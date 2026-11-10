# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys
from collections import OrderedDict
from torch.cuda.amp import autocast, GradScaler
import torch
from torch.nn.parallel import DistributedDataParallel

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.data import build_reid_test_loader, build_reid_train_loader
from fastreid.evaluation.testing import flatten_results_dict
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.modeling import build_model
from fastreid.solver import build_lr_scheduler, build_optimizer
from fastreid.evaluation import inference_on_dataset, print_csv_format, ReidEvaluator
from fastreid.utils.checkpoint import Checkpointer, PeriodicCheckpointer
from fastreid.utils import comm
from fastreid.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter
)

logger = logging.getLogger("fastreid")


def get_evaluator(cfg, dataset_name, output_dir=None):
    data_loader, num_query = build_reid_test_loader(cfg, dataset_name=dataset_name)
    return data_loader, ReidEvaluator(cfg, num_query, output_dir)


def do_test(cfg, model):
    results = OrderedDict()
    for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
        logger.info("Prepare testing set")
        try:
            data_loader, evaluator = get_evaluator(cfg, dataset_name)
        except NotImplementedError:
            logger.warn(
                "No evaluator found. implement its `build_evaluator` method."
            )
            results[dataset_name] = {}
            continue
        results_i = inference_on_dataset(model, data_loader, evaluator, flip_test=cfg.TEST.FLIP.ENABLED)
        results[dataset_name] = results_i

        if comm.is_main_process():
            assert isinstance(
                results, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results
            )
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            results_i['dataset'] = dataset_name
            print_csv_format(results_i)

    if len(results) == 1:
        results = list(results.values())[0]

    return results


def do_train(cfg, model, resume=False, data_loader=None):
    
    data_loader_iter = iter(data_loader)

    scaler = GradScaler() if cfg.SOLVER.AMP.ENABLED else None

    model.train()
    optimizer = build_optimizer(cfg, model)[0]

    iters_per_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH
    scheduler = build_lr_scheduler(cfg, optimizer, iters_per_epoch)

    checkpointer = Checkpointer(
        model,
        cfg.OUTPUT_DIR,
        save_to_disk=comm.is_main_process(),
        optimizer=optimizer,
        **scheduler
    )

    start_epoch = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("epoch", -1) + 1
    )
    iteration = start_iter = start_epoch * iters_per_epoch

    max_epoch = cfg.SOLVER.MAX_EPOCH
    max_iter = max_epoch * iters_per_epoch
    warmup_iters = cfg.SOLVER.WARMUP_ITERS
    delay_epochs = cfg.SOLVER.DELAY_EPOCHS

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_epoch)
    if len(cfg.DATASETS.TESTS) == 1:
        metric_name = "metric"
    else:
        metric_name = cfg.DATASETS.TESTS[0] + "/metric"

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR)
        ]
        if comm.is_main_process()
        else []
    )

    logger.info("Start training from epoch {}".format(start_epoch))
    with EventStorage(start_iter) as storage:
        for epoch in range(start_epoch, max_epoch):
            storage.epoch = epoch
            for _ in range(iters_per_epoch):
                data = next(data_loader_iter)
                storage.iter = iteration

                # ========== AMP: 前向传播 ==========
                with autocast(enabled=cfg.SOLVER.AMP.ENABLED):
                    loss_dict = model(data)
                    losses = sum(loss_dict.values())
                    assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()

                # ========== AMP: 反向传播 ==========
                if cfg.SOLVER.AMP.ENABLED:
                    scaler.scale(losses).backward()
                else:
                    losses.backward()

                # ========== AMP: 参数更新 ==========
                if cfg.SOLVER.AMP.ENABLED:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

                if iteration - start_iter > 5 and \
                        ((iteration + 1) % 200 == 0 or iteration == max_iter - 1) and \
                        ((iteration + 1) % iters_per_epoch != 0):
                    for writer in writers:
                        writer.write()

                iteration += 1

                if iteration <= warmup_iters:
                    scheduler["warmup_sched"].step()

            # Write metrics after each epoch
            for writer in writers:
                writer.write()

            if iteration > warmup_iters and (epoch + 1) > delay_epochs:
                scheduler["lr_sched"].step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (epoch + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
            ):
                results = do_test(cfg, model)
            else:
                results = {}
            flatten_results = flatten_results_dict(results)

            metric_dict = dict(metric=flatten_results[metric_name] if metric_name in flatten_results else -1)
            periodic_checkpointer.step(epoch, **metric_dict)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

# def cluster_token(cfg, dataset):
#     from fastreid.modeling.backbones.build import build_backbone
#     import torch.nn.functional as F
#     from sklearn.cluster import KMeans
#     import numpy as np
#     import random
#     from collections import defaultdict

#     backbone = build_backbone(cfg)
#     backbone.eval()
#     device = torch.device(cfg.MODEL.DEVICE)
#     backbone.to(device)

#     # ========== 1. 按 pid 分组收集图片索引 ==========
#     # CommDataset 使用 img_items，格式: (img_path, pid_str, camid_str, viewid_str)
#     if hasattr(dataset, 'img_items'):
#         samples = dataset.img_items
#     elif hasattr(dataset, 'data'):
#         samples = dataset.data
#     elif hasattr(dataset, 'samples'):
#         samples = dataset.samples
#     else:
#         samples = dataset

#     pid_to_indices = defaultdict(list)

#     for idx, sample in enumerate(samples):
#         # sample 格式: (img_path, pid_str, camid_str, viewid_str)
#         if isinstance(sample, (list, tuple)) and len(sample) >= 2:
#             pid_str = sample[1]
#             pid_to_indices[pid_str].append(idx)

#     all_pids = sorted(pid_to_indices.keys())
#     num_pids = len(all_pids)
#     logger.info(f"Total pids: {num_pids}, total images: {len(samples)}")

#     # ========== 2. 构建 DataLoader 用于特征提取 ==========
#     from torch.utils.data import DataLoader

#     test_batch_size = cfg.TEST.IMS_PER_BATCH  # 1024
#     sampler = torch.utils.data.SequentialSampler(dataset)
#     data_loader = DataLoader(
#         dataset,
#         batch_size=test_batch_size,
#         sampler=sampler,
#         num_workers=cfg.DATALOADER.NUM_WORKERS,
#         pin_memory=True,
#         collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
#     )

#     logger.info(f"Extracting features with batch_size={test_batch_size}")

#     # ========== 3. 提取所有图片的特征并存储 ==========
#     # 先提取所有图片的 patch 特征，按索引存储
#     all_features = {}  # idx -> [128, 768] tensor

#     with torch.no_grad():
#         for batch_idx, batch_data in enumerate(data_loader):
#             # 计算当前 batch 的起始索引
#             start_idx = batch_idx * test_batch_size

#             # 处理 batch_data 格式
#             if isinstance(batch_data, dict):
#                 images = batch_data['images']
#             elif isinstance(batch_data, (list, tuple)):
#                 images = batch_data[0]
#             else:
#                 images = batch_data

#             images = images.to(device)
#             features = backbone.forward_features(images)  # [B, 128, 768]

#             B = features.shape[0]
#             for i in range(B):
#                 img_idx = start_idx + i
#                 if img_idx < len(samples):
#                     all_features[img_idx] = features[i].cpu()  # [128, 768]

#             if (batch_idx + 1) % 10 == 0:
#                 logger.info(f"Processed {(batch_idx + 1) * test_batch_size} / {len(samples)} images")

#     logger.info(f"Extracted features for {len(all_features)} images")

#     # ========== 4. 按 pid 采样描述子 ==========
#     # 每个 pid 取 4 张图（不足则重复采样），每张图取 100 个描述子
#     num_select_per_pid = 4   # 每个 pid 选多少张图
#     num_patches_per_img = 100  # 每张图选多少个 patch

#     all_descriptors = []  # 收集所有描述子用于聚类

#     for pid in all_pids:
#         indices = pid_to_indices[pid]
#         num_imgs = len(indices)

#         # 如果不足 4 张，重复采样直到有 4 张
#         if num_imgs < num_select_per_pid:
#             # 重复采样：循环使用已有图片
#             selected_indices = []
#             while len(selected_indices) < num_select_per_pid:
#                 selected_indices.extend(indices)
#             selected_indices = selected_indices[:num_select_per_pid]
#         else:
#             # 随机选 4 张
#             selected_indices = random.sample(indices, num_select_per_pid)

#         # 对选中的每张图，随机采样 100 个描述子
#         for img_idx in selected_indices:
#             if img_idx in all_features:
#                 feature = all_features[img_idx]  # [128, 768]
#                 N = feature.shape[0]  # 128 patches

#                 # 随机选 100 个 patch（从 128 个中选）
#                 selected_patch_indices = random.sample(range(N), num_patches_per_img)
#                 selected_desc = feature[selected_patch_indices, :]  # [100, 768]
#                 all_descriptors.append(selected_desc)

#     # 合并所有描述子: [num_pids * 4 * 100, 768]
#     all_descriptors = torch.cat(all_descriptors, dim=0)  # [N_total, 768]
#     logger.info(f"Total descriptors for clustering: {all_descriptors.shape}")

#     # ========== 5. 聚类 ==========
#     descriptors_np = all_descriptors.numpy()

#     # K-Means 聚类，8 个中心
#     logger.info("Starting K-Means clustering with k=8...")
#     kmeans = KMeans(n_clusters=8, random_state=42, n_init=1)
#     kmeans.fit(descriptors_np)

#     # 获取 8 个聚类中心: [8, 768]
#     cluster_centers = kmeans.cluster_centers_

#     # ========== 6. 保存到 cfg ==========
#     cfg.defrost()
#     cfg.MODEL.CLUSTER_INIT = cluster_centers.tolist()  # 转为 list 保存
#     cfg.freeze()

#     logger.info(f"Cluster centers saved to cfg.MODEL.CLUSTER_INIT, shape: {cluster_centers.shape}")

#     # 清理
#     del backbone
#     torch.cuda.empty_cache()
def cluster_token(cfg, dataset):
    from fastreid.modeling.backbones.build import build_backbone
    from sklearn.cluster import KMeans
    import numpy as np
    import random
    random.seed(42)
    np.random.seed(42)
    from collections import defaultdict

    backbone = build_backbone(cfg)
    backbone.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    backbone.to(device)

    # ========== 1. 获取样本并按 pid 分组 ==========
    samples = dataset.img_items
    pid_to_indices = defaultdict(list)

    for idx, sample in enumerate(samples):
        if isinstance(sample, (list, tuple)) and len(sample) >= 4:
            pid_str = sample[1]
            pid_to_indices[pid_str].append(idx)

    all_pids = sorted(pid_to_indices.keys())
    num_pids = len(all_pids)
    logger.info(f"Total pids: {num_pids}, total images: {len(samples)}")

    # ========== 2. 构建 DataLoader 用于特征提取 ==========
    from torch.utils.data import DataLoader

    test_batch_size = cfg.TEST.IMS_PER_BATCH
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=test_batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
    )

    logger.info(f"Extracting features with batch_size={test_batch_size}")

    # ========== 3. 提取所有图片的 patch 特征 ==========
    all_patch_features = {}   # idx -> [128, 768]

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            start_idx = batch_idx * test_batch_size

            if isinstance(batch_data, dict):
                images = batch_data['images']
            elif isinstance(batch_data, (list, tuple)):
                images = batch_data[0]
            else:
                images = batch_data

            images = images.to(device)
            features = backbone.forward_features(images)  # [B, 129, 768]

            B = features.shape[0]
            for i in range(B):
                img_idx = start_idx + i
                if img_idx < len(samples):
                    all_patch_features[img_idx] = features[i, 1:, :].cpu()   # [128, 768] patches

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {(batch_idx + 1) * test_batch_size} / {len(samples)} images")

    logger.info(f"Extracted patch features for {len(all_patch_features)} images")

    # ========== 4. Query Tokens: 按 pid 采样 patch 描述子，聚 8 个中心 ==========
    num_select_per_pid = 6      # 每个 pid 选多少张图
    num_patches_per_img = 96   # 每张图选多少个 patch
    num_query_clusters = 8

    logger.info("Clustering QUERY tokens from patch descriptors...")

    all_descriptors = []  # 收集所有 patch 描述子

    for pid in all_pids:
        indices = pid_to_indices[pid]
        num_imgs = len(indices)

        # 不足 4 张则重复采样
        if num_imgs < num_select_per_pid:
            selected_indices = []
            while len(selected_indices) < num_select_per_pid:
                selected_indices.extend(indices)
            selected_indices = selected_indices[:num_select_per_pid]
        else:
            selected_indices = random.sample(indices, num_select_per_pid)

        # 对选中的每张图，随机采样指定数量的 patch 描述子（不含 cls）
        for img_idx in selected_indices:
            if img_idx in all_patch_features:
                feature = all_patch_features[img_idx]  # [128, 768]
                N = feature.shape[0]

                selected_patch_indices = random.sample(range(N), num_patches_per_img)
                selected_desc = feature[selected_patch_indices, :]  # [num_patches_per_img, 768]
                all_descriptors.append(selected_desc)

    # 合并所有描述子
    all_descriptors = torch.cat(all_descriptors, dim=0)  # [num_pids * num_select_per_pid * num_patches_per_img, 768]
    logger.info(f"Total patch descriptors for query clustering: {all_descriptors.shape}")

    # K-Means 聚类 8 个中心
    descriptors_np = all_descriptors.numpy()
    logger.info(f"Starting K-Means for query tokens with k={num_query_clusters}...")
    kmeans_query = KMeans(n_clusters=num_query_clusters, random_state=42, n_init=10)
    kmeans_query.fit(descriptors_np)

    query_centers = torch.from_numpy(kmeans_query.cluster_centers_).float()  # [8, 768]
    logger.info(f"Query tokens shape: {query_centers.shape}")

    # ========== 5. 保存到 cfg ==========
    cfg.defrost()
    cfg.MODEL.CLUSTER_INIT_QUERY = query_centers.tolist()   # [8, 768]
    cfg.freeze()

    logger.info(f"Query tokens saved to cfg.MODEL.CLUSTER_INIT_QUERY, shape: {query_centers.shape}")

    # 清理
    del backbone
    torch.cuda.empty_cache()
def main(args):
    cfg = setup(args)
    data_loader = build_reid_train_loader(cfg)
    num_classes = data_loader.dataset.num_classes  # 或 dataset.num_classes / dataset.num_train_pids
    # import pdb; pdb.set_trace()  # 断点
    cfg.defrost()
    cfg.MODEL.HEADS.NUM_CLASSES = num_classes
    cluster_token(cfg,data_loader.dataset)


    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=True
        )

    do_train(cfg, model, resume=args.resume,data_loader=data_loader)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )