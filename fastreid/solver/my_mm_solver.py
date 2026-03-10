import torch
from fastreid.solver.build import OPTIMIZER_REGISTRY, maybe_add_gradient_clipping, maybe_add_freeze_layer

@OPTIMIZER_REGISTRY.register()
def build_mm_sgd_optimizer(cfg, model):
    """
    1) image_encoder 用 BASE_LR
    2) fake_text_net 用 INV_LR（配置里新增）
    3) 其余选项（momentum, weight_decay, clip, freeze）全走原逻辑
    """
    # 基本的 momentum/nesterov
    optim_defaults = {
        "momentum": cfg.SOLVER.MOMENTUM,
        "nesterov": cfg.SOLVER.NESTEROV,
    }

    # 1) 视觉分支
    img_group = {
        "params": model.image_encoder.parameters(),
        "lr": cfg.SOLVER.BASE_LR,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
        **optim_defaults
    }

    # 2) 反演网络分支
    inv_group = {
        "params": model.fake_text_net.parameters(),
        # 配置里新增 INV_LR、INV_WEIGHT_DECAY
        "lr": getattr(cfg.SOLVER, "INV_LR", cfg.SOLVER.BASE_LR * 5),
        "weight_decay": getattr(cfg.SOLVER, "INV_WEIGHT_DECAY", cfg.SOLVER.WEIGHT_DECAY),
        **optim_defaults
    }

    groups = [img_group, inv_group]
    optim = torch.optim.SGD(groups)
    # 把梯度裁剪 & freeze layer 混进去
    optim_cls = maybe_add_freeze_layer(cfg,
                 maybe_add_gradient_clipping(cfg, optim.__class__))
    return optim_cls(groups), None