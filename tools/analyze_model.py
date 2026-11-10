#!/usr/bin/env python3
# encoding: utf-8

"""
分析模型 Params / GFLOPs / FPS

FPS:
- CUDA Event timing
- fixed input
- optional AMP FP16
- inference_mode

Example:

CUDA_VISIBLE_DEVICES=4 python3 tools/analyze_model.py \
    --config-file ./configs/CARGO/visa.yml \
    --amp \
    MODEL.WEIGHTS ./logs/CARGO/ViSA/model_best.pth
"""


import argparse
import sys

import torch


sys.path.append('.')


from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer



# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------

def setup_cfg(args):

    cfg = get_cfg()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.defrost()

    cfg.MODEL.BACKBONE.PRETRAIN = False

    cfg.freeze()

    return cfg



# ---------------------------------------------------------------
# Params
# ---------------------------------------------------------------

def count_params(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable



def fmt(num):

    if num >= 1e9:
        return f"{num/1e9:.2f}B ({num:,})"

    if num >= 1e6:
        return f"{num/1e6:.2f}M ({num:,})"

    return f"{num:,}"



# ---------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------

@torch.no_grad()
def compute_flops(model, input_shape):

    from thop import profile


    dev = model.device

    bs, c, h, w = input_shape


    dummy = torch.randn(
        bs,
        c,
        h,
        w,
        device=dev
    )


    inputs = {
        "images": dummy,
        "camids":
            torch.zeros(
                bs,
                dtype=torch.long,
                device=dev
            ),

        "viewids":
            ['Aerial'] * bs
    }


    macs, _ = profile(
        model,
        inputs=(inputs,),
        verbose=False
    )


    return macs / 1e9



# ---------------------------------------------------------------
# FPS
# ---------------------------------------------------------------

@torch.inference_mode()
def measure_fps(
        model,
        input_shape,
        warmup=50,
        iters=200,
        amp=False):


    dev = model.device


    bs, c, h, w = input_shape


    # -----------------------------
    # fixed input
    # -----------------------------

    images = torch.randn(
        bs,
        c,
        h,
        w,
        device=dev
    )


    camids = torch.zeros(
        bs,
        dtype=torch.long,
        device=dev
    )


    viewids = ['Aerial'] * bs


    inputs = {
        "images": images,
        "camids": camids,
        "viewids": viewids
    }



    # -----------------------------
    # warmup
    # -----------------------------

    for _ in range(warmup):

        with torch.cuda.amp.autocast(
            enabled=amp
        ):

            _ = model(inputs)


    torch.cuda.synchronize()



    # -----------------------------
    # CUDA timing
    # -----------------------------

    starter = torch.cuda.Event(
        enable_timing=True
    )

    ender = torch.cuda.Event(
        enable_timing=True
    )



    starter.record()


    for _ in range(iters):

        with torch.cuda.amp.autocast(
            enabled=amp
        ):

            _ = model(inputs)



    ender.record()


    torch.cuda.synchronize()



    total_ms = starter.elapsed_time(
        ender
    )


    avg_ms = total_ms / iters


    fps = (
        bs * iters /
        (total_ms / 1000)
    )


    return fps, avg_ms



# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():


    parser = argparse.ArgumentParser(
        description="Params / GFLOPs / FPS"
    )


    parser.add_argument(
        "--config-file",
        required=True
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )


    parser.add_argument(
        "--num-warmup",
        type=int,
        default=50
    )


    parser.add_argument(
        "--num-iter",
        type=int,
        default=200
    )


    parser.add_argument(
        "--amp",
        action="store_true",
        help="use FP16 inference"
    )


    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        default=[]
    )


    args = parser.parse_args()



    torch.backends.cudnn.benchmark = True



    cfg = setup_cfg(args)



    print(f"Config: {args.config_file}")

    print(
        f"Model: {cfg.MODEL.META_ARCHITECTURE}"
    )

    print(
        f"Backbone: {cfg.MODEL.BACKBONE.NAME}"
    )



    # -----------------------------
    # build model
    # -----------------------------

    model = DefaultTrainer.build_model(cfg)


    model.eval()



    if cfg.MODEL.WEIGHTS:

        print(
            f"Weights: {cfg.MODEL.WEIGHTS}"
        )

        Checkpointer(model).load(
            cfg.MODEL.WEIGHTS
        )



    dev = model.device


    print(
        f"Device: {dev}"
    )



    h, w = cfg.INPUT.SIZE_TEST


    input_shape = (
        args.batch_size,
        3,
        h,
        w
    )



    # -----------------------------
    # Params
    # -----------------------------

    total_p, train_p = count_params(model)


    print("\n" + "="*50)
    print("1. Params")
    print("="*50)


    print(
        f"Total: {fmt(total_p)}"
    )

    print(
        f"Trainable: {fmt(train_p)}"
    )



    # -----------------------------
    # FLOPs
    # -----------------------------

    print("\n" + "="*50)
    print("2. GFLOPs")
    print("="*50)


    try:

        gflops = compute_flops(
            model,
            input_shape
        )

        print(
            f"GFLOPs: {gflops:.2f} G"
        )


    except Exception as e:

        gflops = None

        print(
            "[ERROR]",
            e
        )



    # -----------------------------
    # FPS
    # -----------------------------

    print("\n" + "="*50)
    print("3. FPS")
    print("="*50)


    fps, latency = measure_fps(
        model,
        input_shape,
        args.num_warmup,
        args.num_iter,
        args.amp
    )


    print(
        f"FPS: {fps:.2f}"
    )


    print(
        f"Latency: {latency:.3f} ms / batch"
    )



    print("\n" + "="*50)
    print("Summary")
    print("="*50)


    print(
        f"Params: {fmt(total_p)}"
    )


    if gflops:

        print(
            f"GFLOPs: {gflops:.2f} G"
        )


    print(
        f"FPS: {fps:.2f}"
    )



if __name__ == "__main__":

    main()