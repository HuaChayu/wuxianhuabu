#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantize_h3_g64.py — 把 MiniMax-H3 的「BF16 融合/剪枝本体」离线重打包成
                     MLX affine 4bit / group_size=64 量化权重（safetensors）。

【为什么要离线转，而不是在主工程里写加载器】
  社区融合权重（如 MATLOWAI 的 Fused Turbo、xmarre 的 r1024）大多是 ComfyUI
  血统：BF16 稠密权重、键名 `X.weight`。而无限画布的 MLX 侧只认「量化三元组」
  `X.weight`(U32 打包) + `X.weight.scales` + `X.weight.biases`，走 quantizedMM。
  若直接读 BF16 本体：显存/内存翻倍、推理变慢；若在 Swift 里做量化，则需要把
  40GB 级张量全量载入再转换，会爆内存。
  正确做法就是本脚本：离线、逐张量流式处理，主工程零改动（只换权重目录）。

【核心策略】（与原一次性脚本 quantize_h3_g64.py 完全一致，勿改判定条件）
  只有同时满足以下 4 个条件的张量才量化：
      dtype == BF16  且  ndim == 2  且  末维 >= 64  且  末维 % 64 == 0
  其余一律原样保留，原因：
    - 1D 张量（norm.weight / bias 等）：MLX quantize 只接受 2D，且量化 1D 精度损失大
    - `blocks.N.adaln_proj.linear.weight` [96768, 8]：末维 8 < 64，是剪枝后的折叠系数表
    - `adaln_t_table` [1025, 8] F32：时间步查表，必须保持 F32 精度
    - F32 / F16 张量：保持原精度

【实测结果（2026-09-11）】
  源：MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy.safetensors (BF16, 534 tensor, ~37.47 GiB)
  出：MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-mlx-4bit-g64.safetensors (10.61 GiB)
  209 个 2D 权重走 g64（产出 209×3 = 627 条），325 个保留，合计 952 条；
  还原相对误差 9.1%~11.3%，余弦相似度 0.9936~0.9958。

【用法】
  python3 quantize_h3_g64.py                    # 用下面的 DEFAULT_* 路径
  python3 quantize_h3_g64.py --src A.safetensors --dst B.safetensors
  python3 quantize_h3_g64.py --dry-run          # 只统计会量化/保留各多少个，不写盘

【依赖】numpy、mlx（Apple Silicon 原生）。注意必须用 Python 的 mlx 包，
       不是 Swift MLX；两者量化算法一致，产物可互通。
"""

import argparse
import json
import os
import struct
import time

import numpy as np
import mlx.core as mx

# ---- 默认路径（按需改成你的） ----------------------------------------------
DEFAULT_SRC = os.path.expanduser(
    "~/Downloads/h3-fused/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy.safetensors"
)
DEFAULT_DST = os.path.expanduser(
    "~/Downloads/h3-fused/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-mlx-4bit/"
    "transformer.safetensors"
)

GS, BITS = 64, 4          # group_size / bits，与 config.json 的 quant 配置保持一致
TARGET_DTYPE = "BF16"     # 只量化 BF16；F32/F16 一般是要保精度的关键张量


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_bf16(raw: bytes, shape):
    """BF16 字节流 -> mx.array(bfloat16)。

    safetensors 里的 BF16 就是 uint16 位模式，numpy 没有原生 bfloat16，
    所以按 uint16 读出来再 view 成 bfloat16，全程零精度损失。
    兜底路径：uint16 -> uint32 左移 16 位 -> float32 视图（等价，稍慢）。
    """
    u16 = np.frombuffer(raw, dtype=np.uint16).copy()
    arr = mx.array(u16).reshape(shape)
    try:
        return arr.view(mx.bfloat16)
    except Exception:
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return mx.array(f32).reshape(shape)


def raw_to_mx(raw: bytes, dt: str, shape):
    """按 dtype 把裸字节还原成 mx.array（不量化分支用）。"""
    if dt == "BF16":
        return to_bf16(raw, shape)
    if dt == "F32":
        return mx.array(np.frombuffer(raw, dtype=np.float32).copy()).reshape(shape)
    if dt == "F16":
        return mx.array(np.frombuffer(raw, dtype=np.float16).copy()).reshape(shape)
    raise ValueError(f"unsupported dtype: {dt}")


def should_quant(dt: str, shape) -> bool:
    """量化判定：与主工程 config.json 的量化约定严格对应。"""
    return (
        dt == TARGET_DTYPE
        and len(shape) == 2
        and shape[-1] >= GS
        and shape[-1] % GS == 0
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="H3 BF16 -> MLX affine 4bit g64")
    ap.add_argument("--src", default=DEFAULT_SRC, help="输入 BF16 safetensors")
    ap.add_argument("--dst", default=DEFAULT_DST, help="输出量化 safetensors")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写盘")
    args = ap.parse_args()

    # --- 1. 解析 safetensors 头：前 8 字节是 header 长度，随后是 JSON header ---
    with open(args.src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
        data0 = 8 + n
    hdr.pop("__metadata__", None)
    keys = list(hdr)
    log(f"源张量 {len(keys)} 个，数据区起点 {data0}")

    # mmap 整个文件，逐张量切片，不把 40GB 一次性读入内存（流式的关键）
    mm = np.memmap(args.src, dtype=np.uint8, mode="r")

    outs, nq, nk = {}, 0, 0
    t0 = time.time()
    for i, k in enumerate(keys):
        v = hdr[k]
        dt, shp = v["dtype"], tuple(v["shape"])
        a, b = v["data_offsets"]
        raw = mm[data0 + a: data0 + b]          # 零拷贝视图

        if should_quant(dt, shp):
            w = to_bf16(raw, shp)
            # MLX 官方 affine 量化：group_size 个元素共享一组 scale/bias
            wq, sc, bi = mx.quantize(w, group_size=GS, bits=BITS)
            mx.eval(wq, sc, bi)                 # 立即求值，避免惰性图堆积吃内存
            outs[k] = wq                        # U32 打包权重
            outs[k + ".scales"] = sc            # BF16
            outs[k + ".biases"] = bi            # BF16
            nq += 1
        else:
            outs[k] = raw_to_mx(raw, dt, shp)
            mx.eval(outs[k])
            nk += 1

        if (i + 1) % 25 == 0 or i + 1 == len(keys):
            log(f"  {i+1}/{len(keys)}  量化 {nq} / 保留 {nk}  ({time.time()-t0:.0f}s)")

    log(f"统计：量化 {nq} 个（产出 {nq*3} 条）/ 原始保留 {nk} 个 / 合计 {len(outs)} 条")

    if args.dry_run:
        log("dry-run，跳过写盘")
        return

    # --- 2. 写盘 ---------------------------------------------------------------
    # 目录不存在则建（输出目录可能只有软链的其余组件，transformer 单独放这里）
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    log(f"开始写盘 -> {args.dst}")
    mx.save_safetensors(args.dst, outs)
    size = os.path.getsize(args.dst)
    log(f"完成: {size/2**30:.2f} GiB  ({args.dst})")


if __name__ == "__main__":
    main()
