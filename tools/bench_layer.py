#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一整层 decoder 的端到端吞吐基准（设备侧串联，最后一个 token 才 sync）。

    source env.sh
    python3 tools/bench_layer.py                       # dim=512/8 头/ffn=1024
    python3 tools/bench_layer.py --dim 1024 --ffn 2048 --tokens 128

和 `examples/python_model_layer.py` 里的 `[layer]` 那行不同：那边逐 token
下载回来跟 NumPy 对账，时间被「上传/下载」主导；这里**不下载**，只把 token
一路喂下去、最后一次 `sync`，量到的才是这层本身。

注意 `TransformerLayer` 的稳态要走录好的 `LaunchPlan`：第一个 token 会把
这一层的 launch 序列录下来（所以首个 token 明显慢），之后每 token 只打几个
补丁再一次性重放。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from k100lc_kernels import Runtime, TransformerLayer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3, help="重复几轮，报最好的一轮")
    args = ap.parse_args()

    dim, nh, ffn = args.dim, args.heads, args.ffn
    hd = dim // nh
    nt = min(args.tokens, args.max_len - 1)
    rt = Runtime()
    rng = np.random.default_rng(4)

    def mk(shape, s=1.0):
        return rng.standard_normal(shape).astype(np.float32) * s

    layer = TransformerLayer(
        rt, dim=dim, n_heads=nh, head_dim=hd, ffn=ffn, max_len=args.max_len,
        w_qkv=mk((3 * dim, dim), .02), w_o=mk((dim, dim), .02),
        w_gate_up=mk((2 * ffn, dim), .02), w_down=mk((dim, ffn), .02),
        w_norm1=mk(dim, .5) + 1, w_norm2=mk(dim, .5) + 1, eps=1e-6, tag="bl")
    x = rt.alloc(dim * 4)
    rt.upload(x, mk(dim))
    layer.forward_device(x, 0)                  # 首个 token：录 plan
    rt.sync()
    print(f"dim={dim} n_heads={nh} head_dim={hd} ffn={ffn} max_len={args.max_len}"
          f" 每 token 走 LaunchPlan 重放 {len(layer._plan.kernels)} 个内核")
    best = None
    for _ in range(args.runs):
        layer.reset()
        layer.forward_device(x, 0)
        rt.sync()
        t0 = time.perf_counter()
        for pos in range(1, nt + 1):
            layer.forward_device(x, pos)
        rt.sync()
        dt = (time.perf_counter() - t0) / nt
        best = dt if best is None else min(best, dt)
    print(f"{nt} 个 token 最好一轮：{best * 1e3:.3f} ms/token"
          f"（不含上传/下载）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
