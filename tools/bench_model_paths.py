#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型级线性层路径基准：f32 vs compressed-tensors INT4（W4A16）。

    source env.sh
    python3 tools/bench_model_paths.py --n 17408 --k 5120 --rows 1 --iters 50

用合成权重跑 `model.F32Linear` / `model.DotLinear`，报每层耗时、等效权重带宽，
并按 400 个同形状层估算权重流时间。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from k100lc_kernels import (DotLinear, F32Linear, Runtime, Workspace,  # noqa: E402
                            pack_int4_group128)


def bench(fn, iters: int) -> float:
    fn()
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=17408)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    rng = np.random.default_rng(2026)
    w = rng.standard_normal((args.n, args.k), dtype=np.float32) * 0.02
    x = rng.standard_normal((args.rows, args.k), dtype=np.float32)
    packed, scales = pack_int4_group128(w)
    qbytes = len(packed) + len(scales)
    fbytes = w.nbytes

    rt = Runtime()
    ws = Workspace(rt)
    px = ws.buffer("x", x.nbytes)
    rt.upload(px, x.reshape(-1))

    f32 = F32Linear(rt, args.n, args.k, w, ws, "bench_f32")
    int4 = DotLinear(rt, args.n, args.k, "int4", packed, (scales,), ws, "bench_int4")

    def run_f32():
        f32.forward_device(px, args.rows, sync=True)

    def run_int4():
        int4.forward_device(px, args.rows, sync=True)

    tf = bench(run_f32, args.iters)
    ti = bench(run_int4, args.iters)
    print(f"shape N={args.n} K={args.k} M={args.rows}")
    print(f"  f32  weights {fbytes/1e6:7.1f} MB  {tf*1e3:7.3f} ms  "
          f"{fbytes/tf/1e9:6.1f} GB/s  {args.n*args.k*2*args.rows/tf/1e12:6.2f} TFLOP/s")
    print(f"  int4 weights {qbytes/1e6:7.1f} MB  {ti*1e3:7.3f} ms  "
          f"{qbytes/ti/1e9:6.1f} GB/s  {args.n*args.k*2*args.rows/ti/1e12:6.2f} TFLOP/s"
          f"  speedup={tf/ti:.2f}x")
    print(f"  400 个同形状层的权重流时间：f32 {400*tf*1e3:.1f} ms/token，"
          f"int4 {400*ti*1e3:.1f} ms/token")
    ws.free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
