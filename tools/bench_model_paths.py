#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型级线性层路径基准：f32 / INT4 W4A16 / W4A8 / W4A4 / 预填充 GEMM。

    source env.sh
    python3 tools/bench_model_paths.py --n 17408 --k 5120 --rows 1 --iters 50
    python3 tools/bench_model_paths.py --n 17408 --k 5120 --rows 128 --iters 20

用合成权重跑 `model.F32Linear` / `DotLinear` / `Int4Linear`，报每层耗时、
等效权重带宽与相对 f32 的加速比；`--rows` 是 128 的倍数时还会跑
`Int4Linear` 的 W4A4 预填充 GEMM。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from k100lc_kernels import (DotLinear, F32Linear, Int4Linear, Runtime,  # noqa: E402
                            Workspace, pack_int4_group128)


def bench(fn, iters: int) -> float:
    fn()
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / max(1e-12, float(np.abs(b).max())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=17408)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    rng = np.random.default_rng(2026)
    w = (rng.standard_normal((args.n, args.k), dtype=np.float32) * 0.02)
    x = rng.standard_normal((args.rows, args.k), dtype=np.float32)
    ref = x @ w.T
    packed, scales = pack_int4_group128(w)
    qbytes = len(packed) + len(scales)
    fbytes = w.nbytes

    rt = Runtime()
    ws = Workspace(rt)
    px = ws.buffer("x", x.nbytes)
    rt.upload(px, x.reshape(-1))

    layers = [
        ("f32        ", F32Linear(rt, args.n, args.k, w, ws, "bench_f32"), fbytes),
        ("int4 W4A16 ", DotLinear(rt, args.n, args.k, "int4", packed, (scales,),
                                  ws, "bench_w4a16"), qbytes),
        ("int4 W4A8  ", Int4Linear(rt, args.n, args.k, packed, scales, "w4a8",
                                   ws, "bench_w4a8"), qbytes),
        ("int4 W4A4  ", Int4Linear(rt, args.n, args.k, packed, scales, "w4a4",
                                   ws, "bench_w4a4"), qbytes),
    ]
    print(f"shape N={args.n} K={args.k} M={args.rows}")
    print(f"  f32 weights {fbytes/1e6:.1f} MB, int4 weights {qbytes/1e6:.1f} MB")
    t_f32 = None
    for name, layer, wb in layers:
        def run(layer=layer):
            layer.forward_device(px, args.rows, sync=True)
        t = bench(run, args.iters)
        out = rt.download(layer.forward_device(px, args.rows, sync=True),
                          args.rows * args.n, np.float32).reshape(args.rows, args.n)
        if t_f32 is None:
            t_f32 = t
        print(f"  {name} {t*1e3:7.3f} ms  {wb/t/1e9:6.1f} GB/s  "
              f"speedup={t_f32/t:5.2f}x  relerr={rel(out, ref):.3e}")
    if args.rows >= 128 and args.rows % 128 == 0 and args.n % 64 == 0:
        layer = layers[3][1]
        t = bench(lambda: layer.forward_device(px, args.rows, sync=True),
                  max(5, args.iters // 4))
        out = rt.download(layer.forward_device(px, args.rows, sync=True),
                          args.rows * args.n, np.float32).reshape(args.rows, args.n)
        tmac = args.rows * args.n * args.k / t / 1e12
        print(f"  prefill GEMM W4A4 {t*1e3:7.3f} ms  {tmac:5.1f} TMAC/s  "
              f"relerr={rel(out, ref):.3e}")
    ws.free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
