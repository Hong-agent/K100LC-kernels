#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoE 稠密全专家 vs token→expert 分桶 基准（INT4 W4A8 专家）。

    source env.sh
    python3 tools/bench_moe.py --rows 512 --dim 512 --n-exp 8 --topk 2 --iters 5

路由用确定性的 round-robin，让每个专家恰好拿到 `rows*topk/n_exp` 行，
两条路径都走 W4A4 GEMM，便于比较纯计算/访存的差别。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from k100lc_kernels import (Int4Linear, MoECombine, MoEExperts, Runtime,  # noqa: E402
                            Workspace, dequant_int4_group128, pack_int4_group128)


def bench(fn, iters: int) -> float:
    fn()
    for _ in range(2):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--n-exp", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-check", action="store_true",
                    help="跳过主机参考（大模型更快出数）")
    args = ap.parse_args()
    if args.rows % args.n_exp or (args.rows * args.topk) % args.n_exp:
        raise SystemExit("rows 与 rows*topk 都要能被 n_exp 整除（保证桶大小整齐）")

    rng = np.random.default_rng(args.seed)
    dim, n_exp, rows, topk = args.dim, args.n_exp, args.rows, args.topk
    x = rng.standard_normal((rows, dim), dtype=np.float32)
    w = [rng.normal(0, 0.05, (dim, dim)).astype(np.float32) for _ in range(n_exp)]
    packed = [pack_int4_group128(wi) for wi in w]
    dqw = [dequant_int4_group128(p, s, dim, dim) for p, s in packed]
    # round-robin 路由：每个专家恰好 rows*topk/n_exp 行
    ids = np.stack([(np.arange(rows) + e) % n_exp for e in range(topk)], axis=1)
    weights = rng.random((rows, topk), dtype=np.float32)

    if not args.no_check:
        ref = np.stack([
            sum(weights[r, e] * (x[r] @ dqw[ids[r, e]].T)
                for e in range(topk))
            for r in range(rows)])

    rt = Runtime()
    ws = Workspace(rt)
    px = ws.buffer("x", x.nbytes)
    rt.upload(px, x.reshape(-1))
    experts = [Int4Linear(rt, dim, dim, packed[e][0], packed[e][1], "w4a8",
                          ws, f"exp{e}", prefill="gemm")
               for e in range(n_exp)]

    # 稠密：每个专家跑全部行
    dense = MoECombine(rt, rows, dim, n_exp, ws, "dense")
    all_w = np.zeros((rows, n_exp), dtype=np.float32)
    for r in range(rows):
        for e in range(topk):
            all_w[r, ids[r, e]] += weights[r, e]
    rt.upload(dense.weights, all_w.reshape(-1))

    def run_dense():
        for e, expert in enumerate(experts):
            expert.forward_device(px, rows, out_dev=dense.slot(e), sync=False)
        dense.forward_device(sync=True)

    # 分桶：每个专家只处理自己收到的 token
    moe = MoEExperts(rt, experts, rows, dim, dim, topk, ws, "bucket")

    def run_bucket():
        moe.forward_device(px, rows, ids, weights, sync=True)

    t_dense = bench(run_dense, args.iters)
    t_bucket = bench(run_bucket, args.iters)
    run_bucket()
    out = rt.download(moe.y, rows * dim, np.float32).reshape(rows, dim)
    err = float("nan") if args.no_check else \
        float(np.abs(out - ref).max() / max(1e-12, float(np.abs(ref).max())))
    mac_dense = rows * n_exp * dim * dim
    mac_bucket = rows * topk * dim * dim
    print(f"rows={rows} dim={dim} n_exp={n_exp} topk={topk}")
    print(f"  dense  {t_dense*1e3:8.3f} ms  {mac_dense/t_dense/1e12:5.1f} TMAC/s  "
          f"({mac_dense/1e6:.0f} MMAC)")
    print(f"  bucket {t_bucket*1e3:8.3f} ms  {mac_bucket/t_bucket/1e12:5.1f} TMAC/s  "
          f"({mac_bucket/1e6:.0f} MMAC)  speedup={t_dense/t_bucket:.2f}x  relerr={err:.2e}")
    ws.free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
