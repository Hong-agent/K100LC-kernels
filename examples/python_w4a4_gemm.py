#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 模型预填充的最强算力路径：RT4 int4 × int4 GEMM（`v_dot8_i32_i4`）。

    source env.sh

    python3 examples/python_w4a4_gemm.py                 # 真 27B 层，M=128/512/2048
    python3 examples/python_w4a4_gemm.py --m 128,512 --iters 10
    python3 examples/python_w4a4_gemm.py --selftest      # 合成权重自检

权重 = RT4（int4 q + 组优先 f32 尺度，GEMM 用）；
激活 = `quant_rows_k` 量化的 int4（每 128 元素一组）。M 按 BM=128 分块，
每块拆成 1D 启动（绕开运行时 2D grid 的坑）。指标：TMAC/s（对照 v_dot8
峰值 75.6）与等效权重带宽（每 128 行重读一遍权重）。

真权重路径用 `K100LC_RT4=/path/model.rt4` 指定；没有真权重时用
`--selftest` 跑合成权重。
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from k100lc_kernels import RT4File, W4Runner, Runtime  # noqa: E402

DEFAULT_RT4 = pathlib.Path(os.environ.get(
    "K100LC_RT4", ROOT / "models" / "model.rt4"))
DEFAULT_LAYER = "model.language_model.layers.0.mlp.gate_proj.weight"


def _ref(f: RT4File, t, x: np.ndarray, chunk: int = 512) -> np.ndarray:
    n = t.shape[0]
    y = np.empty((x.shape[0], n), dtype=np.float32)
    for s in range(0, n, chunk):
        c = min(chunk, n - s)
        w = f.dequant(t, rows=c, start=s)        # [c,K] f32
        y[:, s:s + c] = x @ w.T
    return y


def _bench(run, iters: int) -> float:
    run()
    for _ in range(2):
        run()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    return (time.perf_counter() - t0) / iters


def selftest() -> int:
    rng = np.random.default_rng(3)
    n, k, m = 128, 512, 256
    B = rng.integers(-8, 8, size=(n, k), dtype=np.int32)
    x = rng.standard_normal((m, k)).astype(np.float32)
    wq = ((B & 0xF)[:, 0::2] | ((B & 0xF)[:, 1::2] << 4)).astype(np.uint8).tobytes()
    wsc = np.ones((k // 128, n), dtype=np.float32)
    rt = Runtime()
    pwq = rt.alloc(len(wq)); rt.upload(pwq, wq)
    pwsc = rt.alloc(wsc.nbytes); rt.upload(pwsc, wsc)
    runner = W4Runner(rt)
    y = runner.gemm(pwq, pwsc, x, n, k)
    # 参考：把 x 也按 quant_rows_k 的语义量化成 int4（每 128 一组，scale=amax/7）
    xq = np.empty((m, k), dtype=np.float32)
    for r in range(m):
        for g in range(k // 128):
            blk = x[r, g * 128:(g + 1) * 128]
            sc = max(np.abs(blk).max() / 7.0, 1e-30)
            xq[r, g * 128:(g + 1) * 128] = np.clip(np.rint(blk / sc), -8, 7) * sc
    ref = xq @ B.astype(np.float32).T
    err = float(np.abs(y - ref).max() / max(1e-9, float(np.abs(ref).max())))
    ok = err < 1e-3
    print(f"selftest GEMM M={m} N={n} K={k}: relerr {err:.3e} "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rt4", default=str(DEFAULT_RT4))
    ap.add_argument("--name", default=DEFAULT_LAYER)
    ap.add_argument("--m", default="128,512,2048", help="M 列表（128 的倍数）")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest or not pathlib.Path(args.rt4).is_file():
        if not args.selftest:
            print(f"找不到 {args.rt4}，改用合成权重自检")
        return selftest()

    ms = [int(v) for v in args.m.split(",") if v]
    f = RT4File(args.rt4)
    t = f.tensor(args.name)
    n, k = t.shape
    if t.kind != "i4" or n % 64 or k % 128:
        print(f"{t.name} kind={t.kind} shape={t.shape}，不满足 GEMM 约束")
        return 1
    wbytes = t.q_bytes + t.s_bytes
    print(f"RT4 {args.rt4}")
    print(f"层 {t.name}  [{n}x{k}]  int4 权重 {wbytes / 1e6:.2f} MB  "
          f"BM={W4Runner.BM} BN={W4Runner.BN}")

    rt = Runtime()
    pwq, pwsc = f.upload_weight_gemm(rt, t)
    runner = W4Runner(rt)
    rng = np.random.default_rng(1)
    m0 = min(ms)
    x0 = rng.standard_normal((m0, k)).astype(np.float32)
    ref = _ref(f, t, x0)
    y = runner.gemm(pwq, pwsc, x0, n, k)
    err = float(np.abs(y - ref).max() / max(1e-12, float(np.abs(ref).max())))
    print(f"  M={m0} 对 f32 参考 relerr {err:.3e}（int4 激活量化预期 ~1e-1）")

    for m in ms:
        x = rng.standard_normal((m, k)).astype(np.float32)
        runner.gemm_device(pwq, pwsc, x, n, k)          # 预热 + 上传 x + 量化
        dt_g = _bench(lambda: runner.gemm_device(pwq, pwsc, x, n, k,
                                                 upload=False, quantize=False),
                      args.iters)
        dt = _bench(lambda: runner.gemm_device(pwq, pwsc, x, n, k, upload=False),
                    args.iters)
        tmac = m * n * k / dt / 1e12
        tmac_g = m * n * k / dt_g / 1e12
        gbps = (m // W4Runner.BM) * wbytes / dt / 1e9
        print(f"  M={m:5d}: 端到端 {dt * 1e3:7.3f} ms / {tmac:5.1f} TMAC/s"
              f"（量化+GEMM）；只算 GEMM {dt_g * 1e3:7.3f} ms / {tmac_g:5.1f} TMAC/s"
              f"；等效权重带宽 {gbps:6.1f} GB/s")
    rt.free(pwq); rt.free(pwsc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
