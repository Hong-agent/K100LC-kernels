#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 模型线性层的两条“最强通路”实跑：W4A8（v_dot4）与 W4A4（v_dot8）。

    source env.sh

    # 真权重（用 K100LC_RT4=/path/model.rt4 指定）
    python3 examples/python_w4a4_w4a8_gemv.py
    python3 examples/python_w4a4_w4a8_gemv.py --rows 1,4 --iters 50

    # 没下模型时用合成权重自检
    python3 examples/python_w4a4_w4a8_gemv.py --selftest

指标：
  * GB/s  = 每步读过的 int4 权重字节（q+s）÷ 时间，对照本卡实测解码上限 566 GB/s；
  * TMAC/s = M·N·K ÷ 时间，对照 v_dot8 峰值 75.6 TMAC/s（W4A4）。
W4A8 用 int8 激活，精度约 1%；W4A4 用 int4 激活，精度约 10%，但算力是前者的两倍。
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


def _ref(f: RT4File, t, x: np.ndarray, chunk: int = 2048) -> np.ndarray:
    n, k = t.shape
    y = np.empty((x.shape[0], n), dtype=np.float32)
    for s in range(0, n, chunk):
        c = min(chunk, n - s)
        w = f.dequant(t, rows=c, start=s)          # [c,K] f32
        y[:, s:s + c] = x @ w.T
    return y


def _bench(run, iters: int) -> float:
    run()
    for _ in range(3):
        run()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    return (time.perf_counter() - t0) / iters


def _report(tag: str, y: np.ndarray, ref: np.ndarray, dt: float,
            m: int, n: int, k: int, wbytes: int) -> None:
    err = float(np.abs(y - ref).max() / max(1e-12, float(np.abs(ref).max())))
    gbps = wbytes / dt / 1e9
    tmac = m * n * k / dt / 1e12
    print(f"  {tag}: {dt * 1e3:7.3f} ms  {gbps:6.1f} GB/s  {tmac:6.1f} TMAC/s  "
          f"relerr {err:.3e}")


def selftest() -> int:
    """无 RT4 文件时的合成权重自检。"""
    rng = np.random.default_rng(0)
    n, k, grp = 512, 512, 128
    q = rng.integers(-8, 8, size=(n, k), dtype=np.int32)
    s = rng.uniform(1e-3, 2e-2, size=(n, k // grp)).astype(np.float32)
    wq = ((q & 0xF)[:, 0::2] | ((q & 0xF)[:, 1::2] << 4)).astype(np.uint8).tobytes()
    ws = s.astype(np.float16).tobytes()
    wref = (q.reshape(n, k // grp, grp).astype(np.float32) * s[:, :, None]).reshape(n, k)
    rt = Runtime()
    pwq = rt.alloc(len(wq)); rt.upload(pwq, wq)
    pws = rt.alloc(len(ws)); rt.upload(pws, ws)
    runner = W4Runner(rt)
    rc = 0
    for m in (1, 2, 4):
        x = rng.standard_normal((m, k)).astype(np.float32)
        ref = x @ wref.T
        for kind, tol in (("w4a8", 3e-2), ("w4a4", 3e-1)):
            y = runner.gemv(kind, pwq, pws, x, n, k)
            err = float(np.abs(y - ref).max() / np.abs(ref).max())
            ok = err < tol
            rc |= 0 if ok else 1
            print(f"selftest M={m} {kind}: relerr {err:.3e} {'OK' if ok else 'FAIL'} "
                  f"(阈值 {tol:g})")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rt4", default=str(DEFAULT_RT4))
    ap.add_argument("--name", default=DEFAULT_LAYER)
    ap.add_argument("--rows", default="1",
                    help="M 列表，逗号分隔（1..4），默认 1")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest or not pathlib.Path(args.rt4).is_file():
        if not args.selftest:
            print(f"找不到 {args.rt4}，改用合成权重自检")
        return selftest()

    rows = [int(v) for v in args.rows.split(",") if v]
    f = RT4File(args.rt4)
    t = f.tensor(args.name)
    if t.kind != "i4":
        print(f"{t.name} kind={t.kind}，本示例只支持 i4 线性层")
        return 1
    n, k = t.shape
    wbytes = t.q_bytes + t.s_bytes
    print(f"RT4 {args.rt4}")
    print(f"层 {t.name}  [{n}x{k}]  int4 权重 {wbytes / 1e6:.2f} MB  "
          f"group={t.group}")

    rt = Runtime()
    pwq, pws = f.upload_weight(rt, t)
    runner = W4Runner(rt)
    rng = np.random.default_rng(1)
    rc = 0
    for m in rows:
        x = rng.standard_normal((m, k)).astype(np.float32)
        ref = _ref(f, t, x)
        for kind in ("w4a8", "w4a4"):
            # 计时只算设备侧（含激活量化 + GEMV + sync），不含 H2D/D2H
            runner.gemv_device(kind, pwq, pws, x, n, k)      # 先上传 x
            dt = _bench(lambda: runner.gemv_device(kind, pwq, pws, x, n, k,
                                                   upload=False), args.iters)
            y = runner.gemv(kind, pwq, pws, x, n, k)
            _report(f"M={m} {kind.upper()}", y, ref, dt, m, n, k, wbytes)
        # 两条通路的相互差（W4A4 的 int4 激活量化误差）
        y8 = runner.gemv("w4a8", pwq, pws, x, n, k)
        y4 = runner.gemv("w4a4", pwq, pws, x, n, k)
        d = float(np.abs(y4 - y8).max() / np.abs(y8).max())
        print(f"  M={m} W4A4 vs W4A8 相对差 {d:.3e}")
    runner.close()
    rt.free(pwq); rt.free(pws)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
