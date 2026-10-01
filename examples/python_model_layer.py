#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型级 API 的合成端到端验证 + 计时。

不依赖任何模型权重：随机生成 f32 / INT4 权重，走

    RMSNorm -> MLP(gate/up -> SwiGLU -> down)

并与 NumPy 参考对账；另外验证 MoE 专家输出加权合并内核。

    source env.sh
    python3 examples/python_model_layer.py --rows 4 --dim 512 --ffn 1024
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from k100lc_kernels import (DotLinear, F32Linear, MLP, MoECombine, RMSNorm,  # noqa: E402
                            Runtime, Workspace, dequant_int4_group128,
                            pack_int4_group128)
from iq_dequant import dequant_iq4_nl, dequant_q4_0_fast  # noqa: E402

FAILURES: list[str] = []


def _bench(fn, iters: int = 50) -> float:
    fn()
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3      # ms


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / max(1e-12, float(np.abs(b).max())))


def run_f32(rt: Runtime, rows: int, dim: int, ffn: int,
            rng: np.random.Generator) -> None:
    ws = Workspace(rt)
    x = rng.standard_normal((rows, dim), dtype=np.float32)
    wn = rng.normal(0, 0.1, dim).astype(np.float32) + 1.0
    wg = rng.normal(0, 0.05, (ffn, dim)).astype(np.float32)
    wu = rng.normal(0, 0.05, (ffn, dim)).astype(np.float32)
    wd = rng.normal(0, 0.05, (dim, ffn)).astype(np.float32)

    norm = RMSNorm(rt, dim, wn, 1e-6, ws, "norm")
    mlp = MLP(rt, F32Linear(rt, ffn, dim, wg, ws, "gate"),
              F32Linear(rt, ffn, dim, wu, ws, "up"),
              F32Linear(rt, dim, ffn, wd, ws, "down"), ws, "mlp")

    def step():
        h = norm.forward_device(px, rows, sync=False)
        return mlp.forward_device(h, rows, sync=True)

    px = rt.alloc(x.nbytes)
    rt.upload(px, x)
    out = rt.download(step(), rows * dim, np.float32).reshape(rows, dim)

    rn = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * wn
    g = rn @ wg.T
    ref = (g / (1 + np.exp(-g))) * (rn @ wu.T) @ wd.T
    err = _rel(out, ref)
    if err > 1e-4:
        FAILURES.append("f32")
    print(f"[f32 ] rows={rows} dim={dim} ffn={ffn} max_rel={err:.2e} "
          f"time={_bench(step, 50):.3f} ms")
    ws.free()


def run_int4(rt: Runtime, rows: int, dim: int, ffn: int,
             rng: np.random.Generator) -> None:
    ws = Workspace(rt)
    x = rng.standard_normal((rows, dim), dtype=np.float32)
    wn = rng.normal(0, 0.1, dim).astype(np.float32) + 1.0
    wg = rng.normal(0, 0.05, (ffn, dim)).astype(np.float32)
    wu = rng.normal(0, 0.05, (ffn, dim)).astype(np.float32)
    wd = rng.normal(0, 0.05, (dim, ffn)).astype(np.float32)
    pg, sg = pack_int4_group128(wg)
    pu, su = pack_int4_group128(wu)
    pd, sd = pack_int4_group128(wd)

    norm = RMSNorm(rt, dim, wn, 1e-6, ws, "norm")
    mlp = MLP(rt,
              DotLinear(rt, ffn, dim, "int4", pg, (sg,), ws, "gate"),
              DotLinear(rt, ffn, dim, "int4", pu, (su,), ws, "up"),
              DotLinear(rt, dim, ffn, "int4", pd, (sd,), ws, "down"), ws, "mlp")

    def step():
        h = norm.forward_device(px, rows, sync=False)
        return mlp.forward_device(h, rows, sync=True)

    px = rt.alloc(x.nbytes)
    rt.upload(px, x)
    out = rt.download(step(), rows * dim, np.float32).reshape(rows, dim)

    rn = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * wn
    dg = dequant_int4_group128(pg, sg, ffn, dim)
    du = dequant_int4_group128(pu, su, ffn, dim)
    dd = dequant_int4_group128(pd, sd, dim, ffn)
    g = rn @ dg.T
    ref = (g / (1 + np.exp(-g))) * (rn @ du.T)
    ref = ref @ dd.T
    err = _rel(out, ref)
    if err > 1e-4:
        FAILURES.append("int4")
    print(f"[int4] rows={rows} dim={dim} ffn={ffn} max_rel={err:.2e} "
          f"time={_bench(step, 50):.3f} ms")
    ws.free()


def run_moe(rt: Runtime, rows: int, dim: int, n_exp: int,
            rng: np.random.Generator) -> None:
    ws = Workspace(rt)
    exp_out = rng.standard_normal((n_exp, rows, dim), dtype=np.float32)
    weights = rng.random((rows, n_exp), dtype=np.float32)
    moe = MoECombine(rt, rows, dim, n_exp, ws, "moe")
    out = moe.forward(exp_out, weights)
    ref = np.einsum("re,erd->rd", weights, exp_out)
    moe.rt.upload(moe.exp_out, exp_out.reshape(-1))
    moe.rt.upload(moe.weights, weights.reshape(-1))
    err = _rel(out, ref)
    if err > 1e-4:
        FAILURES.append("moe")
    print(f"[moe ] rows={rows} dim={dim} n_exp={n_exp} max_rel={err:.2e} "
          f"time={_bench(lambda: moe.forward_device(sync=True), 50):.3f} ms")
    ws.free()


def run_gguf(rt: Runtime, dim: int, rng: np.random.Generator) -> None:
    """GGUF 量化路径：q4_0 / iq4nl 的原始字节直接进 DotLinear。"""
    ws = Workspace(rt)
    n, k = 16, max(64, dim)
    x = rng.standard_normal((2, k), dtype=np.float32)

    def synth(nblocks: int, block_bytes: int) -> bytes:
        r = bytearray(rng.integers(0, 256, size=nblocks * block_bytes,
                                   dtype=np.uint8).tobytes())
        for i in range(nblocks):
            r[i * block_bytes:i * block_bytes + 2] = np.float16(0.02).tobytes()
        return bytes(r)

    cases = [("q4_0", 32, 18, dequant_q4_0_fast),
             ("iq4nl", 32, 18, dequant_iq4_nl)]
    for kind, elems, block_bytes, ref_fn in cases:
        raw = synth(n * (k // elems), block_bytes)
        ref = ref_fn(raw).reshape(n, k).astype(np.float32)
        lin = DotLinear(rt, n, k, kind, raw, (), ws, kind)
        out = lin.forward(x)
        err = _rel(out, x @ ref.T)
        if err > 1e-4:
            FAILURES.append(kind)
        print(f"[gguf] {kind:6s} n={n} k={k} max_rel={err:.2e}")
    ws.free()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--moe-exp", type=int, default=4)
    args = ap.parse_args()

    rng = np.random.default_rng(2026)
    rt = Runtime()
    run_f32(rt, args.rows, args.dim, args.ffn, rng)
    run_int4(rt, args.rows, args.dim, args.ffn, rng)
    run_moe(rt, args.rows, args.dim, args.moe_exp, rng)
    run_gguf(rt, args.dim, rng)
    if FAILURES:
        print("失败:", ", ".join(FAILURES))
        return 1
    print("全部对账通过 ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
