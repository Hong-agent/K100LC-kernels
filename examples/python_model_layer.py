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

from k100lc_kernels import (DotLinear, F32Linear, Int4Linear, MLP, MoECombine,  # noqa: E402
                            Attention, MoEExperts, RMSNorm, RoPE, Runtime,
                            Workspace, dequant_int4_group128, pack_int4_group128)
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


def run_moe_bucketed(rt: Runtime, rows: int, dim: int, n_exp: int,
                     rng: np.random.Generator) -> None:
    """token→expert 分桶 MoE：与稠密「所有专家 × 全部行」对账并比时间。"""
    topk = min(2, n_exp)
    ws = Workspace(rt)
    x = rng.standard_normal((rows, dim), dtype=np.float32)
    wts = [rng.normal(0, 0.05, (dim, dim)).astype(np.float32)
           for _ in range(n_exp)]
    ids = rng.integers(0, n_exp, size=(rows, topk))
    weights = rng.random((rows, topk), dtype=np.float32)
    ref = np.stack([
        sum(weights[r, e] * (x[r] @ wts[ids[r, e]].T) for e in range(topk))
        for r in range(rows)])

    experts = [F32Linear(rt, dim, dim, wts[e], ws, f"moe_exp{e}")
               for e in range(n_exp)]
    moe = MoEExperts(rt, experts, rows, dim, dim, topk, ws, "moe_bucket")
    out = moe.forward(x, ids, weights)
    err = _rel(out, ref)
    if err > 1e-4:
        FAILURES.append("moe_bucketed")
    t_bucket = _bench(lambda: moe.forward(x, ids, weights), 5)

    # 稠密参照：每个专家都跑全部行，再用 moe_combine_k 合并
    dense = MoECombine(rt, rows, dim, n_exp, ws, "moe_dense")
    all_w = np.zeros((rows, n_exp), dtype=np.float32)
    for r in range(rows):
        for e in range(topk):
            all_w[r, ids[r, e]] += weights[r, e]
    px = ws.buffer("moe_bucket.x", x.nbytes)
    rt.upload(px, x.reshape(-1))

    def dense_run():
        for e, expert in enumerate(experts):
            expert.forward_device(px, rows, out_dev=dense.slot(e), sync=False)
        dense.rt.upload(dense.weights, all_w.reshape(-1))
        dense.forward_device(sync=True)

    t_dense = _bench(dense_run, 5)
    print(f"[moe ] rows={rows} dim={dim} n_exp={n_exp} topk={topk} "
          f"max_rel={err:.2e}  bucketed={t_bucket:.3f} ms  dense={t_dense:.3f} ms "
          f"speedup={t_dense/t_bucket:.2f}x")
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


def run_int4_fast(rt: Runtime, rows: int, dim: int, ffn: int,
                  rng: np.random.Generator) -> None:
    """INT4 的三条通路：W4A16（原字节）/ W4A8 / W4A4（转换后）。

    参考 = INT4 反量化权重 × f32 激活；W4A8/W4A4 的误差只剩激活量化。
    """
    ws = Workspace(rt)
    x = rng.standard_normal((rows, dim), dtype=np.float32)
    w = rng.normal(0, 0.05, (ffn, dim)).astype(np.float32)
    packed, scales = pack_int4_group128(w)
    ref = x @ dequant_int4_group128(packed, scales, ffn, dim).T
    cases = [
        ("w4a16", DotLinear(rt, ffn, dim, "int4", packed, (scales,), ws, "w4a16"), 1e-4),
        ("w4a8", Int4Linear(rt, ffn, dim, packed, scales, "w4a8", ws, "w4a8"), 3e-2),
        ("w4a4", Int4Linear(rt, ffn, dim, packed, scales, "w4a4", ws, "w4a4"), 3e-1),
    ]
    for name, layer, tol in cases:
        out = layer.forward(x)
        err = _rel(out, ref)
        if err > tol:
            FAILURES.append(name)
        t = _bench(lambda: layer.forward(x), 20)
        print(f"[{name:5s}] rows={rows} n={ffn} k={dim} max_rel={err:.2e} "
              f"time={t:.3f} ms")
    ws.free()


def run_rope(rt: Runtime, dim: int, rng) -> None:
    """RoPE（rotate-half）与 NumPy 参考对账 + 计时。"""
    rope = RoPE(rt, dim=dim, max_len=256, tag="rope")
    for rows, pos in ((1, 0), (1, 63), (7, 5), (64, 100), (200, 0)):
        if pos + rows > 256:
            continue
        x = rng.standard_normal((rows, dim)).astype(np.float32)
        got = rope.forward(x, pos=pos)
        half = dim // 2
        j = np.arange(half)
        theta = 10000.0 ** (-2.0 * j / dim)
        p = np.arange(pos, pos + rows)[:, None] * theta[None, :]
        c = np.cos(p).astype(np.float32)
        s = np.sin(p).astype(np.float32)
        a, b = x[:, :half], x[:, half:]
        ref = np.concatenate([a * c - b * s, a * s + b * c], axis=1)
        err = _rel(got, ref)
        if err > 1e-5:
            FAILURES.append(f"rope(rows={rows}, pos={pos})")
        t = _bench(lambda: rope.forward(x, pos=pos), 20)
        print(f"[rope ] rows={rows} pos={pos} dim={dim} max_rel={err:.2e} "
              f"time={t * 1e3:.1f} us")
    rope.ws.free()


def run_attention(rt: Runtime, dim: int, max_len: int, rng) -> None:
    """解码注意力（`Attention`）与 NumPy 参考对账 + 计时。

    覆盖：`n_kv` 不是 256 的倍数（走尾部 -1e30 掩码）、`pad` 不能被切块数整除、
    缓存远没填满（代价只按当前长度算）、以及长上下文（pad ≥ 2048 时 softmax 走
    分块路：`block_max_k`/`reduce_max1_k`/`block_exp_sum_k`/`reduce_sum1_k` +
    输出侧 `div_scalar_k`；`Vt·P` 走 `gemv_f32_rows8_split_k`）。
    """
    # 100/512：尾部掩码；300/1024：pad=512 的切块整除；1000/16384：缓存没填满；
    # 3000/4096：pad=3072（分块 + 掩码，且 Vt 行距 4096 > 列数）；4096/4096：整块
    for n_kv, max_len_ in ((100, 512), (300, 1024), (1000, 16384),
                           (3000, 4096), (4096, 4096)):
        max_len = max(max_len_, n_kv)
        attn = Attention(rt, dim=dim, max_len=max_len, tag=f"attn{n_kv}")
        k = rng.standard_normal((n_kv, dim)).astype(np.float32)
        v = rng.standard_normal((n_kv, dim)).astype(np.float32)
        q = rng.standard_normal(dim).astype(np.float32)
        attn.append(k, v)
        got = attn.forward(q)
        s = (k @ q) / np.sqrt(dim)
        e = np.exp(s - s.max())
        ref = (e / e.sum()) @ v
        err = _rel(got, ref)
        if err > 1e-5:
            FAILURES.append(f"attention(n_kv={n_kv})")
        t = _bench(lambda: attn.forward(q), 10)
        print(f"[attn ] n_kv={n_kv} dim={dim} max_rel={err:.2e} time={t * 1e3:.1f} us")
        attn.ws.free()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--moe-exp", type=int, default=4)
    args = ap.parse_args()

    rng = np.random.default_rng(2026)
    rt = Runtime()
    run_rope(rt, 128, rng)
    run_attention(rt, 128, 640, rng)
    run_f32(rt, args.rows, args.dim, args.ffn, rng)
    run_int4(rt, args.rows, args.dim, args.ffn, rng)
    run_moe(rt, args.rows, args.dim, args.moe_exp, rng)
    run_moe_bucketed(rt, max(16, args.rows * 4), args.dim, args.moe_exp, rng)
    run_gguf(rt, args.dim, rng)
    run_int4_fast(rt, args.rows, args.dim, args.ffn, rng)
    if FAILURES:
        print("失败:", ", ".join(FAILURES))
        return 1
    print("全部对账通过 ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
