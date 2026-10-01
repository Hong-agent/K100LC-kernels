#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真模型整层 GEMV：compressed-tensors INT4 checkpoint 原字节 → `int4_dot_k`。

不做任何转换/重量化：直接把 `weight_packed`（I32，8 个 4bit 码/字）和
`weight_scale`（BF16，每 128 个 k 一个）搬进显存，跑

    int4_dot_k    → 每个 (128 权重块 × 一行) 一个 work-item，写 partial
    reduce_blocks_k → 每行 nbpr 个 partial 求和成 y

然后和 `tools/ct_int4.py` 的参考解码（f32 矩阵乘）逐行对账，并报等效带宽。

    K100LC_INT4_MODEL=/path/to/checkpoint \
    python3 examples/python_int4_gemv.py                      # 默认 gate_proj（44.6 MB）
    python3 examples/python_int4_gemv.py --rows 1024 --iters 50
    python3 examples/python_int4_gemv.py --layer model.language_model.layers.0.mlp.down_proj

先 `source env.sh`（或自己保证 PYTHONPATH 含 `python/` 与 `tools/`）。
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
from ct_int4 import DEFAULT_MODEL_DIR, Int4Linear  # noqa: E402
from gen_gemv_qdot import div_magic  # noqa: E402
from k100lc_kernels import Runtime  # noqa: E402

GROUP = 128
BLOCK_BYTES = GROUP // 2           # 一个 128 权重块 = 64 字节


def _magic(d: int, max_i: int) -> int:
    """`(i * M) >> 32 == i // d`（内核用魔法乘代替整数除法）。"""
    return div_magic(d, max_i)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="model.language_model.layers.3.mlp.gate_proj")
    ap.add_argument("--rows", type=int, default=0, help="只跑前 N 行（0 = 全部）")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL_DIR)
    args = ap.parse_args()

    lin = Int4Linear(args.layer, args.model_dir)
    rows = args.rows or lin.n
    rows = min(rows, lin.n)
    nbpr = lin.nbpr                    # 一行的 128 权重块数 = K/128
    nblocks = rows * nbpr
    print(f"{args.layer}\n  N={lin.n} K={lin.k} nbpr={nbpr} → 取前 {rows} 行，"
          f"{nblocks} 个块，权重 {rows * nbpr * BLOCK_BYTES / 1e6:.1f} MB")

    w_raw = lin.packed_rows(0, rows)   # checkpoint 原字节，不解释
    s_raw = lin.scale_rows(0, rows)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(lin.k).astype(np.float32)

    rt = Runtime()
    pw = rt.alloc(len(w_raw))
    ps = rt.alloc(len(s_raw))
    px = rt.alloc(x.nbytes)
    pp = rt.alloc(nblocks * 4)
    py = rt.alloc(rows * 4)
    pid = rt.alloc(rows * 4)
    rt.upload(pw, w_raw)
    rt.upload(ps, s_raw)
    rt.upload(px, x)
    rt.upload(pid, np.arange(rows, dtype=np.uint32))

    # 魔法数（整数除法的乘数）只算一次：div_magic 会对整个 i 域穷举校验，
    # 放进计时循环里会把 CPU 时间算进去。
    m_nbpr = _magic(nbpr, nblocks)
    m_rows = _magic(rows, rows)
    argv = [pw, px, pp, nblocks, 64, nbpr, m_nbpr, rows, m_rows, ps, pid,
            nbpr * BLOCK_BYTES, rows, m_rows]

    def launch() -> None:
        # grid = workgroup 个数（不是总 work-item 数）：ceil(nblocks / 64)
        rt.launch("int4_dot_k", (nblocks + 63) // 64, 64, argv)
        rt.launch("reduce_blocks_k", (rows + 63) // 64, 64, [pp, py, rows, nbpr])

    launch()
    rt.sync()
    y = rt.download(py, rows, np.float32)

    # 参考：按 512 行分块解码（避免一次开几百 MB），和 f32 矩阵乘结果比
    ref = np.empty(rows, dtype=np.float32)
    for r0 in range(0, rows, 512):
        nr = min(512, rows - r0)
        ref[r0:r0 + nr] = lin.dequant_rows(r0, nr) @ x
    err = np.abs(y - ref)
    rel = float(err.max() / max(1.0, np.abs(ref).max()))
    print(f"  与参考（checkpoint 原始解码 → f32 矩阵乘）:")
    print(f"    max_abs={err.max():.3e}  相对={rel:.2e}  "
          f"{'一致 ✔' if rel < 1e-3 else '不一致 ✘'}")

    # 计时：权重字节 / 耗时 = 等效权重带宽
    launch()
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        launch()
    rt.sync()
    dt = (time.perf_counter() - t0) / args.iters
    gb = (rows * nbpr * BLOCK_BYTES + rows * nbpr * 2) / 1e9
    print(f"  {dt * 1e3:.3f} ms/次（{args.iters} 次平均）→ 等效权重带宽 "
          f"{gb / dt:.0f} GB/s")

    for p in (pw, ps, px, pp, py, pid):
        rt.free(p)
    return 0 if rel < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
