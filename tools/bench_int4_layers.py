#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按层类型实测 `int4_dot_k`，再推算整模型每 token 的权重流时间。

checkpoint 里 400 个量化线性层只有少数几种形状（gate/up、down、q/k/v/o、
线性注意力的 in_proj/out_proj），每种形状的 `int4_dot_k` 表现不同（K 越大
激活复用越差、`nbpr` 越大归约越长）。这个脚本：

1. 用 `tools/ct_int4.py` 枚举所有量化层，按 `(N, K)` 分组统计字节数；
2. 每种形状挑一个层，把 `weight_packed`/`weight_scale` 原样搬到显存，
   跑 `int4_dot_k` + `reduce_blocks_k`，报耗时与等效权重带宽；
3. 按「该形状的总字节 / 该形状的带宽」推算整模型每 token 的权重流时间。

    python3 tools/bench_int4_layers.py                 # 全部形状
    python3 tools/bench_int4_layers.py --iters 50
    python3 tools/bench_int4_layers.py --classes 4     # 只看最大的 4 种
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
from ct_int4 import DEFAULT_MODEL_DIR, Int4Linear, list_linears, read_header  # noqa: E402
from gen_gemv_qdot import div_magic  # noqa: E402
from k100lc_kernels import Runtime  # noqa: E402

BLOCK_BYTES = 64          # 128 个 4bit 权重


def layer_bytes(hdr: dict, name: str) -> int:
    tot = 0
    for suf in (".weight_packed", ".weight_scale"):
        off = hdr[name + suf]["data_offsets"]
        tot += off[1] - off[0]
    return tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--classes", type=int, default=0, help="只测最大的 N 种形状")
    args = ap.parse_args()

    hdr = read_header(args.model_dir / "model.safetensors")
    names = list_linears(args.model_dir)

    groups: dict[tuple[int, int], list[str]] = collections.defaultdict(list)
    for nm in names:
        shape = tuple(int(v) for v in hdr[nm + ".weight_packed"]["shape"])
        groups[(shape[0], shape[1] * 8)].append(nm)

    classes = sorted(groups.items(), key=lambda kv: -sum(layer_bytes(hdr, n)
                                                         for n in kv[1]))
    if args.classes:
        classes = classes[: args.classes]
    total_bytes = sum(layer_bytes(hdr, n) for n in names)
    print(f"{args.model_dir}: {len(names)} 个量化层，共 {total_bytes/1e9:.3f} GB，"
          f"{len(classes)} 种形状（按字节降序）")

    rt = Runtime()
    est_s = 0.0
    for (n, k), members in classes:
        rep = members[0]
        lin = Int4Linear(rep, args.model_dir)
        nbpr = lin.nbpr
        nblocks = lin.n * nbpr
        w_raw = lin.packed_rows(0, lin.n)
        s_raw = lin.scale_rows(0, lin.n)
        x = np.random.default_rng(0).standard_normal(k).astype(np.float32)
        ids = np.arange(lin.n, dtype=np.uint32)
        pw, ps, px, pp, py, pid = (rt.alloc(len(w_raw)), rt.alloc(len(s_raw)),
                                   rt.alloc(k * 4), rt.alloc(nblocks * 4),
                                   rt.alloc(lin.n * 4), rt.alloc(ids.nbytes))
        rt.upload(pw, w_raw)
        rt.upload(ps, s_raw)
        rt.upload(px, x)
        rt.upload(pid, ids)
        m1, m2 = div_magic(nbpr, nblocks), div_magic(lin.n, lin.n + 1)
        argv = [pw, px, pp, nblocks, 64, nbpr, m1, lin.n, m2, ps, pid,
                nbpr * BLOCK_BYTES, lin.n, m2]

        def run() -> None:
            rt.launch("int4_dot_k", (nblocks + 63) // 64, 64, argv)
            rt.launch("reduce_blocks_k", (lin.n + 63) // 64, 64, [pp, py, lin.n, nbpr])

        run()
        rt.sync()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            run()
        rt.sync()
        dt = (time.perf_counter() - t0) / args.iters

        nbytes = sum(layer_bytes(hdr, nm) for nm in members)
        ms = dt * 1e3
        gbs = len(w_raw) / 1e9 / dt
        print(f"  N={n:6d} K={k:6d} nbpr={nbpr:4d} × {len(members):3d} 层 "
              f"({nbytes/1e9:6.2f} GB): {ms:7.3f} ms/层 → {gbs:6.1f} GB/s"
              f"    {rep}")
        est_s += nbytes / (len(w_raw) / dt)   # 该形状的字节 / 该形状的字节率
        for p in (pw, ps, px, pp, py, pid):
            rt.free(p)

    print(f"整模型权重流推算：{est_s*1e3:.1f} ms/token（≈ {1.0/est_s:.1f} tok/s 上限，"
          f"只算权重搬运，不含 attention/归一化/采样等其它开销）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
