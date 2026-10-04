#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`rmsnorm_deep_k`：`rmsnorm_fast_k` 的**加深流水**变体（一次发 16 条 load）。

同一套算法、同一份 ABI、**数值与 `rmsnorm_fast_k` 逐位相同**（NB 只改一次发
几条 load，Σx² 的加法顺序不变）。差别只在访存往返数：
单 workgroup 处理整行 `dim`，每 lane 要读 `dim/64` 个元素，
`rmsnorm_fast_k`（NB=4）要 `dim/256` 次完整访存往返，NB=16 只要 1/4。

本机实测（rows=1，单次 launch + 同步，取 400 次均值）：

| dim | rmsnorm_fast_k (NB=4) | rmsnorm_deep_k (NB=16) |
|---|---:|---:|
| 128 | ~7.5 us | ~7.5 us（主循环不跑，两者等价） |
| 512 | 10.4 us | 18.2 us（**更慢**，主循环不跑、退化成逐条） |
| 2560 | 15.5 us | 16.3 us |
| 5120 | 26.5 us | **13.0 us** |
| 10240 | 48.2 us | **20.2 us** |

所以调用方按 `dim` 分派：`dim >= 2048` 用 deep，更小仍旧用 `rmsnorm_fast_k`。
消费方（FASTASM）的两种用法正好落在两端——整层 hidden（~5120）走 deep，
q/k norm 的 `dim = head_dim`（128）本来就不进主循环。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import gen_rmsnorm as G           # noqa: E402
from kernel_lab import build_one, run_one  # noqa: E402

NAME = "rmsnorm_deep_k"
_DEEP = {"nb": 16, "lx": 100, "lw": 120}

ARGS = G.ARGS
KERNARG_SIZE = G.KERNARG_SIZE


def gen_asm() -> str:
    return G.make_variant(NAME, **_DEEP)[1]


def selftest() -> int:
    hsaco = build_one(NAME, gen_asm(), ARGS, KERNARG_SIZE,
                      group_segment=G.GROUP_SEGMENT, tag="deep")
    rng = np.random.default_rng(20261012)
    rc = 0
    for rows, dim in ((1, 64), (3, 128), (4, 512), (2, 1024), (1, 2048),
                      (1, 4096), (1, 5120), (2, 10240)):
        x = rng.standard_normal((rows, dim)).astype(np.float32) * 2.0
        w = (rng.random(dim).astype(np.float32) + 0.5)
        eps = 1e-6
        o = run_one(hsaco, NAME,
                    [{"buffer": "y"}, {"buffer": "x"}, {"buffer": "w"},
                     {"scalar": {"dtype": "u32", "value": dim}},
                     {"scalar": {"dtype": "f32", "value": eps}}],
                    {"x": {"dtype": "f32", "values": x.reshape(-1).tolist()},
                     "w": {"dtype": "f32", "values": w.tolist()},
                     "y": {"dtype": "f32", "values": [0.0] * (rows * dim)}},
                    grid=rows * 64, workgroup=64)
        got = np.array(o["y"], np.float32).reshape(rows, dim)
        ref = G._ref(x, w, eps)
        rel = float(np.abs(got - ref).max() / max(1e-9, float(np.abs(ref).max())))
        ok = rel < 1e-5
        rc |= 0 if ok else 1
        print(f"rows={rows} dim={dim}: max_rel={rel:.2e} {'✔' if ok else '✘'}")
    return rc


def main() -> int:
    rc = selftest()
    print(f"{NAME}:", "全部一致 ✔" if rc == 0 else "有不一致 ✘")
    return rc


KERNELS = [(NAME, gen_asm, ARGS, KERNARG_SIZE)]

if __name__ == "__main__":
    raise SystemExit(main())
