#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""行 top-k（KMAX=16，一行一个线程，插入式）+ 自检。

签名：
    topk_k(const float* x, int32_t* idx, float* val,
           uint32_t rows, uint32_t cols, uint32_t k)
grid=ceil(rows/64), workgroup=64；k<=16。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "topk_k"
KMAX = 16
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
        scalar_arg(24, 4), scalar_arg(28, 4), scalar_arg(32, 4)]
KERNARG_SIZE = 40


def gen_asm() -> str:
    topv = [10 + i for i in range(KMAX)]
    topi = [30 + i for i in range(KMAX)]
    L = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",   # x
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",   # idx
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",  # val
        "s_load_dword s22, s[4:5], 0x18",         # rows
        "s_load_dword s23, s[4:5], 0x1c",         # cols
        "s_load_dword s24, s[4:5], 0x20",         # k
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v26, 0",
        "v_mov_b32_e32 v1, s6",
        "v_lshlrev_b32_e32 v1, 6, v1",
        "v_add_u32_e32 v1, v1, v0",               # row
        "v_cmp_gt_u32_e32 vcc, s22, v1",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        # xrow = x + row*cols*4
        "v_mov_b32_e32 v3, s23",
        "v_mul_lo_u32 v2, v1, v3",
        "v_lshlrev_b32_e32 v2, 2, v2",
        "v_mov_b32_e32 v5, s17",
        "v_add_co_u32_e32 v4, vcc, s16, v2",
        "v_addc_co_u32_e32 v5, vcc, v5, v26, vcc",
    ]
    for i in range(KMAX):
        L.append(f"v_mov_b32_e32 v{topv[i]}, 0xff800000")
        L.append(f"v_mov_b32_e32 v{topi[i]}, 0")
    L += [
        "s_mov_b32 s25, 0",
        "L_col:",
        "s_cmp_lt_u32 s25, s23",
        "s_cbranch_scc0 L_done_col",
        "v_mov_b32_e32 v7, s25",
        "v_lshlrev_b32_e32 v7, 2, v7",
        "v_mov_b32_e32 v251, v5",
        "v_add_co_u32_e32 v250, vcc, v4, v7",
        "v_addc_co_u32_e32 v251, vcc, v251, v26, vcc",
        "global_load_dword v6, v[250:251], off",
        "s_waitcnt vmcnt(0)",
        "v_mov_b32_e32 v7, s25",                  # 当前下标
    ]
    for i in range(KMAX):
        v, ix = topv[i], topi[i]
        L += [
            f"v_cmp_lt_f32_e32 vcc, v{v}, v6",          # v < new  ⇔ new > v
            # v_cndmask dst = vcc ? src1 : src0
            f"v_cndmask_b32_e32 v8, v{v}, v6, vcc",     # new_top
            f"v_cndmask_b32_e32 v6, v6, v{v}, vcc",     # new leftover
            f"v_mov_b32_e32 v{v}, v8",
            f"v_cndmask_b32_e32 v9, v{ix}, v7, vcc",
            f"v_cndmask_b32_e32 v7, v7, v{ix}, vcc",
            f"v_mov_b32_e32 v{ix}, v9",
        ]
    L += [
        "s_add_i32 s25, s25, 1",
        "s_branch L_col",
        "L_done_col:",
        # row*k
        "v_mov_b32_e32 v27, s24",
        "v_mul_lo_u32 v28, v1, v27",
    ]
    for i in range(KMAX):
        L += [
            f"s_mov_b32 s26, {i}",
            "s_cmp_lt_u32 s26, s24",
            f"s_cbranch_scc0 L_skip_{i}",
            (f"v_mov_b32_e32 v29, v28" if i == 0
             else f"v_add_u32_e32 v29, {i}, v28"),
            "v_lshlrev_b32_e32 v29, 2, v29",
            "v_mov_b32_e32 v253, s21",
            "v_add_co_u32_e32 v252, vcc, s20, v29",
            "v_addc_co_u32_e32 v253, vcc, v253, v26, vcc",
            f"global_store_dword v[252:253], v{topv[i]}, off",
            "v_mov_b32_e32 v253, s19",
            "v_add_co_u32_e32 v252, vcc, s18, v29",
            "v_addc_co_u32_e32 v253, vcc, v253, v26, vcc",
            f"global_store_dword v[252:253], v{topi[i]}, off",
            f"L_skip_{i}:",
        ]
    L += ["L_end:", "s_or_b64 exec, exec, s[0:1]", "s_endpgm"]
    return "\n".join(L) + "\n"


def main() -> int:
    rows, cols, k = 16, 512, 10
    rng = np.random.default_rng(8)
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    ref_ix = np.argsort(-x, axis=1)[:, :k]
    ref_val = np.take_along_axis(x, ref_ix, axis=1)
    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="topk")
    out = run_one(hsaco, NAME,
                  [{"buffer": "x"}, {"buffer": "idx"}, {"buffer": "val"},
                   {"scalar": {"dtype": "u32", "value": rows}},
                   {"scalar": {"dtype": "u32", "value": cols}},
                   {"scalar": {"dtype": "u32", "value": k}}],
                  {"x": {"dtype": "f32", "values": x.reshape(-1).tolist()},
                   "idx": {"dtype": "i32", "values": [0] * (rows * k)},
                   "val": {"dtype": "f32", "values": [0.0] * (rows * k)}},
                  grid=rows * 64, workgroup=64)
    got_ix = np.array(out["idx"], dtype=np.int32).reshape(rows, k)
    got_val = np.array(out["val"], dtype=np.float32).reshape(rows, k)
    ok = np.array_equal(got_ix, ref_ix) and np.allclose(got_val, ref_val)
    print("topk idx equal:", np.array_equal(got_ix, ref_ix))
    print("topk max_abs:", float(np.abs(got_val - ref_val).max()))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
