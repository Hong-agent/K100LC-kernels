#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""行 softmax（warp-per-row，LDS 树形归约）+ 自检。

签名：softmax_k(float* y, const float* x, uint32_t rows, uint32_t cols, uint32_t group_size)
grid=rows, workgroup=64；要求 cols % 64 == 0。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "softmax_k"
ARGS = [buffer_arg(0), buffer_arg(8), scalar_arg(16, 4),
        scalar_arg(20, 4), scalar_arg(24, 4)]
KERNARG_SIZE = 32


def _lds_reduce(L, op, off, reg="v8"):
    L += [
        f"v_cmp_gt_u32_e32 vcc, {off}, v0",
        "s_and_saveexec_b64 s[2:3], vcc",
        f"v_mov_b32_e32 v19, {off * 4}",
        "v_add_u32_e32 v19, v11, v19",
        "ds_read_b32 v18, v19",
        "s_waitcnt lgkmcnt(0)",
        f"{op} {reg}, {reg}, v18",
        f"ds_write_b32 v11, {reg}",
        "s_or_b64 exec, exec, s[2:3]",
        "s_barrier",
    ]


def gen_asm() -> str:
    L = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",   # y
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",   # x
        "s_load_dword s20, s[4:5], 0x10",         # rows
        "s_load_dword s21, s[4:5], 0x14",         # cols
        "s_load_dword s22, s[4:5], 0x18",         # group_size
        "s_waitcnt lgkmcnt(0)",
        "s_lshr_b32 s22, s21, 6",                 # cols/64
        "v_mov_b32_e32 v3, 0",
        # row offset = blockIdx.x * cols * 4
        "v_mov_b32_e32 v1, s6",
        "v_mul_lo_u32 v1, v1, s21",
        "v_lshlrev_b32_e32 v1, 2, v1",
        "v_mov_b32_e32 v5, s19",
        "v_add_co_u32_e32 v4, vcc, s18, v1",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",   # x row
        "v_mov_b32_e32 v7, s17",
        "v_add_co_u32_e32 v6, vcc, s16, v1",
        "v_addc_co_u32_e32 v7, vcc, v7, v3, vcc",   # y row
        # max
        "v_mov_b32_e32 v8, 0",
        "s_mov_b32 s23, 0",
        "L_max:",
        "v_mov_b32_e32 v9, s23",
        "v_lshlrev_b32_e32 v9, 6, v9",
        "v_add_u32_e32 v9, v9, v0",
        "v_lshlrev_b32_e32 v10, 2, v9",
        "v_mov_b32_e32 v12, v5",
        "v_add_co_u32_e32 v11, vcc, v4, v10",
        "v_addc_co_u32_e32 v12, vcc, v12, v3, vcc",
        "global_load_dword v13, v[11:12], off",
        "s_waitcnt vmcnt(0)",
        "v_max_f32_e32 v8, v8, v13",
        "s_add_i32 s23, s23, 1",
        "s_cmp_lt_u32 s23, s22",
        "s_cbranch_scc1 L_max",
        # LDS max reduce + broadcast
        "v_lshlrev_b32_e32 v11, 2, v0",
        "ds_write_b32 v11, v8",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
    ]
    for off in (32, 16, 8, 4, 2, 1):
        _lds_reduce(L, "v_max_f32_e32", off, "v8")
    L += [
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[2:3], vcc",
        "v_mov_b32_e32 v16, 256",
        "ds_write_b32 v16, v8",
        "s_or_b64 exec, exec, s[2:3]",
        "s_barrier",
        "v_mov_b32_e32 v16, 256",
        "ds_read_b32 v14, v16",
        "s_waitcnt lgkmcnt(0)",
        # exp + sum + store
        "v_mov_b32_e32 v15, 0",
        "s_mov_b32 s23, 0",
        "L_exp:",
        "v_mov_b32_e32 v9, s23",
        "v_lshlrev_b32_e32 v9, 6, v9",
        "v_add_u32_e32 v9, v9, v0",
        "v_lshlrev_b32_e32 v10, 2, v9",
        "v_mov_b32_e32 v12, v5",
        "v_add_co_u32_e32 v11, vcc, v4, v10",
        "v_addc_co_u32_e32 v12, vcc, v12, v3, vcc",
        "global_load_dword v13, v[11:12], off",
        "s_waitcnt vmcnt(0)",
        "v_sub_f32_e32 v13, v13, v14",
        "v_mov_b32_e32 v17, 0x3fb8aa3b",          # log2(e)
        "v_mul_f32_e32 v13, v13, v17",
        "s_nop 0",
        "v_exp_f32_e32 v13, v13",
        "s_nop 0",
        "v_add_f32_e32 v15, v15, v13",
        "v_mov_b32_e32 v12, v7",
        "v_add_co_u32_e32 v11, vcc, v6, v10",
        "v_addc_co_u32_e32 v12, vcc, v12, v3, vcc",
        "global_store_dword v[11:12], v13, off",
        "s_add_i32 s23, s23, 1",
        "s_cmp_lt_u32 s23, s22",
        "s_cbranch_scc1 L_exp",
        # LDS sum reduce + broadcast
        "v_lshlrev_b32_e32 v11, 2, v0",
        "ds_write_b32 v11, v15",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
    ]
    for off in (32, 16, 8, 4, 2, 1):
        _lds_reduce(L, "v_add_f32_e32", off, "v15")
    L += [
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[2:3], vcc",
        "v_mov_b32_e32 v16, 260",
        "ds_write_b32 v16, v15",
        "s_or_b64 exec, exec, s[2:3]",
        "s_barrier",
        "v_mov_b32_e32 v16, 260",
        "ds_read_b32 v15, v16",
        "s_waitcnt lgkmcnt(0)",
        "s_nop 0",
        "v_rcp_f32_e32 v17, v15",
        "s_nop 0",
        # normalize
        "s_mov_b32 s23, 0",
        "L_norm:",
        "v_mov_b32_e32 v9, s23",
        "v_lshlrev_b32_e32 v9, 6, v9",
        "v_add_u32_e32 v9, v9, v0",
        "v_lshlrev_b32_e32 v10, 2, v9",
        "v_mov_b32_e32 v12, v7",
        "v_add_co_u32_e32 v11, vcc, v6, v10",
        "v_addc_co_u32_e32 v12, vcc, v12, v3, vcc",
        "global_load_dword v13, v[11:12], off",
        "s_waitcnt vmcnt(0)",
        "v_mul_f32_e32 v13, v13, v17",
        "global_store_dword v[11:12], v13, off",
        "s_add_i32 s23, s23, 1",
        "s_cmp_lt_u32 s23, s22",
        "s_cbranch_scc1 L_norm",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


def main() -> int:
    rows, cols = 32, 2560
    rng = np.random.default_rng(6)
    x = rng.standard_normal((rows, cols), dtype=np.float32) * 2.0
    ref = np.exp(x - x.max(axis=1, keepdims=True))
    ref /= ref.sum(axis=1, keepdims=True)
    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE,
                      group_segment=1024, tag="softmax")
    out = run_one(hsaco, NAME,
                  [{"buffer": "y"}, {"buffer": "x"},
                   {"scalar": {"dtype": "u32", "value": rows}},
                   {"scalar": {"dtype": "u32", "value": cols}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"y": {"dtype": "f32", "values": [0.0] * (rows * cols)},
                   "x": {"dtype": "f32", "values": x.reshape(-1, order="C").tolist()}},
                  grid=rows * 64, workgroup=64)
    got = np.array(out["y"], dtype=np.float32).reshape(rows, cols)
    d = np.abs(got - ref)
    print(f"softmax max_abs={d.max():.3e}")
    ok = d.max() < 1e-5
    print("GPU softmax vs numpy：", "一致 ✔" if ok else "不一致 ✘")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
