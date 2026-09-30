#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LayerNorm（warp-per-row，LDS 归约）+ 自检。

签名：
    layernorm_k(float* y, const float* x, const float* w, const float* b,
                uint32_t rows, uint32_t cols, float eps, uint32_t group_size)
grid=rows, workgroup=64；要求 cols % 64 == 0。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "layernorm_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16), buffer_arg(24),
        scalar_arg(32, 4), scalar_arg(36, 4), scalar_arg(40, 4), scalar_arg(44, 4)]
KERNARG_SIZE = 48


def _lds_reduce(L, op, off, reg):
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


def _broadcast(L, reg, slot):
    L += [
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[2:3], vcc",
        f"v_mov_b32_e32 v22, {slot}",
        f"ds_write_b32 v22, {reg}",
        "s_or_b64 exec, exec, s[2:3]",
        "s_barrier",
        f"v_mov_b32_e32 v22, {slot}",
        f"ds_read_b32 v{17 if reg == 'v15' else 18}, v22",
        "s_waitcnt lgkmcnt(0)",
    ]


def gen_asm() -> str:
    L = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",   # y
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",   # x
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",  # w
        "s_load_dwordx2 s[22:23], s[4:5], 0x18",  # b
        "s_load_dword s24, s[4:5], 0x20",         # rows
        "s_load_dword s25, s[4:5], 0x24",         # cols
        "s_load_dword s26, s[4:5], 0x28",         # eps (f32)
        "s_load_dword s27, s[4:5], 0x2c",         # group_size
        "s_waitcnt lgkmcnt(0)",
        "s_lshr_b32 s28, s25, 6",
        "v_mov_b32_e32 v3, 0",
        "v_mov_b32_e32 v1, s6",
        "v_mul_lo_u32 v1, v1, s25",
        "v_lshlrev_b32_e32 v1, 2, v1",
        "v_mov_b32_e32 v5, s19",
        "v_add_co_u32_e32 v4, vcc, s18, v1",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",   # x row
        "v_mov_b32_e32 v7, s17",
        "v_add_co_u32_e32 v6, vcc, s16, v1",
        "v_addc_co_u32_e32 v7, vcc, v7, v3, vcc",   # y row
        "v_mov_b32_e32 v15, 0",                     # sum
        "v_mov_b32_e32 v16, 0",                     # sumsq
        "s_mov_b32 s29, 0",
        "L_sum:",
        "v_mov_b32_e32 v9, s29",
        "v_lshlrev_b32_e32 v9, 6, v9",
        "v_add_u32_e32 v9, v9, v0",
        "v_lshlrev_b32_e32 v10, 2, v9",
        "v_mov_b32_e32 v12, v5",
        "v_add_co_u32_e32 v11, vcc, v4, v10",
        "v_addc_co_u32_e32 v12, vcc, v12, v3, vcc",
        "global_load_dword v13, v[11:12], off",
        "s_waitcnt vmcnt(0)",
        "v_add_f32_e32 v15, v15, v13",
        "v_fma_f32 v16, v13, v13, v16",
        "v_mov_b32_e32 v12, v7",
        "v_add_co_u32_e32 v11, vcc, v6, v10",
        "v_addc_co_u32_e32 v12, vcc, v12, v3, vcc",
        "global_store_dword v[11:12], v13, off",
        "s_add_i32 s29, s29, 1",
        "s_cmp_lt_u32 s29, s28",
        "s_cbranch_scc1 L_sum",
        # reduce sum
        "v_lshlrev_b32_e32 v11, 2, v0",
        "ds_write_b32 v11, v15",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
    ]
    for off in (32, 16, 8, 4, 2, 1):
        _lds_reduce(L, "v_add_f32_e32", off, "v15")
    _broadcast(L, "v15", 256)
    L += [
        # reduce sumsq
        "v_lshlrev_b32_e32 v11, 2, v0",
        "ds_write_b32 v11, v16",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
    ]
    for off in (32, 16, 8, 4, 2, 1):
        _lds_reduce(L, "v_add_f32_e32", off, "v16")
    _broadcast(L, "v16", 260)
    L += [
        # inv_cols / mean / var / rsqrt
        "v_cvt_f32_u32_e32 v19, s25",
        "s_nop 0",
        "v_rcp_f32_e32 v19, v19",
        "s_nop 0",
        "v_mul_f32_e32 v20, v17, v19",           # mean
        "v_mul_f32_e32 v21, v18, v19",           # E[x^2]
        "v_fma_f32 v21, -v20, v20, v21",         # var
        "v_mov_b32_e32 v23, s26",
        "v_add_f32_e32 v21, v21, v23",           # + eps
        "s_nop 0",
        "v_rsq_f32_e32 v21, v21",
        "s_nop 0",
        # normalize
        "s_mov_b32 s29, 0",
        "L_norm:",
        "v_mov_b32_e32 v9, s29",
        "v_lshlrev_b32_e32 v9, 6, v9",
        "v_add_u32_e32 v9, v9, v0",
        "v_lshlrev_b32_e32 v10, 2, v9",
        "v_mov_b32_e32 v12, v7",
        "v_add_co_u32_e32 v11, vcc, v6, v10",
        "v_addc_co_u32_e32 v12, vcc, v12, v3, vcc",
        "global_load_dword v13, v[11:12], off",
        "s_waitcnt vmcnt(0)",
        "v_sub_f32_e32 v13, v13, v20",
        "v_mul_f32_e32 v13, v13, v21",
        # * w[i]
        "v_mov_b32_e32 v12, s21",
        "v_add_co_u32_e32 v11, vcc, s20, v10",
        "v_addc_co_u32_e32 v12, vcc, v12, v3, vcc",
        "global_load_dword v14, v[11:12], off",
        "s_waitcnt vmcnt(0)",
        "v_mul_f32_e32 v13, v13, v14",
        # + b[i]
        "v_mov_b32_e32 v12, s23",
        "v_add_co_u32_e32 v11, vcc, s22, v10",
        "v_addc_co_u32_e32 v12, vcc, v12, v3, vcc",
        "global_load_dword v14, v[11:12], off",
        "s_waitcnt vmcnt(0)",
        "v_add_f32_e32 v13, v13, v14",
        # 写回 y row（v11:12 此时指向 b，需要重算 y）
        "v_mov_b32_e32 v12, v7",
        "v_add_co_u32_e32 v11, vcc, v6, v10",
        "v_addc_co_u32_e32 v12, vcc, v12, v3, vcc",
        "global_store_dword v[11:12], v13, off",
        "s_add_i32 s29, s29, 1",
        "s_cmp_lt_u32 s29, s28",
        "s_cbranch_scc1 L_norm",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


def main() -> int:
    rows, cols = 32, 2560
    rng = np.random.default_rng(7)
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    w = rng.standard_normal(cols, dtype=np.float32)
    b = rng.standard_normal(cols, dtype=np.float32)
    eps = 1e-5
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    ref = (x - mean) / np.sqrt(var + eps) * w + b
    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE,
                      group_segment=1024, tag="layernorm")
    out = run_one(hsaco, NAME,
                  [{"buffer": "y"}, {"buffer": "x"}, {"buffer": "w"}, {"buffer": "b"},
                   {"scalar": {"dtype": "u32", "value": rows}},
                   {"scalar": {"dtype": "u32", "value": cols}},
                   {"scalar": {"dtype": "f32", "value": eps}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"y": {"dtype": "f32", "values": [0.0] * (rows * cols)},
                   "x": {"dtype": "f32", "values": x.reshape(-1).tolist()},
                   "w": {"dtype": "f32", "values": w.tolist()},
                   "b": {"dtype": "f32", "values": b.tolist()}},
                  grid=rows * 64, workgroup=64)
    got = np.array(out["y"], dtype=np.float32).reshape(rows, cols)
    d = np.abs(got - ref)
    print(f"layernorm max_abs={d.max():.3e}")
    ok = d.max() < 1e-4
    print("GPU LayerNorm vs numpy：", "一致 ✔" if ok else "不一致 ✘")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
