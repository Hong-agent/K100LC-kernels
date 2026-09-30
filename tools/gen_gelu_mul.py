#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GELU(gate) * up（tanh 近似）f32 逐元素内核 + 自检。

签名：
    gelu_mul_k(float* y, const float* gate, const float* up,
               uint32_t n, uint32_t group_size)

grid=ceil(n/64)，workgroup=64；group_size 固定传 64。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "gelu_mul_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
        scalar_arg(24, 4), scalar_arg(28, 4)]
KERNARG_SIZE = 32


def gen_asm() -> str:
    L = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",   # y
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",   # gate
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",  # up
        "s_load_dword s22, s[4:5], 0x18",         # n
        "s_load_dword s23, s[4:5], 0x1c",         # group_size
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v1, s6",
        "v_lshlrev_b32_e32 v1, 6, v1",            # blockIdx*64
        "v_add_u32_e32 v1, v1, v0",               # gid
        "v_cmp_gt_u32_e32 vcc, s22, v1",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        "v_lshlrev_b32_e32 v2, 2, v1",            # byte offset
        "v_mov_b32_e32 v3, 0",
        # gate
        "v_mov_b32_e32 v5, s19",
        "v_add_co_u32_e32 v4, vcc, s18, v2",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "global_load_dword v6, v[4:5], off",
        # up
        "v_mov_b32_e32 v5, s21",
        "v_add_co_u32_e32 v4, vcc, s20, v2",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "global_load_dword v7, v[4:5], off",
        "s_waitcnt vmcnt(0)",
        # tanh 近似
        "v_mul_f32_e32 v8, v6, v6",
        "v_mul_f32_e32 v9, v8, v6",
        "v_mov_b32_e32 v10, 0x3d372713",          # 0.044715
        "v_fma_f32 v9, v9, v10, v6",              # g + 0.044715 g^3
        "v_mov_b32_e32 v10, 0x3f4c422a",          # sqrt(2/pi)
        "v_mul_f32_e32 v9, v9, v10",
        "v_mov_b32_e32 v10, 0x4038aa3b",          # 2*log2(e)
        "v_mul_f32_e32 v9, v9, v10",
        "s_nop 0",
        "v_exp_f32_e32 v9, v9",                   # 2^z = e^{z*ln2}
        "s_nop 0",
        "v_mov_b32_e32 v10, 0x3f800000",          # 1.0
        "v_add_f32_e32 v9, v10, v9",              # 1 + e
        "s_nop 0",
        "v_rcp_f32_e32 v9, v9",
        "s_nop 0",
        "v_mov_b32_e32 v10, 0x40000000",          # 2.0
        "v_mul_f32_e32 v9, v9, v10",              # 2/(1+e)
        "v_mov_b32_e32 v10, 0x3f800000",
        "v_sub_f32_e32 v9, v10, v9",              # tanh
        "v_add_f32_e32 v9, v10, v9",              # 1 + tanh
        "v_mov_b32_e32 v10, 0x3f000000",          # 0.5
        "v_mul_f32_e32 v9, v9, v10",
        "v_mul_f32_e32 v9, v9, v6",               # gelu
        "v_mul_f32_e32 v9, v9, v7",               # * up
        "v_mov_b32_e32 v5, s17",
        "v_add_co_u32_e32 v4, vcc, s16, v2",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "global_store_dword v[4:5], v9, off",
        "L_end:",
        "s_or_b64 exec, exec, s[0:1]",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


def main() -> int:
    n = 4096
    rng = np.random.default_rng(3)
    g = rng.standard_normal(n).astype(np.float32)
    u = rng.standard_normal(n).astype(np.float32)
    ref = (0.5 * g * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) *
                                    (g + 0.044715 * g ** 3))) * u).astype(np.float32)
    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="gelu")
    out = run_one(hsaco, NAME,
                  [{"buffer": "y"}, {"buffer": "g"}, {"buffer": "u"},
                   {"scalar": {"dtype": "u32", "value": n}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"y": {"dtype": "f32", "values": [0.0] * n},
                   "g": {"dtype": "f32", "values": g.tolist()},
                   "u": {"dtype": "f32", "values": u.tolist()}},
                  grid=n, workgroup=64)
    got = np.array(out["y"], dtype=np.float32)
    d = np.abs(got - ref)
    print(f"gelu_mul max_abs={d.max():.3e} max_rel={(d / np.maximum(np.abs(ref), 1e-6)).max():.3e}")
    ok = d.max() < 1e-4 * max(1.0, float(np.abs(ref).max()))
    print("GPU GELU*up vs numpy：", "一致 ✔" if ok else "不一致 ✘")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
