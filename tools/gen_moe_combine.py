#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoE 专家输出加权合并内核：`moe_combine_k` + 自检。

语义：

    y[r, d] = sum_e weights[r, e] * exp_out[e, r, d]

布局：

    y        float32 [rows, dim]
    exp_out  float32 [n_exp, rows, dim]
    weights  float32 [rows, n_exp]

实现：**一个 workgroup 一行**（grid=rows, workgroup=64），lane 沿 dim 以
64 为步长循环；对每个 d 再内层循环 n_exp。这样不需要主机提供除法魔法数，
`dim` 取任何值都能用（旧版按元素切分，`dim=2048` 时魔法除无法精确表示）。

签名：
    moe_combine_k(float* y, const float* exp_out, const float* weights,
                  uint32_t n_exp, uint32_t rows, uint32_t dim)
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "moe_combine_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
        scalar_arg(24, 4), scalar_arg(28, 4), scalar_arg(32, 4)]
KERNARG_SIZE = 40


def gen_asm() -> str:
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{NAME}:")
    em("s_load_dwordx2 s[16:17], s[4:5], 0x0")     # y
    em("s_load_dwordx2 s[18:19], s[4:5], 0x8")     # exp_out
    em("s_load_dwordx2 s[20:21], s[4:5], 0x10")    # weights
    em("s_load_dword s22, s[4:5], 0x18")           # n_exp
    em("s_load_dword s23, s[4:5], 0x1c")           # rows
    em("s_load_dword s24, s[4:5], 0x20")           # dim
    em("s_waitcnt lgkmcnt(0)")
    em("s_cmp_lt_u32 s6, s23")
    em("s_cbranch_scc0 L_end")
    em("s_mul_i32 s25, s23, s24")                  # 专家跨度（元素）= rows*dim
    em("v_mov_b32_e32 v7, 0")
    em("v_mov_b32_e32 v15, 64")
    # weights 行基址 = weights + row*n_exp*4
    em("v_mov_b32_e32 v1, s6")
    em("v_mul_lo_u32 v1, v1, s22")
    em("v_lshlrev_b32_e32 v1, 2, v1")
    # y 行地址
    em("v_mov_b32_e32 v2, s6")
    em("v_mul_lo_u32 v2, v2, s24")
    em("v_lshlrev_b32_e32 v2, 2, v2")
    em("v_mov_b32_e32 v3, s16")
    em("v_mov_b32_e32 v4, s17")
    em("v_add_co_u32_e32 v3, vcc, v3, v2")
    em("v_addc_co_u32_e32 v4, vcc, v4, v7, vcc")
    # exp 行基址 = exp_out + row*dim*4
    em("v_mov_b32_e32 v5, s18")
    em("v_mov_b32_e32 v6, s19")
    em("v_add_co_u32_e32 v5, vcc, v5, v2")
    em("v_addc_co_u32_e32 v6, vcc, v6, v7, vcc")
    em("v_mov_b32_e32 v8, v0")                     # d = lane
    em("L_d:")
    em("v_cmp_gt_u32_e32 vcc, s24, v8")
    em("s_and_saveexec_b64 s[0:1], vcc")
    em("s_cbranch_execz L_dend")
    em("v_mov_b32_e32 v9, 0")                      # acc
    em("s_mov_b32 s26, 0")                         # e
    em("L_e:")
    em("s_cmp_lt_u32 s26, s22")
    em("s_cbranch_scc0 L_estore")
    # w = weights[row*n_exp + e]
    em("v_mov_b32_e32 v10, s26")
    em("v_lshlrev_b32_e32 v10, 2, v10")
    em("v_add_u32_e32 v10, v1, v10")
    em("v_mov_b32_e32 v11, s20")
    em("v_mov_b32_e32 v12, s21")
    em("v_add_co_u32_e32 v11, vcc, v11, v10")
    em("v_addc_co_u32_e32 v12, vcc, v12, v7, vcc")
    em("global_load_dword v13, v[11:12], off")
    # exp 元素 = e*rows*dim + row*dim + d
    em("v_mov_b32_e32 v14, s26")
    em("v_mul_lo_u32 v14, v14, s25")
    em("v_add_u32_e32 v14, v14, v8")
    em("v_lshlrev_b32_e32 v14, 2, v14")
    em("v_mov_b32_e32 v11, v5")
    em("v_mov_b32_e32 v12, v6")
    em("v_add_co_u32_e32 v11, vcc, v11, v14")
    em("v_addc_co_u32_e32 v12, vcc, v12, v7, vcc")
    em("global_load_dword v14, v[11:12], off")
    em("s_waitcnt vmcnt(0)")
    em("v_fma_f32 v9, v13, v14, v9")
    em("s_add_i32 s26, s26, 1")
    em("s_branch L_e")
    em("L_estore:")
    em("v_lshlrev_b32_e32 v10, 2, v8")
    em("v_mov_b32_e32 v11, v3")
    em("v_mov_b32_e32 v12, v4")
    em("v_add_co_u32_e32 v11, vcc, v11, v10")
    em("v_addc_co_u32_e32 v12, vcc, v12, v7, vcc")
    em("global_store_dword v[11:12], v9, off")
    em("L_dend:")
    em("s_or_b64 exec, exec, s[0:1]")
    em("v_add_u32_e32 v8, v8, v15")
    em("v_cmp_gt_u32_e32 vcc, s24, v8")
    em("s_cbranch_vccnz L_d")
    em("L_end:")
    em("s_endpgm")
    return "\n".join(L) + "\n"


def selftest() -> int:
    rng = np.random.default_rng(11)
    rows, dim, n_exp = 3, 2048, 5          # dim 故意取魔法除无法表示的 2048
    exp_out = rng.standard_normal((n_exp, rows, dim), dtype=np.float32)
    weights = rng.random((rows, n_exp), dtype=np.float32)
    ref = np.einsum("re,erd->rd", weights, exp_out).astype(np.float32)
    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE,
                      tag="moe_combine")
    out = run_one(hsaco, NAME,
                  [{"buffer": "y"}, {"buffer": "e"}, {"buffer": "w"},
                   {"scalar": {"dtype": "u32", "value": n_exp}},
                   {"scalar": {"dtype": "u32", "value": rows}},
                   {"scalar": {"dtype": "u32", "value": dim}}],
                  {"y": {"dtype": "f32", "values": [0.0] * (rows * dim)},
                   "e": {"dtype": "f32", "values": exp_out.reshape(-1).tolist()},
                   "w": {"dtype": "f32", "values": weights.reshape(-1).tolist()}},
                  grid=rows * 64, workgroup=64)
    got = np.asarray(out["y"], dtype=np.float32).reshape(rows, dim)
    err = float(np.abs(got - ref).max())
    ok = err < 1e-4 * max(1.0, float(np.abs(ref).max()))
    print(f"{NAME} rows={rows} dim={dim} n_exp={n_exp} max_abs={err:.3e} "
          f"{'✔' if ok else '✘'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
