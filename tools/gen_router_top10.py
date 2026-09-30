#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoE router top-10（一行一个线程）+ 自检。

签名：
    router_top10_k(const float* logits, int32_t* ids, float* weights,
                   uint32_t rows, uint32_t n_experts)

语义：对每行先取 logits 最大的 10 个专家，再在 top-10 内做 softmax/renormalize。
这与「全体 softmax 后取 top-10 再 renormalize」等价（softmax 单调）。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "router_top10_k"
K = 10
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
        scalar_arg(24, 4), scalar_arg(28, 4)]
KERNARG_SIZE = 32


def gen_asm() -> str:
    topv = [10 + i for i in range(K)]
    topi = [20 + i for i in range(K)]
    wreg = [40 + i for i in range(K)]
    L = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",   # logits
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",   # ids
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",  # weights
        "s_load_dword s22, s[4:5], 0x18",         # rows
        "s_load_dword s23, s[4:5], 0x1c",         # n_experts
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v38, 0",
        "v_mov_b32_e32 v1, s6",
        "v_lshlrev_b32_e32 v1, 6, v1",
        "v_add_u32_e32 v1, v1, v0",
        "v_cmp_gt_u32_e32 vcc, s22, v1",
        "s_and_saveexec_b64 s[2:3], vcc",
        "s_cbranch_execz L_end",
        # logits row
        "v_mov_b32_e32 v3, s23",
        "v_mul_lo_u32 v2, v1, v3",
        "v_lshlrev_b32_e32 v2, 2, v2",
        "v_mov_b32_e32 v5, s17",
        "v_add_co_u32_e32 v4, vcc, s16, v2",
        "v_addc_co_u32_e32 v5, vcc, v5, v38, vcc",
    ]
    for i in range(K):
        L.append(f"v_mov_b32_e32 v{topv[i]}, 0xff800000")
        L.append(f"v_mov_b32_e32 v{topi[i]}, 0")
    L += [
        "s_mov_b32 s24, 0",
        "L_col:",
        "s_cmp_lt_u32 s24, s23",
        "s_cbranch_scc0 L_done_col",
        "v_mov_b32_e32 v7, s24",
        "v_lshlrev_b32_e32 v7, 2, v7",
        "v_mov_b32_e32 v251, v5",
        "v_add_co_u32_e32 v250, vcc, v4, v7",
        "v_addc_co_u32_e32 v251, vcc, v251, v38, vcc",
        "global_load_dword v6, v[250:251], off",
        "s_waitcnt vmcnt(0)",
        "v_mov_b32_e32 v7, s24",
    ]
    for i in range(K):
        v, ix = topv[i], topi[i]
        L += [
            f"v_cmp_lt_f32_e32 vcc, v{v}, v6",
            # cndmask: dst = vcc ? src1 : src0
            f"v_cndmask_b32_e32 v8, v{v}, v6, vcc",
            f"v_cndmask_b32_e32 v6, v6, v{v}, vcc",
            f"v_mov_b32_e32 v{v}, v8",
            f"v_cndmask_b32_e32 v9, v{ix}, v7, vcc",
            f"v_cndmask_b32_e32 v7, v7, v{ix}, vcc",
            f"v_mov_b32_e32 v{ix}, v9",
        ]
    L += [
        "s_add_i32 s24, s24, 1",
        "s_branch L_col",
        "L_done_col:",
        # max of top-10
        "v_mov_b32_e32 v30, v10",
    ]
    for i in range(1, K):
        L.append(f"v_max_f32_e32 v30, v30, v{topv[i]}")
    L += [
        "v_mov_b32_e32 v31, 0x3fb8aa3b",           # log2(e)
        "v_mov_b32_e32 v32, 0",                     # sum
    ]
    for i in range(K):
        L += [
            f"v_sub_f32_e32 v33, v{topv[i]}, v30",
            "v_mul_f32_e32 v33, v33, v31",
            "s_nop 0",
            "v_exp_f32_e32 v33, v33",
            "s_nop 0",
            f"v_mov_b32_e32 v{wreg[i]}, v33",
            "v_add_f32_e32 v32, v32, v33",
        ]
    L += [
        "s_nop 0",
        "v_rcp_f32_e32 v32, v32",
        "s_nop 0",
        # row*10
        "v_mov_b32_e32 v34, 10",
        "v_mul_lo_u32 v35, v1, v34",
    ]
    for i in range(K):
        L += [
            f"v_mul_f32_e32 v{wreg[i]}, v{wreg[i]}, v32",
            f"v_add_u32_e32 v36, {i}, v35",
            "v_lshlrev_b32_e32 v36, 2, v36",
            "v_mov_b32_e32 v253, s21",
            "v_add_co_u32_e32 v252, vcc, s20, v36",
            "v_addc_co_u32_e32 v253, vcc, v253, v38, vcc",
            f"global_store_dword v[252:253], v{wreg[i]}, off",
            "v_mov_b32_e32 v253, s19",
            "v_add_co_u32_e32 v252, vcc, s18, v36",
            "v_addc_co_u32_e32 v253, vcc, v253, v38, vcc",
            f"global_store_dword v[252:253], v{topi[i]}, off",
        ]
    L += ["L_end:", "s_or_b64 exec, exec, s[2:3]", "s_endpgm"]
    return "\n".join(L) + "\n"


def main() -> int:
    rows, n_exp = 8, 512
    rng = np.random.default_rng(9)
    logits = rng.standard_normal((rows, n_exp), dtype=np.float32)
    top = np.argsort(-logits, axis=1)[:, :K]
    ex = np.exp(logits[np.arange(rows)[:, None], top] -
                logits.max(axis=1, keepdims=True))
    ref_w = ex / ex.sum(axis=1, keepdims=True)
    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="router")
    out = run_one(hsaco, NAME,
                  [{"buffer": "logits"}, {"buffer": "ids"}, {"buffer": "w"},
                   {"scalar": {"dtype": "u32", "value": rows}},
                   {"scalar": {"dtype": "u32", "value": n_exp}}],
                  {"logits": {"dtype": "f32", "values": logits.reshape(-1).tolist()},
                   "ids": {"dtype": "i32", "values": [0] * (rows * K)},
                   "w": {"dtype": "f32", "values": [0.0] * (rows * K)}},
                  grid=rows * 64, workgroup=64)
    gid = np.array(out["ids"], dtype=np.int32).reshape(rows, K)
    gw = np.array(out["w"], dtype=np.float32).reshape(rows, K)
    print("router ids equal:", np.array_equal(gid, top))
    print("router weight max_abs:", float(np.abs(gw - ref_w).max()))
    return 0 if np.array_equal(gid, top) and np.abs(gw - ref_w).max() < 1e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
