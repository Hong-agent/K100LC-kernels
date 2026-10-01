#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoE 专家输出加权合并内核：`moe_combine_k` + 自检。

语义：

    y[r, d] = sum_e weights[r, e] * exp_out[e, r, d]

布局：

    y        float32 [rows, dim]
    exp_out  float32 [n_exp, rows, dim]      # 每个被选中专家的完整输出
    weights  float32 [rows, n_exp]           # router 权重（未选中的写 0）

因为 `i = r*dim + d`，专家 e 的元素下标就是 `e*rows*dim + i`，所以一个
work-item 处理一个输出元素、内层循环 n_exp 即可，不需要 gather。

签名：
    moe_combine_k(float* y, const float* exp_out, const float* weights,
                  uint32_t n_exp, uint32_t rows, uint32_t dim, uint32_t magic_dim)

grid = ceil(rows*dim/64), workgroup = 64。
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
        scalar_arg(24, 4), scalar_arg(28, 4), scalar_arg(32, 4), scalar_arg(36, 4)]
KERNARG_SIZE = 40


def div_magic(d: int, max_i: int) -> int:
    """返回 M，使 `(i*M)>>32 == i//d` 对 0 <= i < max_i 成立。"""
    if d <= 1:
        raise ValueError("div_magic 只支持 d >= 2")
    m = (1 << 32) // d + 1
    if m >= (1 << 32):
        raise ValueError(f"div_magic: d={d} 太大")
    i = np.arange(max_i, dtype=np.uint64)
    if not np.array_equal((i * np.uint64(m)) >> np.uint64(32),
                          i // np.uint64(d)):
        raise ValueError(f"div_magic: d={d} max_i={max_i} 无法精确表示")
    return int(m)


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
    em("s_load_dword s25, s[4:5], 0x24")           # magic_dim
    em("s_waitcnt lgkmcnt(0)")
    # i = blockIdx.x*64 + tid
    em("v_mov_b32_e32 v1, s6")
    em("v_lshlrev_b32_e32 v1, 6, v1")
    em("v_add_u32_e32 v1, v0, v1")
    # 总元素数 = rows*dim
    em("s_mul_i32 s26, s23, s24")
    em("v_cmp_gt_u32_e32 vcc, s26, v1")
    em("s_and_saveexec_b64 s[0:1], vcc")
    em("s_cbranch_execz L_end")
    # r = i / dim
    em("v_mov_b32_e32 v3, s25")
    em("v_mul_hi_u32 v2, v1, v3")
    # weights 行内起始字节偏移 = r*n_exp*4
    em("v_mul_lo_u32 v4, v2, s22")
    em("v_lshlrev_b32_e32 v4, 2, v4")
    # 专家跨度（元素）= rows*dim
    em("s_mul_i32 s27, s23, s24")
    em("v_mov_b32_e32 v13, s27")
    em("v_mov_b32_e32 v14, 0")                     # 0（进位用）
    em("v_mov_b32_e32 v8, 0")                      # acc
    em("s_mov_b32 s29, 0")                         # e
    em("L_e:")
    em("s_cmp_lt_u32 s29, s22")
    em("s_cbranch_scc0 L_store")
    # w = weights[r*n_exp + e]
    em("v_mov_b32_e32 v5, s29")
    em("v_lshlrev_b32_e32 v5, 2, v5")
    em("v_add_u32_e32 v5, v4, v5")
    em("v_mov_b32_e32 v6, s20")
    em("v_mov_b32_e32 v7, s21")
    em("v_add_co_u32_e32 v6, vcc, v6, v5")
    em("v_addc_co_u32_e32 v7, vcc, v7, v14, vcc")
    em("global_load_dword v10, v[6:7], off")
    em("s_waitcnt vmcnt(0)")
    # 专家元素下标 = e*expert_stride + i
    em("v_mov_b32_e32 v11, s29")
    em("v_mul_lo_u32 v11, v11, v13")
    em("v_add_u32_e32 v11, v11, v1")
    em("v_lshlrev_b32_e32 v11, 2, v11")
    em("v_mov_b32_e32 v6, s18")
    em("v_mov_b32_e32 v7, s19")
    em("v_add_co_u32_e32 v6, vcc, v6, v11")
    em("v_addc_co_u32_e32 v7, vcc, v7, v14, vcc")
    em("global_load_dword v12, v[6:7], off")
    em("s_waitcnt vmcnt(0)")
    em("v_fma_f32 v8, v10, v12, v8")
    em("s_add_i32 s29, s29, 1")
    em("s_branch L_e")
    em("L_store:")
    em("v_lshlrev_b32_e32 v5, 2, v1")
    em("v_mov_b32_e32 v6, s16")
    em("v_mov_b32_e32 v7, s17")
    em("v_add_co_u32_e32 v6, vcc, v6, v5")
    em("v_addc_co_u32_e32 v7, vcc, v7, v14, vcc")
    em("global_store_dword v[6:7], v8, off")
    em("L_end:")
    em("s_or_b64 exec, exec, s[0:1]")
    em("s_endpgm")
    return "\n".join(L) + "\n"


def selftest() -> int:
    rng = np.random.default_rng(11)
    rows, dim, n_exp = 3, 129, 5
    exp_out = rng.standard_normal((n_exp, rows, dim), dtype=np.float32)
    weights = rng.random((rows, n_exp), dtype=np.float32)
    ref = np.einsum("re,erd->rd", weights, exp_out).astype(np.float32)
    magic = div_magic(dim, rows * dim)
    total = rows * dim
    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE,
                      tag="moe_combine")
    out = run_one(hsaco, NAME,
                  [{"buffer": "y"}, {"buffer": "e"}, {"buffer": "w"},
                   {"scalar": {"dtype": "u32", "value": n_exp}},
                   {"scalar": {"dtype": "u32", "value": rows}},
                   {"scalar": {"dtype": "u32", "value": dim}},
                   {"scalar": {"dtype": "u32", "value": magic}}],
                  {"y": {"dtype": "f32", "values": [0.0] * total},
                   "e": {"dtype": "f32", "values": exp_out.reshape(-1).tolist()},
                   "w": {"dtype": "f32", "values": weights.reshape(-1).tolist()}},
                  grid=total * 64, workgroup=64)
    got = np.asarray(out["y"], dtype=np.float32).reshape(rows, dim)
    err = float(np.abs(got - ref).max())
    scale = max(1.0, float(np.abs(ref).max()))
    ok = err < 1e-4 * scale
    print(f"{NAME} rows={rows} dim={dim} n_exp={n_exp} max_abs={err:.3e} "
          f"{'✔' if ok else '✘'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
