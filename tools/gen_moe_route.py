#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoE token→expert 分桶需要的两个内核 + 自检。

* `gather_rows_k(dst, src, idx, rows, dim)`
      按索引把行 gather 到一起：`dst[r,:] = src[idx[r],:]`。
      一个 workgroup 一行，lane 用 `dwordx4` 搬 4 个 f32。

* `moe_combine_gather_k(y, exp_out, weights, pos, n_sel, rows, dim, magic_dim)`
      按路由位置把专家输出加权合并：
      `y[r,:] = Σ_e weights[r,e] * exp_out[e, pos[r,e], :]`。

与 `moe_combine_k` 的区别：专家输出不是「每个 token 在固定行」，而是排在
「该专家实际收到的 token」桶里，用 `pos[r,e]` 找到位置。这样 token 只需
进入被选中的专家，不需要每个专家都跑全部行。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

GATHER_NAME = "gather_rows_k"
GATHER_ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
               scalar_arg(24, 4), scalar_arg(28, 4)]
GATHER_KERNARG = 32

COMBINE_NAME = "moe_combine_gather_k"
# (y, exp_bucket, weights, base, n_sel, rows, dim)
#   base[r*n_sel+e] = 该 (r,e) 在 exp_bucket 里的行号（主机预计算）
COMBINE_ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16), buffer_arg(24),
                scalar_arg(32, 4), scalar_arg(36, 4), scalar_arg(40, 4)]
COMBINE_KERNARG = 48


def div_magic(d: int, max_i: int) -> int:
    if d <= 0:
        raise ValueError("div_magic: d 必须为正")
    if max_i <= 0:
        return 0
    if d == 1:
        if max_i <= 1:
            return 0
        raise ValueError("div_magic: d=1 只在下标恒为 0 时可用")
    m = (1 << 32) // d + 1
    if m >= (1 << 32):
        raise ValueError(f"div_magic: d={d} 太大")
    i = np.arange(max_i, dtype=np.uint64)
    if not np.array_equal((i * np.uint64(m)) >> np.uint64(32),
                          i // np.uint64(d)):
        raise ValueError(f"div_magic: d={d} max_i={max_i} 无法精确表示")
    return int(m)


def gen_gather_asm() -> str:
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{GATHER_NAME}:")
    em("s_load_dwordx2 s[16:17], s[4:5], 0x0")     # dst
    em("s_load_dwordx2 s[18:19], s[4:5], 0x8")     # src
    em("s_load_dwordx2 s[20:21], s[4:5], 0x10")    # idx
    em("s_load_dword s22, s[4:5], 0x18")           # rows
    em("s_load_dword s23, s[4:5], 0x1c")           # dim
    em("s_waitcnt lgkmcnt(0)")
    em("s_cmp_lt_u32 s6, s22")
    em("s_cbranch_scc0 L_end")
    em("v_mov_b32_e32 v8, 0")
    em("v_mov_b32_e32 v16, 0")
    em("v_mov_b32_e32 v24, 256")
    # idx[row]
    em("v_mov_b32_e32 v1, s6")
    em("v_lshlrev_b32_e32 v1, 2, v1")
    em("v_mov_b32_e32 v2, s20")
    em("v_mov_b32_e32 v3, s21")
    em("v_add_co_u32_e32 v2, vcc, v2, v1")
    em("v_addc_co_u32_e32 v3, vcc, v3, v8, vcc")
    em("global_load_dword v4, v[2:3], off")
    em("s_waitcnt vmcnt(0)")
    # dst 行地址 = dst + row*dim*4
    em("v_mov_b32_e32 v5, s6")
    em("v_mul_lo_u32 v5, v5, s23")
    em("v_lshlrev_b32_e32 v5, 2, v5")
    em("v_mov_b32_e32 v6, s16")
    em("v_mov_b32_e32 v7, s17")
    em("v_add_co_u32_e32 v6, vcc, v6, v5")
    em("v_addc_co_u32_e32 v7, vcc, v7, v8, vcc")
    # src 行地址 = src + idx*dim*4
    em("v_mul_lo_u32 v9, v4, s23")
    em("v_lshlrev_b32_e32 v9, 2, v9")
    em("v_mov_b32_e32 v10, s18")
    em("v_mov_b32_e32 v11, s19")
    em("v_add_co_u32_e32 v10, vcc, v10, v9")
    em("v_addc_co_u32_e32 v11, vcc, v11, v8, vcc")
    # d = lane*4，步进 256
    em("v_lshlrev_b32_e32 v12, 2, v0")
    em("L_loop:")
    em("v_cmp_gt_u32_e32 vcc, s23, v12")
    em("s_and_saveexec_b64 s[0:1], vcc")
    em("s_cbranch_execz L_next")
    em("v_lshlrev_b32_e32 v13, 2, v12")           # 字节偏移
    em("v_mov_b32_e32 v14, v10")
    em("v_mov_b32_e32 v15, v11")
    em("v_add_co_u32_e32 v14, vcc, v14, v13")
    em("v_addc_co_u32_e32 v15, vcc, v15, v16, vcc")
    em("global_load_dwordx4 v[20:23], v[14:15], off")
    em("s_waitcnt vmcnt(0)")
    em("v_mov_b32_e32 v14, v6")
    em("v_mov_b32_e32 v15, v7")
    em("v_add_co_u32_e32 v14, vcc, v14, v13")
    em("v_addc_co_u32_e32 v15, vcc, v15, v16, vcc")
    em("global_store_dwordx4 v[14:15], v[20:23], off")
    em("L_next:")
    em("s_or_b64 exec, exec, s[0:1]")
    em("v_add_u32_e32 v12, v12, v24")
    em("v_cmp_gt_u32_e32 vcc, s23, v12")
    em("s_cbranch_vccnz L_loop")
    em("L_end:")
    em("s_endpgm")
    return "\n".join(L) + "\n"


def gen_combine_gather_asm() -> str:
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{COMBINE_NAME}:")
    em("s_load_dwordx2 s[16:17], s[4:5], 0x0")     # y
    em("s_load_dwordx2 s[18:19], s[4:5], 0x8")     # exp_bucket
    em("s_load_dwordx2 s[20:21], s[4:5], 0x10")    # weights
    em("s_load_dwordx2 s[22:23], s[4:5], 0x18")    # base
    em("s_load_dword s24, s[4:5], 0x20")           # n_sel
    em("s_load_dword s25, s[4:5], 0x24")           # rows
    em("s_load_dword s26, s[4:5], 0x28")           # dim
    em("s_waitcnt lgkmcnt(0)")
    em("s_cmp_lt_u32 s6, s25")
    em("s_cbranch_scc0 L_end")
    em("v_mov_b32_e32 v7, 0")
    em("v_mov_b32_e32 v15, 64")
    # idx 行基址字节偏移 = row*n_sel*4（weights 与 base 共用）
    em("v_mov_b32_e32 v1, s6")
    em("v_mul_lo_u32 v1, v1, s24")
    em("v_lshlrev_b32_e32 v1, 2, v1")
    # y 行地址 = y + row*dim*4
    em("v_mov_b32_e32 v2, s6")
    em("v_mul_lo_u32 v2, v2, s26")
    em("v_lshlrev_b32_e32 v2, 2, v2")
    em("v_mov_b32_e32 v3, s16")
    em("v_mov_b32_e32 v4, s17")
    em("v_add_co_u32_e32 v3, vcc, v3, v2")
    em("v_addc_co_u32_e32 v4, vcc, v4, v7, vcc")
    em("v_mov_b32_e32 v8, v0")                     # d = lane
    em("L_d:")
    em("v_cmp_gt_u32_e32 vcc, s26, v8")
    em("s_and_saveexec_b64 s[0:1], vcc")
    em("s_cbranch_execz L_dend")
    em("v_mov_b32_e32 v9, 0")                      # acc
    em("s_mov_b32 s29, 0")                         # e
    em("L_e:")
    em("s_cmp_lt_u32 s29, s24")
    em("s_cbranch_scc0 L_estore")
    em("v_mov_b32_e32 v10, s29")
    em("v_lshlrev_b32_e32 v10, 2, v10")
    em("v_add_u32_e32 v10, v1, v10")               # weights/base 字节偏移
    # w = weights[idx]
    em("v_mov_b32_e32 v11, s20")
    em("v_mov_b32_e32 v12, s21")
    em("v_add_co_u32_e32 v11, vcc, v11, v10")
    em("v_addc_co_u32_e32 v12, vcc, v12, v7, vcc")
    em("global_load_dword v13, v[11:12], off")
    # b = base[idx]
    em("v_mov_b32_e32 v11, s22")
    em("v_mov_b32_e32 v12, s23")
    em("v_add_co_u32_e32 v11, vcc, v11, v10")
    em("v_addc_co_u32_e32 v12, vcc, v12, v7, vcc")
    em("global_load_dword v14, v[11:12], off")
    em("s_waitcnt vmcnt(0)")
    # exp 元素 = b*dim + d
    em("v_mul_lo_u32 v14, v14, s26")
    em("v_add_u32_e32 v14, v14, v8")
    em("v_lshlrev_b32_e32 v14, 2, v14")
    em("v_mov_b32_e32 v11, s18")
    em("v_mov_b32_e32 v12, s19")
    em("v_add_co_u32_e32 v11, vcc, v11, v14")
    em("v_addc_co_u32_e32 v12, vcc, v12, v7, vcc")
    em("global_load_dword v14, v[11:12], off")
    em("s_waitcnt vmcnt(0)")
    em("v_fma_f32 v9, v13, v14, v9")
    em("s_add_i32 s29, s29, 1")
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
    em("v_cmp_gt_u32_e32 vcc, s26, v8")
    em("s_cbranch_vccnz L_d")
    em("L_end:")
    em("s_endpgm")
    return "\n".join(L) + "\n"


def selftest() -> int:
    rng = np.random.default_rng(13)
    rows, dim = 7, 128
    src = rng.standard_normal((rows, dim), dtype=np.float32)
    idx = np.array([3, 0, 5, 1, 6, 2, 4], dtype=np.uint32)
    hsaco = build_one(GATHER_NAME, gen_gather_asm(), GATHER_ARGS,
                      kernarg_size=GATHER_KERNARG, tag="gather_rows")
    out = run_one(hsaco, GATHER_NAME,
                  [{"buffer": "dst"}, {"buffer": "src"}, {"buffer": "idx"},
                   {"scalar": {"dtype": "u32", "value": rows}},
                   {"scalar": {"dtype": "u32", "value": dim}}],
                  {"dst": {"dtype": "f32", "values": [0.0] * (rows * dim)},
                   "src": {"dtype": "f32", "values": src.reshape(-1).tolist()},
                   "idx": {"dtype": "u32", "values": idx.tolist()}},
                  grid=rows * 64, workgroup=64)
    got = np.asarray(out["dst"], dtype=np.float32).reshape(rows, dim)
    ok_gather = bool(np.array_equal(got, src[idx]))
    print(f"{GATHER_NAME} max_abs={np.abs(got - src[idx]).max():.3e} "
          f"{'✔' if ok_gather else '✘'}")

    n_sel = 3
    dim2 = 2048          # 旧版魔法除在 2048×行数时无法精确表示
    erows = rows + 2
    exp_off = np.arange(n_sel + 1, dtype=np.uint32) * erows
    exp_bucket = rng.standard_normal((exp_off[-1], dim2), dtype=np.float32)
    weights = rng.random((rows, n_sel), dtype=np.float32)
    pos = rng.integers(0, erows, size=(rows, n_sel)).astype(np.uint32)
    base = exp_off[:n_sel][None, :] + pos              # 槽位 e 固定指向专家 e
    ref = np.empty((rows, dim2), dtype=np.float32)
    for r in range(rows):
        ref[r] = sum(weights[r, e] * exp_bucket[exp_off[e] + pos[r, e]]
                     for e in range(n_sel))
    total = rows * dim2
    hsaco = build_one(COMBINE_NAME, gen_combine_gather_asm(), COMBINE_ARGS,
                      kernarg_size=COMBINE_KERNARG, tag="moe_combine_gather")
    out = run_one(hsaco, COMBINE_NAME,
                  [{"buffer": "y"}, {"buffer": "e"}, {"buffer": "w"},
                   {"buffer": "base"},
                   {"scalar": {"dtype": "u32", "value": n_sel}},
                   {"scalar": {"dtype": "u32", "value": rows}},
                   {"scalar": {"dtype": "u32", "value": dim2}}],
                  {"y": {"dtype": "f32", "values": [0.0] * total},
                   "e": {"dtype": "f32", "values": exp_bucket.reshape(-1).tolist()},
                   "w": {"dtype": "f32", "values": weights.reshape(-1).tolist()},
                   "base": {"dtype": "u32", "values": base.reshape(-1).tolist()}},
                  grid=rows * 64, workgroup=64)
    got = np.asarray(out["y"], dtype=np.float32).reshape(rows, dim2)
    err = float(np.abs(got - ref).max())
    ok_comb = err < 1e-4 * max(1.0, float(np.abs(ref).max()))
    print(f"{COMBINE_NAME} max_abs={err:.3e} {'✔' if ok_comb else '✘'}")
    return 0 if (ok_gather and ok_comb) else 1


if __name__ == "__main__":
    raise SystemExit(selftest())


# 供 build_native_kernels.py 收集
KERNELS = [
    (GATHER_NAME, gen_gather_asm, GATHER_ARGS, GATHER_KERNARG),
    (COMBINE_NAME, gen_combine_gather_asm, COMBINE_ARGS, COMBINE_KERNARG),
]
