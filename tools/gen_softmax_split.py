#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分块 softmax（`block_max_k` + `reduce_max_k` + `block_exp_sum_k`）+ 自检。

    python3 tools/gen_softmax_split.py

为什么：`softmax_vec_k` 把**一整行**交给**一个** warp 做三趟扫描（max、
exp+sum、normalize），单 warp 的访存延迟全暴露出来——n_kv=16384 实测 126 us，
而数据只有 64 KB，纯粹被「一个 warp 串行 64 轮 × 3 趟」拖住。

这里改成多 workgroup 分块（解码注意力只有一行，列数 = 上下文长度）：

    sh        = log2(块数)          # 块数必须是 2 的幂
    chunk     = cols >> sh          # 每块的列数（256 的倍数）
    块 b      = 列 [b*chunk, (b+1)*chunk)

1. `block_max_k(part, x, cols, sh)`：每块 64 lane × 4 宽载入，块内 6 步 LDS
   树形归约求本块最大值 → `part[b]`（grid = 块数）。
2. `reduce_max_k(part, y, nrows, nbpr)`：把 `nbpr` 个分块最大值归约成 `y`
   （与 `reduce_blocks_k` 同构，只是求和换成取最大）。
3. `block_exp_sum_k(part, y, x, M, cols, sh, inv)`：每块算
   `y[i] = exp((x[i]-M)*inv)` 并写出（**不归一化**），同时把本块的 exp 和
   归约到 `part[b]`。归一化那一步改到输出侧去做（只需要除 dim 个数，而不是
   整行 cols 个数），所以这里省掉一整趟读写。

这样三步各只要 ~1 轮访存，块之间完全并行；调用方（Attention）用
`reduce_blocks_k` 把 `part_sum` 求和成 `L`，再用 `scale_mul_k` 把 `dim` 个
输出元素乘 `1/L`。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

MAXP_NAME = "block_max_k"
MAXP_ARGS = [buffer_arg(0), buffer_arg(8), scalar_arg(16, 4), scalar_arg(20, 4)]
MAXP_KERNARG = 24

RMAX_NAME = "reduce_max1_k"
RMAX_ARGS = [buffer_arg(0), buffer_arg(8), scalar_arg(16, 4)]
RMAX_KERNARG = 24

RSUM_NAME = "reduce_sum1_k"
RSUM_ARGS = [buffer_arg(0), buffer_arg(8), scalar_arg(16, 4)]
RSUM_KERNARG = 24

# 注意 M 必须是**设备指针**（`reduce_max1_k` 刚写出来的值），所以是 buffer 参数：
# 早先按 by-value 传会让内核把指针低 32 位当成 M。softmax 对平移不敏感，
# 所以 M 是「随便一个有限值」时结果看着还是对的——只有当那串位模式恰好是个
# 极大的浮点（→ exp 全下溢成 0 → L=0 → 除以 0）时才暴露，属于最难查的一类。
EXP_NAME = "block_exp_sum_k"
EXP_ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16), buffer_arg(24),
            scalar_arg(32, 4), scalar_arg(36, 4), scalar_arg(40, 4)]
EXP_KERNARG = 48

DIV_NAME = "div_scalar_k"
DIV_ARGS = [buffer_arg(0), buffer_arg(8), scalar_arg(16, 4)]
DIV_KERNARG = 24

def _store_part(L: list[str], part_off: int, reg: str) -> None:
    """lane 0 把归约结果写进 `part[blockIdx.x]`。"""
    L += [
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[2:3], vcc",
        "v_mov_b32_e32 v12, s6",
        "v_lshlrev_b32_e32 v12, 2, v12",
        "v_mov_b32_e32 v13, s17",
        "v_mov_b32_e32 v14, 0",
        f"v_add_co_u32_e32 v12, vcc, s{part_off}, v12",
        "v_addc_co_u32_e32 v13, vcc, v13, v14, vcc",
        f"global_store_dword v[12:13], {reg}, off",
        "s_or_b64 exec, exec, s[2:3]",
    ]


def gen_max_part_asm() -> str:
    """`block_max_k(part, x, cols, sh)`：分块最大值，grid = 1<<sh。"""
    L = [
        ".text", f"k_{MAXP_NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",    # part
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",    # x
        "s_load_dword s20, s[4:5], 0x10",          # cols
        "s_load_dword s21, s[4:5], 0x14",          # sh
        "s_waitcnt lgkmcnt(0)",
        # s22 = 每块字节数 = (cols*4) >> sh ; s23 = 每 lane 轮数 = cols >> (sh+8)
        "s_lshl_b32 s22, s20, 2",
        "s_lshr_b32 s22, s22, s21",
        "s_add_i32 s24, s21, 8",
        "s_lshr_b32 s23, s20, s24",
        "v_mov_b32_e32 v1, s6",
        "v_mul_lo_u32 v1, v1, s22",
        "v_lshlrev_b32_e32 v2, 4, v0",
        "v_add_u32_e32 v1, v1, v2",
        "v_mov_b32_e32 v3, 0",
        "v_mov_b32_e32 v5, s19",
        "v_add_co_u32_e32 v4, vcc, s18, v1",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "v_mov_b32_e32 v8, 0xff800000",            # -inf
        "v_mov_b32_e32 v10, 1024",                 # 每轮 256 个元素
        "s_mov_b32 s26, 0",
        "L_max:",
        "s_cmp_lt_u32 s26, s23",
        "s_cbranch_scc0 L_done",
        "global_load_dwordx4 v[20:23], v[4:5], off",
        "s_waitcnt vmcnt(0)",
        "v_max_f32_e32 v20, v20, v21",
        "v_max_f32_e32 v22, v22, v23",
        "v_max_f32_e32 v20, v20, v22",
        "v_max_f32_e32 v8, v8, v20",
        "v_add_co_u32_e32 v4, vcc, v4, v10",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "s_add_i32 s26, s26, 1",
        "s_branch L_max",
        "L_done:",
    ]
    L += _tree_lines("v_max_f32_e32", "v8")
    _store_part(L, 16, "v8")
    L.append("s_endpgm")
    return "\n".join(L) + "\n"


def gen_reduce1_asm(name: str, op: str, init: str) -> str:
    """`reduce_XXX1_k(part, y, nbpr)`：把 ≤64 个分块值归约成 `y[0]`。

    一开始写成「lane 0 串行扫 nbpr 个」的话，每次 `s_waitcnt vmcnt(0)` 都是一次
    完整的访存往返，nbpr=64 时实测 14.8 us（纯延迟）。改成 **一个 lane 读一个
    分块值 + 6 步 LDS 树形归约**：只要 1 次往返，实测 ~2 us。
    要求 nbpr ≤ 64、grid = 1 个 workgroup（解码注意力的块数按 64 封顶，满足）。
    """
    return "\n".join([
        ".text", f"k_{name}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",    # part
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",    # y
        "s_load_dword s20, s[4:5], 0x10",          # nbpr
        "s_waitcnt lgkmcnt(0)",
        f"v_mov_b32_e32 v8, {init}",               # 越界 lane 的中性元
        "v_cmp_gt_u32_e32 vcc, s20, v0",           # lane < nbpr 才读
        "s_and_saveexec_b64 s[0:1], vcc",
        "v_lshlrev_b32_e32 v2, 2, v0",
        "v_mov_b32_e32 v4, s16",
        "v_mov_b32_e32 v5, s17",
        "v_mov_b32_e32 v3, 0",
        "v_add_co_u32_e32 v4, vcc, v4, v2",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "global_load_dword v8, v[4:5], off",
        "s_waitcnt vmcnt(0)",
        "s_or_b64 exec, exec, s[0:1]",
    ] + _tree_lines(op, "v8") + [
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[2:3], vcc",
        "v_mov_b32_e32 v12, s18",
        "v_mov_b32_e32 v13, s19",
        "global_store_dword v[12:13], v8, off",
        "s_or_b64 exec, exec, s[2:3]",
        "s_endpgm",
    ]) + "\n"


def _tree_lines(op: str, reg: str) -> list[str]:
    """6 步 LDS 树形归约（槽位 = lane*4），结果留在 lane 0。"""
    L = ["v_lshlrev_b32_e32 v11, 2, v0",
         f"ds_write_b32 v11, {reg}",
         "s_waitcnt lgkmcnt(0)",
         "s_barrier"]
    for off in (32, 16, 8, 4, 2, 1):
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
    return L


def gen_reduce_max1_asm() -> str:
    return gen_reduce1_asm(RMAX_NAME, "v_max_f32_e32", "0xff800000")


def gen_reduce_sum1_asm() -> str:
    return gen_reduce1_asm(RSUM_NAME, "v_add_f32_e32", "0")


def gen_exp_sum_asm() -> str:
    """`block_exp_sum_k(part, y, x, Mptr, cols, sh, inv)`。

    `y[i] = exp((x[i]-M)*inv)`（未归一化），同时把本块的 exp 和归约进 `part[b]`。
    `M` 是 `reduce_max1_k` 刚写出来的**设备指针**：softmax 对平移不敏感，所以
    早先按 by-value 传指针低 32 位时结果常常「看着对」，只有那串位模式恰好是
    个极大的浮点数时才 exp 全下溢、除以 0 → NaN（实测踩到过）。
    """
    L = [
        ".text", f"k_{EXP_NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",    # part
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",    # y
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",   # x
        "s_load_dwordx2 s[22:23], s[4:5], 0x18",   # Mptr
        "s_load_dword s28, s[4:5], 0x20",          # cols
        "s_load_dword s29, s[4:5], 0x24",          # sh
        "s_load_dword s31, s[4:5], 0x28",          # inv
        "s_waitcnt lgkmcnt(0)",
        # s24 = 每块字节数 = (cols*4) >> sh ; s25 = 每 lane 轮数 = cols >> (sh+8)
        "s_lshl_b32 s24, s28, 2",
        "s_lshr_b32 s24, s24, s29",
        "s_add_i32 s30, s29, 8",
        "s_lshr_b32 s25, s28, s30",
        "v_mov_b32_e32 v1, s6",
        "v_mul_lo_u32 v1, v1, s24",
        "v_lshlrev_b32_e32 v2, 4, v0",
        "v_add_u32_e32 v1, v1, v2",
        "v_mov_b32_e32 v3, 0",
        "v_mov_b32_e32 v5, s21",
        "v_add_co_u32_e32 v4, vcc, s20, v1",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",  # x 地址
        "v_mov_b32_e32 v7, s19",
        "v_add_co_u32_e32 v6, vcc, s18, v1",
        "v_addc_co_u32_e32 v7, vcc, v7, v3, vcc",  # y 地址
        "v_mov_b32_e32 v8, 0",                     # 本块的 exp 和
        "v_mov_b32_e32 v17, s22",
        "v_mov_b32_e32 v18, s23",
        "global_load_dword v9, v[17:18], off",     # M（所有 lane 同一个地址）
        "s_waitcnt vmcnt(0)",
        "v_mov_b32_e32 v10, s31",                  # inv = 1/sqrt(dim)
        "v_mov_b32_e32 v16, 0x3fb8aa3b",           # log2(e)：v_exp_f32 是 2^x
        "v_mov_b32_e32 v15, 1024",
        "s_mov_b32 s26, 0",
        "L_exp:",
        "s_cmp_lt_u32 s26, s25",
        "s_cbranch_scc0 L_done",
        "global_load_dwordx4 v[20:23], v[4:5], off",
        "s_waitcnt vmcnt(0)",
    ]
    for r in ("v20", "v21", "v22", "v23"):
        L += [
            f"v_sub_f32_e32 {r}, {r}, v9",
            f"v_mul_f32_e32 {r}, {r}, v10",
            f"v_mul_f32_e32 {r}, {r}, v16",
            "s_nop 0",
            f"v_exp_f32_e32 {r}, {r}",
            "s_nop 0",
            f"v_add_f32_e32 v8, v8, {r}",
        ]
    L += [
        "global_store_dwordx4 v[6:7], v[20:23], off",
        "v_add_co_u32_e32 v4, vcc, v4, v15",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "v_add_co_u32_e32 v6, vcc, v6, v15",
        "v_addc_co_u32_e32 v7, vcc, v7, v3, vcc",
        "s_add_i32 s26, s26, 1",
        "s_branch L_exp",
        "L_done:",
    ]
    L += _tree_lines("v_add_f32_e32", "v8")
    _store_part(L, 16, "v8")
    L.append("s_endpgm")
    return "\n".join(L) + "\n"


def build_split_hsaco(tag: str = "smx"):
    mx = build_one(MAXP_NAME, gen_max_part_asm(), MAXP_ARGS,
                   kernarg_size=MAXP_KERNARG, group_segment=256, tag=tag)
    rm = build_one(RMAX_NAME, gen_reduce_max1_asm(), RMAX_ARGS,
                   kernarg_size=RMAX_KERNARG, group_segment=256, tag=tag)
    rs = build_one(RSUM_NAME, gen_reduce_sum1_asm(), RSUM_ARGS,
                   kernarg_size=RSUM_KERNARG, group_segment=256, tag=tag)
    ex = build_one(EXP_NAME, gen_exp_sum_asm(), EXP_ARGS,
                   kernarg_size=EXP_KERNARG, group_segment=256, tag=tag)
    dv = build_one(DIV_NAME, gen_div_asm(), DIV_ARGS,
                   kernarg_size=DIV_KERNARG, tag=tag)
    return mx, rm, rs, ex, dv


def gen_div_asm() -> str:
    """`div_scalar_k(y, s, n)`：`y[i] /= s[0]`（`s` 是**设备指针**，不占主机标量）。

    归一化放在输出侧（只除 dim 个数，而不是整行 cols 个数），所以需要按设备上
    的 `L` 做一次除法——用 `v_rcp_f32` 求倒数再乘（相对误差 ~1e-7，和 softmax
    的其它步骤同量级）。
    """
    return "\n".join([
        ".text", f"k_{DIV_NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",    # y
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",    # s（设备指针）
        "s_load_dword s20, s[4:5], 0x10",          # n
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v3, 0",
        "v_mov_b32_e32 v1, s6",
        "v_mul_lo_u32 v1, v1, 64",
        "v_add_u32_e32 v1, v0, v1",                # i
        "v_cmp_gt_u32_e32 vcc, s20, v1",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        "v_lshlrev_b32_e32 v2, 2, v1",
        "v_mov_b32_e32 v4, s16",
        "v_mov_b32_e32 v5, s17",
        "v_add_co_u32_e32 v4, vcc, v4, v2",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "v_mov_b32_e32 v9, s18",
        "v_mov_b32_e32 v10, s19",
        "global_load_dword v6, v[9:10], off",      # 所有 lane 读同一个地址（广播）
        "s_waitcnt vmcnt(0)",
        "s_nop 0",
        "v_rcp_f32_e32 v7, v6",
        "s_nop 0",
        "global_load_dword v8, v[4:5], off",
        "s_waitcnt vmcnt(0)",
        "v_mul_f32_e32 v8, v8, v7",
        "global_store_dword v[4:5], v8, off",
        "L_end:",
        "s_or_b64 exec, exec, s[0:1]",
        "s_endpgm",
    ]) + "\n"


def selftest() -> int:
    rng = np.random.default_rng(20261002)
    rc = 0
    mx, rm, rs, ex, dv = build_split_hsaco()
    for cols, sh in ((1024, 2), (4096, 4), (4096, 0), (16384, 6), (16384, 2)):
        nblk = 1 << sh
        x = (rng.standard_normal(cols).astype(np.float32) * 3.0)
        inv = float(1.0 / np.sqrt(128.0))
        # 1) 分块最大值
        o1 = run_one(mx, MAXP_NAME,
                     [{"buffer": "part"}, {"buffer": "x"},
                      {"scalar": {"dtype": "u32", "value": cols}},
                      {"scalar": {"dtype": "u32", "value": sh}}],
                     {"part": {"dtype": "f32", "values": [0.0] * nblk},
                      "x": {"dtype": "f32", "values": x.tolist()}},
                     grid=nblk * 64, workgroup=64)
        part = np.array(o1["part"], np.float32)
        ref_part = x.reshape(nblk, cols // nblk).max(axis=1)
        bad1 = float(np.abs(part - ref_part).max())
        # 2) 归约最大值
        o2 = run_one(rm, RMAX_NAME,
                     [{"buffer": "part"}, {"buffer": "y"},
                      {"scalar": {"dtype": "u32", "value": nblk}}],
                     {"part": {"dtype": "f32", "values": part.tolist()},
                      "y": {"dtype": "f32", "values": [0.0]}},
                     grid=64, workgroup=64)
        M = float(o2["y"][0])
        bad2 = abs(M - float(x.max()))
        # 3) exp + 分块和
        o3 = run_one(ex, EXP_NAME,
                     [{"buffer": "part"}, {"buffer": "y"}, {"buffer": "x"},
                      {"buffer": "m"},
                      {"scalar": {"dtype": "u32", "value": cols}},
                      {"scalar": {"dtype": "u32", "value": sh}},
                      {"scalar": {"dtype": "f32", "value": inv}}],
                     {"part": {"dtype": "f32", "values": [0.0] * nblk},
                      "y": {"dtype": "f32", "values": [0.0] * cols},
                      "x": {"dtype": "f32", "values": x.tolist()},
                      "m": {"dtype": "f32", "values": [M]}},
                     grid=nblk * 64, workgroup=64)
        y = np.array(o3["y"], np.float32)
        psum = np.array(o3["part"], np.float32)
        o5 = run_one(rs, RSUM_NAME,
                     [{"buffer": "part"}, {"buffer": "y"},
                      {"scalar": {"dtype": "u32", "value": nblk}}],
                     {"part": {"dtype": "f32", "values": psum.tolist()},
                      "y": {"dtype": "f32", "values": [0.0]}},
                     grid=64, workgroup=64)
        e = np.exp((x - M) * inv)
        bad3 = float(np.abs(y - e).max() / max(1e-9, float(np.abs(e).max())))
        bad4 = float(np.abs(o5["y"][0] - e.sum()) / max(1e-9, float(e.sum())))
        bad4b = float(np.abs(psum.sum() - e.sum()) / max(1e-9, float(e.sum())))
        soft = y / y.sum()
        ref_soft = np.exp(x * inv - (x * inv).max())
        ref_soft = ref_soft / ref_soft.sum()
        bad5 = float(np.abs(soft - ref_soft).max())
        # 4) 输出侧归一化：y /= L（L 是设备上的标量）
        L = float(o5["y"][0])
        o4 = run_one(dv, DIV_NAME,
                     [{"buffer": "y"}, {"buffer": "L"},
                      {"scalar": {"dtype": "u32", "value": cols}}],
                     {"y": {"dtype": "f32", "values": y.tolist()},
                      "L": {"dtype": "f32", "values": [L]}},
                     grid=(cols + 63) // 64 * 64, workgroup=64)
        soft2 = np.array(o4["y"], np.float32)
        bad6 = float(np.abs(soft2 - ref_soft).max())
        ok = max(bad1, bad2, bad3, bad4, bad4b, bad5, bad6) < 1e-5
        rc |= 0 if ok else 1
        print(f"cols={cols:6d} nblk={nblk:3d}: 块最大 {bad1:.2e} 全局最大 {bad2:.2e} "
              f"exp {bad3:.2e} 分块和 {bad4:.2e} softmax {bad5:.2e} "
              f"除法归一 {bad6:.2e} "
              f"{'✔' if ok else '✘'}")
    return rc


def main() -> int:
    rc = selftest()
    print("softmax 分块三件套 vs numpy:", "全部一致 ✔" if rc == 0 else "有不一致 ✘")
    return rc


# `tools/build_native_kernels.py` 从这里取内核
KERNELS = [
    (MAXP_NAME, gen_max_part_asm, MAXP_ARGS, MAXP_KERNARG),
    (RMAX_NAME, gen_reduce_max1_asm, RMAX_ARGS, RMAX_KERNARG),
    (RSUM_NAME, gen_reduce_sum1_asm, RSUM_ARGS, RSUM_KERNARG),
    (EXP_NAME, gen_exp_sum_asm, EXP_ARGS, EXP_KERNARG),
    (DIV_NAME, gen_div_asm, DIV_ARGS, DIV_KERNARG),
]


if __name__ == "__main__":
    raise SystemExit(main())
