#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化权重「原生解码 + 点积」融合 GEMV（gfx926 `.s` 生成器）+ 自检。

    python3 tools/gen_gemv_qdot.py            # 11 类编码自检（真权重 + numpy 对账）

## 为什么要融合

老路子是「先把整块专家矩阵解成 f32 scratch，再 `gemv_f32_k`」：

    IQ4_NL 4.5 bit/权重 → f32 32 bit/权重，写一遍 + 读一遍 = **约 14 倍**的
    专家权重访存。专家权重每 token 约 1.3 GB，解成 f32 后就是 19 GB/token，
    直接把解码速度压到 30 tok/s 量级。

融合版内核**只读原编码**（4.5 bit），边解边点积，访存降到 1/7。

## 设计

一个 work-item 负责「一个 32 权重块 × 一行」的点积，写 `partial[块号]`；
再用 `reduce_k` 把每行的 `nbpr` 个块和加起来。这样：

* work-item 之间访存**完全合并**（w 按块号连续，x 按块内偏移连续）；
* work-item 数量 = 总块数（几十万），占用率足够掩盖访存延迟；
* 复用 `*_dequant_k` 已经验过位的解码逻辑，只把「写 32 个 f32」换成
  「读 4 个 x + 4 次 FMA」，尺度 d 的处理完全不变。

## 索引

    i        = 全局块号（0 .. nblocks-1），也是 partial 的下标
    q1, b    = divmod(i, nbpr)          b = 本块在该行内的块号
    e        = q1 / rows_per_exp        第几个专家组（单矩阵时恒为 0）
    xblk     = e * nbpr + b             激活里对应的块号

除法用「魔法乘」：`q = (i * M) >> 32`。M 由主机按实际 i 上界穷举校验过，
保证与整数除法逐位相同（见 `div_magic`）。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

# IQ4_NL 16 项码本（kvalues_iq4nl）拆成 4 个 dword（小端）
LUT_LO_HI = 0xF6EADDCF
LUT_LO_LO = 0xBFAD9881
LUT_HI_HI = 0x71594535
LUT_HI_LO = 0x26190D01


# --------------------------------------------------------------------------
# 主机侧：除法魔法数
# --------------------------------------------------------------------------
def div_magic(d: int, max_i: int) -> int:
    """返回 M，使 `(i * M) >> 32 == i // d` 对一切 `0 <= i < max_i` 成立。"""
    if d <= 0:
        raise ValueError("div_magic: d 必须为正")
    if max_i <= 0:
        return 0
    m = (1 << 32) // d + 1
    if m >= (1 << 32):
        raise ValueError(f"div_magic: d={d} 太大（商溢出）")
    i = np.arange(max_i, dtype=np.uint64)
    if not np.array_equal((i * np.uint64(m)) >> np.uint64(32), i // np.uint64(d)):
        raise ValueError(f"div_magic: d={d} max_i={max_i} 无法用 32 位魔法数表示")
    return int(m)


def _divmod(L: list[str], i: str, d: str, magic: str, q: str, r: str, t: str) -> None:
    """`q = i/d`、`r = i - q*d`（魔法乘；主机已保证精确）。"""
    L.append(f"v_mul_hi_u32 {q}, {i}, {magic}")
    L.append(f"v_mul_lo_u32 {t}, {q}, {d}")
    L.append(f"v_sub_u32_e32 {r}, {i}, {t}")


# 各类融合点积内核统一的前 9 个参数（0x00..0x2c），额外表指针从 0x30 起。
def dot_args(n_tables: int = 0) -> list[dict]:
    a = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
         scalar_arg(24, 4), scalar_arg(28, 4), scalar_arg(32, 4),
         scalar_arg(36, 4), scalar_arg(40, 4), scalar_arg(44, 4)]
    a += [buffer_arg(48 + 8 * i) for i in range(n_tables)]
    # 专家组索引表 + 专家组字节跨度 + 「权重侧每专家组行数」及其除法魔法数。
    # 注意 X 侧的专家组（`rows_per_exp`）和权重侧的专家组（这里）可以不同：
    # gate/up 的 10 个专家共用同一份 x，X 侧只有一组，权重侧每 640 行换一个专家。
    o = 48 + 8 * n_tables
    a += [buffer_arg(o), scalar_arg(o + 8, 4), scalar_arg(o + 12, 4),
          scalar_arg(o + 16, 4)]
    return a


def dot_kernarg(n_tables: int = 0) -> int:
    return ((48 + 8 * n_tables) + 20 + 7) // 8 * 8


# v1 = 全局块号 i；v6:v7 = 激活里对应块的地址；v70..v73 = 累加器。
DOT_ACC = [70, 71, 72, 73]


def dot_head(L: list[str], n_tables: int = 0, acc=None) -> None:
    """公共前导：取参数、算 i、魔法除法得 xblk、建立 x 地址、清零累加器。"""
    acc = DOT_ACC if acc is None else acc
    em = L.append
    em("s_load_dwordx2 s[16:17], s[4:5], 0x0")    # w
    em("s_load_dwordx2 s[18:19], s[4:5], 0x8")    # x
    em("s_load_dwordx2 s[20:21], s[4:5], 0x10")   # partial
    em("s_load_dword s22, s[4:5], 0x18")          # nblocks
    em("s_load_dword s23, s[4:5], 0x1c")          # group_size
    em("s_load_dword s24, s[4:5], 0x20")          # nbpr
    em("s_load_dword s25, s[4:5], 0x24")          # magic_nbpr
    em("s_load_dword s26, s[4:5], 0x28")          # rows_per_exp
    em("s_load_dword s27, s[4:5], 0x2c")          # magic_rpe
    # 注意：目标对 >= s[30:31] 的 `s_load_dwordx2 s[x:x+1]` 在本机实测会读到
    # 垃圾（同偏移用两条 `s_load_dword` 就正确），所以表指针拆成两条 4 字节读。
    for i in range(n_tables):
        em(f"s_load_dword s{28 + 2 * i}, s[4:5], {0x30 + 8 * i:#x}")
        em(f"s_load_dword s{29 + 2 * i}, s[4:5], {0x34 + 8 * i:#x}")
    nid = 28 + 2 * n_tables
    em(f"s_load_dword s{nid}, s[4:5], {0x30 + 8 * n_tables:#x}")      # ids
    em(f"s_load_dword s{nid + 1}, s[4:5], {0x34 + 8 * n_tables:#x}")
    em(f"s_load_dword s{nid + 2}, s[4:5], {0x38 + 8 * n_tables:#x}")  # 专家组跨度
    em(f"s_load_dword s{nid + 3}, s[4:5], {0x3c + 8 * n_tables:#x}")  # 权重侧每专家行数
    em(f"s_load_dword s{nid + 4}, s[4:5], {0x40 + 8 * n_tables:#x}")  # 其魔法数
    em("s_waitcnt lgkmcnt(0)")
    em("v_mov_b32_e32 v3, 0")
    em("v_mov_b32_e32 v1, s6")
    em("v_mul_lo_u32 v1, v1, s23")
    em("v_add_u32_e32 v1, v0, v1")                # i
    em("v_cmp_gt_u32_e32 vcc, s22, v1")
    em("s_and_saveexec_b64 s[0:1], vcc")
    em("s_cbranch_execz L_end")
    _divmod(L, "v1", "s24", "s25", "v50", "v52", "v51")   # q1, b
    _divmod(L, "v50", "s26", "s27", "v53", "v55", "v54")  # e
    em("v_mul_lo_u32 v56, v53, s24")
    em("v_add_u32_e32 v56, v56, v52")              # xblk
    # 权重侧的专家组号 ew = q1 / rows_per_w（gate/up 的 x 只有一组，ew 单独算）
    _divmod(L, "v50", f"s{nid + 3}", f"s{nid + 4}", "v125", "v126", "v127")
    em("v_mul_lo_u32 v124, v126, s24")
    em("v_add_u32_e32 v124, v124, v52")
    em("v_lshlrev_b32_e32 v121, 2, v125")
    em(f"v_mov_b32_e32 v122, s{nid}")
    em(f"v_mov_b32_e32 v123, s{nid + 1}")
    em("v_add_co_u32_e32 v122, vcc, v122, v121")
    em("v_addc_co_u32_e32 v123, vcc, v123, v3, vcc")
    em("global_load_dword v121, v[122:123], off")
    em("s_waitcnt vmcnt(0)")
    em(f"v_mul_lo_u32 v120, v121, s{nid + 2}")
    for r in acc:
        em(f"v_mov_b32_e32 v{r}, 0")


def dot_tail(L: list[str], acc=None) -> None:
    """公共收尾：4 路求和 → partial[i]。"""
    acc = DOT_ACC if acc is None else acc
    em = L.append
    em(f"v_add_f32_e32 v{acc[0]}, v{acc[0]}, v{acc[1]}")
    em(f"v_add_f32_e32 v{acc[2]}, v{acc[2]}, v{acc[3]}")
    em(f"v_add_f32_e32 v{acc[0]}, v{acc[0]}, v{acc[2]}")
    em("v_lshlrev_b32_e32 v57, 2, v1")
    em("v_mov_b32_e32 v59, s21")
    em("v_add_co_u32_e32 v58, vcc, s20, v57")
    em("v_addc_co_u32_e32 v59, vcc, v59, v3, vcc")
    em(f"global_store_dword v[58:59], v{acc[0]}, off")
    em("L_end:")
    em("s_or_b64 exec, exec, s[0:1]")
    em("s_endpgm")


def dot_xaddr(L: list[str], blk_shift: int) -> None:
    """v56（xblk）→ v6:v7 = x + xblk*BLK*4，`blk_shift = log2(BLK*4)`。"""
    em = L.append
    em(f"v_lshlrev_b32_e32 v56, {blk_shift}, v56")
    em("v_mov_b32_e32 v6, s18")
    em("v_mov_b32_e32 v7, s19")
    em("v_add_co_u32_e32 v6, vcc, v6, v56")
    em("v_addc_co_u32_e32 v7, vcc, v7, v3, vcc")


# --------------------------------------------------------------------------
# 行和内核（partial[总块数] → y[总行数]）
# --------------------------------------------------------------------------
REDUCE_NAME = "reduce_blocks_k"
REDUCE_ARGS = [buffer_arg(0), buffer_arg(8), scalar_arg(16, 4), scalar_arg(20, 4)]
REDUCE_KERNARG = 24


def gen_reduce_asm() -> str:
    L = [
        ".text", f"k_{REDUCE_NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",    # partial
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",    # y
        "s_load_dword s20, s[4:5], 0x10",          # nrows
        "s_load_dword s21, s[4:5], 0x14",          # nbpr
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v3, 0",
        "v_mov_b32_e32 v1, s6",
        "v_mul_lo_u32 v1, v1, 64",
        "v_add_u32_e32 v1, v0, v1",                # row
        "v_cmp_gt_u32_e32 vcc, s20, v1",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        "v_mul_lo_u32 v2, v1, s21",
        "v_lshlrev_b32_e32 v2, 2, v2",
        "v_mov_b32_e32 v4, s16",
        "v_mov_b32_e32 v5, s17",
        "v_add_co_u32_e32 v4, vcc, v4, v2",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "v_mov_b32_e32 v6, 0",                     # acc
        "v_mov_b32_e32 v8, 4",                     # 每步 4 字节
        "s_mov_b32 s22, 0",
        "L_loop:",
        "s_cmp_lt_u32 s22, s21",
        "s_cbranch_scc0 L_done",
        "global_load_dword v7, v[4:5], off",
        "s_waitcnt vmcnt(0)",
        "v_add_f32_e32 v6, v6, v7",
        "v_add_co_u32_e32 v4, vcc, v4, v8",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "s_add_i32 s22, s22, 1",
        "s_branch L_loop",
        "L_done:",
        "v_lshlrev_b32_e32 v9, 2, v1",
        "v_mov_b32_e32 v11, s19",
        "v_add_co_u32_e32 v10, vcc, s18, v9",
        "v_addc_co_u32_e32 v11, vcc, v11, v3, vcc",
        "global_store_dword v[10:11], v6, off",
        "L_end:",
        "s_or_b64 exec, exec, s[0:1]",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# IQ4_NL 融合点积（32 权重 / 18 字节块）
# --------------------------------------------------------------------------
IQ4NL_NAME = "iq4nl_dot_k"
IQ4NL_ARGS = dot_args(0)
IQ4NL_KERNARG = dot_kernarg(0)


def gen_iq4nl_dot_asm() -> str:
    L: list[str] = []
    em = L.append

    em(".text")
    em(f"k_{IQ4NL_NAME}:")
    dot_head(L, 0)
    dot_xaddr(L, 7)                                # BLK=32 → *128 B
    # waddr = w + i*18
    em("v_mul_lo_u32 v2, v124, 18")
    em("v_add_u32_e32 v2, v2, v120")
    em("v_mov_b32_e32 v4, s16")
    em("v_mov_b32_e32 v5, s17")
    em("v_add_co_u32_e32 v4, vcc, v4, v2")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")
    em(f"v_mov_b32_e32 v9, {LUT_LO_HI:#010x}")
    em(f"v_mov_b32_e32 v10, {LUT_LO_LO:#010x}")
    em(f"v_mov_b32_e32 v11, {LUT_HI_HI:#010x}")
    em(f"v_mov_b32_e32 v12, {LUT_HI_LO:#010x}")
    em("v_mov_b32_e32 v13, 0x0000ff00")
    em("global_load_ushort v8, v[4:5], off")
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v8, v8")                 # 块尺度 d

    def lut(src: str) -> None:
        em(f"v_and_b32_e32 v28, 0x07070707, {src}")
        em("v_perm_b32 v25, v9, v10, v28")
        em("v_perm_b32 v26, v11, v12, v28")
        em(f"v_and_b32_e32 v27, 0x08080808, {src}")
        em("v_lshrrev_b32_e32 v27, 3, v27")
        em("v_perm_b32 v28, v3, v13, v27")
        em("v_bfi_b32 v29, v28, v26, v25")

    def fma4(off: int) -> None:
        em(f"global_load_dwordx4 v[44:47], v[6:7], off offset:{off}")
        em("s_waitcnt vmcnt(0)")
        for i in range(4):
            em(f"v_lshrrev_b32_e32 v30, {8 * i}, v29")
            em("v_lshlrev_b32_e32 v30, 24, v30")
            em("v_ashrrev_i32_e32 v30, 24, v30")
            em("v_cvt_f32_i32_e32 v30, v30")
            em("v_mul_f32_e32 v30, v8, v30")
            em(f"v_fma_f32 v{DOT_ACC[i]}, v30, v{44 + i}, v{DOT_ACC[i]}")

    for j in range(4):
        for b in range(4):
            em(f"global_load_ubyte v{14 + b}, v[4:5], off offset:{2 + 4 * j + b}")
        em("s_waitcnt vmcnt(0)")
        em("v_lshlrev_b32_e32 v22, 8, v15")
        em("v_or_b32_e32 v22, v14, v22")
        em("v_lshlrev_b32_e32 v23, 8, v17")
        em("v_or_b32_e32 v23, v16, v23")
        em("v_lshlrev_b32_e32 v23, 16, v23")
        em("v_or_b32_e32 v18, v22, v23")            # qs dword j
        em("v_and_b32_e32 v24, 0x0f0f0f0f, v18")
        lut("v24")
        fma4(16 * j)
        em("v_and_b32_e32 v24, 0xf0f0f0f0, v18")
        em("v_lshrrev_b32_e32 v24, 4, v24")
        em("v_and_b32_e32 v24, 0x0f0f0f0f, v24")
        lut("v24")
        fma4(64 + 16 * j)

    dot_tail(L)
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# IQ2_S 融合点积（256 权重 / 82 字节块，256×u64 网格表）
# --------------------------------------------------------------------------
IQ2S_NAME = "iq2s_dot_k"
IQ2S_ARGS = dot_args(1)
IQ2S_KERNARG = dot_kernarg(1)


def _iq23_value(L, g, bit, scale, out, sgn_bit=None):
    byte = bit & 3
    L += [
        f"v_lshrrev_b32_e32 v35, {8 * byte}, {g}",
        "v_and_b32_e32 v35, 0xff, v35",
        "v_cvt_f32_u32_e32 v35, v35",
        f"v_lshrrev_b32_e32 v36, {bit & 7 if sgn_bit is None else sgn_bit}, v28",
        "v_and_b32_e32 v36, 1, v36",
        "v_cvt_f32_u32_e32 v36, v36",
        "v_fma_f32 v36, v36, -2.0, 1.0",
        "v_mul_f32_e32 v35, v35, v36",
        f"v_mul_f32_e32 {out}, {scale}, v35",
    ]


def _dot_fma4(L, elem):
    L.append(f"global_load_dwordx4 v[64:67], v[6:7], off offset:{4 * elem}")
    L.append("s_waitcnt vmcnt(0)")
    for j in range(4):
        L.append(f"v_fma_f32 v{DOT_ACC[j]}, v{40 + j}, v{64 + j}, v{DOT_ACC[j]}")


def gen_iq2s_dot_asm() -> str:
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{IQ2S_NAME}:")
    dot_head(L, 1)
    dot_xaddr(L, 10)
    em("v_mov_b32_e32 v9, s28")
    em("v_mov_b32_e32 v10, s29")
    em("v_mov_b32_e32 v80, 82")
    em("v_mul_lo_u32 v2, v124, v80")
    em("v_add_u32_e32 v2, v2, v120")
    em("v_mov_b32_e32 v4, s16")
    em("v_mov_b32_e32 v5, s17")
    em("v_add_co_u32_e32 v4, vcc, v4, v2")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")
    em("global_load_ushort v8, v[4:5], off")
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v8, v8")
    for ib in range(8):
        em(f"global_load_ubyte v13, v[4:5], off offset:{66 + ib}")
        em(f"global_load_ubyte v15, v[4:5], off offset:{74 + ib}")
        em("s_waitcnt vmcnt(0)")
        for reg, sh, mul in (("v44", 0, None), ("v45", 4, None)):
            if sh:
                em(f"v_lshrrev_b32_e32 v16, {sh}, v15")
            else:
                em("v_and_b32_e32 v16, 0xf, v15")
            em("v_cvt_f32_u32_e32 v16, v16")
            em("v_add_f32_e32 v16, 0.5, v16")
            em("v_mul_f32_e32 v16, 0.5, v16")
            em("v_mul_f32_e32 v16, 0.5, v16")
            em(f"v_mul_f32_e32 {reg}, v8, v16")
        for l in range(4):
            em(f"global_load_ubyte v14, v[4:5], off offset:{2 + 4 * ib + l}")
            em(f"global_load_ubyte v28, v[4:5], off offset:{34 + 4 * ib + l}")
            em("s_waitcnt vmcnt(0)")
            em(f"v_lshlrev_b32_e32 v16, {8 - 2 * l}, v13")
            em("v_and_b32_e32 v16, 0x300, v16")
            em("v_or_b32_e32 v14, v14, v16")
            em("v_lshlrev_b32_e32 v16, 3, v14")
            em("v_mov_b32_e32 v29, v9")
            em("v_mov_b32_e32 v30, v10")
            em("v_add_co_u32_e32 v29, vcc, v29, v16")
            em("v_addc_co_u32_e32 v30, vcc, v30, v3, vcc")
            em("global_load_dwordx2 v[33:34], v[29:30], off")
            em("s_waitcnt vmcnt(0)")
            scale = "v44" if l < 2 else "v45"
            for j in range(4):
                _iq23_value(L, "v33", j, scale, f"v{40 + j}")
            elem = ib * 32 + 8 * l
            _dot_fma4(L, elem)
            for j in range(4):
                _iq23_value(L, "v34", 4 + j, scale, f"v{40 + j}")
            _dot_fma4(L, elem + 4)
    dot_tail(L)
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# IQ3_S 融合点积（256 权重 / 110 字节块，512×u32 网格表）
# --------------------------------------------------------------------------
IQ3S_NAME = "iq3s_dot_k"
IQ3S_ARGS = dot_args(1)
IQ3S_KERNARG = dot_kernarg(1)


def gen_iq3s_dot_asm() -> str:
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{IQ3S_NAME}:")
    dot_head(L, 1)
    dot_xaddr(L, 10)
    em("v_mov_b32_e32 v9, s28")
    em("v_mov_b32_e32 v10, s29")
    em("v_mov_b32_e32 v80, 110")
    em("v_mul_lo_u32 v2, v124, v80")
    em("v_add_u32_e32 v2, v2, v120")
    em("v_mov_b32_e32 v4, s16")
    em("v_mov_b32_e32 v5, s17")
    em("v_add_co_u32_e32 v4, vcc, v4, v2")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")
    em("global_load_ushort v8, v[4:5], off")
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v8, v8")
    for pair in range(4):
        em(f"global_load_ubyte v15, v[4:5], off offset:{106 + pair}")
        em("s_waitcnt vmcnt(0)")
        em("v_and_b32_e32 v16, 0xf, v15")
        em("v_cvt_f32_u32_e32 v16, v16")
        em("v_fma_f32 v16, v16, 2.0, 1.0")
        em("v_mul_f32_e32 v44, v8, v16")
        em("v_lshrrev_b32_e32 v17, 4, v15")
        em("v_cvt_f32_u32_e32 v17, v17")
        em("v_fma_f32 v17, v17, 2.0, 1.0")
        em("v_mul_f32_e32 v45, v8, v17")
        for half in range(2):
            em(f"global_load_ubyte v13, v[4:5], off offset:{66 + 2 * pair + half}")
            for l in range(4):
                q0, q1 = 20 + 2 * l, 21 + 2 * l
                em(f"global_load_ubyte v{q0}, v[4:5], off offset:"
                   f"{2 + 16 * pair + 8 * half + 2 * l}")
                em(f"global_load_ubyte v{q1}, v[4:5], off offset:"
                   f"{3 + 16 * pair + 8 * half + 2 * l}")
                em(f"global_load_ubyte v28, v[4:5], off offset:"
                   f"{74 + 8 * pair + 4 * half + l}")
                em("s_waitcnt vmcnt(0)")
                scale = "v44" if half == 0 else "v45"
                elem_base = pair * 64 + half * 32 + 8 * l
                for gi, qreg in ((0, q0), (1, q1)):
                    sh = (8 - 2 * l) if gi == 0 else (7 - 2 * l)
                    em(f"v_lshlrev_b32_e32 v16, {sh}, v13")
                    em("v_and_b32_e32 v16, 0x100, v16")
                    em(f"v_or_b32_e32 v14, v{qreg}, v16")
                    em("v_lshlrev_b32_e32 v17, 2, v14")
                    em("v_mov_b32_e32 v29, v9")
                    em("v_mov_b32_e32 v30, v10")
                    em("v_add_co_u32_e32 v29, vcc, v29, v17")
                    em("v_addc_co_u32_e32 v30, vcc, v30, v3, vcc")
                    em("global_load_dword v33, v[29:30], off")
                    em("s_waitcnt vmcnt(0)")
                    for j in range(4):
                        _iq23_value(L, "v33", j + (0 if gi == 0 else 4), scale,
                                    f"v{40 + j}")
                    elem = elem_base + 4 * gi
                    _dot_fma4(L, elem)
    dot_tail(L)
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# Q2_0 融合点积（64 权重 / 18 字节块）
# --------------------------------------------------------------------------
Q20_NAME = "q2_0_dot_k"
Q20_ARGS = dot_args(0)
Q20_KERNARG = dot_kernarg(0)


def gen_q2_0_dot_asm() -> str:
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{Q20_NAME}:")
    dot_head(L, 0)
    dot_xaddr(L, 8)                                # BLK=64 → *256 B
    em("v_mul_lo_u32 v2, v124, 18")
    em("v_add_u32_e32 v2, v2, v120")
    em("v_mov_b32_e32 v4, s16")
    em("v_mov_b32_e32 v5, s17")
    em("v_add_co_u32_e32 v4, vcc, v4, v2")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")
    em("global_load_ushort v8, v[4:5], off")
    for b in range(16):
        em(f"global_load_ubyte v{48 + b}, v[4:5], off offset:{2 + b}")
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v8, v8")
    for b in range(16):
        for k in range(4):
            if k == 0:
                em(f"v_and_b32_e32 v{24 + k}, 3, v{48 + b}")
            else:
                em(f"v_lshrrev_b32_e32 v{24 + k}, {2 * k}, v{48 + b}")
                em(f"v_and_b32_e32 v{24 + k}, 3, v{24 + k}")
            em(f"v_cvt_f32_u32_e32 v{24 + k}, v{24 + k}")
            em(f"v_add_f32_e32 v{24 + k}, -1.0, v{24 + k}")
            em(f"v_mul_f32_e32 v{24 + k}, v8, v{24 + k}")
        off = 16 * b
        em(f"global_load_dwordx4 v[64:67], v[6:7], off offset:{off}")
        em("s_waitcnt vmcnt(0)")
        for j in range(4):
            em(f"v_fma_f32 v{DOT_ACC[j]}, v{24 + j}, v{64 + j}, v{DOT_ACC[j]}")
    dot_tail(L)
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# IQ4_XS 融合点积（256 权重 / 136 字节块）
# --------------------------------------------------------------------------
IQ4XS_ACC = [110, 111, 112, 113]
IQ4XS_NAME = "iq4xs_dot_k"
IQ4XS_ARGS = dot_args(0)
IQ4XS_KERNARG = dot_kernarg(0)


def gen_iq4xs_dot_asm() -> str:
    L: list[str] = []
    em = L.append

    def decode(selector: str) -> None:
        em(f"v_and_b32_e32 v31, 0x07070707, {selector}")
        em("v_perm_b32 v25, v9, v10, v31")
        em("v_perm_b32 v26, v11, v12, v31")
        em(f"v_and_b32_e32 v28, 0x08080808, {selector}")
        em("v_lshrrev_b32_e32 v28, 3, v28")
        em("v_perm_b32 v27, v3, v13, v28")
        em("v_bfi_b32 v30, v27, v26, v25")

    def fma4(off: int) -> None:
        for i in range(4):
            em(f"v_lshrrev_b32_e32 v{40 + i}, {8 * i}, v30")
            em(f"v_lshlrev_b32_e32 v{40 + i}, 24, v{40 + i}")
            em(f"v_ashrrev_i32_e32 v{40 + i}, 24, v{40 + i}")
            em(f"v_cvt_f32_i32_e32 v{40 + i}, v{40 + i}")
            em(f"v_mul_f32_e32 v{40 + i}, v46, v{40 + i}")
        # 注意：v48..v79 被 qs 的 dwordx2 占着，x 临时寄存器要用更高的。
        em(f"global_load_dwordx4 v[100:103], v[6:7], off offset:{off}")
        em("s_waitcnt vmcnt(0)")
        for i in range(4):
            em(f"v_fma_f32 v{IQ4XS_ACC[i]}, v{40 + i}, v{100 + i}, v{IQ4XS_ACC[i]}")

    em(".text")
    em(f"k_{IQ4XS_NAME}:")
    dot_head(L, 0, IQ4XS_ACC)
    dot_xaddr(L, 10)
    em("v_mov_b32_e32 v9, 0xf6eaddcf")
    em("v_mov_b32_e32 v10, 0xbfad9881")
    em("v_mov_b32_e32 v11, 0x71594535")
    em("v_mov_b32_e32 v12, 0x26190d01")
    em("v_mov_b32_e32 v13, 0x0000ff00")
    em("v_mov_b32_e32 v47, 0xc2000000")
    em("v_mov_b32_e32 v80, 136")
    em("v_mul_lo_u32 v2, v124, v80")
    em("v_add_u32_e32 v2, v2, v120")
    em("v_mov_b32_e32 v4, s16")
    em("v_mov_b32_e32 v5, s17")
    em("v_add_co_u32_e32 v4, vcc, v4, v2")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")
    em("global_load_ushort v8, v[4:5], off")
    em("global_load_ubyte v16, v[4:5], off offset:2")
    em("global_load_ubyte v17, v[4:5], off offset:3")
    em("global_load_dword v14, v[4:5], off offset:4")
    for m in range(16):
        em(f"global_load_dwordx2 v[{48 + 2 * m}:{49 + 2 * m}], v[4:5], "
           f"off offset:{8 + 8 * m}")
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v8, v8")
    em("v_lshlrev_b32_e32 v17, 8, v17")
    em("v_or_b32_e32 v15, v16, v17")
    for ib in range(8):
        em(f"v_lshrrev_b32_e32 v44, {4 * ib}, v14")
        em("v_and_b32_e32 v44, 0xf, v44")
        em(f"v_lshrrev_b32_e32 v45, {2 * ib}, v15")
        em("v_and_b32_e32 v45, 3, v45")
        em("v_lshlrev_b32_e32 v45, 4, v45")
        em("v_or_b32_e32 v44, v44, v45")
        em("v_cvt_f32_u32_e32 v46, v44")
        em("v_add_f32_e32 v46, v47, v46")
        em("v_mul_f32_e32 v46, v8, v46")
        for j in range(4):
            qreg = 48 + 4 * ib + j
            em(f"v_and_b32_e32 v20, 0x0f0f0f0f, v{qreg}")
            decode("v20")
            fma4(128 * ib + 16 * j)
            em(f"v_and_b32_e32 v20, 0xf0f0f0f0, v{qreg}")
            em("v_lshrrev_b32_e32 v20, 4, v20")
            em("v_and_b32_e32 v20, 0x0f0f0f0f, v20")
            decode("v20")
            fma4(128 * ib + 64 + 16 * j)
    dot_tail(L, IQ4XS_ACC)
    return "\n".join(L) + "\n"


def _gen_q6k_dot_asm() -> str:
    """Q6_K 融合点积（256 权重 / 210 字节块：ql[128] + qh[64] + sc[16] + d）。

    元素映射（与 `iq_dequant.dequant_q6_k` 一致）：
        half∈{0,1}，l∈[0,32)：
          q1 = (ql[l]&0xF)      | ((qh[l]>>0 &3)<<4) - 32 → 元素 half*128+l
          q2 = (ql[l+32]&0xF)   | ((qh[l]>>2 &3)<<4) - 32 → half*128+32+l
          q3 = (ql[l]>>4)       | ((qh[l]>>4 &3)<<4) - 32 → half*128+64+l
          q4 = (ql[l+32]>>4)    | ((qh[l]>>6 &3)<<4) - 32 → half*128+96+l
        尺度：16 个 int8，q1/q2/q3/q4 分别用 sc[half*8+is+0/2/4/6]（is=l//16）。
    """
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{Q6K_NAME}:")
    dot_head(L, 0)
    dot_xaddr(L, 10)                               # BLK=256 → *1024 B
    em("v_mov_b32_e32 v48, 210")                   # v_mul_lo_u32 的大立即数不可靠
    em("v_mul_lo_u32 v2, v124, v48")
    em("v_add_u32_e32 v2, v2, v120")
    em("v_mov_b32_e32 v4, s16")
    em("v_mov_b32_e32 v5, s17")
    em("v_add_co_u32_e32 v4, vcc, v4, v2")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")
    # global_load_ushort 没有 offset 形式：把指针先推到 d 的位置
    em("v_mov_b32_e32 v44, v4")
    em("v_mov_b32_e32 v45, v5")
    em("v_mov_b32_e32 v46, 208")
    em("v_add_co_u32_e32 v44, vcc, v44, v46")
    em("v_addc_co_u32_e32 v45, vcc, v45, v3, vcc")
    em("global_load_ushort v8, v[44:45], off")
    em("global_load_dwordx4 v[32:35], v[4:5], off offset:192")
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v8, v8")
    em("v_mov_b32_e32 v47, 0xc2000000")            # -32.0f（不是 inline 常量）
    # 16 个 int8 尺度 × d → v128..v143
    for n in range(16):
        q, b = n // 4, n % 4
        em(f"v_lshrrev_b32_e32 v36, {8 * b}, v{32 + q}")
        em("v_lshlrev_b32_e32 v36, 24, v36")
        em("v_ashrrev_i32_e32 v36, 24, v36")
        em("v_cvt_f32_i32_e32 v36, v36")
        em(f"v_mul_f32_e32 v{128 + n}, v8, v36")
    for half in range(2):
        for l0 in range(0, 32, 4):
            sb = half * 8 + l0 // 16
            em(f"global_load_dword v20, v[4:5], off offset:{half * 64 + l0}")
            em(f"global_load_dword v21, v[4:5], off offset:{half * 64 + 32 + l0}")
            em(f"global_load_dword v22, v[4:5], off offset:{128 + half * 32 + l0}")
            for base in (0, 32, 64, 96):
                em(f"global_load_dwordx4 v[{96 + base // 8}:{99 + base // 8}], "
                   f"v[6:7], off offset:{4 * (half * 128 + base + l0)}")
            em("s_waitcnt vmcnt(0)")
            for si, (dreg, hi, base, qs) in enumerate(
                    (("v20", 0, 0, 0), ("v21", 0, 32, 2),
                     ("v20", 1, 64, 4), ("v21", 1, 96, 6))):
                xb = 96 + base // 8
                for j in range(4):
                    em(f"v_lshrrev_b32_e32 v36, {8 * j}, {dreg}")
                    em("v_and_b32_e32 v36, 0xff, v36")
                    if hi == 0:
                        em("v_and_b32_e32 v36, 0xf, v36")
                    else:
                        em("v_lshrrev_b32_e32 v36, 4, v36")
                    em(f"v_lshrrev_b32_e32 v37, {8 * j}, v22")
                    em("v_and_b32_e32 v37, 0xff, v37")
                    if qs:
                        em(f"v_lshrrev_b32_e32 v37, {qs}, v37")
                    em("v_and_b32_e32 v37, 3, v37")
                    em("v_lshlrev_b32_e32 v37, 4, v37")
                    em("v_or_b32_e32 v36, v36, v37")
                    em("v_cvt_f32_u32_e32 v36, v36")
                    em("v_add_f32_e32 v36, v47, v36")
                    em(f"v_mul_f32_e32 v36, v{128 + sb + 2 * si}, v36")
                    em(f"v_fma_f32 v{DOT_ACC[j]}, v36, v{xb + j}, v{DOT_ACC[j]}")
    dot_tail(L)
    return "\n".join(L) + "\n"


Q6K_NAME = "q6k_dot_k"
Q6K_ARGS = dot_args(0)
Q6K_KERNARG = dot_kernarg(0)


def gen_q6k_dot_asm() -> str:
    return _gen_q6k_dot_asm()


# --------------------------------------------------------------------------
# Q4_K / Q5_K / Q8_0 / Q4_0 融合点积
# --------------------------------------------------------------------------
Q4K_NAME = "q4k_dot_k"
Q4K_ARGS = dot_args(0)
Q4K_KERNARG = dot_kernarg(0)

Q5K_NAME = "q5k_dot_k"
Q5K_ARGS = dot_args(0)
Q5K_KERNARG = dot_kernarg(0)

Q80_NAME = "q8_0_dot_k"
Q80_ARGS = dot_args(0)
Q80_KERNARG = dot_kernarg(0)

Q40_NAME = "q4_0_dot_k"
Q40_ARGS = dot_args(0)
Q40_KERNARG = dot_kernarg(0)


def _emit_waddr(L: list[str], block_bytes: int) -> None:
    """v4:v5 = w + ids[ew]*stride + 专家内块号*块字节数。"""
    em = L.append
    em(f"v_mov_b32_e32 v48, {block_bytes}")
    em("v_mul_lo_u32 v2, v124, v48")
    em("v_add_u32_e32 v2, v2, v120")
    em("v_mov_b32_e32 v4, s16")
    em("v_mov_b32_e32 v5, s17")
    em("v_add_co_u32_e32 v4, vcc, v4, v2")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")


def _emit_nibble(L: list[str], dst: str, src: str, shift: int) -> None:
    """dst = (src >> shift) & 0xF；shift=0 时不发多余指令。"""
    if shift:
        L.append(f"v_lshrrev_b32_e32 {dst}, {shift}, {src}")
        L.append(f"v_and_b32_e32 {dst}, 0xf, {dst}")
    else:
        L.append(f"v_and_b32_e32 {dst}, 0xf, {src}")


def _emit_k_scale(L: list[str], j: int, a_reg: str, nb_reg: str) -> None:
    """Q4_K/Q5_K 第 j 组（j=0..7）的 `a=d*sc`、`nb=-(dmin*m)`。

    尺度字节 s0..s11 已装在 v16..v18；d、dmin 在 v9、v10。
    `_scale_min_k4` 的分段规则与 `iq_dequant.dequant_q4_k` 一致：
        j<4:  sc=s[j]&63,               m=s[j+4]&63
        j>=4: sc=(s[j+4]&15)|((s[j-4]>>6)<<4)
              m =(s[j+4]>>4)|((s[j]>>6)<<4)
    """
    em = L.append

    def ld_byte(dst: str, idx: int) -> None:
        dw, sh = idx // 4, 8 * (idx % 4)
        if sh:
            em(f"v_lshrrev_b32_e32 {dst}, {sh}, v{16 + dw}")
        else:
            em(f"v_mov_b32_e32 {dst}, v{16 + dw}")
        em(f"v_and_b32_e32 {dst}, 0xff, {dst}")

    if j < 4:
        ld_byte("v60", j)
        em("v_and_b32_e32 v60, 63, v60")
        ld_byte("v61", j + 4)
        em("v_and_b32_e32 v61, 63, v61")
    else:
        ld_byte("v62", j + 4)
        em("v_and_b32_e32 v60, 15, v62")
        ld_byte("v63", j - 4)
        em("v_lshrrev_b32_e32 v63, 6, v63")
        em("v_lshlrev_b32_e32 v63, 4, v63")
        em("v_or_b32_e32 v60, v60, v63")
        em("v_lshrrev_b32_e32 v61, 4, v62")
        ld_byte("v63", j)
        em("v_lshrrev_b32_e32 v63, 6, v63")
        em("v_lshlrev_b32_e32 v63, 4, v63")
        em("v_or_b32_e32 v61, v61, v63")
    em("v_cvt_f32_u32_e32 v60, v60")
    em(f"v_mul_f32_e32 {a_reg}, v9, v60")          # a = d * sc
    em("v_cvt_f32_u32_e32 v61, v61")
    em(f"v_mul_f32_e32 {nb_reg}, v10, v61")        # dmin * m
    em(f"v_mul_f32_e32 {nb_reg}, -1.0, {nb_reg}")  # nb = -(dmin * m)


def gen_q4_0_dot_asm() -> str:
    """Q4_0：32 权重 / 18 字节块（f16 d + 16B qs），值 = d*(q-8)。"""
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{Q40_NAME}:")
    dot_head(L, 0)
    dot_xaddr(L, 7)                                # BLK=32 → *128 B
    _emit_waddr(L, 18)
    em("global_load_ushort v8, v[4:5], off")
    for b in range(16):
        em(f"global_load_ubyte v{20 + b}, v[4:5], off offset:{2 + b}")
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v8, v8")
    em("v_mov_b32_e32 v36, 8")
    for q in range(4):
        # 低半字节 → 元素 4q..4q+3
        em(f"global_load_dwordx4 v[40:43], v[6:7], off offset:{16 * q}")
        em("s_waitcnt vmcnt(0)")
        for k in range(4):
            _emit_nibble(L, "v38", f"v{20 + 4 * q + k}", 0)
            em("v_sub_u32_e32 v38, v38, v36")
            em("v_cvt_f32_i32_e32 v38, v38")
            em("v_mul_f32_e32 v38, v38, v8")
            em(f"v_fma_f32 v{DOT_ACC[k]}, v38, v{40 + k}, v{DOT_ACC[k]}")
        # 高半字节 → 元素 16+4q..16+4q+3
        em(f"global_load_dwordx4 v[40:43], v[6:7], off offset:{4 * (16 + 4 * q)}")
        em("s_waitcnt vmcnt(0)")
        for k in range(4):
            _emit_nibble(L, "v38", f"v{20 + 4 * q + k}", 4)
            em("v_sub_u32_e32 v38, v38, v36")
            em("v_cvt_f32_i32_e32 v38, v38")
            em("v_mul_f32_e32 v38, v38, v8")
            em(f"v_fma_f32 v{DOT_ACC[k]}, v38, v{40 + k}, v{DOT_ACC[k]}")
    dot_tail(L)
    return "\n".join(L) + "\n"


def gen_q8_0_dot_asm() -> str:
    """Q8_0：32 权重 / 34 字节块（f16 d + 32×int8），值 = d*q。"""
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{Q80_NAME}:")
    dot_head(L, 0)
    dot_xaddr(L, 7)                                # BLK=32 → *128 B
    _emit_waddr(L, 34)
    em("global_load_ushort v8, v[4:5], off")
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v8, v8")
    for q in range(8):
        for k in range(4):
            em(f"global_load_ubyte v{20 + k}, v[4:5], off offset:{2 + 4 * q + k}")
        em(f"global_load_dwordx4 v[40:43], v[6:7], off offset:{16 * q}")
        em("s_waitcnt vmcnt(0)")
        for k in range(4):
            em(f"v_lshlrev_b32_e32 v30, 24, v{20 + k}")
            em("v_ashrrev_i32_e32 v30, 24, v30")
            em("v_cvt_f32_i32_e32 v30, v30")
            em("v_mul_f32_e32 v30, v30, v8")
            em(f"v_fma_f32 v{DOT_ACC[k]}, v30, v{40 + k}, v{DOT_ACC[k]}")
    dot_tail(L)
    return "\n".join(L) + "\n"


def gen_q4k_dot_asm() -> str:
    """Q4_K：256 权重 / 144 字节块，8 组 32 元素各带 (d*sc, dmin*m)。"""
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{Q4K_NAME}:")
    dot_head(L, 0)
    dot_xaddr(L, 10)                               # BLK=256 → *1024 B
    _emit_waddr(L, 144)
    em("global_load_dword v8, v[4:5], off")        # d | dmin
    em("global_load_dword v16, v[4:5], off offset:4")    # s0..s3
    em("global_load_dword v17, v[4:5], off offset:8")    # s4..s7
    em("global_load_dword v18, v[4:5], off offset:12")   # s8..s11
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v9, v8")
    em("v_lshrrev_b32_e32 v10, 16, v8")
    em("v_cvt_f32_f16_e32 v10, v10")               # dmin
    for s in range(4):
        em(f"global_load_dwordx4 v[20:23], v[4:5], off offset:{16 + 32 * s}")
        em(f"global_load_dwordx4 v[24:27], v[4:5], off offset:{32 + 32 * s}")
        em("s_waitcnt vmcnt(0)")
        _emit_k_scale(L, 2 * s, "v92", "v93")
        _emit_k_scale(L, 2 * s + 1, "v94", "v95")
        for half, a_reg, nb_reg in ((0, "v92", "v93"), (1, "v94", "v95")):
            xbase = 256 * s + 128 * half
            for l0 in range(0, 32, 4):
                em(f"global_load_dwordx4 v[40:43], v[6:7], off "
                   f"offset:{xbase + 4 * l0}")
                em("s_waitcnt vmcnt(0)")
                qreg = f"v{20 + l0 // 4}"
                for k in range(4):
                    _emit_nibble(L, "v30", qreg, 8 * k + 4 * half)
                    em("v_cvt_f32_u32_e32 v30, v30")
                    em(f"v_fma_f32 v30, v30, {a_reg}, {nb_reg}")
                    em(f"v_fma_f32 v{DOT_ACC[k]}, v30, v{40 + k}, v{DOT_ACC[k]}")
    dot_tail(L)
    return "\n".join(L) + "\n"


def gen_q5k_dot_asm() -> str:
    """Q5_K：256 权重 / 176 字节块；第 5 位来自 qh[32] 的 bit(2s)/bit(2s+1)。"""
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{Q5K_NAME}:")
    dot_head(L, 0)
    dot_xaddr(L, 10)                               # BLK=256 → *1024 B
    _emit_waddr(L, 176)
    em("global_load_dword v8, v[4:5], off")        # d | dmin
    em("global_load_dword v16, v[4:5], off offset:4")    # s0..s3
    em("global_load_dword v17, v[4:5], off offset:8")    # s4..s7
    em("global_load_dword v18, v[4:5], off offset:12")   # s8..s11
    em("global_load_dwordx4 v[80:83], v[4:5], off offset:16")   # qh[0:16]
    em("global_load_dwordx4 v[84:87], v[4:5], off offset:32")   # qh[16:32]
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v9, v8")
    em("v_lshrrev_b32_e32 v10, 16, v8")
    em("v_cvt_f32_f16_e32 v10, v10")               # dmin
    for s in range(4):
        em(f"global_load_dwordx4 v[20:23], v[4:5], off offset:{48 + 32 * s}")
        em(f"global_load_dwordx4 v[24:27], v[4:5], off offset:{64 + 32 * s}")
        em("s_waitcnt vmcnt(0)")
        _emit_k_scale(L, 2 * s, "v92", "v93")
        _emit_k_scale(L, 2 * s + 1, "v94", "v95")
        for half, a_reg, nb_reg in ((0, "v92", "v93"), (1, "v94", "v95")):
            xbase = 256 * s + 128 * half
            for l0 in range(0, 32, 4):
                em(f"global_load_dwordx4 v[40:43], v[6:7], off "
                   f"offset:{xbase + 4 * l0}")
                em("s_waitcnt vmcnt(0)")
                qreg = f"v{20 + l0 // 4}"
                qhreg = f"v{80 + l0 // 4}"
                for k in range(4):
                    _emit_nibble(L, "v30", qreg, 8 * k + 4 * half)
                    sh = 8 * k + 2 * s + half
                    if sh:
                        em(f"v_lshrrev_b32_e32 v31, {sh}, {qhreg}")
                    else:
                        em(f"v_mov_b32_e32 v31, {qhreg}")
                    em("v_and_b32_e32 v31, 1, v31")
                    em("v_lshlrev_b32_e32 v31, 4, v31")
                    em("v_or_b32_e32 v30, v30, v31")
                    em("v_cvt_f32_u32_e32 v30, v30")
                    em(f"v_fma_f32 v30, v30, {a_reg}, {nb_reg}")
                    em(f"v_fma_f32 v{DOT_ACC[k]}, v30, v{40 + k}, v{DOT_ACC[k]}")
    dot_tail(L)
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# 参考实现（numpy）：核对 GPU 结果
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IQ3_XXS 融合点积（256 权重 / 98 字节块，256×u32 网格表 + 128B 符号表）
# --------------------------------------------------------------------------
IQ3XXS_NAME = "iq3xxs_dot_k"
IQ3XXS_ARGS = dot_args(2)          # + grid 表、ksigns 表
IQ3XXS_KERNARG = dot_kernarg(2)


def gen_iq3xxs_dot_asm() -> str:
    L: list[str] = []
    em = L.append

    def value(g: str, bit: int) -> None:
        """g（4 个幅值字节）第 (bit&3) 字节 × 符号 → v40+（bit&3）。"""
        byte = bit & 3
        em(f"v_lshrrev_b32_e32 v35, {8 * byte}, {g}")
        em("v_and_b32_e32 v35, 0xff, v35")
        em("v_cvt_f32_u32_e32 v35, v35")
        em(f"v_lshrrev_b32_e32 v36, {bit}, v28")
        em("v_and_b32_e32 v36, 1, v36")
        em("v_cvt_f32_u32_e32 v36, v36")
        em("v_fma_f32 v36, v36, -2.0, 1.0")
        em("v_mul_f32_e32 v35, v35, v36")
        em(f"v_mul_f32_e32 v{40 + byte}, v14, v35")

    em(".text")
    em(f"k_{IQ3XXS_NAME}:")
    dot_head(L, 2)
    dot_xaddr(L, 10)                               # BLK=256 → *1024 B
    em("v_mov_b32_e32 v9, s28")                    # grid lo
    em("v_mov_b32_e32 v10, s29")                   # grid hi
    em("v_mov_b32_e32 v11, s30")                   # ksigns lo
    em("v_mov_b32_e32 v12, s31")                   # ksigns hi
    em("v_mov_b32_e32 v80, 98")
    em("v_mul_lo_u32 v2, v124, v80")
    em("v_add_u32_e32 v2, v2, v120")
    em("v_mov_b32_e32 v4, s16")
    em("v_mov_b32_e32 v5, s17")
    em("v_add_co_u32_e32 v4, vcc, v4, v2")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")   # waddr
    em("global_load_ushort v8, v[4:5], off")
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v8, v8")                 # d

    for ib in range(8):
        for q in range(4):
            em(f"global_load_ubyte v{15 + q}, v[4:5], off offset:{66 + 4 * ib + q}")
        for q in range(8):
            em(f"global_load_ubyte v{20 + q}, v[4:5], off offset:{2 + 8 * ib + q}")
        em("s_waitcnt vmcnt(0)")
        em("v_lshlrev_b32_e32 v16, 8, v16")
        em("v_or_b32_e32 v15, v15, v16")
        em("v_lshlrev_b32_e32 v18, 8, v18")
        em("v_or_b32_e32 v17, v17, v18")
        em("v_lshlrev_b32_e32 v18, 16, v17")
        em("v_or_b32_e32 v13, v15, v18")           # aux
        em("v_lshrrev_b32_e32 v15, 28, v13")
        em("v_cvt_f32_u32_e32 v15, v15")
        em("v_add_f32_e32 v15, 0.5, v15")
        em("v_mul_f32_e32 v15, 0.5, v15")
        em("v_mul_f32_e32 v14, v8, v15")           # db
        for l in range(4):
            if 7 * l == 0:
                em("v_and_b32_e32 v17, 127, v13")
            else:
                em(f"v_lshrrev_b32_e32 v17, {7 * l}, v13")
                em("v_and_b32_e32 v17, 127, v17")
            em("v_mov_b32_e32 v29, v11")
            em("v_mov_b32_e32 v30, v12")
            em("v_add_co_u32_e32 v29, vcc, v29, v17")
            em("v_addc_co_u32_e32 v30, vcc, v30, v3, vcc")
            em("global_load_ubyte v28, v[29:30], off")
            em("s_waitcnt vmcnt(0)")
            for gi, greg in ((0, "v33"), (1, "v34")):
                qreg = 20 + 2 * l + gi
                em(f"v_lshlrev_b32_e32 v17, 2, v{qreg}")
                em("v_mov_b32_e32 v31, v9")
                em("v_mov_b32_e32 v32, v10")
                em("v_add_co_u32_e32 v31, vcc, v31, v17")
                em("v_addc_co_u32_e32 v32, vcc, v32, v3, vcc")
                em(f"global_load_dword {greg}, v[31:32], off")
                em("s_waitcnt vmcnt(0)")
                base = 0 if gi == 0 else 4
                for j in range(4):
                    value(greg, base + j)
                elem = ib * 32 + 8 * l + 4 * gi
                em(f"global_load_dwordx4 v[64:67], v[6:7], off offset:{4 * elem}")
                em("s_waitcnt vmcnt(0)")
                for j in range(4):
                    em(f"v_fma_f32 v{DOT_ACC[j]}, v{40 + j}, v{64 + j}, v{DOT_ACC[j]}")
    dot_tail(L)
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# 参考实现（numpy）：核对 GPU 结果
# --------------------------------------------------------------------------
def ref_dot(w_raw: np.ndarray, nbpr: int, rows_per_exp: int,
            x: np.ndarray) -> np.ndarray:
    """w_raw: [总块数, 18] uint8；x: [xrow数, k] f32；返回 partial[总块数]。"""
    d = w_raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs = w_raw[:, 2:18]
    vals = np.empty((qs.shape[0], 32), dtype=np.int32)
    vals[:, 0:16] = (qs & 0x0F).astype(np.int32)
    vals[:, 16:32] = (qs >> 4).astype(np.int32)
    lut = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                    1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)
    wv = lut[vals] * d[:, None]
    nblk = w_raw.shape[0]
    out = np.empty(nblk, dtype=np.float32)
    xb = np.ascontiguousarray(x).reshape(-1, 32)
    for g in range(nblk):
        q1, b = divmod(g, nbpr)
        e = q1 // rows_per_exp
        out[g] = float(np.dot(wv[g], xb[e * nbpr + b]))
    return out


def _load_iq4nl_blocks(blocks: int,
                       prefer: str = "blk.0.ffn_down_shexp.weight") -> bytes:
    from gguf_sample import load_index
    idx = load_index()
    tensors = [t for t in idx["tensors"]
               if t["type"] == "IQ4_NL" and t["bytes"] >= blocks * 18]
    if not tensors:
        raise SystemExit("索引里找不到足够大的 IQ4_NL 张量")
    tensors.sort(key=lambda t: t["bytes"])
    t = next((x for x in tensors if x["name"] == prefer), tensors[0])
    src = pathlib.Path(idx["sources"][t["source"]]["path"])
    with open(src, "rb") as f:
        f.seek(t["offset"])
        raw = f.read(blocks * 18)
    if len(raw) != blocks * 18 or max(raw) == 0:
        raise SystemExit("权重读取失败（没下完？）")
    return raw


def selftest_iq4nl(nbpr: int = 20, rows_per_exp: int = 40) -> int:
    n_exp = 2
    total_rows = rows_per_exp * n_exp
    nblocks = total_rows * nbpr
    raw = _load_iq4nl_blocks(nblocks)
    w_raw = np.frombuffer(raw, dtype=np.uint8).reshape(nblocks, 18)
    k = nbpr * 32
    x = np.random.default_rng(5).standard_normal((n_exp, k)).astype(np.float32)
    m_nbpr = div_magic(nbpr, nblocks)
    m_rpe = div_magic(rows_per_exp, nblocks // nbpr + 1)
    ref = ref_dot(w_raw, nbpr, rows_per_exp, x)
    ref_y = ref.reshape(total_rows, nbpr).sum(axis=1)

    hsaco = build_one(IQ4NL_NAME, gen_iq4nl_dot_asm(), IQ4NL_ARGS,
                      kernarg_size=IQ4NL_KERNARG, tag="qdot")
    out = run_one(hsaco, IQ4NL_NAME,
                  [{"buffer": "w"}, {"buffer": "x"}, {"buffer": "p"},
                   {"scalar": {"dtype": "u32", "value": nblocks}},
                   {"scalar": {"dtype": "u32", "value": 64}},
                   {"scalar": {"dtype": "u32", "value": nbpr}},
                   {"scalar": {"dtype": "u32", "value": m_nbpr}},
                   {"scalar": {"dtype": "u32", "value": rows_per_exp}},
                   {"scalar": {"dtype": "u32", "value": m_rpe}},
                   {"buffer": "ids"},
                   {"scalar": {"dtype": "u32",
                               "value": rows_per_exp * nbpr * 18}},
                   {"scalar": {"dtype": "u32", "value": rows_per_exp}},
                   {"scalar": {"dtype": "u32",
                               "value": div_magic(rows_per_exp,
                                                  nblocks // nbpr + 1)}}],
                  {"w": {"dtype": "u8", "values": list(raw)},
                   "x": {"dtype": "f32", "values": x.reshape(-1).tolist()},
                   "p": {"dtype": "f32", "values": [0.0] * nblocks},
                   "ids": {"dtype": "u32", "values": list(range(n_exp))}},
                  grid=nblocks, workgroup=64)
    got = np.array(out["p"], dtype=np.float32)
    dmax = float(np.abs(got - ref).max())
    print(f"iq4nl_dot: nblocks={nblocks} nbpr={nbpr} rows/exp={rows_per_exp} "
          f"partial max_abs={dmax:.3e}")

    rh = build_one(REDUCE_NAME, gen_reduce_asm(), REDUCE_ARGS,
                   kernarg_size=REDUCE_KERNARG, tag="qdot")
    out2 = run_one(rh, REDUCE_NAME,
                   [{"buffer": "p"}, {"buffer": "y"},
                    {"scalar": {"dtype": "u32", "value": total_rows}},
                    {"scalar": {"dtype": "u32", "value": nbpr}}],
                   {"p": {"dtype": "f32", "values": list(got)},
                    "y": {"dtype": "f32", "values": [0.0] * total_rows}},
                   grid=total_rows, workgroup=64)
    y = np.array(out2["y"], dtype=np.float32)
    emax = float(np.abs(y - ref_y).max())
    scale = max(1.0, float(np.abs(ref_y).max()))
    print(f"reduce: max_abs={emax:.3e} (相对 {emax / scale:.2e})")
    ok = dmax < 1e-5 * scale and emax < 1e-4 * scale
    print("IQ4_NL 融合点积 + 归约：", "一致 ✔" if ok else "不一致 ✘")
    return 0 if ok else 1


def _synth_blocks(qtype: str, block_bytes: int, blocks: int) -> bytes:
    """索引里没有该编码时（例如 Q4_0）的合成权重块。

    随机字节的 f16 尺度可能是 NaN/Inf，所以把 d / dmin 写成正常值。
    """
    rng = np.random.default_rng(17)
    raw = bytearray(rng.integers(0, 256, size=blocks * block_bytes,
                                 dtype=np.uint8).tobytes())
    if qtype in ("Q4_0", "Q8_0"):
        for i in range(blocks):
            off = i * block_bytes
            raw[off:off + 2] = np.float16(0.01).tobytes()
    elif qtype in ("Q4_K", "Q5_K"):
        for i in range(blocks):
            off = i * block_bytes
            raw[off:off + 2] = np.float16(0.01).tobytes()
            raw[off + 2:off + 4] = np.float16(0.005).tobytes()
    return bytes(raw)


def _load_blocks(qtype: str, block_bytes: int, blocks: int,
                 prefer: str | None = None,
                 allow_synth: bool = False) -> tuple[bytes, int]:
    from gguf_sample import load_index
    idx = load_index()
    tensors = [t for t in idx["tensors"]
               if t["type"] == qtype and t["bytes"] >= blocks * block_bytes]
    if not tensors:
        if allow_synth:
            return _synth_blocks(qtype, block_bytes, blocks), 0
        raise SystemExit(f"索引里找不到足够大的 {qtype} 张量")
    tensors.sort(key=lambda t: t["bytes"])
    t = next((x for x in tensors if x["name"] == prefer), tensors[0])
    src = pathlib.Path(idx["sources"][t["source"]]["path"])
    with open(src, "rb") as f:
        f.seek(t["offset"])
        raw = f.read(blocks * block_bytes)
    if len(raw) != blocks * block_bytes or max(raw) == 0:
        raise SystemExit("权重读取失败（没下完？）")
    return raw, int(src.stat().st_size)


def selftest_iq3xxs(nbpr: int = 10, rows_per_exp: int = 24) -> int:
    """IQ3_XXS：256 权重 / 98 字节块，需要 256×u32 网格表 + 128 B 符号表。"""
    import struct
    from iq_dequant import dequant_iq3_xxs
    from iq_tables import iq3xxs_grid, ksigns_iq2xs

    n_exp = 2
    total_rows = rows_per_exp * n_exp
    nblocks = total_rows * nbpr
    raw, _ = _load_blocks("IQ3_XXS", 98, nblocks, "blk.0.ffn_gate_exps.weight")
    w_raw = np.frombuffer(raw, dtype=np.uint8).reshape(nblocks, 98)
    k = nbpr * 256
    x = np.random.default_rng(7).standard_normal((n_exp, k)).astype(np.float32)
    # 参考：用 iq_dequant 解出 [nblocks*256]，按块切成 [nbpr,256] 与 x 的块做点积
    wv = dequant_iq3_xxs(raw).reshape(nblocks, 256).astype(np.float32)
    xb = np.ascontiguousarray(x).reshape(-1, 256)
    ref = np.empty(nblocks, dtype=np.float32)
    for g in range(nblocks):
        q1, b = divmod(g, nbpr)
        e = q1 // rows_per_exp
        ref[g] = float(np.dot(wv[g], xb[e * nbpr + b]))
    ref_y = ref.reshape(total_rows, nbpr).sum(axis=1)

    m_nbpr = div_magic(nbpr, nblocks)
    m_rpe = div_magic(rows_per_exp, nblocks // nbpr + 1)
    grid = struct.pack("<%dI" % len(iq3xxs_grid), *[int(v) for v in iq3xxs_grid])
    ks = bytes(int(v) for v in ksigns_iq2xs)
    hsaco = build_one(IQ3XXS_NAME, gen_iq3xxs_dot_asm(), IQ3XXS_ARGS,
                      kernarg_size=IQ3XXS_KERNARG, tag="qdot")
    out = run_one(hsaco, IQ3XXS_NAME,
                  [{"buffer": "w"}, {"buffer": "x"}, {"buffer": "p"},
                   {"scalar": {"dtype": "u32", "value": nblocks}},
                   {"scalar": {"dtype": "u32", "value": 64}},
                   {"scalar": {"dtype": "u32", "value": nbpr}},
                   {"scalar": {"dtype": "u32", "value": m_nbpr}},
                   {"scalar": {"dtype": "u32", "value": rows_per_exp}},
                   {"scalar": {"dtype": "u32", "value": m_rpe}},
                   {"buffer": "g"}, {"buffer": "ks"},
                   {"buffer": "ids"},
                   {"scalar": {"dtype": "u32",
                               "value": rows_per_exp * nbpr * 98}},
                   {"scalar": {"dtype": "u32", "value": rows_per_exp}},
                   {"scalar": {"dtype": "u32",
                               "value": div_magic(rows_per_exp,
                                                  nblocks // nbpr + 1)}}],
                  {"w": {"dtype": "u8", "values": list(raw)},
                   "x": {"dtype": "f32", "values": x.reshape(-1).tolist()},
                   "p": {"dtype": "f32", "values": [0.0] * nblocks},
                   "ids": {"dtype": "u32", "values": list(range(n_exp))},
                   "g": {"dtype": "u8", "values": list(grid)},
                   "ks": {"dtype": "u8", "values": list(ks)}},
                  grid=nblocks, workgroup=64)
    got = np.array(out["p"], dtype=np.float32)
    dmax = float(np.abs(got - ref).max())
    scale = max(1.0, float(np.abs(ref).max()))
    print(f"iq3xxs_dot: nblocks={nblocks} nbpr={nbpr} partial max_abs={dmax:.3e}")
    ok = dmax < 1e-4 * scale
    print("IQ3_XXS 融合点积：", "一致 ✔" if ok else "不一致 ✘")
    if not ok:
        for i in np.argsort(-np.abs(got - ref))[:4]:
            print(f"  p[{i}] gpu={got[i]:.6g} ref={ref[i]:.6g}")
    return 0 if ok else 1


def _selftest_grid(qtype: str, kernel: str, gen, args, kernarg: int, block_bytes: int,
                   decode, table_bytes: bytes, nbpr: int = 10,
                   rows_per_exp: int = 24, prefer: str | None = None) -> int:
    """IQ2_S / IQ3_S 这类「单张网格表」的融合点积通用自检。"""
    n_exp = 2
    total_rows = rows_per_exp * n_exp
    nblocks = total_rows * nbpr
    raw, _ = _load_blocks(qtype, block_bytes, nblocks, prefer)
    k = nbpr * 256
    x = np.random.default_rng(11).standard_normal((n_exp, k)).astype(np.float32)
    wv = decode(raw).reshape(nblocks, 256).astype(np.float32)
    xb = np.ascontiguousarray(x).reshape(-1, 256)
    ref = np.empty(nblocks, dtype=np.float32)
    for g in range(nblocks):
        q1, b = divmod(g, nbpr)
        e = q1 // rows_per_exp
        ref[g] = float(np.dot(wv[g], xb[e * nbpr + b]))
    m1 = div_magic(nbpr, nblocks)
    m2 = div_magic(rows_per_exp, nblocks // nbpr + 1)
    hsaco = build_one(kernel, gen(), args, kernarg_size=kernarg, tag="qdot")
    out = run_one(hsaco, kernel,
                  [{"buffer": "w"}, {"buffer": "x"}, {"buffer": "p"},
                   {"scalar": {"dtype": "u32", "value": nblocks}},
                   {"scalar": {"dtype": "u32", "value": 64}},
                   {"scalar": {"dtype": "u32", "value": nbpr}},
                   {"scalar": {"dtype": "u32", "value": m1}},
                   {"scalar": {"dtype": "u32", "value": rows_per_exp}},
                   {"scalar": {"dtype": "u32", "value": m2}},
                   {"buffer": "g"}, {"buffer": "ids"},
                   {"scalar": {"dtype": "u32",
                               "value": rows_per_exp * nbpr * block_bytes}},
                   {"scalar": {"dtype": "u32", "value": rows_per_exp}},
                   {"scalar": {"dtype": "u32",
                               "value": div_magic(rows_per_exp,
                                                  nblocks // nbpr + 1)}}],
                  {"w": {"dtype": "u8", "values": list(raw)},
                   "x": {"dtype": "f32", "values": x.reshape(-1).tolist()},
                   "p": {"dtype": "f32", "values": [0.0] * nblocks},
                   "ids": {"dtype": "u32", "values": list(range(n_exp))},
                   "g": {"dtype": "u8", "values": list(table_bytes)}},
                  grid=nblocks, workgroup=64)
    got = np.array(out["p"], dtype=np.float32)
    dmax = float(np.abs(got - ref).max())
    scale = max(1.0, float(np.abs(ref).max()))
    print(f"{kernel}: nblocks={nblocks} nbpr={nbpr} max_abs={dmax:.3e}")
    ok = dmax < 1e-4 * scale
    print(f"{qtype} 融合点积：", "一致 ✔" if ok else "不一致 ✘")
    if not ok:
        for i in np.argsort(-np.abs(got - ref))[:4]:
            print(f"  p[{i}] gpu={got[i]:.6g} ref={ref[i]:.6g}")
    return 0 if ok else 1


def selftest_iq2s() -> int:
    import struct
    from iq_dequant import dequant_iq2_s
    from iq_tables import iq2s_grid
    grid = struct.pack("<%dQ" % len(iq2s_grid), *[int(v) for v in iq2s_grid])
    return _selftest_grid("IQ2_S", IQ2S_NAME, gen_iq2s_dot_asm, IQ2S_ARGS,
                          IQ2S_KERNARG, 82, dequant_iq2_s, grid,
                          prefer="blk.1.ffn_gate_exps.weight")


def selftest_iq3s() -> int:
    import struct
    from iq_dequant import dequant_iq3_s
    from iq_tables import iq3s_grid
    grid = struct.pack("<%dI" % len(iq3s_grid), *[int(v) for v in iq3s_grid])
    return _selftest_grid("IQ3_S", IQ3S_NAME, gen_iq3s_dot_asm, IQ3S_ARGS,
                          IQ3S_KERNARG, 110, dequant_iq3_s, grid,
                          prefer="blk.2.ffn_gate_exps.weight")


def _selftest_notable(qtype: str, kernel: str, gen, args, kernarg: int,
                      block_bytes: int, blk_elems: int, decode,
                      nbpr: int = 10, rows_per_exp: int = 24,
                      prefer: str | None = None,
                      allow_synth: bool = False) -> int:
    n_exp = 2
    total_rows = rows_per_exp * n_exp
    nblocks = total_rows * nbpr
    raw, _ = _load_blocks(qtype, block_bytes, nblocks, prefer, allow_synth)
    k = nbpr * blk_elems
    x = np.random.default_rng(13).standard_normal((n_exp, k)).astype(np.float32)
    wv = decode(raw).reshape(nblocks, blk_elems).astype(np.float32)
    xb = np.ascontiguousarray(x).reshape(-1, blk_elems)
    ref = np.empty(nblocks, dtype=np.float32)
    for g in range(nblocks):
        q1, b = divmod(g, nbpr)
        e = q1 // rows_per_exp
        ref[g] = float(np.dot(wv[g], xb[e * nbpr + b]))
    m1 = div_magic(nbpr, nblocks)
    m2 = div_magic(rows_per_exp, nblocks // nbpr + 1)
    hsaco = build_one(kernel, gen(), args, kernarg_size=kernarg, tag="qdot")
    out = run_one(hsaco, kernel,
                  [{"buffer": "w"}, {"buffer": "x"}, {"buffer": "p"},
                   {"scalar": {"dtype": "u32", "value": nblocks}},
                   {"scalar": {"dtype": "u32", "value": 64}},
                   {"scalar": {"dtype": "u32", "value": nbpr}},
                   {"scalar": {"dtype": "u32", "value": m1}},
                   {"scalar": {"dtype": "u32", "value": rows_per_exp}},
                   {"scalar": {"dtype": "u32", "value": m2}},
                   {"buffer": "ids"},
                   {"scalar": {"dtype": "u32",
                               "value": rows_per_exp * nbpr * block_bytes}},
                   {"scalar": {"dtype": "u32", "value": rows_per_exp}},
                   {"scalar": {"dtype": "u32",
                               "value": div_magic(rows_per_exp,
                                                  nblocks // nbpr + 1)}}],
                  {"w": {"dtype": "u8", "values": list(raw)},
                   "x": {"dtype": "f32", "values": x.reshape(-1).tolist()},
                   "p": {"dtype": "f32", "values": [0.0] * nblocks},
                   "ids": {"dtype": "u32", "values": list(range(n_exp))}},
                  grid=nblocks, workgroup=64)
    got = np.array(out["p"], dtype=np.float32)
    dmax = float(np.abs(got - ref).max())
    scale = max(1.0, float(np.abs(ref).max()))
    print(f"{kernel}: nblocks={nblocks} nbpr={nbpr} max_abs={dmax:.3e}")
    ok = dmax < 1e-4 * scale
    print(f"{qtype} 融合点积：", "一致 ✔" if ok else "不一致 ✘")
    return 0 if ok else 1


def selftest_q2_0() -> int:
    from iq_dequant import dequant_q2_0
    return _selftest_notable("Q2_0", Q20_NAME, gen_q2_0_dot_asm, Q20_ARGS,
                             Q20_KERNARG, 18, 64, dequant_q2_0, nbpr=40,
                             prefer="blk.0.ffn_down_exps.weight")


def selftest_iq4xs() -> int:
    from iq_dequant import dequant_iq4_xs
    return _selftest_notable("IQ4_XS", IQ4XS_NAME, gen_iq4xs_dot_asm, IQ4XS_ARGS,
                             IQ4XS_KERNARG, 136, 256, dequant_iq4_xs,
                             prefer="blk.0.attn_q.weight")


def selftest_q6k() -> int:
    from iq_dequant import dequant_q6_k
    return _selftest_notable("Q6_K", Q6K_NAME, gen_q6k_dot_asm, Q6K_ARGS,
                             Q6K_KERNARG, 210, 256, dequant_q6_k,
                             prefer="blk.3.attn_q.weight")


def selftest_q4k() -> int:
    from iq_dequant import dequant_q4_k_fast
    return _selftest_notable("Q4_K", Q4K_NAME, gen_q4k_dot_asm, Q4K_ARGS,
                             Q4K_KERNARG, 144, 256, dequant_q4_k_fast,
                             prefer="blk.0.ffn_gate_shexp.weight")


def selftest_q5k() -> int:
    from iq_dequant import dequant_q5_k_fast
    return _selftest_notable("Q5_K", Q5K_NAME, gen_q5k_dot_asm, Q5K_ARGS,
                             Q5K_KERNARG, 176, 256, dequant_q5_k_fast,
                             prefer="blk.4.ffn_up_shexp.weight")


def selftest_q8_0() -> int:
    from iq_dequant import dequant_q8_0_fast
    return _selftest_notable("Q8_0", Q80_NAME, gen_q8_0_dot_asm, Q80_ARGS,
                             Q80_KERNARG, 34, 32, dequant_q8_0_fast,
                             prefer="blk.47.ffn_down_shexp.weight")


def selftest_q4_0() -> int:
    from iq_dequant import dequant_q4_0_fast
    return _selftest_notable("Q4_0", Q40_NAME, gen_q4_0_dot_asm, Q40_ARGS,
                             Q40_KERNARG, 18, 32, dequant_q4_0_fast,
                             allow_synth=True)


def main() -> int:
    rc = selftest_iq4nl()
    rc |= selftest_iq3xxs()
    rc |= selftest_iq2s()
    rc |= selftest_iq3s()
    rc |= selftest_q2_0()
    rc |= selftest_iq4xs()
    rc |= selftest_q6k()
    rc |= selftest_q4k()
    rc |= selftest_q5k()
    rc |= selftest_q8_0()
    rc |= selftest_q4_0()
    return rc


# 供 build_native_kernels.py 收集（name, gen_asm, args, kernarg_size）
KERNELS = [
    (IQ4NL_NAME, gen_iq4nl_dot_asm, IQ4NL_ARGS, IQ4NL_KERNARG),
    (IQ3XXS_NAME, gen_iq3xxs_dot_asm, IQ3XXS_ARGS, IQ3XXS_KERNARG),
    (IQ2S_NAME, gen_iq2s_dot_asm, IQ2S_ARGS, IQ2S_KERNARG),
    (IQ3S_NAME, gen_iq3s_dot_asm, IQ3S_ARGS, IQ3S_KERNARG),
    (Q20_NAME, gen_q2_0_dot_asm, Q20_ARGS, Q20_KERNARG),
    (IQ4XS_NAME, gen_iq4xs_dot_asm, IQ4XS_ARGS, IQ4XS_KERNARG),
    (Q6K_NAME, gen_q6k_dot_asm, Q6K_ARGS, Q6K_KERNARG),
    (Q4K_NAME, gen_q4k_dot_asm, Q4K_ARGS, Q4K_KERNARG),
    (Q5K_NAME, gen_q5k_dot_asm, Q5K_ARGS, Q5K_KERNARG),
    (Q80_NAME, gen_q8_0_dot_asm, Q80_ARGS, Q80_KERNARG),
    (Q40_NAME, gen_q4_0_dot_asm, Q40_ARGS, Q40_KERNARG),
    (REDUCE_NAME, gen_reduce_asm, REDUCE_ARGS, REDUCE_KERNARG),
]


if __name__ == "__main__":
    raise SystemExit(main())
