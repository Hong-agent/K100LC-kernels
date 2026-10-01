#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""融合的**多头解码注意力**：`flash_dec_part_k` + `flash_dec_comb_k`。

    python3 tools/gen_flash_decode.py        # 自检（含 combine）

为什么要有它：解码注意力（M=1）现在按「每个头」走 K·q GEMV + 分块 softmax +
Vt·P + 归一化，8 个头就是 50~60 次 launch；这台机器一次 launch 固定开销约
11 us（见 docs/MODEL_RUNTIME.md），启动次数本身就是主要代价。这两个内核把
「所有头」的整条链路压成 **2 次 launch**：

    flash_dec_part_k  grid = 头数 × 分块数：每个 workgroup 算一段 KV 的
                      (m, l, Σ p·v) 部分和（两趟扫描）
    flash_dec_comb_k  grid = 头数：按 exp(m_s - M) 把各分块合起来、除以 Σ

布局（按头连续，一个 launch 覆盖所有头）：

    q  [n_heads, dh]           kt [n_heads, dh, max_len]   （K **转置**）
    v  [n_heads, max_len, dh]  po [n_heads, nsplit, dh]
    pm / pl [n_heads, nsplit]  out [n_heads, dh]

K 用转置布局是为访存：算分数时 lane j 负责一段里的第 j 行，
`kt_h[d*max_len + base + j]` 一条指令就是 64 个连续地址（完全合并）；V 用行
主序是为了算输出：lane d 负责一个输出维，扫 j 时 `v_h[(base+j)*dh + d]` 同样
连续。

数值：段内 `p_j = exp((s_j - m_s)·inv)`、`o_s = Σ p_j v_j`、`l_s = Σ p_j`；
combine 里 `M = max_s m_s`、`w_s = exp((m_s - M)·inv)`，
`out = Σ_s w_s·o_s / Σ_s w_s·l_s`。整段都被 mask 掉（pad > n_kv 时可能出现）
时把 m 夹到 0，于是 p / l / o 全 0、权重也是 0。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

PART_NAME = "flash_dec_part_k"
PART_ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16), buffer_arg(24),
             buffer_arg(32), buffer_arg(40),
             scalar_arg(48, 4), scalar_arg(52, 4), scalar_arg(56, 4),
             scalar_arg(60, 4), scalar_arg(64, 4), scalar_arg(68, 4),
             scalar_arg(72, 4)]
PART_KERNARG = 80

COMB_NAME = "flash_dec_comb_k"
COMB_ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16), buffer_arg(24),
             scalar_arg(32, 4), scalar_arg(36, 4), scalar_arg(40, 4)]
COMB_KERNARG = 48


def _tree(L: list[str], op: str, dst: str, slot: str, save: int) -> None:
    """64 lane 的 LDS 树形归约（结果留在 lane 0 的 `dst` 里）。"""
    for off in (32, 16, 8, 4, 2, 1):
        L += [
            f"v_cmp_gt_u32_e32 vcc, {off}, v0",
            f"s_and_saveexec_b64 s[{save}:{save + 1}], vcc",
            f"v_mov_b32_e32 v62, {off * 4}",
            f"v_add_u32_e32 v62, {slot}, v62",
            "ds_read_b32 v63, v62",
            "s_waitcnt lgkmcnt(0)",
            f"{op} {dst}, {dst}, v63",
            f"ds_write_b32 {slot}, {dst}",
            f"s_or_b64 exec, exec, s[{save}:{save + 1}]",
            "s_barrier",
        ]


def gen_part_asm() -> str:
    """`flash_dec_part_k(po, pm, pl, q, kt, v, n_kv, dh, pad, max_len, cshift, inv)`。

    grid = `n_heads << cshift`；`head = blk >> cshift`、
    `split = blk & ((1<<cshift)-1)`、`R = pad >> cshift`（每段行数，64 的倍数）。
    LDS：`R` 个 f32 分数（第二趟就地改成 p）+ 归约槽（`R*4` 起，64 lane × 4 B）。

    v1.9.8：第一趟（d 循环）和第三趟（j 循环）的批量宽度从 4 放宽到 8——每个
    lane 一次发出的在途 load 翻倍，`s_waitcnt vmcnt(0)` 的条数减半。实测这两条
    等待各占整核的 ~10%（见 CHANGELOG 1.9.7 的分相账）。
    """
    dbatch = 8                     # 第一趟：一次发 dbatch 个 d 的 kt/q
    jbatch = 8                     # 第四趟：一次发 jbatch 个 j 的 p/v
    KT0 = 44                       # kt / v 的载入目标基址
    QT0 = KT0 + dbatch             # q 的载入目标基址（= 48，和原来的 4 宽一致）
    DT = KT0 + 2 * dbatch          # d 的临时寄存器（= 52，原来的 v52）

    def d_elt(k: int) -> list[str]:
        """第一趟里第 k 个 d 的 (kt 读, q 读)。k=0 时顺带把 d 放进 v{DT}。"""
        if k == 0:
            pre_kt = [f"v_mov_b32_e32 v{DT}, s37"]
            mul = f"v_mul_lo_u32 v22, v{DT}, s31"
            pre_q = [f"v_lshlrev_b32_e32 v26, 2, v{DT}"]
        else:
            pre_kt = [f"v_add_u32_e32 v22, {k}, v{DT}"]
            mul = "v_mul_lo_u32 v22, v22, s31"
            pre_q = [f"v_add_u32_e32 v26, {k}, v{DT}",
                     "v_lshlrev_b32_e32 v26, 2, v26"]
        return pre_kt + [
            mul,
            "v_add_u32_e32 v22, v22, v21",
            "v_lshlrev_b32_e32 v22, 2, v22",
            "v_mov_b32_e32 v23, v8",
            "v_mov_b32_e32 v24, v9",
            "v_add_co_u32_e32 v23, vcc, v23, v22",
            "v_addc_co_u32_e32 v24, vcc, v24, v1, vcc",
            f"global_load_dword v{KT0 + k}, v[23:24], off",
            *pre_q,
            "v_mov_b32_e32 v27, v12",
            "v_mov_b32_e32 v28, v13",
            "v_add_co_u32_e32 v27, vcc, v27, v26",
            "v_addc_co_u32_e32 v28, vcc, v28, v1, vcc",
            f"global_load_dword v{QT0 + k}, v[27:28], off",
        ]

    def j_elt(k: int) -> list[str]:
        """第四趟里第 k 个 j 的 (LDS 读 p, global 读 v)。"""
        head = (["v_mov_b32_e32 v42, s37"] if k == 0 else
                ["v_mov_b32_e32 v52, s37", f"v_add_u32_e32 v42, {k}, v52"])
        return head + [
            "v_lshlrev_b32_e32 v24, 2, v42",
            "ds_read_b32 v53, v24",
            "v_add_u32_e32 v25, v5, v42",
            "v_mul_lo_u32 v25, v25, s39",
            "v_add_u32_e32 v25, v25, v22",
            "v_lshlrev_b32_e32 v25, 2, v25",
            "v_mov_b32_e32 v26, v10",
            "v_mov_b32_e32 v27, v11",
            "v_add_co_u32_e32 v26, vcc, v26, v25",
            "v_addc_co_u32_e32 v27, vcc, v27, v1, vcc",
            f"global_load_dword v{KT0 + k}, v[26:27], off",
            f"ds_read_b32 v{54 + k}, v24",
        ]

    L: list[str] = [
        ".text", f"k_{PART_NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",     # po
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",     # pm
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",    # pl
        "s_load_dwordx2 s[22:23], s[4:5], 0x18",    # q
        "s_load_dwordx2 s[24:25], s[4:5], 0x20",    # kt
        "s_load_dwordx2 s[26:27], s[4:5], 0x28",    # v
        "s_load_dword s28, s[4:5], 0x30",           # n_kv
        "s_load_dword s29, s[4:5], 0x34",           # dh
        "s_load_dword s30, s[4:5], 0x38",           # pad
        "s_load_dword s31, s[4:5], 0x3c",           # max_len
        "s_load_dword s32, s[4:5], 0x40",           # cshift
        "s_load_dword s33, s[4:5], 0x44",           # inv
        "s_load_dword s39, s[4:5], 0x48",           # vstride（V 的行距，元素）
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v1, 0",
        "v_mov_b32_e32 v2, s6",                     # blk
        "v_lshrrev_b32_e32 v3, s32, v2",            # head
        "v_mov_b32_e32 v4, 1",
        "v_lshlrev_b32_e32 v4, s32, v4",
        "v_add_u32_e32 v4, -1, v4",
        "v_and_b32_e32 v4, v4, v2",                 # split
        "s_lshr_b32 s34, s30, s32",                 # R（每段行数）
        "s_lshr_b32 s38, s34, 6",                   # R/64：行块数（每个 c 管 64 行）
        "v_mov_b32_e32 v34, s34",                   # R（VGPR 版）
        "v_mov_b32_e32 v35, s29",                   # dh（VGPR 版）
        "v_mov_b32_e32 v36, s31",                   # max_len（VGPR 版）
        "v_mul_lo_u32 v5, s34, v4",                 # base = split*R
        # kt_h = kt + head*dh*max_len*4
        "v_mul_lo_u32 v7, v3, s29",
        "v_mul_lo_u32 v7, v7, s31",
        "v_lshlrev_b32_e32 v7, 2, v7",
        "v_mov_b32_e32 v8, s24",
        "v_mov_b32_e32 v9, s25",
        "v_add_co_u32_e32 v8, vcc, v8, v7",
        "v_addc_co_u32_e32 v9, vcc, v9, v1, vcc",
        # v_h = v + head*dh*4（V 里每个头占相连的 dh 列）
        "v_mul_lo_u32 v7, v3, s29",
        "v_lshlrev_b32_e32 v7, 2, v7",
        "v_mov_b32_e32 v10, s26",
        "v_mov_b32_e32 v11, s27",
        "v_add_co_u32_e32 v10, vcc, v10, v7",
        "v_addc_co_u32_e32 v11, vcc, v11, v1, vcc",
        # q_h = q + head*dh*4
        "v_mul_lo_u32 v7, v3, s29",
        "v_lshlrev_b32_e32 v7, 2, v7",
        "v_mov_b32_e32 v12, s22",
        "v_mov_b32_e32 v13, s23",
        "v_add_co_u32_e32 v12, vcc, v12, v7",
        "v_addc_co_u32_e32 v13, vcc, v13, v1, vcc",
        # po 段 = po + (head*nsplit + split)*dh*4
        "v_shl_add:",
        "v_lshlrev_b32_e32 v15, s32, v3",
        "v_add_u32_e32 v15, v15, v4",
        "v_mul_lo_u32 v15, v15, s29",
        "v_lshlrev_b32_e32 v15, 2, v15",
        "v_mov_b32_e32 v16, s16",
        "v_mov_b32_e32 v17, s17",
        "v_add_co_u32_e32 v16, vcc, v16, v15",
        "v_addc_co_u32_e32 v17, vcc, v17, v1, vcc",
        "v_lshlrev_b32_e32 v18, 2, v0",             # lane*4
        "v_lshlrev_b32_e32 v41, 2, v34",            # 归约槽基址 R*4（广播用）
        "v_add_u32_e32 v19, v41, v18",              # 本 lane 的槽 = R*4 + lane*4

        # 整段都被 mask（base >= n_kv）→ 写 pm=-inf / pl=0 / po=0 直接退出：
        # 解码早期 nsplit 里绝大多数段是空的，原来每段还要把 R 行扫两遍
        "s_mov_b32 s42, 1",
        "s_lshl_b32 s42, s42, s32",
        "s_mov_b32 s43, 1",
        "s_sub_u32 s43, s42, s43",                  # nsplit-1
        "s_and_b32 s44, s6, s43",                   # split
        "s_mul_i32 s45, s44, s34",                  # base
        "s_cmp_lt_u32 s45, s28",                    # base < n_kv ？
        "s_cbranch_scc0 L_masked",
        # 本段真正要扫的行数：`min(R, n_kv - base)`（base < n_kv 已由上面保证）。
        # 解码早期 n_kv < R，原来每段都要把 R 行扫两遍（63/64 是 mask 掉的），
        # 实测 n_kv=1 时 flash_dec_part_k 要 30 us，几乎全是白扫。
        "s_sub_u32 s46, s28, s45",                  # n_kv - base
        "s_cmp_lt_u32 s46, s34",
        "s_cbranch_scc1 L_reff_ok",
        "s_mov_b32 s46, s34",                       # 超过 R 就按 R 算
        "L_reff_ok:",
        "s_add_u32 s47, s46, 63",                   # 行块数 = ceil(R_eff / 64)
        "s_lshr_b32 s47, s47, 6",
        # ---------------- 第一趟：分数 ----------------
        "s_mov_b32 s36, 0",                         # c
        "L_c1:", "s_cmp_lt_u32 s36, s47",
        "s_cbranch_scc0 L_c1_done",
        "v_mov_b32_e32 v20, 0",                     # acc
        "s_mov_b32 s37, 0",                         # d
        "v_mov_b32_e32 v21, s36",
        "v_lshlrev_b32_e32 v21, 6, v21",
        "v_add_u32_e32 v21, v21, v5",
        "v_add_u32_e32 v21, v21, v0",               # row = base + c*64 + lane
        # ---- 一次发 4 个 d 的 load（k 与 q 各 4 条），只等一次 ----
        # 不批量的话每个 d 都要一趟完整访存往返：4 个 workgroup 的小 case 实测
        # 181 us（128 个 d × 0.5 us 的纯延迟）。
        "L_d4:",
        f"s_add_i32 s41, s37, {dbatch}",
        "s_cmp_lt_u32 s41, s29",
        "s_cbranch_scc0 L_d1",
        *[x for k in range(dbatch) for x in d_elt(k)],
        "s_waitcnt vmcnt(0)",
        *[f"v_fma_f32 v20, v{KT0 + k}, v{QT0 + k}, v20" for k in range(dbatch)],
        f"s_add_i32 s37, s37, {dbatch}",
        "s_branch L_d4",
        # ---- 收尾：剩下不到 4 个的单拍循环 ----
        "L_d1:", "s_cmp_lt_u32 s37, s29",
        "s_cbranch_scc0 L_d1_done",
        "v_mov_b32_e32 v40, s37",
        "v_mul_lo_u32 v22, v40, s31",
        "v_add_u32_e32 v22, v22, v21",
        "v_lshlrev_b32_e32 v22, 2, v22",
        "v_mov_b32_e32 v23, v8",
        "v_mov_b32_e32 v24, v9",
        "v_add_co_u32_e32 v23, vcc, v23, v22",
        "v_addc_co_u32_e32 v24, vcc, v24, v1, vcc",
        "global_load_dword v25, v[23:24], off",
        "v_lshlrev_b32_e32 v26, 2, v40",
        "v_mov_b32_e32 v27, v12",
        "v_mov_b32_e32 v28, v13",
        "v_add_co_u32_e32 v27, vcc, v27, v26",
        "v_addc_co_u32_e32 v28, vcc, v28, v1, vcc",
        "global_load_dword v29, v[27:28], off",
        "s_waitcnt vmcnt(0)",
        "v_fma_f32 v20, v25, v29, v20",
        "s_add_i32 s37, s37, 1",
        "s_branch L_d1",
        "L_d1_done:",
        "v_mov_b32_e32 v37, s28",                   # n_kv（VGPR 版）
        "v_cmp_lt_u32_e64 vcc, v21, v37",           # row < n_kv ？
        "v_mov_b32_e32 v31, 0xff800000",
        # 本机语义：vcc 为真时取**第二个**源（见 gen_gemv_w4a4_r2.py 的注释）
        "v_cndmask_b32_e32 v20, v31, v20, vcc",     # row < n_kv → 保留 acc
        "v_mov_b32_e32 v32, s36",
        "v_lshlrev_b32_e32 v32, 6, v32",
        "v_add_u32_e32 v32, v32, v0",
        "v_lshlrev_b32_e32 v32, 2, v32",
        "ds_write_b32 v32, v20",
        "s_add_i32 s36, s36, 1",
        "s_branch L_c1",
        "L_c1_done:",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
        # ---------------- 第二趟：max ----------------
        "v_mov_b32_e32 v20, 0xff800000",
        "s_mov_b32 s36, 0",
        "L_c2:", "s_cmp_lt_u32 s36, s47",
        "s_cbranch_scc0 L_c2_done",
        "v_mov_b32_e32 v32, s36",
        "v_lshlrev_b32_e32 v32, 6, v32",
        "v_add_u32_e32 v32, v32, v0",
        "v_lshlrev_b32_e32 v32, 2, v32",
        "ds_read_b32 v30, v32",
        "s_waitcnt lgkmcnt(0)",
        "v_max_f32_e32 v20, v20, v30",
        "s_add_i32 s36, s36, 1",
        "s_branch L_c2",
        "L_c2_done:",
        "ds_write_b32 v19, v20",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
    ]
    _tree(L, "v_max_f32_e32", "v20", "v19", 2)
    L += [
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[4:5], vcc",
        "ds_write_b32 v41, v20",                    # 只有 lane 0 执行：写槽 0
        "s_or_b64 exec, exec, s[4:5]",
        "s_barrier",
        "ds_read_b32 v20, v41",                     # 所有 lane 读**同一个**槽
        "s_waitcnt lgkmcnt(0)",
        # 整段都被 mask：m 是 -inf → 夹到 0（p / l / o 全 0，combine 权重也是 0）
        "v_mov_b32_e32 v31, 0xff800000",
        "v_cmp_eq_f32_e32 vcc, v20, v31",
        "v_mov_b32_e32 v30, 0",
        "v_cndmask_b32_e32 v20, v20, v30, vcc",     # m == -inf → 夹到 0
        # ---------------- 第二趟：exp + 和 ----------------
        "v_mov_b32_e32 v21, 0",                     # l_local
        "v_mov_b32_e32 v22, 0x3fb8aa3b",            # log2(e)
        "v_mov_b32_e32 v23, s33",                   # inv
        "s_mov_b32 s36, 0",
        "L_c3:", "s_cmp_lt_u32 s36, s47",
        "s_cbranch_scc0 L_c3_done",
        "v_mov_b32_e32 v32, s36",
        "v_lshlrev_b32_e32 v32, 6, v32",
        "v_add_u32_e32 v32, v32, v0",
        "v_lshlrev_b32_e32 v32, 2, v32",
        "ds_read_b32 v30, v32",
        "s_waitcnt lgkmcnt(0)",
        "v_sub_f32_e32 v30, v30, v20",
        "v_mul_f32_e32 v30, v30, v23",
        "v_mul_f32_e32 v30, v30, v22",
        "s_nop 0",
        "v_exp_f32_e32 v30, v30",
        "s_nop 0",
        "ds_write_b32 v32, v30",                    # 就地改成 p
        "v_add_f32_e32 v21, v21, v30",
        "s_add_i32 s36, s36, 1",
        "s_branch L_c3",
        "L_c3_done:",
        "ds_write_b32 v19, v21",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
    ]
    _tree(L, "v_add_f32_e32", "v21", "v19", 2)
    L += [
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[4:5], vcc",
        "ds_write_b32 v41, v21",
        "s_or_b64 exec, exec, s[4:5]",
        "s_barrier",
        "ds_read_b32 v21, v41",                     # l（所有 lane 同值）
        "s_waitcnt lgkmcnt(0)",
        # lane 0 写 pm / pl：下标 = head*nsplit + split
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[4:5], vcc",
        "v_lshlrev_b32_e32 v30, s32, v3",
        "v_add_u32_e32 v30, v30, v4",
        "v_lshlrev_b32_e32 v31, 2, v30",
        "v_mov_b32_e32 v32, s18",
        "v_mov_b32_e32 v33, s19",
        "v_add_co_u32_e32 v32, vcc, v32, v31",
        "v_addc_co_u32_e32 v33, vcc, v33, v1, vcc",
        "global_store_dword v[32:33], v20, off",    # pm
        "v_mov_b32_e32 v32, s20",
        "v_mov_b32_e32 v33, s21",
        "v_add_co_u32_e32 v32, vcc, v32, v31",
        "v_addc_co_u32_e32 v33, vcc, v33, v1, vcc",
        "global_store_dword v[32:33], v21, off",    # pl
        "s_or_b64 exec, exec, s[4:5]",
        # ---------------- 第三趟：o = Σ p·v ----------------
        "s_mov_b32 s36, 0",                         # i（输出维块，步长 64）
        "L_i:", "s_cmp_lt_u32 s36, s29",
        "s_cbranch_scc0 L_i_done",
        "v_mov_b32_e32 v22, s36",
        "v_add_u32_e32 v22, v22, v0",               # d = i*64 + lane
        "v_cmp_lt_u32_e64 vcc, v22, v35",
        "s_and_saveexec_b64 s[6:7], vcc",
        "s_cbranch_execz L_i_next",
        "v_mov_b32_e32 v23, 0",                     # o
        "s_mov_b32 s37, 0",                         # j
        # 同样一次发 4 个 j（4 条 LDS 读 p + 4 条 global 读 v），只等一次
        "L_j4:",
        f"s_add_i32 s41, s37, {jbatch}",
        "s_cmp_lt_u32 s41, s46",
        "s_cbranch_scc0 L_j",
        *[x for k in range(jbatch) for x in j_elt(k)],
        "s_waitcnt vmcnt(0)",
        *[f"v_fma_f32 v23, v{KT0 + k}, v{54 + k}, v23" for k in range(jbatch)],
        f"s_add_i32 s37, s37, {jbatch}",
        "s_branch L_j4",
        "L_j:", "s_cmp_lt_u32 s37, s46",
        "s_cbranch_scc0 L_j_done",
        "v_mov_b32_e32 v42, s37",
        "v_lshlrev_b32_e32 v24, 2, v42",
        "ds_read_b32 v30, v24",
        "v_add_u32_e32 v25, v5, v42",
        "v_mul_lo_u32 v25, v25, s39",
        "v_add_u32_e32 v25, v25, v22",
        "v_lshlrev_b32_e32 v25, 2, v25",
        "v_mov_b32_e32 v26, v10",
        "v_mov_b32_e32 v27, v11",
        "v_add_co_u32_e32 v26, vcc, v26, v25",
        "v_addc_co_u32_e32 v27, vcc, v27, v1, vcc",
        "global_load_dword v31, v[26:27], off",
        "s_waitcnt vmcnt(0)",
        "v_fma_f32 v23, v30, v31, v23",
        "s_add_i32 s37, s37, 1",
        "s_branch L_j",
        "L_j_done:",
        "v_lshlrev_b32_e32 v25, 2, v22",
        "v_mov_b32_e32 v26, v16",
        "v_mov_b32_e32 v27, v17",
        "v_add_co_u32_e32 v26, vcc, v26, v25",
        "v_addc_co_u32_e32 v27, vcc, v27, v1, vcc",
        "global_store_dword v[26:27], v23, off",
        "L_i_next:",
        "s_or_b64 exec, exec, s[6:7]",
        "s_add_i32 s36, s36, 64",
        "s_branch L_i",
        "L_i_done:",
        # ---------------- 整段被 mask 的出口 ----------------
        "s_endpgm",
        "L_masked:",
        "v_mov_b32_e32 v20, 0xff800000",            # m = -inf
        "v_mov_b32_e32 v21, 0",                     # l = 0
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[4:5], vcc",
        "v_lshlrev_b32_e32 v30, s32, v3",
        "v_add_u32_e32 v30, v30, v4",               # head*nsplit + split
        "v_lshlrev_b32_e32 v31, 2, v30",
        "v_mov_b32_e32 v32, s18",
        "v_mov_b32_e32 v33, s19",
        "v_add_co_u32_e32 v32, vcc, v32, v31",
        "v_addc_co_u32_e32 v33, vcc, v33, v1, vcc",
        "global_store_dword v[32:33], v20, off",    # pm = -inf
        "v_mov_b32_e32 v32, s20",
        "v_mov_b32_e32 v33, s21",
        "v_add_co_u32_e32 v32, vcc, v32, v31",
        "v_addc_co_u32_e32 v33, vcc, v33, v1, vcc",
        "global_store_dword v[32:33], v21, off",    # pl = 0
        "s_or_b64 exec, exec, s[4:5]",
        "s_mov_b32 s36, 0",
        "L_mz:", "s_cmp_lt_u32 s36, s29",
        "s_cbranch_scc0 L_mz_done",
        "v_mov_b32_e32 v22, s36",
        "v_add_u32_e32 v22, v22, v0",               # d = i*64 + lane
        "v_cmp_lt_u32_e64 vcc, v22, v35",
        "s_and_saveexec_b64 s[6:7], vcc",
        "s_cbranch_execz L_mz_next",
        "v_lshlrev_b32_e32 v25, 2, v22",
        "v_mov_b32_e32 v26, v16",
        "v_mov_b32_e32 v27, v17",
        "v_add_co_u32_e32 v26, vcc, v26, v25",
        "v_addc_co_u32_e32 v27, vcc, v27, v1, vcc",
        "v_mov_b32_e32 v23, 0",
        "global_store_dword v[26:27], v23, off",    # po = 0
        "L_mz_next:",
        "s_or_b64 exec, exec, s[6:7]",
        "s_add_i32 s36, s36, 64",
        "s_branch L_mz",
        "L_mz_done:",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


COMB_UNROLL = 4          # `flash_dec_comb_k` 的 `L_s` 循环按几个分块一组展开


def gen_comb_asm(unroll: int = COMB_UNROLL) -> str:
    """`flash_dec_comb_k(out, po, pm, pl, dh, nsplit, inv)`：grid = n_heads。

    原来 `L_s` 循环每个分块单独发 3 条 load、单独等一次 `vmcnt(0)`：nsplit 个
    分块就是 nsplit 次访存往返，而往返之间还夹着「SALU 写 s31 → VALU 取 s31 →
    算地址 → load」的依赖链，延迟完全暴露（实测 nsplit=64 时整个内核里循环占
    74/85 us）。改成按 `unroll` 个分块一组：

    * 组内 3·unroll 条 load **全发出去**，只等一次 `vmcnt(0)`；
    * `pm/pl` 同一头下 nsplit 个分块是连续的 4 字节，组内 m_k / l_k 直接吃
      地址上的**立即数偏移**，一条地址算到底；
    * `po` 的行距 `dh*4` 运行时才知道，用寄存器累加。

    往返次数从 nsplit 降到 ceil(nsplit/unroll)。`nsplit` 不是 unroll 的倍数时
    尾部退回原来的单步路径，语义（含 `nsplit<unroll`）不变。
    """
    if unroll not in (1, 2, 4):
        raise ValueError("unroll 只能是 1/2/4")
    pool = [r for r in range(12, 32) if r != 20]      # v20 = d，保留

    def take(n: int) -> list[int]:
        return [pool.pop(0) for _ in range(n)]

    mr = take(unroll)                                 # m_s
    lr = take(unroll)                                 # l_s
    orr = take(unroll)                                # o_s[d]
    v_ib, v_off, v_tmp, v_a, v_b, v_w = take(6)

    def addr(slo: int, shi: int, voff: int) -> list[str]:
        return [f"v_mov_b32_e32 v{v_a}, s{slo}",
                f"v_mov_b32_e32 v{v_b}, s{shi}",
                f"v_add_co_u32_e32 v{v_a}, vcc, v{v_a}, v{voff}",
                f"v_addc_co_u32_e32 v{v_b}, vcc, v{v_b}, v1, vcc"]

    def weigh(k: int) -> list[str]:
        r = mr[k], lr[k], orr[k]
        return [f"v_sub_f32_e32 v{v_w}, v{r[0]}, v3",
                f"v_mul_f32_e32 v{v_w}, v{v_w}, v11",
                f"v_mul_f32_e32 v{v_w}, v{v_w}, v10",
                "s_nop 0",
                f"v_exp_f32_e32 v{v_w}, v{v_w}",
                "s_nop 0",
                f"v_fma_f32 v8, v{v_w}, v{r[2]}, v8",
                f"v_fma_f32 v9, v{v_w}, v{r[1]}, v9"]

    # 单步：idx → 三对地址 → 三条 load → 等一次
    one: list[str] = [
        f"v_mul_lo_u32 v{v_ib}, v2, s29",             # head*nsplit
        f"v_mov_b32_e32 v{v_tmp}, s31",               # s（VGPR 版）
        f"v_add_u32_e32 v{v_ib}, v{v_ib}, v{v_tmp}",
        f"v_lshlrev_b32_e32 v{v_off}, 2, v{v_ib}",    # pm/pl 字节偏移
        *addr(20, 21, v_off),
        f"global_load_dword v{mr[0]}, v[{v_a}:{v_b}], off",
        *addr(22, 23, v_off),
        f"global_load_dword v{lr[0]}, v[{v_a}:{v_b}], off",
        f"v_mul_lo_u32 v{v_off}, v{v_ib}, s28",       # v_off 用完，改存 po 偏移
        f"v_add_u32_e32 v{v_off}, v{v_off}, v20",
        f"v_lshlrev_b32_e32 v{v_off}, 2, v{v_off}",
        *addr(18, 19, v_off),
        f"global_load_dword v{orr[0]}, v[{v_a}:{v_b}], off",
        "s_waitcnt vmcnt(0)",
        *weigh(0),
        "s_add_i32 s31, s31, 1",
        "s_branch L_s",
    ]

    # 展开组：一次发 3·unroll 条 load、只等一次
    grp: list[str] = []
    if unroll > 1:
        grp += [
            f"v_mul_lo_u32 v{v_ib}, v2, s29",
            f"v_mov_b32_e32 v{v_tmp}, s31",
            f"v_add_u32_e32 v{v_ib}, v{v_ib}, v{v_tmp}",
            f"v_lshlrev_b32_e32 v{v_off}, 2, v{v_ib}",
            *addr(20, 21, v_off),
        ]
        grp += [f"global_load_dword v{mr[k]}, v[{v_a}:{v_b}], off "
                f"offset:{4 * k}" for k in range(unroll)]
        grp += addr(22, 23, v_off)
        grp += [f"global_load_dword v{lr[k]}, v[{v_a}:{v_b}], off "
                f"offset:{4 * k}" for k in range(unroll)]
        grp += [
            f"v_mul_lo_u32 v{v_off}, v{v_ib}, s28",   # v_off 用完，改存 po 偏移
            f"v_add_u32_e32 v{v_off}, v{v_off}, v20",
            f"v_lshlrev_b32_e32 v{v_off}, 2, v{v_off}",
            f"v_mov_b32_e32 v{v_tmp}, s28",           # v_tmp 用完，改存 dh*4 步长
            f"v_lshlrev_b32_e32 v{v_tmp}, 2, v{v_tmp}",
            *addr(18, 19, v_off),
        ]
        for k in range(unroll):
            if k:
                grp += [f"v_add_u32_e32 v{v_a}, v{v_a}, v{v_tmp}",
                        f"v_addc_co_u32_e32 v{v_b}, vcc, v{v_b}, v1, vcc"]
            grp += [f"global_load_dword v{orr[k]}, v[{v_a}:{v_b}], off"]
        grp += ["s_waitcnt vmcnt(0)"]
        for k in range(unroll):
            grp += weigh(k)
        grp += [f"s_add_i32 s31, s31, {unroll}", "s_branch L_s"]

    L: list[str] = [
        ".text", f"k_{COMB_NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",       # out
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",       # po
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",      # pm
        "s_load_dwordx2 s[22:23], s[4:5], 0x18",      # pl
        "s_load_dword s28, s[4:5], 0x20",             # dh
        "s_load_dword s29, s[4:5], 0x24",             # nsplit
        "s_load_dword s30, s[4:5], 0x28",             # inv
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v1, 0",
        "v_mov_b32_e32 v2, s6",                       # head
        "v_mov_b32_e32 v25, s28",                     # dh（VGPR 版）
        "v_cmp_lt_u32_e64 vcc, v0, v25",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        # M = max_s pm[head][s]：同样按 4 个分块一组发 load、只等一次
        "v_mul_lo_u32 v4, v2, s29",                   # head*nsplit（循环不变量）
        "v_lshlrev_b32_e32 v5, 2, v4",
        "v_mov_b32_e32 v6, s20",
        "v_mov_b32_e32 v7, s21",
        "v_add_co_u32_e32 v6, vcc, v6, v5",
        "v_addc_co_u32_e32 v7, vcc, v7, v1, vcc",
        "v_mov_b32_e32 v3, 0xff800000",
        "s_mov_b32 s31, 0",
        "L_m:", "s_cmp_lt_u32 s31, s29",
        "s_cbranch_scc0 L_m_done",
        "s_add_i32 s33, s31, 3",
        "s_cmp_lt_u32 s33, s29",
        "s_cbranch_scc0 L_m_one",
        "v_mov_b32_e32 v4, s31",
        "v_lshlrev_b32_e32 v4, 2, v4",
        "v_add_co_u32_e32 v8, vcc, v6, v4",
        "v_addc_co_u32_e32 v9, vcc, v7, v1, vcc",
        "global_load_dword v12, v[8:9], off offset:0",
        "global_load_dword v13, v[8:9], off offset:4",
        "global_load_dword v14, v[8:9], off offset:8",
        "global_load_dword v15, v[8:9], off offset:12",
        "s_waitcnt vmcnt(0)",
        "v_max_f32_e32 v3, v3, v12",
        "v_max_f32_e32 v3, v3, v13",
        "v_max_f32_e32 v3, v3, v14",
        "v_max_f32_e32 v3, v3, v15",
        "s_add_i32 s31, s31, 4",
        "s_branch L_m",
        "L_m_one:",
        "v_mov_b32_e32 v4, s31",
        "v_lshlrev_b32_e32 v4, 2, v4",
        "v_add_co_u32_e32 v8, vcc, v6, v4",
        "v_addc_co_u32_e32 v9, vcc, v7, v1, vcc",
        "global_load_dword v12, v[8:9], off",
        "s_waitcnt vmcnt(0)",
        "v_max_f32_e32 v3, v3, v12",
        "s_add_i32 s31, s31, 1",
        "s_branch L_m",
        "L_m_done:",
        "v_mov_b32_e32 v10, 0x3fb8aa3b",              # log2(e)
        "v_mov_b32_e32 v11, s30",                     # inv
        # 逐维块（dh > 64 时一个 workgroup 要跑多轮）：
        #   d = i + lane，num/den 归约各分块，最后 out[head][d] = num/den
        "s_mov_b32 s32, 0",                           # i
        "L_i:", "s_cmp_lt_u32 s32, s28",
        "s_cbranch_scc0 L_i_done",
        "v_mov_b32_e32 v20, s32",
        "v_add_u32_e32 v20, v20, v0",                 # d = i + lane
        "v_mov_b32_e32 v25, s28",                     # dh（VGPR 版）
        "v_cmp_lt_u32_e64 vcc, v20, v25",
        "s_and_saveexec_b64 s[6:7], vcc",
        "s_cbranch_execz L_i_next",
        "v_mov_b32_e32 v8, 0",                        # num
        "v_mov_b32_e32 v9, 0",                        # den
        "s_mov_b32 s31, 0",
        "L_s:", "s_cmp_lt_u32 s31, s29",
        "s_cbranch_scc0 L_s_done",
    ]
    if unroll > 1:
        L += [f"s_add_i32 s33, s31, {unroll - 1}",    # s+unroll-1 < nsplit ?
              "s_cmp_lt_u32 s33, s29",
              "s_cbranch_scc0 L_s_one"]
        L += grp
        L += ["L_s_one:"]
    L += one
    L += [
        "L_s_done:",
        "s_nop 0",
        "v_rcp_f32_e32 v19, v9",
        "s_nop 0",
        "v_mul_f32_e32 v21, v8, v19",
        "v_mul_lo_u32 v22, v2, s28",
        "v_add_u32_e32 v22, v22, v20",
        "v_lshlrev_b32_e32 v22, 2, v22",
        "v_mov_b32_e32 v23, s16",
        "v_mov_b32_e32 v24, s17",
        "v_add_co_u32_e32 v23, vcc, v23, v22",
        "v_addc_co_u32_e32 v24, vcc, v24, v1, vcc",
        "global_store_dword v[23:24], v21, off",
        "L_i_next:",
        "s_or_b64 exec, exec, s[6:7]",
        "s_add_i32 s32, s32, 64",
        "s_branch L_i",
        "L_i_done:",
        "L_end:",
        "s_or_b64 exec, exec, s[0:1]",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


def build_flash(tag: str = "fdec"):
    part = build_one(PART_NAME, gen_part_asm(), PART_ARGS,
                     kernarg_size=PART_KERNARG, group_segment=65536, tag=tag)
    comb = build_one(COMB_NAME, gen_comb_asm(), COMB_ARGS,
                     kernarg_size=COMB_KERNARG, tag=tag)
    return part, comb


def _ref(n_heads, dh, n_kv, q, kt, v, inv):
    """numpy 参考：逐头 `softmax(q·k/sqrt(dh))·v`。

    `kt [n_heads, dh, max_len]`、`v [max_len, n_heads*dh]`（每头占相连的 dh 列，
    和运行时 `FlashAttention` 的缓存布局一致）。
    """
    out = np.zeros((n_heads, dh), np.float32)
    for h in range(n_heads):
        s = (kt[h][:, :n_kv].T @ q[h]).astype(np.float32)
        p = np.exp((s - s.max()) * inv).astype(np.float32)
        out[h] = (p @ v[:n_kv, h * dh:(h + 1) * dh]) / p.sum()
    return out


def selftest() -> int:
    part, comb = build_flash()
    rng = np.random.default_rng(20261005)
    rc = 0
    # (n_heads, dh, n_kv, pad, cshift)；要求 `R = pad >> cshift` 是 64 的倍数
    # （内核按 64 行一块扫，R 不是 64 倍数会漏行）
    cases = [(1, 64, 64, 64, 0), (1, 128, 100, 256, 0),
             (2, 128, 300, 512, 3), (4, 64, 1000, 1024, 4),
             (3, 128, 5000, 5120, 4), (2, 128, 4096, 4096, 6),
             (2, 128, 8000, 8192, 6), (8, 64, 200, 256, 2)]
    for n_heads, dh, n_kv, pad, cshift in cases:
        nsplit = 1 << cshift
        assert (pad >> cshift) % 64 == 0, (pad, cshift)
        max_len = pad
        q = rng.standard_normal((n_heads, dh)).astype(np.float32)
        kt = rng.standard_normal((n_heads, dh, max_len)).astype(np.float32)
        v = rng.standard_normal((max_len, n_heads * dh)).astype(np.float32)
        inv = np.float32(1.0 / np.sqrt(dh))
        o = run_one(part, PART_NAME,
                    [{"buffer": "po"}, {"buffer": "pm"}, {"buffer": "pl"},
                     {"buffer": "q"}, {"buffer": "kt"}, {"buffer": "v"},
                     {"scalar": {"dtype": "u32", "value": n_kv}},
                     {"scalar": {"dtype": "u32", "value": dh}},
                     {"scalar": {"dtype": "u32", "value": pad}},
                     {"scalar": {"dtype": "u32", "value": max_len}},
                     {"scalar": {"dtype": "u32", "value": cshift}},
                     {"scalar": {"dtype": "f32", "value": float(inv)}},
                     {"scalar": {"dtype": "u32", "value": n_heads * dh}}],  # vstride
                    {"po": {"dtype": "f32", "values": [0.0] * (n_heads * nsplit * dh)},
                     "pm": {"dtype": "f32", "values": [0.0] * (n_heads * nsplit)},
                     "pl": {"dtype": "f32", "values": [0.0] * (n_heads * nsplit)},
                     "q": {"dtype": "f32", "values": q.reshape(-1).tolist()},
                     "kt": {"dtype": "f32", "values": kt.reshape(-1).tolist()},
                     "v": {"dtype": "f32", "values": v.reshape(-1).tolist()}},
                    grid=n_heads * nsplit * 64, workgroup=64)
        o2 = run_one(comb, COMB_NAME,
                     [{"buffer": "out"}, {"buffer": "po"}, {"buffer": "pm"},
                      {"buffer": "pl"},
                      {"scalar": {"dtype": "u32", "value": dh}},
                      {"scalar": {"dtype": "u32", "value": nsplit}},
                      {"scalar": {"dtype": "f32", "value": float(inv)}}],
                     {"out": {"dtype": "f32", "values": [0.0] * (n_heads * dh)},
                      "po": {"dtype": "f32", "values": o["po"]},
                      "pm": {"dtype": "f32", "values": o["pm"]},
                      "pl": {"dtype": "f32", "values": o["pl"]}},
                     grid=n_heads * 64, workgroup=64)
        got = np.array(o2["out"], np.float32).reshape(n_heads, dh)
        ref = _ref(n_heads, dh, n_kv, q, kt, v, inv)
        rel = float(np.abs(got - ref).max() / max(1e-9, float(np.abs(ref).max())))
        ok = rel < 1e-4
        rc |= 0 if ok else 1
        print(f"heads={n_heads} dh={dh} n_kv={n_kv} pad={pad} nsplit={nsplit}: "
              f"max_rel={rel:.2e} {'✔' if ok else '✘'}")
        if not ok:
            print("   got", got[0][:4], " ref", ref[0][:4])
    return rc


# `tools/build_native_kernels.py` 从这里取内核
KERNELS = [
    (PART_NAME, gen_part_asm, PART_ARGS, PART_KERNARG),
    (COMB_NAME, gen_comb_asm, COMB_ARGS, COMB_KERNARG),
]


def main() -> int:
    rc = selftest()
    print("flash decode（多头融合）:", "全部一致 ✔" if rc == 0 else "有不一致 ✘")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
