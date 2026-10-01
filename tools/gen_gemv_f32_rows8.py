#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8 行 / workgroup 的 f32 GEMV（`gemv_f32_rows8_k`）+ 自检。

    python3 tools/gen_gemv_f32_rows8.py

为什么要有这个内核：`gemv_f32_warp_k`（一行一个 warp）把**每一行**都单独做一次
6 步 LDS 树形归约 + 6 次 `s_barrier`。实测它的耗时**几乎与 k 无关**（nrows=16384
时 k=128 → 126 us、k=512 → 134 us）——瓶颈不是访存而是**每行的归约/同步开销**，
所以解码注意力的 `K·q`（n_kv 行 × dim=128）被这个固定开销拖到 ~66 GB/s。

这里的映射（workgroup = 64 lane = 1 warp）：

    g  = lane >> 3        # 0..7   —— 该 lane 负责第 g 行
    li = lane & 7         # 0..7   —— 行内第 li 段（每段 16 个 float）
    行号 = blockIdx.x * 8 + g

于是：

* 每行 128 个 float 由 8 个 lane 用 4 条 `global_load_dwordx4` 读完（lane li 读
  `li*16 .. li*16+15`），warp 内 8 个组各读自己那一行 → 每条指令 8 个 128B
  事务、完全占满；
* 归约只在 **8 个 lane**（同一行的 8 段）之间做，树只有 3 步；而 3 次
  `s_barrier` 是**整个 workgroup 共用**的 → 每 8 行只要 3 次同步，
  `gemv_f32_warp_k` 是每行 6 次（每 8 行 48 次）。同步次数少 16 倍。

前提：`k % 128 == 0`（每段 16 个 float、8 段一轮）。不满足时调用方回退到
`gemv_f32_warp_k`。签名与 `gemv_f32_warp_k` **完全相同**，只是 grid 是
`ceil(nrows/8)` 个 workgroup（尾部不足 8 行的部分请用 `gemv_f32_warp_k` 收尾）。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "gemv_f32_rows8_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
        scalar_arg(24, 4), scalar_arg(28, 4), scalar_arg(32, 4)]
KERNARG_SIZE = 40

# split-K 版本：`out = W·x` 里每个 workgroup 只扫 k 的一段，把 8 行的**部分和**
# 写进 `part[row*nsplit + s]`，再由 `reduce_blocks_k` 沿 nsplit 求和。
# 用于解码注意力的 `Vt·P`（nrows=dim=128 太少、k=pad 很大，"一行一个 warp"
# 的版本只有 128 个 workgroup、每个还得串行 k/64 次访存往返，实测 129 us）。
SPLIT_NAME = "gemv_f32_rows8_split_k"
SPLIT_ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
              scalar_arg(24, 4), scalar_arg(28, 4), scalar_arg(32, 4),
              scalar_arg(36, 4)]
SPLIT_KERNARG = 40
ROWS_PER_WG = 8
LANES_PER_ROW = 8
ELEMS_PER_LANE = 16                 # 4 × dwordx4


def gen_asm() -> str:
    L = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",   # w
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",   # x
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",  # y
        "s_load_dword s22, s[4:5], 0x18",         # nrows
        "s_load_dword s23, s[4:5], 0x1c",         # k
        "s_waitcnt lgkmcnt(0)",
        # s24 = 行距（字节）= k*4；s25 = 每行轮数 = k/128
        "s_lshl_b32 s24, s23, 2",
        "s_lshr_b32 s26, s23, 7",
        # g = lane>>3（第几行）、li = lane&7（行内第几段）
        "v_lshrrev_b32_e32 v1, 3, v0",
        "v_and_b32_e32 v2, 7, v0",
        # row = blockIdx.x*8 + g
        "v_mov_b32_e32 v3, s6",
        "v_lshlrev_b32_e32 v3, 3, v3",
        "v_add_u32_e32 v4, v1, v3",
        # row < nrows ？否则整条 lane 不干活（只启动整组时不会触发，留作保险）
        "v_mov_b32_e32 v5, s22",
        "v_cmp_lt_u32_e64 vcc, v4, v5",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        # 本 lane 的行内字节偏移 = li*64
        "v_lshlrev_b32_e32 v6, 6, v2",
        # w 地址 = w + row*行距 + li*64
        "v_mul_lo_u32 v7, v4, s24",
        "v_add_u32_e32 v7, v7, v6",
        "v_mov_b32_e32 v8, s16",
        "v_mov_b32_e32 v9, s17",
        "v_mov_b32_e32 v10, 0",
        "v_add_co_u32_e32 v8, vcc, v8, v7",
        "v_addc_co_u32_e32 v9, vcc, v9, v10, vcc",
        # x 地址 = x + li*64（8 个组读同一段 → 硬件广播）
        "v_mov_b32_e32 v11, s18",
        "v_mov_b32_e32 v12, s19",
        "v_add_co_u32_e32 v11, vcc, v11, v6",
        "v_addc_co_u32_e32 v12, vcc, v12, v10, vcc",
        # 累加器（4 段）
        "v_mov_b32_e32 v20, 0",
        "v_mov_b32_e32 v21, 0",
        "v_mov_b32_e32 v22, 0",
        "v_mov_b32_e32 v23, 0",
        "s_mov_b32 s25, 0",
        "L_loop:",
        # 一次把 4 段 w（16 个 float）和 4 段 x 都发出去，只等一次
        "global_load_dwordx4 v[28:31], v[8:9], off",
        "global_load_dwordx4 v[32:35], v[8:9], off offset:16",
        "global_load_dwordx4 v[36:39], v[8:9], off offset:32",
        "global_load_dwordx4 v[40:43], v[8:9], off offset:48",
        "global_load_dwordx4 v[44:47], v[11:12], off",
        "global_load_dwordx4 v[48:51], v[11:12], off offset:16",
        "global_load_dwordx4 v[52:55], v[11:12], off offset:32",
        "global_load_dwordx4 v[56:59], v[11:12], off offset:48",
        "s_waitcnt vmcnt(0)",
        "v_fma_f32 v20, v28, v44, v20",
        "v_fma_f32 v21, v29, v45, v21",
        "v_fma_f32 v22, v30, v46, v22",
        "v_fma_f32 v23, v31, v47, v23",
        "v_fma_f32 v20, v32, v48, v20",
        "v_fma_f32 v21, v33, v49, v21",
        "v_fma_f32 v22, v34, v50, v22",
        "v_fma_f32 v23, v35, v51, v23",
        "v_fma_f32 v20, v36, v52, v20",
        "v_fma_f32 v21, v37, v53, v21",
        "v_fma_f32 v22, v38, v54, v22",
        "v_fma_f32 v23, v39, v55, v23",
        "v_fma_f32 v20, v40, v56, v20",
        "v_fma_f32 v21, v41, v57, v21",
        "v_fma_f32 v22, v42, v58, v22",
        "v_fma_f32 v23, v43, v59, v23",
        # 下一轮：两个矩阵各前进 512 字节（128 个 float）
        "v_mov_b32_e32 v60, 512",
        "v_add_co_u32_e32 v8, vcc, v8, v60",
        "v_addc_co_u32_e32 v9, vcc, v9, v10, vcc",
        "v_add_co_u32_e32 v11, vcc, v11, v60",
        "v_addc_co_u32_e32 v12, vcc, v12, v10, vcc",
        "s_add_i32 s25, s25, 1",
        "s_cmp_lt_u32 s25, s26",
        "s_cbranch_scc1 L_loop",
        # 段内 4 路合并
        "v_add_f32_e32 v20, v20, v21",
        "v_add_f32_e32 v22, v22, v23",
        "v_add_f32_e32 v24, v20, v22",
        # 行内 8 lane 树形归约（3 步），槽位 = lane id（g*8+li == lane）
        "v_lshlrev_b32_e32 v25, 2, v0",
        "ds_write_b32 v25, v24",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
    ]
    for off in (4, 2, 1):
        L += [
            f"v_cmp_gt_u32_e32 vcc, {off}, v2",        # li < off
            "s_and_saveexec_b64 s[2:3], vcc",
            "v_lshlrev_b32_e32 v26, 2, v0",
            f"v_add_u32_e32 v26, {off * 4}, v26",
            "ds_read_b32 v27, v26",
            "s_waitcnt lgkmcnt(0)",
            "v_add_f32_e32 v24, v24, v27",
            "ds_write_b32 v25, v24",
            "s_or_b64 exec, exec, s[2:3]",
            "s_barrier",
        ]
    L += [
        # li == 0 的 lane 写 y[block*8+g]
        "v_cmp_eq_u32_e32 vcc, 0, v2",
        "s_and_saveexec_b64 s[2:3], vcc",
        "v_mov_b32_e32 v28, s6",
        "v_lshlrev_b32_e32 v28, 3, v28",
        "v_add_u32_e32 v28, v28, v1",
        "v_lshlrev_b32_e32 v28, 2, v28",
        "v_mov_b32_e32 v29, s20",
        "v_mov_b32_e32 v30, s21",
        "v_mov_b32_e32 v31, 0",
        "v_add_co_u32_e32 v29, vcc, v29, v28",
        "v_addc_co_u32_e32 v30, vcc, v30, v31, vcc",
        "ds_read_b32 v27, v25",
        "s_waitcnt lgkmcnt(0)",
        "global_store_dword v[29:30], v27, off",
        "s_or_b64 exec, exec, s[2:3]",
        "L_end:",
        "s_or_b64 exec, exec, s[0:1]",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


def _ref(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (w @ x).astype(np.float32)


def gen_split_asm() -> str:
    """`gemv_f32_rows8_split_k(w, x, part, nrows, k, stride, sh)`。

    workgroup 数 = `(nrows/8) << sh`：`blk = blockIdx.x >> sh` 选 8 行一组，
    `s = blockIdx.x & (2^sh-1)` 选 k 的第几段（每段 `chunk = k >> sh` 列）。
    行内映射与 `gemv_f32_rows8_k` 完全一样，只是列起点加上 `s*chunk`、结果写
    到 `part[row*2^sh + s]`。`stride` 是 w 的行距（元素数），所以调用方既能
    用「行距 = 列数」的紧凑矩阵，也能用在缓存没填满、行距 = max_len 的 Vt 上。
    """
    L = [
        ".text", f"k_{SPLIT_NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",   # w
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",   # x
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",  # part
        "s_load_dword s22, s[4:5], 0x18",         # nrows
        "s_load_dword s23, s[4:5], 0x1c",         # k（要扫的列数）
        "s_load_dword s27, s[4:5], 0x20",         # stride（行距，元素）
        "s_load_dword s28, s[4:5], 0x24",         # sh = log2(nsplit)
        "s_waitcnt lgkmcnt(0)",
        # s24 = 行距字节；s25 = 每段字节；s26 = 每 lane 轮数 = (k>>sh)/128
        "s_lshl_b32 s24, s27, 2",
        "s_lshl_b32 s25, s23, 2",
        "s_lshr_b32 s25, s25, s28",
        "s_add_i32 s29, s28, 7",
        "s_lshr_b32 s26, s23, s29",
        # blk = blockIdx.x >> sh ; s = blockIdx.x & (nsplit-1)
        "v_mov_b32_e32 v1, s6",
        "v_lshrrev_b32_e32 v2, s28, v1",
        "v_mov_b32_e32 v62, 1",
        "v_lshlrev_b32_e32 v30, s28, v62",
        "v_sub_u32_e32 v30, v30, v62",            # nsplit-1
        "v_and_b32_e32 v3, v30, v1",              # s
        # g = lane>>3 ; li = lane&7 ; row = blk*8 + g
        "v_lshrrev_b32_e32 v4, 3, v0",
        "v_and_b32_e32 v5, 7, v0",
        "v_lshlrev_b32_e32 v2, 3, v2",
        "v_add_u32_e32 v6, v2, v4",               # row
        "v_mov_b32_e32 v14, s22",
        "v_cmp_lt_u32_e64 vcc, v6, v14",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        # 字节偏移：li*64、row*行距、s*段长
        "v_lshlrev_b32_e32 v7, 6, v5",
        "v_mul_lo_u32 v8, v6, s24",
        "v_mul_lo_u32 v31, v3, s25",
        "v_add_u32_e32 v8, v8, v31",
        "v_add_u32_e32 v8, v8, v7",               # w 偏移
        "v_add_u32_e32 v31, v31, v7",             # x 偏移
        "v_mov_b32_e32 v11, 0",
        "v_mov_b32_e32 v9, s16",
        "v_mov_b32_e32 v10, s17",
        "v_add_co_u32_e32 v9, vcc, v9, v8",
        "v_addc_co_u32_e32 v10, vcc, v10, v11, vcc",
        "v_mov_b32_e32 v12, s18",
        "v_mov_b32_e32 v13, s19",
        "v_add_co_u32_e32 v12, vcc, v12, v31",
        "v_addc_co_u32_e32 v13, vcc, v13, v11, vcc",
        # 累加器
        "v_mov_b32_e32 v20, 0",
        "v_mov_b32_e32 v21, 0",
        "v_mov_b32_e32 v22, 0",
        "v_mov_b32_e32 v23, 0",
        "s_mov_b32 s30, 0",
        "L_loop:",
        "s_cmp_lt_u32 s30, s26",
        "s_cbranch_scc0 L_done",
        "global_load_dwordx4 v[28:31], v[9:10], off",
        "global_load_dwordx4 v[32:35], v[9:10], off offset:16",
        "global_load_dwordx4 v[36:39], v[9:10], off offset:32",
        "global_load_dwordx4 v[40:43], v[9:10], off offset:48",
        "global_load_dwordx4 v[44:47], v[12:13], off",
        "global_load_dwordx4 v[48:51], v[12:13], off offset:16",
        "global_load_dwordx4 v[52:55], v[12:13], off offset:32",
        "global_load_dwordx4 v[56:59], v[12:13], off offset:48",
        "s_waitcnt vmcnt(0)",
        "v_fma_f32 v20, v28, v44, v20",
        "v_fma_f32 v21, v29, v45, v21",
        "v_fma_f32 v22, v30, v46, v22",
        "v_fma_f32 v23, v31, v47, v23",
        "v_fma_f32 v20, v32, v48, v20",
        "v_fma_f32 v21, v33, v49, v21",
        "v_fma_f32 v22, v34, v50, v22",
        "v_fma_f32 v23, v35, v51, v23",
        "v_fma_f32 v20, v36, v52, v20",
        "v_fma_f32 v21, v37, v53, v21",
        "v_fma_f32 v22, v38, v54, v22",
        "v_fma_f32 v23, v39, v55, v23",
        "v_fma_f32 v20, v40, v56, v20",
        "v_fma_f32 v21, v41, v57, v21",
        "v_fma_f32 v22, v42, v58, v22",
        "v_fma_f32 v23, v43, v59, v23",
        "v_mov_b32_e32 v62, 512",
        "v_add_co_u32_e32 v9, vcc, v9, v62",
        "v_addc_co_u32_e32 v10, vcc, v10, v11, vcc",
        "v_add_co_u32_e32 v12, vcc, v12, v62",
        "v_addc_co_u32_e32 v13, vcc, v13, v11, vcc",
        "s_add_i32 s30, s30, 1",
        "s_branch L_loop",
        "L_done:",
        "v_add_f32_e32 v20, v20, v21",
        "v_add_f32_e32 v22, v22, v23",
        "v_add_f32_e32 v24, v20, v22",
        # 行内 8 lane 树形归约（3 步）
        "v_lshlrev_b32_e32 v25, 2, v0",
        "ds_write_b32 v25, v24",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
    ]
    for off in (4, 2, 1):
        L += [
            f"v_cmp_gt_u32_e32 vcc, {off}, v5",        # li < off
            "s_and_saveexec_b64 s[2:3], vcc",
            "v_lshlrev_b32_e32 v26, 2, v0",
            f"v_add_u32_e32 v26, {off * 4}, v26",
            "ds_read_b32 v27, v26",
            "s_waitcnt lgkmcnt(0)",
            "v_add_f32_e32 v24, v24, v27",
            "ds_write_b32 v25, v24",
            "s_or_b64 exec, exec, s[2:3]",
            "s_barrier",
        ]
    L += [
        # li == 0 的 lane 写 part[(row<<sh) + s]
        "v_cmp_eq_u32_e32 vcc, 0, v5",
        "s_and_saveexec_b64 s[2:3], vcc",
        "v_lshlrev_b32_e32 v28, s28, v6",          # row << sh
        "v_add_u32_e32 v28, v28, v3",              # + s
        "v_lshlrev_b32_e32 v28, 2, v28",
        "v_mov_b32_e32 v29, s20",
        "v_mov_b32_e32 v30, s21",
        "v_mov_b32_e32 v31, 0",
        "v_add_co_u32_e32 v29, vcc, v29, v28",
        "v_addc_co_u32_e32 v30, vcc, v30, v31, vcc",
        "ds_read_b32 v27, v25",
        "s_waitcnt lgkmcnt(0)",
        "global_store_dword v[29:30], v27, off",
        "s_or_b64 exec, exec, s[2:3]",
        "L_end:",
        "s_or_b64 exec, exec, s[0:1]",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


def check(cases) -> bool:
    ok = True
    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE,
                      group_segment=64 * 4, tag="rows8")
    for nrows, k in cases:
        rng = np.random.default_rng(11 + nrows + k)
        w = rng.standard_normal((nrows, k)).astype(np.float32)
        x = rng.standard_normal(k).astype(np.float32)
        ref = _ref(w, x)
        ngroups = nrows // ROWS_PER_WG
        out = run_one(hsaco, NAME,
                      [{"buffer": "w"}, {"buffer": "x"}, {"buffer": "y"},
                       {"scalar": {"dtype": "u32", "value": nrows}},
                       {"scalar": {"dtype": "u32", "value": k}},
                       {"scalar": {"dtype": "u32", "value": 64}}],
                      {"w": {"dtype": "f32", "values": w.reshape(-1).tolist()},
                       "x": {"dtype": "f32", "values": x.tolist()},
                       "y": {"dtype": "f32", "values": [0.0] * nrows}},
                      grid=ngroups * 64, workgroup=64)
        got = np.array(out["y"], dtype=np.float32)[:ngroups * ROWS_PER_WG]
        d = np.abs(got - ref[:ngroups * ROWS_PER_WG])
        rel = float(d.max() / max(1e-9, float(np.abs(ref).max())))
        good = rel < 1e-5
        ok = ok and good
        print(f"nrows={nrows} k={k}: max_rel={rel:.3e} "
              f"{'一致 ✔' if good else '不一致 ✘'}")
    return ok


def main() -> int:
    ok = check([(64, 128), (256, 128), (512, 256), (128, 512), (2048, 128)])
    print("gemv_f32_rows8_k vs numpy:", "全部一致 ✔" if ok else "有不一致 ✘")
    ok2 = check_split()
    print("gemv_f32_rows8_split_k vs numpy:",
          "全部一致 ✔" if ok2 else "有不一致 ✘")
    return 0 if (ok and ok2) else 1


# `tools/build_native_kernels.py` 从这里取两个内核
KERNELS = [
    (NAME, gen_asm, ARGS, KERNARG_SIZE),
    (SPLIT_NAME, gen_split_asm, SPLIT_ARGS, SPLIT_KERNARG),
]


def check_split() -> bool:
    """split-K 版：分块部分和相加 == 完整 GEMV；行距可以大于列数。"""
    hsaco = build_one(SPLIT_NAME, gen_split_asm(), SPLIT_ARGS,
                      kernarg_size=SPLIT_KERNARG, group_segment=64 * 4,
                      tag="rows8s")
    ok = True
    for nrows, k, stride, sh in ((128, 2048, 2048, 4), (128, 16384, 16384, 6),
                                 (8, 4096, 8192, 2), (16, 1024, 1024, 1)):
        rng = np.random.default_rng(900 + sh + k % 7)
        w = rng.standard_normal((nrows, stride)).astype(np.float32)
        x = rng.standard_normal(k).astype(np.float32)
        nsplit = 1 << sh
        out = run_one(hsaco, SPLIT_NAME,
                      [{"buffer": "w"}, {"buffer": "x"}, {"buffer": "part"},
                       {"scalar": {"dtype": "u32", "value": nrows}},
                       {"scalar": {"dtype": "u32", "value": k}},
                       {"scalar": {"dtype": "u32", "value": stride}},
                       {"scalar": {"dtype": "u32", "value": sh}}],
                      {"w": {"dtype": "f32", "values": w.reshape(-1).tolist()},
                       "x": {"dtype": "f32", "values": x.tolist()},
                       "part": {"dtype": "f32", "values": [0.0] * (nrows * nsplit)}},
                      grid=(nrows // ROWS_PER_WG) * nsplit * 64, workgroup=64)
        got = np.array(out["part"], np.float32).reshape(nrows, nsplit)
        ref = (w[:, :k] @ x).astype(np.float32)
        d = float(np.abs(got.sum(axis=1) - ref).max()
                  / max(1e-9, float(np.abs(ref).max())))
        good = d < 1e-5
        ok = ok and good
        print(f"split nrows={nrows} k={k} stride={stride} nsplit={nsplit}: "
              f"rel={d:.2e} {'一致 ✔' if good else '不一致 ✘'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
