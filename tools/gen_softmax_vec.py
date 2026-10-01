#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""行 softmax 的**向量化**版本：`softmax_vec_k` + 自检。

和 `softmax_k` 的差别只在访存方式：

* `softmax_k`：lane l 处理列 `{l, l+64, l+128, …}`，每个元素一条
  `global_load_dword` 后面跟一条 `s_waitcnt vmcnt(0)`——一个 warp 扫 16K 列要
  256 轮，每轮一次全排空，实测 291 us（解码注意力里最大的一块）。
* `softmax_vec_k`：lane l **一次处理连续 4 列**（`global_load_dwordx4`），
  一个 warp 一轮覆盖 256 个连续元素——访存完全合并、指令数少 4 倍、
  轮数少 4 倍（16K 列只要 64 轮）。

约束：`cols % 256 == 0`（64 lane × 4 列）。max / exp+sum / normalize 三趟
都用 dwordx4，LDS 树形归约与 `softmax_k` 一致。

签名：`softmax_vec_k(float* y, const float* x, uint32_t rows, uint32_t cols,
uint32_t group_size)`；grid = rows，workgroup = 64。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "softmax_vec_k"
ARGS = [buffer_arg(0), buffer_arg(8), scalar_arg(16, 4),
        scalar_arg(20, 4), scalar_arg(24, 4)]
KERNARG_SIZE = 32


def _lds_reduce(L: list[str], op: str, off: int, reg: str = "v8") -> None:
    """LDS 树形归约（与 `softmax_k` 逐字节同构）。"""
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


def _addr(L: list[str], base_lo: str, base_hi: str, dst_lo: str = "v11",
          dst_hi: str = "v12") -> None:
    """`dst = base + v10`（v10 是字节偏移）。"""
    L += [
        f"v_mov_b32_e32 {dst_hi}, {base_hi}",
        f"v_add_co_u32_e32 {dst_lo}, vcc, {base_lo}, v10",
        f"v_addc_co_u32_e32 {dst_hi}, vcc, {dst_hi}, v3, vcc",
    ]


def _off(L: list[str]) -> None:
    """v10 = ((i*256) + lane*4) * 4 —— 一轮 256 个元素的字节偏移。"""
    L += [
        "v_mov_b32_e32 v10, s23",
        "v_lshlrev_b32_e32 v10, 10, v10",     # i*1024 字节 = i*256 元素
        "v_add_u32_e32 v10, v10, v9",         # + lane*16 字节
    ]


def gen_asm() -> str:
    L: list[str] = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",    # y
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",    # x
        "s_load_dword s20, s[4:5], 0x10",          # rows
        "s_load_dword s21, s[4:5], 0x14",          # cols
        "v_mov_b32_e32 v1, 0",
        "s_waitcnt lgkmcnt(0)",
        "s_lshr_b32 s22, s21, 8",                  # 每 lane 的轮数 = cols/256
        "v_mov_b32_e32 v3, 0",
        # row 偏移 = blockIdx.x * cols * 4
        "v_mov_b32_e32 v1, s6",
        "v_mul_lo_u32 v1, v1, s21",
        "v_lshlrev_b32_e32 v1, 2, v1",
        "v_mov_b32_e32 v5, s19",
        "v_add_co_u32_e32 v4, vcc, s18, v1",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",  # x row
        "v_mov_b32_e32 v7, s17",
        "v_add_co_u32_e32 v6, vcc, s16, v1",
        "v_addc_co_u32_e32 v7, vcc, v7, v3, vcc",  # y row
        "v_lshlrev_b32_e32 v9, 4, v0",             # lane*16 字节
        # ---------------- max ----------------
        "v_mov_b32_e32 v8, 0xff800000",            # -inf
        "s_mov_b32 s23, 0",
        "L_max:",
    ]
    _off(L)
    _addr(L, "s18", "s19")                          # 这里用行基址寄存器 v4/v5
    L += [
        "v_mov_b32_e32 v12, v5",
        "v_add_co_u32_e32 v11, vcc, v4, v10",
        "v_addc_co_u32_e32 v12, vcc, v12, v3, vcc",
        "global_load_dwordx4 v[20:23], v[11:12], off",
        "s_waitcnt vmcnt(0)",
        "v_max_f32_e32 v20, v20, v21",
        "v_max_f32_e32 v22, v22, v23",
        "v_max_f32_e32 v20, v20, v22",
        "v_max_f32_e32 v8, v8, v20",
        "s_add_i32 s23, s23, 1",
        "s_cmp_lt_u32 s23, s22",
        "s_cbranch_scc1 L_max",
        # LDS max reduce + broadcast
        "v_lshlrev_b32_e32 v11, 2, v0",
        "ds_write_b32 v11, v8",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
    ]
    for off in (32, 16, 8, 4, 2, 1):
        _lds_reduce(L, "v_max_f32_e32", off, "v8")
    L += [
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[2:3], vcc",
        "v_mov_b32_e32 v16, 256",
        "ds_write_b32 v16, v8",
        "s_or_b64 exec, exec, s[2:3]",
        "s_barrier",
        "v_mov_b32_e32 v16, 256",
        "ds_read_b32 v14, v16",                     # v14 = 行最大值
        "s_waitcnt lgkmcnt(0)",
        # ---------------- exp + sum + store ----------------
        "v_mov_b32_e32 v15, 0",
        "v_mov_b32_e32 v17, 0x3fb8aa3b",            # log2(e)
        "s_mov_b32 s23, 0",
        "L_exp:",
    ]
    _off(L)
    L += [
        "v_mov_b32_e32 v12, v5",
        "v_add_co_u32_e32 v11, vcc, v4, v10",
        "v_addc_co_u32_e32 v12, vcc, v12, v3, vcc",
        "global_load_dwordx4 v[20:23], v[11:12], off",
        "s_waitcnt vmcnt(0)",
    ]
    for r in ("v20", "v21", "v22", "v23"):
        L += [
            f"v_sub_f32_e32 {r}, {r}, v14",
            f"v_mul_f32_e32 {r}, {r}, v17",
            "s_nop 0",
            f"v_exp_f32_e32 {r}, {r}",
        ]
    L += [
        "v_add_f32_e32 v15, v15, v20",
        "v_add_f32_e32 v15, v15, v21",
        "v_add_f32_e32 v15, v15, v22",
        "v_add_f32_e32 v15, v15, v23",
        "v_mov_b32_e32 v12, v7",
        "v_add_co_u32_e32 v11, vcc, v6, v10",
        "v_addc_co_u32_e32 v12, vcc, v12, v3, vcc",
        "global_store_dwordx4 v[11:12], v[20:23], off",
        "s_add_i32 s23, s23, 1",
        "s_cmp_lt_u32 s23, s22",
        "s_cbranch_scc1 L_exp",
        # LDS sum reduce + broadcast
        "v_lshlrev_b32_e32 v11, 2, v0",
        "ds_write_b32 v11, v15",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
    ]
    for off in (32, 16, 8, 4, 2, 1):
        _lds_reduce(L, "v_add_f32_e32", off, "v15")
    L += [
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[2:3], vcc",
        "v_mov_b32_e32 v16, 260",
        "ds_write_b32 v16, v15",
        "s_or_b64 exec, exec, s[2:3]",
        "s_barrier",
        "v_mov_b32_e32 v16, 260",
        "ds_read_b32 v15, v16",
        "s_waitcnt lgkmcnt(0)",
        "s_nop 0",
        "v_rcp_f32_e32 v17, v15",
        "s_nop 0",
        # ---------------- normalize ----------------
        "s_mov_b32 s23, 0",
        "L_norm:",
    ]
    _off(L)
    L += [
        "v_mov_b32_e32 v12, v7",
        "v_add_co_u32_e32 v11, vcc, v6, v10",
        "v_addc_co_u32_e32 v12, vcc, v12, v3, vcc",
        "global_load_dwordx4 v[20:23], v[11:12], off",
        "s_waitcnt vmcnt(0)",
    ]
    for r in ("v20", "v21", "v22", "v23"):
        L.append(f"v_mul_f32_e32 {r}, {r}, v17")
    L += [
        "global_store_dwordx4 v[11:12], v[20:23], off",
        "s_add_i32 s23, s23, 1",
        "s_cmp_lt_u32 s23, s22",
        "s_cbranch_scc1 L_norm",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


def selftest(rows: int = 8, cols: int = 1024) -> int:
    rng = np.random.default_rng(6)
    x = rng.standard_normal((rows, cols), dtype=np.float32) * 2.0
    ref = np.exp(x - x.max(axis=1, keepdims=True))
    ref /= ref.sum(axis=1, keepdims=True)
    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE,
                      group_segment=1024, tag="softmaxvec")
    out = run_one(hsaco, NAME,
                  [{"buffer": "y"}, {"buffer": "x"},
                   {"scalar": {"dtype": "u32", "value": rows}},
                   {"scalar": {"dtype": "u32", "value": cols}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"y": {"dtype": "f32", "values": [0.0] * (rows * cols)},
                   "x": {"dtype": "f32", "values": x.reshape(-1).tolist()}},
                  grid=rows * 64, workgroup=64)
    got = np.array(out["y"], dtype=np.float32).reshape(rows, cols)
    d = float(np.abs(got - ref).max())
    print(f"{NAME} rows={rows} cols={cols} max_abs={d:.3e} "
          f"{'一致 ✔' if d < 1e-5 else '不一致 ✘'}")
    return 0 if d < 1e-5 else 1


def main() -> int:
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
