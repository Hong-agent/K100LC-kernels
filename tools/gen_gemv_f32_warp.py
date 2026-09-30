#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""warp-per-row 的 f32 GEMV（合并访存）+ 自检。

    python3 tools/gen_gemv_f32_warp.py

老版本 `gemv_f32_k` 每个 lane 处理一整行：迭代 i 时 lane l 读 `row_l*k + i`
—— 相邻 lane 的地址差 k*4 字节（本模型 10 KB），每次访存要 64 条 cache line
才拿到 1/8 有用的数据。新版让**一个 warp 处理一行**、lane 交错读：
迭代 i 时 64 个 lane 读连续的 256 字节 → 完全合并。

前提：k 是 64 的倍数（本模型所有矩阵满足）。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "gemv_f32_warp_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
        scalar_arg(24, 4), scalar_arg(28, 4), scalar_arg(32, 4)]
KERNARG_SIZE = 40


def gen_asm() -> str:
    L = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",   # w
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",   # x
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",  # y
        "s_load_dword s22, s[4:5], 0x18",         # nrows
        "s_load_dword s23, s[4:5], 0x1c",         # k
        "s_waitcnt lgkmcnt(0)",
        "s_cmp_ge_u32 s6, s22",                   # row = blockIdx.x ≥ nrows → 退出
        "s_cbranch_scc1 L_end",
        # wrow = w + row*k
        "v_mov_b32_e32 v1, s6",
        "v_mov_b32_e32 v2, s23",
        "v_lshlrev_b32_e32 v2, 2, v2",            # k*4 字节
        "v_mul_lo_u32 v1, v1, v2",
        "v_mov_b32_e32 v3, 0",
        "v_mov_b32_e32 v4, s16",
        "v_mov_b32_e32 v5, s17",
        "v_add_co_u32_e32 v4, vcc, v4, v1",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "v_mov_b32_e32 v6, s18",
        "v_mov_b32_e32 v7, s19",
        "v_mov_b32_e32 v8, 0",                    # acc
        "v_lshlrev_b32_e32 v9, 2, v0",            # lane*4
        "v_mov_b32_e32 v10, 256",                 # 64 lanes * 4B
        "s_lshr_b32 s24, s23, 6",                 # niter = k/64
        "s_mov_b32 s25, 0",
        "L_loop:",
        "v_add_co_u32_e32 v12, vcc, v4, v9",
        "v_addc_co_u32_e32 v13, vcc, v5, v3, vcc",
        "global_load_dword v14, v[12:13], off",
        "v_add_co_u32_e32 v12, vcc, v6, v9",
        "v_addc_co_u32_e32 v13, vcc, v7, v3, vcc",
        "global_load_dword v15, v[12:13], off",
        "s_waitcnt vmcnt(0)",
        "v_fma_f32 v8, v14, v15, v8",
        "v_add_u32_e32 v9, v10, v9",
        "s_add_i32 s25, s25, 1",
        "s_cmp_lt_u32 s25, s24",
        "s_cbranch_scc1 L_loop",
        # LDS 树形归约（已验证；ds_bpermute 语义实测与预期不符，见 ROADMAP）
        "v_lshlrev_b32_e32 v11, 2, v0",
        "ds_write_b32 v11, v8",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
    ]
    for off in (32, 16, 8, 4, 2, 1):
        L += [
            f"v_cmp_gt_u32_e32 vcc, {off}, v0",
            "s_and_saveexec_b64 s[2:3], vcc",
            f"v_mov_b32_e32 v19, {off * 4}",
            "v_add_u32_e32 v19, v11, v19",
            "ds_read_b32 v18, v19",
            "s_waitcnt lgkmcnt(0)",
            "v_add_f32_e32 v8, v8, v18",
            "ds_write_b32 v11, v8",
            "s_or_b64 exec, exec, s[2:3]",
            "s_barrier",
        ]
    L += [
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[2:3], vcc",
        "v_mov_b32_e32 v16, s6",
        "v_lshlrev_b32_e32 v16, 2, v16",
        "v_mov_b32_e32 v17, s21",
        "v_add_co_u32_e32 v16, vcc, s20, v16",
        "v_addc_co_u32_e32 v17, vcc, v17, v3, vcc",
        "v_mov_b32_e32 v11, 0",
        "ds_read_b32 v18, v11",
        "s_waitcnt lgkmcnt(0)",
        "global_store_dword v[16:17], v18, off",
        "s_or_b64 exec, exec, s[2:3]",
        "L_end:",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


def main() -> int:
    nrows, k = 256, 640
    rng = np.random.default_rng(3)
    w = rng.standard_normal((nrows, k)).astype(np.float32)
    x = rng.standard_normal(k).astype(np.float32)
    ref = (w @ x).astype(np.float32)
    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE,
                      group_segment=64 * 4 * 4, tag="warp")
    out = run_one(hsaco, NAME,
                  [{"buffer": "w"}, {"buffer": "x"}, {"buffer": "y"},
                   {"scalar": {"dtype": "u32", "value": nrows}},
                   {"scalar": {"dtype": "u32", "value": k}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"w": {"dtype": "f32", "values": w.reshape(-1).tolist()},
                   "x": {"dtype": "f32", "values": x.tolist()},
                   "y": {"dtype": "f32", "values": [0.0] * nrows}},
                  grid=nrows * 64, workgroup=64)   # run_one 的 grid 是总 work-item 数
    got = np.array(out["y"], dtype=np.float32)
    d = np.abs(got - ref)
    print(f"warp-per-row GEMV: max_abs={d.max():.3e}")
    ok = d.max() < 1e-2 * max(1.0, float(np.abs(ref).max()))
    print("vs numpy:", "一致 ✔" if ok else "不一致 ✘")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
