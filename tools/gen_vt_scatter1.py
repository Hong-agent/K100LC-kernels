#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单行 KV 追加（`vt_scatter_v1_k`）：一次 launch 完成「K 转置写 + V 拷贝」。

解码时每个 token 只追加 **1 行**。`vt_scatter_v_k` 是给「一次追加一整块
(64 行)」设计的：它跑 `for t in range(0,64)` 和 `for d in range(0,64)` 两圈
固定 64 次的循环，只在 `r < rows` / `c < dim` 时干活；rows=1、dim=512 时
(grid=8) 实测 **22.9 us**，其中真正要做的只有「读 1 个 K + 读 1 个 V + 两笔
store」——**15.8 us 全花在两圈空转的标量循环上**（每圈 ~64 次 SALU 依赖链）。

这里按「一个 lane 一列」直接铺开，连循环都不需要：

    vt[c*ystride + y0] = k[c]        （K：转置写，c = blk*64 + lane）
    vc[y0*dim + c]     = v[c]        （V：原样拷）

grid = `dim/64`、workgroup = 64；要求 `dim % 64 == 0`。rows > 1 时仍然走
`vt_scatter_v_k`（那才是它擅长的形状）。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "vt_scatter_v1_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16), buffer_arg(24),
        scalar_arg(32, 4), scalar_arg(36, 4), scalar_arg(40, 4)]
KERNARG_SIZE = 48


def gen_asm() -> str:
    return "\n".join([
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",     # vt（Kt [dim, max_len]）
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",     # vc（V [max_len, dim]）
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",    # k
        "s_load_dwordx2 s[22:23], s[4:5], 0x18",    # v
        "s_load_dword s24, s[4:5], 0x20",           # dim
        "s_load_dword s25, s[4:5], 0x24",           # ystride（Kt 的行距）
        "s_load_dword s26, s[4:5], 0x28",           # y0（写到第几列）
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v1, 0",
        "v_mov_b32_e32 v2, s6",                     # blk
        "v_lshlrev_b32_e32 v2, 6, v2",
        "v_add_u32_e32 v2, v2, v0",                 # c = blk*64 + lane
        "v_mov_b32_e32 v3, s24",
        "v_cmp_lt_u32_e64 vcc, v2, v3",             # c < dim ？
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        "v_lshlrev_b32_e32 v4, 2, v2",              # c*4（k / v 的列偏移）
        "v_mov_b32_e32 v5, s20",
        "v_mov_b32_e32 v6, s21",
        "v_add_co_u32_e32 v5, vcc, v5, v4",
        "v_addc_co_u32_e32 v6, vcc, v6, v1, vcc",
        "global_load_dword v7, v[5:6], off",        # k[c]
        "v_mov_b32_e32 v5, s22",
        "v_mov_b32_e32 v6, s23",
        "v_add_co_u32_e32 v5, vcc, v5, v4",
        "v_addc_co_u32_e32 v6, vcc, v6, v1, vcc",
        "global_load_dword v8, v[5:6], off",        # v[c]
        "s_waitcnt vmcnt(0)",
        # vt[c*ystride + y0] = k[c]
        "v_mul_lo_u32 v9, v2, s25",
        "v_mov_b32_e32 v15, s26",
        "v_add_u32_e32 v9, v9, v15",
        "v_lshlrev_b32_e32 v9, 2, v9",
        "v_mov_b32_e32 v10, s16",
        "v_mov_b32_e32 v11, s17",
        "v_add_co_u32_e32 v10, vcc, v10, v9",
        "v_addc_co_u32_e32 v11, vcc, v11, v1, vcc",
        "global_store_dword v[10:11], v7, off",
        # vc[y0*dim + c] = v[c]
        "s_mul_i32 s27, s26, s24",                  # y0*dim
        "v_mov_b32_e32 v15, s27",
        "v_add_u32_e32 v12, v15, v2",
        "v_lshlrev_b32_e32 v12, 2, v12",
        "v_mov_b32_e32 v13, s18",
        "v_mov_b32_e32 v14, s19",
        "v_add_co_u32_e32 v13, vcc, v13, v12",
        "v_addc_co_u32_e32 v14, vcc, v14, v1, vcc",
        "global_store_dword v[13:14], v8, off",
        "L_end:",
        "s_or_b64 exec, exec, s[0:1]",
        "s_endpgm",
    ]) + "\n"


def selftest() -> int:
    hsaco = build_one(NAME, gen_asm(), ARGS, KERNARG_SIZE, tag="vts1")
    rng = np.random.default_rng(20261011)
    rc = 0
    for dim, max_len, y0 in ((64, 8, 0), (128, 16, 5), (512, 64, 0),
                             (512, 256, 63), (1024, 300, 299), (64, 64, 1)):
        k = rng.standard_normal(dim).astype(np.float32)
        v = rng.standard_normal(dim).astype(np.float32)
        vt = np.full((dim, max_len), -7.0, np.float32)
        vc = np.full((max_len, dim), -7.0, np.float32)
        o = run_one(hsaco, NAME,
                    [{"buffer": "vt"}, {"buffer": "vc"}, {"buffer": "k"},
                     {"buffer": "v"},
                     {"scalar": {"dtype": "u32", "value": dim}},
                     {"scalar": {"dtype": "u32", "value": max_len}},
                     {"scalar": {"dtype": "u32", "value": y0}}],
                    {"k": {"dtype": "f32", "values": k.tolist()},
                     "v": {"dtype": "f32", "values": v.tolist()},
                     "vt": {"dtype": "f32", "values": vt.reshape(-1).tolist()},
                     "vc": {"dtype": "f32", "values": vc.reshape(-1).tolist()}},
                    grid=(dim // 64) * 64, workgroup=64)
        got_vt = np.array(o["vt"], np.float32).reshape(dim, max_len)
        got_vc = np.array(o["vc"], np.float32).reshape(max_len, dim)
        ref_vt = vt.copy(); ref_vt[:, y0] = k
        ref_vc = vc.copy(); ref_vc[y0] = v
        bad = int((got_vt != ref_vt).sum()) + int((got_vc != ref_vc).sum())
        ok = bad == 0
        rc |= 0 if ok else 1
        print(f"dim={dim} max_len={max_len} y0={y0}: 不同元素={bad} "
              f"{'✔' if ok else '✘'}")
    return rc


def main() -> int:
    rc = selftest()
    print("vt_scatter_v1_k:", "全部一致 ✔" if rc == 0 else "有不一致 ✘")
    return rc


KERNELS = [(NAME, gen_asm, ARGS, KERNARG_SIZE)]

if __name__ == "__main__":
    raise SystemExit(main())
