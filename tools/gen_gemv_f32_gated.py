#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门控 f32 GEMV：`gemv_f32_gated_acc_k` —— `y += W·(silu(g) * u)`。

    python3 tools/gen_gemv_f32_gated.py

门控 MLP 的最后一步原本是三个内核：`gate/up` GEMV → `silu_mul_k` →
`down` GEMV（再并一个残差就是四个）。D 层 decoder 里这是一整条依赖链，
而每条 dispatch 的 GPU 侧开销实测约 7 us。这里把「激活」融进 `down` 的
GEMV：`down` 的输入 `act[j] = silu(g[j]) * u[j]` 在**读进来的那一步**算，
顺带把残差也累加进去：

    y[row] += Σ_j W[row, j] · silu(g[j]) · u[j]

结构就是 `gemv_f32_warp_k`（warp-per-row、lane 交错读、LDS 树形归约）：
每轮多读一路 `u`、多算一次 silu（`exp(-g)` 用 `v_exp_f32` 的 2^x 通路，
所以先乘 `-log2(e)`），最后 `v_fma_f32` 累加。

签名：`(w, g, u, y, nrows, k, 64)`；`k` 必须是 64 的倍数，grid = nrows，
wg = 64。`g`/`u` 分别是门控 MLP 的 gate 与 up 输出（各 `k` 个元素）。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "gemv_f32_gated_acc_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16), buffer_arg(24),
        scalar_arg(32, 4), scalar_arg(36, 4), scalar_arg(40, 4)]
KERNARG_SIZE = 48
LOG2E = "0x3fb8aa3b"


def gen_asm() -> str:
    L = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",   # w
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",   # g（gate）
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",  # u（up）
        "s_load_dwordx2 s[22:23], s[4:5], 0x18",  # y
        "s_load_dword s26, s[4:5], 0x20",         # nrows
        "s_load_dword s27, s[4:5], 0x24",         # k
        "s_waitcnt lgkmcnt(0)",
        "s_cmp_ge_u32 s6, s26",                   # row ≥ nrows → 退出
        "s_cbranch_scc1 L_end",
        # wrow = w + row*k*4
        "v_mov_b32_e32 v1, s6",
        "v_mov_b32_e32 v2, s27",
        "v_lshlrev_b32_e32 v2, 2, v2",
        "v_mul_lo_u32 v1, v1, v2",
        "v_mov_b32_e32 v3, 0",
        "v_mov_b32_e32 v4, s16",
        "v_mov_b32_e32 v5, s17",
        "v_add_co_u32_e32 v4, vcc, v4, v1",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "v_mov_b32_e32 v6, s18",                  # g 基址
        "v_mov_b32_e32 v7, s19",
        "v_mov_b32_e32 v18, s20",                 # u 基址
        "v_mov_b32_e32 v19, s21",
        "v_mov_b32_e32 v8, 0",                    # acc
        "v_lshlrev_b32_e32 v9, 2, v0",            # lane*4
        "v_mov_b32_e32 v10, 256",
        "v_mov_b32_e32 v20, %s" % LOG2E,
        "v_mul_f32_e32 v20, -1.0, v20",           # -log2(e)（silu 用 exp(-g)）
        "v_mov_b32_e32 v21, 1.0",
        "s_lshr_b32 s28, s27, 6",                 # niter = k/64
        "s_mov_b32 s29, 0",
        "L_loop:",
        "v_add_co_u32_e32 v12, vcc, v4, v9",
        "v_addc_co_u32_e32 v13, vcc, v5, v3, vcc",
        "global_load_dword v14, v[12:13], off",   # w
        "v_add_co_u32_e32 v12, vcc, v6, v9",
        "v_addc_co_u32_e32 v13, vcc, v7, v3, vcc",
        "global_load_dword v15, v[12:13], off",   # g
        "v_add_co_u32_e32 v16, vcc, v18, v9",
        "v_addc_co_u32_e32 v17, vcc, v19, v3, vcc",
        "global_load_dword v22, v[16:17], off",   # u
        "s_waitcnt vmcnt(0)",
        # silu(g) = g / (1 + exp(-g))；v_exp_f32 是 2^x → 先乘 -log2(e)
        "v_mul_f32_e32 v23, v15, v20",
        "s_nop 0",
        "v_exp_f32_e32 v23, v23",
        "s_nop 0",
        "v_add_f32_e32 v23, v21, v23",
        "s_nop 0",
        "v_rcp_f32_e32 v23, v23",
        "s_nop 0",
        "v_mul_f32_e32 v23, v23, v15",            # silu(g)
        "v_mul_f32_e32 v23, v23, v22",            # * up
        "v_fma_f32 v8, v14, v23, v8",
        "v_add_u32_e32 v9, v10, v9",
        "s_add_i32 s29, s29, 1",
        "s_cmp_lt_u32 s29, s28",
        "s_cbranch_scc1 L_loop",
        # LDS 树形归约（和 gemv_f32_warp_k 同一套）
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
        "v_mov_b32_e32 v17, s23",
        "v_add_co_u32_e32 v16, vcc, s22, v16",
        "v_addc_co_u32_e32 v17, vcc, v17, v3, vcc",
        "v_mov_b32_e32 v11, 0",
        "ds_read_b32 v18, v11",
        "s_waitcnt lgkmcnt(0)",
        "global_load_dword v24, v[16:17], off",   # y 旧值（残差累加）
        "s_waitcnt vmcnt(0)",
        "v_add_f32_e32 v18, v18, v24",
        "global_store_dword v[16:17], v18, off",
        "s_or_b64 exec, exec, s[2:3]",
        "L_end:",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


def selftest() -> int:
    h = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE,
                  group_segment=1024, tag="gated")
    rng = np.random.default_rng(20261006)
    ok = True
    for nrows, k in ((64, 256), (128, 1024), (512, 1024), (65, 512)):
        w = rng.standard_normal((nrows, k)).astype(np.float32)
        g = rng.standard_normal(k).astype(np.float32)
        u = rng.standard_normal(k).astype(np.float32)
        y0 = rng.standard_normal(nrows).astype(np.float32)
        o = run_one(h, NAME,
                    [{"buffer": "w"}, {"buffer": "g"}, {"buffer": "u"},
                     {"buffer": "y"},
                     {"scalar": {"dtype": "u32", "value": nrows}},
                     {"scalar": {"dtype": "u32", "value": k}},
                     {"scalar": {"dtype": "u32", "value": 64}}],
                    {"w": {"dtype": "f32", "values": w.reshape(-1).tolist()},
                     "g": {"dtype": "f32", "values": g.tolist()},
                     "u": {"dtype": "f32", "values": u.tolist()},
                     "y": {"dtype": "f32", "values": y0.tolist()}},
                    grid=nrows * 64, workgroup=64)
        got = np.array(o["y"], np.float32)
        act = (g / (1.0 + np.exp(-g))) * u
        ref = (y0 + w @ act).astype(np.float32)
        rel = float(np.abs(got - ref).max() / max(1e-9, float(np.abs(ref).max())))
        good = rel < 1e-4          # silu 走 rcp 近似，容差放宽到 1e-4
        ok = ok and good
        print(f"nrows={nrows} k={k}: max_rel={rel:.2e} {'✔' if good else '✘'}")
    return 0 if ok else 1


KERNELS = [(NAME, gen_asm, ARGS, KERNARG_SIZE)]


def main() -> int:
    rc = selftest()
    print("gemv_f32_gated_acc_k vs numpy:", "一致 ✔" if rc == 0 else "不一致 ✘")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
