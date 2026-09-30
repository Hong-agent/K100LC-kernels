#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用 f32 GEMV（gfx926 `.s` 生成器）+ 自检。

    python3 tools/gen_gemv_f32.py

    __global__ void gemv_f32_k(const float* w, const float* x, float* y,
                               uint32_t nrows, uint32_t nchunks, uint32_t group_size)

`y[n] = Σ_k w[n,k]·x[k]`，w 行主序 `[nrows, k]` f32，按 4 个元素一轮展开。
配合各编码的 `*_dequant_k`（输出 f32 scratch）使用。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "gemv_f32_k"
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
        "s_load_dword s23, s[4:5], 0x1c",         # nchunks
        "s_load_dword s24, s[4:5], 0x20",         # group_size
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v3, 0",
        "v_mov_b32_e32 v1, s6",
        "v_mul_lo_u32 v1, v1, s24",
        "v_add_u32_e32 v1, v0, v1",
        "v_cmp_gt_u32_e32 vcc, s22, v1",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        # wrow = w + row*nchunks*16
        "v_mov_b32_e32 v2, s23",
        "v_lshlrev_b32_e32 v2, 4, v2",
        "v_mul_lo_u32 v2, v1, v2",
        "v_mov_b32_e32 v4, s16",
        "v_mov_b32_e32 v5, s17",
        "v_add_co_u32_e32 v4, vcc, v4, v2",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "v_mov_b32_e32 v6, s18",
        "v_mov_b32_e32 v7, s19",
        "v_lshlrev_b32_e32 v44, 2, v1",
        "v_mov_b32_e32 v45, s21",
        "v_add_co_u32_e32 v44, vcc, s20, v44",
        "v_addc_co_u32_e32 v45, vcc, v45, v3, vcc",
        "v_mov_b32_e32 v46, 16",
        "s_mov_b32 s25, 0",
        "v_mov_b32_e32 v20, 0",
        "v_mov_b32_e32 v21, 0",
        "v_mov_b32_e32 v22, 0",
        "v_mov_b32_e32 v23, 0",
        "L_loop:",
        "global_load_dwordx4 v[24:27], v[4:5], off",
        "global_load_dwordx4 v[28:31], v[6:7], off",
        "s_waitcnt vmcnt(0)",
        "v_fma_f32 v20, v24, v28, v20",
        "v_fma_f32 v21, v25, v29, v21",
        "v_fma_f32 v22, v26, v30, v22",
        "v_fma_f32 v23, v27, v31, v23",
        "v_add_co_u32_e32 v4, vcc, v4, v46",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "v_add_co_u32_e32 v6, vcc, v6, v46",
        "v_addc_co_u32_e32 v7, vcc, v7, v3, vcc",
        "s_add_i32 s25, s25, 1",
        "s_cmp_lt_u32 s25, s23",
        "s_cbranch_scc1 L_loop",
        "v_add_f32_e32 v20, v20, v21",
        "v_add_f32_e32 v22, v22, v23",
        "v_add_f32_e32 v20, v20, v22",
        "global_store_dword v[44:45], v20, off",
        "L_end:",
        "s_or_b64 exec, exec, s[0:1]",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


def main() -> int:
    nrows, k = 256, 2560
    nchunks = k // 4
    rng = np.random.default_rng(5)
    w = rng.standard_normal((nrows, k)).astype(np.float32)
    x = rng.standard_normal(k).astype(np.float32)
    ref = (w @ x).astype(np.float32)

    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="f32")
    out = run_one(hsaco, NAME,
                  [{"buffer": "w"}, {"buffer": "x"}, {"buffer": "y"},
                   {"scalar": {"dtype": "u32", "value": nrows}},
                   {"scalar": {"dtype": "u32", "value": nchunks}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"w": {"dtype": "f32", "values": w.reshape(-1).tolist()},
                   "x": {"dtype": "f32", "values": x.tolist()},
                   "y": {"dtype": "f32", "values": [0.0] * nrows}},
                  grid=nrows, workgroup=64)
    got = np.array(out["y"], dtype=np.float32)
    diff = np.abs(got - ref)
    print(f"nrows={nrows} k={k}  max_abs={diff.max():.3e}")
    ok = diff.max() < 1e-2 * max(1.0, float(np.abs(ref).max()))
    print("f32 GEMV vs numpy：", "一致 ✔" if ok else "不一致 ✘")
    if not ok:
        for i in np.argsort(-diff)[:5]:
            print(f"  y[{i}] gpu={got[i]:.6g} ref={ref[i]:.6g}")
    print("HSACO:", hsaco)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
