#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用 int8 权重 GEMV（gfx926 `.s` 生成器）+ 自检。

    python3 tools/gen_gemv_i8.py

内核签名：

    __global__ void gemv_i8_k(const int8_t* w, const float* ws, const float* x,
                              float* y, uint32_t nrows, uint32_t k,
                              uint32_t group_size)

布局：`w` 行主序 `[nrows, k]` int8，`ws` 行主序 `[nrows, k/32]` f32
（每 32 个权重一个尺度），x 是 f32 `[k]`。

    y[n] = Σ_s ws[n,s] · Σ_{j<32} w[n, 32s+j] · x[32s+j]

这是「原生编码 → int8 scratch → 通用 GEMV」路线的后半段：每种 i-quant 只需
写一个 `*_to_i8` 解码内核，GEMV/尺度折叠全共用。一个 work-item 算一行。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "gemv_i8_k"
# 注意：行距/组数等由主机算好传入，不在内核里做标量乘加（见 docs/KERNEL-DEV.md
# 「s_mul_i32 个别目标寄存器编码异常」）。
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16), buffer_arg(24),
        scalar_arg(32, 4), scalar_arg(36, 4), scalar_arg(40, 4),
        scalar_arg(44, 4), scalar_arg(48, 4)]
KERNARG_SIZE = 56


def gen_asm() -> str:
    L: list[str] = []
    em = L.append

    em(".text")
    em(f"k_{NAME}:")
    em("s_load_dwordx2 s[16:17], s[4:5], 0x0")    # w
    em("s_load_dwordx2 s[18:19], s[4:5], 0x8")    # ws
    em("s_load_dwordx2 s[20:21], s[4:5], 0x10")   # x
    em("s_load_dwordx2 s[22:23], s[4:5], 0x18")   # y
    em("s_load_dword s24, s[4:5], 0x20")          # nrows
    em("s_load_dword s25, s[4:5], 0x24")          # ngroups
    em("s_load_dword s26, s[4:5], 0x28")          # rowbytes
    em("s_load_dword s27, s[4:5], 0x2c")          # ws_stride
    em("s_load_dword s28, s[4:5], 0x30")          # group_size
    em("s_waitcnt lgkmcnt(0)")
    em("v_mov_b32_e32 v3, 0")
    em("v_mov_b32_e32 v1, s6")
    em("v_mul_lo_u32 v1, v1, s28")
    em("v_add_u32_e32 v1, v0, v1")                # row
    em("v_cmp_gt_u32_e32 vcc, s24, v1")
    em("s_and_saveexec_b64 s[0:1], vcc")
    em("s_cbranch_execz L_end")
    # wrow = w + row*rowbytes
    em("v_mov_b32_e32 v2, s26")
    em("v_mul_lo_u32 v2, v1, v2")
    em("v_mov_b32_e32 v4, s16")
    em("v_mov_b32_e32 v5, s17")
    em("v_add_co_u32_e32 v4, vcc, v4, v2")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")
    # wsrow = ws + row*ws_stride
    em("v_mov_b32_e32 v2, s27")
    em("v_mul_lo_u32 v2, v1, v2")
    em("v_mov_b32_e32 v6, s18")
    em("v_mov_b32_e32 v7, s19")
    em("v_add_co_u32_e32 v6, vcc, v6, v2")
    em("v_addc_co_u32_e32 v7, vcc, v7, v3, vcc")
    # x 固定基址
    em("v_mov_b32_e32 v8, s20")
    em("v_mov_b32_e32 v9, s21")
    # y + row*4
    em("v_lshlrev_b32_e32 v44, 2, v1")
    em("v_mov_b32_e32 v45, s23")
    em("v_add_co_u32_e32 v44, vcc, s22, v44")
    em("v_addc_co_u32_e32 v45, vcc, v45, v3, vcc")
    em("v_mov_b32_e32 v46, 32")
    em("v_mov_b32_e32 v47, 4")
    em("v_mov_b32_e32 v48, 128")
    em("s_mov_b32 s29, 0")
    em("v_mov_b32_e32 v49, 4")
    em("v_mov_b32_e32 v102, 0")

    em("L_loop:")
    # 32 个权重 int8 → v23..v30
    for q in range(8):
        off = 16 * (q // 4)
        em(f"global_load_dword v{23 + q}, v[4:5], off offset:{off + 4 * (q % 4)}")
    # 32 个 x f32 → v64..v95
    for q in range(8):
        em(f"global_load_dwordx4 v[{64 + 4 * q}:{67 + 4 * q}], v[8:9], off offset:{16 * q}")
    em("s_waitcnt vmcnt(0)")
    em("v_mov_b32_e32 v96, 0")
    em("v_mov_b32_e32 v97, 0")
    em("v_mov_b32_e32 v98, 0")
    em("v_mov_b32_e32 v99, 0")
    for i in range(32):
        wreg = 23 + i // 4
        sh = 8 * (i % 4)
        em(f"v_lshrrev_b32_e32 v100, {sh}, v{wreg}")
        em("v_lshlrev_b32_e32 v100, 24, v100")
        em("v_ashrrev_i32_e32 v100, 24, v100")
        em("v_cvt_f32_i32_e32 v100, v100")
        em(f"v_fma_f32 v{96 + (i % 4)}, v100, v{64 + i}, v{96 + (i % 4)}")
    em("v_add_f32_e32 v96, v96, v97")
    em("v_add_f32_e32 v98, v98, v99")
    em("v_add_f32_e32 v96, v96, v98")
    em("global_load_dword v101, v[6:7], off")
    em("s_waitcnt vmcnt(0)")
    em("v_mul_f32_e32 v96, v101, v96")
    em("v_add_f32_e32 v102, v102, v96")
    # 前进：w += 32，x += 128B，ws += 4
    em("v_add_co_u32_e32 v4, vcc, v4, v46")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")
    em("v_add_co_u32_e32 v8, vcc, v8, v48")
    em("v_addc_co_u32_e32 v9, vcc, v9, v3, vcc")
    em("v_add_co_u32_e32 v6, vcc, v6, v49")
    em("v_addc_co_u32_e32 v7, vcc, v7, v3, vcc")
    em("s_add_i32 s29, s29, 1")
    em("s_cmp_lt_u32 s29, s25")
    em("s_cbranch_scc1 L_loop")
    em("global_store_dword v[44:45], v102, off")
    em("L_end:")
    em("s_or_b64 exec, exec, s[0:1]")
    em("s_endpgm")
    return "\n".join(L) + "\n"


def main() -> int:
    nrows, k = 256, 640
    ngroups = k // 32
    rng = np.random.default_rng(11)
    w = rng.integers(-127, 128, size=(nrows, k), dtype=np.int8)
    ws = (rng.random((nrows, ngroups)).astype(np.float32) * 0.02 + 0.001)
    x = rng.standard_normal(k).astype(np.float32)
    ref = np.einsum("nsk,sk->ns", w.astype(np.float32).reshape(nrows, ngroups, 32),
                    x.reshape(ngroups, 32))
    ref = (ref * ws).sum(axis=1).astype(np.float32)

    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="i8")
    out = run_one(hsaco, NAME,
                  [{"buffer": "w"}, {"buffer": "ws"}, {"buffer": "x"}, {"buffer": "y"},
                   {"scalar": {"dtype": "u32", "value": nrows}},
                   {"scalar": {"dtype": "u32", "value": ngroups}},
                   {"scalar": {"dtype": "u32", "value": k}},
                   {"scalar": {"dtype": "u32", "value": ngroups * 4}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"w": {"dtype": "i8", "values": w.reshape(-1).tolist()},
                   "ws": {"dtype": "f32", "values": ws.reshape(-1).tolist()},
                   "x": {"dtype": "f32", "values": x.tolist()},
                   "y": {"dtype": "f32", "values": [0.0] * nrows}},
                  grid=nrows, workgroup=64)
    got = np.array(out["y"], dtype=np.float32)
    diff = np.abs(got - ref)
    print(f"nrows={nrows} k={k}  max_abs={diff.max():.3e}  max_rel="
          f"{(diff / np.maximum(np.abs(ref), 1e-6)).max():.3e}")
    ok = diff.max() < 1e-2 * max(1.0, float(np.abs(ref).max()))
    print("int8 GEMV vs numpy：", "一致 ✔" if ok else "不一致 ✘")
    if not ok:
        for i in np.argsort(-diff)[:5]:
            print(f"  y[{i}] gpu={got[i]:.6g} ref={ref[i]:.6g}")
    print("HSACO:", hsaco)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
