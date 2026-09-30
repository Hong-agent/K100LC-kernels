#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q4_0 → f32 GPU 解码内核 + 自检。

签名：q4_0_dequant_k(const uint8_t* w, float* y, uint32_t nblocks, uint32_t group_size)
块 = f16 d + 16 字节；低半字节 → y[0..15]，高半字节 → y[16..31]，值 = d*(q-8)。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "q4_0_dequant_k"
ARGS = [buffer_arg(0), buffer_arg(8), scalar_arg(16, 4), scalar_arg(20, 4)]
KERNARG_SIZE = 24


def gen_asm() -> str:
    L = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",
        "s_load_dword s20, s[4:5], 0x10",
        "s_load_dword s21, s[4:5], 0x14",
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v3, 0",
        "v_mov_b32_e32 v1, s6",
        "v_lshlrev_b32_e32 v1, 6, v1",
        "v_add_u32_e32 v1, v1, v0",
        "v_cmp_gt_u32_e32 vcc, s20, v1",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        # wptr = w + id*18 + 2
        "v_mul_lo_u32 v2, v1, 18",
        "v_add_u32_e32 v2, 2, v2",
        "v_mov_b32_e32 v5, s17",
        "v_add_co_u32_e32 v4, vcc, s16, v2",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        # ybase = y + id*128
        "v_lshlrev_b32_e32 v2, 7, v1",
        "v_mov_b32_e32 v7, s19",
        "v_add_co_u32_e32 v6, vcc, s18, v2",
        "v_addc_co_u32_e32 v7, vcc, v7, v3, vcc",
        # d
        "v_mov_b32_e32 v17, -1",
        "v_add_co_u32_e32 v8, vcc, 0xfffffffe, v4",
        "v_addc_co_u32_e32 v9, vcc, v17, v5, vcc",
        "global_load_ushort v10, v[8:9], off",
        "s_waitcnt vmcnt(0)",
        "v_cvt_f32_f16_e32 v10, v10",
        "v_mov_b32_e32 v12, 8",
        "s_mov_b32 s22, 0",
        "L_loop:",
        "global_load_ubyte v11, v[4:5], off",
        "s_waitcnt vmcnt(0)",
        # 低半字节
        "v_and_b32_e32 v13, 0x0f, v11",
        "v_sub_u32_e32 v13, v13, v12",
        "v_cvt_f32_i32_e32 v13, v13",
        "v_mul_f32_e32 v13, v13, v10",
        "global_store_dword v[6:7], v13, off",
        # 高半字节 -> y+64
        "v_lshrrev_b32_e32 v14, 4, v11",
        "v_sub_u32_e32 v14, v14, v12",
        "v_cvt_f32_i32_e32 v14, v14",
        "v_mul_f32_e32 v14, v14, v10",
        "v_add_co_u32_e32 v15, vcc, 64, v6",
        "v_addc_co_u32_e32 v16, vcc, 0, v7, vcc",
        "global_store_dword v[15:16], v14, off",
        # 前进
        "v_add_co_u32_e32 v4, vcc, 1, v4",
        "v_addc_co_u32_e32 v5, vcc, 0, v5, vcc",
        "v_add_co_u32_e32 v6, vcc, 4, v6",
        "v_addc_co_u32_e32 v7, vcc, 0, v7, vcc",
        "s_add_i32 s22, s22, 1",
        "s_cmp_lt_u32 s22, 16",
        "s_cbranch_scc1 L_loop",
        "L_end:",
        "s_or_b64 exec, exec, s[0:1]",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


def main() -> int:
    nblocks = 256
    rng = np.random.default_rng(5)
    raw = bytearray(rng.integers(0, 256, size=nblocks * 18, dtype=np.uint8).tobytes())
    for i in range(nblocks):
        raw[i * 18:i * 18 + 2] = np.float16(0.01).tobytes()
    raw = np.frombuffer(bytes(raw), dtype=np.uint8)
    ref = np.empty(nblocks * 32, dtype=np.float32)
    for i in range(nblocks):
        b = raw[i * 18:(i + 1) * 18]
        d = np.frombuffer(b[:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
        q = b[2:18].astype(np.int32)
        ref[i * 32:i * 32 + 16] = (q & 0xF) - 8
        ref[i * 32 + 16:i * 32 + 32] = (q >> 4) - 8
        ref[i * 32:(i + 1) * 32] *= d
    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="q4_0")
    out = run_one(hsaco, NAME,
                  [{"buffer": "w"}, {"buffer": "y"},
                   {"scalar": {"dtype": "u32", "value": nblocks}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"w": {"dtype": "u8", "values": list(raw.tobytes())},
                   "y": {"dtype": "f32", "values": [0.0] * (nblocks * 32)}},
                  grid=nblocks, workgroup=64)
    got = np.array(out["y"], dtype=np.float32)
    d = np.abs(got - ref)
    print(f"Q4_0 nblocks={nblocks} max_abs={d.max():.3e}")
    ok = d.max() == 0.0
    print("GPU Q4_0 vs 参考：", "bit-exact ✔" if ok else "不一致 ✘")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
