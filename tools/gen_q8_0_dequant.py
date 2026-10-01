#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q8_0 → f32 GPU 解码内核 + 真权重自检。

签名：q8_0_dequant_k(const uint8_t* w, float* y, uint32_t nblocks, uint32_t group_size)
一个 work-item 解一个 34 字节块（f16 d + 32×int8）。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from iq_dequant import dequant_q8_0_fast  # noqa: E402
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "q8_0_dequant_k"
ARGS = [buffer_arg(0), buffer_arg(8), scalar_arg(16, 4), scalar_arg(20, 4)]
KERNARG_SIZE = 24


def gen_asm() -> str:
    L = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",   # w
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",   # y
        "s_load_dword s20, s[4:5], 0x10",         # nblocks
        "s_load_dword s21, s[4:5], 0x14",         # group_size
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v3, 0",
        "v_mov_b32_e32 v1, s6",
        "v_lshlrev_b32_e32 v1, 6, v1",
        "v_add_u32_e32 v1, v1, v0",               # block id
        "v_cmp_gt_u32_e32 vcc, s20, v1",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        # wptr = w + id*34 + 2
        "v_mul_lo_u32 v2, v1, 34",
        "v_add_u32_e32 v2, 2, v2",
        "v_mov_b32_e32 v5, s17",
        "v_add_co_u32_e32 v4, vcc, s16, v2",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        # yptr = y + id*256*4
        "v_lshlrev_b32_e32 v2, 7, v1",           # 32 floats/block = 128B
        "v_mov_b32_e32 v7, s19",
        "v_add_co_u32_e32 v6, vcc, s18, v2",
        "v_addc_co_u32_e32 v7, vcc, v7, v3, vcc",
        # d = f16 at w-2
        "v_mov_b32_e32 v17, -1",
        "v_add_co_u32_e32 v8, vcc, 0xfffffffe, v4",  # wptr-2 低位
        "v_addc_co_u32_e32 v9, vcc, v17, v5, vcc",
        "global_load_ushort v10, v[8:9], off",
        "s_waitcnt vmcnt(0)",
        "v_cvt_f32_f16_e32 v10, v10",
        "s_mov_b32 s22, 0",
        "L_loop:",
        "global_load_ubyte v11, v[4:5], off",
        "s_waitcnt vmcnt(0)",
        "v_lshlrev_b32_e32 v11, 24, v11",
        "v_ashrrev_i32_e32 v11, 24, v11",
        "v_cvt_f32_i32_e32 v11, v11",
        "v_mul_f32_e32 v11, v11, v10",
        "global_store_dword v[6:7], v11, off",
        "v_add_co_u32_e32 v4, vcc, 1, v4",
        "v_addc_co_u32_e32 v5, vcc, 0, v5, vcc",
        "v_add_co_u32_e32 v6, vcc, 4, v6",
        "v_addc_co_u32_e32 v7, vcc, 0, v7, vcc",
        "s_add_i32 s22, s22, 1",
        "s_cmp_lt_u32 s22, 32",
        "s_cbranch_scc1 L_loop",
        "L_end:",
        "s_or_b64 exec, exec, s[0:1]",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


def main() -> int:
    try:
        from gguf_sample import load_index
        idx = load_index()
        t = next((x for x in idx["tensors"] if x["type"] == "Q8_0"), None)
    except (OSError, SystemExit):
        idx, t = None, None
    if t is None:
        nblocks = 256
        rng = np.random.default_rng(4)
        r = bytearray(rng.integers(0, 256, size=nblocks * 34, dtype=np.uint8).tobytes())
        for i in range(nblocks):
            r[i * 34:i * 34 + 2] = np.float16(0.01).tobytes()
        raw = bytes(r)
    else:
        nblocks = 256
        src = pathlib.Path(idx["sources"][t["source"]]["path"])
        with open(src, "rb") as f:
            f.seek(t["offset"])
            raw = f.read(nblocks * 34)
    ref = dequant_q8_0_fast(raw)
    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="q8_0")
    out = run_one(hsaco, NAME,
                  [{"buffer": "w"}, {"buffer": "y"},
                   {"scalar": {"dtype": "u32", "value": nblocks}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"w": {"dtype": "u8", "values": list(raw)},
                   "y": {"dtype": "f32", "values": [0.0] * (nblocks * 32)}},
                  grid=nblocks, workgroup=64)
    got = np.array(out["y"], dtype=np.float32)
    d = np.abs(got - ref)
    print(f"Q8_0 nblocks={nblocks} max_abs={d.max():.3e}")
    ok = d.max() == 0.0
    print("GPU Q8_0 vs 参考：", "bit-exact ✔" if ok else "不一致 ✘")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
