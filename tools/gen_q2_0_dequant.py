#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q2_0 → f32 解码内核（gfx926 生成器 + 真权重自检）。

    python3 tools/gen_q2_0_dequant.py

    __global__ void q2_0_dequant_k(const uint8_t* w, float* y,
                                   uint32_t nblocks, uint32_t group_size)

Q2_0：64 元素 / 18 字节（f16 d + u8 qs[16]），码 = 2 bit，4 码/字节低位在前，
值 = (码-1)·d，码本 = {-1,0,1,2}。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from iq_dequant import dequant_q2_0  # noqa: E402
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "q2_0_dequant_k"
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
        "v_mul_lo_u32 v1, v1, s21",
        "v_add_u32_e32 v1, v0, v1",
        "v_cmp_gt_u32_e32 vcc, s20, v1",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        "v_mul_lo_u32 v2, v1, 18",
        "v_mov_b32_e32 v4, s16",
        "v_mov_b32_e32 v5, s17",
        "v_add_co_u32_e32 v4, vcc, v4, v2",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "v_lshlrev_b32_e32 v6, 8, v1",
        "v_mov_b32_e32 v7, s19",
        "v_add_co_u32_e32 v6, vcc, s18, v6",
        "v_addc_co_u32_e32 v7, vcc, v7, v3, vcc",
        "global_load_ushort v8, v[4:5], off",
    ]
    for b in range(16):
        L.append(f"global_load_ubyte v{48 + b}, v[4:5], off offset:{2 + b}")
    L.append("s_waitcnt vmcnt(0)")
    L.append("v_cvt_f32_f16_e32 v8, v8")
    for b in range(16):
        for k in range(4):
            if k == 0:
                L.append(f"v_and_b32_e32 v{24 + k}, 3, v{48 + b}")
            else:
                L.append(f"v_lshrrev_b32_e32 v{24 + k}, {2 * k}, v{48 + b}")
                L.append(f"v_and_b32_e32 v{24 + k}, 3, v{24 + k}")
            # 码 0/1/2/3 → 值 -1/0/1/2；先转 f32 再减 1 再乘 d
            L.append(f"v_cvt_f32_u32_e32 v{24 + k}, v{24 + k}")
            L.append(f"v_add_f32_e32 v{24 + k}, -1.0, v{24 + k}")
            L.append(f"v_mul_f32_e32 v{24 + k}, v8, v{24 + k}")
        L.append(f"global_store_dwordx4 v[6:7], v[24:27], off offset:{16 * b}")
    L += ["L_end:", "s_or_b64 exec, exec, s[0:1]", "s_endpgm"]
    return "\n".join(L) + "\n"


def main() -> int:
    idx = json.loads((ROOT / "build/flashnext-iq3s-shard1.index.json").read_text(encoding="utf-8"))
    t = next(x for x in idx["tensors"] if x["type"] == "Q2_0" and x["shape"][0] % 64 == 0)
    nblocks = 2048
    src = pathlib.Path(idx["sources"][t["source"]]["path"])
    with open(src, "rb") as f:
        f.seek(t["offset"])
        raw = f.read(nblocks * 18)
    ref = dequant_q2_0(raw)

    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="q2_0")
    out = run_one(hsaco, NAME,
                  [{"buffer": "w"}, {"buffer": "y"},
                   {"scalar": {"dtype": "u32", "value": nblocks}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"w": {"dtype": "u8", "values": list(raw)},
                   "y": {"dtype": "f32", "values": [0.0] * (nblocks * 64)}},
                  grid=nblocks, workgroup=64)
    got = np.array(out["y"], dtype=np.float32)
    diff = np.abs(got - ref)
    print(f"{t['name']}: nblocks={nblocks} max_abs={diff.max():.3e}")
    ok = diff.max() == 0.0
    print("GPU Q2_0 vs 参考：", "bit-exact ✔" if ok else "不一致 ✘")
    if not ok:
        for i in np.flatnonzero(diff > 0)[:6]:
            print(f"  y[{i}] gpu={got[i]:.8g} ref={ref[i]:.8g}")
    print("HSACO:", hsaco)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
