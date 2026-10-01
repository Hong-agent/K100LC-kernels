#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IQ3_XXS → f32 解码内核（gfx926 生成器 + 真权重自检）。

    python3 tools/gen_iq3xxs_dequant.py

块：256 元素 / 98 字节 = f16 d + u8 qs[64] + u32 scales[8]（偏移 66，2 字节对齐）。
每 32 元素（ib）：`db = d·(0.5+(aux>>28))·0.5`；4 个子组各 8 元素 =
两个 4 元素网格值（256×u32 表）+ 7 bit 符号（ksigns 128 项表）。
网格表与 ksigns 表作为小缓冲传入（1 KB + 128 B，常驻 L1）。
"""
from __future__ import annotations

import json
import pathlib
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from iq_dequant import dequant_iq3_xxs  # noqa: E402
from iq_tables import iq3xxs_grid, ksigns_iq2xs  # noqa: E402
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "iq3xxs_dequant_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16), buffer_arg(24),
        scalar_arg(32, 4), scalar_arg(36, 4)]
KERNARG_SIZE = 40


def _value(L: list[str], g: str, bit: int, out: str) -> None:
    """g（4 个幅值字节）的第 (bit&3) 个字节 × sg 的 bit 号符号 → out（f32）。"""
    byte = bit & 3
    L += [
        f"v_lshrrev_b32_e32 v35, {8 * byte}, {g}",
        "v_and_b32_e32 v35, 0xff, v35",
        "v_cvt_f32_u32_e32 v35, v35",
        f"v_lshrrev_b32_e32 v36, {bit}, v28",
        "v_and_b32_e32 v36, 1, v36",
        "v_cvt_f32_u32_e32 v36, v36",
        "v_fma_f32 v36, v36, -2.0, 1.0",
        "v_mul_f32_e32 v35, v35, v36",
        f"v_mul_f32_e32 {out}, v14, v35",
    ]


def gen_asm() -> str:
    L = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",   # w
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",   # y
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",  # grid
        "s_load_dwordx2 s[22:23], s[4:5], 0x18",  # ksigns
        "s_load_dword s24, s[4:5], 0x20",         # nblocks
        "s_load_dword s25, s[4:5], 0x24",         # group_size
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v3, 0",
        "v_mov_b32_e32 v1, s6",
        "v_mul_lo_u32 v1, v1, s25",
        "v_add_u32_e32 v1, v0, v1",
        "v_cmp_gt_u32_e32 vcc, s24, v1",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        "v_mov_b32_e32 v9, s20",     # grid lo
        "v_mov_b32_e32 v10, s21",    # grid hi
        "v_mov_b32_e32 v11, s22",    # ksigns lo
        "v_mov_b32_e32 v12, s23",    # ksigns hi
        "v_mov_b32_e32 v80, 98",     # 块字节数
        "v_mul_lo_u32 v2, v1, v80",
        "v_mov_b32_e32 v4, s16",
        "v_mov_b32_e32 v5, s17",
        "v_add_co_u32_e32 v4, vcc, v4, v2",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        "v_lshlrev_b32_e32 v6, 10, v1",
        "v_mov_b32_e32 v7, s19",
        "v_add_co_u32_e32 v6, vcc, s18, v6",
        "v_addc_co_u32_e32 v7, vcc, v7, v3, vcc",
        "global_load_ushort v8, v[4:5], off",
        "s_waitcnt vmcnt(0)",
        "v_cvt_f32_f16_e32 v8, v8",
    ]
    for ib in range(8):
        # aux 的 4 字节（global_load_ushort 没有 offset 形式）
        for k in range(4):
            L.append(f"global_load_ubyte v{15 + k}, v[4:5], off offset:{66 + 4 * ib + k}")
        # 本组的 8 个 qs 字节 → v20..v27
        for k in range(8):
            L.append(f"global_load_ubyte v{20 + k}, v[4:5], off offset:{2 + 8 * ib + k}")
        L.append("s_waitcnt vmcnt(0)")
        L += [
            "v_lshlrev_b32_e32 v16, 8, v16",
            "v_or_b32_e32 v15, v15, v16",
            "v_lshlrev_b32_e32 v18, 8, v18",
            "v_or_b32_e32 v17, v17, v18",
            "v_lshlrev_b32_e32 v18, 16, v17",
            "v_or_b32_e32 v13, v15, v18",       # aux
            "v_lshrrev_b32_e32 v15, 28, v13",   # top
            "v_cvt_f32_u32_e32 v15, v15",
            "v_add_f32_e32 v15, 0.5, v15",
            "v_mul_f32_e32 v15, 0.5, v15",
            "v_mul_f32_e32 v14, v8, v15",       # db
        ]
        for l in range(4):
            # sgn_code = (aux >> 7l) & 127
            if 7 * l == 0:
                L.append("v_and_b32_e32 v17, 127, v13")
            else:
                L.append(f"v_lshrrev_b32_e32 v17, {7 * l}, v13")
                L.append("v_and_b32_e32 v17, 127, v17")
            L += [
                "v_mov_b32_e32 v29, v11",
                "v_mov_b32_e32 v30, v12",
                "v_add_co_u32_e32 v29, vcc, v29, v17",
                "v_addc_co_u32_e32 v30, vcc, v30, v3, vcc",
                "global_load_ubyte v28, v[29:30], off",
                "s_waitcnt vmcnt(0)",
            ]
            for gi, greg in ((0, "v33"), (1, "v34")):
                qreg = 20 + 2 * l + gi
                L += [
                    f"v_lshlrev_b32_e32 v17, 2, v{qreg}",
                    "v_mov_b32_e32 v31, v9",
                    "v_mov_b32_e32 v32, v10",
                    "v_add_co_u32_e32 v31, vcc, v31, v17",
                    "v_addc_co_u32_e32 v32, vcc, v32, v3, vcc",
                    f"global_load_dword {greg}, v[31:32], off",
                    "s_waitcnt vmcnt(0)",
                ]
                base_bit = 0 if gi == 0 else 4
                for j in range(4):
                    _value(L, greg, base_bit + j, f"v{40 + j}")
                elem = ib * 32 + 8 * l + 4 * gi
                L.append(f"global_store_dwordx4 v[6:7], v[40:43], off offset:{4 * elem}")
    L += ["L_end:", "s_or_b64 exec, exec, s[0:1]", "s_endpgm"]
    return "\n".join(L) + "\n"


def main() -> int:
    from gguf_sample import load_index
    idx = load_index()
    t = next(x for x in idx["tensors"] if x["type"] == "IQ3_XXS")
    nblocks = 256
    src = pathlib.Path(idx["sources"][t["source"]]["path"])
    with open(src, "rb") as f:
        f.seek(t["offset"])
        raw = f.read(nblocks * 98)
    ref = dequant_iq3_xxs(raw)
    grid = struct.pack("<%dI" % len(iq3xxs_grid), *[int(v) for v in iq3xxs_grid])
    ksigns = bytes(int(v) for v in ksigns_iq2xs)

    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="iq3xxs")
    out = run_one(hsaco, NAME,
                  [{"buffer": "w"}, {"buffer": "y"}, {"buffer": "grid"}, {"buffer": "ks"},
                   {"scalar": {"dtype": "u32", "value": nblocks}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"w": {"dtype": "u8", "values": list(raw)},
                   "y": {"dtype": "f32", "values": [0.0] * (nblocks * 256)},
                   "grid": {"dtype": "u8", "values": list(grid)},
                   "ks": {"dtype": "u8", "values": list(ksigns)}},
                  grid=nblocks, workgroup=64)
    got = np.array(out["y"], dtype=np.float32)
    diff = np.abs(got - ref)
    print(f"{t['name']}: nblocks={nblocks} max_abs={diff.max():.3e}")
    ok = diff.max() == 0.0
    print("GPU IQ3_XXS vs 参考：", "bit-exact ✔" if ok else "不一致 ✘")
    if not ok:
        for i in np.flatnonzero(diff > 0)[:6]:
            print(f"  y[{i}] gpu={got[i]:.8g} ref={ref[i]:.8g}")
    print("HSACO:", hsaco)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
