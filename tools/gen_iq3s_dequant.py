#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IQ3_S → f32 解码内核（gfx926 生成器 + 真权重自检）。

    python3 tools/gen_iq3s_dequant.py

块：256 元素 / 110 字节 = f16 d + qs[64] + qh[8] + signs[32] + scales[4]。
每 64 元素（pair）：两档尺度 `d·(1+2·sc)`；两个 half 各 4 组、每组 8 元素 =
512×u32 网格（idx 带 qh 高位）+ 一个符号字节。
"""
from __future__ import annotations

import json
import pathlib
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from iq_dequant import dequant_iq3_s  # noqa: E402
from iq_tables import iq3s_grid  # noqa: E402
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "iq3s_dequant_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
        scalar_arg(24, 4), scalar_arg(28, 4)]
KERNARG_SIZE = 32


def _value(L: list[str], g: str, bit: int, scale: str, out: str) -> None:
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
        f"v_mul_f32_e32 {out}, {scale}, v35",
    ]


def gen_asm() -> str:
    L = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",   # w
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",   # y
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",  # grid
        "s_load_dword s24, s[4:5], 0x18",         # nblocks
        "s_load_dword s25, s[4:5], 0x1c",         # group_size
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v3, 0",
        "v_mov_b32_e32 v1, s6",
        "v_mul_lo_u32 v1, v1, s25",
        "v_add_u32_e32 v1, v0, v1",
        "v_cmp_gt_u32_e32 vcc, s24, v1",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        "v_mov_b32_e32 v9, s20",
        "v_mov_b32_e32 v10, s21",
        "v_mov_b32_e32 v80, 110",
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
    for pair in range(4):
        L += [
            f"global_load_ubyte v15, v[4:5], off offset:{106 + pair}",  # scales[pair]
        ]
        L.append("s_waitcnt vmcnt(0)")
        # db0 = d*(1+2*(sc&0xF))
        L += [
            "v_and_b32_e32 v16, 0xf, v15",
            "v_cvt_f32_u32_e32 v16, v16",
            "v_fma_f32 v16, v16, 2.0, 1.0",
            "v_mul_f32_e32 v44, v8, v16",
        ]
        # db1 = d*(1+2*(sc>>4))
        L += [
            "v_lshrrev_b32_e32 v17, 4, v15",
            "v_cvt_f32_u32_e32 v17, v17",
            "v_fma_f32 v17, v17, 2.0, 1.0",
            "v_mul_f32_e32 v45, v8, v17",
        ]
        for half in range(2):
            qhb_off = 66 + 2 * pair + half
            L += [f"global_load_ubyte v13, v[4:5], off offset:{qhb_off}"]
            for l in range(4):
                q0 = 20 + 2 * l
                q1 = 21 + 2 * l
                L += [
                    f"global_load_ubyte v{q0}, v[4:5], off offset:{2 + 16 * pair + 8 * half + 2 * l}",
                    f"global_load_ubyte v{q1}, v[4:5], off offset:{3 + 16 * pair + 8 * half + 2 * l}",
                    f"global_load_ubyte v28, v[4:5], off offset:{74 + 8 * pair + 4 * half + l}",
                ]
                L.append("s_waitcnt vmcnt(0)")
                scale = "v44" if half == 0 else "v45"
                elem_base = pair * 64 + half * 32 + 8 * l
                for gi, qreg in ((0, q0), (1, q1)):
                    sh = (8 - 2 * l) if gi == 0 else (7 - 2 * l)
                    L += [
                        f"v_lshlrev_b32_e32 v16, {sh}, v13",
                        "v_and_b32_e32 v16, 0x100, v16",
                        f"v_or_b32_e32 v14, v{qreg}, v16",
                        "v_lshlrev_b32_e32 v17, 2, v14",
                        "v_mov_b32_e32 v29, v9",
                        "v_mov_b32_e32 v30, v10",
                        "v_add_co_u32_e32 v29, vcc, v29, v17",
                        "v_addc_co_u32_e32 v30, vcc, v30, v3, vcc",
                        f"global_load_dword v33, v[29:30], off",
                    ]
                    L.append("s_waitcnt vmcnt(0)")
                    for j in range(4):
                        _value(L, "v33", j + (0 if gi == 0 else 4), scale, f"v{40 + j}")
                    elem = elem_base + 4 * gi
                    L.append(f"global_store_dwordx4 v[6:7], v[40:43], off offset:{4 * elem}")
    L += ["L_end:", "s_or_b64 exec, exec, s[0:1]", "s_endpgm"]
    return "\n".join(L) + "\n"


def main() -> int:
    idx = json.loads((ROOT / "build/flashnext-iq3s-shard1.index.json").read_text(encoding="utf-8"))
    t = next(x for x in idx["tensors"] if x["type"] == "IQ3_S")
    nblocks = 256
    src = pathlib.Path(idx["sources"][t["source"]]["path"])
    with open(src, "rb") as f:
        f.seek(t["offset"])
        raw = f.read(nblocks * 110)
    ref = dequant_iq3_s(raw)
    grid = struct.pack("<%dI" % len(iq3s_grid), *[int(v) for v in iq3s_grid])

    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="iq3s")
    out = run_one(hsaco, NAME,
                  [{"buffer": "w"}, {"buffer": "y"}, {"buffer": "grid"},
                   {"scalar": {"dtype": "u32", "value": nblocks}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"w": {"dtype": "u8", "values": list(raw)},
                   "y": {"dtype": "f32", "values": [0.0] * (nblocks * 256)},
                   "grid": {"dtype": "u8", "values": list(grid)}},
                  grid=nblocks, workgroup=64)
    got = np.array(out["y"], dtype=np.float32)
    diff = np.abs(got - ref)
    print(f"{t['name']}: nblocks={nblocks} max_abs={diff.max():.3e}")
    ok = diff.max() == 0.0
    print("GPU IQ3_S vs 参考：", "bit-exact ✔" if ok else "不一致 ✘")
    if not ok:
        for i in np.flatnonzero(diff > 0)[:6]:
            print(f"  y[{i}] gpu={got[i]:.8g} ref={ref[i]:.8g}")
    print("HSACO:", hsaco)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
