#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IQ4_XS → f32 解码内核（gfx926 生成器 + 真权重自检）。

    python3 tools/gen_iq4xs_dequant.py

IQ4_XS：256 元素 / 136 字节 = f16 d + u16 scales_h + u8 scales_l[4] + u8 qs[128]。
每 32 元素一个 6 bit 尺度 ls（低 4 位来自 scales_l，高 2 位来自 scales_h），
`dl = d*(ls-32)`；码值同样查 16 项 kvalues_iq4nl。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from iq_dequant import dequant_iq4_xs  # noqa: E402
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "iq4xs_dequant_k"
ARGS = [buffer_arg(0), buffer_arg(8), scalar_arg(16, 4), scalar_arg(20, 4)]
KERNARG_SIZE = 24


def _decode(L: list[str], selector: str) -> None:
    L += [
        f"v_and_b32_e32 v31, 0x07070707, {selector}",
        "v_perm_b32 v25, v9, v10, v31",
        "v_perm_b32 v26, v11, v12, v31",
        f"v_and_b32_e32 v28, 0x08080808, {selector}",
        "v_lshrrev_b32_e32 v28, 3, v28",
        "v_perm_b32 v27, v3, v13, v28",
        "v_bfi_b32 v30, v27, v26, v25",
    ]


def _store4(L: list[str], off: int) -> None:
    """v30 的 4 个 int8 × v46(尺度) → 4 个 f32 → dwordx4 存到 y+off。"""
    for i in range(4):
        L += [
            f"v_lshrrev_b32_e32 v{40 + i}, {8 * i}, v30",
            f"v_lshlrev_b32_e32 v{40 + i}, 24, v{40 + i}",
            f"v_ashrrev_i32_e32 v{40 + i}, 24, v{40 + i}",
            f"v_cvt_f32_i32_e32 v{40 + i}, v{40 + i}",
            f"v_mul_f32_e32 v{40 + i}, v46, v{40 + i}",
        ]
    L.append(f"global_store_dwordx4 v[6:7], v[40:43], off offset:{off}")


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
        "v_mov_b32_e32 v9, 0xf6eaddcf",
        "v_mov_b32_e32 v10, 0xbfad9881",
        "v_mov_b32_e32 v11, 0x71594535",
        "v_mov_b32_e32 v12, 0x26190d01",
        "v_mov_b32_e32 v13, 0x0000ff00",
        "v_mov_b32_e32 v47, 0xc2000000",   # -32.0f
        "v_mov_b32_e32 v80, 136",          # 块字节数
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
        "global_load_ubyte v16, v[4:5], off offset:2",
        "global_load_ubyte v17, v[4:5], off offset:3",
        "global_load_dword v14, v[4:5], off offset:4",
    ]
    # qs[128] 在块内偏移 8，8 字节对齐：16 次 dwordx2
    for m in range(16):
        L.append(f"global_load_dwordx2 v[{48 + 2 * m}:{49 + 2 * m}], v[4:5], off offset:{8 + 8 * m}")
    L.append("s_waitcnt vmcnt(0)")
    L.append("v_cvt_f32_f16_e32 v8, v8")
    L += ["v_lshlrev_b32_e32 v17, 8, v17", "v_or_b32_e32 v15, v16, v17"]
    for ib in range(8):
        # ls = ((scales_l >> 4*(ib//2)) & 0xF) | (((scales_h >> 2*ib) & 3) << 4)
        L += [
            f"v_lshrrev_b32_e32 v44, {4 * ib}, v14",
            "v_and_b32_e32 v44, 0xf, v44",
            f"v_lshrrev_b32_e32 v45, {2 * ib}, v15",
            "v_and_b32_e32 v45, 3, v45",
            "v_lshlrev_b32_e32 v45, 4, v45",
            "v_or_b32_e32 v44, v44, v45",
            "v_cvt_f32_u32_e32 v46, v44",
            "v_add_f32_e32 v46, v47, v46",
            "v_mul_f32_e32 v46, v8, v46",
        ]
        for j in range(4):
            qreg = 48 + 4 * ib + j
            # 低半字节 → 本组元素 0..15
            L.append(f"v_and_b32_e32 v20, 0x0f0f0f0f, v{qreg}")
            _decode(L, "v20")
            _store4(L, 128 * ib + 16 * j)
            # 高半字节 → 本组元素 16..31
            L.append(f"v_and_b32_e32 v20, 0xf0f0f0f0, v{qreg}")
            L.append("v_lshrrev_b32_e32 v20, 4, v20")
            L.append("v_and_b32_e32 v20, 0x0f0f0f0f, v20")
            _decode(L, "v20")
            _store4(L, 128 * ib + 64 + 16 * j)
    L += ["L_end:", "s_or_b64 exec, exec, s[0:1]", "s_endpgm"]
    return "\n".join(L) + "\n"


def main() -> int:
    idx = json.loads((ROOT / "build/flashnext-iq3s-shard1.index.json").read_text(encoding="utf-8"))
    t = next(x for x in idx["tensors"] if x["type"] == "IQ4_XS" and x["shape"][0] % 256 == 0)
    nblocks = 512
    src = pathlib.Path(idx["sources"][t["source"]]["path"])
    with open(src, "rb") as f:
        f.seek(t["offset"])
        raw = f.read(nblocks * 136)
    ref = dequant_iq4_xs(raw)

    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="iq4xs")
    out = run_one(hsaco, NAME,
                  [{"buffer": "w"}, {"buffer": "y"},
                   {"scalar": {"dtype": "u32", "value": nblocks}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"w": {"dtype": "u8", "values": list(raw)},
                   "y": {"dtype": "f32", "values": [0.0] * (nblocks * 256)}},
                  grid=nblocks, workgroup=64)
    got = np.array(out["y"], dtype=np.float32)
    diff = np.abs(got - ref)
    print(f"{t['name']}: nblocks={nblocks} max_abs={diff.max():.3e}")
    ok = diff.max() == 0.0
    print("GPU IQ4_XS vs 参考：", "bit-exact ✔" if ok else "不一致 ✘")
    if not ok:
        for i in np.flatnonzero(diff > 0)[:6]:
            print(f"  y[{i}] gpu={got[i]:.8g} ref={ref[i]:.8g}")
    print("HSACO:", hsaco)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
