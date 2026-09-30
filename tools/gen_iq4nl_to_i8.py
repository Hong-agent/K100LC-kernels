#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IQ4_NL → int8 + 每块尺度（gfx926 内核生成器 + 真权重自检）。

    python3 tools/gen_iq4nl_to_i8.py

内核签名：

    __global__ void iq4nl_to_i8_k(const uint8_t* w, int8_t* out, float* scale,
                                  uint32_t nblocks, uint32_t group_size)

一个 work-item 解一个 18 字节 IQ4_NL 块：32 个码值按自然顺序写成 32 个 int8，
块的 f16 尺度单独写成 f32。之后由 `gemv_i8_k` 做通用 int8 GEMV。
数值关系：`out[e] * scale[b] == dequant_iq4_nl(w)[e]`。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from iq_dequant import dequant_iq4_nl  # noqa: E402
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "iq4nl_to_i8_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
        scalar_arg(24, 4), scalar_arg(28, 4)]
KERNARG_SIZE = 32


def _decode_into(L: list[str], selector: str) -> None:
    L += [
        f"v_and_b32_e32 v31, 0x07070707, {selector}",
        "v_perm_b32 v25, v9, v10, v31",
        "v_perm_b32 v26, v11, v12, v31",
        f"v_and_b32_e32 v27, 0x08080808, {selector}",
        "v_lshrrev_b32_e32 v27, 3, v27",
        "v_perm_b32 v28, v3, v13, v27",
        "v_bfi_b32 v30, v28, v26, v25",
    ]


def gen_asm() -> str:
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{NAME}:")
    em("s_load_dwordx2 s[16:17], s[4:5], 0x0")    # w
    em("s_load_dwordx2 s[18:19], s[4:5], 0x8")    # out
    em("s_load_dwordx2 s[20:21], s[4:5], 0x10")   # scale
    em("s_load_dword s22, s[4:5], 0x18")          # nblocks
    em("s_load_dword s23, s[4:5], 0x1c")          # group_size
    em("s_waitcnt lgkmcnt(0)")
    em("v_mov_b32_e32 v3, 0")
    em("v_mov_b32_e32 v1, s6")
    em("v_mul_lo_u32 v1, v1, s23")
    em("v_add_u32_e32 v1, v0, v1")                # block id
    em("v_cmp_gt_u32_e32 vcc, s22, v1")
    em("s_and_saveexec_b64 s[0:1], vcc")
    em("s_cbranch_execz L_end")
    em("v_mul_lo_u32 v2, v1, 18")
    em("v_mov_b32_e32 v4, s16")
    em("v_mov_b32_e32 v5, s17")
    em("v_add_co_u32_e32 v4, vcc, v4, v2")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")   # waddr
    em("v_lshlrev_b32_e32 v6, 5, v1")
    em("v_lshlrev_b32_e32 v7, 0, v1")              # 占位，后面覆盖
    em("v_mov_b32_e32 v7, s19")
    em("v_add_co_u32_e32 v6, vcc, s18, v6")
    em("v_addc_co_u32_e32 v7, vcc, v7, v3, vcc")   # out + block*32
    em("v_lshlrev_b32_e32 v44, 2, v1")
    em("v_mov_b32_e32 v45, s21")
    em("v_add_co_u32_e32 v44, vcc, s20, v44")
    em("v_addc_co_u32_e32 v45, vcc, v45, v3, vcc") # scale + block*4
    em("v_mov_b32_e32 v9, 0xf6eaddcf")
    em("v_mov_b32_e32 v10, 0xbfad9881")
    em("v_mov_b32_e32 v11, 0x71594535")
    em("v_mov_b32_e32 v12, 0x26190d01")
    em("v_mov_b32_e32 v13, 0x0000ff00")
    em("global_load_ushort v8, v[4:5], off")
    for b in range(16):
        em(f"global_load_ubyte v{48 + b}, v[4:5], off offset:{2 + b}")
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v8, v8")
    em("global_store_dword v[44:45], v8, off")
    for j in range(4):
        bse = 48 + 4 * j
        em(f"v_lshlrev_b32_e32 v22, 8, v{bse + 1}")
        em(f"v_or_b32_e32 v22, v{bse}, v22")
        em(f"v_lshlrev_b32_e32 v23, 8, v{bse + 3}")
        em(f"v_or_b32_e32 v23, v{bse + 2}, v23")
        em("v_lshlrev_b32_e32 v23, 16, v23")
        em(f"v_or_b32_e32 v18, v22, v23")
        em("v_and_b32_e32 v24, 0x0f0f0f0f, v18")
        _decode_into(L, "v24")
        em(f"global_store_dword v[6:7], v30, off offset:{4 * j}")
        em("v_and_b32_e32 v24, 0xf0f0f0f0, v18")
        em("v_lshrrev_b32_e32 v24, 4, v24")
        em("v_and_b32_e32 v24, 0x0f0f0f0f, v24")
        _decode_into(L, "v24")
        em(f"global_store_dword v[6:7], v30, off offset:{16 + 4 * j}")
    em("L_end:")
    em("s_or_b64 exec, exec, s[0:1]")
    em("s_endpgm")
    return "\n".join(L) + "\n"


def main() -> int:
    idx = json.loads((ROOT / "build/flashnext-iq3s-shard1.index.json").read_text(encoding="utf-8"))
    t = next(x for x in idx["tensors"] if x["name"] == "blk.0.ffn_down_shexp.weight")
    nblocks = 4096
    src = pathlib.Path(idx["sources"][t["source"]]["path"])
    with open(src, "rb") as f:
        f.seek(t["offset"])
        raw = f.read(nblocks * 18)
    ref = dequant_iq4_nl(raw)

    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="to_i8")
    out = run_one(hsaco, NAME,
                  [{"buffer": "w"}, {"buffer": "out"}, {"buffer": "scale"},
                   {"scalar": {"dtype": "u32", "value": nblocks}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"w": {"dtype": "u8", "values": list(raw)},
                   "out": {"dtype": "i8", "values": [0] * (nblocks * 32)},
                   "scale": {"dtype": "f32", "values": [0.0] * nblocks}},
                  grid=nblocks, workgroup=64)
    codes = np.array(out["out"], dtype=np.int8).reshape(nblocks, 32)
    scales = np.array(out["scale"], dtype=np.float32).reshape(nblocks, 1)
    got = (codes.astype(np.float32) * scales).reshape(-1)
    diff = np.abs(got - ref)
    print(f"nblocks={nblocks} max_abs={diff.max():.3e}")
    ok = diff.max() == 0.0
    print("GPU IQ4_NL→int8 × scale vs 参考：", "bit-exact ✔" if ok else "不一致 ✘")
    if not ok:
        bad = np.flatnonzero(diff > 0)[:5]
        for i in bad:
            print(f"  e[{i}] got={got[i]:.8g} ref={ref[i]:.8g}")
    print("HSACO:", hsaco)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
