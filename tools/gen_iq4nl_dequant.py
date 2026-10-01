#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IQ4_NL 原生解码内核（gfx926 `.s` 生成器）+ 真权重 GPU 自检。

    python3 tools/gen_iq4nl_dequant.py            # 生成 → 上卡 → 与参考解码对账
    python3 tools/gen_iq4nl_dequant.py --blocks 4096

内核签名：

    __global__ void iq4nl_dequant_k(const uint8_t* w, float* y,
                                    uint32_t nblocks, uint32_t group_size)

一个 work-item 解一个 18 字节 IQ4_NL 块（f16 d + 16 字节 qs），写 32 个 f32。
它不追求吞吐（比 GEMV 简单得多），目的是把下面两条最容易错的路径先验对：

1. 16 项非线性码本用 `v_perm_b32` 查表（selector 0..7 才干净，>=8 要 mask 合并）；
2. 18 字节块、qs 在 `+2` 处非 4 字节对齐（全部用 `global_load_ubyte + offset`）。

数值参考 = `tools/iq_dequant.py::dequant_iq4_nl`（已与 ggml `to_float` 逐位一致）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from iq_dequant import dequant_iq4_nl  # noqa: E402
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "iq4nl_dequant_k"
ARGS = [buffer_arg(0), buffer_arg(8), scalar_arg(16, 4), scalar_arg(20, 4)]
KERNARG_SIZE = 24

# kvalues_iq4nl = [-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113]
# 低半表 8 字节 → 两个 dword（小端）；高半表同理。
LUT_LO_HI = 0xF6EADDCF   # 码 4..7 的字节
LUT_LO_LO = 0xBFAD9881   # 码 0..3 的字节
LUT_HI_HI = 0x71594535   # 码 12..15 的字节
LUT_HI_LO = 0x26190D01   # 码 8..11 的字节


def gen_asm() -> str:
    L: list[str] = []

    def em(s: str) -> None:
        L.append(s)

    em(".text")
    em(f"k_{NAME}:")
    em("s_load_dwordx2 s[16:17], s[4:5], 0x0")     # w
    em("s_load_dwordx2 s[18:19], s[4:5], 0x8")     # y
    em("s_load_dword s20, s[4:5], 0x10")           # nblocks
    em("s_load_dword s21, s[4:5], 0x14")           # group_size
    em("s_waitcnt lgkmcnt(0)")
    em("v_mov_b32_e32 v3, 0")
    em("v_mov_b32_e32 v1, s6")                     # blockIdx.x
    em("v_mul_lo_u32 v1, v1, s21")
    em("v_add_u32_e32 v1, v0, v1")                 # gid = blockIdx*gsize + tid
    em("v_cmp_gt_u32_e32 vcc, s20, v1")            # gid < nblocks ?
    em("s_and_saveexec_b64 s[0:1], vcc")
    em("s_cbranch_execz L_end")
    em("v_mul_lo_u32 v2, v1, 18")                  # w 字节偏移
    em("v_mov_b32_e32 v4, s16")
    em("v_mov_b32_e32 v5, s17")
    em("v_add_co_u32_e32 v4, vcc, v4, v2")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")   # waddr 64 位
    em("v_lshlrev_b32_e32 v6, 7, v1")              # gid*128
    em("v_mov_b32_e32 v7, s19")
    em("v_add_co_u32_e32 v6, vcc, s18, v6")
    em("v_addc_co_u32_e32 v7, vcc, v7, v3, vcc")   # yaddr 64 位
    em(f"v_mov_b32_e32 v9, {LUT_LO_HI:#010x}")
    em(f"v_mov_b32_e32 v10, {LUT_LO_LO:#010x}")
    em(f"v_mov_b32_e32 v11, {LUT_HI_HI:#010x}")
    em(f"v_mov_b32_e32 v12, {LUT_HI_LO:#010x}")
    em("v_mov_b32_e32 v13, 0x0000ff00")            # mask perm 的源
    em("global_load_ushort v8, v[4:5], off")
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v8, v8")                 # 块尺度 d

    def lut(src: str, dst_res: str, dst_a: str, dst_b: str,
            dst_t: str, dst_mask: str, dst_sm: str) -> None:
        """src（selector dword，每字节 0..15）→ dst_res（4 个 int8 打包）。

        关键：v_perm 的 selector 只有 bit0..2 干净，>=8 的行为不稳定，
        所以查表必须先用 `& 0x07070707` 把 bit3 清掉，再用 bit3 造半表 mask。
        """
        em(f"v_and_b32_e32 {dst_sm}, 0x07070707, {src}")
        em(f"v_perm_b32 {dst_a}, v9, v10, {dst_sm}")
        em(f"v_perm_b32 {dst_b}, v11, v12, {dst_sm}")
        em(f"v_and_b32_e32 {dst_t}, 0x08080808, {src}")
        em(f"v_lshrrev_b32_e32 {dst_t}, 3, {dst_t}")
        em(f"v_perm_b32 {dst_mask}, v3, v13, {dst_t}")
        em(f"v_bfi_b32 {dst_res}, {dst_mask}, {dst_b}, {dst_a}")

    def store4(res: str, base_off: int) -> None:
        for i in range(4):
            em(f"v_lshrrev_b32_e32 v30, {8 * i}, {res}")
            em("v_lshlrev_b32_e32 v30, 24, v30")
            em("v_ashrrev_i32_e32 v30, 24, v30")   # 符号扩展 int8
            em("v_cvt_f32_i32_e32 v30, v30")
            em("v_mul_f32_e32 v30, v8, v30")
            em(f"global_store_dword v[6:7], v30, off offset:{base_off + 4 * i}")

    # 4 个 qs dword：每个 4 字节 = 低半字节流 4 个元素 + 高半字节流 4 个元素。
    for j in range(4):
        for b in range(4):
            em(f"global_load_ubyte v{14 + b}, v[4:5], off offset:{2 + 4 * j + b}")
        em("s_waitcnt vmcnt(0)")
        em("v_lshlrev_b32_e32 v22, 8, v15")
        em("v_or_b32_e32 v22, v14, v22")
        em("v_lshlrev_b32_e32 v23, 8, v17")
        em("v_or_b32_e32 v23, v16, v23")
        em("v_lshlrev_b32_e32 v23, 16, v23")
        em("v_or_b32_e32 v18, v22, v23")           # qs dword j
        # 低半字节流：元素 4j..4j+3
        em("v_and_b32_e32 v24, 0x0f0f0f0f, v18")
        lut("v24", "v29", "v25", "v26", "v27", "v28", "v31")
        store4("v29", 16 * j)
        # 高半字节流：元素 16+4j..19+4j
        em("v_and_b32_e32 v24, 0xf0f0f0f0, v18")
        em("v_lshrrev_b32_e32 v24, 4, v24")
        em("v_and_b32_e32 v24, 0x0f0f0f0f, v24")
        lut("v24", "v29", "v25", "v26", "v27", "v28", "v31")
        store4("v29", 64 + 16 * j)

    em("L_end:")
    em("s_or_b64 exec, exec, s[0:1]")
    em("s_endpgm")
    return "\n".join(L) + "\n"


def pick_iq4nl_blocks(nblocks: int) -> bytes:
    from gguf_sample import load_index
    idx = load_index()
    tensors = [t for t in idx["tensors"] if t["type"] == "IQ4_NL" and t["bytes"] >= nblocks * 18]
    if not tensors:
        raise SystemExit("索引里找不到 IQ4_NL 张量")
    tensors.sort(key=lambda t: t["bytes"])
    t = next((x for x in tensors if x["name"] == "blk.0.ffn_down_shexp.weight"), tensors[0])
    src = pathlib.Path(idx["sources"][t["source"]]["path"])
    if not src.is_file():
        raise SystemExit(f"权重文件不存在：{src}")
    with open(src, "rb") as f:
        f.seek(t["offset"])
        raw = f.read(nblocks * 18)
    if len(raw) != nblocks * 18 or max(raw) == 0:
        raise SystemExit("读到全零/短读，权重可能没下完")
    return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=1024)
    ap.add_argument("--workgroup", type=int, default=64)
    args = ap.parse_args()

    raw = pick_iq4nl_blocks(args.blocks)
    ref = dequant_iq4_nl(raw)
    asm_text = gen_asm()
    hsaco = build_one(NAME, asm_text, ARGS, kernarg_size=KERNARG_SIZE, tag="dequant")
    out = run_one(hsaco, NAME,
                  [{"buffer": "w"}, {"buffer": "y"},
                   {"scalar": {"dtype": "u32", "value": args.blocks}},
                   {"scalar": {"dtype": "u32", "value": args.workgroup}}],
                  {"w": {"dtype": "u8", "values": list(raw)},
                   "y": {"dtype": "f32", "values": [0.0] * (args.blocks * 32)}},
                  grid=args.blocks, workgroup=args.workgroup)
    got = np.array(out["y"], dtype=np.float32)
    diff = np.abs(got - ref)
    print(f"blocks={args.blocks}  max_abs={diff.max():.3e}  "
          f"max_rel={(diff / np.maximum(np.abs(ref), 1e-30)).max():.3e}")
    ok = diff.max() == 0.0 or diff.max() < 1e-6
    print("GPU 解码 vs 参考解码：", "bit-exact ✔" if ok else "不一致 ✘")
    if not ok:
        bad = np.flatnonzero(diff > 1e-6)[:8]
        for i in bad:
            print(f"  y[{i}] gpu={got[i]:.6g} ref={ref[i]:.6g}")
    print("HSACO:", hsaco)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
