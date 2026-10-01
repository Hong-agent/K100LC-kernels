#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IQ4_NL 原生解码 GEMV 内核（gfx926 `.s` 生成器）+ 真权重 GPU 自检。

    python3 tools/gen_iq4nl_gemv.py

内核签名：

    __global__ void iq4nl_gemv_k(const uint8_t* w, const float* x, float* y,
                                 uint32_t nrows, uint32_t k, uint32_t group_size)

布局与 `gemv_w4a8` 同源：`y[n] = Σ_k W[n,k]·x[k]`，W 是 GGUF 里的
IQ4_NL `[ne0=k, ne1=n]` 行主序（每行 k/32 个 18 字节块）。

这是**第一个正确性版本**：一个 work-item 算一行，激活保持 f32、逐元素
`v_fma_f32`。它不追吞吐（后续再上 `v_dot4_i32_i8` + int8 激活），目的
是把「原生解码 + GEMV 拼装」的数值先跑对。
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

NAME = "iq4nl_gemv_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
        scalar_arg(24, 4), scalar_arg(28, 4), scalar_arg(32, 4)]
KERNARG_SIZE = 40


def _decode_into(L: list[str], selector: str) -> None:
    """把 selector dword（4 个 0..15 码）解成 4 个 int8，结果放 v30。"""
    L += [
        f"v_and_b32_e32 v31, 0x07070707, {selector}",
        "v_perm_b32 v25, v9, v10, v31",
        "v_perm_b32 v26, v11, v12, v31",
        f"v_and_b32_e32 v27, 0x08080808, {selector}",
        "v_lshrrev_b32_e32 v27, 3, v27",
        "v_perm_b32 v28, v3, v13, v27",
        "v_bfi_b32 v30, v28, v26, v25",
    ]


def _fma4(L: list[str], xoff: int) -> None:
    """v30 的 4 个 int8 与 x[+xoff] 的 4 个 f32 做 FMA，累加进 v32..v35。"""
    L.append(f"global_load_dwordx4 v[36:39], v[6:7], off offset:{xoff}")
    L.append("s_waitcnt vmcnt(0)")
    for i in range(4):
        L += [
            f"v_lshrrev_b32_e32 v40, {8 * i}, v30",
            "v_lshlrev_b32_e32 v40, 24, v40",
            "v_ashrrev_i32_e32 v40, 24, v40",
            "v_cvt_f32_i32_e32 v40, v40",
            f"v_fma_f32 v{32 + i}, v40, v{36 + i}, v{32 + i}",
        ]


def gen_asm() -> str:
    L: list[str] = []
    em = L.append

    em(".text")
    em(f"k_{NAME}:")
    em("s_load_dwordx2 s[16:17], s[4:5], 0x0")    # w
    em("s_load_dwordx2 s[18:19], s[4:5], 0x8")    # x
    em("s_load_dwordx2 s[20:21], s[4:5], 0x10")   # y
    em("s_load_dword s22, s[4:5], 0x18")          # nrows
    em("s_load_dword s23, s[4:5], 0x1c")          # k
    em("s_load_dword s24, s[4:5], 0x20")          # group_size
    em("s_waitcnt lgkmcnt(0)")
    em("v_mov_b32_e32 v3, 0")
    em("v_mov_b32_e32 v1, s6")
    em("v_mul_lo_u32 v1, v1, s24")
    em("v_add_u32_e32 v1, v0, v1")                # row
    em("v_cmp_gt_u32_e32 vcc, s22, v1")
    em("s_and_saveexec_b64 s[0:1], vcc")
    em("s_cbranch_execz L_end")
    em("s_lshr_b32 s25, s23, 5")                  # NB = k/32
    em("s_mul_i32 s26, s25, 18")                  # rowbytes
    em("v_mul_lo_u32 v2, v1, s26")
    em("v_mov_b32_e32 v4, s16")
    em("v_mov_b32_e32 v5, s17")
    em("v_add_co_u32_e32 v4, vcc, v4, v2")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")  # wrow
    em("v_mov_b32_e32 v6, s18")
    em("v_mov_b32_e32 v7, s19")                   # x 块基址（每轮 +128B）
    em("v_lshlrev_b32_e32 v44, 2, v1")
    em("v_mov_b32_e32 v45, s21")
    em("v_add_co_u32_e32 v44, vcc, s20, v44")
    em("v_addc_co_u32_e32 v45, vcc, v45, v3, vcc")  # y + row*4
    em("v_mov_b32_e32 v9, 0xf6eaddcf")
    em("v_mov_b32_e32 v10, 0xbfad9881")
    em("v_mov_b32_e32 v11, 0x71594535")
    em("v_mov_b32_e32 v12, 0x26190d01")
    em("v_mov_b32_e32 v13, 0x0000ff00")
    em("v_mov_b32_e32 v46, 18")
    em("v_mov_b32_e32 v47, 128")
    em("s_mov_b32 s27, 0")
    em("v_mov_b32_e32 v41, 0")

    em("L_loop:")
    em("v_mov_b32_e32 v32, 0")
    em("v_mov_b32_e32 v33, 0")
    em("v_mov_b32_e32 v34, 0")
    em("v_mov_b32_e32 v35, 0")
    em("global_load_ushort v8, v[4:5], off")
    for b in range(16):
        em(f"global_load_ubyte v{48 + b}, v[4:5], off offset:{2 + b}")
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v8, v8")
    # q_j = b_{4j} | b_{4j+1}<<8 | b_{4j+2}<<16 | b_{4j+3}<<24 → v18..v21
    for j in range(4):
        bse = 48 + 4 * j
        em(f"v_lshlrev_b32_e32 v22, 8, v{bse + 1}")
        em(f"v_or_b32_e32 v22, v{bse}, v22")
        em(f"v_lshlrev_b32_e32 v23, 8, v{bse + 3}")
        em(f"v_or_b32_e32 v23, v{bse + 2}, v23")
        em("v_lshlrev_b32_e32 v23, 16, v23")
        em(f"v_or_b32_e32 v{18 + j}, v22, v23")
        # 低半字节流：元素 4j..4j+3
        em(f"v_and_b32_e32 v24, 0x0f0f0f0f, v{18 + j}")
        _decode_into(L, "v24")
        _fma4(L, 16 * j)
        # 高半字节流：元素 16+4j..19+4j
        em(f"v_and_b32_e32 v24, 0xf0f0f0f0, v{18 + j}")
        em("v_lshrrev_b32_e32 v24, 4, v24")
        em("v_and_b32_e32 v24, 0x0f0f0f0f, v24")
        _decode_into(L, "v24")
        _fma4(L, 64 + 16 * j)

    em("v_add_f32_e32 v32, v32, v33")
    em("v_add_f32_e32 v34, v34, v35")
    em("v_add_f32_e32 v32, v32, v34")
    em("v_mul_f32_e32 v32, v8, v32")
    em("v_add_f32_e32 v41, v41, v32")
    em("v_add_co_u32_e32 v4, vcc, v4, v46")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")
    em("v_add_co_u32_e32 v6, vcc, v6, v47")
    em("v_addc_co_u32_e32 v7, vcc, v7, v3, vcc")
    em("s_add_i32 s27, s27, 1")
    em("s_cmp_lt_u32 s27, s25")
    em("s_cbranch_scc1 L_loop")

    em("global_store_dword v[44:45], v41, off")
    em("L_end:")
    em("s_or_b64 exec, exec, s[0:1]")
    em("s_endpgm")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="blk.0.ffn_down_shexp.weight")
    ap.add_argument("--rows", type=int, default=2560, help="只测前 N 行（0=全部）")
    ap.add_argument("--workgroup", type=int, default=64)
    args = ap.parse_args()

    from gguf_sample import load_index
    idx = load_index()
    t = next((x for x in idx["tensors"] if x["name"] == args.tensor), None)
    if t is None or t["type"] != "IQ4_NL":
        raise SystemExit(f"{args.tensor}: 不是 IQ4_NL 张量")
    k, n = t["shape"][0], t["shape"][1]
    if k % 32:
        raise SystemExit(f"K={k} 不是 32 的倍数")
    rows = min(args.rows or n, n)
    rowbytes = k // 32 * 18
    src = pathlib.Path(idx["sources"][t["source"]]["path"])
    with open(src, "rb") as f:
        f.seek(t["offset"])
        raw = f.read(rows * rowbytes)
    if len(raw) != rows * rowbytes or max(raw) == 0:
        raise SystemExit("权重读取失败（未下完？）")
    W = dequant_iq4_nl(raw).reshape(rows, k)
    x = np.random.default_rng(3).standard_normal(k).astype(np.float32)
    ref = (W.astype(np.float32) @ x).astype(np.float32)

    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="gemv")
    out = run_one(hsaco, NAME,
                  [{"buffer": "w"}, {"buffer": "x"}, {"buffer": "y"},
                   {"scalar": {"dtype": "u32", "value": rows}},
                   {"scalar": {"dtype": "u32", "value": k}},
                   {"scalar": {"dtype": "u32", "value": args.workgroup}}],
                  {"w": {"dtype": "u8", "values": list(raw)},
                   "x": {"dtype": "f32", "values": [float(v) for v in x]},
                   "y": {"dtype": "f32", "values": [0.0] * rows}},
                  grid=rows, workgroup=args.workgroup)
    got = np.array(out["y"], dtype=np.float32)
    diff = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), 1e-6)
    print(f"{args.tensor}: rows={rows} k={k}  max_abs={diff.max():.3e}  "
          f"max_rel={(diff / denom).max():.3e}")
    ok = diff.max() < 1e-3 * max(1.0, float(np.abs(ref).max()))
    print("GPU GEMV vs 参考：", "一致 ✔" if ok else "不一致 ✘")
    if not ok:
        for i in np.argsort(-diff)[:5]:
            print(f"  y[{i}] gpu={got[i]:.6g} ref={ref[i]:.6g}")
    print("HSACO:", hsaco)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
