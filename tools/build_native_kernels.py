#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把本包的原生内核打成一个 `build/native_kernels.hsaco`。

    python3 tools/build_native_kernels.py

只包含 Python 生成器产出的原生内核；FASTASM 基线内核另有自己的 HSACO，
两者最后由 `tools/merge_hsacos.py` 合并成完整内核包。
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import asm  # noqa: E402
from kernel_lab import _max_registers  # noqa: E402

import gen_gemv_f32  # noqa: E402
import gen_gemv_f32_warp  # noqa: E402
import gen_gemv_i8  # noqa: E402
import gen_gelu_mul  # noqa: E402
import gen_q4_0_dequant  # noqa: E402
import gen_q8_0_dequant  # noqa: E402
import gen_softmax  # noqa: E402
import gen_layernorm  # noqa: E402
import gen_topk  # noqa: E402
import gen_router_top10  # noqa: E402
import gen_gemv_qdot  # noqa: E402
import gen_iq2s_dequant  # noqa: E402
import gen_iq3s_dequant  # noqa: E402
import gen_iq3xxs_dequant  # noqa: E402
import gen_iq4nl_dequant  # noqa: E402
import gen_iq4nl_to_i8  # noqa: E402
import gen_iq4xs_dequant  # noqa: E402
import gen_q2_0_dequant  # noqa: E402
import gen_int4_dot  # noqa: E402
import gen_quant_fast  # noqa: E402
import gen_gemv_w4a4_r2  # noqa: E402
import gen_moe_combine  # noqa: E402

MODULES = [
    gen_gemv_f32, gen_gemv_f32_warp, gen_gemv_i8, gen_iq4nl_dequant, gen_iq4nl_to_i8,
    gen_q2_0_dequant, gen_iq4xs_dequant, gen_iq3xxs_dequant,
    gen_iq2s_dequant, gen_iq3s_dequant, gen_gelu_mul,
    gen_q4_0_dequant, gen_q8_0_dequant, gen_softmax, gen_layernorm, gen_topk,
    gen_router_top10,
    gen_gemv_qdot,
    gen_int4_dot,
    gen_quant_fast,
    gen_gemv_w4a4_r2,
    gen_moe_combine,
]


def main() -> int:
    out = ROOT / "build" / "native_kernels"
    out.mkdir(parents=True, exist_ok=True)
    spec = []
    jobs = []
    for mod in MODULES:
        if hasattr(mod, "KERNELS"):
            jobs.extend((nm, gen(), ar, ks) for nm, gen, ar, ks in mod.KERNELS)
        else:
            jobs.append((mod.NAME, mod.gen_asm(), mod.ARGS, mod.KERNARG_SIZE))
    for name, asm_text, args, ksize in jobs:
        src = out / f"{name}.s"
        src.write_text(asm_text, encoding="utf-8")
        code, _ins, _lbl = asm.assemble(src)
        (out / f"{name}.bin").write_bytes(code)
        ngpr, vgpr = _max_registers(asm_text)
        spec.append({
            "name": name,
            "code": str(out / f"{name}.bin"),
            "args": args,
            "kernarg_size": ksize,
            "kernarg_align": 8,
            "sgpr_count": max(ngpr, 4),
            "vgpr_count": max(vgpr, 1),
            # warp-per-row GEMV 用 LDS 做归约（64 lane × 4B）
            "group_segment": 1024 if name in ("gemv_f32_warp_k", "softmax_k",
                                              "layernorm_k") else 0,
            "private_segment": 0,
        })
        print(f"  {name:22s} code={len(code):6d}B vgpr={max(vgpr, 1):3d} sgpr={max(ngpr, 4):2d}")
    spec_path = out / "native_kernels.spec.json"
    spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    hsaco = ROOT / "build" / "native_kernels.hsaco"
    subprocess.run([sys.executable, str(ROOT / "tools/make_hsaco_multi.py"),
                    str(spec_path), str(hsaco)], check=True, capture_output=True)
    print(f"-> {hsaco}（{hsaco.stat().st_size} 字节，{len(spec)} 个内核）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
