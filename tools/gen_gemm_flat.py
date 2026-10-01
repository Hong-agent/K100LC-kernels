#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从基线 `gemm_w4a4` 生成 flat 1D 变体 `gemm_w4a4_flat`。

动机：这套无 DTK 运行时投递 2D grid 时，`blockIdx.y` 的第二个 workgroup
不可靠（见 `docs/ABI.md`）。原来的 `gemm_w4a4<<<dim3(N/BN, M/BM)>>>` 因此
只能按 128 行拆成多次 1D 启动，每块只有 `N/64` 个 workgroup（约 1.1 波），
占用率不足。

flat 变体只改 prologue：把 `bm/bn` 从 `blockIdx.x`（1D 扁平块号）算出来：

    q = flat / NBN,  r = flat % NBN     # NBN = N / BN
    bm = q * 128,    bn = r * 64

除法用主机传入的 `nbn` 与魔法数 `magic`（`q = (flat*magic)>>32`，`s_mul_hi_u32`），
内核其余 500 多条指令逐字节不变。新 ABI 在 K 后面追加两个 u32：

    gemm_w4a4_flat(Ap, Bp, sa, sb, C, M, N, K, nbn, magic)

于是 1 次 1D 启动即可覆盖全部 `(N/64) × (M/128)` 个块，恢复到满网格占用率。
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASM_DIR = ROOT / "kernels" / "asm" / "k_gemm"
BASE_ASM = ASM_DIR / "001__Z9gemm_w4a4PKjS0_PKfS2_Pfiii.s"
NEW_NAME = "_Z14gemm_w4a4_flatPKjS0_PKfS2_Pfiiiii"
NEW_ASM = ASM_DIR / f"002_{NEW_NAME}.s"
SPEC = ROOT / "kernels" / "kernel_spec.json"


def patch_asm(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    marks = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("k__Z9gemm_w4a4"):
            out.append(f"k_{NEW_NAME}:")
            marks += 1
            continue
        if i < 20 and s == "s_lshl_b32 s30, s7, 7":
            # 新参数：nbn(N/BN) 在 0x34、魔法数在 0x38；随后的 s_waitcnt 覆盖它们
            out += ["s_load_dword s36, s[4:5], 0x34",
                    "s_load_dword s37, s[4:5], 0x38"]
            marks += 1
            continue
        if i < 20 and s == "s_lshl_b32 s31, s6, 6":
            marks += 1
            continue
        if i < 20 and s == "s_ashr_i32 s2, s18, 3":
            out += [ln,
                    "s_mul_hi_u32 s38, s6, s37",   # q = flat / nbn
                    "s_mul_i32 s39, s38, s36",
                    "s_sub_u32 s39, s6, s39",      # r = flat % nbn
                    "s_lshl_b32 s30, s38, 7",      # bm = q * BM(128)
                    "s_lshl_b32 s31, s39, 6"]      # bn = r * BN(64)
            marks += 1
            continue
        out.append(ln)
    if marks != 4:
        raise SystemExit(f"gen_gemm_flat: 只改到 {marks}/4 处，基线 .s 变了？")
    return "\n".join(out)


def buf(off: int, write: bool = False) -> dict:
    return {".actual_access": "write_only" if write else "read_only",
            ".address_space": "global", ".offset": off, ".size": 8,
            ".value_kind": "global_buffer"}


def scalar(off: int) -> dict:
    return {".offset": off, ".size": 4, ".value_kind": "by_value"}


def main() -> int:
    # 清理历史拼错/改名的产物（只保留当前 NEW_NAME）
    for old in ASM_DIR.glob("*gemm_w4a4_flat*.s"):
        if old != NEW_ASM:
            old.unlink()
    NEW_ASM.write_text(patch_asm(BASE_ASM.read_text(encoding="utf-8")),
                       encoding="utf-8")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    entry = {
        "name": NEW_NAME,
        "args": [buf(0), buf(8), buf(16), buf(24), buf(32, True),
                 scalar(40), scalar(44), scalar(48), scalar(52), scalar(56)],
        "kernarg_size": 64,
        "kernarg_align": 8,
        "sgpr_count": 48,        # prologue 多用 s36..s39，留出余量
        "vgpr_count": 124,
        "group_segment": 27136,
        "private_segment": 0,
        "asm": f"k_gemm/002_{NEW_NAME}.s",
    }
    spec = [x for x in spec if "gemm_w4a4_flat" not in x.get("name", "")]
    spec.append(entry)
    SPEC.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[gen_gemm_flat] {NEW_ASM.relative_to(ROOT)} + {SPEC.name}（{NEW_NAME}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
