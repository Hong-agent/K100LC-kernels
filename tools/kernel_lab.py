#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新内核开发的一键环路：写 .s → 自研汇编器 → 单核 HSACO → 上卡执行 → 回读。

给 IQ4_NL / MoE 等新内核用。所有步骤都只用仓库内工具与 /opt/hyhal：

    from kernel_lab import build_one, run_one
    hsaco = build_one("probe_k", ASM, ARGS, kernarg_size=24)
    out = run_one(hsaco, "probe_k", args=[...], buffers={...}, grid=1, workgroup=64)

`build_one` 会把产物放在 build/lab/ 下，便于反复调试与事后取证。
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import asm  # noqa: E402

LAB = ROOT / "build" / "lab"


def buffer_arg(offset: int) -> dict:
    return {".actual_access": "read_only", ".address_space": "global",
            ".offset": offset, ".size": 8, ".value_kind": "global_buffer"}


def scalar_arg(offset: int, size: int) -> dict:
    return {".address_space": "global", ".offset": offset, ".size": size,
            ".value_kind": "by_value"}


def _max_registers(text: str) -> tuple[int, int]:
    """从 .s 文本里粗算最高 SGPR / VGPR 编号，供 HSACO 元数据用。"""
    import re
    max_s, max_v = 3, 0
    for m in re.finditer(r"\bs\[(\d+)(?::(\d+))?\]", text):
        max_s = max(max_s, int(m.group(2) or m.group(1)))
    for m in re.finditer(r"\bs(\d+)\b", text):
        max_s = max(max_s, int(m.group(1)))
    for m in re.finditer(r"\bv\[(\d+)(?::(\d+))?\]", text):
        max_v = max(max_v, int(m.group(2) or m.group(1)))
    for m in re.finditer(r"\bv(\d+)\b", text):
        max_v = max(max_v, int(m.group(1)))
    return max_s + 1, max_v + 1


def build_one(name: str, asm_text: str, args: list[dict], kernarg_size: int,
              sgpr_count: int | None = None, vgpr_count: int | None = None,
              kernarg_align: int = 8, group_segment: int = 0,
              private_segment: int = 0, tag: str = "") -> pathlib.Path:
    LAB.mkdir(parents=True, exist_ok=True)
    stem = f"{name}{('_' + tag) if tag else ''}"
    src = LAB / f"{stem}.s"
    src.write_text(asm_text if asm_text.endswith("\n") else asm_text + "\n",
                   encoding="utf-8")
    code, _insts, _labels = asm.assemble(src)
    bin_path = LAB / f"{stem}.bin"
    bin_path.write_bytes(code)
    if sgpr_count is None or vgpr_count is None:
        auto_s, auto_v = _max_registers(asm_text)
        sgpr_count = sgpr_count or max(auto_s, 4)
        vgpr_count = vgpr_count or max(auto_v, 1)
    spec = [{
        "name": name, "code": str(bin_path), "args": args,
        "kernarg_size": kernarg_size, "kernarg_align": kernarg_align,
        "sgpr_count": sgpr_count, "vgpr_count": vgpr_count,
        "group_segment": group_segment, "private_segment": private_segment,
    }]
    spec_path = LAB / f"{stem}.spec.json"
    spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    hsaco = LAB / f"{stem}.hsaco"
    subprocess.run([sys.executable, str(ROOT / "tools/make_hsaco_multi.py"),
                    str(spec_path), str(hsaco)], check=True, capture_output=True)
    return hsaco


def run_one(hsaco: pathlib.Path, kernel: str, args: list[dict], buffers: dict,
            grid: int = 1, workgroup: int = 64, grid_y: int = 1,
            extra_group: int = 0) -> dict:
    """启动一个内核；`grid` 是**总 work-item 数**（见 tools/probe_ids.py）。"""
    os.environ["RT_HSACO"] = str(hsaco)
    from hsa_job import run_job
    return run_job(hsaco, kernel, args, buffers, grid=grid, workgroup=workgroup,
                   grid_y=grid_y, extra_group=extra_group)
