#!/usr/bin/env python3
"""HSACO kernel 符号名与隐藏参数的公共定义。"""
from __future__ import annotations

import re
import subprocess

# amdhsa metadata 的 .value_kind → RtArg.kind 编号
HID = {
    "hidden_block_count_x": 1, "hidden_block_count_y": 2, "hidden_block_count_z": 3,
    "hidden_group_size_x": 4, "hidden_group_size_y": 5, "hidden_group_size_z": 6,
    "hidden_remainder_x": 7, "hidden_remainder_y": 8, "hidden_remainder_z": 9,
    "hidden_global_offset_x": 10, "hidden_global_offset_y": 11,
    "hidden_global_offset_z": 12, "hidden_grid_dims": 13,
}


def demangled_base(symbol: str) -> str:
    """mangled 名 → 源码里的写法（如 gemv_w4a8<1, true, 1>）。"""
    out = subprocess.run(["c++filt", symbol], capture_output=True, text=True,
                         check=True).stdout.strip()
    m = re.match(r"^_Z(\d+)", symbol)
    if not m:
        return out
    start = 2 + len(m.group(1))          # _Z<十进制长度><名字>，长度从下标 2 开始
    base = symbol[start:start + int(m.group(1))]
    i = out.find(base)
    if i < 0:
        return base
    j = i + len(base)
    if j < len(out) and out[j] == "<":   # 模板实参
        depth, k = 0, j
        while k < len(out):
            if out[k] == "<":
                depth += 1
            elif out[k] == ">":
                depth -= 1
                if depth == 0:
                    k += 1
                    break
            k += 1
        return out[i:k]
    return out[i:j]


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)
