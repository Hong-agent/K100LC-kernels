#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从自研 HSACO 生成运行时的内核表头 `nodtk_kernels.h`。

    python3 tools/gen_kernel_table.py build/flashmoe.hsaco build/nodtk/nodtk_kernels.h

`runtime/hsa_rt.{h,cpp}` 依赖这个表：名字、每个显式参数的 (offset,size)、
kernarg/group/private 段大小。数据全部来自 HSACO 自己的 metadata。
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from gen_nodtk import build_table  # noqa: E402


def main() -> int:
    hsaco = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/flashmoe.hsaco")
    out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "build/nodtk/nodtk_kernels.h")
    out.parent.mkdir(parents=True, exist_ok=True)
    header, lookup = build_table(hsaco)
    out.write_text(header, encoding="utf-8")
    print(f"-> {out}：{len(lookup)} 个内核")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
