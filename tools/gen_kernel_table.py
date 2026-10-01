#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从自研 HSACO 生成运行时的内核表头 `nodtk_kernels.h`。

    python3 tools/gen_kernel_table.py build/k100lc_kernels.hsaco build/nodtk_kernels.h

`runtime/hsa_rt.{h,cpp}` 依赖这个表：名字、每个显式参数的 (offset,size)、
kernarg/group/private 段大小。数据全部来自 HSACO 自己的 metadata。
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from amdgpu_meta import read_metadata  # noqa: E402
from hsaco_names import HID, demangled_base, norm  # noqa: E402


def build_table(hsaco: pathlib.Path) -> tuple[str, dict]:
    meta = read_metadata(hsaco.read_bytes())
    kernels = meta.get("amdhsa.kernels", [])
    entries, by_lookup = [], {}
    for k in sorted(kernels, key=lambda x: x[".name"]):
        args = []
        for a in k.get(".args", []):
            vk = str(a.get(".value_kind", "by_value"))
            kind = 0 if vk in ("by_value", "global_buffer") else HID.get(vk)
            if kind is None:
                raise SystemExit(f"{k['.name']}: 不认识的参数类型 {vk}")
            args.append((int(a[".offset"]), int(a[".size"]), kind))
        name = k[".name"]
        lookup = norm(demangled_base(name))
        if lookup in by_lookup:
            raise SystemExit(f"内核名撞车：{lookup}（{name} 与 {by_lookup[lookup]}）")
        idx = len(entries)
        by_lookup[lookup] = name
        entries.append({
            "name": name, "lookup": lookup, "args": args,
            "kernarg_size": int(k.get(".kernarg_segment_size", 0)),
            "group_size": int(k.get(".group_segment_fixed_size", 0)),
            "private_size": int(k.get(".private_segment_fixed_size", 0)),
        })
    lines = ["// 自动生成：tools/gen_kernel_table.py（数据来自自研 HSACO 的 metadata）",
             "#pragma once", "#include <cstdint>", "",
             "#define MAX_KERNARG 1024",
             "struct RtArg { uint32_t off; uint32_t size; uint8_t kind; };",
             "struct RtKernel {",
             "    const char* name;      // mangled 名（HSACO 里的符号）",
             "    const char* lookup;    // 源码里的写法（已去掉空白）",
             "    uint32_t nargs;        // 参数个数（含隐藏）",
             "    const RtArg* args;",
             "    uint32_t kernarg_size, group_size, private_size;",
             "};", ""]
    for i, e in enumerate(entries):
        lines.append(f"static const RtArg k_args_{i}[] = {{")
        for (off, size, kind) in e["args"]:
            label = "by_value/global" if kind == 0 else \
                [n for n, v in HID.items() if v == kind][0]
            lines.append(f"    {{{off}u, {size}u, {kind}}},   // {label}")
        lines.append("};")
        lines.append("")
    lines.append("static const RtKernel k_table[] = {")
    for i, e in enumerate(entries):
        lines.append(f'    {{"{e["name"]}", "{e["lookup"]}", {len(e["args"])}u, k_args_{i}, '
                     f'{e["kernarg_size"]}u, {e["group_size"]}u, {e["private_size"]}u}},')
    lines.append("};")
    lines.append(f"static const int k_table_n = {len(entries)};")
    return "\n".join(lines) + "\n", by_lookup


def main() -> int:
    hsaco = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/k100lc_kernels.hsaco")
    out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "build/nodtk_kernels.h")
    out.parent.mkdir(parents=True, exist_ok=True)
    header, lookup = build_table(hsaco)
    out.write_text(header, encoding="utf-8")
    print(f"-> {out}：{len(lookup)} 个内核")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
