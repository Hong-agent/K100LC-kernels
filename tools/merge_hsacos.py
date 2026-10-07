#!/usr/bin/env python3
"""Merge multiple k100lc-asm-generated HSACOs into one code object."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import msgpack

from make_hsaco_multi import build_elf


SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")


def parse_sections(data: bytes) -> dict[str, dict]:
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    entsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    sections = []
    for i in range(shnum):
        values = SECTION_HEADER.unpack_from(data, shoff + i * entsize)
        sections.append({
            "name_off": values[0], "type": values[1], "flags": values[2],
            "addr": values[3], "offset": values[4], "size": values[5],
            "link": values[6], "entsize": values[9],
        })
    shstr = sections[shstrndx]
    names = data[shstr["offset"]:shstr["offset"] + shstr["size"]]
    for sec in sections:
        end = names.find(b"\0", sec["name_off"])
        sec["name"] = names[sec["name_off"]:end].decode()
        sec["data"] = data[sec["offset"]:sec["offset"] + sec["size"]]
    return {sec["name"]: sec for sec in sections}


def parse_symbols(sections: dict[str, dict]) -> dict[str, dict]:
    dynsym = sections[".dynsym"]
    dynstr = sections[".dynstr"]["data"]
    out = {}
    for i in range(dynsym["size"] // dynsym["entsize"]):
        off = i * dynsym["entsize"]
        st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from(
            "<IBBHQQ", dynsym["data"], off)
        end = dynstr.find(b"\0", st_name)
        name = dynstr[st_name:end].decode()
        if name:
            out[name] = {"value": st_value, "size": st_size,
                         "shndx": st_shndx, "info": st_info}
    return out


def parse_kernels(path: Path) -> list[dict]:
    data = path.read_bytes()
    sections = parse_sections(data)
    symbols = parse_symbols(sections)
    note = sections[".note"]["data"]
    namesz, descsz, ntype = struct.unpack_from("<III", note, 0)
    start = 12 + ((namesz + 3) // 4) * 4
    meta = msgpack.unpackb(note[start:start + descsz], raw=False)
    text = sections[".text"]
    kernels = []
    for k in meta["amdhsa.kernels"]:
        name = k[".name"]
        sym = symbols[name]
        text_off = text["offset"] + (sym["value"] - text["addr"])
        code = data[text_off:text_off + sym["size"]]
        kernels.append({
            "name": name,
            "code": code,
            "args": k[".args"],
            "kernarg_size": k[".kernarg_segment_size"],
            "kernarg_align": k[".kernarg_segment_align"],
            "sgpr_count": k[".sgpr_count"],
            "vgpr_count": k[".vgpr_count"],
            "group_segment": k[".group_segment_fixed_size"],
            "private_segment": k[".private_segment_fixed_size"],
        })
    return kernels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--allow-duplicates", action="store_true",
                    help="允许同名内核（后者覆盖符号；默认拒绝，见下面的注释）")
    args = ap.parse_args()
    kernels = []
    # 同名内核合并后会得到**一个** dynsym 条目（后者覆盖前者），但两份代码都留在
    # .text 里：调用方拿到的名字指向的不是它以为的那份代码。这类冲突在真机上表现
    # 为「改了源码没反应」——实测在另一套移植运行时上，把内核输出改成常量也照样
    # 返回旧结果，最后才发现目录里有一份同名的旧 HSACO。
    origin: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    for path in args.inputs:
        for k in parse_kernels(path):
            if k["name"] in origin:
                duplicates.append((k["name"], origin[k["name"]], str(path)))
            else:
                origin[k["name"]] = str(path)
            kernels.append(k)
    if duplicates and not args.allow_duplicates:
        for name, first, second in duplicates:
            print(f"重复内核名：{name}\n  先出现：{first}\n  又出现：{second}", file=sys.stderr)
        print(f"共 {len(duplicates)} 处冲突；确有同名覆盖的意图时加 --allow-duplicates。",
              file=sys.stderr)
        raise SystemExit(1)
    args.output.write_bytes(build_elf(kernels))
    print(f"wrote {args.output}: merged {len(kernels)} kernels")
    if duplicates:
        print(f"警告：{len(duplicates)} 处同名内核已按顺序覆盖", file=sys.stderr)
    for k in kernels:
        print(f"  {k['name']}: text={len(k['code'])} kernarg={k['kernarg_size']}")


if __name__ == "__main__":
    main()
