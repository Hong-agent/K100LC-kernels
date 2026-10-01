#!/usr/bin/env python3
"""从 HSACO metadata 生成 Python 侧内核目录 catalog.json。"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from amdgpu_meta import read_metadata  # noqa: E402
from hsaco_names import demangled_base, norm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hsaco", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    args = ap.parse_args()
    meta = read_metadata(args.hsaco.read_bytes())
    kernels = []
    for k in sorted(meta.get("amdhsa.kernels", []), key=lambda x: x[".name"]):
        a_list = []
        for a in k.get(".args", []):
            vk = str(a.get(".value_kind", "by_value"))
            a_list.append({"off": int(a[".offset"]), "size": int(a[".size"]),
                           "kind": vk})
        kernels.append({
            "name": k[".name"],
            "lookup": norm(demangled_base(k[".name"])),
            "args": a_list,
            "kernarg_size": int(k.get(".kernarg_segment_size", 0)),
            "group_segment": int(k.get(".group_segment_fixed_size", 0)),
            "private_segment": int(k.get(".private_segment_fixed_size", 0)),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"version": 1, "kernels": kernels},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {args.out}: {len(kernels)} kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
