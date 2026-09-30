#!/usr/bin/env python3
"""把一个 .kkl 编译结果导出/安装到内核包。

    python3 compiler/tools/export_kernel.py compiler/examples/q6k_dequant.kkl \
        --name q6k_dequant --install

--install：把生成的 .s 放进 kernels/asm/k_new/，并在 kernels/kernel_spec.json
追加 spec 条目；之后 `tools/build_all.sh` 会把它并入全量 HSACO + catalog。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
sys.path.insert(0, str(ROOT / "tools"))

from amdgpu_meta import read_metadata  # noqa: E402
from k100lc_compiler import compile_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=pathlib.Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--install", action="store_true")
    args = ap.parse_args()
    out = ROOT / "build" / "compiled"
    hsaco = compile_file(args.source, out, args.name)[0]
    cat = json.loads((out / f"{args.name}.catalog.json").read_text(encoding="utf-8"))
    meta = read_metadata(hsaco.read_bytes())
    k = meta["amdhsa.kernels"][0]
    entry = {
        "name": args.name,
        "asm": f"k_new/{args.name}.s",
        "args": cat["kernels"][0]["args"],
        "kernarg_size": int(k[".kernarg_segment_size"]),
        "kernarg_align": int(k.get(".kernarg_segment_align", 8)),
        "sgpr_count": int(k.get(".sgpr_count", 4)),
        "vgpr_count": int(k.get(".vgpr_count", 4)),
        "group_segment": int(k.get(".group_segment_fixed_size", 0)),
        "private_segment": int(k.get(".private_segment_fixed_size", 0)),
    }
    print(json.dumps(entry, ensure_ascii=False, indent=1))
    if args.install:
        dst_s = ROOT / "kernels" / "asm" / "k_new" / f"{args.name}.s"
        dst_s.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / f"{args.name}.s", dst_s)
        spec_path = ROOT / "kernels" / "kernel_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec = [x for x in spec if x.get("name") != args.name]
        spec.append(entry)
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print(f"installed {dst_s} + kernel_spec.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
