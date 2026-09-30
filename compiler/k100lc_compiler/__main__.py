from __future__ import annotations

import argparse
import ast
import pathlib
import sys

from .core import compile_file


def main() -> int:
    ap = argparse.ArgumentParser(prog="k100lc_compiler")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="编译 .kkl → .s/.hsaco")
    b.add_argument("source", type=pathlib.Path)
    b.add_argument("-o", "--out", type=pathlib.Path, default=pathlib.Path("build/compiler"))
    b.add_argument("--kernel", default=None)
    b.add_argument("--emit-asm", action="store_true")
    l = sub.add_parser("list", help="列出文件里的内核")
    l.add_argument("source", type=pathlib.Path)
    args = ap.parse_args()

    if args.cmd == "list":
        tree = ast.parse(args.source.read_text(encoding="utf-8"))
        for n in tree.body:
            if isinstance(n, ast.FunctionDef):
                print(n.name)
        return 0

    outs = compile_file(args.source, args.out, args.kernel)
    for p in outs:
        print(p)
        if args.emit_asm:
            print(p.with_suffix(".s").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
