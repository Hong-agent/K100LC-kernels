#!/usr/bin/env python3
"""编译 vadd.kkl 并用常驻 Runtime 动态启动（不重建 libfm_engine）。"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "compiler"))

from k100lc_compiler import compile_file  # noqa: E402
from k100lc_kernels import Runtime  # noqa: E402


def main() -> int:
    out = pathlib.Path("/tmp/k100lc_vadd")
    hsaco = compile_file(ROOT / "compiler/examples/vadd.kkl", out)[0]
    rt = Runtime(hsaco=hsaco, catalog=hsaco.with_suffix(".catalog.json"))
    n = 256
    a = np.arange(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    pa, pb, po = rt.alloc(n * 4), rt.alloc(n * 4), rt.alloc(n * 4)
    rt.upload(pa, a)
    rt.upload(pb, b)
    rt.memset(po, 0, n * 4)
    rt.launch_dyn("vadd", (n + 63) // 64, 1, 64, 1, [pa, pb, po, n])
    rt.sync()
    got = rt.download(po, n, np.float32)
    print("vadd max_abs", float(np.abs(got - (a + b)).max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
