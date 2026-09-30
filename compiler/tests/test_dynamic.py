#!/usr/bin/env python3
"""动态 Runtime 启动编译器产物的最小测试（单 HSACO/进程）。"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "compiler"))

from k100lc_compiler import compile_source  # noqa: E402
from k100lc_kernels import Runtime  # noqa: E402


def main() -> int:
    src = """
def dadd(a: ptr[f32], b: ptr[f32], o: ptr[f32], n: u32):
    i = gid()
    if i < n:
        o[i] = a[i] + b[i]
"""
    out = pathlib.Path("/tmp/k100lc_dynamic")
    hsaco = compile_source(src, out, "dadd")[0]
    rt = Runtime(hsaco=hsaco, catalog=hsaco.with_suffix(".catalog.json"))
    n = 333
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal(n, dtype=np.float32), rng.standard_normal(n, dtype=np.float32)
    pa, pb, po = rt.alloc(n * 4), rt.alloc(n * 4), rt.alloc(n * 4)
    rt.upload(pa, a)
    rt.upload(pb, b)
    rt.memset(po, 0, n * 4)
    rt.launch_dyn("dadd", (n + 63) // 64, 1, 64, 1, [pa, pb, po, n])
    rt.sync()
    d = np.abs(rt.download(po, n, np.float32) - (a + b)).max()
    assert d == 0.0, d
    print("dynamic runtime ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
