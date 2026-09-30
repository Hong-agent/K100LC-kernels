#!/usr/bin/env python3
"""编译 compiler/examples 下的内核并和 numpy 对账。"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
sys.path.insert(0, str(ROOT / "tools"))

from k100lc_compiler import compile_file, compile_source  # noqa: E402
from kernel_lab import run_one  # noqa: E402


def check_vadd(out_dir: pathlib.Path) -> None:
    h = compile_file(ROOT / "compiler/examples/vadd.kkl", out_dir)[0]
    n = 1024
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(n, dtype=np.float32), rng.standard_normal(n, dtype=np.float32)
    o = run_one(h, "vadd",
                [{"buffer": "a"}, {"buffer": "b"}, {"buffer": "out"},
                 {"scalar": {"dtype": "u32", "value": n}}],
                {"a": {"dtype": "f32", "values": a.tolist()},
                 "b": {"dtype": "f32", "values": b.tolist()},
                 "out": {"dtype": "f32", "values": [0.0] * n}},
                grid=n, workgroup=64)
    assert np.abs(np.array(o["out"], np.float32) - (a + b)).max() == 0.0
    print("vadd ok")


def check_silu(out_dir: pathlib.Path) -> None:
    h = compile_file(ROOT / "compiler/examples/silu.kkl", out_dir)[0]
    n = 512
    x = np.linspace(-4, 4, n, dtype=np.float32)
    o = run_one(h, "silu",
                [{"buffer": "x"}, {"buffer": "y"},
                 {"scalar": {"dtype": "u32", "value": n}}],
                {"x": {"dtype": "f32", "values": x.tolist()},
                 "y": {"dtype": "f32", "values": [0.0] * n}},
                grid=n, workgroup=64)
    ref = (x / (1.0 + np.exp(-x))).astype(np.float32)
    d = np.abs(np.array(o["y"], np.float32) - ref).max()
    assert d < 1e-4, d
    print(f"silu ok (max_abs={d:.3e})")


def check_axpy(out_dir: pathlib.Path) -> None:
    h = compile_file(ROOT / "compiler/examples/axpy.kkl", out_dir)[0]
    n = 256
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal(n, dtype=np.float32), rng.standard_normal(n, dtype=np.float32)
    alpha = 0.75
    o = run_one(h, "axpy",
                [{"buffer": "a"}, {"buffer": "x"}, {"buffer": "y"},
                 {"scalar": {"dtype": "u32", "value": n}},
                 {"scalar": {"dtype": "f32", "value": alpha}}],
                {"a": {"dtype": "f32", "values": x.tolist()},
                 "x": {"dtype": "f32", "values": x.tolist()},
                 "y": {"dtype": "f32", "values": y.tolist()}},
                grid=n, workgroup=64)
    d = np.abs(np.array(o["y"], np.float32) - (alpha * x + y)).max()
    assert d < 1e-5, d
    print(f"axpy ok (max_abs={d:.3e})")


def check_loop(out_dir: pathlib.Path) -> None:
    src = """
def scaleloop(x: ptr[f32], y: ptr[f32], n: u32, s: f32):
    i = gid()
    if i < n:
        v = x[i]
        for k in range(0, 4):
            v = v * s
        y[i] = v
"""
    h = compile_source(src, out_dir, "scaleloop")[0]
    n = 128
    x = np.linspace(-2, 2, n, dtype=np.float32)
    s = 1.5
    o = run_one(h, "scaleloop",
                [{"buffer": "x"}, {"buffer": "y"},
                 {"scalar": {"dtype": "u32", "value": n}},
                 {"scalar": {"dtype": "f32", "value": s}}],
                {"x": {"dtype": "f32", "values": x.tolist()},
                 "y": {"dtype": "f32", "values": [0.0] * n}},
                grid=n, workgroup=64)
    d = np.abs(np.array(o["y"], np.float32) - x * s ** 4).max()
    assert d < 1e-5, d
    print(f"loop ok (max_abs={d:.3e})")


def main() -> int:
    out = pathlib.Path("/tmp/k100lc_compiler_test")
    check_vadd(out)
    check_silu(out)
    check_axpy(out)
    check_loop(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
