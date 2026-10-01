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


def check_many_vars(out_dir: pathlib.Path) -> None:
    """回归：局部变量多到碰临时寄存器区时，必须仍然算对。

    以前变量区和临时区都从 v64 附近往上涨，62 个局部变量起两者重叠，
    生成的内核**不报错但算错**（v1.4.0 修）。
    """
    nvar = 150
    body = "\n".join(f"    t{i} = x[{i}] + 1.0" for i in range(nvar))
    total = "\n".join(f"    acc = acc + t{i}" for i in range(nvar))
    src = f"""
def many(x: ptr[f32], y: ptr[f32], n: u32):
{body}
    acc = 0.0
{total}
    y[0] = acc
"""
    h = compile_source(src, out_dir, "many")[0]
    x = np.arange(256, dtype=np.float32)
    o = run_one(h, "many",
                [{"buffer": "x"}, {"buffer": "y"},
                 {"scalar": {"dtype": "u32", "value": 256}}],
                {"x": {"dtype": "f32", "values": x.tolist()},
                 "y": {"dtype": "f32", "values": [0.0] * 4}},
                grid=1, workgroup=64)
    got, ref = float(o["y"][0]), float(np.sum(x[:nvar] + 1.0))
    assert abs(got - ref) < 1e-2, f"{got} != {ref}"
    print(f"many_vars({nvar}) ok (got={got:.1f})")


def check_long_expr(out_dir: pathlib.Path) -> None:
    """回归：语句内临时值必须能复用。

    `tmp_v` 原来是单调递增的——`x[0] + x[1] + … + x[n]` 这种长表达式每个
    `+` 都占一个新寄存器，60 项就把 v245 用光（其实同时只有 2~3 个是活的）。
    v1.5.0 加了临时寄存器池：用完即归还。
    """
    nterm = 300
    terms = " + ".join(f"x[{i}]" for i in range(nterm))
    src = f"""
def longexpr(x: ptr[f32], y: ptr[f32], n: u32):
    y[0] = {terms}
"""
    h = compile_source(src, out_dir, "longexpr")[0]
    x = np.arange(2048, dtype=np.float32)
    o = run_one(h, "longexpr",
                [{"buffer": "x"}, {"buffer": "y"},
                 {"scalar": {"dtype": "u32", "value": 2048}}],
                {"x": {"dtype": "f32", "values": x.tolist()},
                 "y": {"dtype": "f32", "values": [0.0] * 4}},
                grid=1, workgroup=64)
    got, ref = float(o["y"][0]), float(np.sum(x[:nterm]))
    assert abs(got - ref) <= 1e-6 * max(1.0, abs(ref)), f"{got} != {ref}"
    print(f"long_expr({nterm} 项) ok")


def main() -> int:
    out = pathlib.Path("/tmp/k100lc_compiler_test")
    check_vadd(out)
    check_silu(out)
    check_axpy(out)
    check_loop(out)
    check_many_vars(out)
    check_long_expr(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
