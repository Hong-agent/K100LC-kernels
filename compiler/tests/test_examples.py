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


def check_f32_cmp(out_dir: pathlib.Path) -> None:
    """回归：f32 的 6 个比较运算符都要能编、且语义正确。

    v1.6.2 之前 `v_cmp_*_f32_e32` 的第一个源在编码表里是 ssrc（标量/内联
    常量），两个 VGPR 相比汇编器直接拒绝——也就是**所有 f32 比较都编不过**。
    现在统一走 `v_cmp_lt_f32_e64 vcc, x, y`，其余运算符用交换操作数 / 取反
    掩码拼出来。
    """
    ops = {"<": np.less, "<=": np.less_equal, ">": np.greater,
           ">=": np.greater_equal, "==": np.equal, "!=": np.not_equal}
    n = 128
    x = np.linspace(-1, 1, n, dtype=np.float32)
    x[5] = x[9]                      # 造一对相等的，覆盖 == / !=
    y = np.roll(x, 3)
    for op, ref_fn in ops.items():
        src = f"""
def cmptest(x: ptr[f32], y: ptr[f32], out: ptr[f32], n: u32):
    i = gid()
    if i < n:
        if x[i] {op} y[i]:
            out[i] = 1.0
        else:
            out[i] = 0.0
"""
        h = compile_source(src, out_dir, "cmptest")[0]
        o = run_one(h, "cmptest",
                    [{"buffer": "x"}, {"buffer": "y"}, {"buffer": "out"},
                     {"scalar": {"dtype": "u32", "value": n}}],
                    {"x": {"dtype": "f32", "values": x.tolist()},
                     "y": {"dtype": "f32", "values": y.tolist()},
                     "out": {"dtype": "f32", "values": [0.0] * n}},
                    grid=n, workgroup=64)
        got = np.array(o["out"], np.float32)
        ref = ref_fn(x, y).astype(np.float32)
        assert np.array_equal(got, ref), f"f32 {op}: 不等个数 {int((got != ref).sum())}"
    print("f32 比较（6 个运算符）ok")


def check_varying_else(out_dir: pathlib.Path) -> None:
    """回归：varying 条件的 if/else（v1.6.2 之前只支持无 else）+ min/max。"""
    src = """
def vsel(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        v = x[i]
        if v > 0.0:
            y[i] = min(v, 3.0)
        else:
            y[i] = max(v, -2.0) * 10.0
"""
    h = compile_source(src, out_dir, "vsel")[0]
    n = 256
    x = np.linspace(-6, 6, n, dtype=np.float32)
    o = run_one(h, "vsel",
                [{"buffer": "x"}, {"buffer": "y"},
                 {"scalar": {"dtype": "u32", "value": n}}],
                {"x": {"dtype": "f32", "values": x.tolist()},
                 "y": {"dtype": "f32", "values": [0.0] * n}},
                grid=n, workgroup=64)
    got = np.array(o["y"], np.float32)
    ref = np.where(x > 0.0, np.minimum(x, 3.0),
                   np.maximum(x, -2.0) * 10.0).astype(np.float32)
    d = float(np.abs(got - ref).max())
    assert d < 1e-5, d
    print(f"varying if/else + min/max ok (max_abs={d:.3e})")


def check_dsl_surface(out_dir: pathlib.Path) -> None:
    """把文档里写的 DSL 特性扫一遍（v1.6.3 就是这么发现 `^` 没接线、
    嵌套 varying if/else 算错的）。"""
    cases = [
        ("一元负号 + 位运算 & | ^ >>",
         """
def dsl(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        a = u32(x[i])
        b = (a & 7) | ((a >> 1) ^ 3)
        y[i] = f32(b) + (-1.0)
""",
         np.array([0, 1, 2, 3, 4, 5, 6, 7], np.float32),
         lambda a: ((a.astype(np.uint32) & 7)
                    | ((a.astype(np.uint32) >> 1) ^ 3)).astype(np.float32) - 1.0),
        ("嵌套 varying if/else",
         """
def dsl(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        if x[i] > 0.0:
            if x[i] > 2.0:
                y[i] = 2.0
            else:
                y[i] = 1.0
        else:
            y[i] = -1.0
""",
         np.array([-3, -1, 0, 0.5, 1, 2, 3, 5], np.float32),
         lambda a: np.where(a > 0, np.where(a > 2, 2.0, 1.0), -1.0)),
        ("增强赋值 += -= *= /=",
         """
def dsl(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        v = x[i]
        v += 1.0
        v *= 3.0
        v -= 0.5
        v /= 2.0
        y[i] = v
""",
         np.array([-2, -1, 0, 1, 2], np.float32),
         lambda a: ((a + 1.0) * 3.0 - 0.5) / 2.0),
        ("for range 两种写法 + break/continue 之外的全量走一遍",
         """
def dsl(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        v = x[i]
        for j in range(1, 4):
            v = v * 2.0
        for k in range(0, 2):
            v = v + 1.0
        y[i] = v
""",
         np.array([0, 1, 2], np.float32),
         lambda a: (a * 8.0) + 2.0),
        ("and（两个 varying 掩码合并）",
         """
def dsl(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        y[i] = 0.0
        if x[i] > 0.0 and x[i] < 2.0:
            y[i] = 1.0
""",
         np.array([-2, -1, 0, 0.5, 1, 2, 3], np.float32),
         lambda a: np.where((a > 0) & (a < 2), 1.0, 0.0)),
        ("or（uniform + varying 混合）",
         """
def dsl(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        y[i] = 0.0
        if n > 4 and (x[i] < 0.0 or x[i] > 2.0):
            y[i] = 1.0
""",
         np.array([-2, -1, 0, 0.5, 1, 2, 3], np.float32),
         lambda a: np.where((a < 0) | (a > 2), 1.0, 0.0)),
    ]
    for title, src, x, ref_fn in cases:
        n = len(x)
        h = compile_source(src, out_dir, "dsl")[0]
        o = run_one(h, "dsl",
                    [{"buffer": "x"}, {"buffer": "y"},
                     {"scalar": {"dtype": "u32", "value": n}}],
                    {"x": {"dtype": "f32", "values": list(map(float, x))},
                     "y": {"dtype": "f32", "values": [0.0] * n}},
                    grid=n, workgroup=64)
        got = np.array(o["y"], np.float32)
        ref = np.asarray(ref_fn(x), np.float32)
        d = float(np.abs(got - ref).max())
        assert d < 1e-4, f"{title}: max_abs={d}"
    print(f"DSL 特性扫描（{len(cases)} 项）ok")


def check_int_cmp_and_divmod(out_dir: pathlib.Path) -> None:
    """回归：整数比较 6 个运算符（u32 / s32）+ 2 的幂常数除与取模。

    v1.6.4 之前编码表里两个源都能是 VGPR 的整数比较只有
    `v_cmp_lt_u32_e64` / `v_cmp_gt_i32_e64` / `v_cmp_eq_u32_e32`，
    所以 u32 的 `<= >= !=`、s32 的 `<= > >= == !=` 全都编不过。
    """
    n = 64
    x = np.arange(n, dtype=np.float32)
    xs = (np.arange(n, dtype=np.float32) - 20.0)     # 有负数，覆盖 s32
    cmp_ops = {"<": np.less, "<=": np.less_equal, ">": np.greater,
               ">=": np.greater_equal, "==": np.equal, "!=": np.not_equal}
    for cast, vals, conv in (("u32", x, lambda a: a.astype(np.uint32)),
                             ("s32", xs, lambda a: a.astype(np.int32))):
        for op, ref_fn in cmp_ops.items():
            src = f"""
def icmp(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        if {cast}(x[i]) {op} {cast}(8.0):
            y[i] = 1.0
        else:
            y[i] = 0.0
"""
            h = compile_source(src, out_dir, "icmp")[0]
            o = run_one(h, "icmp",
                        [{"buffer": "x"}, {"buffer": "y"},
                         {"scalar": {"dtype": "u32", "value": n}}],
                        {"x": {"dtype": "f32", "values": vals.tolist()},
                         "y": {"dtype": "f32", "values": [0.0] * n}},
                        grid=n, workgroup=64)
            got = np.array(o["y"], np.float32)
            ref = ref_fn(conv(vals), conv(np.full(n, 8.0, np.float32))).astype(np.float32)
            assert np.array_equal(got, ref), f"{cast} {op} 不一致"
    div_src = """
def idiv(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        a = u32(x[i])
        y[i] = f32((a / 4) + (a % 8) * 100)
"""
    h = compile_source(div_src, out_dir, "idiv")[0]
    o = run_one(h, "idiv",
                [{"buffer": "x"}, {"buffer": "y"},
                 {"scalar": {"dtype": "u32", "value": n}}],
                {"x": {"dtype": "f32", "values": x.tolist()},
                 "y": {"dtype": "f32", "values": [0.0] * n}},
                grid=n, workgroup=64)
    got = np.array(o["y"], np.float32)
    u = x.astype(np.uint32)
    ref = ((u // 4) + (u % 8) * 100).astype(np.float32)
    assert np.array_equal(got, ref), "int / 与 % 不一致"
    try:
        compile_source("def bad(x: ptr[f32], y: ptr[f32]):\n"
                       "    y[0] = f32(u32(x[0]) % 3)\n", out_dir, "bad")
        raise AssertionError("非 2 的幂除数应当报错")
    except Exception as exc:                       # noqa: BLE001
        assert "2 的幂" in str(exc), exc
    print("整数比较（12 项）+ 2 的幂除/取模 ok")


def check_math_builtins(out_dir: pathlib.Path) -> None:
    """回归：取整族内建 + `ubyte` + 整数 max/min。

    `max`/`min` 原来不管操作数类型都发 `v_max_f32_e32`——整数会**静默按
    浮点比大小**（`max(10, 9.5)` 之类的小整数恰好看不出来，大整数就错了）。
    现在按类型选 `v_max_u32` / `v_min_u32`，s32 直接报错。
    """
    cases = [
        ("floor", "floor(x[i])", lambda a: np.floor(a)),
        ("ceil", "ceil(x[i])", lambda a: np.ceil(a)),
        ("trunc", "trunc(x[i])", lambda a: np.trunc(a)),
        ("rint", "rint(x[i])", lambda a: np.rint(a)),
        ("fract", "fract(x[i])", lambda a: a - np.floor(a)),
    ]
    n = 16
    x = np.linspace(-3.2, 3.7, n, dtype=np.float32)
    for name, expr, ref_fn in cases:
        src = f"""
def mb(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        y[i] = {expr}
"""
        h = compile_source(src, out_dir, "mb")[0]
        o = run_one(h, "mb",
                    [{"buffer": "x"}, {"buffer": "y"},
                     {"scalar": {"dtype": "u32", "value": n}}],
                    {"x": {"dtype": "f32", "values": x.tolist()},
                     "y": {"dtype": "f32", "values": [0.0] * n}},
                    grid=n, workgroup=64)
        d = float(np.abs(np.array(o["y"], np.float32) - ref_fn(x).astype(np.float32)).max())
        assert d < 1e-5, f"{name}: max_abs={d}"

    u = np.array([0, 1, 255, 256, 257, 300, 1024, 4095], np.float32)
    int_cases = [
        ("ubyte", "ubyte(u32(x[i]))",
         lambda a: (a.astype(np.uint32) & 0xFF).astype(np.float32)),
        ("u32 max", "f32(max(u32(x[i]), 300))",
         lambda a: np.maximum(a.astype(np.uint32), np.uint32(300)).astype(np.float32)),
        ("u32 min", "f32(min(u32(x[i]), 300))",
         lambda a: np.minimum(a.astype(np.uint32), np.uint32(300)).astype(np.float32)),
    ]
    for name, expr, ref_fn in int_cases:
        src = f"""
def mb(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        y[i] = {expr}
"""
        h = compile_source(src, out_dir, "mb")[0]
        o = run_one(h, "mb",
                    [{"buffer": "x"}, {"buffer": "y"},
                     {"scalar": {"dtype": "u32", "value": len(u)}}],
                    {"x": {"dtype": "f32", "values": u.tolist()},
                     "y": {"dtype": "f32", "values": [0.0] * len(u)}},
                    grid=len(u), workgroup=64)
        got = np.array(o["y"], np.float32)
        assert np.array_equal(got, ref_fn(u)), f"{name}: {got} != {ref_fn(u)}"
    print("取整族内建 + ubyte + 整数 max/min ok")


def check_while(out_dir: pathlib.Path) -> None:
    """回归：`while` 语句（uniform 条件）+ break / continue / `while 1`。

    varying 条件会明确报错——那需要 loop-carried 的 exec 掩码约定，
    还没定，宁可报错也不给错结果。
    """
    cases = [
        ("uniform 计数",
         """
def wk(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        c = 0
        v = x[i]
        while c < 3:
            v = v * 2.0
            c = c + 1
        y[i] = v
""",
         lambda a: a * 8.0),
        ("while + break",
         """
def wk(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        c = 0
        v = x[i]
        while c < 100:
            v = v + 1.0
            c = c + 1
            if c > 2:
                break
        y[i] = v
""",
         lambda a: a + 3.0),
        ("while + continue",
         """
def wk(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        c = 0
        v = 0.0
        while c < 5:
            c = c + 1
            if c > 2:
                continue
            v = v + 1.0
        y[i] = v
""",
         lambda a: np.full(len(a), 2.0, np.float32)),
        ("while 1 + break",
         """
def wk(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        c = 0
        while 1:
            c = c + 1
            if c > 4:
                break
        y[i] = f32(c)
""",
         lambda a: np.full(len(a), 5.0, np.float32)),
    ]
    n = 32
    x = np.arange(n, dtype=np.float32)
    for title, src, ref_fn in cases:
        h = compile_source(src, out_dir, "wk")[0]
        o = run_one(h, "wk",
                    [{"buffer": "x"}, {"buffer": "y"},
                     {"scalar": {"dtype": "u32", "value": n}}],
                    {"x": {"dtype": "f32", "values": x.tolist()},
                     "y": {"dtype": "f32", "values": [0.0] * n}},
                    grid=n, workgroup=64)
        got = np.array(o["y"], np.float32)
        ref = np.asarray(ref_fn(x), np.float32)
        d = float(np.abs(got - ref).max())
        assert d < 1e-5, f"{title}: max_abs={d} got={got[:3]} ref={ref[:3]}"
    try:
        compile_source("def bad(x: ptr[f32], y: ptr[f32]):\n"
                       "    c = 0\n"
                       "    while x[0] > 1.0:\n"
                       "        c = c + 1\n", out_dir, "bad")
        raise AssertionError("varying 条件应当报错")
    except Exception as exc:                       # noqa: BLE001
        assert "uniform" in str(exc), exc
    print("while（4 种写法）+ varying 条件报错 ok")


def check_mixed_types(out_dir: pathlib.Path) -> None:
    """回归：整数与 f32 混用时的类型提升。

    v1.8.1 之前，整数常量/变量出现在 f32 上下文里会被当成**位模式**
    （`v_mov_b32 vN, 1` 是 1.4e-45 而不是 1.0），于是下面这些写法静默算错：
    `x[i] + 1`、`x[i] * m`（m 是整数变量）、`y[i] = 1`、`x[i] > 3`、
    `min(x[i], 3)`、`sqrt(4)`、`fma(x, 2, 1)`、`fabs(-1)`、`floor(2)`……
    """
    n = 16
    x = np.arange(n, dtype=np.float32) + 1.0
    body_cases = [
        ("f32 + 整数字面量", "y[i] = x[i] + 1", lambda a: a + 1.0),
        ("f32 * 整数变量", "m = 3\n        y[i] = x[i] * m", lambda a: a * 3.0),
        ("f32 / 整数变量", "d = 2\n        y[i] = x[i] / d", lambda a: a / 2.0),
        ("f32 指针存整数常量", "y[i] = 2", lambda a: np.full(len(a), 2.0)),
        ("f32 比较 vs 整数", "y[i] = 0.0\n        if x[i] > 3:\n            y[i] = 1.0",
         lambda a: np.where(a > 3, 1.0, 0.0)),
        ("min(x, 3)", "y[i] = min(x[i], 3)", lambda a: np.minimum(a, 3.0)),
        ("max(x, 1)", "y[i] = max(x[i], 1)", lambda a: np.maximum(a, 1.0)),
        ("sqrt(4)", "y[i] = sqrt(4)", lambda a: np.full(len(a), 2.0)),
        ("rsqrt(4)", "y[i] = rsqrt(4)", lambda a: np.full(len(a), 0.5)),
        ("fma(x, 2, 1)", "y[i] = fma(x[i], 2, 1)", lambda a: a * 2 + 1),
        ("fabs(-1)", "y[i] = fabs(-1)", lambda a: np.ones(len(a))),
        ("floor(2)", "y[i] = floor(2)", lambda a: np.full(len(a), 2.0)),
        ("rint(2)", "y[i] = rint(2)", lambda a: np.full(len(a), 2.0)),
        ("s32 变量参与 f32 运算",
         "m = s32(x[i] - x[i]) - 2\n        y[i] = x[i] * f32(m)",
         lambda a: a * -2.0),
    ]
    for title, body, ref_fn in body_cases:
        src = (f"\ndef mt(x: ptr[f32], y: ptr[f32], n: u32):\n"
               f"    i = gid()\n    if i < n:\n        {body}\n")
        h = compile_source(src, out_dir, "mt")[0]
        o = run_one(h, "mt",
                    [{"buffer": "x"}, {"buffer": "y"},
                     {"scalar": {"dtype": "u32", "value": n}}],
                    {"x": {"dtype": "f32", "values": x.tolist()},
                     "y": {"dtype": "f32", "values": [0.0] * n}},
                    grid=n, workgroup=64)
        got = np.array(o["y"], np.float32)
        ref = np.asarray(ref_fn(x), np.float32)
        d = float(np.abs(got - ref).max())
        assert d < 1e-5, f"{title}: max_abs={d} got={got[:3].tolist()} ref={ref[:3].tolist()}"
    # 全 uniform 的 `while a and b`（之前会被当成 varying 拒掉）
    src = """
def mt(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        a = 0
        b = 0
        while a < 3 and b < 5:
            a = a + 1
            b = b + 1
        y[i] = f32(a * 10 + b)
"""
    h = compile_source(src, out_dir, "mt")[0]
    o = run_one(h, "mt",
                [{"buffer": "x"}, {"buffer": "y"},
                 {"scalar": {"dtype": "u32", "value": n}}],
                {"x": {"dtype": "f32", "values": x.tolist()},
                 "y": {"dtype": "f32", "values": [0.0] * n}},
                grid=n, workgroup=64)
    got = np.array(o["y"], np.float32)
    assert np.array_equal(got, np.full(n, 33.0, np.float32)), got[:4]
    print(f"混合类型提升（{len(body_cases)} 项）+ uniform while-and ok")


def check_transcendental_hazards(out_dir: pathlib.Path) -> None:
    """回归：`v_sqrt_f32` / `v_rsq_f32` / `v_exp_f32` 的 VALU 读-写冒险。

    这几个指令的源刚被 VALU 写过、或者结果立刻被后续指令读，都需要隔一条
    `s_nop 0`。少一条时**同一个 wave 里只有部分 lane 算错**（实测
    `y[i] = sqrt(4)` 时每 16 个元素里第 8~11 号是 0，看起来像"随机丢数据"）。
    """
    n = 64
    x = np.arange(n, dtype=np.float32) * 0.1 + 0.5
    cases = [
        ("sqrt(4)（字面量源）", "y[i] = sqrt(4)", lambda a: np.full(len(a), 2.0), 1e-6),
        ("rsqrt(4)（字面量源）", "y[i] = rsqrt(4)", lambda a: np.full(len(a), 0.5), 1e-6),
        ("sqrt(x)", "y[i] = sqrt(x[i])", lambda a: np.sqrt(a), 1e-6),
        ("rsqrt(x)", "y[i] = rsqrt(x[i])", lambda a: 1.0 / np.sqrt(a), 1e-6),
        ("exp(x)", "y[i] = exp(x[i])", lambda a: np.exp(a), 1e-5),
        ("sqrt(9) + rsqrt(4)", "y[i] = sqrt(9) + rsqrt(4)",
         lambda a: np.full(len(a), 3.5), 1e-6),
        ("exp(x*0.1) + sqrt(2)", "y[i] = exp(x[i] * 0.1) + sqrt(2)",
         lambda a: np.exp(a * 0.1) + np.sqrt(2.0), 1e-5),
        ("1/x（v_rcp 通路）", "y[i] = 1.0 / x[i]", lambda a: 1.0 / a, 1e-6),
        ("x/3（常量倒数）", "y[i] = x[i] / 3.0", lambda a: a / 3.0, 1e-6),
    ]
    for title, body, ref_fn, rtol in cases:
        src = (f"\ndef th(x: ptr[f32], y: ptr[f32], n: u32):\n"
               f"    i = gid()\n    if i < n:\n        {body}\n")
        h = compile_source(src, out_dir, "th")[0]
        o = run_one(h, "th",
                    [{"buffer": "x"}, {"buffer": "y"},
                     {"scalar": {"dtype": "u32", "value": n}}],
                    {"x": {"dtype": "f32", "values": x.tolist()},
                     "y": {"dtype": "f32", "values": [0.0] * n}},
                    grid=n, workgroup=64)
        got = np.array(o["y"], np.float32)
        ref = np.asarray(ref_fn(x), np.float32)
        rel = float(np.abs(got - ref).max() / max(1e-9, float(np.abs(ref).max())))
        assert rel < rtol, f"{title}: rel={rel:.3e} got={got[:6].tolist()}"
    print(f"超越函数 hazard（{len(cases)} 项）ok")


def main() -> int:
    out = pathlib.Path("/tmp/k100lc_compiler_test")
    check_vadd(out)
    check_silu(out)
    check_axpy(out)
    check_loop(out)
    check_many_vars(out)
    check_long_expr(out)
    check_f32_cmp(out)
    check_varying_else(out)
    check_dsl_surface(out)
    check_int_cmp_and_divmod(out)
    check_math_builtins(out)
    check_while(out)
    check_mixed_types(out)
    check_transcendental_hazards(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
