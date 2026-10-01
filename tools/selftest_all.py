#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prebuilt 内核包全内核对账基线（不重新编译任何内核）。

各生成器脚本（`tools/gen_*.py`）都有自己的自检，但它们验证的是**现场重新
汇编出来的那份 HSACO**。真正发货、用户 `Runtime()` 直接加载的是
`prebuilt/k100lc_kernels.hsaco` + `prebuilt/libfm_engine.so`；那两份产物到
目前没有任何整体验证。这个脚本补上这一环：

    合成输入 → `Runtime.launch`（就是用户会走的那条路径）→ 与 NumPy /
    `tools/iq_dequant.py` 的参考实现对账。

用法：

    source env.sh
    python3 tools/selftest_all.py                  # 全部用例
    python3 tools/selftest_all.py --group gemv     # 只看某一族
    python3 tools/selftest_all.py --json build/selftest.json

每个用例返回 `(ok, 指标, 说明)`；退出码非零表示有失败项。
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

import iq_dequant  # noqa: E402
from k100lc_kernels import Runtime  # noqa: E402


# ---------------------------------------------------------------------------
# 登记机制
# ---------------------------------------------------------------------------
CASES: list[tuple[str, str, object]] = []


def case(group: str, name: str):
    def deco(fn):
        CASES.append((group, name, fn))
        return fn
    return deco


def div_magic(d: int, max_i: int) -> int:
    """与 `tools/gen_gemv_qdot.py::div_magic` 相同：`(i*M)>>32 == i//d`。"""
    d = int(d)
    m = (1 << 32) // d + 1
    if m >= (1 << 32):
        raise ValueError(f"div_magic: d={d} 太大")
    i = np.arange(max(1, int(max_i)), dtype=np.uint64)
    if not np.array_equal((i * np.uint64(m)) >> np.uint64(32), i // np.uint64(d)):
        raise ValueError(f"div_magic: d={d} max_i={max_i} 无法精确表示")
    return int(m)


class Ctx:
    """一个用例的设备缓冲集合；退出时统一释放。"""

    def __init__(self, rt: Runtime):
        self.rt = rt
        self.ptrs: list[int] = []

    def buf(self, data) -> int:
        if isinstance(data, np.ndarray):
            nbytes = int(data.nbytes)
        else:
            data = bytes(data)
            nbytes = len(data)
        p = self.rt.alloc(max(nbytes, 4))
        self.rt.upload(p, data)
        self.ptrs.append(p)
        return p

    def out(self, count: int, dtype=np.float32) -> int:
        a = np.zeros(int(count), dtype=dtype)
        return self.buf(a)

    def get(self, p: int, count: int, dtype=np.float32) -> np.ndarray:
        self.rt.sync()
        return self.rt.download(p, int(count), dtype)

    def launch(self, name: str, grid: int, workgroup: int, argv) -> None:
        self.rt.launch(name, int(grid), int(workgroup), list(argv))
        self.rt.sync()

    def close(self) -> None:
        for p in self.ptrs:
            self.rt.free(p)
        self.ptrs.clear()


def judge(metric: float, ref, atol: float = 1e-4, rtol: float = 1e-5):
    """返回 `(指标, 是否通过)`；容差按参考值的幅值缩放。"""
    scale = max(1.0, float(np.abs(np.asarray(ref)).max()))
    return float(metric), bool(float(metric) <= atol + rtol * scale)


# ---------------------------------------------------------------------------
# 合成量化块
# ---------------------------------------------------------------------------
# 每种 GGUF 编码里 f16 字段的偏移；随机字节会让 f16 指数变成 inf/nan，
# 所以这些位置固定成温和值，其余字节全随机。
F16_FIELDS = {
    "Q2_0": [(0, 0.02)], "Q4_0": [(0, 0.02)], "Q8_0": [(0, 0.02)],
    "IQ4_NL": [(0, 0.02)], "IQ4_XS": [(0, 0.02)],
    "IQ2_S": [(0, 0.02)], "IQ3_S": [(0, 0.02)], "IQ3_XXS": [(0, 0.02)],
    "Q4_K": [(0, 0.02), (2, 0.01)], "Q5_K": [(0, 0.02), (2, 0.01)],
    "Q6_K": [(208, 0.02)],
}


def synth_blocks(type_name: str, nblocks: int, seed: int) -> np.ndarray:
    _qk, bb = iq_dequant.GEOMETRY[type_name]
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(nblocks, bb), dtype=np.uint8)
    for off, val in F16_FIELDS[type_name]:
        bits = int(np.float16(val).view(np.uint16))
        raw[:, off] = bits & 0xFF
        raw[:, off + 1] = (bits >> 8) & 0xFF
    return raw


def dot_reference(type_name: str, raw: np.ndarray, nbpr: int, rows_per_exp: int,
                  x: np.ndarray) -> np.ndarray:
    """`*_dot_k` 的参考输出：partial[全局块号] = wv[g] · x[e, b]。"""
    qk, _bb = iq_dequant.GEOMETRY[type_name]
    nblocks = raw.shape[0]
    wv = iq_dequant.dequantize(raw.tobytes(), type_name).reshape(nblocks, qk)
    xb = np.ascontiguousarray(x).reshape(-1, qk)
    out = np.empty(nblocks, dtype=np.float32)
    for g in range(nblocks):
        q1, b = divmod(g, nbpr)
        e = q1 // rows_per_exp
        out[g] = float(np.dot(wv[g].astype(np.float64), xb[e * nbpr + b]))
    return out


# ---------------------------------------------------------------------------
# A. 逐元素
# ---------------------------------------------------------------------------
@case("elementwise", "fill_k")
def t_fill(ctx: Ctx):
    n = 1000
    y = ctx.out(n)
    ctx.launch("fill_k", (n + 63) // 64, 64, [y, np.float32(1.25), n])
    return judge(np.abs(ctx.get(y, n) - 1.25).max(), 1.25)


@case("elementwise", "scale_mul_k")
def t_scale_mul(ctx: Ctx):
    n = 1000
    x = np.random.default_rng(1).standard_normal(n).astype(np.float32)
    xp = ctx.buf(x.copy())
    ctx.launch("scale_mul_k", (n + 63) // 64, 64, [xp, np.float32(0.5), n])
    return judge(np.abs(ctx.get(xp, n) - x * 0.5).max(), x * 0.5)


@case("elementwise", "add_inplace_k")
def t_add_inplace(ctx: Ctx):
    n = 1000
    rng = np.random.default_rng(2)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    ap, bp = ctx.buf(a.copy()), ctx.buf(b)
    ctx.launch("add_inplace_k", (n + 63) // 64, 64, [ap, bp, n])
    return judge(np.abs(ctx.get(ap, n) - (a + b)).max(), a + b)


@case("elementwise", "concat2_k")
def t_concat2(ctx: Ctx):
    """实测契约（用探针钉死）：**按 `pre` 分块交替交织**，不是单纯拼接。

        q = i // pre, r = i % pre
        out[i] = (q 偶 ? a : b)[(q // 2) * pre + r]      对 i ∈ [0, 2n)

    关键是 **第 5 个参数 `n` 是「半长」**：`a` / `b` 各 `n` 个元素，
    **输出要 `2n` 个**（每个线程写 2 个）。按 `n` 给输出缓冲会写穿到相邻
    缓冲，表现为「同一组参数复跑结果不同」（取决于隔壁放了什么）。
    grid = `ceil(n/64)`、wg = 64。
    """
    pre, n = 8, 32
    rng = np.random.default_rng(3)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    op, ap, bp = ctx.out(2 * n), ctx.buf(a), ctx.buf(b)
    ctx.launch("concat2_k", (n + 63) // 64, 64, [op, ap, bp, pre, n])
    ref = np.empty(2 * n, dtype=np.float32)
    for i in range(2 * n):
        q, r = divmod(i, pre)
        src = a if q % 2 == 0 else b
        ref[i] = src[(q // 2) * pre + r]
    return judge(np.abs(ctx.get(op, 2 * n) - ref).max(), ref, atol=0.0, rtol=0.0)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


@case("elementwise", "silu_mul_k")
def t_silu_mul(ctx: Ctx):
    n = 1000
    rng = np.random.default_rng(4)
    gate = rng.standard_normal(n).astype(np.float32)
    up = rng.standard_normal(n).astype(np.float32)
    y, gp, up_ = ctx.out(n), ctx.buf(gate), ctx.buf(up)
    ctx.launch("silu_mul_k", (n + 63) // 64, 64, [y, gp, up_, n])
    ref = _silu(gate) * up
    return judge(np.abs(ctx.get(y, n) - ref).max(), ref)


@case("elementwise", "sigmoid_mul_k")
def t_sigmoid_mul(ctx: Ctx):
    n = 1000
    rng = np.random.default_rng(5)
    gate = rng.standard_normal(n).astype(np.float32)
    up = rng.standard_normal(n).astype(np.float32)
    y, gp, up_ = ctx.out(n), ctx.buf(gate), ctx.buf(up)
    ctx.launch("sigmoid_mul_k", (n + 63) // 64, 64, [y, gp, up_, n])
    # 与 silu_mul_k 不同：sigmoid 作用在**第二个**参数上（实测 a * sigmoid(b)）
    ref = gate * (1.0 / (1.0 + np.exp(-up)))
    return judge(np.abs(ctx.get(y, n) - ref).max(), ref)


@case("elementwise", "gelu_mul_k")
def t_gelu_mul(ctx: Ctx):
    n = 1000
    rng = np.random.default_rng(6)
    gate = rng.standard_normal(n).astype(np.float32)
    up = rng.standard_normal(n).astype(np.float32)
    y, gp, up_ = ctx.out(n), ctx.buf(gate), ctx.buf(up)
    ctx.launch("gelu_mul_k", (n + 63) // 64, 64, [y, gp, up_, n, 64])
    ref = _gelu_tanh(gate) * up
    return judge(np.abs(ctx.get(y, n) - ref).max(), ref)


@case("elementwise", "l2norm_k")
def t_l2norm(ctx: Ctx):
    rows, n = 4, 128
    rng = np.random.default_rng(7)
    x = rng.standard_normal((rows, n)).astype(np.float32)
    eps = 1e-5
    xp = ctx.buf(x.copy())
    ctx.launch("l2norm_k", rows, 64, [xp, n, np.float32(eps)])
    ref = x / np.sqrt((x ** 2).sum(axis=1, keepdims=True) + eps)
    return judge(np.abs(ctx.get(xp, rows * n).reshape(rows, n) - ref).max(), ref)


# ---------------------------------------------------------------------------
# B. 归一化 / softmax / top-k / 路由
# ---------------------------------------------------------------------------
@case("norm", "rmsnorm_k")
def t_rmsnorm(ctx: Ctx):
    rows, dim = 4, 256
    rng = np.random.default_rng(8)
    x = rng.standard_normal((rows, dim)).astype(np.float32)
    w = rng.standard_normal(dim).astype(np.float32)
    eps = 1e-5
    y, xp, wp = ctx.out(rows * dim), ctx.buf(x), ctx.buf(w)
    # 第 6 个参数是 flag：**必须传 0** 才是标准 RMSNorm；
    # 非 0 会走另一条缩放分支（实测约 ×2，model.RMSNorm 传的就是 0）。
    ctx.launch("rmsnorm_k", rows, 64, [y, xp, wp, dim, np.float32(eps), 0])
    ref = x / np.sqrt((x ** 2).mean(axis=1, keepdims=True) + eps) * w
    return judge(np.abs(ctx.get(y, rows * dim).reshape(rows, dim) - ref).max(), ref)


@case("norm", "rmsnorm_gated_k")
def t_rmsnorm_gated(ctx: Ctx):
    rows, dim = 4, 256
    rng = np.random.default_rng(9)
    x = rng.standard_normal((rows, dim)).astype(np.float32)
    w = rng.standard_normal(dim).astype(np.float32)
    gate = rng.standard_normal((rows, dim)).astype(np.float32)
    eps = 1e-5
    y = ctx.out(rows * dim)
    xp, wp, gp = ctx.buf(x), ctx.buf(w), ctx.buf(gate)
    ctx.launch("rmsnorm_gated_k", rows, 64, [y, xp, wp, gp, dim, np.float32(eps)])
    # 门控不是逐元素相乘，而是 **SiLU**：y = rms(x)*w*siLU(gate)（实测确认）
    ref = x / np.sqrt((x ** 2).mean(axis=1, keepdims=True) + eps) * w * _silu(gate)
    return judge(np.abs(ctx.get(y, rows * dim).reshape(rows, dim) - ref).max(), ref)


@case("norm", "layernorm_k")
def t_layernorm(ctx: Ctx):
    rows, cols = 4, 256
    rng = np.random.default_rng(10)
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    w = rng.standard_normal(cols).astype(np.float32)
    b = rng.standard_normal(cols).astype(np.float32)
    eps = 1e-5
    y = ctx.out(rows * cols)
    xp, wp, bp = ctx.buf(x), ctx.buf(w), ctx.buf(b)
    ctx.launch("layernorm_k", rows, 64, [y, xp, wp, bp, rows, cols, np.float32(eps), 64])
    ref = (x - x.mean(axis=1, keepdims=True)) / np.sqrt(
        x.var(axis=1, keepdims=True) + eps) * w + b
    return judge(np.abs(ctx.get(y, rows * cols).reshape(rows, cols) - ref).max(), ref)


@case("norm", "softmax_k")
def t_softmax(ctx: Ctx):
    rows, cols = 6, 256
    rng = np.random.default_rng(11)
    x = rng.standard_normal((rows, cols)).astype(np.float32) * 3
    y, xp = ctx.out(rows * cols), ctx.buf(x)
    ctx.launch("softmax_k", rows, 64, [y, xp, rows, cols, 64])
    e = np.exp(x - x.max(axis=1, keepdims=True))
    ref = e / e.sum(axis=1, keepdims=True)
    got = ctx.get(y, rows * cols).reshape(rows, cols)
    return judge(np.abs(got - ref).max(), ref)


@case("norm", "softmax_vec_k")
def t_softmax_vec(ctx: Ctx):
    """向量化行 softmax：lane 一次处理**连续 4 列**（`global_load_dwordx4`），
    一个 warp 一轮覆盖 256 个元素——访存完全合并。要求 `cols % 256 == 0`。

    实测比 `softmax_k`（lane 按 64 步长跳、每元素一条 load + 全排空）快
    **2.3 倍**（cols=16384：126 vs 287 us），是解码注意力里最大的一块。
    """
    rows, cols = 6, 1024
    rng = np.random.default_rng(11)
    x = rng.standard_normal((rows, cols)).astype(np.float32) * 3
    y, xp = ctx.out(rows * cols), ctx.buf(x)
    ctx.launch("softmax_vec_k", rows, 64, [y, xp, rows, cols, 64])
    e = np.exp(x - x.max(axis=1, keepdims=True))
    ref = e / e.sum(axis=1, keepdims=True)
    return judge(np.abs(ctx.get(y, rows * cols).reshape(rows, cols) - ref).max(), ref)


@case("norm", "topk_k")
def t_topk(ctx: Ctx):
    rows, cols, k = 3, 512, 8
    rng = np.random.default_rng(12)
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    xp = ctx.buf(x)
    ip, vp = ctx.out(rows * k, np.int32), ctx.out(rows * k)
    ctx.launch("topk_k", (rows + 63) // 64, 64, [xp, ip, vp, rows, cols, k])
    order = np.argsort(-x, axis=1)[:, :k]
    vals = np.take_along_axis(x, order, axis=1)
    got_i = ctx.get(ip, rows * k, np.int32).reshape(rows, k)
    got_v = ctx.get(vp, rows * k).reshape(rows, k)
    err = max(np.abs(got_v - vals).max(),
              float(np.abs(got_i.astype(np.int64) - order).max()))
    return judge(err, np.concatenate([vals.reshape(-1), order.reshape(-1)]))


@case("norm", "router_top10_k")
def t_router_top10(ctx: Ctx):
    rows, n_exp, k = 4, 256, 10
    rng = np.random.default_rng(13)
    logits = rng.standard_normal((rows, n_exp)).astype(np.float32)
    lp = ctx.buf(logits)
    ip, wp = ctx.out(rows * k, np.int32), ctx.out(rows * k)
    ctx.launch("router_top10_k", rows, 64, [lp, ip, wp, rows, n_exp])
    top = np.argsort(-logits, axis=1)[:, :k]
    ex = np.exp(logits[np.arange(rows)[:, None], top] - logits.max(axis=1, keepdims=True))
    ref_w = ex / ex.sum(axis=1, keepdims=True)
    got_i = ctx.get(ip, rows * k, np.int32).reshape(rows, k)
    got_w = ctx.get(wp, rows * k).reshape(rows, k)
    err = max(float(np.abs(got_i - top).max()), float(np.abs(got_w - ref_w).max()))
    return judge(err, np.concatenate([top.reshape(-1), ref_w.reshape(-1)]), atol=1e-5)


@case("norm", "argmax_k")
def t_argmax(ctx: Ctx):
    rows, cols = 3, 1024
    rng = np.random.default_rng(14)
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    xp = ctx.buf(x)
    out = ctx.buf(np.zeros(rows * 4, dtype=np.uint8))
    for r in range(rows):
        ctx.launch("argmax_k", (cols + 63) // 64, 64,
                   [xp + r * cols * 4, cols, out + r * 4])
    got = ctx.get(out, rows, np.int32)
    return judge(np.abs(got.astype(np.int64) - x.argmax(axis=1)).max(),
                 np.zeros(1), atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# C. MoE
# ---------------------------------------------------------------------------
@case("moe", "gather_rows_k")
def t_gather_rows(ctx: Ctx):
    rows, dim = 7, 256
    rng = np.random.default_rng(15)
    src = rng.standard_normal((rows, dim)).astype(np.float32)
    idx = np.array([3, 0, 5, 1, 6, 2, 4], dtype=np.uint32)
    dp, sp, ip = ctx.out(rows * dim), ctx.buf(src), ctx.buf(idx)
    ctx.launch("gather_rows_k", rows, 64, [dp, sp, ip, rows, dim])
    got = ctx.get(dp, rows * dim).reshape(rows, dim)
    return judge(np.abs(got - src[idx]).max(), src[idx], atol=0.0, rtol=0.0)


@case("moe", "moe_combine_k")
def t_moe_combine(ctx: Ctx):
    rows, dim, n_exp = 5, 512, 3
    rng = np.random.default_rng(16)
    exp_out = rng.standard_normal((n_exp, rows, dim)).astype(np.float32)
    weights = rng.random((rows, n_exp)).astype(np.float32)
    y, ep, wp = ctx.out(rows * dim), ctx.buf(exp_out), ctx.buf(weights)
    ctx.launch("moe_combine_k", rows, 64, [y, ep, wp, n_exp, rows, dim])
    ref = np.einsum("re,erd->rd", weights, exp_out)
    return judge(np.abs(ctx.get(y, rows * dim).reshape(rows, dim) - ref).max(), ref)


@case("moe", "moe_combine_gather_k")
def t_moe_combine_gather(ctx: Ctx):
    rows, n_sel, erows, dim = 5, 3, 8, 1024   # dim=1024 覆盖旧版魔法除边界
    rng = np.random.default_rng(17)
    exp_off = np.arange(n_sel + 1, dtype=np.uint32) * erows
    exp_bucket = rng.standard_normal((exp_off[-1], dim)).astype(np.float32)
    weights = rng.random((rows, n_sel)).astype(np.float32)
    pos = rng.integers(0, erows, size=(rows, n_sel)).astype(np.uint32)
    base = exp_off[:n_sel][None, :] + pos
    y, ep, wp, bp = (ctx.out(rows * dim), ctx.buf(exp_bucket),
                     ctx.buf(weights), ctx.buf(base))
    ctx.launch("moe_combine_gather_k", rows, 64, [y, ep, wp, bp, n_sel, rows, dim])
    ref = np.zeros((rows, dim), dtype=np.float32)
    for r in range(rows):
        for e in range(n_sel):
            ref[r] += weights[r, e] * exp_bucket[base[r, e]]
    return judge(np.abs(ctx.get(y, rows * dim).reshape(rows, dim) - ref).max(), ref)


# ---------------------------------------------------------------------------
# D. GEMV
# ---------------------------------------------------------------------------
@case("gemv", "gemv_f32_k")
def t_gemv_f32(ctx: Ctx):
    nrows, k = 64, 512
    rng = np.random.default_rng(18)
    w = rng.standard_normal((nrows, k)).astype(np.float32)
    x = rng.standard_normal(k).astype(np.float32)
    wp, xp, yp = ctx.buf(w), ctx.buf(x), ctx.out(nrows)
    ctx.launch("gemv_f32_k", nrows, 64, [wp, xp, yp, nrows, k // 4, 64])
    ref = w @ x
    return judge(np.abs(ctx.get(yp, nrows) - ref).max(), ref)


@case("gemv", "gemv_f32_warp_k")
def t_gemv_f32_warp(ctx: Ctx):
    nrows, k = 64, 640
    rng = np.random.default_rng(19)
    w = rng.standard_normal((nrows, k)).astype(np.float32)
    x = rng.standard_normal(k).astype(np.float32)
    wp, xp, yp = ctx.buf(w), ctx.buf(x), ctx.out(nrows)
    ctx.launch("gemv_f32_warp_k", nrows, 64, [wp, xp, yp, nrows, k, 64])
    ref = w @ x
    return judge(np.abs(ctx.get(yp, nrows) - ref).max(), ref)


@case("gemv", "gemv_i8_k")
def t_gemv_i8(ctx: Ctx):
    nrows, k = 64, 640
    ngroups = k // 32
    rng = np.random.default_rng(20)
    w = rng.integers(-127, 128, size=(nrows, k), dtype=np.int8)
    ws = (rng.random((nrows, ngroups)).astype(np.float32) * 0.02 + 0.001)
    x = rng.standard_normal(k).astype(np.float32)
    wp, wsp, xp, yp = ctx.buf(w), ctx.buf(ws), ctx.buf(x), ctx.out(nrows)
    ctx.launch("gemv_i8_k", nrows, 64,
               [wp, wsp, xp, yp, nrows, ngroups, k, ngroups * 4, 64])
    part = np.einsum("gsk,sk->gs", w.astype(np.float32).reshape(nrows, ngroups, 32),
                     x.reshape(ngroups, 32))
    ref = (part * ws).sum(axis=1)
    return judge(np.abs(ctx.get(yp, nrows) - ref).max(), ref)


# ---------------------------------------------------------------------------
# E. 量化解码
# ---------------------------------------------------------------------------
# (编码, 内核, 需要的小表, 是否带 group_size 参数)
DECODE_ENTRIES = [
    ("Q2_0", "q2_0_dequant_k", (), True),
    ("Q4_0", "q4_0_dequant_k", (), True),
    ("Q8_0", "q8_0_dequant_k", (), True),
    ("IQ4_NL", "iq4nl_dequant_k", (), True),
    ("IQ4_XS", "iq4xs_dequant_k", (), True),
    ("IQ2_S", "iq2s_dequant_k", ("iq2s",), True),
    ("IQ3_S", "iq3s_dequant_k", ("iq3s",), True),
    ("IQ3_XXS", "iq3xxs_dequant_k", ("iq3xxs", "ksigns"), True),
    ("Q4_K", "q4k_dequant", (), False),
    ("Q5_K", "q5k_dequant", (), False),
    ("Q6_K", "q6k_dequant", (), False),
]


def _make_decode_case(type_name: str, kernel: str, tables, with_group: bool,
                      seed: int):
    def run(ctx: Ctx):
        nblocks = 64
        qk, _bb = iq_dequant.GEOMETRY[type_name]
        raw = synth_blocks(type_name, nblocks, seed)
        ref = iq_dequant.dequantize(raw.tobytes(), type_name)
        wp = ctx.buf(raw.tobytes())
        yp = ctx.out(nblocks * qk)
        argv = [wp, yp] + [ctx.buf(_table_bytes(t)) for t in tables]
        argv += [nblocks] + ([64] if with_group else [])
        ctx.launch(kernel, nblocks, 64, argv)
        got = ctx.get(yp, nblocks * qk)
        return judge(np.abs(got - ref).max(), ref, atol=1e-5, rtol=1e-6)
    run.__name__ = f"t_{kernel}"
    return run


for _i, (_tn, _kn, _tb, _wg) in enumerate(DECODE_ENTRIES):
    case("decode", _kn)(_make_decode_case(_tn, _kn, _tb, _wg, 100 + _i))


@case("decode", "int4_dequant_k")
def t_int4_dequant(ctx: Ctx):
    nblocks, block_elems, block_words = 64, 128, 16
    rng = np.random.default_rng(30)
    packed = rng.integers(0, 1 << 32, size=nblocks * block_words,
                          dtype=np.uint64).astype(np.uint32)
    scale_bits = (rng.uniform(1e-3, 5e-2, size=nblocks).astype(np.float32)
                  .view(np.uint32) & np.uint32(0xFFFF0000)) >> 16
    scales = scale_bits.astype(np.uint16)
    nwords = nblocks * block_words
    wp, sp, yp = (ctx.buf(packed), ctx.buf(scales),
                  ctx.out(nblocks * block_elems))
    ctx.launch("int4_dequant_k", (nwords + 63) // 64, 64, [wp, sp, yp, nwords, 64])
    codes = np.empty((nblocks, block_elems), dtype=np.int32)
    for w in range(block_words):
        codes[:, w * 8:(w + 1) * 8] = (
            (packed.reshape(nblocks, block_words)[:, w][:, None]
             >> np.arange(0, 32, 4, dtype=np.uint32)) & 0xF).astype(np.int32)
    bf16 = (scales.astype(np.uint32) << 16).view(np.float32)
    ref = ((codes - 8).astype(np.float32) * bf16[:, None]).reshape(-1)
    return judge(np.abs(ctx.get(yp, nblocks * block_elems) - ref).max(), ref,
                 atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# F. 融合点积（原生解码 × 激活）+ reduce_blocks_k
# ---------------------------------------------------------------------------
DOT_ENTRIES = [
    # (类型, 内核, 每块元素, 额外表)
    ("Q2_0", "q2_0_dot_k", 64, ()),
    ("Q4_0", "q4_0_dot_k", 32, ()),
    ("Q8_0", "q8_0_dot_k", 32, ()),
    ("IQ4_NL", "iq4nl_dot_k", 32, ()),
    ("IQ4_XS", "iq4xs_dot_k", 256, ()),
    ("Q4_K", "q4k_dot_k", 256, ()),
    ("Q5_K", "q5k_dot_k", 256, ()),
    ("Q6_K", "q6k_dot_k", 256, ()),
    ("IQ2_S", "iq2s_dot_k", 256, ("iq2s",)),
    ("IQ3_S", "iq3s_dot_k", 256, ("iq3s",)),
    ("IQ3_XXS", "iq3xxs_dot_k", 256, ("iq3xxs", "ksigns")),
]


def _table_bytes(name: str) -> bytes:
    from iq_tables import iq2s_grid, iq3s_grid, iq3xxs_grid, ksigns_iq2xs
    if name == "iq2s":
        return struct.pack("<%dQ" % len(iq2s_grid), *[int(v) for v in iq2s_grid])
    if name == "iq3s":
        return struct.pack("<%dI" % len(iq3s_grid), *[int(v) for v in iq3s_grid])
    if name == "iq3xxs":
        return struct.pack("<%dI" % len(iq3xxs_grid), *[int(v) for v in iq3xxs_grid])
    if name == "ksigns":
        return bytes(int(v) for v in ksigns_iq2xs)
    raise KeyError(name)


def _make_dot_case(type_name: str, kernel: str, qk: int, tables, seed: int):
    def run(ctx: Ctx):
        nbpr, rows_per_exp, n_exp = 8, 6, 2
        k = nbpr * qk
        nblocks = rows_per_exp * n_exp * nbpr
        raw = synth_blocks(type_name, nblocks, seed)
        rng = np.random.default_rng(seed + 7)
        x = rng.standard_normal((n_exp, k)).astype(np.float32)
        ref = dot_reference(type_name, raw, nbpr, rows_per_exp, x)
        wp, xp, pp = ctx.buf(raw.tobytes()), ctx.buf(x), ctx.out(nblocks)
        tp = [ctx.buf(_table_bytes(t)) for t in tables]
        ids = ctx.buf(np.arange(n_exp, dtype=np.uint32))
        # 参数顺序：w,x,partial,nblocks,wg,nbpr,m_nbpr,rpe,m_rpe,[表],ids,stride,rpw,m_rpw
        _qk, bb = iq_dequant.GEOMETRY[type_name]
        stride = rows_per_exp * nbpr * bb      # 权重侧每专家组字节跨度
        argv = ([wp, xp, pp, nblocks, 64, nbpr, div_magic(nbpr, nblocks),
                 rows_per_exp, div_magic(rows_per_exp, nblocks // nbpr + 1)]
                + tp
                + [ids, stride, rows_per_exp,
                   div_magic(rows_per_exp, nblocks // nbpr + 1)])
        ctx.launch(kernel, nblocks, 64, argv)
        got = ctx.get(pp, nblocks)
        return judge(np.abs(got - ref).max(), ref)
    run.__name__ = f"t_{kernel}"
    return run


for _i, (_tn, _kn, _qk, _tb) in enumerate(DOT_ENTRIES):
    case("dot", _kn)(_make_dot_case(_tn, _kn, _qk, _tb, 200 + _i))


@case("dot", "reduce_blocks_k")
def t_reduce_blocks(ctx: Ctx):
    rows, nbpr = 32, 5
    rng = np.random.default_rng(40)
    partial = rng.standard_normal(rows * nbpr).astype(np.float32)
    pp, yp = ctx.buf(partial), ctx.out(rows)
    ctx.launch("reduce_blocks_k", rows, 64, [pp, yp, rows, nbpr])
    ref = partial.reshape(rows, nbpr).sum(axis=1)
    return judge(np.abs(ctx.get(yp, rows) - ref).max(), ref)


@case("dot", "int4_dot_k")
def t_int4_dot(ctx: Ctx):
    nbpr, rows_per_exp, n_exp = 8, 6, 2
    block_elems, block_words = 128, 16
    k = nbpr * block_elems
    nblocks = rows_per_exp * n_exp * nbpr
    rng = np.random.default_rng(50)
    packed = rng.integers(0, 1 << 32, size=nblocks * block_words,
                          dtype=np.uint64).astype(np.uint32)
    scale_bits = (rng.uniform(1e-3, 5e-2, size=nblocks).astype(np.float32)
                  .view(np.uint32) & np.uint32(0xFFFF0000)) >> 16
    scales = scale_bits.astype(np.uint16)
    x = rng.standard_normal((n_exp, k)).astype(np.float32)
    codes = np.empty((nblocks, block_elems), dtype=np.float32)
    pk = packed.reshape(nblocks, block_words)
    for w in range(block_words):
        codes[:, w * 8:(w + 1) * 8] = (
            (pk[:, w][:, None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 0xF)
    bf16 = (scales.astype(np.uint32) << 16).view(np.float32)
    wv = (codes - 8) * bf16[:, None]
    ref = np.empty(nblocks, dtype=np.float32)
    for g in range(nblocks):
        q1, b = divmod(g, nbpr)
        e = q1 // rows_per_exp
        ref[g] = float(np.dot(wv[g], x[e, b * block_elems:(b + 1) * block_elems]))
    wp, sp, xp, pp = ctx.buf(packed), ctx.buf(scales), ctx.buf(x), ctx.out(nblocks)
    ids = ctx.buf(np.arange(n_exp, dtype=np.uint32))
    argv = [wp, xp, pp, nblocks, 64, nbpr, div_magic(nbpr, nblocks),
            rows_per_exp, div_magic(rows_per_exp, nblocks // nbpr + 1),
            sp, ids, rows_per_exp * nbpr * 64, rows_per_exp,
            div_magic(rows_per_exp, nblocks // nbpr + 1)]
    ctx.launch("int4_dot_k", nblocks, 64, argv)
    got = ctx.get(pp, nblocks)
    return judge(np.abs(got - ref).max(), ref)


# ---------------------------------------------------------------------------
# G. RT4 解码通路（W4A4 / W4A8）—— math 见 kernels/gemv_w4a4_core.h
# ---------------------------------------------------------------------------
def pack_int4_lo(v: np.ndarray) -> np.ndarray:
    """(R,K) 有符号 int4 → (R,K/8) u32，**低半字节 = 更小的 k**。"""
    r, k = v.shape
    out = np.zeros((r, k // 8), dtype=np.uint32)
    for j in range(8):
        out |= (v[:, j::8].astype(np.uint32) & 0xF) << np.uint32(4 * j)
    return out


def unpack_int4_lo(u: np.ndarray, k: int) -> np.ndarray:
    """(R,K/8) u32 → (R,K) int32（符号扩展的两补码 int4）。"""
    u = np.asarray(u, dtype=np.uint32)
    out = np.empty((u.shape[0], k), dtype=np.int32)
    for j in range(8):
        nib = ((u >> np.uint32(4 * j)) & np.uint32(0xF)).astype(np.int32)
        out[:, j::8] = np.where(nib >= 8, nib - 16, nib)
    return out


def quant_act4_ref(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """主机侧 `quant_act4`：组=32，`s = amax/7`（amax=0 时 1），q 夹到 [-8,7]。"""
    m, k = x.shape
    xg = x.reshape(m, k // 32, 32)
    amax = np.abs(xg).max(axis=2)
    s = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float32)
    q = np.rint(xg / s[:, :, None]).astype(np.int32)
    q = np.clip(q, -8, 7)
    return q.reshape(m, k), s


def quant_act_ref(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """主机侧 `quant_act`：组=128，`s = amax/127`，q 夹到 [-128,127]。"""
    m, k = x.shape
    xg = x.reshape(m, k // 128, 128)
    amax = np.abs(xg).max(axis=2)
    s = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.rint(xg / s[:, :, None]).astype(np.int32)
    q = np.clip(q, -128, 127)
    return q.reshape(m, k), s


def w4_gemv_ref(wq: np.ndarray, ws: np.ndarray, qa: np.ndarray,
                asc: np.ndarray, k: int, grp_a: int) -> np.ndarray:
    """`y = Σ_g ws[n,g] · Σ_{k∈g} w[n,k]·(q[m,k]·asc[m,k//grp_a])`；返回 [M,N]。

    与内核一致：**整数组内先做精确 int32 点积**，再用该子组的 f32 尺度缩放，
    最后按权重组乘 `ws` 累加（避免「先逐元素乘尺度再求和」带来的参考误差）。
    """
    wv = unpack_int4_lo(wq, k).astype(np.float32)
    grp = k // ws.shape[1]                       # 权重组（128）
    ngrp = ws.shape[1]
    wsc = ws.view(np.float16).astype(np.float64)
    a = qa.astype(np.float64)
    m = qa.shape[0]
    y = np.zeros((m, wv.shape[0]), dtype=np.float64)
    sub = grp // grp_a                           # 每组里几个激活子组
    for g in range(ngrp):
        sl = slice(g * grp, (g + 1) * grp)
        acc = np.zeros_like(y)
        for s in range(sub):
            kk = slice(g * grp + s * grp_a, g * grp + (s + 1) * grp_a)
            acc += asc[:, g * sub + s][:, None] * (
                a[:, kk] @ wv[:, kk].T.astype(np.float64))
        y += acc * wsc[:, g][None, :]
    return y.astype(np.float32)


def _w4_setup(ctx: Ctx, n: int, k: int, m: int, seed: int):
    rng = np.random.default_rng(seed)
    wq_h = rng.integers(-8, 8, size=(n, k)).astype(np.int32)
    ws_h = rng.uniform(0.5, 1.5, size=(n, k // 128)).astype(np.float32).astype(np.float16)
    x = rng.standard_normal((m, k)).astype(np.float32)
    wq = ctx.buf(pack_int4_lo(wq_h))
    ws = ctx.buf(ws_h)
    px = ctx.buf(x)
    return wq, ws, px, x, wq_h, ws_h


@case("w4", "w4a4 M=1（量化+GEMV 全对账）")
def t_w4a4_m1(ctx: Ctx):
    n, k = 256, 5120
    wq, ws, px, x, wq_h, ws_h = _w4_setup(ctx, n, k, 1, 60)
    qa, asc = quant_act4_ref(x)
    paq = ctx.buf(pack_int4_lo(qa.reshape(1, k)))
    pasc = ctx.buf(asc.reshape(-1))
    py = ctx.out(n)
    ctx.launch("gemv_w4a4<1>", (n + 3) // 4, 256, [wq, ws, paq, pasc, py, n, k])
    ref = w4_gemv_ref(pack_int4_lo(wq_h), ws_h, qa, asc, k, 32)
    return judge(np.abs(ctx.get(py, n) - ref[0]).max(), ref)


@case("w4", "w4a4 M=4（量化+GEMV 全对账）")
def t_w4a4_m4(ctx: Ctx):
    n, k, m = 256, 5120, 4
    wq, ws, px, x, wq_h, ws_h = _w4_setup(ctx, n, k, m, 61)
    qa, asc = quant_act4_ref(x)
    paq = ctx.buf(pack_int4_lo(qa))
    pasc = ctx.buf(asc.reshape(-1))
    py = ctx.out(m * n)
    ctx.launch("gemv_w4a4<4>", (n + 3) // 4, 256, [wq, ws, paq, pasc, py, n, k])
    ref = w4_gemv_ref(pack_int4_lo(wq_h), ws_h, qa, asc, k, 32)
    got = ctx.get(py, m * n).reshape(m, n)
    return judge(np.abs(got - ref).max(), ref)


@case("w4", "w4a4 双行 M=4（r2_k）")
def t_w4a4_r2(ctx: Ctx):
    n, k, m = 256, 5120, 4
    wq, ws, px, x, wq_h, ws_h = _w4_setup(ctx, n, k, m, 62)
    qa, asc = quant_act4_ref(x)
    paq = ctx.buf(pack_int4_lo(qa))
    pasc = ctx.buf(asc.reshape(-1))
    py = ctx.out(m * n)
    ctx.launch("gemv_w4a4_r2_k", (n + 7) // 8, 256, [wq, ws, paq, pasc, py, n, k])
    ref = w4_gemv_ref(pack_int4_lo(wq_h), ws_h, qa, asc, k, 32)
    got = ctx.get(py, m * n).reshape(m, n)
    return judge(np.abs(got - ref).max(), ref)


@case("w4", "w4a8 M=1（量化+GEMV 全对账）")
def t_w4a8_m1(ctx: Ctx):
    n, k = 256, 5120
    wq, ws, px, x, wq_h, ws_h = _w4_setup(ctx, n, k, 1, 63)
    qa, asc = quant_act_ref(x)
    # quant_act 把偶/奇下标的量化值拆成两条 int8 流
    pae = ctx.buf(qa[:, 0::2].astype(np.int8))
    pao = ctx.buf(qa[:, 1::2].astype(np.int8))
    pasc = ctx.buf(asc.reshape(-1))
    asum = qa.reshape(1, k // 128, 128).sum(axis=2).astype(np.int32)
    pasu = ctx.buf(asum.reshape(-1))
    py = ctx.out(n)
    ctx.launch("gemv_w4a8<1,false,1>", (n + 3) // 4, 256,
               [wq, ws, pae, pao, pasc, pasu, py, n, k])
    ref = w4_gemv_ref(pack_int4_lo(wq_h), ws_h, qa, asc, k, 128)
    return judge(np.abs(ctx.get(py, n) - ref[0]).max(), ref)


# ---------------------------------------------------------------------------
# H. 激活量化（所有量化通路的前置步骤）
# ---------------------------------------------------------------------------
def quant_rows_ref(x: np.ndarray, k: int, g: int) -> tuple[np.ndarray, np.ndarray]:
    """主机侧 `quant_rows_fast_k` / `quant_rows_k`：组内 `s = amax/7`，夹 [-8,7]。

    输出布局：`q[row][K/8]` 低半字节 = 偶数 k；`sc[组][row]`（组优先）。
    """
    rows = x.shape[0]
    ng = k // g
    q = np.zeros((rows, k // 8), dtype=np.uint32)
    sc = np.zeros((ng, rows), dtype=np.float32)
    for r in range(rows):
        for gg in range(ng):
            blk = x[r, gg * g:(gg + 1) * g]
            amax = np.float32(np.abs(blk).max())
            s = np.float32(amax / np.float32(7.0)) if amax > 0 else np.float32(1.0)
            sc[gg, r] = s
            codes = np.clip(np.rint(blk / s), -8, 7).astype(np.int64)
            for e in range(0, g, 8):
                w = 0
                for j in range(8):
                    w |= int(codes[e + j] & 0xF) << (4 * j)
                q[r, (gg * g + e) // 8] = w
    return q, sc


@case("quant", "quant_rows_fast_k")
def t_quant_rows_fast(ctx: Ctx):
    rows, k, g = 32, 512, 128
    rng = np.random.default_rng(70)
    x = rng.standard_normal((rows, k)).astype(np.float32)
    qref, scref = quant_rows_ref(x, k, g)
    ng = k // g
    pq = ctx.out(rows * k // 8, np.uint32)
    psc = ctx.out(ng * rows)
    px = ctx.buf(x)
    ctx.launch("quant_rows_fast_k", rows, 256,
               [pq, psc, px, k, g, rows, k, 0, ng])
    q = ctx.get(pq, rows * k // 8, np.uint32).reshape(rows, k // 8)
    sc = ctx.get(psc, ng * rows).reshape(ng, rows)
    # 量化是整数/位运算，要求**逐位一致**
    err = 0.0 if np.array_equal(q, qref) and np.array_equal(sc, scref) else 1.0
    return judge(err, np.zeros(1), atol=0.0, rtol=0.0)


@case("quant", "quant_rows_k")
def t_quant_rows(ctx: Ctx):
    """旧版行量化：与 `quant_rows_fast_k` 输出逐位一致。

    约定：`grid = rows`，**workgroup 必须等于 K/G（组数）**——给大了
    `sc` 会被写坏（实测 wg=64 时 scale 全是垃圾，`q` 却还对）。
    参数是 `(q, sc, x, K, G, rows, in_stride, in_off)`。
    """
    rows, k, g = 32, 512, 128
    ng = k // g
    rng = np.random.default_rng(73)
    x = rng.standard_normal((rows, k)).astype(np.float32)
    qref, scref = quant_rows_ref(x, k, g)
    pq = ctx.out(rows * k // 8, np.uint32)
    psc = ctx.out(ng * rows)
    ctx.launch("quant_rows_k", rows, ng,
               [pq, psc, ctx.buf(x), k, g, rows, k, 0])
    q = ctx.get(pq, rows * k // 8, np.uint32).reshape(rows, k // 8)
    sc = ctx.get(psc, ng * rows).reshape(ng, rows)
    err = 0.0 if np.array_equal(q, qref) and np.array_equal(sc, scref) else 1.0
    return judge(err, np.zeros(1), atol=0.0, rtol=0.0)


@case("quant", "quant_act4")
def t_quant_act4(ctx: Ctx):
    k = 5120
    rng = np.random.default_rng(71)
    x = rng.standard_normal((1, k)).astype(np.float32)
    qref, sref = quant_act4_ref(x)
    paq = ctx.out(k // 2, np.uint8)
    pasc = ctx.out(k // 32)
    ctx.launch("quant_act4", k // 32, 32, [ctx.buf(x), paq, pasc, 1, k])
    got_q = ctx.get(paq, k // 2, np.uint8)
    got_s = ctx.get(pasc, k // 32)
    ref_q = pack_int4_lo(qref).view(np.uint8).reshape(-1)
    ok = bool(np.array_equal(got_q, ref_q)) and bool(
        np.array_equal(got_s.view(np.uint32), sref.reshape(-1).view(np.uint32)))
    return judge(0.0 if ok else 1.0, np.zeros(1), atol=0.0, rtol=0.0)


@case("quant", "quant_act")
def t_quant_act(ctx: Ctx):
    k = 5120
    rng = np.random.default_rng(72)
    x = rng.standard_normal((1, k)).astype(np.float32)
    qref, sref = quant_act_ref(x)
    pae = ctx.out(k // 2, np.int8)
    pao = ctx.out(k // 2, np.int8)
    pasc = ctx.out(k // 128)
    pasu = ctx.out(k // 128, np.int32)
    ctx.launch("quant_act", k // 128, 32, [ctx.buf(x), pae, pao, pasc, pasu, 1, k])
    sum_ref = qref.reshape(1, k // 128, 128).sum(axis=2).astype(np.int32)
    ok = (np.array_equal(ctx.get(pae, k // 2, np.int8), qref[0, 0::2].astype(np.int8))
          and np.array_equal(ctx.get(pao, k // 2, np.int8), qref[0, 1::2].astype(np.int8))
          and np.array_equal(ctx.get(pasc, k // 128).view(np.uint32),
                             sref.reshape(-1).view(np.uint32))
          and np.array_equal(ctx.get(pasu, k // 128, np.int32), sum_ref.reshape(-1)))
    return judge(0.0 if ok else 1.0, np.zeros(1), atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# I. NVFP4（权重 E2M1 + E4M3 块尺度 + f32 全局尺度；激活 int8）
# ---------------------------------------------------------------------------
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def e2m1_decode(codes: np.ndarray) -> np.ndarray:
    """4bit E2M1 码 → f32（bit3 是符号，低 3 位查表）。"""
    c = np.asarray(codes, dtype=np.uint8) & 0xF
    v = E2M1[(c & 7).astype(np.int64)]
    return np.where(c & 8, -v, v).astype(np.float32)


def e4m3_decode(bits: np.ndarray) -> np.ndarray:
    """E4M3（OCP FP8）→ f32；1 符号 + 4 指数（偏置 7）+ 3 尾数。"""
    b = np.asarray(bits, dtype=np.uint8).astype(np.int64)
    s, e, m = (b >> 7) & 1, (b >> 3) & 0xF, b & 7
    val = np.where(e == 0, (m / 8.0) * 2.0 ** -6, (1.0 + m / 8.0) * 2.0 ** (e - 7))
    return np.where(s == 1, -val, val).astype(np.float32)


def pack_nvfp4_codes(codes: np.ndarray) -> np.ndarray:
    """(N,K) 的 4bit 码 → (N,K/8) u32，低半字节 = 更小的 k。"""
    n, k = codes.shape
    out = np.zeros((n, k // 8), dtype=np.uint32)
    for j in range(8):
        out |= (codes[:, j::8].astype(np.uint32) & 0xF) << np.uint32(4 * j)
    return out


def nvfp4_quant_ref(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`nvfp4_quant_act` 的主机参考：块=16，`s = amax/127`，码夹到 [-128,127]。

    输出 `(a, b, scale)`：a/b 是偶/奇下标的 int8 码（与 `quant_act` 同约定）。
    """
    m, k = x.shape
    xb = x.reshape(m, k // 16, 16)
    amax = np.abs(xb).max(axis=2)
    s = np.where(amax > 0, amax / np.float32(127.0), np.float32(1.0)).astype(np.float32)
    q = np.clip(np.rint(xb / s[:, :, None]), -128, 127).astype(np.int8).reshape(m, k)
    return q[:, 0::2], q[:, 1::2], s.reshape(m, -1)


def unpack_nvfp4_codes(wq: np.ndarray, k: int) -> np.ndarray:
    """(N,K/8) u32 → (N,K) 的 4bit 码。"""
    n = wq.shape[0]
    out = np.empty((n, k), dtype=np.uint8)
    for j in range(8):
        out[:, j::8] = ((wq >> np.uint32(4 * j)) & np.uint32(0xF)).astype(np.uint8)
    return out


@case("nvfp4", "nvfp4_quant_act")
def t_nvfp4_quant(ctx: Ctx):
    k = 5120
    rng = np.random.default_rng(90)
    x = (rng.standard_normal((1, k)).astype(np.float32) * 3)
    qa, qb, s = nvfp4_quant_ref(x)
    pa = ctx.out(k // 2, np.int8)
    pb = ctx.out(k // 2, np.int8)
    ps = ctx.out(k // 16)
    ctx.launch("nvfp4_quant_act", k // 16, 64, [ctx.buf(x), pa, pb, ps, 1, k])
    ok = (np.array_equal(ctx.get(pa, k // 2, np.int8), qa.reshape(-1))
          and np.array_equal(ctx.get(pb, k // 2, np.int8), qb.reshape(-1))
          and np.array_equal(ctx.get(ps, k // 16).view(np.uint32),
                             s.reshape(-1).view(np.uint32)))
    return judge(0.0 if ok else 1.0, np.zeros(1), atol=0.0, rtol=0.0)


@case("nvfp4", "nvfp4_gemv 全部 24 个模板变体")
def t_nvfp4_gemv_all(ctx: Ctx):
    """`nvfp4_gemv<A,B>`（A=1..4、B=1..4）与 `nvfp4_gemv_wide<A,B>`
    （A=1..4、B=1..2）共 24 个变体，全部用同一条参考对账。

    `grid = ceil(N/A)`、`wg = 256`。数学：
    `y[n] = gscale · Σ_k E2M1(w)·E4M3(ws[n,k/16])·(q_act[k]·asc[k/16])`。
    """
    n, k = 32, 512
    rng = np.random.default_rng(202)
    codes = rng.integers(0, 16, size=(n, k)).astype(np.uint8)
    wq = pack_nvfp4_codes(codes)
    ws = rng.integers(1, 0x7F, size=(n, k // 16)).astype(np.uint8)   # 避开 E4M3 NaN
    x = rng.standard_normal((1, k)).astype(np.float32) * 2
    qa, qb, s = nvfp4_quant_ref(x)
    gscale = float(np.float32(0.75))
    pwq, pws = ctx.buf(wq), ctx.buf(ws)
    pa = ctx.buf(qa.reshape(-1))
    pb = ctx.buf(qb.reshape(-1))
    psc = ctx.buf(s.reshape(-1))
    act = np.empty((1, k), dtype=np.int32)
    act[0, 0::2] = qa
    act[0, 1::2] = qb
    wv = e2m1_decode(unpack_nvfp4_codes(wq, k))
    wsc = e4m3_decode(ws)[:, np.repeat(np.arange(k // 16), 16)]
    av = act.astype(np.float32) * s[:, np.repeat(np.arange(k // 16), 16)]
    ref = ((wv * wsc) @ av.T * gscale).reshape(-1)
    worst = 0.0
    bad: list[str] = []
    for wide, bs in ((False, (1, 2, 3, 4)), (True, (1, 2))):
        for a in (1, 2, 3, 4):
            for b in bs:
                name = f"nvfp4_gemv{'_wide' if wide else ''}<{a},{b}>"
                py = ctx.out(n)
                ctx.launch(name, (n + a - 1) // a, 256,
                           [pwq, pws, pa, pb, psc, py, n, k, gscale])
                err = float(np.abs(ctx.get(py, n) - ref).max())
                worst = max(worst, err)
                if err > 1e-2 + 1e-5 * float(np.abs(ref).max()):
                    bad.append(name)
    if bad:
        raise AssertionError(f"以下变体不一致：{bad}")
    return judge(worst, ref)


# ---------------------------------------------------------------------------
# J. 视觉塔（ViT）里的小算子 + 序列小算子
# ---------------------------------------------------------------------------
@case("vit", "vit_bias_kernel")
def t_vit_bias(ctx: Ctx):
    """`y[i] += b[i % dim]`（累加，不是覆盖）。参数 `(y, b, n, dim)`。"""
    n, dim = 128, 8
    rng = np.random.default_rng(110)
    y0 = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(dim).astype(np.float32)
    py = ctx.buf(y0.copy())
    ctx.launch("vit_bias_kernel", (n + 63) // 64, 64, [py, ctx.buf(b), n, dim])
    ref = y0 + b[np.arange(n) % dim]
    return judge(np.abs(ctx.get(py, n) - ref).max(), ref)


@case("vit", "vit_bias_s_kernel")
def t_vit_bias_s(ctx: Ctx):
    """`y[i] += b[i % period]`；参数 `(y, b, n, dim, period)`。

    第 5 个参数**只有等于 `dim` 时**是干净的「按列加 bias」。实测
    （`dim=8`、`b=[1..8]`）：X=8 时 `b[i%8]` 全程正确；X=16 时只在
    `i%16 < 8` 的位置写；X=2/3/4 前 `X*dim` 个像 `b[i%X]` 之后就变模式；
    X=1/24 只有前 8 个像 `b[i]`。其它取值的语义没查清，用例只覆盖
    `X == dim`。要按列加 bias 建议直接用 `vit_bias_kernel`。
    """
    n, dim, period = 512, 64, 64
    rng = np.random.default_rng(111)
    y0 = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(dim).astype(np.float32)
    py = ctx.buf(y0.copy())
    ctx.launch("vit_bias_s_kernel", (n + 255) // 256, 256,
               [py, ctx.buf(b), n, dim, period])
    ref = y0 + b[np.arange(n) % period]
    return judge(np.abs(ctx.get(py, n) - ref).max(), ref)


@case("vit", "vit_gelu_kernel")
def t_vit_gelu(ctx: Ctx):
    """**精确 erf 形式**的 GELU（不是 tanh 近似）：实测与 erf 差 2e-8、
    与 tanh 近似差 4e-4，足以区分。参数 `(y, x, n, workgroup)`。"""
    n = 256
    x = np.linspace(-6, 6, n, dtype=np.float32)
    py = ctx.out(n)
    ctx.launch("vit_gelu_kernel", (n + 63) // 64, 64, [py, ctx.buf(x), n, 64])
    erf = np.vectorize(math.erf)
    ref = (0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))).astype(np.float32)
    return judge(np.abs(ctx.get(py, n) - ref).max(), ref, atol=1e-6, rtol=1e-6)


@case("vit", "vit_ln_kernel")
def t_vit_ln(ctx: Ctx):
    """标准 LayerNorm；**grid = rows、wg = 64**（与 `layernorm_k` 同约定）。
    参数 `(y, x, w, b, rows, dim, eps)`。"""
    rows, dim = 8, 128
    rng = np.random.default_rng(112)
    x = rng.standard_normal((rows, dim)).astype(np.float32)
    w = rng.standard_normal(dim).astype(np.float32)
    b = rng.standard_normal(dim).astype(np.float32)
    eps = 1e-5
    py = ctx.out(rows * dim)
    ctx.launch("vit_ln_kernel", rows, 64,
               [py, ctx.buf(x), ctx.buf(w), ctx.buf(b), rows, dim, np.float32(eps)])
    ref = (x - x.mean(axis=1, keepdims=True)) / np.sqrt(
        x.var(axis=1, keepdims=True) + eps) * w + b
    return judge(np.abs(ctx.get(py, rows * dim).reshape(rows, dim) - ref).max(), ref)


@case("attn", "attn_pv_part")
def t_attn_pv(ctx: Ctx):
    """`out[d] = Σ_j P[j]·Vt[d,j]` 的分块版（编译器生成的核）。

    参数 `(partial, p, vt, n_kv, vstride, nb_shift, jper)`：
    一个 work-item 负责 `(d, blk)`，`d = gid >> nb_shift`、`blk = gid & (2^nb_shift-1)`，
    块内做 `jper` 个 FMA 写 `partial[d*nb + blk]`；再交给 `reduce_blocks_k` 归约。
    `vstride` 是 Vt 的行距（可以与 n_kv 不同，方便直接挂在 `[dim, max_len]` 缓存上）。
    """
    dim, n_kv, nb, stride = 128, 512, 8, 640
    jper = n_kv // nb
    rng = np.random.default_rng(120)
    p = rng.random(n_kv).astype(np.float32)
    p /= p.sum()
    vt = np.zeros((dim, stride), np.float32)
    vt[:, :n_kv] = rng.standard_normal((dim, n_kv))
    part = ctx.out(dim * nb)
    ctx.launch("attn_pv_part", (dim * nb + 63) // 64, 64,
               [part, ctx.buf(p), ctx.buf(vt), n_kv, stride,
                nb.bit_length() - 1, jper])
    got = ctx.get(part, dim * nb).reshape(dim, nb).sum(axis=1)
    ref = vt[:, :n_kv] @ p
    return judge(np.abs(got - ref).max(), ref)


@case("seq", "split_qkv_k")
def t_split_qkv(ctx: Ctx):
    """把一个 token 的 `[qn + kn + vn]` 行拆成 q / k / v 三段。

    实测语义（探针确认）：源码行距 = `qn + kn + vn`；
    `q[t*qn+i] = x[t*row+i]`、`k[t*kn+i] = x[t*row+qn+i]`、
    `v[t*vn+i] = x[t*row+qn+kn+i]`；grid = T，wg = 64。

    **限制：`qn+kn+vn` 必须 ≤ 64**（一个 lane 一个元素、没有行内循环）。
    超出不报错、只是静默算错（实测 row=80 时最后 16 个元素错，误差数恒为
    `row-64`）；`tools/size_sweep.py` 会打印这条已知限制。
    """
    T, qn, kn, vn = 5, 4, 2, 3
    rng = np.random.default_rng(80)
    x = rng.standard_normal(T * (qn + kn + vn)).astype(np.float32)
    pq, pk, pv = ctx.out(T * qn), ctx.out(T * kn), ctx.out(T * vn)
    ctx.launch("split_qkv_k", T, 64, [pq, pk, pv, ctx.buf(x), T, qn, kn, vn])
    xt = x.reshape(T, qn + kn + vn)
    ref_q = xt[:, :qn].reshape(-1)
    ref_k = xt[:, qn:qn + kn].reshape(-1)
    ref_v = xt[:, qn + kn:].reshape(-1)
    got = np.concatenate([ctx.get(pq, T * qn), ctx.get(pk, T * kn),
                          ctx.get(pv, T * vn)])
    ref = np.concatenate([ref_q, ref_k, ref_v])
    return judge(np.abs(got - ref).max(), ref, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", help="只跑这个分组")
    ap.add_argument("--only", help="只跑名字里含这个子串的用例")
    ap.add_argument("--json", help="把结果写成 JSON")
    ap.add_argument("--repeat", type=int, default=1, help="每个用例重复次数，报告最差")
    args = ap.parse_args()

    rt = Runtime()
    rows = []
    bad = 0
    t0 = time.time()
    for group, name, fn in CASES:
        if args.group and args.group != group:
            continue
        if args.only and args.only not in name:
            continue
        metric, ok, note = float("nan"), False, ""
        try:
            for _ in range(max(1, args.repeat)):
                ctx = Ctx(rt)
                try:
                    m, good = fn(ctx)
                finally:
                    ctx.close()
                if not (m == m):        # nan
                    metric, ok, note = float("nan"), False, "指标为 NaN"
                    break
                metric = float(m) if metric != metric else max(metric, float(m))
                ok = bool(good)
                if not ok:
                    break
        except Exception as exc:                       # noqa: BLE001
            ok, note = False, f"{type(exc).__name__}: {exc}"
            metric = float("nan")
        if not ok:
            bad += 1
        rows.append({"group": group, "name": name, "ok": bool(ok),
                     "metric": None if metric != metric else round(metric, 12),
                     "note": note})
        flag = "ok  " if ok else "FAIL"
        print(f"{flag} {group:<12} {name:<24} max_abs={metric:.3e}"
              + (f"   {note}" if note else ""))

    dt = time.time() - t0
    print(f"\n{len(rows)} 个用例，{len(rows) - bad} 通过，{bad} 失败，"
          f"{dt:.1f}s")
    if args.json:
        p = pathlib.Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"total": len(rows), "failed": bad,
                                 "seconds": round(dt, 2), "cases": rows},
                                ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"报告 → {p}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
