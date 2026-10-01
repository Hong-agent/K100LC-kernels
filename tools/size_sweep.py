#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""尺寸扫描：把「按固定尺寸对过账」的内核换一批尺寸再跑一遍。

`tools/selftest_all.py` 每个内核只跑一组尺寸，而这张卡上**已经出现过两次
尺寸相关的坑**（`concat2_k` 的参数是半长、`vit_bias_s_kernel` 的周期参数），
所以这里把容易踩尺寸的内核按一批尺寸扫一遍。

    source env.sh
    python3 tools/size_sweep.py          # 全部
    python3 tools/size_sweep.py --group norm

**历史上探明的尺寸坑**（现在都已修掉/澄清，扫描里都覆盖）：

* `split_qkv_k`：旧版每个 token 的行（`qn+kn+vn`）**不能超过 64 个元素**
  （一个 lane 一个元素、没有行内循环，超出静默算错）。v1.8.0 用编译器 DSL
  重写了这个内核（加了行内并行循环），现在任意行宽都正确。
* `concat2_k`：第 5 个参数是「半长」，输出要给 `2n`。
* `vit_bias_s_kernel`：第 5 个参数只有等于 `dim` 时语义干净。
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from k100lc_kernels import Runtime  # noqa: E402

FAILS: list[str] = []


class Sweep:
    def __init__(self, rt: Runtime):
        self.rt = rt
        self.ptrs: list[int] = []

    def buf(self, a) -> int:
        a = np.ascontiguousarray(a) if isinstance(a, np.ndarray) else bytes(a)
        n = a.nbytes if isinstance(a, np.ndarray) else len(a)
        p = self.rt.alloc(max(int(n), 4))
        self.rt.upload(p, a)
        self.ptrs.append(p)
        return p

    def get(self, p, n, dt=np.float32):
        self.rt.sync()
        return self.rt.download(p, int(n), dt)

    def run(self, name, grid, wg, argv):
        self.rt.launch(name, int(grid), int(wg), list(argv))
        self.rt.sync()

    def free(self):
        for p in self.ptrs:
            self.rt.free(p)
        self.ptrs.clear()


def check(s: Sweep, title: str, bad: int, note: str = "") -> None:
    if bad:
        FAILS.append(f"{title} ({bad} 个元素不对)")
    print(f"  {'✔' if bad == 0 else '✘ ' + str(bad)} {title} {note}")


def g_elementwise(s: Sweep, rng) -> None:
    print("[逐元素：n 边界（含 63/64/65）]")
    for n in (1, 63, 64, 65, 127, 128, 1000, 4096):
        y = s.buf(np.zeros(n, np.float32))
        s.run("fill_k", (n + 63) // 64, 64, [y, np.float32(1.25), n])
        check(s, f"fill n={n}", int((s.get(y, n) != 1.25).sum()))
        x = np.arange(n, dtype=np.float32)
        xp = s.buf(x.copy())
        s.run("scale_mul_k", (n + 63) // 64, 64, [xp, np.float32(2.0), n])
        check(s, f"scale n={n}", int((s.get(xp, n) != x * 2).sum()))
        ap, bp = s.buf(x.copy()), s.buf(np.ones(n, np.float32))
        s.run("add_inplace_k", (n + 63) // 64, 64, [ap, bp, n])
        check(s, f"add n={n}", int((s.get(ap, n) != x + 1.0).sum()))
    for n in (64, 65, 1000, 4096, 16384):
        a = rng.standard_normal(n).astype(np.float32)
        b = rng.standard_normal(n).astype(np.float32)
        y = s.buf(np.zeros(n, np.float32))
        s.run("silu_mul_k", (n + 63) // 64, 64, [y, s.buf(a), s.buf(b), n])
        check(s, f"silu_mul n={n}",
              int((np.abs(s.get(y, n) - a / (1 + np.exp(-a)) * b) > 1e-5).sum()))


def g_norm(s: Sweep, rng) -> None:
    print("[归一化 / softmax：dim / cols 扫描]")
    for dim in (64, 128, 256, 512, 1024, 2560, 5120):
        rows = 3
        x = rng.standard_normal((rows, dim)).astype(np.float32)
        w = rng.standard_normal(dim).astype(np.float32)
        y = s.buf(np.zeros(rows * dim, np.float32))
        s.run("rmsnorm_k", rows, 64, [y, s.buf(x), s.buf(w), dim, np.float32(1e-5), 0])
        ref = x / np.sqrt((x ** 2).mean(1, keepdims=True) + 1e-5) * w
        check(s, f"rmsnorm dim={dim}",
              int((np.abs(s.get(y, rows * dim).reshape(rows, dim) - ref) > 1e-4).sum()))
    for cols in (64, 128, 256, 512, 1024, 4096, 16384):
        x = (rng.standard_normal((3, cols)).astype(np.float32) * 2)
        e = np.exp(x - x.max(1, keepdims=True))
        ref = e / e.sum(1, keepdims=True)
        y = s.buf(np.zeros(3 * cols, np.float32))
        s.run("softmax_k", 3, 64, [y, s.buf(x), 3, cols, 64])
        check(s, f"softmax_k cols={cols}",
              int((np.abs(s.get(y, 3 * cols).reshape(3, cols) - ref) > 1e-5).sum()))
        if cols % 256 == 0:
            y2 = s.buf(np.zeros(3 * cols, np.float32))
            s.run("softmax_vec_k", 3, 64, [y2, s.buf(x), 3, cols, 64])
            check(s, f"softmax_vec_k cols={cols}",
                  int((np.abs(s.get(y2, 3 * cols).reshape(3, cols) - ref) > 1e-5).sum()))
    for n in (32, 64, 128, 256, 1024, 5120):
        x = rng.standard_normal((4, n)).astype(np.float32)
        xp = s.buf(x.copy())
        s.run("l2norm_k", 4, 64, [xp, n, np.float32(1e-5)])
        ref = x / np.sqrt((x ** 2).sum(1, keepdims=True) + 1e-5)
        check(s, f"l2norm n={n}",
              int((np.abs(s.get(xp, 4 * n).reshape(4, n) - ref) > 1e-5).sum()))


def g_gemv(s: Sweep, rng) -> None:
    print("[GEMV / 归约：(n,k) 扫描]")
    for n, k in ((1, 64), (64, 64), (1000, 64), (17, 512), (3000, 5120)):
        w = rng.standard_normal((n, k)).astype(np.float32)
        x = rng.standard_normal(k).astype(np.float32)
        y = s.buf(np.zeros(n, np.float32))
        s.run("gemv_f32_warp_k", n, 64, [s.buf(w), s.buf(x), y, n, k, 64])
        ref = w @ x
        rel = float(np.abs(s.get(y, n) - ref).max() / max(1.0, float(np.abs(ref).max())))
        check(s, f"gemv_warp n={n} k={k}", 0 if rel < 1e-5 else 1, f"rel={rel:.1e}")
    for nbpr in (1, 2, 3, 5, 8, 16, 17):
        rows = 64
        p = rng.standard_normal(rows * nbpr).astype(np.float32)
        y = s.buf(np.zeros(rows, np.float32))
        s.run("reduce_blocks_k", 1, 64, [s.buf(p), y, rows, nbpr])
        ref = p.reshape(rows, nbpr).sum(1)
        check(s, f"reduce nbpr={nbpr}",
              int((np.abs(s.get(y, rows) - ref) > 1e-4).sum()))
    for rows, dim in ((1, 64), (7, 256), (33, 5120)):
        src = rng.standard_normal((rows, dim)).astype(np.float32)
        idx = rng.permutation(rows).astype(np.uint32)
        dst = s.buf(np.zeros(rows * dim, np.float32))
        s.run("gather_rows_k", rows, 64, [dst, s.buf(src), s.buf(idx), rows, dim])
        check(s, f"gather_rows r={rows} d={dim}",
              int((s.get(dst, rows * dim).reshape(rows, dim) != src[idx]).sum()))


def g_moe_topk(s: Sweep, rng) -> None:
    print("[MoE / top-k / router]")
    for rows, dim, ne in ((1, 64, 2), (7, 2048, 4), (16, 5120, 8)):
        eo = rng.standard_normal((ne, rows, dim)).astype(np.float32)
        w = rng.random((rows, ne)).astype(np.float32)
        y = s.buf(np.zeros(rows * dim, np.float32))
        s.run("moe_combine_k", rows, 64, [y, s.buf(eo), s.buf(w), ne, rows, dim])
        ref = np.einsum("re,erd->rd", w, eo)
        check(s, f"moe_combine r={rows} d={dim} e={ne}",
              int((np.abs(s.get(y, rows * dim).reshape(rows, dim) - ref)
                   > 1e-4 * max(1.0, float(np.abs(ref).max()))).sum()))
    for rows, cols, k in ((1, 64, 4), (17, 1000, 16), (5, 100, 1)):
        x = rng.standard_normal((rows, cols)).astype(np.float32)
        ip = s.buf(np.zeros(rows * k, np.int32))
        vp = s.buf(np.zeros(rows * k, np.float32))
        s.run("topk_k", (rows + 63) // 64, 64, [s.buf(x), ip, vp, rows, cols, k])
        order = np.argsort(-x, axis=1)[:, :k]
        vals = np.take_along_axis(x, order, axis=1)
        bad = int((s.get(ip, rows * k, np.int32).reshape(rows, k) != order).sum())
        bad += int((s.get(vp, rows * k).reshape(rows, k) != vals).sum())
        check(s, f"topk r={rows} c={cols} k={k}", bad)
    for rows, ne in ((1, 64), (8, 1024), (2, 40)):
        lg = rng.standard_normal((rows, ne)).astype(np.float32)
        ip = s.buf(np.zeros(rows * 10, np.int32))
        wp = s.buf(np.zeros(rows * 10, np.float32))
        s.run("router_top10_k", rows, 64, [s.buf(lg), ip, wp, rows, ne])
        top = np.argsort(-lg, axis=1)[:, :10]
        ex = np.exp(lg[np.arange(rows)[:, None], top] - lg.max(1, keepdims=True))
        rw = ex / ex.sum(1, keepdims=True)
        bad = int((s.get(ip, rows * 10, np.int32).reshape(rows, 10) != top).sum())
        bad += int((np.abs(s.get(wp, rows * 10).reshape(rows, 10) - rw) > 1e-5).sum())
        check(s, f"router r={rows} e={ne}", bad)


def g_seq_vit(s: Sweep, rng) -> None:
    print("[序列 / 视觉塔]")
    # v1.8.0 起 split_qkv_k 的行宽没有 64 的限制了（旧版有）
    for T, qn, kn, vn in ((1, 4, 2, 3), (5, 16, 8, 8), (3, 7, 1, 2), (8, 32, 16, 16),
                          (5, 64, 32, 48), (3, 200, 100, 257), (4, 1, 1, 1)):
        x = rng.standard_normal(T * (qn + kn + vn)).astype(np.float32)
        pq = s.buf(np.zeros(T * qn, np.float32))
        pk = s.buf(np.zeros(T * kn, np.float32))
        pv = s.buf(np.zeros(T * vn, np.float32))
        s.run("split_qkv_k", T, 64, [pq, pk, pv, s.buf(x), T, qn, kn, vn])
        xt = x.reshape(T, qn + kn + vn)
        bad = int((s.get(pq, T * qn) != xt[:, :qn].reshape(-1)).sum())
        bad += int((s.get(pk, T * kn) != xt[:, qn:qn + kn].reshape(-1)).sum())
        bad += int((s.get(pv, T * vn) != xt[:, qn + kn:].reshape(-1)).sum())
        check(s, f"split_qkv T={T} qn={qn} kn={kn} vn={vn}", bad,
              f"(row={qn + kn + vn})")
    for dim in (64, 128, 512):
        rows = 3
        x = rng.standard_normal((rows, dim)).astype(np.float32)
        w = rng.standard_normal(dim).astype(np.float32)
        b = rng.standard_normal(dim).astype(np.float32)
        y = s.buf(np.zeros(rows * dim, np.float32))
        s.run("vit_ln_kernel", rows, 64,
              [y, s.buf(x), s.buf(w), s.buf(b), rows, dim, np.float32(1e-5)])
        ref = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-5) * w + b
        check(s, f"vit_ln dim={dim}",
              int((np.abs(s.get(y, rows * dim).reshape(rows, dim) - ref) > 1e-4).sum()))
    erf = np.vectorize(math.erf)
    for n in (64, 1000, 4096):
        x = rng.standard_normal(n).astype(np.float32)
        y = s.buf(np.zeros(n, np.float32))
        s.run("vit_gelu_kernel", (n + 63) // 64, 64, [y, s.buf(x), n, 64])
        ref = (0.5 * x * (1 + erf(x / math.sqrt(2)))).astype(np.float32)
        check(s, f"vit_gelu n={n}", int((np.abs(s.get(y, n) - ref) > 1e-6).sum()))
    for n, dim in ((1000, 16), (4096, 128)):
        b = rng.standard_normal(dim).astype(np.float32)
        y0 = rng.standard_normal(n).astype(np.float32)
        y = s.buf(y0.copy())
        s.run("vit_bias_kernel", (n + 63) // 64, 64, [y, s.buf(b), n, dim])
        check(s, f"vit_bias n={n} dim={dim}",
              int((s.get(y, n) != y0 + b[np.arange(n) % dim]).sum()))


def g_attn(s: Sweep, rng) -> None:
    """解码注意力 `Attention` 的尺寸扫描（dim / n_kv / max_len 三代组合）。

    这条路上有三个「按长度分派」的开关，尺寸一变就会换内核，所以必须换着尺寸扫：

    * `K·q`：dim 是 128 的倍数且 ≤ 512 时走 `gemv_f32_rows8_k`，否则回退；
    * softmax：pad ≥ 2048 走分块四件套（`block_max_k`/`reduce_max1_k`/
      `block_exp_sum_k`/`reduce_sum1_k`）+ 输出侧 `div_scalar_k`，否则
      `softmax_vec_k`；
    * `Vt·P`：dim%8==0 且 pad%128==0 时走 `gemv_f32_rows8_split_k`，否则
      回退到 `attn_pv_part`/`gemv_f32_warp_k`。

    覆盖点还包括：`n_kv` 不是 256 的倍数（尾部 -1e30 掩码）、`pad > n_kv`
    （Vt 行距 = max_len > 列数）、以及 n_kv=1 这种退化情形。
    """
    from k100lc_kernels.model import Attention  # 延迟导入，避免无关组变慢

    print("[解码注意力：dim / n_kv / max_len 扫描]")
    for dim in (64, 128, 256):
        for n_kv, max_len in ((1, 256), (255, 256), (300, 512), (768, 1024),
                              (1000, 2048), (3000, 4096), (4096, 4096)):
            if n_kv > max_len:
                continue
            attn = Attention(s.rt, dim=dim, max_len=max_len,
                             tag=f"sw{dim}_{n_kv}")
            k = rng.standard_normal((n_kv, dim)).astype(np.float32)
            v = rng.standard_normal((n_kv, dim)).astype(np.float32)
            q = rng.standard_normal(dim).astype(np.float32)
            attn.append(k, v)
            got = attn.forward(q)
            score = (k @ q) / np.sqrt(dim)
            e = np.exp(score - score.max())
            ref = (e / e.sum()) @ v
            rel = float(np.abs(got - ref).max() / max(1e-9, float(np.abs(ref).max())))
            check(s, f"attention dim={dim} n_kv={n_kv} max_len={max_len}",
                  0 if rel < 1e-5 else 1,
                  f"max_rel={rel:.2e}")
            attn.ws.free()
    # V → Vt 的分块转置（编译器 + DSL 共享内存生成的核）：rows 跨 64 边界、y0 非零
    for rows, dim, y0 in ((1, 128, 0), (1, 128, 5), (63, 128, 0), (64, 128, 9),
                          (65, 128, 0), (130, 256, 3), (256, 64, 0)):
        max_len = 512
        cshift = (dim // 64).bit_length() - 1
        v = rng.standard_normal((rows, dim)).astype(np.float32)
        vt = np.full((dim, max_len), -7.0, np.float32)
        pvt = s.buf(vt)
        s.run("vt_scatter_k", ((rows + 63) // 64) * (dim // 64), 64,
              [pvt, s.buf(v), rows, dim, dim, max_len, y0, cshift])
        ref = vt.copy()
        ref[:, y0:y0 + rows] = v.T
        check(s, f"vt_scatter rows={rows} dim={dim} y0={y0}",
              int((s.get(pvt, dim * max_len).reshape(dim, max_len) != ref).sum()))


GROUPS = {
    "elementwise": g_elementwise,
    "norm": g_norm,
    "gemv": g_gemv,
    "moe": g_moe_topk,
    "seq": g_seq_vit,
    "attn": g_attn,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", help="只跑这一组")
    args = ap.parse_args()
    rt = Runtime()
    rng = np.random.default_rng(1234)
    for name, fn in GROUPS.items():
        if args.group and args.group != name:
            continue
        s = Sweep(rt)
        try:
            fn(s, rng)
        finally:
            s.free()
    print()
    if FAILS:
        print(f"失败 {len(FAILS)} 项：")
        for f in FAILS:
            print("  ", f)
        return 1
    print("尺寸扫描全部通过 ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
