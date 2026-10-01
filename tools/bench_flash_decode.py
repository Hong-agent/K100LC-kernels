#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解码注意力（`flash_dec_part_k` + `flash_dec_comb_k`）的分项耗时与长上下文缩放。

    source env.sh
    python3 tools/bench_flash_decode.py                    # 真实调用：part+comb
    python3 tools/bench_flash_decode.py --dh 64 --ctx 512,2048,4096
    python3 tools/bench_flash_decode.py --what comb --splits 1,4,16,64

三种量法：

* `--what forward`（默认）走 `FlashAttention.forward_device`：两条 launch、
  一次 sync，就是模型里的真实调用。按 `n_kv` 扫，看长上下文怎么缩放。
* `--what part` 只发 `flash_dec_part_k`（grid = 头数 × 分块数）。
* `--what comb` 只发 `flash_dec_comb_k`（grid = 头数），按 `--splits` 扫
  `nsplit`——它原先每个分块一次访存往返，是长上下文里唯一线性增长的环节。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from k100lc_kernels import Runtime  # noqa: E402
from k100lc_kernels.model import FlashAttention, cdiv  # noqa: E402


class Timer:
    def __init__(self, rt: Runtime, iters: int):
        self.rt, self.iters = rt, iters

    def __call__(self, fn) -> float:
        fn()
        self.rt.sync()
        t0 = time.perf_counter()
        for _ in range(self.iters):
            fn()
        self.rt.sync()
        return (time.perf_counter() - t0) / self.iters * 1e6


def bench_comb(rt: Runtime, nh: int, dh: int, splits: list[int], iters: int) -> None:
    out = rt.alloc(nh * dh * 4)
    po = rt.alloc(nh * 64 * dh * 4)
    pm = rt.alloc(nh * 64 * 4)
    pl = rt.alloc(nh * 64 * 4)
    rng = np.random.default_rng(7)
    rt.upload(po, rng.standard_normal(nh * 64 * dh).astype(np.float32))
    rt.upload(pm, rng.standard_normal(nh * 64).astype(np.float32))
    rt.upload(pl, np.abs(rng.standard_normal(nh * 64)).astype(np.float32) + 0.5)
    inv = float(np.float32(1.0 / np.sqrt(dh)))
    timer = Timer(rt, iters)
    print(f"comb  n_heads={nh} dh={dh} grid={nh} workgroup=64")
    print(f"{'nsplit':>7} {'us':>9} {'us/head':>9} {'GB/s(po)':>9}")
    for ns in splits:
        us = timer(lambda: rt.launch("flash_dec_comb_k", nh, 64,
                                     [out, po, pm, pl, dh, ns, inv]))
        gb = nh * ns * dh * 4 / (us * 1e-6) / 1e9
        print(f"{ns:7d} {us:9.2f} {us / nh:9.2f} {gb:9.1f}")
    for p in (out, po, pm, pl):
        rt.free(p)


def _attn(rt: Runtime, nh: int, dh: int, n_kv: int, tag: str) -> FlashAttention:
    attn = FlashAttention(rt, nh, dh, max_len=n_kv, tag=tag)
    rng = np.random.default_rng(11)
    k = rt.alloc(n_kv * nh * dh * 4)
    v = rt.alloc(n_kv * nh * dh * 4)
    rt.upload(k, rng.standard_normal(n_kv * nh * dh).astype(np.float32))
    rt.upload(v, rng.standard_normal(n_kv * nh * dh).astype(np.float32))
    attn.append_device(k, v, n_kv)
    rt.sync()
    rt.free(k)
    rt.free(v)
    return attn


def bench_forward(rt: Runtime, nh: int, dh: int, ctxs: list[int], iters: int) -> None:
    timer = Timer(rt, iters)
    print(f"forward  n_heads={nh} dh={dh}（part + comb，每条各一次 sync）")
    print(f"{'n_kv':>7} {'nsplit':>7} {'part':>9} {'comb':>9} {'total':>9} "
          f"{'GB/s(KV)':>9}")
    for n_kv in ctxs:
        attn = _attn(rt, nh, dh, n_kv, f"bfd{nh}_{dh}_{n_kv}")
        pad, nsplit, _R = attn.plan(n_kv)
        cshift = nsplit.bit_length() - 1
        inv = float(np.float32(1.0 / np.sqrt(dh)))
        t_part = timer(lambda: rt.launch(
            "flash_dec_part_k", nh * nsplit, 64,
            [attn.po, attn.pm, attn.pl, attn.q, attn.kt, attn.v, n_kv, dh,
             pad, attn.max_len, cshift, inv, attn.dim]))
        t_comb = timer(lambda: rt.launch(
            "flash_dec_comb_k", nh, 64,
            [attn.out, attn.po, attn.pm, attn.pl, dh, nsplit, inv]))
        t_all = timer(lambda: attn.forward_device(attn.q, n_kv=n_kv))
        # 每个 token 要读 K + V 两份缓存（f32，每头 dh 宽）
        gb = 2 * n_kv * nh * dh * 4 / (t_all * 1e-6) / 1e9
        print(f"{n_kv:7d} {nsplit:7d} {t_part:9.2f} {t_comb:9.2f} {t_all:9.2f} "
              f"{gb:9.1f}")
        attn.ws.free()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=("forward", "part", "comb"), default="forward")
    ap.add_argument("--nheads", type=int, default=8)
    ap.add_argument("--dh", type=int, default=128)
    ap.add_argument("--ctx", type=str, default="128,512,1024,2048,4096")
    ap.add_argument("--splits", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    rt = Runtime()
    if args.what == "comb":
        bench_comb(rt, args.nheads, args.dh,
                   [int(s) for s in args.splits.split(",") if s], args.iters)
    else:
        bench_forward(rt, args.nheads, args.dh,
                      [int(s) for s in args.ctx.split(",") if s], args.iters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
