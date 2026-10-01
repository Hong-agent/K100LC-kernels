#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解码（M=1..4）权重带宽基准：把「每条权重通路能跑多快」量成 GB/s。

解码阶段是纯访存瓶颈（每个权重只参与一次 MAC），所以唯一有意义的指标是
**权重带宽达成率** = 权重字节数 / 单次耗时 / HBM 峰值（本卡实测 896 GB/s）。

    source env.sh
    python3 tools/bench_decode.py                          # 默认真实形状
    python3 tools/bench_decode.py --n 17408 --k 5120 --iters 50
    python3 tools/bench_decode.py --paths w4a4,w4a8        # 只看某几条通路

覆盖：RT4 W4A4（M=1..4、双行变体）、RT4 W4A8（M=1..4）、compressed-tensors
INT4（`int4_dot_k`，W4A16）、GGUF 原生点积（`iq4nl_dot_k` / `q4k_dot_k`）、
以及 f32 参考路（`gemv_f32_warp_k`）。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

from k100lc_kernels import Runtime  # noqa: E402
from k100lc_kernels.model import div_magic  # noqa: E402

HBM_PEAK = 896e9        # 实测峰值读带宽（docs/PERFORMANCE.md）


class Bench:
    def __init__(self, rt: Runtime):
        self.rt = rt
        self._bufs: dict[int, int] = {}

    def buf(self, key, nbytes: int) -> int:
        p = self._bufs.get(key)
        if p is None or p[1] < nbytes:
            if p is not None:
                self.rt.free(p[0])
            q = self.rt.alloc(int(nbytes))
            self._bufs[key] = (q, int(nbytes))
            return q
        return p[0]

    def upload(self, key, data) -> int:
        n = data.nbytes if isinstance(data, np.ndarray) else len(data)
        p = self.buf(key, n)
        self.rt.upload(p, data)
        return p

    def time(self, fn, iters: int) -> float:
        fn()                      # warmup
        self.rt.sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        self.rt.sync()
        return (time.perf_counter() - t0) / iters

    def free(self) -> None:
        for p, _n in self._bufs.values():
            self.rt.free(p)
        self._bufs.clear()


def report(name: str, seconds: float, wbytes: int) -> dict:
    """`seconds` 是单次迭代的秒数。"""
    gbs = wbytes / seconds / 1e9
    print(f"  {name:<26} {seconds * 1e6:8.1f} us   {gbs:7.1f} GB/s   "
          f"{100 * gbs * 1e9 / HBM_PEAK:5.1f}% 峰值")
    return {"path": name, "us": round(seconds * 1e6, 1), "gbs": round(gbs, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=17408)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--paths", default="w4a4,w4a8,w4a4r2,ct4,gguf,f32")
    ap.add_argument("--layers", type=int, default=0,
                    help=">0 时额外跑「连续入队 N 层、只 sync 一次」，报 ms/token")
    args = ap.parse_args()

    n, k, iters = args.n, args.k, args.iters
    want = set(args.paths.split(","))
    rng = np.random.default_rng(0)
    rt = Runtime()
    b = Bench(rt)

    print(f"K100_LC 解码权重带宽基准：N={n} K={k} iters={iters} "
          f"threads={args.threads}（HBM 峰值 {HBM_PEAK / 1e9:.0f} GB/s）")

    # ---------------- RT4 权重（W4A4 / W4A8 共用） ----------------
    if want & {"w4a4", "w4a8", "w4a4r2"}:
        wq = np.ascontiguousarray(rng.integers(0, 1 << 32, size=n * (k // 8),
                                               dtype=np.uint64).astype(np.uint32))
        ws = (rng.uniform(1e-3, 5e-2, size=n * (k // 128)).astype(np.float32)
              .view(np.uint32) & np.uint32(0xFFFF0000)) >> 16
        ws = np.ascontiguousarray(ws.astype(np.uint16))
        pwq = b.upload("wq", wq)
        pws = b.upload("ws", ws)
        wbytes = wq.nbytes + ws.nbytes
        nwarp = args.threads // 64

        if "w4a4" in want:
            print(f"[W4A4] 权重 {wbytes / 1e6:.1f} MB")
            for m in (1, 2, 4):
                x = rng.standard_normal((m, k)).astype(np.float32)
                aq = b.buf(f"aq{m}", m * k // 2)
                asc = b.buf(f"asc{m}", m * (k // 32) * 4)
                py = b.buf(f"y{m}", m * n * 4)
                px = b.upload(f"x{m}", x)
                for r in range(m):
                    rt.launch("quant_act4", k // 32, 32,
                              [px + r * k * 4, aq + r * (k // 2),
                               asc + r * (k // 32) * 4, 1, k])

                def run(m=m):
                    for r in range(m):
                        rt.launch("quant_act4", k // 32, 32,
                                  [px + r * k * 4, aq + r * (k // 2),
                                   asc + r * (k // 32) * 4, 1, k])
                    grid = (n + nwarp - 1) // nwarp
                    rt.launch(f"gemv_w4a4<{m}>", grid, args.threads,
                              [pwq, pws, aq, asc, py, n, k])
                report(f"w4a4 M={m}（含激活量化）", b.time(run, iters), wbytes)

        if "w4a4r2" in want and k % 8 == 0:
            m = 4
            x = rng.standard_normal((m, k)).astype(np.float32)
            px = b.upload("xr2", x)
            aq = b.buf("aqr2", m * k // 2)
            asc = b.buf("ascr2", m * (k // 32) * 4)
            py = b.buf("yr2", m * n * 4)

            def run_r2():
                for r in range(m):
                    rt.launch("quant_act4", k // 32, 32,
                              [px + r * k * 4, aq + r * (k // 2),
                               asc + r * (k // 32) * 4, 1, k])
                rt.launch("gemv_w4a4_r2_k", (n + 7) // 8, 256,
                          [pwq, pws, aq, asc, py, n, k])
            report("w4a4 双行 M=4", b.time(run_r2, iters), wbytes)

        if "w4a8" in want:
            print(f"[W4A8] 权重 {wbytes / 1e6:.1f} MB")
            for m in (1, 4):
                x = rng.standard_normal((m, k)).astype(np.float32)
                px = b.upload(f"x8{m}", x)
                ae = b.buf(f"ae{m}", m * k // 2)
                ao = b.buf(f"ao{m}", m * k // 2)
                asc = b.buf(f"asc8{m}", m * (k // 128) * 4)
                asu = b.buf(f"asu{m}", m * (k // 128) * 4)
                py = b.buf(f"y8{m}", m * n * 4)
                rpp = 2 if m >= 2 else 1
                grid = (n + nwarp * rpp - 1) // (nwarp * rpp)

                def run8(m=m, grid=grid):
                    for r in range(m):
                        rt.launch("quant_act", k // 128, 32,
                                  [px + r * k * 4, ae + r * (k // 2),
                                   ao + r * (k // 2), asc + r * (k // 128) * 4,
                                   asu + r * (k // 128) * 4, 1, k])
                    rt.launch(f"gemv_w4a8<{m},false,{rpp}>", grid, args.threads,
                              [pwq, pws, ae, ao, asc, asu, py, n, k])
                report(f"w4a8 M={m}（含激活量化）", b.time(run8, iters), wbytes)
        b.free()

    # ---------------- compressed-tensors INT4（W4A16） ----------------
    if "ct4" in want:
        print("[INT4 W4A16：int4_dot_k + reduce_blocks_k]")
        nbpr = k // 128
        nblocks = n * nbpr
        wraw = np.ascontiguousarray(rng.integers(0, 1 << 32, size=n * (k // 8),
                                                 dtype=np.uint64).astype(np.uint32))
        scale = (rng.uniform(1e-3, 5e-2, size=n * nbpr).astype(np.float32)
                 .view(np.uint32) & np.uint32(0xFFFF0000)) >> 16
        scale = np.ascontiguousarray(scale.astype(np.uint16))
        pw = b.upload("ctw", wraw)
        ps = b.upload("cts", scale)
        x = rng.standard_normal(k).astype(np.float32)
        px = b.upload("ctx", x)
        parts = b.buf("ctp", nblocks * 4)
        py = b.buf("cty", n * 4)
        ids = b.upload("ctid", np.arange(n, dtype=np.uint32))
        argv = [pw, px, parts, nblocks, 64, nbpr, div_magic(nbpr, nblocks),
                n, div_magic(n, n),
                ps, ids, k // 2, n, div_magic(n, n)]

        def run_ct():
            rt.launch("int4_dot_k", (nblocks + 63) // 64, 64, argv)
            rt.launch("reduce_blocks_k", (n + 63) // 64, 64, [parts, py, n, nbpr])
        report("int4_dot_k（W4A16）", b.time(run_ct, iters),
               wraw.nbytes + scale.nbytes)
        b.free()

    # ---------------- GGUF 原生点积 ----------------
    if "gguf" in want:
        print("[GGUF 原生点积]")
        for kind, qk, bb, kernel in (("iq4nl", 32, 18, "iq4nl_dot_k"),
                                     ("q4k", 256, 144, "q4k_dot_k"),
                                     ("q6k", 256, 210, "q6k_dot_k")):
            nbpr = k // qk
            if nbpr == 0 or (n * nbpr) % 64:
                continue
            nblocks = n * nbpr
            raw = rng.integers(0, 256, size=nblocks * bb, dtype=np.uint8)
            d = int(np.float16(0.02).view(np.uint16))
            raw = raw.reshape(nblocks, bb)
            raw[:, 0], raw[:, 1] = d & 0xFF, (d >> 8) & 0xFF
            raw = np.ascontiguousarray(raw.reshape(-1))
            pw = b.upload(f"gw{kind}", raw)
            x = rng.standard_normal(k).astype(np.float32)
            px = b.upload(f"gx{kind}", x)
            parts = b.buf(f"gp{kind}", nblocks * 4)
            py = b.buf(f"gy{kind}", n * 4)
            ids = b.upload(f"gid{kind}", np.arange(n, dtype=np.uint32))
            argv = [pw, px, parts, nblocks, 64, nbpr, div_magic(nbpr, nblocks),
                    n, div_magic(n, n),
                    ids, nblocks // n * bb, n, div_magic(n, n)]

            def run_g(argv=argv, kernel=kernel):
                rt.launch(kernel, (nblocks + 63) // 64, 64, argv)
                rt.launch("reduce_blocks_k", (n + 63) // 64, 64,
                          [parts, py, n, nbpr])
            report(f"{kernel}", b.time(run_g, iters), raw.nbytes)
            b.free()

    # ---------------- NVFP4（E2M1 权重 + E4M3 块尺度，int8 激活） ----------------
    if "nvfp4" in want:
        print("[NVFP4 解码：nvfp4_gemv<A,1>]")
        wq = np.ascontiguousarray(rng.integers(0, 1 << 32, size=n * (k // 8),
                                               dtype=np.uint64).astype(np.uint32))
        ws = np.ascontiguousarray(rng.integers(1, 0x7F, size=n * (k // 16),
                                               dtype=np.uint8))
        pq, pws = b.upload("nwq", wq), b.upload("nws", ws)
        qa = b.upload("nqa", rng.integers(-127, 127, k, dtype=np.int8))
        qb = b.upload("nqb", rng.integers(-127, 127, k, dtype=np.int8))
        psc = b.upload("nsc", np.full(k // 16, 0.01, np.float32))
        py = b.buf("ny", n * 4)
        wbytes = wq.nbytes + ws.nbytes
        for a_rows in (1, 2, 4):
            def run_nv(a_rows=a_rows):
                rt.launch(f"nvfp4_gemv<{a_rows},1>", (n + a_rows - 1) // a_rows, 256,
                          [pq, pws, qa, qb, psc, py, n, k, 1.0])
            report(f"nvfp4_gemv<{a_rows},1>", b.time(run_nv, iters), wbytes)
        b.free()

    # ---------------- f32 参考路 ----------------
    if "f32" in want:
        nf = min(n, 4096)          # f32 权重太大，缩小 N 只做参考
        print(f"[f32 参考：gemv_f32_warp_k，N={nf}]")
        w = rng.standard_normal((nf, k)).astype(np.float32)
        pw = b.upload("fw", w)
        x = rng.standard_normal(k).astype(np.float32)
        px = b.upload("fx", x)
        py = b.buf("fy", nf * 4)

        def run_f():
            rt.launch("gemv_f32_warp_k", nf, 64, [pw, px, py, nf, k, 64])
        report("gemv_f32_warp_k", b.time(run_f, iters), w.nbytes)
        b.free()

    # ---------------- 端到端：连续入队 N 层、只 sync 一次 ----------------
    if args.layers > 0 and want & {"w4a4", "w4a8"}:
        print(f"[端到端] 连续入队 {args.layers} 层（复用同一份权重，模拟逐 token 的"
              f"权重流），只 sync 一次")
        wq = np.ascontiguousarray(rng.integers(0, 1 << 32, size=n * (k // 8),
                                               dtype=np.uint64).astype(np.uint32))
        ws = (rng.uniform(1e-3, 5e-2, size=n * (k // 128)).astype(np.float32)
              .view(np.uint32) & np.uint32(0xFFFF0000)) >> 16
        ws = np.ascontiguousarray(ws.astype(np.uint16))
        pwq, pws = b.upload("lwq", wq), b.upload("lws", ws)
        x = rng.standard_normal(k).astype(np.float32)
        px = b.upload("lx", x)
        nwarp = args.threads // 64
        for path in ("w4a8", "w4a4"):
            if path not in want:
                continue
            if path == "w4a8":
                ae = b.buf("lae", k // 2)
                ao = b.buf("lao", k // 2)
                asc = b.buf("lasc", (k // 128) * 4)
                asu = b.buf("lasu", (k // 128) * 4)
                grid = (n + nwarp - 1) // nwarp

                def run_layer():
                    rt.launch("quant_act", k // 128, 32,
                              [px, ae, ao, asc, asu, 1, k])
                    rt.launch("gemv_w4a8<1,false,1>", grid, args.threads,
                              [pwq, pws, ae, ao, asc, asu, px, n, k])
            else:
                aq = b.buf("laq", k // 2)
                asc = b.buf("lasc4", (k // 32) * 4)
                grid = (n + nwarp - 1) // nwarp

                def run_layer():
                    rt.launch("quant_act4", k // 32, 32, [px, aq, asc, 1, k])
                    rt.launch("gemv_w4a4<1>", grid, args.threads,
                              [pwq, pws, aq, asc, px, n, k])
            run_layer()
            rt.sync()
            t0 = time.perf_counter()
            for _ in range(args.layers):
                run_layer()
            rt.sync()
            ms = (time.perf_counter() - t0) * 1e3
            print(f"  {path}: {args.layers} 层 {ms:7.2f} ms → {ms / args.layers * 1e3:6.1f}"
                  f" us/层   {wq.nbytes / (ms / args.layers * 1e-3) / 1e9:5.1f} GB/s")
        b.free()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
