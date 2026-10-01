"""RT4（int4 权重）AI 模型支持：格式加载 + W4A4/W4A8 最强通路 GEMV。

RT4 是 int4 大模型权重的单文件打包格式：一个 `.rt4` 权重文件 +
一个同名 `.json` manifest（记录每层的形状、q/s 区偏移与 group）。

    i4 线性层 [N,K]，group=128：
      q 区  N*(K/2) 字节   有符号 int4（-8..7），沿 K 两两打包，低半字节 = 偶数 k
      s 区  N*(K/128)*2    f16 尺度，w[n,k] = q[n,k] * s[n,k//128]

本模块把两条“最强通路”包成可直接给 AI 模型用的接口：

* `W4A8`：int4 权重 × int8 激活（`v_dot4_i32_i8`），解码路径实测 566 GB/s；
* `W4A4`：int4 权重 × int4 激活（`v_dot8_i32_i4`，本卡最高算力 75.6 TMAC/s）。

```python
from k100lc_kernels import Runtime, RT4File, W4Runner
rt = Runtime()
f = RT4File("/path/model.rt4")               # 同级 .rt4.json 自动加载
t = f.tensor("model.language_model.layers.0.mlp.gate_proj.weight")
wq, ws = f.upload_weight(rt, t)
r = W4Runner(rt)
y = r.gemv("w4a8", wq, ws, x, t.shape[0], t.shape[1])   # x: [M,K], M=1..4
```
"""
from __future__ import annotations

import dataclasses
import json
import mmap
import pathlib

import numpy as np

from .runtime import Runtime

GRP = 128                      # 权重分组（RT4 固定）
SG = 32                        # W4A4 的激活子分组（quant_act4）
W4A8_KERNEL = {1: "gemv_w4a8<1,false,1>", 2: "gemv_w4a8<2,false,2>",
               3: "gemv_w4a8<3,false,2>", 4: "gemv_w4a8<4,false,2>"}
W4A4_KERNEL = {1: "gemv_w4a4<1>", 2: "gemv_w4a4<2>",
               3: "gemv_w4a4<3>", 4: "gemv_w4a4<4>"}


def _div_magic(d: int, max_i: int) -> int:
    """返回 M，使 `(i * M) >> 32 == i // d` 对一切 `0 <= i < max_i` 成立。

    与 `tools/gen_gemv_qdot.py::div_magic` 同一算法；`gemm_w4a4_flat` 用它把
    扁平 blockIdx.x 还原成 (bm, bn)。
    """
    if d <= 1:
        raise ValueError("_div_magic 只支持 d >= 2（d=1 时直接走分块回退）")
    m = (1 << 32) // d + 1
    if m >= (1 << 32):
        raise ValueError(f"_div_magic: d={d} 太大")
    i = np.arange(max_i, dtype=np.uint64)
    if not np.array_equal((i * np.uint64(m)) >> np.uint64(32), i // np.uint64(d)):
        raise ValueError(f"_div_magic: d={d} max_i={max_i} 无法精确表示")
    return int(m)


@dataclasses.dataclass(frozen=True)
class RT4Tensor:
    name: str
    kind: str
    shape: tuple[int, int]
    group: int
    q_off: int
    s_off: int
    nbytes: int

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    @property
    def q_bytes(self) -> int:
        return self.rows * (self.cols // 2)

    @property
    def s_bytes(self) -> int:
        return self.rows * (self.cols // self.group) * 2


class RT4File:
    """RT4 权重文件 + manifest 的只读访问（mmap，不复制整份权重）。"""

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        if self.path.suffix != ".rt4":
            raise ValueError(f"RT4File 需要 .rt4 文件：{self.path}")
        self.manifest_path = self.path.with_suffix(".rt4.json")
        if not self.manifest_path.is_file():
            cand = self.path.with_suffix(".json")
            if not cand.is_file():
                raise FileNotFoundError(f"找不到 manifest：{self.manifest_path}")
            self.manifest_path = cand
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "RT4-v1":
            raise ValueError(f"不支持的 RT4 格式：{self.manifest.get('format')}")
        self.group = int(self.manifest.get("group", GRP))
        self._by_name = {t["name"]: t for t in self.manifest["tensors"]}
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        self._mm.close()
        self._fh.close()

    def __enter__(self) -> "RT4File":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def tensor(self, name: str) -> RT4Tensor:
        try:
            t = self._by_name[name]
        except KeyError as e:
            raise KeyError(f"RT4 里没有张量 {name}") from e
        return RT4Tensor(name=t["name"], kind=t["kind"],
                         shape=tuple(t["shape"]), group=int(t.get("group", self.group)),
                         q_off=int(t["q_off"]), s_off=int(t["s_off"]),
                         nbytes=int(t["nbytes"]))

    def names(self, kind: str | None = None) -> list[str]:
        return [t["name"] for t in self.manifest["tensors"]
                if kind is None or t["kind"] == kind]

    def weight_bytes(self, t: RT4Tensor) -> tuple[bytes, bytes]:
        """返回 (int4 打包字节, f16 尺度字节)。"""
        if t.kind != "i4":
            raise ValueError(f"{t.name} 的 RT4 kind={t.kind}，W4A8/W4A4 只支持 i4")
        if t.cols % t.group or t.cols % 8:
            raise ValueError(f"{t.name}: K={t.cols} 不是 {t.group} 的整数倍")
        return (self._mm[t.q_off:t.q_off + t.q_bytes],
                self._mm[t.s_off:t.s_off + t.s_bytes])

    def upload_weight(self, rt: Runtime, t: RT4Tensor) -> tuple[int, int]:
        """把一层的 q/s 上传显存，返回 (wq_ptr, ws_ptr)。调用方负责 free。"""
        wq, ws = self.weight_bytes(t)
        pwq = rt.alloc(len(wq))
        rt.upload(pwq, wq)
        pws = rt.alloc(len(ws))
        rt.upload(pws, ws)
        return pwq, pws

    def upload_weight_gemm(self, rt: Runtime, t: RT4Tensor) -> tuple[int, int]:
        """GEMM 路径的权重：int4 q + **组优先 f32 尺度** sb[g*N+n]。

        RT4 原生尺度是行优先 f16 s[N,K/G]；`gemm_w4a4` 需要组优先 f32，
        这里做一次小转置（N*K/G 个 float，一层几 MB）。q 区布局与 GEMM 一致，
        无需改动。返回 (wq_ptr, wsc_gm_ptr)。
        """
        wq, ws = self.weight_bytes(t)
        n, k = t.shape
        s = np.frombuffer(ws, dtype=np.float16).astype(np.float32).reshape(n, k // t.group)
        wsc = np.ascontiguousarray(s.T)          # [K/G, N]
        pwq = rt.alloc(len(wq))
        rt.upload(pwq, wq)
        pwsc = rt.alloc(wsc.nbytes)
        rt.upload(pwsc, wsc)
        return pwq, pwsc

    def dequant(self, t: RT4Tensor, rows: int | None = None,
                start: int = 0) -> np.ndarray:
        """参考反量化成 f32 [rows?, K]（默认整层）；分块避免大层占满主机内存。"""
        wq, ws = self.weight_bytes(t)
        if rows is None:
            rows = t.rows - start
        out = np.empty((rows, t.cols), dtype=np.float32)
        qrow = t.cols // 2
        srow = t.cols // t.group
        for r in range(rows):
            rr = start + r
            q = np.frombuffer(wq[rr * qrow:(rr + 1) * qrow], dtype=np.uint8)
            lo = (q & 0x0F).astype(np.int32)
            hi = (q >> 4).astype(np.int32)
            lo[lo >= 8] -= 16
            hi[hi >= 8] -= 16
            vals = np.empty(t.cols, dtype=np.int32)
            vals[0::2] = lo
            vals[1::2] = hi
            s = np.frombuffer(ws[rr * srow * 2:(rr + 1) * srow * 2],
                              dtype=np.float16).astype(np.float32)
            out[r] = (vals.reshape(-1, t.group) * s[:, None]).reshape(-1)
        return out


class W4Runner:
    """在 `Runtime` 上跑 W4A8 / W4A4 GEMV（M=1..4，AI 模型解码/MTP 校验批）。"""

    # gemm_w4a4 的分块/量化组（与 kernels/gemm_core.h 一致）
    BM = 128        # tokens per block（M 必须按它对齐）
    BN = 64         # output channels per block（N 必须是它的倍数）
    QG = 128        # 量化组（K 必须是它的倍数）

    def __init__(self, rt: Runtime):
        self.rt = rt
        self._bufs: dict[tuple[str, int], int] = {}

    def _buf(self, name: str, nbytes: int) -> int:
        key = (name, nbytes)
        if key not in self._bufs:
            self._bufs[key] = self.rt.alloc(nbytes)
        return self._bufs[key]

    def close(self) -> None:
        for p in self._bufs.values():
            self.rt.free(p)
        self._bufs.clear()

    def gemv_device(self, kind: str, wq: int, ws: int, x: np.ndarray,
                    n: int, k: int, threads: int = 256,
                    upload: bool = True) -> int:
        """同 `gemv`，但只做设备侧工作并返回 y 的设备指针（不下载）。

        `upload=False` 用于反复计时的场景：调用方先 `upload=True` 一次，
        之后每轮只跑「激活量化 + GEMV + sync」，把主机→设备传输排除在外。
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != k:
            raise ValueError(f"x 形状 {x.shape}，期望 [M,{k}]")
        m = x.shape[0]
        if m not in (1, 2, 3, 4):
            raise ValueError(f"W4 GEMV 只支持 M=1..4（当前 {m}），预填充请走 GEMM")
        nwarp = threads // 64
        px = self._buf("x", m * k * 4)
        py = self._buf("y", m * n * 4)
        if upload:
            self.rt.upload(px, x.reshape(-1))
        if kind == "w4a8":
            if k % GRP:
                raise ValueError(f"W4A8 要求 K 是 {GRP} 的整数倍（当前 {k}）")
            ae = self._buf("w4a8_ae", m * k // 2)
            ao = self._buf("w4a8_ao", m * k // 2)
            asc = self._buf("w4a8_asc", m * (k // GRP) * 4)
            asu = self._buf("w4a8_asu", m * (k // GRP) * 4)
            # 每行一次 1D 启动 + 指针偏移。这台驱动的 2D 网格里
            # quant_act 第 2 行起的 q/asum 输出不可靠（scale 对、q 错），
            # 拆成 M 次 1D 后逐字节与主机参考一致；M≤4，开销可忽略。
            for r in range(m):
                self.rt.launch("quant_act", k // GRP, 32,
                               [px + r * k * 4, ae + r * (k // 2), ao + r * (k // 2),
                                asc + r * (k // GRP) * 4,
                                asu + r * (k // GRP) * 4, 1, k])
            rpp = 2 if m >= 2 else 1
            grid = (n + nwarp * rpp - 1) // (nwarp * rpp)
            self.rt.launch(W4A8_KERNEL[m], grid, threads,
                           [wq, ws, ae, ao, asc, asu, py, n, k])
        elif kind == "w4a4":
            if k % SG:
                raise ValueError(f"W4A4 要求 K 是 {SG} 的整数倍（当前 {k}）")
            aq = self._buf("w4a4_aq", m * k // 2)
            asc = self._buf("w4a4_asc", m * (k // SG) * 4)
            for r in range(m):
                self.rt.launch("quant_act4", k // SG, 32,
                               [px + r * k * 4, aq + r * (k // 2),
                                asc + r * (k // SG) * 4, 1, k])
            grid = (n + nwarp - 1) // nwarp
            self.rt.launch(W4A4_KERNEL[m], grid, threads,
                           [wq, ws, aq, asc, py, n, k])
        else:
            raise ValueError(f"未知 W4 通路 {kind!r}（用 w4a8 / w4a4）")
        self.rt.sync()
        return py

    def gemv(self, kind: str, wq: int, ws: int, x: np.ndarray,
             n: int, k: int, threads: int = 256) -> np.ndarray:
        """kind ∈ {"w4a8","w4a4"}；x [M,K] f32，M=1..4；返回 y [M,N] f32。"""
        x = np.ascontiguousarray(x, dtype=np.float32)
        py = self.gemv_device(kind, wq, ws, x, n, k, threads)
        return self.rt.download(py, x.shape[0] * n, np.float32).reshape(x.shape[0], n)

    def gemm_device(self, wq: int, wsc: int, x: np.ndarray, n: int, k: int,
                    threads: int = 256, upload: bool = True,
                    quantize: bool = True) -> int:
        """预填充最强算力路径：int4×int4 GEMM（`v_dot8_i32_i4`）。

        `wq,wsc` 来自 `RT4File.upload_weight_gemm`；`x` [M,K] f32，M 必须是
        BM=128 的整数倍（模型侧把 M 对齐到 128 再调）。返回 y 的设备指针。
        为了绕开运行时 2D grid 的坑，走 `gemm_w4a4_flat`：`bm/bn` 由扁平
        `blockIdx.x` 用主机传入的 nbn+魔法数还原，一次 1D 启动覆盖全部
        `(N/64)×(M/128)` 个块；激活量化也用一次 1D 的 `quant_rows_k`。
        `nbn == 1` 时魔法除法退化，回退成按 128 行分块启动（原 2D 语义）。

        `quantize=False`（配合先跑一次 `quantize=True`）只计 GEMM 本体，
        用于把「激活量化」与「最强算力 GEMM」分开计时。
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != k:
            raise ValueError(f"x 形状 {x.shape}，期望 [M,{k}]")
        m = x.shape[0]
        if m % self.BM:
            raise ValueError(f"GEMM 要求 M 是 {self.BM} 的整数倍（当前 {m}）")
        if n % self.BN:
            raise ValueError(f"GEMM 要求 N 是 {self.BN} 的整数倍（当前 {n}）")
        if k % self.QG:
            raise ValueError(f"GEMM 要求 K 是 {self.QG} 的整数倍（当前 {k}）")
        px = self._buf("gx", m * k * 4)
        paq = self._buf("gaq", m * k // 2)
        pasc = self._buf("gasc", (k // self.QG) * m * 4)
        pc = self._buf("gc", m * n * 4)
        if upload:
            self.rt.upload(px, x.reshape(-1))
        nq = k // self.QG                       # quant_rows_k 的 workgroup 大小
        nbn = n // self.BN
        if nbn >= 2:
            if quantize:
                # 多线程/向量化量化：一个 warp 一组，访存合并；与 quant_rows_k 逐位一致
                self.rt.launch("quant_rows_fast_k", m, 256,
                               [paq, pasc, px, k, self.QG, m, k, 0, nq])
            grid = nbn * (m // self.BM)
            self.rt.launch("gemm_w4a4_flat", grid, threads,
                           [paq, wq, pasc, wsc, pc, m, n, k,
                            nbn, _div_magic(nbn, grid)])
            self.rt.sync()
            return pc
        # nbn == 1：回退到按 128 行拆的 1D 启动
        for t in range(m // self.BM):
            aq_t = paq + t * self.BM * (k // 8) * 4
            sc_t = pasc + t * (k // self.QG) * self.BM * 4
            x_t = px + t * self.BM * k * 4
            c_t = pc + t * self.BM * n * 4
            if quantize:
                self.rt.launch("quant_rows_fast_k", self.BM, 256,
                               [aq_t, sc_t, x_t, k, self.QG, self.BM, k, 0, nq])
            self.rt.launch("gemm_w4a4", n // self.BN, threads,
                           [aq_t, wq, sc_t, wsc, c_t, self.BM, n, k])
        self.rt.sync()
        return pc

    def gemm(self, wq: int, wsc: int, x: np.ndarray, n: int, k: int,
             threads: int = 256) -> np.ndarray:
        """预填充 GEMM，返回 y [M,N] f32。"""
        x = np.ascontiguousarray(x, dtype=np.float32)
        pc = self.gemm_device(wq, wsc, x, n, k, threads)
        return self.rt.download(pc, x.shape[0] * n, np.float32).reshape(x.shape[0], n)
