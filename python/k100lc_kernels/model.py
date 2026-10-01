"""K100LC 模型级运行时：把内核包成可组合的推理算子。

设计目标：

* **权重只上传一次**：线性层构造时把权重放进显存，之后逐 token 复用；
* **工作缓冲一次分配**：`Workspace` 按 key 复用中间张量，避免每层 alloc/free；
* **一次 forward 只 sync 一次**：引擎是单队列 in-order，层内多个 kernel 可以
  连续入队，最后统一 `rt.sync()`；
* **多种权重格式统一接口**：f32、compressed-tensors INT4（W4A16）、GGUF
  `*_dot_k`、RT4 W4A8 / W4A4，都是 `forward_device(x_dev, rows)` / `forward(x)`。

典型用法（f32 两层 MLP）：

```python
from k100lc_kernels import Runtime
from k100lc_kernels.model import F32Linear, MLP, RMSNorm, Workspace

rt = Runtime()
ws = Workspace(rt)
norm = RMSNorm(rt, dim, w, ws, "norm0")
mlp = MLP(rt, F32Linear(rt, ffn, dim, wg, ws, "gate"),
          F32Linear(rt, ffn, dim, wu, ws, "up"),
          F32Linear(rt, dim, ffn, wd, ws, "down"),
          ws, "mlp0")
h = norm.forward(x)          # numpy in/out
y = mlp.forward(h)
```

注意：`Workspace` 复用缓冲区，同一个 `Workspace` 里的不同层请给唯一 `tag`；
跨 forward 的中间张量（例如要留给下一步的 KV cache）不要放在会被覆盖的
临时 key 上。
"""
from __future__ import annotations

import numpy as np

from .runtime import Runtime

__all__ = [
    "cdiv", "div_magic", "Workspace",
    "QUANT_SPECS", "DotLinear", "F32Linear", "RT4Linear", "Int4Linear",
    "RMSNorm", "SwiGLU", "MLP", "MoECombine", "MoEExperts", "KVCache", "Sampler",
    "Attention",
    "run_sequence",
]


def cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def div_magic(d: int, max_i: int) -> int:
    """返回 M，使 `(i*M)>>32 == i//d` 对 0 <= i < max_i 成立。

    与 `tools/gen_gemv_qdot.py::div_magic` 同一算法，避免运行时依赖 tools/。
    """
    d = int(d)
    max_i = int(max_i)
    if d <= 0:
        raise ValueError("div_magic: d 必须为正")
    if max_i <= 0:
        return 0
    if d == 1:
        if max_i <= 1:
            return 0
        raise ValueError("div_magic: d=1 只在下标恒为 0 时可用")
    m = (1 << 32) // d + 1
    if m >= (1 << 32):
        raise ValueError(f"div_magic: d={d} 太大")
    i = np.arange(max_i, dtype=np.uint64)
    if not np.array_equal((i * np.uint64(m)) >> np.uint64(32),
                          i // np.uint64(d)):
        raise ValueError(f"div_magic: d={d} max_i={max_i} 无法精确表示")
    return int(m)


class Workspace:
    """按 key 复用的设备缓冲集合。

    `buffer(key, nbytes)` 在容量不足时重新分配并释放旧块；同一个 key 的调用
    返回同一块显存，适合逐 token 重复使用的中间张量。
    """

    def __init__(self, rt: Runtime):
        self.rt = rt
        self._bufs: dict[str, tuple[int, int]] = {}

    def buffer(self, key: str, nbytes: int) -> int:
        cur = self._bufs.get(key)
        if cur is not None and cur[1] >= nbytes:
            return cur[0]
        if cur is not None:
            self.rt.free(cur[0])
        p = self.rt.alloc(int(nbytes))
        self._bufs[key] = (p, int(nbytes))
        return p

    def upload(self, key: str, data) -> int:
        n = len(data) if isinstance(data, (bytes, bytearray)) else data.nbytes
        p = self.buffer(key, n)
        self.rt.upload(p, data)
        return p

    def size(self, key: str) -> int:
        return self._bufs.get(key, (0, 0))[1]

    def free(self) -> None:
        for p, _ in self._bufs.values():
            self.rt.free(p)
        self._bufs.clear()

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc) -> None:
        self.free()


def run_sequence(rt: Runtime, ops, x_dev: int, sync: bool = True) -> int:
    """把一串「设备指针 → 设备指针」的算子连续入队，最后只 sync 一次。

    这是本包推荐的整段前向方式：GPU 每层约几十微秒到几百微秒，而
    `sync` 会把 CPU 和 GPU 串起来；先全部入队再等一次，实测每层吞吐
    （W4A8 17408×5120）从 99.0 us 降到 81.4 us。
    """
    for op in ops:
        x_dev = op(x_dev)
    if sync:
        rt.sync()
    return x_dev


# 量化权重格式 → GPU 内核 / 块大小 / 需要的表指针
QUANT_SPECS: dict[str, dict] = {
    "int4":   dict(kernel="int4_dot_k",   elems=128, block_bytes=64,  tables=("scale",)),
    "q2_0":   dict(kernel="q2_0_dot_k",   elems=64,  block_bytes=18,  tables=()),
    "iq4nl":  dict(kernel="iq4nl_dot_k",  elems=32,  block_bytes=18,  tables=()),
    "iq4xs":  dict(kernel="iq4xs_dot_k",  elems=256, block_bytes=136, tables=()),
    "q4k":    dict(kernel="q4k_dot_k",    elems=256, block_bytes=144, tables=()),
    "q5k":    dict(kernel="q5k_dot_k",    elems=256, block_bytes=176, tables=()),
    "q6k":    dict(kernel="q6k_dot_k",    elems=256, block_bytes=210, tables=()),
    "q8_0":   dict(kernel="q8_0_dot_k",   elems=32,  block_bytes=34,  tables=()),
    "q4_0":   dict(kernel="q4_0_dot_k",   elems=32,  block_bytes=18,  tables=()),
    "iq2s":   dict(kernel="iq2s_dot_k",   elems=256, block_bytes=82,  tables=("grid",)),
    "iq3s":   dict(kernel="iq3s_dot_k",   elems=256, block_bytes=110, tables=("grid",)),
    "iq3xxs": dict(kernel="iq3xxs_dot_k", elems=256, block_bytes=98,
                   tables=("grid", "ksigns")),
}


class DotLinear:
    """量化权重的「原生解码 + 点积」线性层（`*_dot_k` + `reduce_blocks_k`）。

    支持 compressed-tensors INT4 与 GGUF 的 11 类编码；权重以原始字节上传，
    不做反量化、不重排。`tables` 按 `QUANT_SPECS[kind]["tables"]` 的顺序传：

    * `int4`：1 张表 = BF16 尺度数组（`weight_scale` 的原始字节）；
    * `iq2s` / `iq3s`：1 张表 = 网格表；
    * `iq3xxs`：2 张表 = 网格表 + 符号表。
    """

    def __init__(self, rt: Runtime, n: int, k: int, kind: str,
                 weight_bytes, tables=(), ws: Workspace | None = None,
                 tag: str | None = None):
        if kind not in QUANT_SPECS:
            raise ValueError(f"未知量化类型 {kind!r}，可选：{sorted(QUANT_SPECS)}")
        spec = QUANT_SPECS[kind]
        if k % spec["elems"]:
            raise ValueError(f"{kind}: K={k} 不是块大小 {spec['elems']} 的整数倍")
        self.rt = rt
        self.n, self.k = int(n), int(k)
        self.kind = kind
        self.spec = spec
        self.nbpr = self.k // spec["elems"]
        self.tag = tag or f"dot_{kind}_{id(self):x}"
        self._own_ws = ws is None
        self.ws = ws or Workspace(rt)
        self.w = self.ws.buffer(self.tag + ".w", len(weight_bytes))
        rt.upload(self.w, weight_bytes)
        self.tables: list[int] = []
        for i, t in enumerate(tables):
            p = self.ws.buffer(f"{self.tag}.t{i}", len(t))
            rt.upload(p, t)
            self.tables.append(p)
        if len(self.tables) != len(spec["tables"]):
            raise ValueError(f"{kind} 需要 {len(spec['tables'])} 张表，"
                             f"收到 {len(self.tables)} 张")
        # 对 dense 层：nblocks = 输出行数 N × 每行块数 nbpr；权重行才是
        # `*_dot_k` 里的「行」，批大小 M 通过 M 次 1D 启动实现。
        self.nblocks = self.n * self.nbpr
        self.partial = self.ws.buffer(f"{self.tag}.partial", self.nblocks * 4)
        self.ids = self.ws.buffer(f"{self.tag}.ids", self.n * 4)
        rt.upload(self.ids, np.arange(self.n, dtype=np.uint32))
        self.m_nbpr = div_magic(self.nbpr, self.nblocks)
        self.m_rows = div_magic(self.n, self.n)
        self._out_cache: dict[int, int] = {}

    def _out(self, rows: int) -> int:
        out = self._out_cache.get(rows)
        if out is None:
            out = self.ws.buffer(f"{self.tag}.out", rows * self.n * 4)
            self._out_cache[rows] = out
        return out

    def forward_device(self, x_dev: int, rows: int, sync: bool = False,
                       out_dev: int | None = None) -> int:
        """在设备上算 `y = W·x`，返回 y 的设备指针（默认不 sync）。"""
        spec = self.spec
        out = int(out_dev) if out_dev is not None else self._out(rows)
        base = [self.w, int(x_dev), self.partial, self.nblocks, 64, self.nbpr,
                self.m_nbpr, self.n, self.m_rows]
        base += self.tables
        base += [self.ids, self.nbpr * spec["block_bytes"], self.n, self.m_rows]
        for r in range(rows):
            argv = list(base)
            argv[1] = int(x_dev) + r * self.k * 4
            self.rt.launch(spec["kernel"], cdiv(self.nblocks, 64), 64, argv)
            self.rt.launch("reduce_blocks_k", cdiv(self.n, 64), 64,
                           [self.partial, out + r * self.n * 4, self.n, self.nbpr])
        if sync:
            self.rt.sync()
        return out

    def forward_experts(self, x_dev: int, rows: int, ids_dev: int,
                        stride_bytes: int, rows_per_exp: int, rows_per_w: int,
                        out_dev: int | None = None, sync: bool = False) -> int:
        """MoE 路径：权重专家组由 `ids_dev` 选，x 侧每组 `rows_per_exp` 行。

        `out_dev` 为空时用本层自己的行优先输出缓冲（`[rows, N]`）。
        """
        spec = self.spec
        out = int(out_dev) if out_dev is not None else self._out(1)
        argv = [self.w, int(x_dev), self.partial, self.nblocks, 64, self.nbpr,
                self.m_nbpr, rows_per_exp, div_magic(rows_per_exp, self.n)]
        argv += self.tables
        argv += [int(ids_dev), stride_bytes, rows_per_w,
                 div_magic(rows_per_w, self.n)]
        self.rt.launch(spec["kernel"], cdiv(self.nblocks, 64), 64, argv)
        self.rt.launch("reduce_blocks_k", cdiv(self.n, 64), 64,
                       [self.partial, out, self.n, self.nbpr])
        if sync:
            self.rt.sync()
        return out

    def forward(self, x: np.ndarray, sync: bool = True):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        rows = x.shape[0]
        px = self.ws.buffer(self.tag + ".x", x.nbytes)
        self.rt.upload(px, x.reshape(-1))
        out = self.forward_device(px, rows, sync=sync)
        if not sync:
            return out
        return self.rt.download(out, rows * self.n, np.float32).reshape(rows, self.n)


class F32Linear:
    """f32 权重线性层；解码 M=1..4 时每行一次 `gemv_f32_warp_k`。"""

    def __init__(self, rt: Runtime, n: int, k: int, weight,
                 ws: Workspace | None = None, tag: str | None = None):
        self.rt = rt
        self.n, self.k = int(n), int(k)
        self.tag = tag or f"f32_{id(self):x}"
        self._own_ws = ws is None
        self.ws = ws or Workspace(rt)
        self.w = self.ws.buffer(self.tag + ".w", self.n * self.k * 4)
        rt.upload(self.w, np.ascontiguousarray(weight, dtype=np.float32))
        self._rows_cache: dict[int, dict] = {}

    def _bind(self, rows: int) -> dict:
        b = self._rows_cache.get(rows)
        if b is None:
            b = dict(out=self.ws.buffer(self.tag + ".out", rows * self.n * 4))
            self._rows_cache[rows] = b
        return b

    def forward_device(self, x_dev: int, rows: int, sync: bool = False,
                       out_dev: int | None = None) -> int:
        b = self._bind(rows)
        out = int(out_dev) if out_dev is not None else b["out"]
        for r in range(rows):
            self.rt.launch("gemv_f32_warp_k", self.n, 64,
                           [self.w, int(x_dev) + r * self.k * 4,
                            out + r * self.n * 4, self.n, self.k, 64])
        if sync:
            self.rt.sync()
        return out

    def forward(self, x: np.ndarray, sync: bool = True):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        rows = x.shape[0]
        px = self.ws.buffer(self.tag + ".x", x.nbytes)
        self.rt.upload(px, x.reshape(-1))
        out = self.forward_device(px, rows, sync=sync)
        if not sync:
            return out
        return self.rt.download(out, rows * self.n, np.float32).reshape(rows, self.n)


class RT4Linear:
    """RT4 INT4 线性层（W4A8 / W4A4），支持 M=1..4 解码与 128 对齐的预填充。"""

    def __init__(self, rt: Runtime, n: int, k: int, wq: int, ws_scale: int,
                 path: str = "w4a8", runner=None, tag: str | None = None):
        from .rt4 import W4Runner
        self.rt = rt
        self.n, self.k = int(n), int(k)
        self.wq, self.ws_scale = int(wq), int(ws_scale)
        self.path = path
        self.runner = runner or W4Runner(rt)
        self.tag = tag or f"rt4_{id(self):x}"

    def forward_device(self, x_dev: int, rows: int, sync: bool = False,
                       out_dev: int | None = None) -> int:
        return self.runner.gemv_device(self.path, self.wq, self.ws_scale, None,
                                       self.n, self.k, x_dev=x_dev, m=rows,
                                       y_dev=out_dev,
                                       sync=sync)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        return self.runner.gemv(self.path, self.wq, self.ws_scale, x,
                                self.n, self.k)


class Int4Linear:
    """compressed-tensors INT4 线性层，可选三条通路。

    ================  ==========================  ===========================
    path              内核                          实测（17408×5120, M=1）
    ================  ==========================  ===========================
    ``"w4a16"``       ``int4_dot_k``             0.269 ms / 171 GB/s
    ``"w4a8"``        ``gemv_w4a8`` + int8 激活  0.103 ms / 446 GB/s（2.6×）
    ``"w4a4"``        ``gemv_w4a4`` + int4 激活  0.089 ms / 515 GB/s（3.0×）
    ================  ==========================  ===========================

    ``w4a16`` 保持 checkpoint 原字节、精度最高；``w4a8`` / ``w4a4`` 在构造时把
    offset-binary 码转成 RT4 的两补码布局、BF16 尺度转 f16，之后逐 token 复用。
    预填充（M 是 128 的倍数）自动走 W4A4 GEMM（``gemm_w4a4_flat``），
    避免逐行 GEMV。
    """

    def __init__(self, rt: Runtime, n: int, k: int, packed: bytes,
                 scales_bf16: bytes, path: str = "w4a8",
                 ws: Workspace | None = None, tag: str | None = None,
                 runner=None, prefill: str = "gemm"):
        from .quant import ct_int4_to_rt4, int4_scale_group_first_f32
        if path not in ("w4a16", "w4a8", "w4a4"):
            raise ValueError('path 必须是 "w4a16" / "w4a8" / "w4a4"')
        if prefill not in ("gemm", "gemv"):
            raise ValueError('prefill 必须是 "gemm" / "gemv"')
        self.rt = rt
        self.n, self.k, self.path = int(n), int(k), path
        self.prefill = prefill
        self.tag = tag or f"int4_{path}_{id(self):x}"
        self._own_ws = ws is None
        self.ws = ws or Workspace(rt)
        self._dot = None
        self._runner = None
        if path == "w4a16":
            self._dot = DotLinear(rt, n, k, "int4", packed, (scales_bf16,),
                                  self.ws, self.tag)
            return
        from .rt4 import W4Runner
        q, sf = ct_int4_to_rt4(packed, scales_bf16)
        self.wq = self.ws.buffer(self.tag + ".wq", len(q))
        rt.upload(self.wq, q)
        self.ws_f16 = self.ws.buffer(self.tag + ".ws", len(sf))
        rt.upload(self.ws_f16, sf)
        gm = int4_scale_group_first_f32(scales_bf16, self.n, self.k)
        self.wsc_gm = self.ws.buffer(self.tag + ".wsc", gm.nbytes)
        rt.upload(self.wsc_gm, gm.reshape(-1))
        self._runner = runner or W4Runner(rt)
        self._y_cache: dict[int, int] = {}

    def _y(self, rows: int) -> int:
        y = self._y_cache.get(rows)
        if y is None:
            y = self.ws.buffer(self.tag + ".y", rows * self.n * 4)
            self._y_cache[rows] = y
        return y

    def forward_device(self, x_dev: int, rows: int, sync: bool = False,
                       out_dev: int | None = None) -> int:
        if self._dot is not None:
            return self._dot.forward_device(x_dev, rows, sync=sync,
                                            out_dev=out_dev)
        y = int(out_dev) if out_dev is not None else self._y(rows)
        if rows <= 4:
            return self._runner.gemv_device(
                self.path, self.wq, self.ws_f16, None, self.n, self.k,
                x_dev=x_dev, y_dev=y, m=rows, sync=sync)
        if rows % 128 == 0 and self.n % 64 == 0 and self.prefill == "gemm":
            return self._runner.gemm_device(
                self.wq, self.wsc_gm, None, self.n, self.k,
                x_dev=x_dev, c_dev=y, m=rows, sync=sync)
        # 其它批大小：按 4 行一组走 GEMV（保持正确性）
        for r0 in range(0, rows, 4):
            m = min(4, rows - r0)
            self._runner.gemv_device(
                self.path, self.wq, self.ws_f16, None, self.n, self.k,
                x_dev=x_dev + r0 * self.k * 4, y_dev=y + r0 * self.n * 4,
                m=m, sync=False)
        if sync:
            self.rt.sync()
        return y

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        rows = x.shape[0]
        px = self.ws.buffer(self.tag + ".x", x.nbytes)
        self.rt.upload(px, x.reshape(-1))
        out = self.forward_device(px, rows, sync=True)
        return self.rt.download(out, rows * self.n, np.float32) \
            .reshape(rows, self.n)


class RMSNorm:
    """RMSNorm：`y = x / sqrt(mean(x^2)+eps) * w`（`rmsnorm_k`，flag=0）。"""

    def __init__(self, rt: Runtime, dim: int, weight, eps: float = 1e-6,
                 ws: Workspace | None = None, tag: str | None = None):
        if dim % 64:
            raise ValueError(f"RMSNorm dim={dim} 必须是 64 的倍数")
        self.rt = rt
        self.dim, self.eps = int(dim), float(eps)
        self.tag = tag or f"rms_{id(self):x}"
        self._own_ws = ws is None
        self.ws = ws or Workspace(rt)
        self.w = self.ws.buffer(self.tag + ".w", self.dim * 4)
        rt.upload(self.w, np.ascontiguousarray(weight, dtype=np.float32))

    def forward_device(self, x_dev: int, rows: int, sync: bool = False) -> int:
        out = self.ws.buffer(self.tag + ".out", rows * self.dim * 4)
        self.rt.launch("rmsnorm_k", rows, 64,
                       [out, int(x_dev), self.w, self.dim, self.eps, 0])
        if sync:
            self.rt.sync()
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        rows = x.shape[0]
        px = self.ws.buffer(self.tag + ".x", x.nbytes)
        self.rt.upload(px, x.reshape(-1))
        out = self.forward_device(px, rows, sync=True)
        return self.rt.download(out, rows * self.dim, np.float32).reshape(rows, self.dim)


class SwiGLU:
    """`y = silu(gate) * up`（`silu_mul_k`）。"""

    def __init__(self, rt: Runtime, ws: Workspace | None = None, tag: str = "swiglu"):
        self.rt = rt
        self.tag = tag
        self._own_ws = ws is None
        self.ws = ws or Workspace(rt)

    def forward_device(self, gate_dev: int, up_dev: int, n: int,
                       out_dev: int | None = None, sync: bool = False) -> int:
        out = int(out_dev) if out_dev is not None else \
            self.ws.buffer(self.tag + ".out", n * 4)
        self.rt.launch("silu_mul_k", cdiv(n, 64), 64,
                       [out, int(gate_dev), int(up_dev), n])
        if sync:
            self.rt.sync()
        return out


class MLP:
    """门控 MLP：`down(act(gate(x)) * up(x))`，全部 kernel 连续入队。"""

    def __init__(self, rt: Runtime, gate, up, down,
                 ws: Workspace | None = None, tag: str = "mlp",
                 act: str = "silu"):
        self.rt = rt
        if act != "silu":
            raise ValueError("当前 model.MLP 只封装 silu 门控")
        self.tag = tag
        self.gate, self.up, self.down = gate, up, down
        self.act = SwiGLU(rt, ws, tag + ".act")

    def forward_device(self, x_dev: int, rows: int, sync: bool = False) -> int:
        h1 = self.gate.forward_device(x_dev, rows, sync=False)
        h2 = self.up.forward_device(x_dev, rows, sync=False)
        act = self.act.forward_device(h1, h2, rows * self.gate.n, sync=False)
        out = self.down.forward_device(act, rows, sync=False)
        if sync:
            self.rt.sync()
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        rows = x.shape[0]
        ws = self.gate.ws
        px = ws.buffer(self.tag + ".in.x", x.nbytes)
        self.rt.upload(px, x.reshape(-1))
        out = self.forward_device(px, rows, sync=True)
        return self.rt.download(out, rows * self.down.n, np.float32) \
            .reshape(rows, self.down.n)


class MoECombine:
    """`y[r,:] = sum_e weights[r,e] * exp_out[e,r,:]`（`moe_combine_k`）。"""

    def __init__(self, rt: Runtime, rows: int, dim: int, n_exp: int,
                 ws: Workspace | None = None, tag: str = "moe"):
        self.rt = rt
        self.rows, self.dim, self.n_exp = int(rows), int(dim), int(n_exp)
        self.tag = tag
        self._own_ws = ws is None
        self.ws = ws or Workspace(rt)
        self.exp_out = self.ws.buffer(tag + ".exp", rows * n_exp * dim * 4)
        self.weights = self.ws.buffer(tag + ".w", rows * n_exp * 4)
        self.out = self.ws.buffer(tag + ".out", rows * dim * 4)

    def slot(self, e: int) -> int:
        """第 e 个专家输出的设备指针（布局 `[n_exp, rows, dim]`）。"""
        return self.exp_out + e * self.rows * self.dim * 4

    def forward_device(self, weights_dev: int | None = None,
                       sync: bool = False) -> int:
        w = int(weights_dev) if weights_dev is not None else self.weights
        self.rt.launch("moe_combine_k", self.rows, 64,
                       [self.out, self.exp_out, w, self.n_exp,
                        self.rows, self.dim])
        if sync:
            self.rt.sync()
        return self.out

    def forward(self, exp_out: np.ndarray, weights: np.ndarray) -> np.ndarray:
        exp_out = np.ascontiguousarray(exp_out, dtype=np.float32)
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        self.rt.upload(self.exp_out, exp_out.reshape(-1))
        self.rt.upload(self.weights, weights.reshape(-1))
        out = self.forward_device(sync=True)
        return self.rt.download(out, self.rows * self.dim, np.float32) \
            .reshape(self.rows, self.dim)


class MoEExperts:
    """token→expert 分桶的 MoE 前向（比「所有专家 × 全部行」省 topk 倍算力）。

    流程：

    1. 主机侧按 router 结果把 token 分成每个专家的桶（`route()`）；
    2. `gather_rows_k` 把 x 的对应行 gather 到连续桶里；
    3. 每个非空专家只在自己的桶上跑线性层，输出写进 `exp_bucket`；
    4. `moe_combine_gather_k` 按 `pos[r,e]` 把专家结果加权合并。

    构造时给定 `rows`（token 数）与 `topk`，之后所有缓冲都复用。
    """

    def __init__(self, rt: Runtime, experts, rows: int, dim: int, out_dim: int,
                 topk: int, ws: Workspace | None = None, tag: str = "moe_experts"):
        if dim % 4:
            raise ValueError(f"dim={dim} 必须是 4 的倍数（gather_rows_k 用 dwordx4）")
        self.rt = rt
        self.experts = list(experts)
        self.n_exp = len(self.experts)
        self.rows, self.dim, self.out_dim, self.topk = \
            int(rows), int(dim), int(out_dim), int(topk)
        if topk > self.n_exp:
            raise ValueError(f"topk={topk} 大于专家数 {self.n_exp}")
        self.tag = tag
        self._own_ws = ws is None
        self.ws = ws or Workspace(rt)
        cap = rows * topk
        self.x_bucket = self.ws.buffer(tag + ".xb", cap * dim * 4)
        self.exp_bucket = self.ws.buffer(tag + ".eb", cap * out_dim * 4)
        self.idx = self.ws.buffer(tag + ".idx", cap * 4)
        self.base = self.ws.buffer(tag + ".base", cap * 4)
        self.weights = self.ws.buffer(tag + ".w", cap * 4)
        self.y = self.ws.buffer(tag + ".y", rows * out_dim * 4)

    def route(self, ids, weights):
        """返回 `(bucket_rows, pos, weights, offsets, valid)`（主机 NumPy 数组）。"""
        ids = np.asarray(ids)
        weights = np.asarray(weights, dtype=np.float32)
        rows, topk = ids.shape
        if rows != self.rows or topk != self.topk:
            raise ValueError(f"路由形状 {ids.shape} 与构造时的 "
                             f"({self.rows},{self.topk}) 不一致")
        valid = (ids >= 0) & (ids < self.n_exp) & (weights != 0.0)
        flat_idx = np.flatnonzero(valid.reshape(-1))
        ex = ids.reshape(-1)[flat_idx].astype(np.int64)
        row_of = (flat_idx // topk).astype(np.uint32)
        counts = np.bincount(ex, minlength=self.n_exp).astype(np.int64)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
        order = np.argsort(ex, kind="stable")       # 稳定排序 → 桶内顺序 = pos
        bucket = row_of[order]
        rank = np.empty(len(flat_idx), dtype=np.int64)
        rank[order] = (np.arange(len(flat_idx))
                       - np.repeat(offsets[:-1].astype(np.int64), counts))
        pos_flat = np.zeros(rows * topk, dtype=np.uint32)
        pos_flat[flat_idx] = rank.astype(np.uint32)
        return bucket, pos_flat.reshape(rows, topk), weights, offsets, valid

    def forward_device(self, x_dev: int, rows: int, ids, weights,
                       sync: bool = False) -> int:
        bucket, pos, w, offsets, valid = self.route(ids, weights)
        total = int(bucket.size)
        if total == 0:
            self.rt.upload(self.y, np.zeros(rows * self.out_dim, dtype=np.float32))
            if sync:
                self.rt.sync()
            return self.y
        self.rt.upload(self.idx, bucket)
        self.rt.launch("gather_rows_k", total, 64,
                       [self.x_bucket, int(x_dev), self.idx, total, self.dim])
        # base[r,e] = 该 (r,e) 在 exp_bucket 里的行号 = 专家桶偏移 + 桶内位置
        ids_arr = np.asarray(ids)
        base = np.zeros((rows, self.topk), dtype=np.uint32)
        base[valid] = offsets[ids_arr[valid]] + pos[valid]
        self.rt.upload(self.base, base.reshape(-1))
        self.rt.upload(self.weights, w.reshape(-1))
        for e, expert in enumerate(self.experts):
            n = int(offsets[e + 1] - offsets[e])
            if n == 0:
                continue
            expert.forward_device(
                self.x_bucket + int(offsets[e]) * self.dim * 4, n,
                out_dev=self.exp_bucket + int(offsets[e]) * self.out_dim * 4,
                sync=False)
        self.rt.launch("moe_combine_gather_k", rows, 64,
                       [self.y, self.exp_bucket, self.weights, self.base,
                        self.topk, rows, self.out_dim])
        if sync:
            self.rt.sync()
        return self.y

    def forward(self, x: np.ndarray, ids, weights) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        rows = x.shape[0]
        px = self.ws.buffer(self.tag + ".x", x.nbytes)
        self.rt.upload(px, x.reshape(-1))
        out = self.forward_device(px, rows, ids, weights, sync=True)
        return self.rt.download(out, rows * self.out_dim, np.float32) \
            .reshape(rows, self.out_dim)


class Attention:
    """解码（M=1）注意力：`scores = K·q` → softmax → `P·V`。

    全部用**已对账**的内核拼出来，不依赖包里语义未验证的 `fa_decode_*`。
    关键是「矩阵乘向量」这一步全部交给 `gemv_f32_warp_k`（warp-per-row、写死在
    汇编里、实测 430 GB/s），而不是自己写标量循环：

    ==============  ==========================================================
    步骤            表达式 / 内核
    ==============  ==========================================================
    `scores=K·q`    `scores[n] = Σ_d K[n,d]·q[d]` → `gemv_f32_warp_k`（**K 当权重矩阵**）
    尾部掩码        `fill_k`（padding 段填 -1e30）
    缩放            `scale_mul_k`（1/√dim）
    softmax         `softmax_k`
    `out=P·V`       `out[d] = Σ_j Vt[d,j]·P[j]` → `gemv_f32_warp_k`（**Vt 当权重矩阵**）
    ==============  ==========================================================

    布局：`K [max_len, dim]` 行主序；`Vt [dim, max_len]`（**转置**，这样
    `out = Vt·P` 就是标准 GEMV）；`max_len` 必须是 64 的倍数（softmax 要求）。
    K / Vt 里没写到的格子保持 0，尾部用 -1e30 掩码掉。

    ```python
    attn = Attention(rt, dim=128, max_len=4096)
    attn.append(k, v)                 # k [rows, dim]、v [rows, dim]
    out = attn.forward(q)             # q [dim] → [dim]
    ```
    """

    def __init__(self, rt: Runtime, dim: int, max_len: int,
                 ws: Workspace | None = None, tag: str = "attn",
                 n_split: int = 128):
        if dim % 64:
            raise ValueError(f"Attention dim={dim} 必须是 64 的倍数")
        if max_len % 64:
            raise ValueError(f"Attention max_len={max_len} 必须是 64 的倍数")
        if max_len % 256:
            raise ValueError(f"Attention max_len={max_len} 必须是 256 的倍数"
                             "（softmax_vec_k 一个 warp 一轮覆盖 256 列）")
        if n_split <= 0 or (n_split & (n_split - 1)):
            raise ValueError("n_split 必须是 2 的幂")
        self.rt = rt
        self.dim, self.max_len = int(dim), int(max_len)
        self.n_split = int(n_split)
        self.tag = tag
        self._own_ws = ws is None
        self.ws = ws or Workspace(rt)
        # K / Vt 必须从 0 开始（没写到的格子要参与 softmax / GEMV，得是 0）
        self.k = self.ws.buffer(tag + ".k", max_len * dim * 4)
        rt.memset(self.k, 0, max_len * dim * 4)
        self.vt = self.ws.buffer(tag + ".vt", dim * max_len * 4)
        rt.memset(self.vt, 0, dim * max_len * 4)
        self.scores = self.ws.buffer(tag + ".scores", max_len * 4)
        self.probs = self.ws.buffer(tag + ".probs", max_len * 4)
        self.part = self.ws.buffer(tag + ".part", dim * n_split * 4)
        self.out = self.ws.buffer(tag + ".out", dim * 4)
        self.q = self.ws.buffer(tag + ".q", dim * 4)
        self.length = 0

    def append_device(self, k_dev: int, v_dev: int, rows: int) -> None:
        """追加 `rows` 个 token 的 K/V；V 转置后写进 `Vt`。"""
        if self.length + rows > self.max_len:
            raise ValueError(f"KV 溢出：{self.length}+{rows} > {self.max_len}")
        off = self.length
        self.rt.copy_dev(self.k + off * self.dim * 4, int(k_dev),
                         rows * self.dim * 4)
        v = self.rt.download(int(v_dev), rows * self.dim, np.float32) \
            .reshape(rows, self.dim)
        self._store_vt(v, off, rows)
        self.length += rows

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        k = np.ascontiguousarray(k, dtype=np.float32).reshape(-1, self.dim)
        v = np.ascontiguousarray(v, dtype=np.float32).reshape(-1, self.dim)
        rows = k.shape[0]
        if self.length + rows > self.max_len:
            raise ValueError(f"KV 溢出：{self.length}+{rows} > {self.max_len}")
        off = self.length
        self.rt.upload(self.k + off * self.dim * 4, k.reshape(-1))
        self._store_vt(v, off, rows)
        self.length += rows

    def _store_vt(self, v: np.ndarray, off: int, rows: int) -> None:
        buf = np.empty(rows, dtype=np.float32)
        for d in range(self.dim):
            np.copyto(buf, v[:, d])
            self.rt.upload(self.vt + (d * self.max_len + off) * 4, buf)

    def reset(self) -> None:
        self.length = 0

    def forward_device(self, q_dev: int, n_kv: int | None = None,
                       sync: bool = False) -> int:
        """在设备上算一次解码注意力，返回 `out[dim]` 的设备指针。"""
        n_kv = self.length if n_kv is None else int(n_kv)
        if n_kv <= 0:
            raise ValueError("n_kv 必须为正")
        # 只算「当前长度向上取整到 64」这么多列：缓存没填满时**不要**按
        # max_len 算（那会做几十倍无用功——softmax 是行扫描，代价正比于列数）。
        # 向上取整到 256：`softmax_vec_k` 一个 lane 处理连续 4 列、一个 warp
        # 一轮覆盖 256 个元素（访存完全合并），实测比 `softmax_k` 快 2.3 倍。
        pad = -(-n_kv // 256) * 256
        # 1) scores = K·q（K 当权重矩阵：行=位置，列=dim）
        self.rt.launch("gemv_f32_warp_k", pad, 64,
                       [self.k, int(q_dev), self.scores, pad, self.dim, 64])
        # 2) 尾部掩码 + 缩放
        if pad > n_kv:
            self.rt.launch("fill_k", -(-(pad - n_kv) // 64), 64,
                           [self.scores + n_kv * 4, -1e30, pad - n_kv])
        self.rt.launch("scale_mul_k", -(-pad // 64), 64,
                       [self.scores, 1.0 / float(self.dim) ** 0.5, pad])
        # 3) softmax
        self.rt.launch("softmax_vec_k", 1, 64,
                       [self.probs, self.scores, 1, pad, 64])
        # 4) out = Vt·P。两条路：
        #    * pad == max_len（缓存填满）→ `gemv_f32_warp_k` 把 Vt 当权重矩阵；
        #    * 否则用 `attn_pv_part`（支持自定义行距，只扫 pad 列）+ 归约——
        #      `gemv_f32_warp_k` 的行距必须等于列数，缓存没填满时只能用前者。
        if pad == self.max_len:
            self.rt.launch("gemv_f32_warp_k", self.dim, 64,
                           [self.vt, self.probs, self.out, self.dim, pad, 64])
        else:
            # 块数必须整除 pad，否则会有列没被扫到（实测 pad=320、nb=128 时
            # 只覆盖 256 列 → 结果错误）。取「不超过 n_split 且能整除 pad 的
            # 最大 2 的幂」；pad 是 64 的倍数，所以至少能取到 64。
            nb = 1
            while nb * 2 <= self.n_split and pad % (nb * 2) == 0:
                nb *= 2
            sh = nb.bit_length() - 1
            self.rt.launch("attn_pv_part", -(-self.dim * nb // 64), 64,
                           [self.part, self.probs, self.vt, pad, self.max_len,
                            sh, pad // nb])
            self.rt.launch("reduce_blocks_k", -(-self.dim // 64), 64,
                           [self.part, self.out, self.dim, nb])
        if sync:
            self.rt.sync()
        return self.out

    def forward(self, q: np.ndarray, n_kv: int | None = None) -> np.ndarray:
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(-1)
        self.rt.upload(self.q, q)
        out = self.forward_device(self.q, n_kv, sync=True)
        return self.rt.download(out, self.dim, np.float32)


class KVCache:
    """f32 KV cache 追加：把新 token 的 K/V 拷进 `[max_len, dim]` 的槽。

    这是主机编排的追加（每次一次 `copy_dev`），适合小批解码；内核包里另有
    `kv_append_k_k` / `kv_append_v_k` 的打包 KV 版本，可用于 int8/int4 KV。
    """

    def __init__(self, rt: Runtime, dim: int, max_len: int,
                 ws: Workspace | None = None, tag: str = "kv"):
        self.rt = rt
        self.dim, self.max_len = int(dim), int(max_len)
        self.tag = tag
        self._own_ws = ws is None
        self.ws = ws or Workspace(rt)
        self.k = self.ws.buffer(tag + ".k", max_len * dim * 4)
        self.v = self.ws.buffer(tag + ".v", max_len * dim * 4)
        self.length = 0

    def append_device(self, k_dev: int, v_dev: int, rows: int,
                      sync: bool = False) -> None:
        if self.length + rows > self.max_len:
            raise ValueError(f"KV cache 溢出：{self.length}+{rows} > {self.max_len}")
        off = self.length * self.dim * 4
        self.rt.copy_dev(self.k + off, int(k_dev), rows * self.dim * 4)
        self.rt.copy_dev(self.v + off, int(v_dev), rows * self.dim * 4)
        self.length += rows
        if sync:
            self.rt.sync()

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        k = np.ascontiguousarray(k, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)
        rows = k.reshape(-1, self.dim).shape[0]
        off = self.length * self.dim * 4
        self.rt.upload(self.k + off, k.reshape(-1))
        self.rt.upload(self.v + off, v.reshape(-1))
        self.length += rows

    def reset(self) -> None:
        self.length = 0


class Sampler:
    """贪心 / top-k / top-p 采样；logits 很小，默认在主机侧做。"""

    def __init__(self, rt: Runtime, ws: Workspace | None = None, tag: str = "sample"):
        self.rt = rt
        self.tag = tag
        self._own_ws = ws is None
        self.ws = ws or Workspace(rt)
        self._idx_cache: dict[int, int] = {}

    def argmax_device(self, logits_dev: int, rows: int, cols: int) -> int:
        """设备侧逐行 argmax，返回 int32 `[rows]` 的设备指针。"""
        idx = self.ws.buffer(self.tag + ".idx", rows * 4)
        if self._idx_cache.get(rows) != idx:
            self._idx_cache[rows] = idx
        for r in range(rows):
            self.rt.launch("argmax_k", cdiv(cols, 64), 64,
                           [int(logits_dev) + r * cols * 4, cols,
                            idx + r * 4])
        self.rt.sync()
        return idx

    @staticmethod
    def greedy(logits: np.ndarray) -> np.ndarray:
        return np.argmax(np.asarray(logits), axis=-1).astype(np.int64)

    @staticmethod
    def sample(logits: np.ndarray, temperature: float = 1.0,
               top_k: int = 0, top_p: float = 1.0,
               rng: np.random.Generator | None = None) -> np.ndarray:
        """对 `[rows, vocab]` 的 logits 做温度 / top-k / top-p 采样。"""
        logits = np.asarray(logits, dtype=np.float32)
        if temperature <= 0:
            return Sampler.greedy(logits)
        rng = rng or np.random.default_rng()
        x = logits / float(temperature)
        if top_k and top_k < x.shape[-1]:
            thresh = np.partition(x, -top_k, axis=-1)[..., -top_k][..., None]
            x = np.where(x < thresh, -np.inf, x)
        if top_p < 1.0:
            order = np.argsort(-x, axis=-1)
            sorted_x = np.take_along_axis(x, order, axis=-1)
            probs = np.exp(sorted_x - sorted_x.max(axis=-1, keepdims=True))
            cdf = np.cumsum(probs, axis=-1)
            cdf /= cdf[..., -1:]
            cut = cdf > top_p
            cut[..., 0] = False
            sorted_x = np.where(cut, -np.inf, sorted_x)
            x = np.empty_like(sorted_x)
            np.put_along_axis(x, order, sorted_x, axis=-1)
        m = x.max(axis=-1, keepdims=True)
        p = np.exp(x - m)
        p /= p.sum(axis=-1, keepdims=True)
        return np.array([rng.choice(p.shape[-1], p=row) for row in p], dtype=np.int64)
