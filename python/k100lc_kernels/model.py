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
    "QUANT_SPECS", "DotLinear", "F32Linear", "RT4Linear",
    "RMSNorm", "SwiGLU", "MLP", "MoECombine", "KVCache", "Sampler",
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

    def forward_device(self, x_dev: int, rows: int, sync: bool = False) -> int:
        """在设备上算 `y = W·x`，返回 y 的设备指针（默认不 sync）。"""
        spec = self.spec
        out = self._out(rows)
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

    def forward_device(self, x_dev: int, rows: int, sync: bool = False) -> int:
        b = self._bind(rows)
        for r in range(rows):
            self.rt.launch("gemv_f32_warp_k", self.n, 64,
                           [self.w, int(x_dev) + r * self.k * 4,
                            b["out"] + r * self.n * 4, self.n, self.k, 64])
        if sync:
            self.rt.sync()
        return b["out"]

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

    def forward_device(self, x_dev: int, rows: int, sync: bool = False) -> int:
        return self.runner.gemv_device(self.path, self.wq, self.ws_scale, None,
                                       self.n, self.k, x_dev=x_dev, m=rows,
                                       sync=sync)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        return self.runner.gemv(self.path, self.wq, self.ws_scale, x,
                                self.n, self.k)


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
        self.magic_dim = div_magic(dim, rows * dim)

    def slot(self, e: int) -> int:
        """第 e 个专家输出的设备指针（布局 `[n_exp, rows, dim]`）。"""
        return self.exp_out + e * self.rows * self.dim * 4

    def forward_device(self, weights_dev: int | None = None,
                       sync: bool = False) -> int:
        w = int(weights_dev) if weights_dev is not None else self.weights
        total = self.rows * self.dim
        self.rt.launch("moe_combine_k", cdiv(total, 64), 64,
                       [self.out, self.exp_out, w, self.n_exp,
                        self.rows, self.dim, self.magic_dim])
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
