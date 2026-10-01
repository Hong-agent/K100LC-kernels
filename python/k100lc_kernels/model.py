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

from .runtime import LaunchPlan, Runtime

__all__ = [
    "cdiv", "div_magic", "Workspace",
    "QUANT_SPECS", "DotLinear", "F32Linear", "RT4Linear", "Int4Linear",
    "RMSNorm", "SwiGLU", "MLP", "MoECombine", "MoEExperts", "RoPE",
    "KVCache", "Sampler",
    "Attention", "FlashAttention", "TransformerLayer",
    "run_sequence", "gemv_f32", "ROWS8_MAX_K", "SPLIT_SOFTMAX_MIN_PAD",
    "SPLITK_VTP",
]


def cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


# `gemv_f32_rows8_k` 只在 k 小的时候更快：它每 8 行只做 3 次 `s_barrier`
# （`gemv_f32_warp_k` 每行 6 次），但每个 workgroup 的 k 循环是串行的、没有
# 双缓冲。实测（nrows=16384、f32）：k=128 → 6.5 倍、256 → 3.6 倍、384 →
# 3.2 倍、512 → 1.5 倍、768/896 → 1.5 倍、**1024 → 0.54 倍（更慢）**，
# 所以按 k 上限切换（512 是留了安全余量的阈值）。
ROWS8_MAX_K = 512

# 分块 softmax 的最小列数：再小的话 4 次启动的开销盖过收益（`softmax_vec_k`
# 一行只要 1~2 轮，本身就很快）。下面用 `pad >= SPLIT_SOFTMAX_MIN_PAD` 判断。
SPLIT_SOFTMAX_MIN_PAD = 2048

# `Attention` 的 `Vt·P` 走 split-K 版（见 `forward_device` 里那段注释）；
# 关掉它就退回「一行一个 warp」，用来做对照基准。
SPLITK_VTP = True


def gemv_f32_acc(rt: Runtime, w: int, x: int, y: int, nrows: int, k: int) -> None:
    """`y += W·x`（把残差并进 GEMV，省一次 `add_inplace_k`）。

    和 `gemv_f32` 一样的选型规则，只是走 `*_acc_k` 变体；没有这两颗内核的旧包
    会退回「GEMV + `add_inplace_k`」两步。
    """
    nrows, k = int(nrows), int(k)
    if (nrows % 8 == 0 and k % 128 == 0 and k <= ROWS8_MAX_K
            and rt.has("gemv_f32_rows8_acc_k")):
        rt.launch("gemv_f32_rows8_acc_k", nrows // 8, 64, [w, x, y, nrows, k, 64])
    elif rt.has("gemv_f32_warp_acc_k"):
        rt.launch("gemv_f32_warp_acc_k", nrows, 64, [w, x, y, nrows, k, 64])
    else:
        gemv_f32(rt, w, x, y, nrows, k)
        rt.launch("add_inplace_k", cdiv(nrows, 64), 64, [y, y, nrows])


def gemv_f32(rt: Runtime, w: int, x: int, y: int, nrows: int, k: int) -> None:
    """f32 GEMV 的分派：小 k 走 8 行/warp 版，其余走 warp-per-row。

    两个内核的参数表完全相同，只是 grid 不同（行数 vs 行数/8）。
    """
    nrows, k = int(nrows), int(k)
    if (nrows % 8 == 0 and k % 128 == 0 and k <= ROWS8_MAX_K
            and rt.has("gemv_f32_rows8_k")):
        rt.launch("gemv_f32_rows8_k", nrows // 8, 64,
                  [w, x, y, nrows, k, 64])
    else:
        rt.launch("gemv_f32_warp_k", nrows, 64, [w, x, y, nrows, k, 64])


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
            gemv_f32(self.rt, self.w, int(x_dev) + r * self.k * 4,
                     out + r * self.n * 4, self.n, self.k)
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
        if self.rt.has("rmsnorm_fast_k"):
            # 包里的 `rmsnorm_k` 是 HIP 编出来的：每个元素一条 load + 一条
            # `s_waitcnt vmcnt(0)`，dim=512 时是 8 次完整访存往返，单次实测
            # 15.1 us（投递地板 7.3）。自研的 `rmsnorm_fast_k` 把 load 按 4 个
            # 一批发，dim=512 只要 2 次往返，实测 7.6 us；数值一致到 1e-7。
            self.rt.launch("rmsnorm_fast_k", rows, 64,
                           [out, int(x_dev), self.w, self.dim, self.eps])
        else:
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
    每一步都按尺寸分派，长上下文（v1.8.3）会比「一行一个 warp + 单 warp
    softmax」的老路快 2~3 倍：

    =================  ==========================================================
    步骤                表达式 / 内核
    =================  ==========================================================
    `scores = K·q`     `scores[n] = Σ_d K[n,d]·q[d]`（**K 当权重矩阵**）→
                       `gemv_f32_rows8_k`（8 行/warp，dim ≤ 512 且 128 的倍数）
                       否则 `gemv_f32_warp_k`
    尾部掩码            `fill_k`（padding 段填 -1e30）
    softmax            `pad ≥ 2048`：`block_max_k` → `reduce_max1_k` →
                       `block_exp_sum_k`（写出**未归一化**的 exp）→ `reduce_sum1_k`
                       否则 `scale_mul_k` + `softmax_vec_k`
    `out = Vt·P`       `out[d] = Σ_j Vt[d,j]·P[j]` → `gemv_f32_rows8_split_k`
                       （split-K，16 段）+ `reduce_blocks_k`，否则
                       `attn_pv_part` / `gemv_f32_warp_k`
    归一化              分块路用 `div_scalar_k` 把 `1/L` 乘回 `dim` 个输出
    =================  ==========================================================

    实测 `dim=128`：n_kv=1000/4096/16384/32768 → 106/121/157/218 us
    （v1.8.2 之前是 132/200/417/682 us）。三个分派开关都有回退路径，
    换旧 hsaco 或换尺寸都不会崩。

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
        # 分块 softmax（`block_max_k`/`reduce_max_k`/`block_exp_sum_k`）用的
        # 中间量：每块一个 max/exp 和（块数 ≤ 64），加两个设备标量。
        self.pmax = self.ws.buffer(tag + ".pmax", 64 * 4)
        self.psum = self.ws.buffer(tag + ".psum", 64 * 4)
        self.mval = self.ws.buffer(tag + ".mval", 4)
        self.lsum = self.ws.buffer(tag + ".lsum", 4)
        self.length = 0

    def append_device(self, k_dev: int, v_dev: int, rows: int) -> None:
        """追加 `rows` 个 token 的 K/V；V 转置后写进 `Vt`。"""
        if self.length + rows > self.max_len:
            raise ValueError(f"KV 溢出：{self.length}+{rows} > {self.max_len}")
        off = self.length
        self.rt.copy_dev(self.k + off * self.dim * 4, int(k_dev),
                         rows * self.dim * 4)
        if self._use_vt_scatter():
            self._vt_scatter_device(int(v_dev), off, rows)
        else:
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
        if self._use_vt_scatter():
            # 只上传一次 V（行主序），转置交给设备侧的 vt_scatter_k
            pv = self.ws.buffer(self.tag + ".vrow", rows * self.dim * 4)
            self.rt.upload(pv, v.reshape(-1))
            self._vt_scatter_device(pv, off, rows)
        else:
            self._store_vt(v, off, rows)
        self.length += rows

    def _use_vt_scatter(self) -> bool:
        """`vt_scatter_k`（编译器生成的 LDS 分块转置）能不能用。

        `dim/64` 必须是 2 的幂（内核按 `blk >> cshift` 切列 tile）。
        """
        return (self.dim % 64 == 0 and (self.dim // 64) & (self.dim // 64 - 1) == 0
                and self.rt.has("vt_scatter_k"))

    def _vt_scatter_device(self, v_dev: int, off: int, rows: int) -> None:
        """设备侧转置：`vt[d*max_len + off + t] = v[t*dim + d]`（一个 kernel）。

        以前这里是「download → numpy 转置 → dim 次 upload」，dim=128、rows=1 时
        实测 **2.12 ms/次**（128 次 4 字节上传），比一次注意力前向还贵三个量级。
        """
        ntile = cdiv(rows, 64) * (self.dim // 64)
        cshift = (self.dim // 64).bit_length() - 1
        self.rt.launch("vt_scatter_k", ntile, 64,
                       [self.vt, int(v_dev), int(rows), self.dim, self.dim,
                        self.max_len, int(off), cshift])

    def _store_vt(self, v: np.ndarray, off: int, rows: int) -> None:
        """兜底（hsaco 里没有 `vt_scatter_k` 时）：主机侧逐列转置上传。"""
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
        inv = 1.0 / float(self.dim) ** 0.5
        # 1) scores = K·q（K 当权重矩阵：行=位置，列=dim）。k=dim、行数=n_kv，
        #    dim ≤ 512 时走 8 行/warp 的版本（每 8 行只要 3 次 s_barrier 而不是
        #    48 次；n_kv=16384、dim=128 实测 126 us → 19.5 us，66 → 429 GB/s）。
        gemv_f32(self.rt, self.k, int(q_dev), self.scores, pad, self.dim)
        # 2) 尾部掩码（真正的分数不可能到 -1e30，所以最大/exp 都会忽略它们）
        if pad > n_kv:
            self.rt.launch("fill_k", -(-(pad - n_kv) // 64), 64,
                           [self.scores + n_kv * 4, -1e30, pad - n_kv])
        # 3) softmax。两条路：
        #    * 分块（默认）：块内 LDS 归约 + 跨块归约，块之间完全并行。原来
        #      `softmax_vec_k` 把整行交给**一个** warp 扫三趟，n_kv=16384 要
        #      126 us；分块后每块只扫 1 轮。
        #    * 老路（`softmax_vec_k`）：hsaco 里没有分块内核时兜底。
        #    分块路的 `probs` 存的是**未归一化**的 exp：归一化挪到输出侧
        #    （只除 dim 个数，而不是整行 pad 个数），最后 `div_scalar_k` 除 L。
        split = (pad >= SPLIT_SOFTMAX_MIN_PAD
                 and all(self.rt.has(n) for n in (
                     "block_max_k", "reduce_max1_k", "block_exp_sum_k",
                     "reduce_sum1_k", "div_scalar_k")))
        if split:
            # 块数取「≤64 且整除 pad/256」的最大 2 的幂（每块列数是 256 的倍数）
            nblk = 1
            while nblk * 2 <= 64 and (pad // 256) % (nblk * 2) == 0:
                nblk *= 2
            sh = nblk.bit_length() - 1
            self.rt.launch("block_max_k", nblk, 64,
                           [self.pmax, self.scores, pad, sh])
            self.rt.launch("reduce_max1_k", 1, 64, [self.pmax, self.mval, nblk])
            self.rt.launch("block_exp_sum_k", nblk, 64,
                           [self.psum, self.probs, self.scores, self.mval,
                            pad, sh, inv])
            self.rt.launch("reduce_sum1_k", 1, 64, [self.psum, self.lsum, nblk])
        else:
            self.rt.launch("scale_mul_k", -(-pad // 64), 64,
                           [self.scores, inv, pad])
            self.rt.launch("softmax_vec_k", 1, 64,
                           [self.probs, self.scores, 1, pad, 64])
        # 4) out = Vt·P。三条路：
        #    * pad == max_len（缓存填满）→ `gemv_f32_warp_k` 把 Vt 当权重矩阵；
        #    * 否则用 `attn_pv_part`（支持自定义行距，只扫 pad 列）+ 归约——
        #      `gemv_f32_warp_k` 的行距必须等于列数，缓存没填满时只能用前者。
        #    * 优先：`gemv_f32_rows8_split_k`。这一步的行数只有 dim（128）、列数
        #      却是整个上下文：「一行一个 warp」只有 128 个 workgroup，每个还要
        #      串行做 k/64 次访存往返（16K 时实测 129 us）。split-K 把列切成
        #      nsplit 段，workgroup 数变成 (dim/8)·nsplit，每段互相独立，最后
        #      用 `reduce_blocks_k` 沿段求和。行距是参数，所以「缓存没填满」
        #      （行距 = max_len）和「填满」两种布局都用同一条路。
        splitk = (SPLITK_VTP and self.rt.has("gemv_f32_rows8_split_k")
                  and self.dim % 8 == 0 and pad % 128 == 0)
        if splitk:
            nsplit = 1
            while nsplit < 16 and (pad // 128) % (nsplit * 2) == 0:
                nsplit *= 2
            sh = nsplit.bit_length() - 1
            self.rt.launch("gemv_f32_rows8_split_k", (self.dim // 8) * nsplit, 64,
                           [self.vt, self.probs, self.part, self.dim, pad,
                            self.max_len, sh])
            self.rt.launch("reduce_blocks_k", -(-self.dim // 64), 64,
                           [self.part, self.out, self.dim, nsplit])
        elif pad == self.max_len:
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
        if split:
            # 分块路的 probs 未归一化：把 1/L 乘回输出（dim 个数，不是 pad 个）
            self.rt.launch("div_scalar_k", -(-self.dim // 64), 64,
                           [self.out, self.lsum, self.dim])
        if sync:
            self.rt.sync()
        return self.out

    def forward(self, q: np.ndarray, n_kv: int | None = None) -> np.ndarray:
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(-1)
        self.rt.upload(self.q, q)
        out = self.forward_device(self.q, n_kv, sync=True)
        return self.rt.download(out, self.dim, np.float32)


class FlashAttention:
    """**多头**解码注意力：一次 forward = 2 个 launch（`flash_dec_part_k` +
    `flash_dec_comb_k`），不管几个头。

    和单头的 `Attention` 的区别不只是头数：

    * K 缓存是**每个头转置**存放的 `Kt [n_heads, head_dim, max_len]`
      （`vt_scatter_k` 一个 launch 就能把 `[rows, n_heads*head_dim]` 整个转过来），
      V 是行主序 `[n_heads, max_len, head_dim]`（追加就是一次 `copy_dev`）。
      这样算分数时 lane j 读 `kt[d][base+j]` 是连续的、算输出时 lane d 读
      `v[base+j][d]` 也是连续的。
    * softmax 在分块内两趟做完（max → exp+和），跨分块的合并由 combine 内核按
      `exp(m_s - M)` 加权后一次除完；**不需要** `softmax_vec_k` 那种单 warp
      扫整行，也不需要外部的 `scale_mul_k`/`fill_k`（mask 在内核里做）。

    约束：`pad = ceil(n_kv/256)*256`、`nsplit` 取「整除 `pad/64` 的最大 2 的幂
    （≤ 64）」，并且 `R = pad/nsplit` 要是 64 的倍数且 ≤ 4096（LDS 16 KB + 256）。
    """

    def __init__(self, rt: Runtime, n_heads: int, head_dim: int, max_len: int,
                 ws: Workspace | None = None, tag: str = "fmha"):
        for v, nm in ((n_heads, "n_heads"), (head_dim, "head_dim")):
            if v <= 0:
                raise ValueError(f"{nm} 必须为正")
        if head_dim > 128:
            raise ValueError("head_dim 目前最大支持 128（内核按 64 一档扫）")
        self.rt = rt
        self.n_heads, self.head_dim = int(n_heads), int(head_dim)
        self.dim = self.n_heads * self.head_dim
        self.max_len = int(max_len)
        self.tag = tag
        self._own_ws = ws is None
        self.ws = ws or Workspace(rt)
        self.kt = self.ws.buffer(tag + ".kt", self.dim * self.max_len * 4)
        self.v = self.ws.buffer(tag + ".v", self.max_len * self.dim * 4)
        self.po = self.ws.buffer(tag + ".po", self.n_heads * 64 * self.head_dim * 4)
        self.pm = self.ws.buffer(tag + ".pm", self.n_heads * 64 * 4)
        self.pl = self.ws.buffer(tag + ".pl", self.n_heads * 64 * 4)
        self.out = self.ws.buffer(tag + ".out", self.dim * 4)
        self.q = self.ws.buffer(tag + ".q", self.dim * 4)
        self.length = 0

    # ---------------- 追加 KV ----------------
    def _copy_k(self, k_dev: int, off: int, rows: int) -> None:
        """K：`[rows, n_heads*head_dim]` → 转置写进 `Kt [dim, max_len]`。

        用 `vt_scatter_k`（编译器 + DSL 共享内存生成的分块转置），一个 launch
        就把**所有头**都转过去：`dim/64` 是 2 的幂时列 tile 天然覆盖每个头。
        """
        if not self.rt.has("vt_scatter_k"):
            raise RuntimeError("内核包里没有 vt_scatter_k（需要重跑 build_all.sh）")
        cshift = (self.dim // 64).bit_length() - 1
        ntile = cdiv(rows, 64) * (self.dim // 64)
        self.rt.launch("vt_scatter_k", ntile, 64,
                       [self.kt, int(k_dev), rows, self.dim, self.dim,
                        self.max_len, off, cshift])

    def append_device(self, k_dev: int, v_dev: int, rows: int) -> None:
        """追加 `rows` 个 token：K 转置进 `Kt`，V 行主序拷进 `V`。

        优先用 `vt_scatter_v_k`（**一个** launch 同时做转置和拷贝）——这样整个
        追加都是 launch，能跟其它 launch 一起进 `Runtime.batch()`；只有旧 hsaco
        里没有这个内核时才退回「转置 + `copy_dev`」两步。
        """
        rows = int(rows)
        if self.length + rows > self.max_len:
            raise ValueError(f"KV 溢出：{self.length}+{rows} > {self.max_len}")
        off = self.length
        if (rows == 1 and self.dim % 64 == 0
                and self.rt.has("vt_scatter_v1_k")):
            # 解码：一次只追加 1 行。`vt_scatter_v_k` 是为「一次一整块 64 行」
            # 设计的，两圈固定 64 次的循环在 rows=1 时基本空转（dim=512、grid=8
            # 实测 22.9 us，其中真活只有「1 个 K + 1 个 V + 两笔 store」）。
            # `vt_scatter_v1_k` 按「一个 lane 一列」直接铺开、没有循环。
            self.rt.launch("vt_scatter_v1_k", self.dim // 64, 64,
                           [self.kt, self.v, int(k_dev), int(v_dev),
                            self.dim, self.max_len, off])
        elif self.rt.has("vt_scatter_v_k") and self._pitch_ok():
            cshift = (self.dim // 64).bit_length() - 1
            ntile = cdiv(rows, 64) * (self.dim // 64)
            self.rt.launch("vt_scatter_v_k", ntile, 64,
                           [self.kt, self.v, int(k_dev), int(v_dev), rows,
                            self.dim, self.dim, self.dim, self.max_len, off,
                            cshift])
        else:
            self._copy_k(k_dev, off, rows)
            self.rt.copy_dev(self.v + off * self.dim * 4, int(v_dev),
                             rows * self.dim * 4)
        self.length += rows

    def _pitch_ok(self) -> bool:
        return (self.dim % 64 == 0
                and (self.dim // 64) & (self.dim // 64 - 1) == 0)

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        k = np.ascontiguousarray(k, dtype=np.float32).reshape(-1, self.dim)
        v = np.ascontiguousarray(v, dtype=np.float32).reshape(-1, self.dim)
        rows = k.shape[0]
        if self.length + rows > self.max_len:
            raise ValueError(f"KV 溢出：{self.length}+{rows} > {self.max_len}")
        pk = self.ws.buffer(self.tag + ".krow", rows * self.dim * 4)
        pv = self.ws.buffer(self.tag + ".vrow", rows * self.dim * 4)
        self.rt.upload(pk, k.reshape(-1))
        self.rt.upload(pv, v.reshape(-1))
        self.append_device(pk, pv, rows)

    def reset(self) -> None:
        self.length = 0

    # ---------------- 前向 ----------------
    def plan(self, n_kv: int | None = None) -> tuple[int, int, int]:
        """返回 `(pad, nsplit, R)`；参数不合法就直接报错（不静默算错）。"""
        n_kv = self.length if n_kv is None else int(n_kv)
        if n_kv <= 0:
            raise ValueError("n_kv 必须为正")
        pad = cdiv(n_kv, 256) * 256
        p64 = pad // 64
        nsplit = 1
        while nsplit * 2 <= 64 and p64 % (nsplit * 2) == 0:
            nsplit *= 2
        R = pad // nsplit
        if R % 64 or R > 4096:
            raise ValueError(
                f"pad={pad} 下 nsplit={nsplit} 不合适（R={R}，要求 64 的倍数且 "
                f"≤ 4096）；换 max_len 或把上下文切小一点")
        return pad, nsplit, R

    def forward_device(self, q_dev: int, n_kv: int | None = None,
                       sync: bool = False, out_dev: int | None = None) -> int:
        """一次解码注意力（q 是**一个** token 的 `[n_heads*head_dim]`）。"""
        pad, nsplit, _R = self.plan(n_kv)
        n_kv = self.length if n_kv is None else int(n_kv)
        cshift = nsplit.bit_length() - 1
        inv = np.float32(1.0 / np.sqrt(self.head_dim))
        out = int(out_dev) if out_dev is not None else self.out
        self.rt.launch("flash_dec_part_k", self.n_heads * nsplit, 64,
                       [self.po, self.pm, self.pl, int(q_dev), self.kt, self.v,
                        n_kv, self.head_dim, pad, self.max_len, cshift,
                        float(inv), self.dim])       # 最后一个是 V 的行距
        self.rt.launch("flash_dec_comb_k", self.n_heads, 64,
                       [out, self.po, self.pm, self.pl, self.head_dim,
                        nsplit, float(inv)])
        if sync:
            self.rt.sync()
        return out

    def forward(self, q: np.ndarray, n_kv: int | None = None,
                sync: bool = True):
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(-1)
        if q.size != self.dim:
            raise ValueError(f"q 的元素数应是 n_heads*head_dim={self.dim}")
        self.rt.upload(self.q, q)
        out = self.forward_device(self.q, n_kv=n_kv, sync=sync)
        if not sync:
            return out
        return self.rt.download(out, self.dim, np.float32)


class TransformerLayer:
    """一层 decoder 的**解码路径**，整层设备侧串联，逐 token 一次 `sync`：

        h = x
        h += W_o · Attention(RoPE(W_qkv·RMSNorm(h)))      # 多头 + KV 缓存
        h += W_down · (silu(gate) * up)                    # 门控 MLP

    设计要点：

    * **权重拼成整块**：QKV 三个投影拼成 `[3*dim, dim]`、gate/up 拼成
      `[2*ffn, dim]`，于是每层各只要 1 次 GEMV（而不是 3 次 / 2 次）。
    * **RoPE 一次处理所有头**：q/k 按 `[n_heads, head_dim]` 摆，`rows = n_heads`、
      `tsh = log2(n_heads)`、`tbase = 位置`，所有头共用同一行表。
    * 注意力用融合的 `FlashAttention`（2 个 launch），K/V 追加 2 次（转置 + 拷贝）。
    * 全程 `sync=False` 入队，最后一次 `sync` —— 实测一次 launch 固定开销约 11 us，
      逐算子 sync 会把 CPU/GPU 串起来。

    ```python
    layer = TransformerLayer(rt, dim=512, n_heads=8, head_dim=64, ffn=2048,
                             max_len=4096, w_qkv=..., w_o=..., w_gate_up=...,
                             w_down=..., w_norm1=..., w_norm2=...)
    for pos, tok in enumerate(tokens):
        h = layer.forward(tok, pos=pos)      # 一次 sync
    ```
    """

    def __init__(self, rt: Runtime, dim: int, n_heads: int, head_dim: int,
                 ffn: int, max_len: int, w_qkv, w_o, w_gate_up, w_down,
                 w_norm1, w_norm2, eps: float = 1e-6,
                 ws: Workspace | None = None, tag: str = "layer"):
        n_heads, head_dim, ffn = int(n_heads), int(head_dim), int(ffn)
        if head_dim % 64:
            raise ValueError("head_dim 必须是 64 的倍数（融合注意力内核）")
        if n_heads & (n_heads - 1) or n_heads < 1:
            raise ValueError("n_heads 必须是 2 的幂（RoPE 的 tsh）")
        if int(dim) != n_heads * head_dim:
            raise ValueError(f"dim({dim}) 必须等于 n_heads*head_dim"
                             f"({n_heads * head_dim})")
        self.rt = rt
        self.dim, self.n_heads, self.head_dim = int(dim), n_heads, head_dim
        self.ffn, self.max_len = ffn, int(max_len)
        self.tag = tag
        self._own_ws = ws is None
        self.ws = ws or Workspace(rt)
        ws = self.ws
        self.norm1 = RMSNorm(rt, dim, w_norm1, eps, ws, tag + ".n1")
        self.qkv = F32Linear(rt, 3 * dim, dim, w_qkv, ws, tag + ".qkv")
        self.rope = RoPE(rt, head_dim, max_len, ws=ws, tag=tag + ".rope")
        self.attn = FlashAttention(rt, n_heads, head_dim, max_len, ws=ws,
                                   tag=tag + ".attn")
        self.o = F32Linear(rt, dim, dim, w_o, ws, tag + ".o")
        self.norm2 = RMSNorm(rt, dim, w_norm2, eps, ws, tag + ".n2")
        self.gate_up = F32Linear(rt, 2 * ffn, dim, w_gate_up, ws, tag + ".gu")
        self.act = SwiGLU(rt, ws, tag + ".act")
        self.down = F32Linear(rt, dim, ffn, w_down, ws, tag + ".down")
        self.h = ws.buffer(tag + ".h", dim * 4)
        # q/k 在 qkv 输出里是相连的两段，RoPE 一次扫 2*n_heads 行就都做了
        self.qk = ws.buffer(tag + ".qk", 2 * dim * 4)
        self.acc = ws.buffer(tag + ".act", max(ffn, dim) * 4)
        self.tsh = n_heads.bit_length() - 1
        self.length = 0
        self._plan = None
        self._plan_idx: dict[str, int] = {}

    def _plan_locate(self, calls: list) -> dict[str, int]:
        """在录下来的调用序列里定位需要打补丁的记录（按内核名）。"""
        want = {"rope_apply_k": "rope", "vt_scatter_v_k": "append",
                "vt_scatter_v1_k": "append1",
                "flash_dec_part_k": "part", "flash_dec_comb_k": "comb"}
        idx = {}
        for i, (kernel, _g, _w, _a) in enumerate(calls):
            key = want.get(kernel)
            if key is not None and key not in idx:
                idx[key] = i
        return idx

    def _patch_plan(self, pos: int) -> None:
        """逐 token 变化的量：RoPE 表行、KV 追加列偏移、注意力长度/分块/grid。"""
        plan, idx = self._plan, self._plan_idx
        if "rope" in idx:
            plan.set_arg(idx["rope"], 7, int(pos))              # tbase
        if "append" in idx:
            plan.set_arg(idx["append"], 9, int(self.length))    # y0
        if "append1" in idx:                                    # 单行版：y0 是第 7 个参数
            plan.set_arg(idx["append1"], 6, int(self.length))
        if "part" in idx:
            pad, nsplit, _R = self.attn.plan(self.length + 1)
            plan.set_arg(idx["part"], 6, self.length + 1)       # n_kv
            plan.set_arg(idx["part"], 8, pad)
            plan.set_arg(idx["part"], 10, nsplit.bit_length() - 1)
            plan.set_grid(idx["part"], self.n_heads * nsplit)
        if "comb" in idx:
            _, nsplit, _R = self.attn.plan(self.length + 1)
            plan.set_arg(idx["comb"], 5, nsplit)

    def reset(self) -> None:
        self.attn.reset()
        self.length = 0
        self._plan = None

    def forward_device(self, x_dev: int, pos: int, sync: bool = False) -> int:
        """跑一个 token：`x_dev` 是 `[dim]`，返回残差流 `h` 的设备指针。"""
        rt, ws = self.rt, self.ws
        if self.length >= self.max_len:
            raise ValueError("KV 缓存满")
        rt.copy_dev(self.h, int(x_dev), self.dim * 4)
        # 第一个 token：边跑边把这一层的 launch 序列记下来（顺序、grid、参数），
        # 之后每个 token 只打几个补丁（位置 / KV 列偏移 / 长度与分块）再一次性
        # 重放——逐 token 重走 Python 路径实测每 token 要 ~0.27 ms，是这条路线的
        # 主要开销；录成 plan 后每 token 只剩几次赋值 + 一次投递。
        if self._plan is None:
            with rt.batch():
                h = self._forward_batched(rt, x_dev, pos)
                calls = [(k, g, w, list(a)) for k, g, w, a, _n in rt._batch]
            self._plan = LaunchPlan(rt, calls)
            self._plan_idx = self._plan_locate(calls)
            rt._batch = None
            if sync:
                rt.sync()
            return h
        self._patch_plan(pos)
        self._plan.run()
        # 重放路径里 append/kernel 序列是录下来的，长度得自己往前走
        self.length += 1
        self.attn.length = self.length
        if sync:
            rt.sync()
        return self.h

    def _forward_batched(self, rt, x_dev: int, pos: int) -> int:
        n1 = self.norm1.forward_device(self.h, 1, sync=False)
        qkv = self.qkv.forward_device(n1, 1, sync=False)
        # [q|k] 一次做完 RoPE：rows = 2*n_heads、tsh 再多 1 位 → 两段的头共用
        # 同一个表行（tbase = 位置），省掉一次 launch
        qk = self.rope.forward_device(qkv, 2 * self.n_heads, pos=pos,
                                      out_dev=self.qk, tsh=self.tsh + 1)
        self.attn.append_device(qk + self.dim * 4, qkv + 2 * self.dim * 4, 1)
        self.length += 1
        attn = self.attn.forward_device(qk)
        # 残差并进输出投影：h += W_o·attn（省一次 add_inplace_k）
        gemv_f32_acc(rt, self.o.w, attn, self.h, self.dim, self.dim)
        n2 = self.norm2.forward_device(self.h, 1, sync=False)
        gu = self.gate_up.forward_device(n2, 1, sync=False)
        if rt.has("gemv_f32_gated_acc_k"):
            # SwiGLU 也融进 down 的 GEMV：h += W_down·(silu(gate)*up)
            rt.launch("gemv_f32_gated_acc_k", self.dim, 64,
                      [self.down.w, gu, gu + self.ffn * 4, self.h, self.dim,
                       self.ffn, 64])
        else:                       # 旧 hsaco：silu_mul + 累加 GEMV 两步
            act = self.act.forward_device(gu, gu + self.ffn * 4, self.ffn,
                                          out_dev=self.acc)
            gemv_f32_acc(rt, self.down.w, act, self.h, self.dim, self.ffn)
        return self.h

    def forward(self, x: np.ndarray, pos: int, sync: bool = True):
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        px = self.ws.buffer(self.tag + ".x", self.dim * 4)
        self.rt.upload(px, x)
        h = self.forward_device(px, pos, sync=sync)
        if not sync:
            return h
        return self.rt.download(h, self.dim, np.float32)


class RoPE:
    """旋转位置编码（RoPE，**rotate-half** 约定）：`rope_apply_k` 的运行时封装。

    表按位置预先算好常驻显存（`[max_len, dim/2]` 的 cos / sin），逐 token 只发
    一个内核，位置由 `pos` 直接指到表里，不用重算：

        theta_j = base ** (-2j/dim)        j = 0..dim/2-1
        cos[pos, j] = cos(pos * theta_j)   sin[pos, j] = sin(pos * theta_j)

    成对的是 `(j, j + dim/2)`（HF Llama / GPT-NeoX 的 `rotate_half`），不是 GPT-J
    的交错 `(2j, 2j+1)`：

        y[j]        = x[j]*cos - x[j+half]*sin
        y[j+half]   = x[j]*sin + x[j+half]*cos

    ```python
    rope = RoPE(rt, dim=128, max_len=4096)
    y = rope.forward(x, pos=0)           # x [rows, dim] → [rows, dim]
    y_dev = rope.forward_device(x_dev, rows, pos, sync=False)
    ```
    """

    def __init__(self, rt: Runtime, dim: int, max_len: int,
                 base: float = 10000.0, ws: Workspace | None = None,
                 tag: str = "rope"):
        if dim <= 0 or dim % 2:
            raise ValueError("RoPE 的 dim 必须是正偶数")
        if max_len <= 0:
            raise ValueError("max_len 必须为正")
        self.rt = rt
        self.dim, self.half = int(dim), int(dim) // 2
        self.max_len, self.base = int(max_len), float(base)
        self.tag = tag
        self._own_ws = ws is None
        self.ws = ws or Workspace(rt)
        j = np.arange(self.half, dtype=np.float64)
        theta = self.base ** (-2.0 * j / self.dim)
        pos = np.arange(self.max_len, dtype=np.float64)[:, None]
        ang = pos * theta[None, :]
        self.cos = self.ws.buffer(tag + ".cos", self.max_len * self.half * 4)
        self.sin = self.ws.buffer(tag + ".sin", self.max_len * self.half * 4)
        rt.upload(self.cos, np.cos(ang).astype(np.float32).reshape(-1))
        rt.upload(self.sin, np.sin(ang).astype(np.float32).reshape(-1))
        self._rows_cache: dict[int, int] = {}

    def _out(self, rows: int) -> int:
        p = self._rows_cache.get(rows)
        if p is None:
            p = self.ws.buffer(f"{self.tag}.out{rows}", rows * self.dim * 4)
            self._rows_cache[rows] = p
        return p

    def forward_device(self, x_dev: int, rows: int, pos: int = 0,
                       sync: bool = False, out_dev: int | None = None,
                       tsh: int = 0) -> int:
        """`tsh`：给同一个 token 的多个头做 RoPE 时，x 按 `[n_heads, head_dim]`
        摆、`rows = n_heads`、`tsh = log2(n_heads)`，所有头共用一行表。"""
        rows, pos = int(rows), int(pos)
        if rows <= 0 or pos < 0 or pos + rows > self.max_len:
            raise ValueError(f"RoPE 位置越界：pos={pos} rows={rows} "
                             f"max_len={self.max_len}")
        y = int(out_dev) if out_dev is not None else self._out(rows)
        # tbase=pos：表行 = 当前位置；tsh：一行表供 2^tsh 个「头」共用
        self.rt.launch("rope_apply_k", rows, 64,
                       [y, int(x_dev), self.cos, self.sin, rows, self.dim,
                        int(tsh), pos])
        if sync:
            self.rt.sync()
        return y

    def forward(self, x: np.ndarray, pos: int = 0, sync: bool = True):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        px = self.ws.buffer(self.tag + ".x", x.nbytes)
        self.rt.upload(px, x.reshape(-1))
        y = self.forward_device(px, x.shape[0], pos=pos, sync=sync)
        if not sync:
            return y
        return self.rt.download(y, int(x.size), np.float32).reshape(x.shape)


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
