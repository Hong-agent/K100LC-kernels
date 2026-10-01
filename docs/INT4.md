# compressed-tensors INT4（W4A16）支持

本包可以直接跑 **compressed-tensors / pack-quantized INT4** 的原始权重字节：
不转换、不重量化、不重排。新增两个内核：

* `int4_dot_k`：主路径，融合 128 权重块解码 + 点积；
* `int4_dequant_k`：解码成 f32，用于逐位对账与回退路径。

示例 checkpoint 目录用 `--model-dir` / `K100LC_INT4_MODEL` 指定；本仓库
不包含权重。

## 1. 存储格式

`config.json` 的 `quantization_config` 满足：

```
format        = pack-quantized
group_size    = 128
num_bits      = 4
type          = int
symmetric     = true
actorder      = static
zero point    = 无
g_idx         = 无
```

每个被量化的线性层是三个张量：

| 张量 | dtype | shape | 内容 |
|---|---|---|---|
| `weight_packed` | I32 | `[N, K/8]` | 8 个 4bit 码 / 字，低半字节 = 更小的 k |
| `weight_scale` | BF16 | `[N, K/128]` | 每 128 个 k 一个尺度，无 global scale |
| `weight_shape` | I64 | `[2]` | `[N, K]` |

反量化：

```
码 = (word >> (4 * (k % 8))) & 0xF                # 低半字节 = 更小的 k
w[n, k] = (码 - 8) * bf16(scale[n, k // 128])     # offset-binary，码 ∈ [0,15]
```

关键性质：尺度数组的线性下标就是「128 权重块」的下标

```
scale_index = n * (K/128) + k // 128
```

所以内核只要拿到块号，就能同时算出权重和尺度两个地址，不需要额外索引表。

一个典型 27B checkpoint 里，量化的是 64 层里的 400 个线性层（注意力 /
线性注意力投影、MLP），总量约 `packed 12.16 GB + scale 0.38 GB = 12.54 GB`；
`embed_tokens`、`lm_head` 与视觉塔不量化。常见的 `nbpr = K/128` 为
`40 / 48 / 136`。

## 2. 内核

| 内核 | 作用 | 显式参数 |
|---|---|---|
| `int4_dot_k` | 一个 work-item = 一个 128 权重块 × 一行，写 `partial[块号]` | `(w, x, partial, nblocks, 64, nbpr, magic_nbpr, rows_per_exp, magic_rpe, scale, ids, stride, rows_per_w, magic_rpw)` |
| `int4_dequant_k` | 一个 work-item = 一个 I32 码字 → 8 个 f32 | `(w, scale, y, nwords, 64)`，`nwords = 块数 * 16` |
| `reduce_blocks_k` | 每行 `nbpr` 个 `partial` 求和 | `(partial, y, nrows, nbpr)` |

`int4_dot_k` 的第 10 个参数（`*_dot_k` 家族的「表指针」槽）是 **BF16 尺度
数组**。地址规则：

```
local      = (row - ew*rows_per_w)*nbpr + b
权重地址   = w     + ids[ew]*stride    + local*64
尺度地址   = scale + ids[ew]*stride/32 + local*2      # 一个 64B 权重块配 2B 尺度
```

也就是说，尺度缓冲必须和 packed 缓冲按同样的块序排（每行 `nbpr` 个 bf16，
紧凑排列）。dense 单矩阵传 `ids = [0..N-1]`、`stride = nbpr*64`、
`rows_per_exp = rows_per_w = N`；MoE 传专家表（每组 `rows_per_w` 行、
`stride = rows_per_w*nbpr*64`），语义与其它 `*_dot_k` 完全一致。

尺度在整块内是常数，所以内核在前 128 个权重上累加 `Σ(码-8)*x`、最后乘一次
尺度；与参考 `Σ((码-8)*尺度)*x` 只差 f32 舍入（实测相对 1e-7 量级）。
`int4_dequant_k` 则按「先减 8、再乘尺度」逐元素来，与主机参考逐位相同。

## 3. 用法

```bash
source env.sh

K100LC_INT4_MODEL=/path/to/checkpoint \
python3 tools/ct_int4.py --model-dir "$K100LC_INT4_MODEL" --info

K100LC_INT4_MODEL=/path/to/checkpoint \
python3 tools/ct_int4.py --model-dir "$K100LC_INT4_MODEL" --selftest

python3 tools/gen_int4_dot.py                 # 合成权重 + 真权重（若已指定）上卡对账
python3 examples/python_int4_gemv.py --model-dir /path/to/checkpoint
python3 tools/bench_int4_layers.py --model-dir /path/to/checkpoint
```

运行时侧（`examples/python_int4_gemv.py` 是完整版本）：

```python
import numpy as np
from ct_int4 import Int4Linear
from k100lc_kernels import Runtime
from gen_gemv_qdot import div_magic

lin = Int4Linear("model.language_model.layers.3.mlp.gate_proj",
                 "/path/to/checkpoint")
rt = Runtime()

pw = rt.alloc(len(lin.packed_rows(0, lin.n)))     # weight_packed
ps = rt.alloc(len(lin.scale_rows(0, lin.n)))      # weight_scale
px = rt.alloc(lin.k * 4)                          # x
pp = rt.alloc(lin.n * lin.nbpr * 4)               # partial
py = rt.alloc(lin.n * 4)                          # y
pid = rt.alloc(lin.n * 4)                         # ids：dense 传 0..N-1

rt.upload(pw, lin.packed_rows(0, lin.n))
rt.upload(ps, lin.scale_rows(0, lin.n))
rt.upload(px, x)
rt.upload(pid, np.arange(lin.n, dtype=np.uint32))

nblocks, nbpr = lin.n * lin.nbpr, lin.nbpr
m_nbpr = div_magic(nbpr, nblocks)
m_rows = div_magic(lin.n, lin.n + 1)
rt.launch("int4_dot_k", (nblocks + 63) // 64, 64,
          [pw, px, pp, nblocks, 64, nbpr, m_nbpr, lin.n, m_rows,
           ps, pid, nbpr * 64, lin.n, m_rows])
rt.launch("reduce_blocks_k", (lin.n + 63) // 64, 64,
          [pp, py, lin.n, nbpr])
rt.sync()
```

注意：

* `div_magic(d, max_i)` 生成的魔法乘数必须在整个下标范围内精确；
  放进计时循环里会多算 CPU 时间，建议提前算一次。
* `int4_dot_k` 的 grid 是 workgroup 个数：`ceil(nblocks/64)`，
  workgroup=64。
* 一个进程只加载一个 HSACO；`int4_dot_k` 已经在
  `prebuilt/k100lc_kernels.hsaco` 里，直接用 `Runtime()` 即可。

## 4. 对账

四层验证，从格式到数值：

1. **码分布**（`tools/ct_int4.py --selftest`）：16 个码的直方图峰值落在 8、
   两侧对称 —— offset-binary 的判别（二补码存档会两端出峰）。
2. **独立实现交叉验证**（可选）：设置 `K100LC_RT4_MANIFEST=/path/model.rt4.json`
   后，与本包之外另一份 INT4 打包实现的解码结果逐元素比。
3. **内核 bit-exact**：`int4_dequant_k` 对真权重与主机参考逐位相同
   （max_abs = 0）。
4. **整层数值**：`int4_dot_k + reduce_blocks_k` 对参考 f32 矩阵乘的相对
   误差在 1e-6 量级。

## 5. 性能（本机 K100_LC）

### 5.1 两条路径

以一个 `17408×5120` 的 MLP gate/up 层（权重 44.6 MB）为例：

| 路径 | 耗时 | 等效权重带宽 |
|---|---:|---:|
| `int4_dot_k` + `reduce_blocks_k`（原生解码融合） | **0.226 ms** | ≈ 200 GB/s |
| 同上，端到端（`--iters 200`） | 0.236 ms | 195 GB/s |
| `int4_dequant_k` → `gemv_f32_warp_k`（解成 f32 再算） | 1.965 ms | — |

融合路径快 **8.7 倍**：省掉 357 MB 的 f32 中间结果（写一遍 + 读一遍）。
（迭代次数少、或紧接在其它负载后面时会偏慢 10~30%；上表用 100~200 次平均、
预热之后的稳定值。）

### 5.2 按层类型

| 形状 | 层数 | 字节 | 耗时/层 | 等效带宽 |
|---|---:|---:|---:|---:|
| gate/up `17408×5120` | 128 | 5.88 GB | 0.246 ms | 181 GB/s |
| down `5120×17408` | 64 | 2.94 GB | 0.497 ms | 90 GB/s |
| in_proj_qkv `10240×5120` | 48 | 1.30 GB | 0.163 ms | 161 GB/s |
| out_proj `5120×6144` | 64 | 1.04 GB | 0.120 ms | 131 GB/s |
| in_proj_z `6144×5120` | 48 | 0.78 GB | 0.118 ms | 133 GB/s |
| q_proj `12288×5120` | 16 | 0.52 GB | 0.185 ms | 170 GB/s |
| k/v_proj `1024×5120` | 32 | 0.09 GB | 0.051 ms | 51 GB/s |
| **合计** | **400** | **12.54 GB** | — | **91.9 ms/token** |

最后一行只算权重搬运，不含 attention、归一化、采样等其它开销。两个偏慢的
点各有原因：`down_proj` 的 K=17408（一行激活 70 KB，L1 放不下，且
`nbpr=136` 让归约变长）；`k/v_proj` 只有 1024 行（4 万个 work-item，
并行度不够）。

### 5.3 瓶颈在解码指令，不在访存

把同一个内核里的 x 加载全部提前、甚至完全去掉 x 加载，耗时都不变
（0.232 ms）；只留「读 64 字节权重 + 存 partial」的最小版本能跑 607 GB/s。
也就是说 4bit 解码（`lshr/and → cvt → add(-8) → fma`，平均 4.9 条指令/权重）
占了大头，访存已经吃满。

想再快就要换数值口径：把激活量化成 int8（W4A8）后用 `v_dot4_i32_i8`
一条指令算 4 个乘积，同时 x 的访存也降到 1/4；见
[`KERNEL_CALLING.md`](KERNEL_CALLING.md) 第 6.5 节。本包的 `int4_dot_k`
走的是**保精度**的 W4A16：激活不量化，数值上只差 f32 舍入。

## 6. 限制

* 假设 `K % 128 == 0`，且尺度是 BF16、每 128 个 k 一个（这类
  checkpoint 的实际布局）。其它 `group_size` 需要改内核里的块长。
* `int4_dot_k` 的 `partial` 布局是「行优先、行内按块号」，与其它
  `*_dot_k` 一致；`reduce_blocks_k` 可以直接复用。
* `int4_dot_k` 是纯 1D 启动，不受 2D grid 真机限制影响。
