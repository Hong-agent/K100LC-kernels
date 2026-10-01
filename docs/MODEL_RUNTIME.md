# 模型级运行时（model.py）

`k100lc_kernels.model` 把内核包成可直接组合的推理算子，覆盖 AI 模型解码
阶段的主要开销：线性层（f32 / INT4 / GGUF / RT4）、RMSNorm、SwiGLU/MLP、
MoE 专家合并、KV cache 与采样。

它的目标不是替代模型框架，而是提供一个 **可复用、可计时、可对账** 的
最小执行层：权重上传一次、工作缓冲复用、一次 forward 只 sync 一次。

```bash
source env.sh
python3 examples/python_model_layer.py --rows 4 --dim 512 --ffn 1024
```

实测输出（K100_LC，合成权重）：

```
[f32 ] rows=4 dim=512 ffn=1024 max_rel=3.59e-07 time=0.218 ms
[int4] rows=4 dim=512 ffn=1024 max_rel=3.70e-07 time=0.290 ms
[moe ] rows=4 dim=512 n_exp=4 max_rel=4.50e-08 time=0.020 ms
[gguf] q4_0   n=16 k=512 max_rel=4.04e-07
[gguf] iq4nl  n=16 k=512 max_rel=3.46e-07
[w4a16] rows=4 n=1024 k=512 max_rel=2.48e-07 time=0.207 ms
[w4a8 ] rows=4 n=1024 k=512 max_rel=8.10e-03 time=0.130 ms
[w4a4 ] rows=4 n=1024 k=512 max_rel=9.97e-02 time=0.117 ms
全部对账通过 ✔
```

## 1. 组件

| 类 / 函数 | 作用 | 主要内核 |
|---|---|---|
| `Workspace` | 按 key 复用设备缓冲，避免逐层 alloc/free | — |
| `F32Linear` | f32 权重线性层，M=1..4 每行一次 GEMV | `gemv_f32_warp_k` |
| `DotLinear` | 量化权重原生解码 + 点积 | `*_dot_k` + `reduce_blocks_k` |
| `Int4Linear` | compressed-tensors INT4 的三条通路（W4A16 / W4A8 / W4A4） | `int4_dot_k` 或 `gemv_w4a8/44` |
| `RT4Linear` | RT4 INT4 权重（W4A8 / W4A4） | `quant_act*` / `gemv_w4a*` |
| `RMSNorm` | `rmsnorm_k`，`flag=0` 使用 `w` | `rmsnorm_k` |
| `SwiGLU` / `MLP` | 门控 MLP，kernel 连续入队 | `silu_mul_k` |
| `MoECombine` | 专家输出加权合并 | `moe_combine_k` |
| `KVCache` | f32 KV 追加（`copy_dev` 编排） | — |
| `Sampler` | 设备 argmax + 主机 top-k / top-p | `argmax_k` |

## 2. 执行模型

`Runtime` 是单队列 in-order；同一个 forward 里的 kernel 可以全部入队，最后
只 `sync()` 一次。这是 `model.py` 的核心约定：

```python
def forward_device(x_dev, rows, sync=False):
    h1 = gate.forward_device(x_dev, rows, sync=False)
    h2 = up.forward_device(x_dev, rows, sync=False)
    act = swiglu.forward_device(h1, h2, rows * gate.n, sync=False)
    out = down.forward_device(act, rows, sync=False)
    if sync:
        rt.sync()
    return out
```

对比「每个 kernel 后 sync 一次」，这样能把每个算子一次的设备同步 + Python
调度开销省掉。实测一次 launch 的固定开销约 **7 us**（Python + HSA 投递），
所以层数多、算子碎时收益明显。

完整 MLP 示例：

```python
from k100lc_kernels import Runtime, Workspace
from k100lc_kernels.model import F32Linear, MLP, RMSNorm

rt = Runtime()
ws = Workspace(rt)
norm = RMSNorm(rt, dim, w_norm, 1e-6, ws, "norm0")
mlp = MLP(rt,
          F32Linear(rt, ffn, dim, w_gate, ws, "gate0"),
          F32Linear(rt, ffn, dim, w_up,   ws, "up0"),
          F32Linear(rt, dim, ffn, w_down, ws, "down0"),
          ws, "mlp0")

x_dev = ...                       # 上一层输出的设备指针
h = norm.forward_device(x_dev, rows, sync=False)
y_dev = mlp.forward_device(h, rows, sync=True)
```

## 3. 权重格式与加载

### 3.1 f32

```python
lin = F32Linear(rt, n, k, w_f32, ws, "gate")
```

### 3.2 compressed-tensors INT4（W4A16，group 128）

直接吃 checkpoint 的 `weight_packed`（I32）+ `weight_scale`（BF16）原始字节：

```python
from k100lc_kernels import DotLinear

packed = open("weight_packed.bin", "rb").read()
scale = open("weight_scale.bin", "rb").read()
lin = DotLinear(rt, n, k, "int4", packed, (scale,), ws, "gate")
```

也可以把 f32 权重现场量化（用于实验 / 对账）：

```python
from k100lc_kernels import pack_int4_group128, dequant_int4_group128

packed, scale = pack_int4_group128(w_f32)       # w_f32: [N,K], K%128==0
ref = dequant_int4_group128(packed, scale, n, k)
```

### 3.3 GGUF 原生编码

`DotLinear` 支持 `QUANT_SPECS` 里的 11 类编码 + `int4`：

```python
from k100lc_kernels.model import QUANT_SPECS
print(sorted(QUANT_SPECS))
# ['int4','iq2s','iq3s','iq3xxs','iq4nl','iq4xs','q2_0','q4_0','q4k','q5k','q6k','q8_0']

lin = DotLinear(rt, n, k, "q4k", raw_bytes, (), ws, "ffn_gate")
# iq2s/iq3s 需要 1 张网格表；iq3xxs 需要网格表 + 符号表：
# lin = DotLinear(rt, n, k, "iq3xxs", raw, (grid_bytes, ksigns_bytes), ws, "gate")
```

### 3.4 RT4 INT4（W4A8 / W4A4）

```python
from k100lc_kernels import RT4File, W4Runner

runner = W4Runner(rt)
with RT4File("/path/model.rt4") as f:
    t = f.tensor("model.layers.0.mlp.gate_proj.weight")
    wq, ws_s = f.upload_weight(rt, t)             # W4A8/W4A4 GEMV 用 f16 尺度
    lin = RT4Linear(rt, t.shape[0], t.shape[1], wq, ws_s, "w4a8", runner)
    y = lin.forward(x)
```

预填充用 `upload_weight_gemm` + `W4Runner.gemm`（M 必须是 128 的倍数）。

> **`threads` 必须保持默认的 256。** `gemv_w4a4<M>` / `gemv_w4a8<M,*,*>` 把
> 「每个 workgroup 4 个 warp、每个 warp 一行权重」写死在代码里，grid 也按
> `ceil(N/4)` 算。传别的 workgroup 大小不会报错，但会**静默算错**（实测
> `threads=512` 时约一半的行是错的）。`gemv_device` 现在会直接拒绝非 256 的取值。

### 3.5 把 compressed-tensors INT4 切到最快通路

compressed-tensors 的 `weight_packed` 与 RT4 的 q 区在字节布局上完全一致
（低半字节 = 偶数 k），只差半字节取值（offset-binary vs 两补码）和尺度
类型（BF16 vs f16）。因此可以每层转换一次，然后直接用本卡最快的
`v_dot4_i32_i8` / `v_dot8_i32_i4` 通路：

```python
from k100lc_kernels import Int4Linear

# packed / scale 是 checkpoint 的 weight_packed(I32) / weight_scale(BF16) 原始字节
fast8 = Int4Linear(rt, n, k, packed, scale, "w4a8", ws, "gate_w4a8")
y = fast8.forward(x)          # M=1..4 解码

fast4 = Int4Linear(rt, n, k, packed, scale, "w4a4", ws, "gate_w4a4")
```

实测（`17408×5120`，M=1，合成权重）：

| 路径 | 每层 | 等效带宽 | 相对 f32 | 相对 f32 参考误差 |
|---|---:|---:|---:|---:|
| f32 | 0.699 ms | 510 GB/s | 1.00× | 9.1e-7 |
| INT4 W4A16 | 0.247 ms | 186 GB/s | 2.83× | 1.2e-1（权重 INT4） |
| INT4 W4A8 | **0.098~0.111 ms** | 414~470 GB/s | **6.1~7.2×** | 1.2e-1 |
| INT4 W4A4 | **0.089~0.095 ms** | 485~514 GB/s | **7.2~7.8×** | 1.5e-1 |

W4A16 与 W4A8 的误差都以权重 INT4 量化为主；W4A8 额外把激活量化到 int8，
在同一份权重上只多约 0.5%~0.8% 的相对误差（`examples/python_model_layer.py`
测得 `8.1e-3`），W4A4 把激活量化到 int4，误差约 1e-1。

预填充（M=128，同一形状）：

| 路径 | 每层 | 算力 | 相对 f32 |
|---|---:|---:|---:|
| f32（128 次 GEMV） | 88.6 ms | — | 1.00× |
| INT4 W4A16（128×2 次 launch） | 28.8 ms | — | 3.08× |
| INT4 + W4A4 GEMM | **0.561 ms** | 20.3 TMAC/s | **158×** |

需要更高精度时用 `Int4Linear(..., "w4a8", prefill="gemv")`：预填充走
W4A8 GEMV（约 0.6% 误差），代价是比 GEMM 慢约 5 倍。

## 4. MoE

公共的三件套：

1. `router_top10_k`：logits → top-10 专家 id + 权重；
2. 每个专家的线性层（`DotLinear` / `Int4Linear` / `RT4Linear`）；
3. 合并内核：`moe_combine_k`（稠密）或 `moe_combine_gather_k`（分桶）。

### 4.1 稠密：所有专家 × 全部行（小 batch 简单）

```python
from k100lc_kernels.model import MoECombine

rows, dim, n_exp = 4, 5120, 64
available = [0, 3, 7]                       # 本例只跑几个专家
moe = MoECombine(rt, rows, dim, len(available), ws, "moe0")
for e, expert in enumerate(selected_experts):
    expert.forward_device(x_dev, rows, out_dev=moe.slot(e), sync=False)
rt.upload(moe.weights, router_weights)          # [rows, n_sel]
y_dev = moe.forward_device(sync=True)
```

`moe_combine_k(y, exp_out, weights, n_exp, rows, dim)`：grid=rows、
workgroup=64，一个 workgroup 负责一行，lane 沿 dim 循环。`exp_out` 布局
`[n_exp, rows, dim]`，`weights` 布局 `[rows, n_exp]`。

### 4.2 分桶：token→expert（省最多 topk 倍算力）

`MoEExperts` 先按 router 结果把 token 分到各专家的桶里，每个专家只处理自己
收到的 token，再用 `moe_combine_gather_k` 按桶内位置合并：

```python
from k100lc_kernels.model import MoEExperts

moe = MoEExperts(rt, experts, rows, dim, out_dim, topk, ws, "moe")
y = moe.forward(x, router_ids, router_weights)   # ids/weights: [rows, topk]
```

底层两个内核：

| 内核 | 作用 | 启动 |
|---|---|---|
| `gather_rows_k(dst, src, idx, rows, dim)` | 按索引把 x 的行 gather 成连续桶 | grid=rows，wg=64；要求 `dim%4==0` |
| `moe_combine_gather_k(y, exp_bucket, weights, base, n_sel, rows, dim)` | 按 `base[r,e]` 合并专家输出 | grid=rows，wg=64 |

`base[r,e] = 专家桶偏移 + 桶内位置`，由主机侧 `MoEExperts.route()` 用 NumPy
一次算好；内核不做任何除法（旧版按元素切分需要魔法除，`dim=2048` 时无法
精确表示）。

实测（INT4 W4A8 专家，round-robin 路由保证每个专家桶大小一致）：

| 配置 | 稠密 | 分桶 | 加速 |
|---|---:|---:|---:|
| rows=512 dim=512 n_exp=8 topk=2 | 0.518 ms | 0.474 ms | 1.09× |
| rows=2048 dim=2048 n_exp=8 topk=2 | 4.968 ms | **1.835 ms** | **2.71×** |

分桶的收益随 batch 增大而增大：解码 M≤4 时两者算力相同，稠密路径反而少一次
gather；预填充（M 几百以上）直接用 `MoEExperts`。

## 5. KV cache

`KVCache` 用 `copy_dev` 把新 token 的 K/V 追加到 `[max_len, dim]` 槽：

```python
from k100lc_kernels.model import KVCache

cache = KVCache(rt, dim=n_kv_heads * head_dim, max_len=4096, ws=ws, tag="kv7")
cache.append_device(k_dev, v_dev, rows, sync=False)   # k_dev/v_dev 是 [rows,dim]
# cache.k / cache.v 是设备指针，cache.length 是当前长度
```

需要 int8/int4 KV 或融合 attention 时，用内核包里的 `kv_append_k_k` /
`kv_append_v_k` / `fa_decode_k` 系列，参数见
[`KERNELS.md`](KERNELS.md) 与 [`KERNEL_CALLING.md`](KERNEL_CALLING.md)。

> 局限：包里目前没有把完整 attention 层（QKV 打包 + RoPE + KV 追加 +
> FlashAttention + 输出投影）封装成一个 Python attention 类，因为各模型的
> QKV 布局与 head 配置不同；这些内核仍然可以逐个调用。

## 6. 采样

```python
from k100lc_kernels.model import Sampler

tok = Sampler.greedy(logits)                          # [rows]
tok = Sampler.sample(logits, temperature=0.8, top_k=50, top_p=0.95)
idx_dev = Sampler(rt, ws).argmax_device(logits_dev, rows, vocab)
```

## 7. 效率实践

真实形状实测（`tools/bench_model_paths.py`，`N=17408 K=5120`，
合成权重）：

| 路径 | 权重显存 | M=1 每层 | M=4 每层 | 400 层 M=1 |
|---|---:|---:|---:|---:|
| f32 | 356.5 MB | 0.699 ms | 2.799 ms | 279.6 ms/token |
| INT4 W4A16 | 46.0 MB | 0.247 ms | 0.945 ms | 98.8 ms/token |
| INT4 W4A8 | 46.0 MB | **0.098~0.111 ms** | **0.156 ms** | **约 39~44 ms/token** |
| INT4 W4A4 | 46.0 MB | **0.089 ms** | **0.152 ms** | **35.6 ms/token** |

同一份权重，W4A8 相对 f32 快 **6~7×**（M=1）/ **17.9×**（M=4），
相对原生的 W4A16 快 **2.5×**。W4A16 的等效带宽低是因为它受 4bit 解码
指令吞吐限制（见 [`INT4.md`](INT4.md) 第 5.3 节），不是访存限制；
W4A8/W4A4 用打包点积指令把解码成本摊掉。

| 实践 | 原因 |
|---|---|
| 权重只 `upload` 一次，常驻显存 | 解码是权重带宽受限，重复上传会走 PCIe |
| 用 `Workspace` 复用中间张量 | 避免逐 token / 逐层 `alloc/free` |
| 一次 forward 只 `sync` 一次 | 单队列 in-order，逐算子 sync 只是浪费 |
| 解码 M=1..4 走 GEMV；预填充 M≥128 走 GEMM | W4A4 GEMM 实测 45 TMAC/s，GEMV 是带宽通路 |
| 减少 launch 次数 | 实测每次 launch 固定开销约 7 us，400 层 × 多次很容易到毫秒级 |
| 权重量化到 INT4/RT4 | 显存与带宽直接降 4~8 倍；精度换速度 |
| MoE 用 `ids`/`stride` 一次覆盖多个专家 | 避免逐专家重复启动 |
| **整段前向只 sync 一次** | 逐层 sync 会把 CPU/GPU 串行；W4A8 17408×5120 实测每层 99.0 us → 81.4 us |

整段前向的写法（`run_sequence` 或直接手动串）：

```python
from k100lc_kernels import run_sequence

# 每层：forward_device(..., sync=False) 返回新的设备指针
y_dev = run_sequence(rt, [lambda d, l=l: l.forward_device(d, rows, sync=False)
                          for l in layers], x_dev, sync=True)
```

不要在一个 token 的层循环里调用 `layer.forward(x)`（它带 host 上传/下载与
sync）；只在最外层输入 token / 取 logits 时用一次。

## 8. 已知限制

* `rmsnorm_k` / `softmax_k` / `layernorm_k` 要求 `dim % 64 == 0`。
* `*_dot_k` 的 `rows_per_exp = 1`（每个 x 行一组）无法用 32 位魔法除表示；
  dense 层请传 `rows_per_exp = N`（`DotLinear` 已经这样做）。
* `private_segment > 0` 的 GDN / `fa_int4` / `vit_attn_kernel` **尚未验证**：
  scratch 由 ROCR 按内核自动准备（`docs/ABI.md` 第 6 节有证据），缺的是按
  语义构造合法输入的参考实现；参数猜错会 fault。
* 2D grid 的 y 维第二个 workgroup 写入不可靠；用 1D 拆行或 flat 内核。
* `KVCache` 是 f32 主机编排版；打包 KV 请用 `kv_append_*` 内核。

## 9. 验证

```bash
source env.sh
python3 examples/python_model_layer.py --rows 1
python3 examples/python_model_layer.py --rows 4 --dim 1024 --ffn 2048
```

脚本会把 f32 / INT4 MLP 与 MoE 合并分别与 NumPy 参考对账，并输出每步耗时；
相对误差大于 `1e-4` 时返回非零退出码。
