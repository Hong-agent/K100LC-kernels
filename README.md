# K100LC-kernels

海光 **K100_LC（gfx926，wave64）** 的可复用 GPU 内核包：自研表驱动汇编器、
**139 个预编译内核**、无 DTK 的常驻 HSA 运行时、Python / C ABI 封装，以及一个
受限 Python DSL → gfx926 汇编 → HSACO 的编译器。只依赖 `/opt/hyhal` 的 HSA
运行时，**不依赖 DTK / hipcc / Docker**。

这个仓库是完整的单项目：克隆下来即可 `source env.sh` 后直接调用
`prebuilt/k100lc_kernels.hsaco` 里的内核，不需要其它本地工程。

| 文档 | 内容 |
|---|---|
| [`docs/KERNEL_CALLING.md`](docs/KERNEL_CALLING.md) | **内核调用方式（详细）**：Python / C ABI、参数打包、grid 语义、逐类调用配方、排错 |
| [`docs/KERNELS.md`](docs/KERNELS.md) | 139 个内核的逐参数总表（从 catalog 自动生成） |
| [`docs/MODEL_RUNTIME.md`](docs/MODEL_RUNTIME.md) | **模型级 API**：线性层 / RMSNorm / MLP / MoE / KV / 采样与效率实践 |
| [`docs/ABI.md`](docs/ABI.md) | kernarg 布局、动态启动、2D grid 限制 |
| [`docs/INT4.md`](docs/INT4.md) | compressed-tensors INT4（W4A16）格式与内核用法 |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | K100_LC 算力 / 带宽实测与复现方法 |

## 仓库内容

| 路径 | 内容 |
|---|---|
| `prebuilt/k100lc_kernels.hsaco` | 完整内核包（139 个内核），可直接加载 |
| `prebuilt/k100lc_base.hsaco` | 基线内核子集 |
| `prebuilt/libfm_engine.so` | 常驻 HSA 引擎（C ABI） |
| `prebuilt/nodtk_kernels.h` | 内核名 / 参数布局的 C 头（由 HSACO metadata 生成） |
| `python/k100lc_kernels/` | Python 封装：`Runtime`、内核目录、RT4 / W4A8 / W4A4 接口 |
| `runtime/` | `fm_engine.{h,cpp}` 引擎与 `hsa_rt.{h,cpp}` HSA 垫片 |
| `kernels/asm/` | 自研 gfx926 汇编 `.s` 源文件 |
| `kernels/*.h` | Attention / GEMM / GEMV 的块参数头 |
| `tools/` | 汇编、生成器、构建、catalog、自检与文档生成脚本 |
| `compiler/` | 受限 Python DSL 编译器（可直接产出可动态启动的 HSACO） |
| `examples/` | Python / C++ 调用示例 |

## 环境要求

* 海光 K100_LC（gfx926）DCU，主机可访问 `/dev/kfd`。
* `/opt/hyhal`（HSA runtime 1.11 一代）与 `libhsa-runtime64`。
* Python 3.10+、NumPy；运行时不需要 torch。
* 重新构建需要 `g++`（C++17）与 `c++filt`。

```bash
source env.sh                 # 设置 PYTHONPATH 与 LD_LIBRARY_PATH
python3 -m k100lc_kernels list
```

## 快速调用（Python）

```python
import numpy as np
from k100lc_kernels import Runtime, info

rt = Runtime()                                  # 默认加载 prebuilt/k100lc_kernels.hsaco
n, k = 256, 640
w = np.random.default_rng(0).standard_normal((n, k), dtype=np.float32)
x = np.random.default_rng(1).standard_normal(k, dtype=np.float32)

pw, px, py = rt.alloc(w.nbytes), rt.alloc(x.nbytes), rt.alloc(n * 4)
rt.upload(pw, w)
rt.upload(px, x)

# 关键：grid 是 workgroup 个数，不是总 work-item 数
rt.launch("gemv_f32_warp_k", n, 64, [pw, px, py, n, k, 64])
rt.sync()
y = rt.download(py, n, np.float32)
print(np.abs(y - w @ x).max())
print(info("gemv_f32_warp_k")["args"])          # catalog 里的参数布局
```

命令行示例：

```bash
python3 examples/python_gemv.py
python3 examples/python_int4_gemv.py --model-dir /path/to/int4-checkpoint
python3 examples/python_w4a4_w4a8_gemv.py --selftest
```

## 快速调用（C++）

```cpp
#include "fm_engine.h"

fm_init("prebuilt/k100lc_kernels.hsaco");
void* p = fm_alloc(1024);
fm_free(p);
```

```bash
g++ -O2 -std=c++17 -I runtime examples/cpp_alloc.cpp \
    -L prebuilt -lfm_engine -L/opt/hyhal/lib -lhsa-runtime64 \
    -Wl,-rpath,$PWD/prebuilt -Wl,-rpath,/opt/hyhal/lib -o /tmp/cpp_alloc
LD_LIBRARY_PATH=/opt/hyhal/lib:$LD_LIBRARY_PATH \
    /tmp/cpp_alloc prebuilt/k100lc_kernels.hsaco
```

## 内核分类（122）

| 分类 | 数量 | 说明 |
|---|---:|---|
| NVFP4 | 27 | `nvfp4_quant_act` / GEMV（M=1..4、1..4 列块） / GEMM |
| INT4（W4A4 / W4A8） | 23 | 激活量化、GEMV M=1..4、两行 GEMV、预填充 GEMM |
| 融合点积 `*_dot_k` | 12 | GGUF 11 类编码 + compressed-tensors INT4 的原生解码 + 点积 |
| 量化解码 | 12 | Q4_0 / Q8_0 / K-quant / I-quant → f32 或 i8 |
| GEMV / GEMM | 3 | f32、int8 通用 GEMV 与旧版 f32 GEMV |
| Transformer 常用算子 | 21 | RMSNorm / LayerNorm / softmax / top-k / router / MoE 合并 / rope / 激活等 |
| Attention / KV / 视觉塔 | 18 | FlashAttention、KV 写入、注意力量化、gather_rows、ViT 算子 |
| 序列模型 / 卷积 | 6 | GDN 递推、深度卷积、SSM gate、QKV 切分 |

完整清单与每个内核的显式参数见 [`docs/KERNELS.md`](docs/KERNELS.md)。

## 模型级支持

`k100lc_kernels.model` 提供可直接组合的推理算子：`F32Linear` /
`DotLinear`（INT4、GGUF 11 类编码）/ `Int4Linear`（W4A16 / W4A8 / W4A4）/
`RT4Linear` / `RMSNorm` / `SwiGLU` / `MLP` / `MoECombine` / `MoEExperts` /
`KVCache` / `Attention`（单头解码注意力）/ `FlashAttention`（**多头**，
一次前向 2 个 launch）/ `RoPE` / `Sampler` / `run_sequence`，以及把整条链串起来
的 **`TransformerLayer`**（RMSNorm → QKV → RoPE → 融合多头注意力 → 输出投影 →
残差 → RMSNorm → SwiGLU → 残差；整层设备侧串联、逐 token 一次 `sync`，
dim=512/8 头/ffn=1024 实测 **0.20 ms/token（10 个内核）**，对账 1.9e-07）。

```bash
python3 examples/python_model_layer.py --rows 4 --dim 512 --ffn 1024
python3 tools/bench_model_paths.py --n 17408 --k 5120 --rows 1 --iters 30
```

合成权重端到端验证：f32 / INT4 MLP 与 NumPy 参考相对误差 `~3.6e-7`，
MoE 合并 `~4.5e-8`，GGUF `q4_0` / `iq4nl` 路径 `~4e-7`。API、MoE 组装与效率实践见
[`docs/MODEL_RUNTIME.md`](docs/MODEL_RUNTIME.md)。

真实形状 `17408×5120` 的线性层基准（M=1，`tools/bench_model_paths.py`）：
f32 0.695 ms/层、INT4 W4A16 **0.225 ms/层（3.1×）**、INT4 W4A8
**0.102 ms/层（6.8×）**、INT4 W4A4 **0.096 ms/层（7.3×）**。

端到端 400 层权重流（连续入队、只 sync 一次，`bench_decode.py --layers 400`）：
W4A8 **33.1 ms/token（82.8 us/层，538 GB/s）**、W4A4 **29.3 ms/token
（73.1 us/层，609 GB/s）**——即峰值的 60% / 68%。

NVFP4（E2M1 权重 + E4M3 块尺度 + f32 全局尺度，int8 激活）也在包里，
同一形状解码实测 `nvfp4_gemv<2,1>` **0.114 ms/层（440 GB/s，峰值 49%）**；
模板里 `<2,1>` 比 `<1,1>` 快 1.5 倍，选模板时注意
（`tools/bench_decode.py --paths nvfp4`）。

MoE 支持两条路：`MoECombine`（所有专家 × 全部行，小 batch 简单）与
`MoEExperts`（token→expert 分桶 + `gather_rows_k` / `moe_combine_gather_k`）。
实测 `rows=2048 dim=2048 n_exp=8 topk=2`：稠密 4.94 ms → 分桶 **1.95 ms
（2.54×）**。

## 自检与基准

| 工具 | 作用 |
|---|---|
| `tools/check_all.sh` | **一条命令跑完全部验证**：重建 → 产物一致性（防止改了生成器忘重建/忘提交）→ 全内核对账 → 编译器回归 → 模型级端到端。`--quick` 跳过重建 |
| `tools/selftest_all.py` | **发货产物**（`prebuilt/` 那份 HSACO）的全内核对账基线：77 个用例覆盖约 100/139 个内核（逐元素 / 归一化 / softmax / top-k / router / MoE / GEMV / 11 类 GGUF 解码 / 融合点积 / RT4 W4A4·W4A8 / 激活量化 / NVFP4 25 个 / ViT 4 个 / `split_qkv_k`），支持 `--group`、`--only`、`--json`、`--repeat`（查非确定性） |
| `tools/size_sweep.py` | **尺寸扫描**：把对过账的内核换一批尺寸再跑一遍（这张卡上出过两次尺寸相关的坑）。已查出的限制：`split_qkv_k` 行必须 ≤ 64 |
| `tools/bench_decode.py` | 解码（M=1..4）每条权重通路的 us/层 与 GB/s |
| `tools/bench_model_paths.py` | f32 / W4A16 / W4A8 / W4A4 / 预填充 GEMM 五路对比 |
| `tools/bench_moe.py` | 稠密全专家 vs 分桶 |

各生成器（`tools/gen_*.py`）自带的自检只管「现场重新汇编出来的那份」，
`selftest_all.py` 补的是打包产物那一环。

改完任何内核 / 运行时 / 编译器后：

```bash
bash tools/check_all.sh
```

## 量化权重通路

**compressed-tensors INT4（W4A16）**：`int4_dot_k` 直接吃
`weight_packed`（I32）+ `weight_scale`（BF16，每 128 个 k 一个），不转换、
不重量化；`int4_dequant_k` 用于逐位对账与回退。格式、地址公式与实测见
[`docs/INT4.md`](docs/INT4.md)。

**RT4 INT4（W4A8 / W4A4）**：`.rt4` 权重文件 + `.rt4.json` manifest，
`k100lc_kernels.RT4File` / `W4Runner` 封装解码 GEMV 与预填充 GEMM：

```python
from k100lc_kernels import Runtime, RT4File, W4Runner

rt = Runtime()
f = RT4File("/path/model.rt4")                    # 同级 .rt4.json 自动加载
t = f.tensor("model.layers.0.mlp.gate_proj.weight")
wq, ws = f.upload_weight(rt, t)
y = W4Runner(rt).gemv("w4a8", wq, ws, x, t.shape[0], t.shape[1])

wq, wsc = f.upload_weight_gemm(rt, t)             # GEMM 用的组优先 f32 尺度
y = W4Runner(rt).gemm(wq, wsc, x, t.shape[0], t.shape[1])   # M 是 128 的倍数
```

**GGUF 原生编码**：`Q2_0 / IQ4_NL / IQ4_XS / Q4_K / Q5_K / Q6_K / Q8_0 /
Q4_0 / IQ2_S / IQ3_S / IQ3_XXS` 都有 `*_dequant_k`（→ f32）与 `*_dot_k`
（原生解码 + 点积 → `partial`）。`*_dot_k` 统一用
`reduce_blocks_k` 做行内归约。

## 构建与重新生成

```bash
bash tools/build_all.sh        # 重建 HSACO / libfm_engine.so / catalog.json，
                               # 并同步到 prebuilt/
python3 tools/gen_kernel_docs.py   # 由 catalog.json 重新生成 docs/KERNELS.md
python3 compiler/tests/test_examples.py
```

构建顺序：`build_kernels.py`（基线汇编）→ `build_native_kernels.py`（生成器
内核）→ `merge_hsacos.py`（合并）→ `gen_kernel_table.py` / `make_catalog.py`
（运行时表与 Python catalog）→ `g++` 编译常驻引擎。

## 两种 grid 语义（最容易踩坑）

| 入口 | `grid` 含义 |
|---|---|
| `Runtime.launch` / `fm_launch` / `Engine.launch` | **workgroup 个数** |
| `tools/kernel_lab.run_one` / `tools/hsa_job.py` | **总 work-item 数**（内部自己除 workgroup） |

内核侧拿到 `v0 = threadIdx.x`、`s6 = blockIdx.x`。更多细节和逐类配方见
[`docs/KERNEL_CALLING.md`](docs/KERNEL_CALLING.md)。

## 已知限制

* Catalog 里 `private_segment > 0` 的 4 个内核（`gdn_k`、`gdn_k2<32>`、
  `fa_int4`、`vit_attn_kernel`）**还没有验证过**：它们的输入语义（GDN 递推 /
  ViT attention / INT4 flash attention）在本仓库里没有参考实现，参数猜错会
  直接 fault——用 `private_segment` 设成 0 的同一组参数复现过，所以那次 fault
  是参数越界、不是 scratch。scratch 本身**不需要**用户操心：hyhal 的
  ROCR 里有完整的按内核动态分配（`AqlQueue::DynamicScratchHandler` /
  `ScratchCache` / `AcquireQueueScratch`），它按内核代码对象里的 scratch 声明
  触发；队列层 `amd_queue_t` 的 scratch 描述符一直是 0，那是正常的
  （这份运行时没有导出 `hsa_amd_queue_set_scratch_allocator` 那个外部入口）。
  新增：设 `RT_HSART_SCRATCH_INFO=1` 可以打印这几个字段，`RT_HSART_NO_SCRATCH=1`
  可以拒绝投递 `private_segment > 0` 的内核。
  这 4 个内核要能用，缺的是**按语义造合法输入并写出参考实现对账**。
* 当前 HSA 路径一个进程只加载一个 HSACO；多个 HSACO 先用
  `tools/merge_hsacos.py` 合并。
* 真机实测：这套运行时投递 2D grid 时 y 维第二个 workgroup 的写入不可靠。
  需要二维并行时优先使用 1D 扁平块号内核（例如 `gemm_w4a4_flat`）。
* 只支持 `/opt/hyhal` 的 HSA 运行时。

* `vit_bias_s_kernel` 的第 5 个参数**只有等于 `dim` 时**是「按列加 bias」
  （其它取值的输出模式没查清）；要按列加 bias 直接用 `vit_bias_kernel`。
* `concat2_k(y, a, b, pre, n)` 的第 5 个参数是**半长**：它会写 **2n** 个
  元素，输出缓冲必须给 `2n`（按 `n` 给会写穿到相邻缓冲）。

## 许可

本包自有代码 Apache-2.0（见 [`LICENSE`](LICENSE)）；i-quant 码本、掩码、
网格表与 GGUF 量化块布局来自 llama.cpp / ggml（MIT）。第三方说明见
[`NOTICE`](NOTICE)。本仓库不包含任何模型权重。
