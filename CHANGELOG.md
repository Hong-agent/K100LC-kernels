# Changelog

## 1.2.0

INT4 推理性能强压榨：把 compressed-tensors INT4 切到 W4A8/W4A4 打包点积通路。

### 新增

- `Int4Linear`：同一份 compressed-tensors INT4 权重可选三条通路
  `w4a16`（原字节）/ `w4a8` / `w4a4`；预填充自动走 W4A4 GEMM，也可用
  `prefill="gemv"` 换更高精度。
- `quant.ct_int4_to_rt4` / `ct_int4_to_twos_complement` /
  `int4_scale_group_first_f32`：offset-binary INT4 → RT4 两补码布局 +
  f16 尺度的主机侧转换。
- `model.run_sequence`：把整段前向连续入队、只 sync 一次。
- `tools/bench_model_paths.py` 扩展为 f32 / W4A16 / W4A8 / W4A4 / 预填充
  GEMM 五路对比，并报相对 f32 参考误差。

### 实测（`N=17408 K=5120`，合成权重）

| 路径 | 权重显存 | M=1 | M=4 | 相对 f32 |
|---|---:|---:|---:|---:|
| f32 | 356.5 MB | 0.699 ms | 2.799 ms | 1.00× |
| INT4 W4A16 | 46.0 MB | 0.247 ms | 0.945 ms | 2.8× |
| INT4 W4A8 | 46.0 MB | **0.098 ms** | **0.156 ms** | **7.2× / 17.9×** |
| INT4 W4A4 | 46.0 MB | **0.089 ms** | **0.152 ms** | **7.8× / 18.5×** |
| 预填充 M=128（W4A4 GEMM） | 46.0 MB | 0.561 ms / 20.3 TMAC/s | — | 158× |

- 400 层权重流：f32 279.6 ms/token → W4A8 39.2 ms/token → W4A4
  35.6 ms/token。
- 整段前向一次 sync（而不是逐层 sync）：W4A8 每层 99.0 us → 81.4 us。
- 精度：W4A16 与原权重误差只来自 INT4 量化；W4A8 额外约 0.5~0.8%
  （激活 int8），W4A4 约 1e-1（激活 int4）。

### 验证

- `python3 examples/python_model_layer.py --rows 4 --dim 512 --ffn 1024`：
  f32 / INT4 / MoE / GGUF / W4A16 / W4A8 / W4A4 全部对账通过。
- `tools/bench_model_paths.py` 复核 M=1 / M=4 / M=128 三档数据。

## 1.1.0

模型级支持与效率优化。

### 新增

- `python/k100lc_kernels/model.py`：模型级运行时
  `Workspace` / `F32Linear` / `DotLinear` / `RT4Linear` / `RMSNorm` /
  `SwiGLU` / `MLP` / `MoECombine` / `KVCache` / `Sampler`。
  权重只上传一次，工作缓冲按 key 复用，一次 forward 只 sync 一次。
- `python/k100lc_kernels/quant.py`：主机侧 INT4 group=128 打包 / 参考解码
  （`pack_int4_group128` / `dequant_int4_group128`）。
- 新内核 `moe_combine_k`：按 router 权重合并多个专家的输出，补齐 MoE
  前向的最后一步（内核总数 120）。
- `examples/python_model_layer.py`：合成权重端到端验证 + 计时，覆盖
  f32 MLP、INT4 MLP、MoE 合并。
- `tools/bench_model_paths.py`：真实形状线性层基准（f32 vs INT4）。
- `docs/MODEL_RUNTIME.md`：模型级 API、权重加载、MoE 组装、效率实践。

### 优化

- `Runtime.launch` / `launch2d` / `launch_dyn` 快路径：缓存编码后的内核名、
  复用 argv 缓冲，单次 launch 的 Python 侧开销从约 10 us 降到约 7 us。
- `W4Runner.gemv_device` / `gemm_device` 支持 `x_dev` / `y_dev` / `c_dev` 与
  `sync=False`，可以把 W4A8/W4A4 也串进「一次 sync」的流水线。
- `DotLinear` 对 dense 层按「输出行 × 每行块数」绑定 partial，支持 M=1..4
  批量，并修正 `div_magic` 在 1 行场景下的边界问题。
- 实测 `17408×5120`：f32 0.701 ms/层、INT4 W4A16 0.240 ms/层（2.93×）；
  400 层权重流 280.6 ms/token → 95.9 ms/token。

### 验证

- `bash tools/build_all.sh`：120 内核重建通过。
- `python3 examples/python_model_layer.py`：f32 / INT4 MLP 与 NumPy 参考
  相对误差约 3.6e-7，MoE 合并约 4.5e-8。

## 1.0.0

首个单项目版本：把 gfx926（K100_LC）的可复用内核、常驻运行时、Python /
C ABI 与编译器整理成一个自包含仓库。

### 内核包

- **119 个预编译内核**（`prebuilt/k100lc_kernels.hsaco`）：NVFP4、INT4、
  f32 / int8 GEMV、预填充 GEMM、归一化 / softmax / top-k / router、
  Attention / KV、ViT、GDN / 卷积等。
- GGUF 11 类主流编码（`Q2_0 / IQ4_NL / IQ4_XS / Q4_K / Q5_K / Q6_K /
  Q8_0 / Q4_0 / IQ2_S / IQ3_S / IQ3_XXS`）具备「原生解码 + 点积」
  `*_dot_k` 内核，权重字节不落地 f32、不重量化。
- compressed-tensors INT4（W4A16，group 128）：`int4_dot_k`（128 权重块
  原生解码 + 点积）与 `int4_dequant_k`（逐位对账 / 回退）。
- RT4 INT4 的 W4A8 / W4A4 解码 GEMV（M=1..4）与预填充 GEMM：
  `quant_act` / `quant_act4` / `quant_rows_fast_k` /
  `gemv_w4a8<M,false,R>` / `gemv_w4a4<M>` / `gemm_w4a4_flat`。
- 融合点积支持专家组索引（`ids` + `stride`），dense 与 MoE 共用同一套
  语义；`reduce_blocks_k` 做行归约。

### 运行时与接口

- 无 DTK 常驻 HSA 引擎 `libfm_engine.so`：alloc / upload / download /
  copy / memset、1D/2D launch、f32 标量参数、`fm_launch_dyn`。
- Python 包 `k100lc_kernels`：`Runtime`、内核 catalog、`RT4File` /
  `W4Runner`。
- C ABI：`runtime/fm_engine.h`；内核表由 `tools/gen_kernel_table.py` 从
  HSACO metadata 生成。
- 动态启动：`Runtime.launch_dyn` / `launch_any`，新编译的 HSACO 不需要
  重建 `libfm_engine.so`。

### 编译器

- 受限 Python DSL → gfx926 汇编 → HSACO；支持 `gid/tid/bid/lane`、
  指针 load/store、if/for、f32/u32/s32 与常用数学内建。
- 编译器产物直接生成 `<name>.catalog.json`，可用 `launch_dyn` 启动。

### 文档

- `docs/KERNEL_CALLING.md`：完整的 Python / C ABI 调用手册、参数打包、
  grid 语义、逐类调用配方与排错。
- `docs/KERNELS.md`：从 catalog 自动生成的 119 内核逐参数总表。
- `docs/ABI.md`、`docs/INT4.md`、`docs/PERFORMANCE.md`。

### 已知限制

- `private_segment > 0` 的内核（`gdn_k`、`gdn_k2<32>`、`fa_int4`、
  `vit_attn_kernel`）需要 HSA queue scratch backing；当前运行时未接通。
- 一个进程只加载一个 HSACO；多个 HSACO 先用 `tools/merge_hsacos.py` 合并。
- 真机实测 2D grid 的 y 维第二个 workgroup 写入不可靠；优先使用 1D
  拆行或 1D flat 内核。
