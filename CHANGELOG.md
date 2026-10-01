# Changelog

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
