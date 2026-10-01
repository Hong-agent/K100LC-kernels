# Changelog

## 1.4.0

发货产物自检基线，以及一批实测语义修正。

### 新增

- `tools/selftest_all.py`：**针对 `prebuilt/` 那份内核包**的全内核对账基线。
  直接加载 `prebuilt/k100lc_kernels.hsaco` + `libfm_engine.so`，走用户会走的
  `Runtime.launch` 路径，与 NumPy / `tools/iq_dequant.py` 的参考实现对账。
  46 个用例覆盖逐元素 / 归一化 / softmax / top-k / router / MoE / GEMV /
  11 类 GGUF 量化解码 / 融合点积 / `reduce_blocks_k`。支持 `--group`、
  `--only`、`--repeat`、`--json`（可当性能与对账的回归基准）。
  各生成器自带的自检只管「现场重新汇编的那份」，覆盖不到打包产物。
- `ROADMAP.md`：长期推进清单（效率 / 特性 / 编译器）与当前状态。
- `tools/bench_decode.py`：解码（M=1..4）权重带宽基准，把每条通路的
  us/层与 GB/s 量出来；覆盖 W4A4/W4A8（含激活量化）、双行变体、
  compressed-tensors INT4、GGUF 原生点积、f32 参考路。

### 优化

- 运行时内核名查找改成哈希缓存（原来每次 launch 都对 122 项内核表做线性
  扫描 + `std::string` 比较）。
- `hipDeviceSynchronize` 只等「自上次同步以来投递过」的槽，不再每次把 64 个
  完成信号全扫一遍。

### 修复

- `Runtime.launch` / `_pack_argv` 支持 numpy 标量（`np.float32`、`np.int32`…）。
  之前把 `np.float32` 直接传进 argv 会 `TypeError`，从数组里取标量是很自然的写法。
- `W4Runner.gemv_device` 显式校验 `threads == 256`。`gemv_w4a4<M>` /
  `gemv_w4a8<M,*,*>` 把「每 workgroup 4 个 warp、每 warp 一行」写死在代码里，
  传别的 workgroup 大小**不报错、只是静默算错**（实测 `threads=512` 时约一半
  行是错的，且因为少算了行看起来还“快”了 1.7 倍）。

### 实测确认（写进文档与自检）

- `sigmoid_mul_k(y, a, b, n)` 是 `y = a * sigmoid(b)`——门控在**第 3 个**参数上，
  与 `silu_mul_k(y, a, b, n) = silu(a) * b` 相反，容易误用。
- `rmsnorm_gated_k(y, x, w, gate, cols, eps)` 的门控是 **SiLU**：
  `y = rms(x) * w * silu(gate)`。
- `rmsnorm_k` 第 6 个参数是 flag：`flag=0` 才是标准 RMSNorm，非 0 等价于用
  `(1+w)` 代替 `w`（`model.RMSNorm` 传的是 0）。
- `concat2_k(y, a, b, pre, n)` 是**按 `pre` 分块交替交织**，不是拼接。
- 自检新增 **w4** 组：W4A4（M=1/M=4/双行）与 W4A8 的「激活量化 + GEMV」
  全对账，参考实现直接按 `kernels/gemv_w4a4_core.h` / `gemv_w4a8_core.h`
  的公式写（组内精确 int32 点积、再乘子组尺度）。这条通路此前**零覆盖**。

### 验证

- `python3 tools/selftest_all.py`：50 个用例全部通过。
- `python3 tools/bench_decode.py --n 17408 --k 5120 --iters 60`（实测，含激活量化）：

| 通路 | us/层 | GB/s | 峰值占比 |
|---|---:|---:|---:|
| W4A4 M=1 | 84.2 | 546 | 61% |
| W4A4 M=4 | 128.8 | 357 | 40% |
| W4A4 双行 M=4 | 117.5 | 391 | 44% |
| W4A8 M=1 | 83.3 | 552 | 62% |
| W4A8 M=4 | 134.5 | 342 | 38% |
| int4_dot_k（W4A16） | 260.6 | 176 | 20% |
| iq4nl_dot_k | 466.3 | 108 | 12% |
| q4k_dot_k | 274.9 | 182 | 20% |

  连续入队（一次 sync）时 `gemv_w4a4<1>` 本体约 69 us（667 GB/s，74% 峰值），
  激活量化再加约 5 us；剩下的是启动/往返延迟与流式带宽的差距。

## 1.3.0

MoE token→expert 分桶，以及合并内核的通用化重写。

### 新增

- `gather_rows_k`：按索引把行 gather 成连续桶（一个 workgroup 一行，
  lane 用 `dwordx4` 搬 4 个 f32）。
- `moe_combine_gather_k`：按 `base[r,e]`（专家桶偏移 + 桶内位置）合并专家
  输出；`base` 由主机 NumPy 一次算好。
- `model.MoEExperts`：token→expert 分桶 MoE，缓冲复用、一次 sync。
- `tools/bench_moe.py`：稠密全专家 vs 分桶基准。

### 优化 / 修复

- `moe_combine_k` / `moe_combine_gather_k` 改成「一个 workgroup 一行、
  lane 沿 dim 循环」，去掉魔法除法参数；旧版按元素切分在 `dim=2048` 时
  魔法除无法精确表示，会直接报错。
- `MoEExperts.route()` 从 Python 逐元素循环改成 NumPy 向量化
  （排序 + `argsort` rank），`rows=512` 时路由开销从毫秒级降到微秒级。
- `F32Linear` / `DotLinear` / `Int4Linear` / `RT4Linear` 支持
  `out_dev=`，MoE 专家可以直接写进合并缓冲区，不需要额外拷贝。

### 实测

| 配置 | 稠密全专家 | 分桶 | 加速 |
|---|---:|---:|---:|
| rows=512 dim=512 n_exp=8 topk=2 | 0.518 ms | 0.474 ms | 1.09× |
| rows=2048 dim=2048 n_exp=8 topk=2 | 4.968 ms | **1.835 ms** | **2.71×** |

内核总数 122；`examples/python_model_layer.py` 增加分桶 MoE 对账，
全部通过。

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
