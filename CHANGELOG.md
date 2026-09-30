# Changelog

## 1.0.0

- 包目录/包名统一为 **K100LC-kernels**（原 `K100LC-内核`）。
- 从 `k100lc-flashmoe` 与 `K100LC-FASTASM-NVFP4` 抽出可复用内核包。
- 合并 **91 个内核**：FASTASM 81 个基线 + 本项目 10 个自研
  (gemv/dequant/IQ→i8)。
- 常驻 HSA 引擎 `libfm_engine.so`：alloc/upload/download/copy/memset、
  1D/2D launch、float 标量参数。
- Python `k100lc_kernels` 包：`Runtime` + `catalog.json` + CLI。
- C++ ABI：`runtime/fm_engine.h`。
- 独立构建验证：`bash tools/build_all.sh` 从源码重建 91 内核 HSACO、
  `libfm_engine.so`、`catalog.json`；`examples/python_gemv.py` 与
  `examples/cpp_alloc.cpp` 均已跑通。
- 新增内核编译器 `compiler/`：受限 Python DSL → gfx926 汇编 → HSACO，
  支持 `gid/tid/bid`、指针 load/store、if/for、f32/u32/s32、常用数学内建；
  `vadd/silu/axpy/loop` 已与 numpy 对账。
- 新增动态内核启动 `fm_launch_dyn` / `Runtime.launch_dyn`：运行时遍历
  HSACO 符号表，用 catalog 参数布局启动；编译器产物无需重建 libfm_engine。
- 新增 `gelu_mul_k`（GELU tanh 近似 × up）通用 FFN 内核；内核总数 92。
- 新增 `q8_0_dequant_k` 与 `q4_0_dequant_k` 主流 GGUF 量化 GPU 解码；
  内核总数 94，均与参考解码 bit-exact。
- 新增 `softmax_k`（warp-per-row + LDS 树形归约）；内核总数 95。
- 用编译器 DSL 生成 `q6k_dequant`（Q6_K → f32），与参考解码 bit-exact；
  内核总数 96。
- 用编译器 DSL 生成 `q4k_dequant` / `q5k_dequant`（Q4_K/Q5_K → f32），
  与参考解码 bit-exact；主流 GGUF K-quant 解码补齐，内核总数 98。
- 新增 `layernorm_k`（warp-per-row + 和/平方和归约）；内核总数 99。
- 新增 `topk_k`（行 top-k，KMAX=16，插入式）；内核总数 100。
- 新增 `router_top10_k`（MoE 路由 top-10 + top-10 内 softmax/renormalize，
  row-per-thread）；内核总数 101。
- 新增 **量化权重「原生解码 + 点积」融合 GEMV**：
  `iq4nl_dot_k` / `iq3xxs_dot_k` / `iq2s_dot_k` / `iq3s_dot_k` /
  `q2_0_dot_k` / `iq4xs_dot_k`（每块一项） + `reduce_blocks_k`（行归约）。
  专家权重只读原编码（4.5 bit 而不是 f32），实测比
  「解码成 f32 scratch + `gemv_f32_k`」快 **12~15 倍**；内核总数 108。
- 修掉一个真机坑：SGPR 分配恰好用满最后一段时，最高一对 SGPR 会被硬件
  改写成常量；`kernel_lab._max_registers` 现在统一多留一段 SGPR。
- 融合点积内核支持**专家索引**（`ids` + `stride`）：MoE 可直接按 router 选中的
  专家取权重，不必先把 10 个专家搬到一起。
- 新增 `q6k_dot_k`（Q6_K 原生解码 + 点积，256 权重 / 210 字节块）；内核总数 109。
  实测：比「每次解成 f32 再 gemv」快 **106 倍**，但比「f32 权重常驻显存后
  `gemv_f32_warp_k`」慢约 1.7 倍（586 GB/s vs 74 GB/s 等效）——它的价值在
  **显存只要 1/5**，适合显存吃紧时替换常驻 f32 权重。
