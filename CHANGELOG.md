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
