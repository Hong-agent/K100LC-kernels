# K100LC 内核编译器

把受限 Python DSL 编译成 gfx926 汇编 → HSACO。它调用本仓库的
`asm.py` / `make_hsaco_multi.py`，生成的 HSACO 可以直接用
`tools/kernel_lab.py`、`tools/hsa_job.py` 或 `Runtime`（需要重建
`nodtk_kernels.h`）执行。

## 快速使用

```bash
cd /home/t/桌面/K100LC-kernels
source env.sh
python3 -m k100lc_compiler list compiler/examples/vadd.kkl
python3 -m k100lc_compiler build compiler/examples/silu.kkl \
    -o /tmp/kbuild --emit-asm
python3 compiler/tests/test_examples.py
python3 compiler/examples/run_vadd.py     # 动态 Runtime 启动编译产物
python3 compiler/tests/test_dynamic.py
```

## DSL 子集

```python
def silu(x: ptr[f32], y: ptr[f32], n: u32):
    i = gid()
    if i < n:
        v = x[i]
        y[i] = v / (1.0 + exp(-v))
```

支持：

* 参数类型：`ptr[f32]`、`ptr[u32]`、`ptr[s32]`、`u32`、`s32`、`f32`
* 内建：`gid()`、`tid()`、`bid()`、`lane()`、`exp`、`sqrt`、`rsqrt`、
  `fma`、`fabs`、`max`；`f32()` / `u32()` / `s32()` 转换
* 语句：赋值、`+= -= *= /=`、`if`（varying 只支持无 else）、
  `for i in range(a, b)`、`break`/`continue`
* 内存：`buf[index]` load/store，下标可以是 varying 表达式
* 运算符：`+ - * /`、`& | ^ <<`、比较 `== != < <= > >=`
* `u8/u16` 指针、`load16`、`f16_to_f32`、`s8`（K-quant 解码所需）

编译出的内核使用标准隐藏 kernarg（`hidden_block_count_x`、
`hidden_group_size_x`…），所以 `gid()` 可用于跨 workgroup。

编译会同时生成 `<name>.catalog.json`；用 `Runtime(hsaco=..., catalog=...)`
加 `launch_dyn(...)` 即可直接启动，不必重建 `libfm_engine.so`。

## 后端已知规则

* `v_exp_f32` 是 **2^x**；编译器在 `exp()` 前乘 `log2(e)`。
* GCN 的 `v_exp_f32` / `v_rcp_f32` 在源/结果刚被 VALU 写入时需要
  `s_nop 0` 隔开，编译器已自动插入。
* 常量除法会折叠；除以常量会变成乘倒数。
* v1 的寄存器分配不做复用/不 spill：小中内核够用，大内核需要后续
  liveness 优化。

## 扩展

* 新内建：在 `k100lc_compiler/core.py::CodeGen.call` 加一条 emission。
* 新语句：在 `CodeGen.stmt` 加 AST 分支。
* 编译器已能生成 Q6_K/Q4_K/Q5_K 解码器（见 `examples/*.kkl`），
  `compiler/tools/export_kernel.py --install` 可直接并入内核包。
* 生成的内核要纳入 `prebuilt/k100lc_kernels.hsaco`：把生成的 `.s`
  放进 `kernels/asm/k_new/` 或写一个 `tools/gen_*.py`，再跑
  `tools/build_all.sh`。
