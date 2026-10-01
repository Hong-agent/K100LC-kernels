# K100LC 内核编译器

把受限 Python DSL 编译成 gfx926 汇编 → HSACO。它调用本仓库的
`asm.py` / `make_hsaco_multi.py`，生成的 HSACO 可以直接用
`tools/kernel_lab.py`、`tools/hsa_job.py` 或 `Runtime`（需要重建
`nodtk_kernels.h`）执行。

## 快速使用

```bash
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
  `fma`、`fabs`、`max`、`min`、`floor`、`ceil`、`trunc`、`rint`（就近偶数）、
  `fract`、`ubyte`（取整数低 8 位按无符号转 f32，解码内核常用）；
  `f32()` / `u32()` / `s32()` 转换
  * `max`/`min` 按操作数类型选指令：f32 走 `v_max_f32`（`min` 用
    `-max(-a,-b)`，编码表没有 f32 的 `v_min`），u32 走 `v_max_u32`/`v_min_u32`，
    s32 明确报错。**v1.7.1 之前不管什么类型都发 `v_max_f32`，整数会静默算错。**
* 语句：赋值、`+= -= *= /=`、`if`（**varying 条件也支持 else**）、
  `for i in range(a, b)`、`break`/`continue`
* 内存：`buf[index]` load/store，下标可以是 varying 表达式
* 运算符：`+ - * /`、`& | ^ << >>`、比较 `== != < <= > >=`（整数与 f32 都支持）
* 整数除 / 取模：**只支持 2 的幂的常量除数**（`a / 4`、`a % 8` 走移位/掩码，
  精确）；一般除数会明确报错——要么自己用 `>>`/`&` 展开，要么在主机侧算好。
  （magic 数除法需要知道取值范围，编译器不知道，宁可报错也不静默算错。）
* `u8/u16` 指针、`load16`、`f16_to_f32`、`s8`（K-quant 解码所需）

后端说明：

* **比较**：编码表里两个源都能是 VGPR 的形式很少——f32 只有
  `v_cmp_lt_f32_e64`，整数只有 `v_cmp_lt_u32_e64` / `v_cmp_gt_i32_e64` /
  `v_cmp_eq_u32_e32`。编译器统一用这几条，其余运算符靠交换操作数 /
  `s_xor_b64 vcc, vcc, -1` 取反拼出来；`==` 是「a≥b 且 b≥a」。
  均等比较用 u32 形式（两个补码下相等与符号无关）。
  因此 `>=` / `<=` 是「不小于 / 不大于」，NaN 上与有序比较不同
  （`check_f32_cmp` 覆盖 f32 的 6 个运算符，`check_int_cmp_and_divmod`
  覆盖 u32/s32 各 6 个）。
* `min(a,b)` 用 `-max(-a,-b)` 实现（编码表里没有 f32 的 `v_min`）。
* varying 条件的 `if/else` 用 exec 掩码切换：`(old & vcc)` 跑 then、
  `(old & ~vcc)` 跑 else，最后 `s_mov_b64 exec, save` 复原。

编译出的内核使用标准隐藏 kernarg（`hidden_block_count_x`、
`hidden_group_size_x`…），所以 `gid()` 可用于跨 workgroup。

编译会同时生成 `<name>.catalog.json`；用 `Runtime(hsaco=..., catalog=...)`
加 `launch_dyn(...)` 即可直接启动，不必重建 `libfm_engine.so`。

## 后端已知规则

* `v_exp_f32` 是 **2^x**；编译器在 `exp()` 前乘 `log2(e)`。
* GCN 的 `v_exp_f32` / `v_rcp_f32` 在源/结果刚被 VALU 写入时需要
  `s_nop 0` 隔开，编译器已自动插入。
* 常量除法会折叠；除以常量会变成乘倒数。
* 寄存器分配：**变量区和临时区不重叠**。编译前先扫一遍 AST，给每个命名
  变量分一个专属 VGPR（v2 起），并给 `gid()/tid()/lane()` 预留匿名槽位；
  地址对（lo/hi）紧接其后，临时值再从 `max(64, 地址对之后)` 起。这样局部
  变量能用到约 197 个（上限 v200，超出会明确报错），临时值仍在 v245 处报错。
  * v1.4.0 之前两边都从 v64 附近往上涨，局部变量超过 61 个就会**重叠**、
    生成的内核不报错但算错（`compiler/tests/test_examples.py::check_many_vars` 是回归测试）。
* **临时寄存器池（用完即归还）**：每条指令发完之后，它的操作数临时值会被
  归还到池子里（`_release`）。语句末尾整体清空（变量从不落在临时池里，
  它们有专属寄存器）。没有这个池的时候 `x[0]+x[1]+…+x[n]` 每个 `+` 都占
  一个新寄存器，60 项就把 v245 用光；现在 300+ 项也能编译，同时存活的
  临时值只有个位数（`check_long_expr` 是回归测试）。
* 仍然不做 spill：只有「同时存活的临时值」真的超过 v245 才会报错。
* `vgpr_count` 按实际用量上报（地址对不再写死在 v254/v255）。以前每个带
  访存的编译内核都声明 256 个 VGPR；现在示例内核都在 68 左右。本机实测这个
  字段对吞吐影响约 3%，主要是元数据正确性。

## 扩展

* 新内建：在 `k100lc_compiler/core.py::CodeGen.call` 加一条 emission。
* 新语句：在 `CodeGen.stmt` 加 AST 分支。
* 编译器已能生成 Q6_K/Q4_K/Q5_K 解码器（见 `examples/*.kkl`），
  `compiler/tools/export_kernel.py --install` 可直接并入内核包。
* 生成的内核要纳入 `prebuilt/k100lc_kernels.hsaco`：把生成的 `.s`
  放进 `kernels/asm/k_new/` 或写一个 `tools/gen_*.py`，再跑
  `tools/build_all.sh`。
