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
  `for i in range(a, b)`、`while cond`、`break`/`continue`
  * `while` 的条件必须是 **uniform（标量）比较**；`while 1:` 也支持
    （无条件的无限循环，靠 `break` 退出）。varying 条件会明确报错——
    那需要 loop-carried 的 exec 掩码约定，还没定。
  * **标量（uniform）变量是整波一份**：SGPR 指令不受 exec 掩码控制，所以
    一个标量变量只在**它自己那个 varying 区**里做记账才是自洽的整波语义。
    跑出这个区（或跑到嵌套的子区、别的区）去改它 → 直接报错（v1.8.2 之前
    是静默算错：`c = 0` 在区外、`if x[i] < 0: c = c + 1` 在区内，会让条件
    不成立的 lane 也一起变成 1）。要 **per-lane** 计数/累加就用 `f32`
    变量（f32 局部变量天生是 VGPR）；要整波一致的记账，就把建和改都放在
    同一个 varying 区里（`q4k/q5k/q6k_dequant` 就是这么写的）。
  * `break` / `continue` 不能出现在 varying 区的 `if` 里：分支是标量指令，
    会把**所有** lane 一起跳出/跳回（同样是 v1.8.2 起直接报错）。
  * **workgroup 共享内存（LDS）**：`sm = lds(n)` 声明 `n` 个 f32 槽（编译期
    定尺寸，可以写算式如 `lds(64 * 65)`），`sm[i]` / `sm[i] = v` 发
    `ds_read_b32` / `ds_write_b32`，`barrier()` 发 `s_waitcnt lgkmcnt(0)` +
    `s_barrier`（跨 lane 交换数据必须先写、再 `barrier()`、再读）。
    生成的 HSACO 会按实际用量声明 `group_segment`（256 B 对齐，上限 64 KB）。
    字面量下标折进 `ds_*` 的 16 位立即偏移，动态下标才花一条移位。
    * 例子：`compiler/examples/vt_scatter.kkl`（64×65 分块转置，每行 +1 填充
      把 64 路 bank 冲突拆成无冲突；实测把解码注意力的 V 转置从 2.12 ms/次
      降到 0.13 ms/次）。
    * 报错都很直接：`lds(动态表达式)`、字面量下标越界、总量超 64 KB、
      `barrier()` 当表达式用。
* 条件：比较、`and` / `or`（非短路——DSL 表达式无副作用，两侧都求值后把掩码
  在 vcc 里按位合并；uniform 与 varying 条件可以混用）
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
* **store 不逐条等**：`global_store_*` 是 fire-and-forget，同一条 lane 上
  store→load 由硬件保序，所以 `store()` 不再发 `s_waitcnt vmcnt(0)`；
  跨 lane / 跨 wave 的可见性交给 `barrier()`（它发
  `s_waitcnt lgkmcnt(0)` + `s_waitcnt vmcnt(0)` + `s_barrier`）。
  **v1.9.3 之前每条 store 后面都等一次**，于是「循环里写 64 次」就是 64 次完整
  访存往返（实测 `vt_scatter_k` 一个 workgroup 43.9 us → 20.7 us）。
* `min(a,b)` 用 `-max(-a,-b)` 实现（编码表里没有 f32 的 `v_min`）。
* varying 条件的 `if/else` 用 exec 掩码切换：`(old & vcc)` 跑 then、
  `(old & ~vcc)` 跑 else，最后 `s_mov_b64 exec, save` 复原。

编译出的内核使用标准隐藏 kernarg（`hidden_block_count_x`、
`hidden_group_size_x`…），所以 `gid()` 可用于跨 workgroup。

编译会同时生成 `<name>.catalog.json`；用 `Runtime(hsaco=..., catalog=...)`
加 `launch_dyn(...)` 即可直接启动，不必重建 `libfm_engine.so`。

## 后端已知规则

* `v_exp_f32` 是 **2^x**；编译器在 `exp()` 前乘 `log2(e)`。
* **超越函数的读-写冒险**：GCN 的 `v_exp_f32` / `v_rcp_f32` / `v_sqrt_f32` /
  `v_rsq_f32`，只要源刚被 VALU 写过、或结果立刻被后续指令读，都需要隔一条
  `s_nop 0`。编译器已在两侧自动插入。**漏一条的后果极其难查：同一个 wave 里
  只有部分 lane 算错**（实测 `y[i] = sqrt(4)` 时每 16 个元素里第 8~11 号是 0，
  看起来像随机丢数据）。回归：`check_transcendental_hazards`（9 项）。
* 常量除法会折叠；除以常量会变成乘倒数。
* **变量 spill 到 LDS**（v1.8.7）：命名变量按**使用次数**排序，最热的
  `VAR_VGPR_BUDGET`（160）个拿专属 VGPR，其余写进 per-workgroup 的 LDS 帧
  （每个 lane 一个 4 字节槽，槽基址走 `ds_*` 的立即偏移、lane 地址一条
  `v_lshlrev ... 2, v0` 全程复用）。所以「几百个局部变量」的核不会再直接
  报错，而是多花几次 LDS 访存。**约束**：spill 帧按 64 lane 布局，这样的内核
  必须以 `workgroup=64` 启动；uniform 标量不 spill（它们本来该是整波一份的
  SGPR，进 per-lane 帧会改变语义）；临时值仍然不 spill（超了还是报错）。
  回归：`check_spill`（230 个变量，对账 1.4e-07）。
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
