# Changelog

## 1.7.3

视觉塔（ViT）小算子的语义查清并纳入对账；顺带查出一个**真实内核缺陷**。

### 新增（自检覆盖）

| 内核 | 语义 | 结果 |
|---|---|---|
| `vit_bias_kernel(y, b, n, dim)` | `y[i] += b[i % dim]`（**累加**，不是覆盖） | 逐位一致 |
| `vit_bias_s_kernel(y, b, n, dim, period)` | `y[i] += b[i % period]` | 见下面的缺陷 |
| `vit_gelu_kernel(y, x, n, 64)` | **精确 erf 形式**的 GELU（不是 tanh 近似） | 与 erf 差 4.8e-7；与 tanh 近似差 4e-4，足以区分 |
| `vit_ln_kernel(y, x, w, b, rows, dim, eps)` | 标准 LayerNorm，**grid = rows、wg = 64**（与 `layernorm_k` 同约定） | 4.8e-7 |

### 查出缺陷：`vit_bias_s_kernel` 的 `period != dim` 会算错

`i % period` 用的是范围有限的魔法除法，实测（`dim=16, n=256`）：

| period | 2 | 3 | 4 | 5 | 8 | 16（=dim） |
|---|---:|---:|---:|---:|---:|---:|
| 第一个错的下标 | 24 | 24 | 48 | 40 | 96 | 全对 |

也就是说**只有 `period == dim`（最常见的「按列加 bias」用法）是可靠的**，
其它周期在大 `n` 下从某个下标起开始算错。用例只覆盖可靠用法，缺陷记在
ROADMAP（要修得动那段手写汇编的除法序列）。

自检用例 57 → 61，**全包覆盖约 84/122**。

## 1.7.2

NVFP4 通路（27 个内核，本包最大的一族）的入口语义查清并纳入对账。

### 新增（自检覆盖）

用探针把 NVFP4 的格式逼出来，再用主机参考逐位/逐项对账：

| 内核 | 语义 | 结果 |
|---|---|---|
| `nvfp4_quant_act` | 块 = 16，`s = amax/127`（amax=0 时取 1），码 = `clamp(rint(x/s), -128, 127)` 的 **int8**；`a`/`b` 是偶/奇下标两条 int8 流 | **逐位一致** |
| `nvfp4_gemv<A,B>` / `nvfp4_gemv_wide<A,B>` | `y[n] = gscale · Σ_k E2M1(w)·E4M3(ws[n,k/16])·(q_act[k]·asc[k/16])`；权重 8 个 4bit 码/u32、低半字节 = 更小的 k；`ws` 是 E4M3 块尺度 | **24 个模板变体全部通过**（A=1..4 × B=1..4，wide 是 A=1..4 × B=1..2），`grid = ceil(N/A)`、`wg = 256` |

参考实现写在 `tools/selftest_all.py` 里（`e2m1_decode` / `e4m3_decode` /
`nvfp4_quant_ref` / `pack_nvfp4_codes`），`E4M3` 按 OCP FP8 解码
（1 符号 + 4 指数偏置 7 + 3 尾数）。

至此 NVFP4 这一族 **27 个内核里 25 个已验证**（只剩两个预填充 GEMM 变体）。
自检用例 55 → 57（用例数少了但覆盖面大了：24 个 gemv 变体合并成一个扫描用例），
**全包覆盖数约 80/122**。

### 待办

`nvfp4_gemm_kernel`（预填充变体）的 grid 约定还没对上：M=128/N=64/K=512 时
按 `(M/128)×(N/64)` 猜的 grid 算出 rel=1.0，还需要原项目文档或更多探针。

### 实测（NVFP4 解码带宽，`tools/bench_decode.py --paths nvfp4`）

`N=17408 K=5120`，权重口径 = `wq + ws` 字节数：

| 内核 | us/层 | GB/s | 峰值占比 |
|---|---:|---:|---:|
| `nvfp4_gemv<1,1>` | 174.4 | 287 | 32% |
| `nvfp4_gemv<2,1>` | **114.0** | **440** | **49%** |
| `nvfp4_gemv<4,1>` | 137.0 | 366 | 41% |

同一形状下 W4A8/W4A4 的 GEMV 本体是 ~667 GB/s（74%）——NVFP4 这条路的
**最优模板是 `<2,1>`**，比 `<1,1>` 快 1.5 倍；离 W4 路还有 ~1.5 倍空间，
但那是 LLVM 生成的汇编，改动成本高，先记在路线图里。

## 1.7.1

编译器内建扩充，顺带修掉 `max`/`min` 对整数的静默错算。

### 修复

- **`max` / `min` 对整数会静默算错**：原来不管操作数类型都发
  `v_max_f32_e32`，也就是说 `max(a, b)` 在两个 u32 上是**按浮点比较位模式**。
  现在按类型选：f32 → `v_max_f32`（`min` 用 `-max(-a,-b)`），
  u32 → `v_max_u32` / `v_min_u32`，s32 明确报错（编码表里没有 i32 版本）。

### 新增

取整族与解码常用内建（都能一条指令实现，`ceil` 是两条）：

| 内建 | 指令 |
|---|---|
| `floor(x)` | `v_floor_f32_e32` |
| `ceil(x)` | `-floor(-x)`（编码表没有 `v_ceil`） |
| `trunc(x)` | `v_trunc_f32_e32` |
| `rint(x)` | `v_rndne_f32_e32`（就近偶数，与 `lrintf` 一致） |
| `fract(x)` | `v_fract_f32_e32` |
| `ubyte(x)` | `v_cvt_f32_ubyte0_e32`（取低 8 位按无符号转 f32） |

### 验证

- `compiler/tests/test_examples.py::check_math_builtins`：6 个取整族内建 +
  `ubyte` + u32 `max`/`min`，逐项与 NumPy 对账（`rint` 比的是就近偶数语义）。
- 编译器回归 10 → 11 项；`bash tools/check_all.sh` 全绿。

### 自检覆盖

- 新增 `quant_rows_k`（旧版行量化）。与 `quant_rows_fast_k` 输出**逐位一致**
  （文档是这么说的，实测确认）。约定是 `grid = rows`、**workgroup 必须等于
  K/G（组数）**——给大了 `sc` 会被写坏（实测 `wg=64` 时 scale 全是垃圾，
  而 `q` 却还对，是个不小的坑），已写进用例与文档。用例总数 54 → 55。

### 语句 / 条件

- **`and` / `or` 支持**（原来直接报「不支持的表达式 BoolOp」）。DSL 表达式
  没有副作用，所以两边都求值、把各自的掩码在 vcc 里按位合并（不是短路求值，
  结果一致）；uniform 条件（在 SCC）会先用 `s_cselect_b64` 展成 64 位掩码，
  因此 `if n > 4 and x[i] > 2` 这种混用也可以。
- `compiler/tests/test_examples.py::check_dsl_surface` 从 4 项扩到 6 项
  （加 `and` 两 varying、`or` 混合类型）。

## 1.7.0

融合点积最后一块：`q6k_dot_k` 批量预载，MoE 口径 **2.35 倍**；11 个
`*_dot_k` / `int4_dot_k` 至此全部改完。

### 优化

`q6k_dot_k` 的内层循环原来每个 `l0` 跟一条 `s_waitcnt vmcnt(0)`（整块 16 次
全排空）。改成**一次处理两个 `l0`（相差 4）**、量化字节与激活一起发出、
只等一次：寄存器上量化字节用 v20..v25、激活 dwordx4 用 v96..111（第一个
`l0`）与 v112..127（第二个），都在原本空闲的区间里。

### 实测（`N=17408 K=5120`，`iters=20`）

| 口径 | 改前 | 改后 |
|---|---:|---:|
| MoE（`rpe=4`，x 89 MB） | 2468 us / 29.6 GB/s | **1052 us / 69.5 GB/s（2.35×）** |
| 稠密（x 常驻） | — | 661 us / 110.6 GB/s |

11 个融合点积的最终成绩（MoE 口径，改前 → 改后）：

| 内核 | 倍数 | 内核 | 倍数 |
|---|---:|---|---:|
| `iq4xs_dot_k` | 5.36× | `q5k_dot_k` | 3.12× |
| `q4k_dot_k` | 3.31× | `iq2s_dot_k` | 2.63× |
| `iq4nl_dot_k` | 2.59× | `iq3s_dot_k` | 2.59× |
| `q6k_dot_k` | 2.35× | `q4_0_dot_k` | 2.36× |
| `iq3xxs_dot_k` | 2.32× | `q8_0_dot_k` | 1.79× |
| `q2_0_dot_k` | 1.58× | `int4_dot_k` | 1.39× |

### 验证

- `tools/selftest_all.py --group dot`：13 个用例全过（数值与改前一致）。
- `bash tools/check_all.sh` 全绿；`tools/bench_decode.py` 的 GGUF 组加入
  `q6k_dot_k`。

### 新增

- 自检新增 **seq** 组：`split_qkv_k`（把 `[qn+kn+vn]` 的一行拆成 q/k/v）。
  语义用探针实测确认后写进用例，可以整行逐位对账；用例总数 53 → 54。

## 1.6.4

编译器：整数比较又查出 8 个运算符编不过，顺带补上 2 的幂常量除/取模。

### 修复

- **整数比较**：编码表里两个源都能是 VGPR 的形式只有
  `v_cmp_lt_u32_e64` / `v_cmp_gt_i32_e64` / `v_cmp_eq_u32_e32` 三条，而编译器
  原来按运算符直译，于是 **u32 的 `<= >= !=`、s32 的 `<= > >= == !=` 全部
  编不过**（`v_cmp_le_u32_e32 vcc, v65, v64` → no encoding form）。
  现在统一用上面三条，其余靠交换操作数 + `s_xor_b64 vcc, vcc, -1` 取反拼；
  `==` / `!=` 用 u32 形式（两个补码下相等与符号无关）。
- uniform（标量条件）的 `!=` 原来直接报「暂不支持」，现在用 `s_cmp_lg_u32`。
- 增强赋值原来只认 `+= -= *= /=`，遇到 `%= &= |= ^= <<= >>=`、`%=` 会
  `KeyError` 崩掉；现在支持全部并在不支持时给出明确报错。

### 新增

- **`%` 运算符**与**整数 `/`**：只支持 2 的幂常量除数（`a / 4` → 右移、
  `a % 8` → 掩码），精确；其它除数明确报错而不是静默算错——magic 数除法
  需要知道取值范围，编译器不知道。

### 验证

- `compiler/tests/test_examples.py::check_int_cmp_and_divmod`：u32/s32 各 6 个
  比较运算符 + 2 的幂除/取模，逐项与 NumPy 对账；非 2 的幂除数断言报错。
- `bash tools/check_all.sh` 全绿（编译器回归 9 → 10 项）。

## 1.6.3

编译器：DSL 特性扫描（拿文档当规格逐项对账）又查出两处。

### 修复

- **嵌套 varying `if/else` 算错**（上一条刚加的 else 支持有 bug）。当时的写法
  在「有任一 lane 满足 then」时用 `s_branch` 跳过了 else——同一个 wave 里
  不满足条件的 lane 就什么都没执行（实测 `x=[0.5]` 该得 1.0 却得 0）。
  现在两条路径都跑、各自用 exec 掩码，并且**条件另存一对 SGPR**：
  then 里的地址计算会用 `vcc` 做进位输出，直接依赖 vcc 一定会被冲掉。
- **`^`（异或）没有接线**：`binop` 里早就有 `v_xor_b32_e32`，但
  `expr()` 的运算符映射漏了 `ast.BitXor`，所以文档写着支持、实际
  `不支持的运算符 BitXor`。

### 新增

- `compiler/tests/test_examples.py::check_dsl_surface`：把文档里写的 DSL
  特性（一元负号、`& | ^ >>`、增强赋值、两种 `range` 写法和嵌套 varying
  if/else）逐项与 NumPy 对账。这一次的两处问题就是它扫出来的。

### 验证

- `compiler/tests/test_examples.py` 9 项全过；`bash tools/check_all.sh` 全绿。

## 1.6.2

编译器：f32 比较本来就是坏的（编不过）、varying `if/else`、`min`。

### 修复

- **所有 f32 比较都编不过**。`v_cmp_*_f32_e32` 的第一个源在编码表里是
  `ssrc`（标量/内联常量），两个 VGPR 相比时汇编器直接
  `no encoding form for v_cmp_gt_f32_e32 vcc, v3, v64` —— 而编译器总是把两边
  都转成 VGPR，所以 `< <= > >= == !=` 一个都不能用（既有测试恰好只用到
  整数比较，一直没暴露）。现在统一发 `v_cmp_lt_f32_e64 vcc, x, y`，
  其余运算符用交换操作数 / `s_xor_b64 vcc, vcc, -1` 取反、`==` 用
  「a≥b 且 b≥a」拼出来。`compiler/tests/test_examples.py::check_f32_cmp`
  逐个运算符对账。
  * 副作用：`>=` / `<=` 实现为「不小于 / 不大于」，NaN 时与有序比较不同。

### 新增

- **varying 条件的 `if/else`**（原来直接报「暂不支持」）。用 exec 掩码切换：
  `(old & vcc)` 跑 then，`(old & ~vcc)` 跑 else，最后复原 exec。
- **`min(a,b)`** 内建。编码表里没有 f32 的 `v_min`（只有 `v_max`/`v_max3`），
  用 `-max(-a,-b)` 精确实现。

### 验证

- `compiler/tests/test_examples.py`：8 项回归（vadd / silu / axpy / loop /
  many_vars / long_expr / f32 比较 6 运算符 / varying if-else + min-max）全过。
- `bash tools/check_all.sh` 全绿。

## 1.6.1

把「scratch 内核不能跑」这条已知限制查清楚了，文档按证据改写。

### 结论修正

原来四处文档都写着「`private_segment > 0` 的 4 个内核（`gdn_k`、
`gdn_k2<32>`、`fa_int4`、`vit_attn_kernel`）因为运行时没接 scratch 所以会
fault」。实测链条：

1. 队列层 `amd_queue_t` 的 `scratch_backing_memory_*` 与
   `scratch_resource_descriptor` **一直是 0**（`hsa_queue_create` 之后、
   投递 `private_segment=4096` 的内核之后都是），而且 hyhal 的头文件与导出
   符号里都没有 `hsa_amd_queue_set_scratch_allocator`。→ 看起来像"没接"。
2. 但 `libhsa-runtime64.so` 里**有完整的 scratch 实现**：
   `rocr::AMD::AqlQueue::DynamicScratchHandler<true/false>`、
   `ScratchCache::alloc/use_reserved`、`AcquireQueueScratch`、
   `InitScratchSRD`。它是**按内核**在投递时准备的（触发条件是内核代码对象
   声明的 scratch 需求），所以队列层描述符恒为 0 是正常的。
3. 因此"描述符为 0"不能当作"不能跑"的依据。

### 新增 / 修改

- `RT_HSART_SCRATCH_INFO=1`：打印队列的 scratch 字段（定位这类问题用）。
- `RT_HSART_NO_SCRATCH=1`：显式拒绝投递 `private_segment > 0` 的内核。
  默认**不拦**——按内核分配由 ROCR 负责，默认拦会误伤。
- 若带 scratch 的内核真的 fault，会给出明确提示（而不是静默 core dump）。
- `README.md` / `docs/ABI.md` / `docs/KERNEL_CALLING.md` /
  `docs/MODEL_RUNTIME.md` 的已知限制改写为：这 4 个内核**尚未验证**，
  缺的是「按语义造合法输入 + 参考实现对账」。

### 实测

猜参数直接投 `vit_attn_kernel` 会 `Invalid address access` 并 core dump；
把同一组参数的 `private_segment` 改成 0 **同样** fault —— 说明那次 fault 来自
参数越界（这 4 个内核没有参考实现，参数只能猜），不是 scratch。

## 1.6.0

编译器：临时寄存器池（语句内 liveness 复用）+ 寄存器元数据收口。

### 优化 / 修复

- **临时值不再单调递增**。原来 `alloc_tmp_v` 只涨不还，`tmp_v` 一路走到
  v245——`y[0] = x[0] + x[1] + … + x[n]` 这种长表达式每个 `+` 占一个新
  寄存器，**60 项就报「VGPR 溢出」**（其实同时只有两三个是活的）。
  现在每条指令发完都会把操作数临时值归还到池子（`_release`），语句末尾
  整体清空。300+ 项表达式正常编译，数值逐位正确
  （`compiler/tests/test_examples.py::check_long_expr` 是回归测试）。
- **地址对不再写死在 v254/v255**。它现在紧跟变量区（两个连续寄存器），
  `vgpr_count` 按真实用量上报：示例内核从 **256 降到 67~68**。
  实测这个声明字段对本机吞吐影响约 3%（358.8 us vs 368.6 us），
  属元数据正确性修复，不是提速。

### 验证

- `bash tools/check_all.sh`：重建 + 产物一致性 + 53 个内核对账用例 +
  编译器 6 项回归 + 模型级端到端，全部通过。

## 1.5.0

融合点积（GGUF 原生解码 / compressed-tensors INT4）的访存重写：
**MoE 口径 1.4~3.3 倍**。

### 优化

问题：`*_dot_k` 与 `int4_dot_k` 里，激活的 `global_load_dwordx4` 后面紧跟着
一条 `s_waitcnt vmcnt(0)`，而 `s_waitcnt vmcnt(0)` 的语义是「等**所有**
未完成的全局载入」。一个 work-item 于是把 8~64 次访存**串行**等下来
（`q4k_dot_k` 77 条载入配 70 条全排空），几乎没有访存并行度。

改法：把一轮要用的激活**一次性发出去，最后只等一次**。

| 内核 | 载入 | 全排空（改前→改后） |
|---|---:|---:|
| `iq4nl_dot_k` | 26 | 12 → 2 |
| `q4_0_dot_k` | 26 | 10 → 2 |
| `q8_0_dot_k` | 42 | 10 → 2 |
| `q4k_dot_k` | 77 | 70 → 14 |
| `q5k_dot_k` | 79 | 70 → 14 |
| `q2_0_dot_k` | 34 | 18 → 2 |
| `iq4xs_dot_k` | 85 | 66 → 10 |
| `iq2s_dot_k` | 178 | 138 → 18 |
| `iq3s_dot_k` | 238 | 166 → 22 |
| `iq3xxs_dot_k` | 258 | 170 → 18 |
| `int4_dot_k` | 38 | 18 → 6 |

`iq2s`/`iq3s`/`iq3xxs`/`iq4xs` 这类一个 work-item 要跑 8 个子块的，
按**子块**批量预载（128 B = 8 条 dwordx4）后再统一等待。

实测（`N=17408 K=5120`，单层权重，`iters=20`）：

| 内核 | 稠密口径（x 常驻） | | MoE 口径（`rpe=4`，x 89 MB） | |
|---|---:|---:|---:|---:|
| | 改前 | 改后 | 改前 | 改后 |
| `iq4nl_dot_k` | 466 us | 354 us（1.32×） | 902 us | **348 us（2.59×）** |
| `q4_0_dot_k` | — | — | 820 us | **347 us（2.36×）** |
| `q8_0_dot_k` | — | — | 1163 us | **651 us（1.79×）** |
| `q4k_dot_k` | 275 us | 265 us（1.04×） | 1560 us | **472 us（3.31×）** |
| `q5k_dot_k` | — | — | 1658 us | **531 us（3.12×）** |
| `iq2s_dot_k` | — | — | 3358 us | **1276 us（2.63×）** |
| `iq3s_dot_k` | — | — | 5165 us | **1997 us（2.59×）** |
| `iq3xxs_dot_k` | — | — | 1771 us | **762 us（2.32×）** |
| `iq4xs_dot_k` | — | — | 1858 us | **346 us（5.36×）** |
| `q2_0_dot_k` | — | — | 728 us | **459 us（1.58×）** |
| `int4_dot_k` | 277 us | 251 us（1.11×） | 470 us | **337 us（1.39×）** |

稠密口径提升有限（激活本来就在 L1 里），MoE 口径是收益所在——也正是
`MoEExperts` 走的那条路。

### 验证

- `python3 tools/selftest_all.py`：53 个用例全部通过（融合点积 13 个用例的
  数值与改前一致）。
- `python3 examples/python_model_layer.py`：全部对账通过；`w4a16` 每层
  0.205 ms → 0.169 ms。

### 待办

`q6k_dot_k` 仍是老写法：它的内层循环一次要 4 条 dwordx4（已经批量到「一轮
只等一次」），再深一层需要两级软件预取；而且它是 ALU 密集型（每个元素
五六条位操作），不是纯访存瓶颈，得先做剖析再动。见 `ROADMAP.md` B7。

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
- `tools/check_all.sh`：一条命令跑完全部验证——重建 → **产物一致性**
  （重建后的 `prebuilt/`、catalog、`docs/KERNELS.md` 必须与仓库一致，
  防止「改了生成器忘了重建 / 忘了提交产物」）→ 全内核对账 → 编译器回归 →
  模型级端到端。`--quick` 跳过重建。
- `tools/bench_decode.py`：解码（M=1..4）权重带宽基准，把每条通路的
  us/层与 GB/s 量出来；覆盖 W4A4/W4A8（含激活量化）、双行变体、
  compressed-tensors INT4、GGUF 原生点积、f32 参考路。

### 优化

- 运行时内核名查找改成哈希缓存（原来每次 launch 都对 122 项内核表做线性
  扫描 + `std::string` 比较）。
- `hipDeviceSynchronize` 只等「自上次同步以来投递过」的槽，不再每次把 64 个
  完成信号全扫一遍。

### 修复

- **编译器：变量寄存器区和临时寄存器区重叠**。两边原来都从 v64 附近往上涨，
  局部变量超过 61 个就开始撞在一起——同一个寄存器既放变量又当临时值，
  生成的内核**不报错但静默算错**。现在编译前先扫一遍 AST：命名变量拿专属
  VGPR（v2 起），`gid()/tid()/lane()` 的匿名槽位按调用点预留，临时值从
  `max(64, 变量区末尾)` 起，两边严格不重叠。可用变量数从 61 提到 ~197，
  超出会明确报错；`compiler/tests/test_examples.py` 增加 `many_vars` 回归。
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
- 自检新增 **quant** 组：`quant_rows_fast_k`、`quant_act4`、`quant_act`
  三个激活量化内核与主机参考**逐位一致**（码字与尺度都要求 bit-exact）。

### 验证

- `python3 tools/selftest_all.py`：53 个用例全部通过。
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
