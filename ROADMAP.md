# K100LC-kernels 长期推进路线图

面向「无人值守持续改进」的工作清单与进度账：**AI 模型运行效率 → 特性覆盖
→ 编译器完善**。每轮改动都遵守同一条纪律：

1. 先量化（有数字或对账，不靠感觉）；
2. 改完必须过 `python3 tools/selftest_all.py`（发货产物的全内核对账基线）；
3. 更新 `CHANGELOG.md` / 相关文档；
4. `git commit`（小步提交，一次一个可验证的主题）。

## 基线（每轮开跑前先跑一遍）

```bash
source env.sh
python3 tools/selftest_all.py --json build/selftest.json   # 覆盖 prebuilt HSACO
python3 examples/python_model_layer.py --rows 4 --dim 512 --ffn 1024
python3 tools/bench_model_paths.py --n 17408 --k 5120 --rows 1 --iters 30
```

`tools/selftest_all.py` 是 1.4.0 新增的**发货产物自检**：直接加载
`prebuilt/k100lc_kernels.hsaco`，把内核跑在用户会走的那条 `Runtime.launch`
路径上，再与 NumPy / `tools/iq_dequant.py` 的参考实现对账。各生成器自带的
自检只管「现场重新汇编出来的那份」，覆盖不到打包产物。

当前覆盖：46 个用例（逐元素 / 归一化 / softmax / top-k / router / MoE /
GEMV / 量化解码 / 融合点积），全部通过。

## 工作流与状态

### A. 运行时与特性解锁

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| A1 | private_segment（scratch）内核可用 | 部分完成 | 结论修正：hyhal 的 ROCR **有**完整 scratch 实现（`AqlQueue::DynamicScratchHandler` / `ScratchCache` / `AcquireQueueScratch`），按内核代码对象按需分配；队列层描述符恒为 0 属正常。实测猜参数投 `vit_attn_kernel` 会 fault，但 `private_segment=0` 同样 fault → 是参数越界。剩：按语义给 4 个内核造合法输入 + 写参考实现 |
| A2 | 2D grid 的 `blockIdx.y` 可靠性 | 待办 | 真机实测第二个 y workgroup 写入不可靠；需要写一个最小复现，定位是运行时投递还是硬件/驱动 |
| A3 | 多 HSACO 并存 | 待办 | 现在一个进程只能 `fm_init` 一份 HSACO；`tools/merge_hsacos.py` 是绕路，长期应支持多份 |
| A4 | `Runtime.launch` 参数健壮性 | 已完成 | 支持 numpy 标量（`np.float32`/`np.int32`），之前直接 `TypeError` |
| A6 | 启动 / 同步开销 | 已完成 | 内核名查找改哈希；sync 只等本轮投递的槽。单次 launch 主机侧约 2~3 us |
| A5 | LDS / `group_segment` 校验 | 待办 | 启动前校验 `group_segment + smem` 是否超过硬件上限，给出明确报错 |

### B. 模型运行效率

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| B1 | 全内核性能基线 | 部分完成 | `tools/bench_decode.py` 已覆盖解码全通路 + **`--layers N` 端到端权重流**（400 层：W4A8 33.1 ms/token、W4A4 29.3 ms/token）；还缺 attention / norm 等族的吞吐基线 |
| B2 | 预填充 INT4 GEMM | 待办（需改内核） | `N=17408 K=5120` 实测：**M=128 20.2、M=256 30.7、M=512、38.1、M=1024 41.8 TMAC/s**（峰值 78）。大 M 稳定 ~54%，小 M 掉到 26%。原因：grid 只按 `(M/128)×(N/64)` 切，M=128 时**只有 272 个 workgroup**（120 CU 才 2.27 个/CU），延迟掩盖不住。**已否掉一个错误方案**：按 K 对半拆成两次 GEMM 并不能提高并行度（grid 与 K 无关），必须上 **split-K 内核**（部分和 + reduce）或把 BM 从 128 改小——两者都要改那个 500+ 行的手写 GEMM |
| B3 | 解码注意力（长上下文） | 已完成 | v1.8.3 把三个瓶颈各修一处，`dim=128` 端到端 n_kv=1000/4096/16384/32768 → 106/121/157/218 us（改前 132/200/417/682 us，**2.66~3.13 倍**）。① `K·q`：新内核 `gemv_f32_rows8_k`（8 行/warp、行内 8 lane 归约、每 8 行只同步 3 次），nrows=16384 k=128 由 126 us（66 GB/s）降到 **19.5 us（429 GB/s）**，65536 行时 660 GB/s；k>512 时它更慢（k=1024 是 0.53×），所以按 `k ≤ 512` 分派。② softmax：`softmax_vec_k` 单 warp 扫三趟（16K 92 us）换成分块四件套 + 输出侧 `div_scalar_k`，未归一化的 exp 直接给 `Vt·P`，省掉一趟读写。③ `Vt·P`：新内核 `gemv_f32_rows8_split_k`（split-K，`(dim/8)·nsplit` 个 workgroup、每段只扫 pad/nsplit 列）129 us → **35.3 us**。踩过的坑：`block_exp_sum_k` 的 M 误传 by-value（读到指针低 32 位；softmax 平移不变所以常常「看着对」，只有那串位恰好是极大浮点时 exp 下溢 → L=0 → NaN），已改真指针并加 3 种子 × 30 尺寸扫描。早先否掉的方案（`attn_score_part` 替 K·q 只得 1.3×、`attn_pv_part` 替 Vt·P 更慢）仍留在记录里。仍待办：`fa_decode_k` 系列的语义、KV in int8/int4 的注意力 | v1.8.5 又把 `append` 的 V 转置搬到设备侧（`vt_scatter_k`，2.12 ms/次 → 0.134 ms/次），`Attention.append` / `append_device` 都不再走主机逐列转置。 v1.8.8 再进一步：新增**融合多头**内核 `flash_dec_part_k` + `flash_dec_comb_k`，不管几个头一次前向只有 2 个 launch；4~8 头比逐头老路快 **2.7~4.6 倍**（8×128/n_kv=1：343 → 74 us），1~2 头反而更慢（分块少时延迟暴露，结论是头 ≥4 用融合路）。
| B4 | 融合算子 | 进行中 | v1.8.9 先有了**整层跑通**的基线：`TransformerLayer`（RMSNorm→QKV→RoPE→融合注意力→残差→RMSNorm→SwiGLU→残差）一次 sync 一个 token，dim=512/8头/ffn=1024 实测 **0.31 ms/token**，逐 token 对账 1.9e-07。瓶颈量出来了：一个 token 约 **16 次 launch**、相邻 launch 有依赖时每次约 19 us，而 10 MB 权重只要 ~17 us —— 下一步融 RMSNorm+量化、qkv+rope、追加KV+rope、注意力+投影、SwiGLU+down 等把次数压下来 |
| B5 | MoE 专家并行度 | 待办 | 分桶路径已有 2.71×；继续做专家内并行 / 权重常驻 |
| B6 | W4 GEMV 的 threads 约束 | 已完成 | 内核把「4 warp/组、1 行/warp」写死，非 256 会静默算错；`W4Runner` 现在直接拒绝 |
| B7 | 融合点积内核带宽 | 已完成 | 11 个融合点积全部改成批量发载入（把每个载入后的 `s_waitcnt vmcnt(0)` 合并），MoE 口径提升 **1.39~5.36 倍**；`q6k_dot_k` 是最后一个（2.35×） |
| B8 | NVFP4 解码带宽 | 待办 | 实测 `nvfp4_gemv<2,1>` 最优：114 us / 440 GB/s（49% 峰值），`<1,1>` 只有 287 GB/s——**用户该选 `<2,1>`**；离 W4 路的 667 GB/s 还有 ~1.5 倍，需要改 LLVM 生成的汇编 |
| B9 | 行和 `reduce_blocks_k` 的访存 | 已完成 | 所有 `*_dot_k` / `int4_dot_k` / 注意力 combine 的收尾步骤。原来每个 lane「一次 load + 一次 `s_waitcnt vmcnt(0)`」沿 `nbpr` 串行（nrows=17408/nbpr=80 要 **43.4 us**），v1.8.4 改成一拍 16 个 load、只等一次，尾部退回单拍循环：同一形状 **31.1 us**，nbpr 不是 16 倍数时更明显（1024/65：34.8 → 10.2 us）。加法顺序不变，输出与旧版**逐位相同**（11 组形状对过） |
| B10 | 融合多头解码注意力 | 已完成 | v1.8.8：`flash_dec_part_k`（grid = 头数×分块数，段内两趟算 (m,l,Σp·v)）+ `flash_dec_comb_k`（按 exp(m_s−M) 合并、一次除完），一次前向 **2 个 launch**（老路 8 头要 40~60 个）。K 转置存（分数阶段完全合并）、V 行主序存（输出阶段完全合并、追加退化成 copy_dev）、两个内层循环 4 宽批量发载入。实测 4~8 头 2.7~4.6 倍（8×128 n_kv=1：343→74 us），1~2 头更慢（分块并行度不足）。踩过 5 个坑（cndmask 取第二源、行块边界写成 R、V 布局、广播读自己的槽、`d` 寄存器被 load 目标覆盖导致时好时坏的越界） |

### C. 特性覆盖

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| C1 | 自检覆盖其余内核 | 进行中 | **全包约 97/136**：W4 解码全通路、激活量化 4 件套、`split_qkv_k`、NVFP4 25/27（`nvfp4_quant_act` + 24 个 gemv 模板变体）、ViT 4/7（`vit_bias`/`vit_bias_s`/`vit_gelu`/`vit_ln`）。待补：Attention/KV 里 `fa_decode_*` 那 18 个（无用例、语义要从汇编逆向；模型侧的 `Attention` 已用自研内核拼出来并对账）、ViT 的 `vit_rope_kernel`/`vit_linear_f16_kernel`/`vit_attn_kernel`、序列/卷积、NVFP4 的 2 个预填充 GEMM（grid 约定未定）、`quant_rows_a8_k`、包的 `rope_k`（模型侧已有自研且已对账的 `rope_apply_k`，见 D9）。已探明但还没固化：`conv1d_silu_k` 是**带左移位的因果卷积**（`y[t] = silu(Σ_j w[d][j]·x[t−1+j][d] + b[d])`，x 有 T+K−1 行），t=0 那一行读的是缓冲区前的数据；`fa_decode_comb_k` 已确认第 3 个指针是 max 归约的输入，但 combine 的公式还没对上 |
| C5 | `vit_bias_s_kernel` 的第 5 个参数 | 待办（语义未查清） | 只有 `X == dim` 是干净的「按列加 bias」；X=16 时只在 `i%16 < dim` 的位置写；X=2/3/4 前 `X*dim` 个元素像 `b[i%X]` 之后就变；X=1/24 只有前 dim 个像 `b[i]`。要按列加 bias 直接用 `vit_bias_kernel`，别用这个 |
| C7 | `split_qkv_k` 的行 ≤ 64 | 已完成 | v1.8.0 用编译器 DSL 重写了这个内核（`bid()` 一行一个 workgroup + lane 并行、`for` 行内循环），任意行宽都正确（扫描到 row=557）；同名同 ABI 替换，旧的 LLVM 版（`row>64` 静默算错）已删除 |
| C6 | `concat2_k` 的参数语义 | 已完成（更正） | 不是缺陷：第 5 个参数 `n` 是**半长**，内核写 `2n` 个元素；按 `n` 给输出会写穿到相邻缓冲，表现为「同参数复跑结果不同」。用例已按 2n 完整对账，README 与 `KERNEL_CALLING.md` 都补了说明 |
| C2 | 新量化格式 | 待办 | Q3_K / Q2_K / MXFP4 / FP8 等的 `*_dot_k` |
| C3 | 采样算子 | 待办 | top-p / repetition penalty 等目前只有主机侧 numpy |
| C4 | `concat2_k` 语义澄清 | 已完成 | 实测是「按 `pre` 分块交替交织」，已写进 `docs/KERNEL_CALLING.md`；如需真拼接要另加内核 |

### D. 编译器完善

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| D1 | 寄存器 liveness / 复用 | 已完成 | v1.8.7 的 spill 顺带把「按使用次数排热/冷」的启发式做进了寄存器规划（最热的 160 个变量优先保寄存器）； v1.4.0 修掉「变量区与临时区重叠」；v1.6.0 加临时寄存器池（用完即归还），长表达式 60 项 → 300+ 项；地址对不再写死 v254/v255，`vgpr_count` 256 → 68 |
| D2 | spill | 已完成 | v1.8.7：**变量**按使用次数排序，最热的 160 个进 VGPR，其余写进 per-workgroup LDS 帧（每 lane 一个 4 字节槽）。之前 230 个变量的内核直接报 「VGPR 变量区已到 v237（上限 v200）」；现在能编、结果对（230 变量求和 rel 1.4e-07），代价实测：73 个 spill 176.8 us → 123 个 spill 184.0 us（**+4.1%**，每个 spill 变量在循环里被读 24 次）。临时值仍不 spill（超 v245 报错，提示把表达式写短）；约束：spill 帧按 64 lane 布局，内核必须以 workgroup=64 启动 |
| D3 | varying `if/else` | 已完成 | v1.6.2 用 exec 掩码实现（v1.6.3 修掉嵌套时 else 被跳过）；顺带修掉「f32 比较全都编不过」与 `^` 没接线 |
| D4 | 新内建 | 进行中 | 已补 `min`/`max`（类型感知）、`floor`/`ceil`/`trunc`/`rint`/`fract`/`ubyte`；运算符补了 `^`、整数比较全表、2 的幂常量 `/` `%`；v1.8.5 补了 workgroup 共享内存（`lds(n)` / `sm[i]` / `barrier()`，见 D8）；后续按需加 `frexp`/`mulhi`/`bfi` 等 |
| D5 | 编译器自检 | 部分完成 | 已有 18 项（vadd/silu/axpy/loop/many_vars/long_expr/f32-比较/varying-ifelse/DSL 特性扫描/整数比较与除模/取整族内建/while/跨 varying 区 uniform 赋值/混合类型提升/超越函数 hazard/共享内存 LDS/RoPE/spill）+ 动态启动；仍缺「生成 vs 参考」的批量随机回归集 |
| D6 | 后端正确性扫描 | 已完成 | 「拿文档当规格逐项对账」这个方法连查出**七类**问题：f32 比较全挂、整数比较 8 个运算符挂、`^` 没接线、嵌套 varying if/else 算错、**整数常量在 f32 上下文被当位模式**（`x[i]+1` 直接丢）、**超越函数漏 `s_nop`**（部分 lane 才错）、**跨 varying 区改 uniform（SGPR）变量**（整波执行，`c=0` 在区外 + `if x[i]<0: c=c+1` 在区内 → 条件不成立的 lane 也变 1，128 个元素错 32 个）。已扫完并固化：一元负号、`& \| ^ << >>`、增强赋值（含 `%=` 等）、两种 `range`、嵌套 for + break 只跳内层、uniform/varying if-else、多内核单文件、`u8/u16` 指针、`load16`/`s8`/`f16_to_f32`、f32/int 比较全表、2 的幂除模、取整族内建、`and`/`or`、跨区标量赋值、varying 区里的 `break`/`continue`、混合类型提升、超越函数 hazard；不支持的特性（varying 条件的 `while`、`return`、指针赋值、链式比较、一般除数）都给出明确报错 |
| D7 | 循环语句 | 部分完成 | v1.7.4 补上 `while`（条件必须 uniform；`while 1` + break 也支持）；varying 条件的 `while` 还需要 loop-carried 掩码约定，仍待办。v1.8.2 起 `while` 的循环变量必须建在 uniform 上下文里——varying 区里建的标量是「整波记账」语义（跨区改直接报错），拿它当循环条件就不再是 uniform |
| D8 | 共享内存（LDS） | 已完成 | v1.8.5：`sm = lds(n)` 声明 f32 槽、`sm[i]` / `sm[i] = v` 走 `ds_read_b32` / `ds_write_b32`、`barrier()` 发 `s_waitcnt lgkmcnt(0)` + `s_barrier`；字面量下标折进 16 位立即偏移，HSACO 按用量声明 `group_segment`（256 B 对齐、上限 64 KB）。报错：非常量尺寸 / 字面量下标越界 / 超 64 KB / `barrier()` 当表达式。回归 `check_lds`（第 16 项）。第一个用户：`vt_scatter_k`（64×65 分块转置，每行 +1 填充拆 bank 冲突），把解码注意力 V 转置 2.12 ms/次 → **0.134 ms/次** |
| D9 | RoPE（旋转位置编码） | 已完成 | v1.8.6：`compiler/examples/rope_apply.kkl`（编译器生成，装进内核包）+ 运行时 `RoPE` 类。**rotate-half** 约定（成对的是 `(j, j+dim/2)`，HF Llama/GPT-NeoX）、逐位置 cos/sin 表 `[max_len, dim/2]` 常驻、逐 token 一个内核（位置直接指到表里）。6 组形状 + 3 组尺寸位**逐位相同**（max_rel=0）；decode（rows=1）一次调用 ~11 us 里 ~7 us 是启动固定开销，rows=2048 时 123 GB/s（只算 x+y；含 cos/sin 读约 300 GB/s）。包的 `rope_k` 语义仍未逆向，模型侧不再依赖它 | v1.8.9 把签名扩成 `(y, x, cs, sn, rows, dim, tsh, tbase)`：表行 = `(row >> tsh) + tbase`，于是一个 token 的所有头可以一次做完（`rows=n_heads`、`tsh=log2(n_heads)`、`tbase=位置`），prefill 多行仍然 `tsh=0`。

### E. 工程质量

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| E1 | 发货产物全内核对账 | 已完成 | `tools/selftest_all.py` |
| E2 | CI 化 | 待办 | 把自检 + 编译器测试串成一条 `tools/check_all.sh` |
| E3 | 文档同步 | 进行中 | 每轮改完同步 `KERNEL_CALLING.md` / `KERNELS.md` / `CHANGELOG.md` |

## 已知限制 / 待确认

* `rmsnorm_k` 的 `flag != 0` 分支等价于用 `(1+w)` 代替 `w`（文档已写，代码未再单独验证）。
* `sigmoid_mul_k` 与 `silu_mul_k` 的门控位置**相反**（一个在第 2 参、一个在第 3 参），
  容易被误用，已在自检里各钉一个用例。
* `gdn_k` / `gdn_k2<32>` / `fa_int4` / `vit_attn_kernel` 仍是「文档里有、真机上大概率
  fault」的状态，未纳入自检（见 A1）。
* `kernels/asm/k_new/` 里由 `compiler/tools/export_kernel.py` 覆盖生成的
  `.s`（`q4k/q5k/q6k_dequant`）是**当时编译器版本**的快照：用当前编译器重新
  导出会得到等价但寄存器分配不同的代码（v1.8.2 验证过新旧编译器输出逐字节
  一致，差异只来自历史版本的寄存器分配/比较指令形式）。要更新快照得重新
  导出并重跑 `tools/check_all.sh`。
