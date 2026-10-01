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
| B3 | 解码注意力（长上下文） | 待办（已量化） | v1.7.6 先落地了一个**正确优先**的 `Attention`（两条 `gemv_f32_warp_k` + `softmax_k`，与 NumPy 对账 ~3e-7）。`dim=128` 实测：n_kv=512/2048/8192/16384/32768 → 64/129/342/565/1016 us。**瓶颈已定位**（n_kv=16384 拆解）：`softmax_k` **一 warp 一行、只有 1 个 warp 扫 16384 列 → 291 us**；`Vt·P` 只有 dim=128 个 warp → 131 us；`K·q` 有 16384 个 warp → 106 us（够用）。两条路线：① 写一个「一 warp 一行 + 4 宽批量载入」的软最大/求和核（手写汇编，参照 `gen_gemv_f32_warp.py` 的结构）；② 攻下 `fa_decode_k`/`fa_decode_comb_k` 的语义。**② 的进展**：`fa_decode_comb_k` 的第 3 个指针是 max 归约的输入已确认，但输出与任何简单公式（`Σw·opart/Σw`、`Σopart/Σl`、转置布局）都对不上，且 `l` 取 2 时直接出 inf/1e32 — 说明参数含义或布局与常规 flash-attention combine 不同，需要原项目文档 |
| B4 | 融合算子 | 待办 | RMSNorm+量化、rope+KV 写入、split_qkv+norm（后者已有 `k_gdn_split_norm_k`）等 |
| B5 | MoE 专家并行度 | 待办 | 分桶路径已有 2.71×；继续做专家内并行 / 权重常驻 |
| B6 | W4 GEMV 的 threads 约束 | 已完成 | 内核把「4 warp/组、1 行/warp」写死，非 256 会静默算错；`W4Runner` 现在直接拒绝 |
| B7 | 融合点积内核带宽 | 已完成 | 11 个融合点积全部改成批量发载入（把每个载入后的 `s_waitcnt vmcnt(0)` 合并），MoE 口径提升 **1.39~5.36 倍**；`q6k_dot_k` 是最后一个（2.35×） |
| B8 | NVFP4 解码带宽 | 待办 | 实测 `nvfp4_gemv<2,1>` 最优：114 us / 440 GB/s（49% 峰值），`<1,1>` 只有 287 GB/s——**用户该选 `<2,1>`**；离 W4 路的 667 GB/s 还有 ~1.5 倍，需要改 LLVM 生成的汇编 |

### C. 特性覆盖

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| C1 | 自检覆盖其余内核 | 进行中 | **全包约 84/122**：W4 解码全通路、激活量化 4 件套、`split_qkv_k`、NVFP4 25/27（`nvfp4_quant_act` + 24 个 gemv 模板变体）、ViT 4/7（`vit_bias`/`vit_bias_s`/`vit_gelu`/`vit_ln`）。待补：Attention/KV（18 个，全部无用例、语义要从汇编逆向）、ViT 的 `vit_rope_kernel`/`vit_linear_f16_kernel`/`vit_attn_kernel`、序列/卷积、NVFP4 的 2 个预填充 GEMM（grid 约定未定）、`quant_rows_a8_k`、`rope_k`。已探明但还没固化：`conv1d_silu_k` 是**带左移位的因果卷积**（`y[t] = silu(Σ_j w[d][j]·x[t−1+j][d] + b[d])`，x 有 T+K−1 行），t=0 那一行读的是缓冲区前的数据；`fa_decode_comb_k` 已确认第 3 个指针是 max 归约的输入，但 combine 的公式还没对上 |
| C5 | `vit_bias_s_kernel` 的第 5 个参数 | 待办（语义未查清） | 只有 `X == dim` 是干净的「按列加 bias」；X=16 时只在 `i%16 < dim` 的位置写；X=2/3/4 前 `X*dim` 个元素像 `b[i%X]` 之后就变；X=1/24 只有前 dim 个像 `b[i]`。要按列加 bias 直接用 `vit_bias_kernel`，别用这个 |
| C6 | `concat2_k` 的参数语义 | 已完成（更正） | 不是缺陷：第 5 个参数 `n` 是**半长**，内核写 `2n` 个元素；按 `n` 给输出会写穿到相邻缓冲，表现为「同参数复跑结果不同」。用例已按 2n 完整对账，README 与 `KERNEL_CALLING.md` 都补了说明 |
| C2 | 新量化格式 | 待办 | Q3_K / Q2_K / MXFP4 / FP8 等的 `*_dot_k` |
| C3 | 采样算子 | 待办 | top-p / repetition penalty 等目前只有主机侧 numpy |
| C4 | `concat2_k` 语义澄清 | 已完成 | 实测是「按 `pre` 分块交替交织」，已写进 `docs/KERNEL_CALLING.md`；如需真拼接要另加内核 |

### D. 编译器完善

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| D1 | 寄存器 liveness / 复用 | 已完成 | v1.4.0 修掉「变量区与临时区重叠」；v1.6.0 加临时寄存器池（用完即归还），长表达式 60 项 → 300+ 项；地址对不再写死 v254/v255，`vgpr_count` 256 → 68 |
| D2 | spill | 待办 | 现在只有「同时存活的临时值」超过 v245 才报错，实际很难触到；真要做 spill 得先有栈帧约定 |
| D3 | varying `if/else` | 已完成 | v1.6.2 用 exec 掩码实现（v1.6.3 修掉嵌套时 else 被跳过）；顺带修掉「f32 比较全都编不过」与 `^` 没接线 |
| D4 | 新内建 | 进行中 | 已补 `min`/`max`（类型感知）、`floor`/`ceil`/`trunc`/`rint`/`fract`/`ubyte`；运算符补了 `^`、整数比较全表、2 的幂常量 `/` `%`；后续按需加 `frexp`/`mulhi`/`bfi` 等 |
| D5 | 编译器自检 | 部分完成 | 已有 11 项（vadd/silu/axpy/loop/many_vars/long_expr/f32-比较/varying-ifelse/DSL 特性扫描/整数比较与除模/取整族内建）+ 动态启动；仍缺「生成 vs 参考」的批量回归集 |
| D6 | 后端正确性扫描 | 已完成 | 「拿文档当规格逐项对账」这个方法连查出四类问题：f32 比较全挂、整数比较 8 个运算符挂、`^` 没接线、嵌套 varying if/else 算错。已扫完并固化：一元负号、`& \| ^ << >>`、增强赋值（含 `%=` 等）、两种 `range`、嵌套 for + break 只跳内层、uniform/varying if-else、多内核单文件、`u8/u16` 指针、`load16`/`s8`/`f16_to_f32`、f32/int 比较全表、2 的幂除模、取整族内建、`and`/`or`；不支持的特性（`while`、`return`、指针赋值、链式比较、一般除数）都给出明确报错 |
| D7 | 循环语句 | 部分完成 | v1.7.4 补上 `while`（条件必须 uniform；`while 1` + break 也支持）；varying 条件的 `while` 还需要 loop-carried 掩码约定，仍待办 |

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
