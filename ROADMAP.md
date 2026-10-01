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
| B1 | 全内核性能基线 | 部分完成 | `tools/bench_decode.py` 已覆盖解码全通路；还缺 attention / norm / MoE 等族的吞吐基线 |
| B2 | 真实 INT4 GEMM（58% 峰值） | 待办 | 权重流式是瓶颈；候选：软件预取 / 双缓冲 / 更大的 BN 复用 |
| B3 | 解码注意力（长上下文） | 待办 | `fa_decode_k` / `fa_decode_rows_k` 的 KV 访存与 GQA 复用 |
| B4 | 融合算子 | 待办 | RMSNorm+量化、rope+KV 写入、split_qkv+norm（后者已有 `k_gdn_split_norm_k`）等 |
| B5 | MoE 专家并行度 | 待办 | 分桶路径已有 2.71×；继续做专家内并行 / 权重常驻 |
| B6 | W4 GEMV 的 threads 约束 | 已完成 | 内核把「4 warp/组、1 行/warp」写死，非 256 会静默算错；`W4Runner` 现在直接拒绝 |
| B7 | 融合点积内核带宽 | 基本完成 | v1.5.0 把「每个激活载入后紧跟 `s_waitcnt vmcnt(0)`」改成批量发载入：11 个融合点积里 10 个已改（MoE 口径提升 1.4~5.4 倍），只剩 `q6k_dot_k`（一次 4 条 dwordx4、ALU 密集，需先剖析） |

### C. 特性覆盖

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| C1 | 自检覆盖其余内核 | 进行中 | 已覆盖 53 个（W4 解码全通路 + 激活量化 3 件套）；待补：Attention/KV、ViT、序列/卷积、NVFP4、`quant_rows_k`/`quant_rows_a8_k`、`rope_k`、`embed_k`、`split_qkv_k` |
| C2 | 新量化格式 | 待办 | Q3_K / Q2_K / MXFP4 / FP8 等的 `*_dot_k` |
| C3 | 采样算子 | 待办 | top-p / repetition penalty 等目前只有主机侧 numpy |
| C4 | `concat2_k` 语义澄清 | 已完成 | 实测是「按 `pre` 分块交替交织」，已写进 `docs/KERNEL_CALLING.md`；如需真拼接要另加内核 |

### D. 编译器完善

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| D1 | 寄存器 liveness / 复用 | 已完成 | v1.4.0 修掉「变量区与临时区重叠」；v1.6.0 加临时寄存器池（用完即归还），长表达式 60 项 → 300+ 项；地址对不再写死 v254/v255，`vgpr_count` 256 → 68 |
| D2 | spill | 待办 | 现在只有「同时存活的临时值」超过 v245 才报错，实际很难触到；真要做 spill 得先有栈帧约定 |
| D3 | varying `if/else` | 已完成 | v1.6.2 用 exec 掩码实现（v1.6.3 修掉嵌套时 else 被跳过）；顺带修掉「f32 比较全都编不过」与 `^` 没接线 |
| D4 | 新内建 | 进行中 | 已补 `min`（用 `-max(-a,-b)`）；后续按需加 `floor/trunc/rndne`、位运算内建等 |
| D5 | 编译器自检 | 部分完成 | 已有 9 项（vadd/silu/axpy/loop/many_vars/long_expr/f32-比较/varying-ifelse/DSL 特性扫描）+ 动态启动；仍缺「生成 vs 参考」的批量回归集 |

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
