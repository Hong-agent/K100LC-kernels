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
| A1 | private_segment（scratch）内核可用 | 待办 | `gdn_k`/`gdn_k2<32>`/`fa_int4`/`vit_attn_kernel` 4 个内核 `private_segment>0`；hyhal **没有** `hsa_amd_queue_set_scratch_allocator` 符号，需要验证 ROCR 是否自动分配 scratch，再决定实现方式 |
| A2 | 2D grid 的 `blockIdx.y` 可靠性 | 待办 | 真机实测第二个 y workgroup 写入不可靠；需要写一个最小复现，定位是运行时投递还是硬件/驱动 |
| A3 | 多 HSACO 并存 | 待办 | 现在一个进程只能 `fm_init` 一份 HSACO；`tools/merge_hsacos.py` 是绕路，长期应支持多份 |
| A4 | `Runtime.launch` 参数健壮性 | 已完成 | 支持 numpy 标量（`np.float32`/`np.int32`），之前直接 `TypeError` |
| A5 | LDS / `group_segment` 校验 | 待办 | 启动前校验 `group_segment + smem` 是否超过硬件上限，给出明确报错 |

### B. 模型运行效率

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| B1 | 全内核性能基线 | 待办 | 自检只对账，不出性能数字；需要一个 `tools/bench_all.py` 记录每个内核的吞吐，作为回归基准 |
| B2 | 真实 INT4 GEMM（58% 峰值） | 待办 | 权重流式是瓶颈；候选：软件预取 / 双缓冲 / 更大的 BN 复用 |
| B3 | 解码注意力（长上下文） | 待办 | `fa_decode_k` / `fa_decode_rows_k` 的 KV 访存与 GQA 复用 |
| B4 | 融合算子 | 待办 | RMSNorm+量化、rope+KV 写入、split_qkv+norm（后者已有 `k_gdn_split_norm_k`）等 |
| B5 | MoE 专家并行度 | 待办 | 分桶路径已有 2.71×；继续做专家内并行 / 权重常驻 |

### C. 特性覆盖

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| C1 | 自检覆盖其余内核 | 进行中 | 已覆盖 46 个；待补：Attention/KV、ViT、序列/卷积、NVFP4、RT4(W4A8/W4A4)、`quant_*`、`rope_k`、`embed_k`、`split_qkv_k` |
| C2 | 新量化格式 | 待办 | Q3_K / Q2_K / MXFP4 / FP8 等的 `*_dot_k` |
| C3 | 采样算子 | 待办 | top-p / repetition penalty 等目前只有主机侧 numpy |
| C4 | `concat2_k` 语义澄清 | 已完成 | 实测是「按 `pre` 分块交替交织」，已写进 `docs/KERNEL_CALLING.md`；如需真拼接要另加内核 |

### D. 编译器完善

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| D1 | 寄存器 liveness / 复用 | 待办 | v1 不做复用不 spill；`alloc_tmp_v` 单调增长（>245 直接报错），变量 VGPR 上限 127 |
| D2 | spill | 待办 | 依赖 D1 |
| D3 | varying `if/else` | 待办 | 目前 varying 分支只支持无 else |
| D4 | 新内建 | 待办 | 需要时按 `core.py::CodeGen.call` 增补 |
| D5 | 编译器自检 | 待办 | `compiler/tests/` 只有少量例子；应加「生成 vs 参考」的回归集 |

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
