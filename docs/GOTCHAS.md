# 已知陷阱（真机实测）

这份文件只记录**在这类硬件/工具链上真机上踩到并复现过**的坑，每条都附带验证
方式。新增条目的格式：现象 → 根因 → 怎么复现/规避。

## 1. 符号重名：合并 HSACO 时会静默覆盖

**现象**：源码改了、内核重编了，跑出来的结果却还是旧的；把内核输出改成常量
也照样返回旧值——看起来像"算错了"，其实是**跑的不是那份代码**。

**根因**：`make_hsaco_multi.build_elf()` 对符号名做 `name_offsets[name] = ...`，
**同名符号后者覆盖前者**，但两份代码都写进 `.text`。按名字查找拿到的是最后
写入的那份，它可能来自另一个目录、另一份 HSACO、或另一次不同参数的编译。

**验证 / 规避**：

```bash
python3 tools/merge_hsacos.py a.hsaco b.hsaco -o out.hsaco
# 有重名时直接报错并列出「先出现 / 又出现」的文件，退出码 1
# 确有同名覆盖的意图时加 --allow-duplicates
```

`prebuilt/k100lc_kernels.hsaco` 与 `prebuilt/k100lc_base.hsaco` 就是命中该检查
的例子（base 是 kernels 的子集，合并时 90 个内核全部重名）。

## 2. `rsqrt.approx.f32`（源码里的 `rsqrtf()`）在 gfx926 上会错

**现象**：用 nvcc 生成 PTX、再经 hyptxas 汇编出来的内核里，`rsqrtf(x)` 在
**每 16 个 lane 的第 8~11 个**返回 `inf`，其余 lane 完全正确——坏的是**lane
位置**，与输入值无关。

**根因**：hyptxas 对 `rsqrt.approx.f32` 的代码生成在这块卡上有问题。在 PTX 层
把它改写成 `sqrt.rn.f32 + div.rn.f32`（复用目的寄存器或另开临时寄存器）**都无效**，
说明问题不止那一条指令，而在寄存器分配/调度一层。

**规避**：源码里写 `1.0f / sqrtf(x)`，实测全对。

**通用诊断手法（推荐纳入自检）**：写一个**每 lane 输入都不同**的探针内核
`out[t] = f(t)`，一眼就能看出坏的是**哪些位置**。lane 位置固定、与数据无关的
错误几乎都是代码生成问题，比逐值比对快得多。

**本仓库是否受影响**：`kernels/asm/*.s` 里没有 `v_rsq*` 系列指令，自研汇编
不经过 hyptxas，因此不受这条影响。

## 3. `host_pool` 必须取自 CPU agent，否则 D2H 会读到上一次的数据

**现象**：连续多次 device→host 拷贝，读回的值里前 64~80 字节（1~2 条 cacheline）
是**上一次**传输的内容。

**根因**：K100_LC 的 GPU agent 会提供一个带 `FINE_GRAINED` 全局标志的内存池，
但该池内存的 `hsa_amd_pointer_info().agentOwner` **仍然是 GPU**——它不是 pinned
主机内存。拿它当拷贝暂存区时，D2H 实际是"显存→显存"，主机再通过 cache 读显存，
脏的 cacheline 会盖住设备刚写入的字节。

**正确做法**（本仓库 `runtime/hsa_rt.cpp::pool_alloc()` 已经这样实现）：

* `host_pool` 从 **CPU agent** 的池里取（`hsa_amd_agent_iterate_memory_pools(cpu,
  ...)`），不要用 GPU agent 报出来的那个；
* 分配之后调 `hsa_amd_agents_allow_access()`，把 GPU 与所有 CPU agent 都加进
  agent 列表，否则拷贝引擎在第一次传输时报 `Invalid address access`。

**不要**试图用 `hsa_amd_memory_lock()` 把普通内存 pin 成设备可访问：在这个
runtime 上它会**直接挂死**（进程停在 0% CPU，实测）。

## 4. `hsa_iterate_agents` 的回调必须返回 `HSA_STATUS_SUCCESS`

返回 `HSA_STATUS_INFO_BREAK` 会让整个 `hsa_iterate_agents` 调用以 0x1 错误结束；
要提前停止时也请每个 agent 都返回 SUCCESS，用自己的标志位判断。

## 5. 主机内存的写回与可见性

从设备侧看，主机内存（`pool_alloc(g_host_pool, ...)` 的产物）在拷贝完成后由
主机直接读是可靠的；但如果同一条主机代码路径既**写**过这块暂存区（H2D 的
方向）又**读**它（D2H 的方向），读回时要先让缓存行失效——**失效（invalidate）
而不是写回（flush）**：对脏行做 flush 会把主机自己的旧数据写回内存，覆盖掉
设备刚写进去的内容。
