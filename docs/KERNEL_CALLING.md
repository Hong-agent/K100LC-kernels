# 内核调用方式详解

本文是 K100LC-kernels 的完整调用手册：从 `Runtime` 初始化、设备内存、参数
打包、grid / workgroup 语义，到每一类内核的调用配方。逐内核的参数偏移与
类型请看自动生成的 [`KERNELS.md`](KERNELS.md)。

如果只是想在自己的模型里用这些内核，先看
[`MODEL_RUNTIME.md`](MODEL_RUNTIME.md)：那里有 `DotLinear` / `MLP` /
`MoECombine` 等现成封装。

## 0. 最短路径

```python
from k100lc_kernels import Runtime

rt = Runtime()                       # 加载 prebuilt/k100lc_kernels.hsaco
p = rt.alloc(1024)                   # 设备内存
rt.upload(p, bytes(1024))            # 主机 → 设备
rt.launch("fill_k", 1, 64, [p, 1.0, 256])   # grid = workgroup 个数
rt.sync()
rt.free(p)
```

只要记住三点：

1. **`grid` 是 workgroup 个数，不是总 work-item 数**（`Runtime.launch` /
   `fm_launch` / `Engine.launch`）。
2. **`argv` 只传显式参数**，隐藏参数（`hidden_*`）由运行时按 HSACO metadata
   自动填。
3. **指针传设备地址（Python `int`）**，`f32` 标量传 Python `float`，整型标量
   传 `int`。

## 1. 调用模型

一次内核调用由四部分组成：

```
HSACO（内核机器码 + metadata）
   │  名称、参数 (offset, size, value_kind)、kernarg/LDS/private 段大小
   ▼
catalog.json（Python 侧可读的同一份布局）
   │
   ▼
libfm_engine.so（常驻 HSA 队列 + 64 槽 kernarg + 显存分配器）
   │  fm_launch / fm_launch_dyn
   ▼
GPU（gfx926，wave64）
```

* **静态路径**：内核已经在 HSACO 的编译期内核表里（`prebuilt/nodtk_kernels.h`
  由 metadata 生成）。`Runtime.launch("lookup", ...)` 直接走它。
* **动态路径**：编译器刚产出的新 HSACO。`Runtime(hsaco=..., catalog=...)` +
  `launch_dyn(...)` 在运行时按名字从 HSACO 符号表取 kernel object，参数布局
  来自 catalog，不需要重新编译 `libfm_engine.so`。

一个进程只能 `fm_init` 一个 HSACO；要把多份内核放在一起，先用
`tools/merge_hsacos.py` 合并。

## 2. 环境准备

```bash
source env.sh
python3 -m k100lc_kernels list                 # 列出内核
python3 -m k100lc_kernels list gemv            # 按子串过滤
python3 -m k100lc_kernels info gemv_f32_warp_k # 打印参数布局
```

`env.sh` 设置：

* `K100LC_KERNELS_ROOT`：仓库根目录；
* `PYTHONPATH`：`python/` 与 `compiler/`；
* `LD_LIBRARY_PATH`：`/opt/hyhal/lib:/opt/hyhal/lib64`。

主机需要能访问 `/dev/kfd`；只支持 `/opt/hyhal` 的 HSA runtime，不需要
DTK / hipcc / Docker。

## 3. 内核发现与 catalog

`python/k100lc_kernels/catalog.json` 从 HSACO metadata 生成，结构是：

```json
{
  "version": 1,
  "kernels": [
    {
      "name": "_Z9gemv_f32_kPKfS0_Pfjjj",
      "lookup": "gemv_f32_k",
      "args": [
        {"off": 0, "size": 8, "kind": "global_buffer"},
        {"off": 24, "size": 4, "kind": "by_value"},
        {"off": 32, "size": 2, "kind": "hidden_group_size_x"}
      ],
      "kernarg_size": 40,
      "group_segment": 0,
      "private_segment": 0
    }
  ]
}
```

Python 侧查询：

```python
from k100lc_kernels import kernels, info

print([k["lookup"] for k in kernels("softmax")])
k = info("softmax_k")
print(k["kernarg_size"], k["group_segment"], k["private_segment"])
print([a for a in k["args"] if not a["kind"].startswith("hidden_")])
```

字段含义：

| 字段 | 含义 |
|---|---|
| `lookup` | 调用时传的名字；模板内核已去掉空白，如 `gemv_w4a8<1,false,1>` |
| `name` | HSACO 里的 mangled 符号 |
| `args` | 参数顺序、`off` 字节偏移、`size` 字节数、`kind` |
| `kernarg_size` | kernarg 段大小（运行时按它清零并拷贝） |
| `group_segment` | LDS 字节数，>0 的内核在工作组内使用共享内存 |
| `private_segment` | scratch 字节数，由 ROCR 按内核自动准备（见第 8 节） |

`kind` 有两类：

* `by_value` / `global_buffer`：**显式参数**，调用方填；
* `hidden_*`：**隐藏参数**，运行时按 launch 配置自动填（块数、组大小、余数、
  grid 维度等）。

## 4. Python 调用详解

### 4.1 创建 Runtime

```python
from k100lc_kernels import Runtime

rt = Runtime()                                    # prebuilt 完整内核包
rt = Runtime(hsaco="build/my.hsaco",
             catalog="build/my.catalog.json")     # 新编译的 HSACO
rt = Runtime(lib="/path/libfm_engine.so")         # 自定义引擎
```

catalog 的查找顺序：

1. 显式传入的 `catalog=`；
2. `<hsaco 路径>.catalog.json`；
3. `<hsaco>.catalog.json`（把 `.catalog.json` 接在完整文件名后）；
4. 默认 HSACO 时使用包内 `python/k100lc_kernels/catalog.json`。

### 4.2 设备内存

```python
p = rt.alloc(1024)                       # 返回设备地址（Python int）
rt.upload(p, np_array)                   # NumPy 数组或 bytes
rt.upload(p, b"\x00" * 64)
y = rt.download(p, 256, np.float32)      # 返回 NumPy 数组
rt.copy_dev(dst, src, nbytes)            # 设备 → 设备
rt.memset(p, 0, 1024)                    # 注意：通过映射区写，带宽很低
rt.free(p)
```

* `alloc` 直接向 HSA 的 GPU data pool 分配，失败抛 `MemoryError`。
* `upload` 传入 NumPy 数组时按 C 连续内存搬运；传入 `bytes` 时按原样搬运。
* `download(ptr, count, dtype)` 的 `count` 是**元素个数**，不是字节数。
* 常用尺寸：

```
f32 [N,K]              N*K*4 字节
int8 [N,K]             N*K   字节
RT4 int4 q 区          N*K/2 字节（低半字节 = 偶数 k）
RT4 f16 尺度           N*(K/128)*2 字节
compressed-tensors int4 packed   N*K/2 字节（8 个码/I32）
compressed-tensors int4 scale    N*(K/128)*2 字节（BF16）
```

### 4.3 `launch`：参数与返回值

```python
rt.launch(kernel, grid, workgroup, argv)
```

| 参数 | 含义 |
|---|---|
| `kernel` | catalog 的 `lookup`（静态表按 lookup 匹配；mangled `name` 仅动态路径接受） |
| `grid` | x 方向的 **workgroup 个数** |
| `workgroup` | 每个 workgroup 的线程数；gfx926 一个 wave 是 64 |
| `argv` | 显式参数列表，顺序与 catalog 中 `kind != hidden_*` 的项一致 |

返回值：无。C 层失败时 `Runtime.launch` 抛 `RuntimeError`。参数个数不匹配、
内核不存在都会在这里报错。

### 4.4 参数打包规则

`Runtime` 把 `argv` 打包成 `uint64_t[]`，C 层按 metadata 里的 `size` 取低
`size` 字节写进 kernarg：

| 参数类型 | Python 传入 | 说明 |
|---|---|---|
| 设备指针 `T*` | `int`（`alloc` 返回值，或指针 + 字节偏移） | 8 字节 |
| `u32` / `s32` | `int` | 只取低 4 字节 |
| `u64` / `s64` | `int` | 8 字节 |
| `f32` | `float` | `struct.pack("<f")` 后按低 4 字节写入 |
| `bool` | — | 本包内核不用标量 bool；模板 bool 已在符号名里 |
| `f16` 标量 | — | 不支持，请传 `u32` 位模式 |

**显式参数必须在前、隐藏参数在后**：C 层用「遇到第一个 `kind != 0` 的参数」
作为显式参数结束标志。本包所有 HSACO 都由 `tools/` 里的构建链生成，满足这个
顺序；手写 metadata 时要注意。

内核不存在、参数个数不匹配等错误的返回值：

```
 0  成功
-1  内核不存在（fm_launch2d 查表失败）
-2  显式参数个数与 argv 长度不一致
-3  动态启动时 kernarg_size > MAX_KERNARG
```

### 4.5 grid / workgroup 语义

内核侧：

* `v0 = threadIdx.x`（workgroup 内 0..workgroup-1；wave 内 lane = `v0 & 63`）
* `s6 = blockIdx.x`
* 隐藏参数里有 `block_count_x`、`group_size_x` 等

**1D 启动**：`launch(name, grid, wg, ...)`。

* 一行一个 workgroup 的内核（`gemv_f32_k`、`gemv_f32_warp_k`、`softmax_k`、
  `layernorm_k`、`gemv_i8_k` 等）：
  `grid = 行数或输出通道数`。
* 一个 work-item 一个块的内核（`*_dot_k`、`nvfp4_gemv` 的模板变体）：
  `grid = ceil(总块数 / workgroup)`。
* grid-stride 内核（`fill_k`、`add_inplace_k`、`silu_mul_k` 等逐元素内核）：
  grid 只决定并行度，内核自己循环到 `n`。给 `1` 也能算对；要性能就按
  `grid ≈ 120 CU × 每 CU 的块数` 给。例如 `fill_k`：

```python
# 计算 n 个 f32；grid 给 1 也能算完，给大只是更快
rt.launch("fill_k", 1, 64, [p, 1.0, n])
rt.launch("fill_k", (n + 63) // 64, 64, [p, 1.0, n])
```

workgroup 建议取 64 的整数倍（256 是常用值），让每个 wave 满。

### 4.6 `launch2d` 与真机 2D grid 限制

```python
rt.launch2d(kernel, gx, gy, wx, wy, argv)
```

* `(gx, gy)` 是 x/y 方向的 workgroup 个数；`(wx, wy)` 是每组的 x/y 线程数。
* 内核侧 `s6 = blockIdx.x`，y 方向可由隐藏的 `block_count_y` / 库函数取。

**真机实测限制**：这套运行时投递 2D grid 时，y 维第二个 workgroup 的写入
不可靠；`quant_act` 会出现 scale 对、q/asum 错，`gemm_w4a4` 第二块会出 NaN。
需要二维并行时，用下面两种方式绕开：

1. 按行/按 128 行拆成多次 1D 启动 + 指针偏移（`RT4File` / `W4Runner`
   就是这么做的）；
2. 使用 1D 扁平块号内核（`gemm_w4a4_flat`）。

### 4.7 `launch_dyn` / `launch_any`：动态内核

```python
rt = Runtime(hsaco="build/my.hsaco", catalog="build/my.catalog.json")
rt.launch_dyn("my_kernel", grid_x, 1, workgroup_x, 1, [a, b, out, n])
```

`launch_dyn` 的参数布局来自 catalog 的全量 `args`（含 `hidden_*`）；运行时遍历
HSACO 符号表找 kernel object，并按 `kind` 自动填隐藏参数。

```python
rt.launch_any("my_kernel", grid, wg, argv)
```

`launch_any` 先尝试动态路径，catalog 里没有这个 lookup 时退回编译期内核表。

> `Runtime.has(name)` 只查询**编译期内核表**。刚用编译器产出、还没合并进
> `prebuilt` 的内核请直接 `launch_dyn`。

### 4.8 计时模板

launch 是异步的；计时要「只 sync 一次」，并且把魔法数、指针、主机传输都
移出计时循环：

```python
def cdiv(a, b):
    return (a + b - 1) // b

magic = div_magic(d, max_i)      # 见 tools/gen_gemv_qdot.py::div_magic
rt.upload(px, x)                 # 只上传一次
rt.launch(...)                   # 预热
rt.sync()

t0 = time.perf_counter()
for _ in range(iters):
    rt.launch(...)               # 不带 sync
rt.sync()                        # 循环结束只等一次
dt = (time.perf_counter() - t0) / iters
```

## 5. C ABI

头文件 `runtime/fm_engine.h`：

```c
int  fm_init(const char* hsaco);
void* fm_alloc(size_t n);
void  fm_free(void* p);
int  fm_upload(void* dev, const void* host, size_t n);
int  fm_download(void* host, const void* dev, size_t n);
int  fm_copy(void* dst, const void* src, size_t n);
int  fm_memset(void* dev, int v, size_t n);
int  fm_sync(void);
int  fm_launch(const char* kernel, uint32_t grid, uint32_t workgroup,
               const uint64_t* argv, int nargs);
int  fm_launch2d(const char* kernel, uint32_t gx, uint32_t gy,
                 uint32_t wx, uint32_t wy, const uint64_t* argv, int nargs);
int  fm_launch_dyn(const char* kernel, uint32_t gx, uint32_t gy,
                   uint32_t wx, uint32_t wy,
                   const uint32_t* layout, uint32_t nargs,
                   uint32_t group_size, uint32_t private_size,
                   uint32_t kernarg_size, const uint64_t* argv, int nargv);
```

C 侧和 Python 侧一样：

* `argv[i]` 是显式参数值；指针给设备地址，浮点标量要先把 `float` 的位模式
  放进 `uint64_t`；
* `grid` / `workgroup` 是工作组的个数 / 线程数；
* 编译链接：

```bash
g++ -O2 -std=c++17 -I runtime examples/cpp_alloc.cpp \
    -L prebuilt -lfm_engine -L/opt/hyhal/lib -lhsa-runtime64 \
    -Wl,-rpath,$PWD/prebuilt -Wl,-rpath,/opt/hyhal/lib -o /tmp/app
LD_LIBRARY_PATH=/opt/hyhal/lib:$LD_LIBRARY_PATH /tmp/app prebuilt/k100lc_kernels.hsaco
```

最小 GEMV 调用：

```cpp
fm_init("prebuilt/k100lc_kernels.hsaco");
float* w = (float*)fm_alloc(n * k * 4);
float* x = (float*)fm_alloc(k * 4);
float* y = (float*)fm_alloc(n * 4);
fm_upload(w, w_host, n * k * 4);
fm_upload(x, x_host, k * 4);
uint64_t argv[6] = {(uint64_t)(uintptr_t)w, (uint64_t)(uintptr_t)x,
                    (uint64_t)(uintptr_t)y, (uint64_t)n, (uint64_t)k,
                    (uint64_t)64};
fm_launch("gemv_f32_warp_k", n, 64, argv, 6);   // grid = n 个 workgroup
fm_sync();
fm_download(y_host, y, n * 4);
```

## 6. 逐类调用配方

以下 `argv` 都是**完整显式参数列表**。所有 `*_dot_k` 的完整参数偏移见
[`KERNELS.md`](KERNELS.md)。

### 6.1 逐元素与简单算子

| lookup | argv | 推荐 grid / workgroup |
|---|---|---|
| `fill_k` | `(y, value, n)` | grid-stride；`ceil(n/64)`，wg=64 |
| `add_inplace_k` | `(y, x, n)` | grid-stride；`ceil(n/64)`，wg=64 |
| `scale_mul_k` | `(y, scale, n)` | grid-stride；`ceil(n/64)`，wg=64 |
| `silu_mul_k` | `(y, a, b, n)` | `y = silu(a) * b`；grid-stride；`ceil(n/64)`，wg=64 |
| `gelu_mul_k` | `(y, gate, up, n, 64)` | `ceil(n/64)`，wg=64；第 5 个参数固定 64 |
| `sigmoid_mul_k` | `(y, x, g, n)` | `y = x * sigmoid(g)`——注意与 `silu_mul_k` 相反，门控在**第 3 个**参数上；grid-stride；`ceil(n/64)`，wg=64 |
| `concat2_k` | `(y, a, b, pre, n)` | **按 `pre` 分块交替交织**（不是单纯拼接）：`q = i//pre, r = i%pre, y[i] = (q 偶 ? a : b)[(q//2)*pre + r]`。**`n` 是半长**：`a`/`b` 各 `n` 个元素、`y` 必须给 **2n**（每个线程写 2 个，按 `n` 给会写穿到相邻缓冲）；grid = `ceil(n/64)`，wg=64 |
| `l2norm_k` | `(x, S, eps)` | grid = 行数，wg=64；原地归一化 |

```python
rt.launch("gelu_mul_k", (n + 63) // 64, 64, [py, pg, pu, n, 64])
```

### 6.2 归一化 / softmax / top-k / router

| lookup | argv | grid / workgroup | 约束 |
|---|---|---|---|
| `rmsnorm_k` | `(y, x, w, cols, eps, flag)` | grid=rows，wg=64 | `cols % 64 == 0`；实测 `flag=0` 使用 `w`，非 0 使用 `1+w` |
| `rmsnorm_gated_k` | `(y, x, w, gate, cols, eps)` | grid=rows，wg=64 | `y = rms(x) * w * silu(gate)`——门控是 **SiLU**，不是逐元素相乘 |
| `layernorm_k` | `(y, x, w, b, rows, cols, eps, 64)` | grid=rows，wg=64 | `cols % 64 == 0` |
| `softmax_k` | `(y, x, rows, cols, 64)` | grid=rows，wg=64 | `cols % 64 == 0` |
| `topk_k` | `(x, idx, val, rows, cols, k)` | `ceil(rows/64)`，wg=64 | `k <= 16` |
| `router_top10_k` | `(logits, ids, weights, rows, n_experts)` | grid=rows，wg=64 | 固定 top-10 |
| `moe_combine_k` | `(y, exp_out, weights, n_exp, rows, dim)` | grid=rows，wg=64 | 稠密专家输出加权合并 |
| `gather_rows_k` | `(dst, src, idx, rows, dim)` | grid=rows，wg=64 | 按索引 gather 行；`dim%4==0` |
| `moe_combine_gather_k` | `(y, exp_bucket, weights, base, n_sel, rows, dim)` | grid=rows，wg=64 | 分桶 MoE 合并；`base` 由主机预计算 |
| `argmax_k` | `(x, n, idx)` | grid-stride | 单行 argmax |

```python
rt.launch("rmsnorm_k", rows, 64, [py, px, pw, cols, 1e-6, flag])
rt.launch("softmax_k", rows, 64, [py, px, rows, cols, 64])
rt.launch("topk_k", (rows + 63) // 64, 64, [px, pidx, pval, rows, cols, 16])
rt.launch("router_top10_k", rows, 64, [plogits, pids, pweights, rows, n_experts])

# MoE 合并（稠密）：y[r,:] = sum_e weights[r,e] * exp_out[e,r,:]
# exp_out 布局 [n_exp, rows, dim]，grid = rows（一个 workgroup 一行）
rt.launch("moe_combine_k", rows, 64,
          [py, pexp, pweights, n_exp, rows, dim])

# 分桶 MoE：先 gather 行，再合并
# base[r,e] = 专家桶偏移 + 桶内位置（主机 NumPy 预计算）
rt.launch("gather_rows_k", total_assigned, 64,
          [px_bucket, px, pidx, total_assigned, dim])
rt.launch("moe_combine_gather_k", rows, 64,
          [py, pexp_bucket, pweights, pbase, topk, rows, dim])
```

### 6.3 f32 / int8 通用 GEMV

| lookup | 语义 | argv | grid / workgroup |
|---|---|---|---|
| `gemv_f32_k` | `y[n] = W[n,k]·x[k]`，一 lane 一行 | `(w, x, y, nrows, k/4, 64)` | grid=nrows，wg=64 |
| `gemv_f32_warp_k` | 同上，warp-per-row | `(w, x, y, nrows, k, 64)` | grid=nrows，wg=64 |
| `gemv_i8_k` | int8 权重 + f32 激活 | `(w, ws, x, y, nrows, ngroups, rowbytes, ws_stride, 64)` | grid=nrows，wg=64 |

```python
n, k = 4096, 5120
rt.launch("gemv_f32_warp_k", n, 64, [pw, px, py, n, k, 64])
```

### 6.4 量化权重「原生解码 + 点积」

这是本包的核心路径：权重不做反量化，内核直接按原编码解码并乘激活，输出
`partial[总块数]`，再由 `reduce_blocks_k` 行归约成 `y[总行数]`。

标准 `*_dot_k` 签名（不同编码的块长和表指针数量不同）：

```
w, x, partial, nblocks, group_size(64), nbpr,
magic_nbpr, rows_per_exp, magic_rpe,
[每个编码自己的表指针],
ids, stride, rows_per_w, magic_rpw
```

| 参数 | 含义 |
|---|---|
| `w` | 压缩权重首地址 |
| `x` | f32 激活，`[总x行, K]` |
| `partial` | 输出，`[总块数]` f32；行优先、行内按块号 |
| `nblocks` | 总块数 = 行数 × `nbpr`；一个 work-item 一个块 |
| `group_size` | workgroup 大小，固定传 64（与 `launch` 的 wg 一致） |
| `nbpr` | 每行的块数 = `K / 块元素数` |
| `magic_nbpr` | `i / nbpr` 的魔法乘数，用 `div_magic(nbpr, nblocks)` |
| `rows_per_exp` | x 侧每个专家组几行；dense 传总行数 |
| `magic_rpe` | `i / rows_per_exp` 的魔法乘数 |
| `ids` | 专家组索引表，u32；dense 传 `[0..rows-1]` |
| `stride` | 专家组在权重里的字节跨度 |
| `rows_per_w` | 权重侧每个专家组几行；dense 传总行数 |
| `magic_rpw` | `i / rows_per_w` 的魔法乘数 |

> `rows_per_exp = 1`（每个 x 行自成专家组）需要「除以 1 得到被除数」，
> 32 位魔法乘无法表示，这类配置当前不支持。dense 单矩阵用
> `rows_per_exp = rows_per_w = N`，此时 `ew` 恒为 0。

块布局：

| lookup | 块元素 | 块字节 | 额外表 |
|---|---:|---:|---|
| `q2_0_dot_k` | 64 | 18 | — |
| `iq4nl_dot_k` | 32 | 18 | — |
| `iq4xs_dot_k` | 256 | 136 | — |
| `q4k_dot_k` | 256 | 144 | — |
| `q5k_dot_k` | 256 | 176 | — |
| `q6k_dot_k` | 256 | 210 | — |
| `q8_0_dot_k` | 32 | 34 | — |
| `q4_0_dot_k` | 32 | 18 | — |
| `iq2s_dot_k` | 256 | 82 | `grid` |
| `iq3s_dot_k` | 256 | 110 | `grid` |
| `iq3xxs_dot_k` | 256 | 98 | `grid` + `ksigns` |
| `int4_dot_k` | 128 | 64 | `scale`（BF16 数组） |

`int4_dot_k` 的地址规则：

```
local      = (row - ew*rows_per_w)*nbpr + b
权重地址   = w     + ids[ew]*stride      + local*64
尺度地址   = scale + ids[ew]*stride/32   + local*2
```

dense 单矩阵：`ids = [0..N-1]`、`stride = nbpr*64`、
`rows_per_exp = rows_per_w = N`。

```python
from gen_gemv_qdot import div_magic   # tools/ 在 PYTHONPATH 上；也可自己实现

nbpr = k // 128
nblocks = rows * nbpr
argv = [pw, px, pp, nblocks, 64, nbpr, div_magic(nbpr, nblocks),
        rows, div_magic(rows, rows),
        pscale, pids, nbpr * 64, rows, div_magic(rows, rows)]
rt.launch("int4_dot_k", (nblocks + 63) // 64, 64, argv)
rt.launch("reduce_blocks_k", (rows + 63) // 64, 64, [pp, py, rows, nbpr])
```

`*_dot_k` 的 `grid` 建议取 `ceil(nblocks / 64)`，workgroup=64。

### 6.5 RT4：W4A8 / W4A4 解码 GEMV

> compressed-tensors INT4 的 `weight_packed` 与这里的 q 区字节布局一致，
> 只需把 offset-binary 码转成两补码、BF16 尺度转 f16，就能走同一条
> W4A8/W4A4 通路：`k100lc_kernels.quant.ct_int4_to_rt4` 做转换，
> `model.Int4Linear(..., "w4a8")` 把它包成线性层。实测 `17408×5120`
> M=1 从 W4A16 的 0.247 ms 降到 **0.098 ms**（2.5×）。

RT4 一层的权重是 `int4 q + f16 尺度`。高层封装：

```python
from k100lc_kernels import Runtime, RT4File, W4Runner

rt = Runtime()
with RT4File("/path/model.rt4") as f:
    t = f.tensor("model.layers.0.mlp.gate_proj.weight")
    wq, ws = f.upload_weight(rt, t)              # (int4 q, f16 scale)
    runner = W4Runner(rt)
    y = runner.gemv("w4a8", wq, ws, x, t.shape[0], t.shape[1])   # x [M,K], M=1..4
    y = runner.gemv("w4a4", wq, ws, x, t.shape[0], t.shape[1])
    runner.close()
```

底层调用（`M` 是 x 的行数，1..4）：

**W4A8**

```python
# 1) 激活量化：每个专家组一个 workgroup，按行拆 1D 启动（避开 2D grid）
rt.launch("quant_act", k // 128, 32,
          [px + r * k * 4, pae + r * (k // 2), pao + r * (k // 2),
           pasc + r * (k // 128) * 4, pasu + r * (k // 128) * 4, 1, k])

# 2) GEMV：R = 1（M=1）或 2（M>=2）
rt.launch("gemv_w4a8<1,false,1>", cdiv(n, (256 // 64) * r), 256,
          [wq, ws, pae, pao, pasc, pasu, py, n, k])
```

| 参数 | 含义 |
|---|---|
| `wq` | int4 权重 q 区 |
| `ws` | f16 尺度，`[N, K/128]` |
| `pae` / `pao` | 激活 int8 的偶数/奇数半 |
| `pasc` | 每组 f32 尺度 |
| `pasu` | 每组 int32 和 |
| `py` | 输出 f32 `[M,N]` |

**W4A4**

```python
rt.launch("quant_act4", k // 32, 32,
          [px + r * k * 4, paq + r * (k // 2),
           pasc + r * (k // 32) * 4, 1, k])
rt.launch("gemv_w4a4<1>", cdiv(n, 256 // 64), 256,
          [wq, ws, paq, pasc, py, n, k])
```

* `quant_act`：grid=`K/128`，wg=32（每行一次 1D 启动，`M` 行就是 M 次）。
* `quant_act4`：grid=`K/32`，wg=32。
* `gemv_w4a8<M,false,R>`：grid=`ceil(N/((threads/64)*R))`，wg=256。
  `R=1`（M=1）、`R=2`（M≥2）。带 `true` 的模板变体走 2D 路径，不使用。
* `gemv_w4a4<M>`：grid=`ceil(N/(threads/64))`，wg=256。
* `gemv_w4a4_r2_k`（M=4）/ `gemv_w4a4_r2_m3_k`（M=3）：同样 7 个参数，
  grid=`ceil(N/8)`，wg=256；每 warp 处理 2 行权重。

### 6.6 RT4：预填充 GEMM

```python
with RT4File("/path/model.rt4") as f:
    t = f.tensor("model.layers.0.mlp.gate_proj.weight")
    wq, wsc = f.upload_weight_gemm(rt, t)     # q + 组优先 f32 尺度
    y = W4Runner(rt).gemm(wq, wsc, x, t.shape[0], t.shape[1])
```

约束：`M % 128 == 0`、`N % 64 == 0`、`K % 128 == 0`。

两条底层路径：

**扁平 1D（推荐，单次 launch 覆盖全网格）**

```python
nq = k // 128
rt.launch("quant_rows_fast_k", m, 256,
          [paq, pasc, px, k, 128, m, k, 0, nq])
grid = (n // 64) * (m // 128)
rt.launch("gemm_w4a4_flat", grid, 256,
          [paq, wq, pasc, wsc, pc, m, n, k, n // 64, div_magic(n // 64, grid)])
```

| lookup | argv | grid / wg |
|---|---|---|
| `quant_rows_k` | `(q, sc, x, K, G, rows, in_stride, in_off)` | grid=rows，wg=组大小 |
| `quant_rows_fast_k` | `(q, sc, x, K, G, rows, in_stride, in_off, NG)` | grid=rows，wg=256 |
| `gemm_w4a4_flat` | `(aq, wq, asc, wsc, c, M, N, K, nbn, magic)` | grid=nbn*(M/128)，wg=256 |
| `gemm_w4a4` | `(aq, wq, asc, wsc, c, M, N, K)` | grid=N/64，wg=256；每次只算 128 行 |

`nbn = N/64`；`magic = div_magic(nbn, grid)`，内核用
`(flat*magic)>>32` 还原 `flat / nbn`。

### 6.7 NVFP4

NVFP4 内核由固定模板生成，显式参数类型从符号名即可读出：

```text
nvfp4_quant_act(float const* x, s8* a, s8* b, float* scale, int M, int K)
nvfp4_gemv<A,B>(u32 const* wq, u8 const* ws, s8 const* a, s8 const* b,
                float const* scale, float* y, int N, int K, float gscale)
nvfp4_gemv_wide<A,B>(同上)
nvfp4_gemm_kernel(u32 const* wq, u8 const* ws, s8 const* a, s8 const* b,
                  float const* scale, float* c, int M, int N, int K, float gscale)
```

NVFP4 的块是 16 个值（E2M1 4bit + E4M3 逐块尺度 + 一个 f32 全局尺度），
所以 `nvfp4_quant_act` 的 `grid = ceil(K/16)`；`nvfp4_gemv*` 的
`workgroup=256`、每个 workgroup 处理 4 行，`grid = ceil(N/4)`。模板参数
`<A,B>` 决定每 lane 的行数 / 每轮列宽；本包不提供 NVFP4 的高层 Python 封装，
底层调用请对照 catalog 参数表与 `kernels/asm/k_nvfp4/` 的汇编。

```python
rt.launch("nvfp4_quant_act", cdiv(k, 16), 64,
          [px, pa, pb, pscale, 1, k])
rt.launch("nvfp4_gemv<1,1>", cdiv(n, 4), 256,
          [pwq, pws, pa, pb, pscale, py, n, k, global_scale])
```

### 6.8 Attention / KV / 序列模型 / 视觉塔

这组内核的参数已经全部列在 [`KERNELS.md`](KERNELS.md) 的对应分类里。
调用前的两条硬性检查：

1. `private_segment > 0` 的内核（`gdn_k`、`gdn_k2<32>`、`fa_int4`、
   `vit_attn_kernel`）**还没有验证**：scratch 由 hyhal 的 ROCR 按内核自动准备
   （见 [`ABI.md`](ABI.md) 第 6 节），真正缺的是按语义构造合法输入的参考实现；
2. `group_segment` 很大的内核（`fa_*`、`gemm_w4a4`、`vit_attn_kernel`）
   需要 LDS 能放下，调用方不要随意改 workgroup。

常用的三个：

```python
# QKV 切分：split_qkv_k(yq, yk, yv, x, T, qn, kn, vn)
#   行宽任意（v1.8.0 起内核由本仓库编译器生成，带行内并行循环）；旧版在
#   qn+kn+vn > 64 时会静默算错，升级后不再有此限制。
rt.launch("split_qkv_k", T, 64, [pq, pk, pv, px, T, qn, kn, vn])
# KV 写入：kv_append_k_k / kv_append_v_k
rt.launch("kv_append_k_k", grid, 64, pkv, pstate, px, ...)
# ViT LayerNorm
rt.launch("vit_ln_kernel", grid, 64, [py, px, pw, pb, rows, cols, eps])
```

## 7. 用编译器写新内核并启动

```bash
python3 -m k100lc_compiler build compiler/examples/silu.kkl \
    -o /tmp/kbuild --emit-asm
```

产物是 `/tmp/kbuild/silu.hsaco` + `/tmp/kbuild/silu.catalog.json`。直接启动：

```python
rt = Runtime(hsaco="/tmp/kbuild/silu.hsaco",
             catalog="/tmp/kbuild/silu.catalog.json")
rt.launch_dyn("silu", cdiv(n, 64), 1, 64, 1, [px, py, n])
```

要把它并进默认内核包：

```bash
python3 tools/merge_hsacos.py prebuilt/k100lc_kernels.hsaco /tmp/kbuild/silu.hsaco \
    -o build/merged.hsaco
python3 tools/make_catalog.py build/merged.hsaco build/merged.catalog.json
```

编译器语言子集、ABI 与已知规则见 [`compiler/README.md`](../compiler/README.md)。

## 8. 排错

| 现象 | 原因 | 处理 |
|---|---|---|
| `RuntimeError: fm_launch(x) 失败` | 参数个数不对 / 内核不存在 | `info(x)["args"]` 数一遍显式参数；确认 lookup 拼写 |
| 结果全 0 或部分行错 | grid 当成总 work-item 数传了 | `Runtime.launch` 的 grid 是 workgroup 个数 |
| 第二行 / 第二块出错 | 用了 2D grid | 改成按行拆 1D 或 1D flat 内核 |
| 进程 fault | 内核 `private_segment > 0` | 那是参数越界（这 4 个内核还没有参考实现可对参数）；scratch 本身由 ROCR 按内核准备，不需要用户接 |
| 结果随 grid 变化 | grid-stride 内核的边界判断依赖 `n` | 检查 `n` 与缓冲区大小；grid 只影响并行度 |
| 除法结果不对 | `magic_*` 不是精确魔法数 | 用 `div_magic(d, max_i)`，且 `max_i` 覆盖最大下标 |
| 新内核 `launch` 找不到 | 还在新 HSACO 里，没合并进 prebuilt | 用 `launch_dyn` 或先 `merge_hsacos.py` |

`private_segment > 0` 的完整列表可以直接从 catalog 查：

```bash
python3 - <<'PY'
import json
for k in json.load(open("python/k100lc_kernels/catalog.json"))["kernels"]:
    if k["private_segment"]:
        print(k["lookup"], k["private_segment"])
PY
```

## 9. 最小验证清单

```bash
source env.sh
python3 -m k100lc_kernels list | wc -l        # 120
python3 examples/python_gemv.py               # max_abs 应在 1e-4 量级
python3 tools/gen_kernel_docs.py              # 重新生成 docs/KERNELS.md
python3 compiler/tests/test_examples.py
```

新加内核后，按 `tools/build_all.sh` 重建，再跑一次上面的清单即可。
