# ABI 与调用约定

本文只讲二进制接口和运行时约定。逐类调用配方见
[`KERNEL_CALLING.md`](KERNEL_CALLING.md)，120 个内核的逐参数表见
[`KERNELS.md`](KERNELS.md)。

## 1. 内核目录（catalog）

`python/k100lc_kernels/catalog.json` 由 `tools/make_catalog.py` 从 HSACO
metadata 生成，结构：

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

| 字段 | 含义 |
|---|---|
| `lookup` | 传给 `Runtime.launch()` / `fm_launch()` 的名字；模板实参已去掉空白 |
| `name` | HSACO 里的 mangled 符号 |
| `args` | 参数顺序、`off`、`size`、`kind` |
| `kernarg_size` | kernarg 段大小 |
| `group_segment` | LDS 字节数 |
| `private_segment` | scratch 字节数；>0 的内核当前运行时无法直接跑 |

`C ABI` 侧的同一份表是 `prebuilt/nodtk_kernels.h`，由
`tools/gen_kernel_table.py` 生成。

## 2. kernarg 布局

每个参数在 kernarg 段里有固定的 `(offset, size, kind)`：

| kind | 编号 | 由谁填 |
|---|---:|---|
| `by_value` / `global_buffer` | 0 | 调用方的 `argv` |
| `hidden_block_count_x/y/z` | 1 / 2 / 3 | 运行时按 grid 填 |
| `hidden_group_size_x/y/z` | 4 / 5 / 6 | 运行时按 workgroup 填 |
| `hidden_remainder_x/y/z` | 7 / 8 / 9 | 运行时按余数填 |
| `hidden_global_offset_x/y/z` | 10 / 11 / 12 | 运行时填 0 |
| `hidden_grid_dims` | 13 | 运行时按 1D/2D 填 |

约定：

* 显式参数连续排在前面，隐藏参数从第一个 `kind != 0` 的参数开始；
  `fm_launch` 内部就是按这个边界数显式参数个数。
* `argv[i]` 是 `uint64_t`；C 层只拷贝低 `size` 字节到 kernarg。
  所以 f32 标量要把 IEEE754 位模式放在低 4 字节（Python `Runtime` 自动处理）。
* 调用方给的 `argv` 长度必须等于显式参数个数，否则 `fm_launch` 返回 `-2`。

## 3. grid / workgroup

```
Runtime.launch(name, grid, workgroup, argv)
    grid       = x 方向 workgroup 个数
    workgroup  = 每个 workgroup 的线程数

Runtime.launch2d(name, gx, gy, wx, wy, argv)
    (gx,gy)    = x/y 方向 workgroup 个数
    (wx,wy)    = x/y 方向线程数
```

内核侧拿到的隐藏参数与 `v0` / `s6`：

* `v0 = threadIdx.x`（0..workgroup-1）
* `s6 = blockIdx.x`
* y 方向的块数 / 线程数在 `hidden_block_count_y`、`hidden_group_size_y` 里

注意：`tools/kernel_lab.run_one` / `tools/hsa_job.py` 的 `grid` 是**总
work-item 数**，内部会自己除 workgroup。两者不要混用。

## 4. 动态启动

```python
rt = Runtime(hsaco="build/foo.hsaco", catalog="build/foo.catalog.json")
rt.launch_dyn("foo", grid_x, 1, workgroup_x, 1, [a, b, out, n])
```

`fm_launch_dyn` 在运行时从 HSACO 符号表解析 kernel object（支持 `foo`、
`foo.k` 等符号形式），参数布局由 catalog 的 `args` 提供，运行时按 `kind`
自动填隐藏参数。这样编译器新产出的 HSACO 不需要重新编译 `libfm_engine.so`。

`Runtime.launch_any` 先走动态路径，catalog 里没有时退回静态表。

## 5. 一个进程一个 HSACO

`fm_init` 一次只加载一份 HSACO。要同时用到多份 HSACO 的内核：

```bash
python3 tools/merge_hsacos.py a.hsaco b.hsaco -o merged.hsaco
python3 tools/make_catalog.py merged.hsaco merged.catalog.json
```

然后 `Runtime(hsaco="merged.hsaco", catalog="merged.catalog.json")`。

## 6. 真机限制

* **2D grid 的 y 维不可靠**：实测 `quant_act` 第二行起输出错、
  `gemm_w4a4` 第二块 NaN。优先用 1D 拆行或 1D flat 内核。
* **`private_segment > 0` 不能直接跑**：当前 HSA queue 没有接 scratch
  backing。涉及 `gdn_k`、`gdn_k2<32>`、`fa_int4`、`vit_attn_kernel`；调用前
  用 `info(lookup)["private_segment"]` 判断。
* **LDS 用量**：`group_segment` 大的内核（`fa_*`、`gemm_w4a4`、
  `vit_attn_kernel`）对 workgroup 大小敏感，不要随意改。
* **只支持 `/opt/hyhal`**。

## 7. 高频内核速查

| lookup | 作用 | 显式参数 |
|---|---|---|
| `gemv_f32_warp_k` | warp-per-row f32 GEMV | `(w,x,y,nrows,k,64)` |
| `gemv_f32_k` | 老版 f32 GEMV | `(w,x,y,nrows,k/4,64)` |
| `gemv_i8_k` | int8 权重 + f32 激活 | `(w,ws,x,y,nrows,ngroups,rowbytes,ws_stride,64)` |
| `fill_k` | f32 填充 | `(y,value,n)` |
| `silu_mul_k` | `y=silu(a)*b` | `(y,a,b,n)` |
| `gelu_mul_k` | `y=gelu(a)*b` | `(y,a,b,n,64)` |
| `softmax_k` | 行 softmax | `(y,x,rows,cols,64)` |
| `layernorm_k` | LayerNorm | `(y,x,w,b,rows,cols,eps,64)` |
| `rmsnorm_k` | RMSNorm | `(y,x,w,cols,eps,flag)`，grid=rows；`flag=0` 使用 `w` |
| `topk_k` | 行 top-k | `(x,idx,val,rows,cols,k)` |
| `router_top10_k` | MoE top-10 + softmax | `(logits,ids,weights,rows,n_experts)` |
| `reduce_blocks_k` | `partial` 行归约 | `(partial,y,nrows,nbpr)` |
| `int4_dot_k` | compressed-tensors INT4 融合点积 | 见 [`INT4.md`](INT4.md) |
| `gemm_w4a4_flat` | W4A4 预填充 GEMM | `(aq,wq,asc,wsc,c,M,N,K,nbn,magic)` |

其余 100 多个内核见 [`KERNELS.md`](KERNELS.md)。
