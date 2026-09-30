# ABI 与调用约定

## 1. 内核目录

`python/k100lc_kernels/catalog.json` 由 `tools/make_catalog.py` 从 HSACO
metadata 生成。每项：

```json
{"lookup": "gemv_f32_warp_k", "name": "_Z...", "args": [...],
 "kernarg_size": 40, "group_segment": 1024, "private_segment": 0}
```

- `lookup`：传给 `Runtime.launch()` / `fm_launch()` 的名字。
- `args`：显式参数顺序（`kind=by_value|global_buffer`）；`hidden_*` 由运行时填。
- `private_segment > 0`：内核有 spill/scratch，当前运行时未接 scratch，
  直接调用可能 fault。

## 2. 参数打包

- 指针参数：传设备指针（`fm_alloc` 返回值，u64）。
- 标量：`u32/s32` 传整数；`f32` 传 Python `float`（运行时按 IEEE754 写 4B）。
- `Runtime.launch/launch2d` 只接受**显式参数**，顺序与 catalog 中
  `kind != hidden_*` 的项一致。

## 3. grid / workgroup

```
Runtime.launch(name, grid, workgroup, argv)
    grid       = number of workgroups (blocks) in x
    workgroup  = threads per workgroup

Runtime.launch2d(name, gx, gy, wx, wy, argv)
    (gx,gy)    = number of workgroups in x/y
    (wx,wy)    = threads per workgroup in x/y
```

内核侧拿到的是 `v0 = threadIdx.x`、`s6 = blockIdx.x`、`blockIdx.y`。
`hsa_job.run_one(grid=...)` 的 `grid` 是**总 work-item 数**，两者不要混。

## 4. 常用内核速查

| lookup | 作用 | 显式参数 |
|---|---|---|
| `gemv_f32_k` | 老版 f32 GEMV，一 lane 一行 | `(w,x,y,nrows,k/4,64)` |
| `gemv_f32_warp_k` | warp-per-row f32 GEMV | `(w,x,y,nrows,k,64)` |
| `gemv_i8_k` | int8 权重 + f32 激活 GEMV | `(w,ws,x,y,nrows,ngroups,rowbytes,ws_stride,64)` |
| `iq4nl_dequant_k` | IQ4_NL → f32 | `(w,y,nblocks,64)` |
| `iq3xxs_dequant_k` | IQ3_XXS → f32 | `(w,y,grid,ksigns,nblocks,64)` |
| `iq2s_dequant_k` | IQ2_S → f32 | `(w,y,grid,nblocks,64)` |
| `iq3s_dequant_k` | IQ3_S → f32 | `(w,y,grid,nblocks,64)` |
| `iq4xs_dequant_k` | IQ4_XS → f32 | `(w,y,nblocks,64)` |
| `q2_0_dequant_k` | Q2_0 → f32 | `(w,y,nblocks,64)` |
| `iq4nl_to_i8_k` | IQ4_NL → int8 + scale | `(w,out,scale,nblocks,64)` |
| `fill_k` | 填充 f32 | `(y,value,n)` |
| `silu_mul_k` | `y=silu(a)*b` | `(y,a,b,n)` |
| `sigmoid_mul_k` | `y=x*sigmoid(g)` | `(y,x,g,n)` |
| `l2norm_k` | 每 S 维 L2 归一化 | `(x,S,eps)`，grid=rows |
| `rmsnorm_gated_k` | RMSNorm + sigmoid 门 | `(y,x,w,g,D,eps)`，grid=rows |
| `conv1d_silu_k` | 深度因果卷积 + SiLU | `(out,in,w,state,T,C,K)` |
| `ssm_ab_gate_k` | a/b 投影 + beta/g | `(gab,beta,gg,wa,wb,x,dt,alog,Hv,K)`，grid=(Hv,T) |
| `gdn_k2<32>` | GDN 递推（**private>0**） | `(out,q,k,v,g,beta,state,snap,T,Hk,Hv,D,rep)` |

## 5. 新增内核

1. 把 `.s` 放进 `kernels/asm/k_new/`，或加一个 `tools/gen_xxx.py`
   （参考 `tools/gen_gemv_f32_warp.py`，在 `build_flashmoe.py` 的 `MODULES`
   里注册）。
2. `bash tools/build_all.sh`。
3. 用 `python3 tools/gen_xxx.py` 或 `tools/kernel_lab.py` 做 bit-exact 对账。
4. 更新 `CHANGELOG.md`。

## 6. 从其他项目更新

`bash tools/update_from.sh <project-root>` 会同步 `kernels/asm/`、
`kernel_spec.json`、`tools/gen_*.py` 与关键运行时文件，然后重建。合并冲突
时以新项目为准；自定义改动请放在 `kernels/asm/k_new/` 或用新的生成器名。
