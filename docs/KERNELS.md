# 内核参数总表

本文件由 `tools/gen_kernel_docs.py` 从 [`python/k100lc_kernels/catalog.json`](../python/k100lc_kernels/catalog.json) 自动生成，请勿手工修改。

预编译 HSACO：`prebuilt/k100lc_kernels.hsaco`；共 **135 个内核**。

调用方法、grid 语义和完整示例先看 [`docs/KERNEL_CALLING.md`](KERNEL_CALLING.md)。本页只回答「某个 lookup 到底要传哪些参数」。

## 分类统计

| 分类 | 内核数 |
|---|---:|
| NVFP4 | 27 |
| INT4（RT4 / compressed-tensors） | 23 |
| 融合点积（原生量化块 × 激活） | 12 |
| 量化解码 | 12 |
| GEMV / GEMM | 5 |
| Transformer 常用算子 | 23 |
| Attention / KV / 视觉塔 | 19 |
| 序列模型 / 卷积 | 6 |
| 其他 | 8 |
| **合计** | **135** |

## 速查表

`argv` 只包含下表列出的**显式参数**，按顺序传给 `Runtime.launch(...)` / `fm_launch(...)`。
`buf:偏移:字节` = 设备指针（u64），`val:偏移:字节` = 标量
（参数类型由内核签名决定；`Runtime` 对 Python `float` 按 f32 打包）。

| lookup | 显式参数 | kernarg | LDS | private |
|---|---|---:|---:|---:|
| `add_inplace_k` | `buf:0:8 buf:8:8 val:16:8` | 280 | 0 | 0 |
| `argmax_comb_k` | `buf:0:8 buf:8:8 val:16:4 buf:24:8` | 288 | 256 | 0 |
| `argmax_k` | `buf:0:8 val:8:8 buf:16:8` | 280 | 256 | 0 |
| `argmax_part_k` | `buf:0:8 val:8:8 val:16:4 buf:24:8 buf:32:8` | 296 | 256 | 0 |
| `attn_pv_part` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4` | 96 | 0 | 0 |
| `attn_q_quant_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4` | 304 | 0 | 0 |
| `block_exp_sum_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4 val:40:4` | 48 | 256 | 0 |
| `block_max_k` | `buf:0:8 buf:8:8 val:16:4 val:20:4` | 24 | 256 | 0 |
| `concat2_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:32:8` | 296 | 0 | 0 |
| `conv1d_silu_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4 val:40:4` | 304 | 0 | 0 |
| `conv_state_update_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4 val:40:4` | 304 | 0 | 0 |
| `div_scalar_k` | `buf:0:8 buf:8:8 val:16:4` | 24 | 0 | 0 |
| `embed_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4` | 296 | 0 | 0 |
| `fa_decode_comb_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4` | 40 | 0 | 0 |
| `fa_decode_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 buf:48:8 buf:56:8 buf:64:8 val:72:4 val:76:4 val:80:4 val:84:4 val:88:4 val:92:4 val:96:4 val:100:4` | 104 | 648 | 0 |
| `fa_decode_rows_comb_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4 val:40:4 val:44:4 val:48:4 val:52:4 val:56:8` | 64 | 0 | 0 |
| `fa_decode_rows_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 buf:48:8 buf:56:8 buf:64:8 buf:72:8 val:80:8 val:88:4 val:92:4 val:96:4 val:100:4 val:104:4 val:108:4 val:112:4 val:116:4 val:120:4 val:124:4 val:128:8 val:136:8 val:144:4` | 148 | 648 | 0 |
| `fa_int4` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 buf:48:8 val:56:4 val:60:4 val:64:4 val:68:4 val:72:4 val:76:4` | 80 | 58880 | 116 |
| `fill_k` | `buf:0:8 val:8:4 val:16:8` | 280 | 0 | 0 |
| `flash_dec_comb_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4 val:40:4` | 48 | 0 | 0 |
| `flash_dec_part_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4 val:60:4 val:64:4 val:68:4 val:72:4` | 80 | 16640 | 0 |
| `gather_heads_k` | `buf:0:8 buf:8:8 val:16:4 val:20:4 val:24:4 val:28:4 val:32:4 val:40:8` | 304 | 0 | 0 |
| `gather_rows_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4` | 32 | 0 | 0 |
| `gdn_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 buf:48:8 val:56:4 val:60:4 val:64:4 val:68:4 val:72:4 val:76:4` | 336 | 0 | 528 |
| `gdn_k2<32>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 buf:48:8 buf:56:8 val:64:4 val:68:4 val:72:4 val:76:4 val:80:4` | 84 | 512 | 144 |
| `gelu_mul_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4` | 32 | 0 | 0 |
| `gemm_w4a4` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 val:40:4 val:44:4 val:48:4` | 52 | 27136 | 0 |
| `gemm_w4a4_flat` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 val:40:4 val:44:4 val:48:4 val:52:4 val:56:4` | 64 | 27136 | 0 |
| `gemv_f32_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4` | 40 | 0 | 0 |
| `gemv_f32_rows8_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4` | 40 | 256 | 0 |
| `gemv_f32_rows8_split_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4` | 40 | 256 | 0 |
| `gemv_f32_warp_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4` | 40 | 1024 | 0 |
| `gemv_i8_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4 val:40:4 val:44:4 val:48:4` | 56 | 0 | 0 |
| `gemv_w4a4<1>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 val:40:4 val:44:4` | 48 | 0 | 0 |
| `gemv_w4a4<2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 val:40:4 val:44:4` | 48 | 0 | 0 |
| `gemv_w4a4<3>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 val:40:4 val:44:4` | 48 | 0 | 0 |
| `gemv_w4a4<4>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 val:40:4 val:44:4` | 48 | 0 | 0 |
| `gemv_w4a4_r2_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 val:40:4 val:44:4` | 48 | 0 | 0 |
| `gemv_w4a4_r2_m3_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 val:40:4 val:44:4` | 48 | 0 | 0 |
| `gemv_w4a8<1,false,1>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 buf:48:8 val:56:4 val:60:4` | 64 | 0 | 0 |
| `gemv_w4a8<1,true,1>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 buf:48:8 val:56:4 val:60:4` | 320 | 0 | 0 |
| `gemv_w4a8<2,false,2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 buf:48:8 val:56:4 val:60:4` | 64 | 0 | 0 |
| `gemv_w4a8<2,true,2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 buf:48:8 val:56:4 val:60:4` | 320 | 0 | 0 |
| `gemv_w4a8<3,false,2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 buf:48:8 val:56:4 val:60:4` | 64 | 0 | 0 |
| `gemv_w4a8<3,true,2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 buf:48:8 val:56:4 val:60:4` | 320 | 0 | 0 |
| `gemv_w4a8<4,false,2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 buf:48:8 val:56:4 val:60:4` | 64 | 0 | 0 |
| `gemv_w4a8<4,true,2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 buf:48:8 val:56:4 val:60:4` | 320 | 0 | 0 |
| `int4_dequant_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4` | 32 | 0 | 0 |
| `int4_dot_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4 buf:48:8 buf:56:8 val:64:4 val:68:4 val:72:4` | 80 | 0 | 0 |
| `iq2s_dequant_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4` | 32 | 0 | 0 |
| `iq2s_dot_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4 buf:48:8 buf:56:8 val:64:4 val:68:4 val:72:4` | 80 | 0 | 0 |
| `iq3s_dequant_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4` | 32 | 0 | 0 |
| `iq3s_dot_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4 buf:48:8 buf:56:8 val:64:4 val:68:4 val:72:4` | 80 | 0 | 0 |
| `iq3xxs_dequant_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4` | 40 | 0 | 0 |
| `iq3xxs_dot_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4 buf:48:8 buf:56:8 buf:64:8 val:72:4 val:76:4 val:80:4` | 88 | 0 | 0 |
| `iq4nl_dequant_k` | `buf:0:8 buf:8:8 val:16:4 val:20:4` | 24 | 0 | 0 |
| `iq4nl_dot_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4 buf:48:8 val:56:4 val:60:4 val:64:4` | 72 | 0 | 0 |
| `iq4nl_to_i8_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4` | 32 | 0 | 0 |
| `iq4xs_dequant_k` | `buf:0:8 buf:8:8 val:16:4 val:20:4` | 24 | 0 | 0 |
| `iq4xs_dot_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4 buf:48:8 val:56:4 val:60:4 val:64:4` | 72 | 0 | 0 |
| `kv_append_k_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4` | 48 | 0 | 0 |
| `kv_append_v_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4 val:40:4 val:44:4 val:48:4 val:52:4` | 56 | 0 | 0 |
| `l2norm_k` | `buf:0:8 val:8:4 val:12:4` | 272 | 128 | 0 |
| `layernorm_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4 val:40:4 val:44:4` | 48 | 1024 | 0 |
| `moe_combine_gather_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4 val:40:4` | 48 | 0 | 0 |
| `moe_combine_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4` | 40 | 0 | 0 |
| `nvfp4_gemm_kernel` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4 val:60:4` | 320 | 10240 | 0 |
| `nvfp4_gemm_kernel_ng` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4 val:60:4` | 320 | 10240 | 0 |
| `nvfp4_gemv<1,1>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<1,2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<1,3>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<1,4>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<2,1>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<2,2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<2,3>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<2,4>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<3,1>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<3,2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<3,3>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<3,4>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<4,1>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<4,2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<4,3>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv<4,4>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv_wide<1,1>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv_wide<1,2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv_wide<2,1>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv_wide<2,2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv_wide<3,1>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv_wide<3,2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv_wide<4,1>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_gemv_wide<4,2>` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4` | 320 | 1024 | 0 |
| `nvfp4_quant_act` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4` | 296 | 4 | 0 |
| `q2_0_dequant_k` | `buf:0:8 buf:8:8 val:16:4 val:20:4` | 24 | 0 | 0 |
| `q2_0_dot_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4 buf:48:8 val:56:4 val:60:4 val:64:4` | 72 | 0 | 0 |
| `q4_0_dequant_k` | `buf:0:8 buf:8:8 val:16:4 val:20:4` | 24 | 0 | 0 |
| `q4_0_dot_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4 buf:48:8 val:56:4 val:60:4 val:64:4` | 72 | 0 | 0 |
| `q4k_dequant` | `buf:0:8 buf:8:8 val:16:4` | 80 | 0 | 0 |
| `q4k_dot_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4 buf:48:8 val:56:4 val:60:4 val:64:4` | 72 | 0 | 0 |
| `q5k_dequant` | `buf:0:8 buf:8:8 val:16:4` | 80 | 0 | 0 |
| `q5k_dot_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4 buf:48:8 val:56:4 val:60:4 val:64:4` | 72 | 0 | 0 |
| `q6k_dequant` | `buf:0:8 buf:8:8 val:16:4` | 80 | 0 | 0 |
| `q6k_dot_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4 buf:48:8 val:56:4 val:60:4 val:64:4` | 72 | 0 | 0 |
| `q8_0_dequant_k` | `buf:0:8 buf:8:8 val:16:4 val:20:4` | 24 | 0 | 0 |
| `q8_0_dot_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4 buf:48:8 val:56:4 val:60:4 val:64:4` | 72 | 0 | 0 |
| `quant_act` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 val:40:4 val:44:4` | 304 | 4 | 0 |
| `quant_act4` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4` | 32 | 0 | 0 |
| `quant_rows_a8_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 val:40:4 val:44:4 val:48:4 val:52:4 val:56:4` | 60 | 0 | 0 |
| `quant_rows_fast_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4` | 48 | 0 | 0 |
| `quant_rows_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4` | 44 | 0 | 0 |
| `reduce_blocks_k` | `buf:0:8 buf:8:8 val:16:4 val:20:4` | 24 | 0 | 0 |
| `reduce_max1_k` | `buf:0:8 buf:8:8 val:16:4` | 24 | 256 | 0 |
| `reduce_sum1_k` | `buf:0:8 buf:8:8 val:16:4` | 24 | 256 | 0 |
| `rmsnorm_gated_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4` | 296 | 128 | 0 |
| `rmsnorm_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4` | 296 | 128 | 0 |
| `rope_apply_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4` | 96 | 0 | 0 |
| `rope_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4 val:44:4 val:48:4 val:52:4` | 312 | 0 | 0 |
| `router_top10_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4` | 32 | 0 | 0 |
| `scale_mul_k` | `buf:0:8 val:8:4 val:16:8` | 280 | 0 | 0 |
| `scatter_heads_k` | `buf:0:8 buf:8:8 val:16:4 val:20:4 val:24:4 val:28:4 val:32:4 val:40:8` | 304 | 0 | 0 |
| `sigmoid_mul_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:8` | 288 | 0 | 0 |
| `silu_mul_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:8` | 288 | 0 | 0 |
| `softmax_k` | `buf:0:8 buf:8:8 val:16:4 val:20:4 val:24:4` | 32 | 1024 | 0 |
| `softmax_vec_k` | `buf:0:8 buf:8:8 val:16:4 val:20:4 val:24:4` | 32 | 1024 | 0 |
| `split_qkv_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4 val:40:4 val:44:4` | 104 | 0 | 0 |
| `ssm_ab_gate_k` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 buf:48:8 buf:56:8 val:64:4 val:68:4` | 328 | 64 | 0 |
| `topk_k` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4` | 40 | 0 | 0 |
| `vit_attn_kernel` | `buf:0:8 buf:8:8 val:16:4 val:20:4 val:24:4 val:28:4 val:32:4` | 296 | 37888 | 304 |
| `vit_bias_kernel` | `buf:0:8 buf:8:8 val:16:4 val:20:4` | 280 | 0 | 0 |
| `vit_bias_s_kernel` | `buf:0:8 buf:8:8 val:16:4 val:20:4 val:24:4` | 288 | 0 | 0 |
| `vit_gelu_kernel` | `buf:0:8 buf:8:8 val:16:8 val:24:4` | 288 | 0 | 0 |
| `vit_linear_f16_kernel` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4 val:40:4` | 304 | 8192 | 0 |
| `vit_ln_kernel` | `buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4 val:40:4` | 304 | 1024 | 0 |
| `vit_rope_kernel` | `buf:0:8 buf:8:8 buf:16:8 val:24:4 val:28:4 val:32:4 val:36:4` | 296 | 0 | 0 |
| `vt_scatter_k` | `buf:0:8 buf:8:8 val:16:4 val:20:4 val:24:4 val:28:4 val:32:4 val:36:4` | 96 | 16640 | 0 |

## 逐内核参数

### NVFP4

#### `nvfp4_gemm_kernel`

- mangled 符号：`_Z17nvfp4_gemm_kernelPKjPKhPKaS4_PKfPfiiif`
- 签名：`nvfp4_gemm_kernel(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, int, float)`
- kernarg：320 B，LDS：10240 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 60 | 4 | `by_value` | 调用方 |
| 10 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 11 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 12 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 13 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 14 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 15 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 16 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 17 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 18 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 19 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 20 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 21 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 22 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemm_kernel_ng`

- mangled 符号：`_Z20nvfp4_gemm_kernel_ngPKjPKhPKaS4_PKfPfiiif`
- 签名：`nvfp4_gemm_kernel_ng(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, int, float)`
- kernarg：320 B，LDS：10240 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 60 | 4 | `by_value` | 调用方 |
| 10 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 11 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 12 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 13 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 14 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 15 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 16 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 17 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 18 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 19 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 20 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 21 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 22 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<1,1>`

- mangled 符号：`_Z10nvfp4_gemvILi1ELi1EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<1, 1>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<1,2>`

- mangled 符号：`_Z10nvfp4_gemvILi1ELi2EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<1, 2>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<1,3>`

- mangled 符号：`_Z10nvfp4_gemvILi1ELi3EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<1, 3>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<1,4>`

- mangled 符号：`_Z10nvfp4_gemvILi1ELi4EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<1, 4>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<2,1>`

- mangled 符号：`_Z10nvfp4_gemvILi2ELi1EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<2, 1>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<2,2>`

- mangled 符号：`_Z10nvfp4_gemvILi2ELi2EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<2, 2>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<2,3>`

- mangled 符号：`_Z10nvfp4_gemvILi2ELi3EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<2, 3>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<2,4>`

- mangled 符号：`_Z10nvfp4_gemvILi2ELi4EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<2, 4>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<3,1>`

- mangled 符号：`_Z10nvfp4_gemvILi3ELi1EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<3, 1>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<3,2>`

- mangled 符号：`_Z10nvfp4_gemvILi3ELi2EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<3, 2>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<3,3>`

- mangled 符号：`_Z10nvfp4_gemvILi3ELi3EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<3, 3>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<3,4>`

- mangled 符号：`_Z10nvfp4_gemvILi3ELi4EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<3, 4>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<4,1>`

- mangled 符号：`_Z10nvfp4_gemvILi4ELi1EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<4, 1>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<4,2>`

- mangled 符号：`_Z10nvfp4_gemvILi4ELi2EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<4, 2>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<4,3>`

- mangled 符号：`_Z10nvfp4_gemvILi4ELi3EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<4, 3>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv<4,4>`

- mangled 符号：`_Z10nvfp4_gemvILi4ELi4EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv<4, 4>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv_wide<1,1>`

- mangled 符号：`_Z15nvfp4_gemv_wideILi1ELi1EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv_wide<1, 1>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv_wide<1,2>`

- mangled 符号：`_Z15nvfp4_gemv_wideILi1ELi2EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv_wide<1, 2>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv_wide<2,1>`

- mangled 符号：`_Z15nvfp4_gemv_wideILi2ELi1EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv_wide<2, 1>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv_wide<2,2>`

- mangled 符号：`_Z15nvfp4_gemv_wideILi2ELi2EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv_wide<2, 2>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv_wide<3,1>`

- mangled 符号：`_Z15nvfp4_gemv_wideILi3ELi1EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv_wide<3, 1>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv_wide<3,2>`

- mangled 符号：`_Z15nvfp4_gemv_wideILi3ELi2EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv_wide<3, 2>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv_wide<4,1>`

- mangled 符号：`_Z15nvfp4_gemv_wideILi4ELi1EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv_wide<4, 1>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_gemv_wide<4,2>`

- mangled 符号：`_Z15nvfp4_gemv_wideILi4ELi2EEvPKjPKhPKaS5_PKfPfiif`
- 签名：`void nvfp4_gemv_wide<4, 2>(unsigned int const*, unsigned char const*, signed char const*, signed char const*, float const*, float*, int, int, float)`
- kernarg：320 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 4 | `by_value` | 调用方 |
| 7 | 52 | 4 | `by_value` | 调用方 |
| 8 | 56 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `nvfp4_quant_act`

- mangled 符号：`_Z15nvfp4_quant_actPKfPaS1_Pfii`
- 签名：`nvfp4_quant_act(float const*, signed char*, signed char*, float*, int, int)`
- kernarg：296 B，LDS：4 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |
| 6 | 40 | 4 | `hidden_block_count_x` | 运行时 |
| 7 | 44 | 4 | `hidden_block_count_y` | 运行时 |
| 8 | 48 | 4 | `hidden_block_count_z` | 运行时 |
| 9 | 52 | 2 | `hidden_group_size_x` | 运行时 |
| 10 | 54 | 2 | `hidden_group_size_y` | 运行时 |
| 11 | 56 | 2 | `hidden_group_size_z` | 运行时 |
| 12 | 58 | 2 | `hidden_remainder_x` | 运行时 |
| 13 | 60 | 2 | `hidden_remainder_y` | 运行时 |
| 14 | 62 | 2 | `hidden_remainder_z` | 运行时 |
| 15 | 80 | 8 | `hidden_global_offset_x` | 运行时 |
| 16 | 88 | 8 | `hidden_global_offset_y` | 运行时 |
| 17 | 96 | 8 | `hidden_global_offset_z` | 运行时 |
| 18 | 104 | 2 | `hidden_grid_dims` | 运行时 |

### INT4（RT4 / compressed-tensors）

#### `gemm_w4a4`

- mangled 符号：`_Z9gemm_w4a4PKjS0_PKfS2_Pfiii`
- 签名：`gemm_w4a4(unsigned int const*, unsigned int const*, float const*, float const*, float*, int, int, int)`
- kernarg：52 B，LDS：27136 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 4 | `by_value` | 调用方 |
| 6 | 44 | 4 | `by_value` | 调用方 |
| 7 | 48 | 4 | `by_value` | 调用方 |

#### `gemm_w4a4_flat`

- mangled 符号：`_Z14gemm_w4a4_flatPKjS0_PKfS2_Pfiiiii`
- 签名：`gemm_w4a4_flat(unsigned int const*, unsigned int const*, float const*, float const*, float*, int, int, int, int, int)`
- kernarg：64 B，LDS：27136 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 4 | `by_value` | 调用方 |
| 6 | 44 | 4 | `by_value` | 调用方 |
| 7 | 48 | 4 | `by_value` | 调用方 |
| 8 | 52 | 4 | `by_value` | 调用方 |
| 9 | 56 | 4 | `by_value` | 调用方 |

#### `gemv_w4a4<1>`

- mangled 符号：`_Z9gemv_w4a4ILi1EEvPKjPKtS1_PKfPfii`
- 签名：`void gemv_w4a4<1>(unsigned int const*, unsigned short const*, unsigned int const*, float const*, float*, int, int)`
- kernarg：48 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 4 | `by_value` | 调用方 |
| 6 | 44 | 4 | `by_value` | 调用方 |

#### `gemv_w4a4<2>`

- mangled 符号：`_Z9gemv_w4a4ILi2EEvPKjPKtS1_PKfPfii`
- 签名：`void gemv_w4a4<2>(unsigned int const*, unsigned short const*, unsigned int const*, float const*, float*, int, int)`
- kernarg：48 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 4 | `by_value` | 调用方 |
| 6 | 44 | 4 | `by_value` | 调用方 |

#### `gemv_w4a4<3>`

- mangled 符号：`_Z9gemv_w4a4ILi3EEvPKjPKtS1_PKfPfii`
- 签名：`void gemv_w4a4<3>(unsigned int const*, unsigned short const*, unsigned int const*, float const*, float*, int, int)`
- kernarg：48 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 4 | `by_value` | 调用方 |
| 6 | 44 | 4 | `by_value` | 调用方 |

#### `gemv_w4a4<4>`

- mangled 符号：`_Z9gemv_w4a4ILi4EEvPKjPKtS1_PKfPfii`
- 签名：`void gemv_w4a4<4>(unsigned int const*, unsigned short const*, unsigned int const*, float const*, float*, int, int)`
- kernarg：48 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 4 | `by_value` | 调用方 |
| 6 | 44 | 4 | `by_value` | 调用方 |

#### `gemv_w4a4_r2_k`

- mangled 符号：`gemv_w4a4_r2_k`
- 签名：`gemv_w4a4_r2_k`
- kernarg：48 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 4 | `by_value` | 调用方 |
| 6 | 44 | 4 | `by_value` | 调用方 |

#### `gemv_w4a4_r2_m3_k`

- mangled 符号：`gemv_w4a4_r2_m3_k`
- 签名：`gemv_w4a4_r2_m3_k`
- kernarg：48 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 4 | `by_value` | 调用方 |
| 6 | 44 | 4 | `by_value` | 调用方 |

#### `gemv_w4a8<1,false,1>`

- mangled 符号：`_Z9gemv_w4a8ILi1ELb0ELi1EEvPKjPKtS1_S1_PKfPKiPfii`
- 签名：`void gemv_w4a8<1, false, 1>(unsigned int const*, unsigned short const*, unsigned int const*, unsigned int const*, float const*, int const*, float*, int, int)`
- kernarg：64 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 8 | `global_buffer` | 调用方 |
| 7 | 56 | 4 | `by_value` | 调用方 |
| 8 | 60 | 4 | `by_value` | 调用方 |

#### `gemv_w4a8<1,true,1>`

- mangled 符号：`_Z9gemv_w4a8ILi1ELb1ELi1EEvPKjPKtS1_S1_PKfPKiPfii`
- 签名：`void gemv_w4a8<1, true, 1>(unsigned int const*, unsigned short const*, unsigned int const*, unsigned int const*, float const*, int const*, float*, int, int)`
- kernarg：320 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 8 | `global_buffer` | 调用方 |
| 7 | 56 | 4 | `by_value` | 调用方 |
| 8 | 60 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `gemv_w4a8<2,false,2>`

- mangled 符号：`_Z9gemv_w4a8ILi2ELb0ELi2EEvPKjPKtS1_S1_PKfPKiPfii`
- 签名：`void gemv_w4a8<2, false, 2>(unsigned int const*, unsigned short const*, unsigned int const*, unsigned int const*, float const*, int const*, float*, int, int)`
- kernarg：64 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 8 | `global_buffer` | 调用方 |
| 7 | 56 | 4 | `by_value` | 调用方 |
| 8 | 60 | 4 | `by_value` | 调用方 |

#### `gemv_w4a8<2,true,2>`

- mangled 符号：`_Z9gemv_w4a8ILi2ELb1ELi2EEvPKjPKtS1_S1_PKfPKiPfii`
- 签名：`void gemv_w4a8<2, true, 2>(unsigned int const*, unsigned short const*, unsigned int const*, unsigned int const*, float const*, int const*, float*, int, int)`
- kernarg：320 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 8 | `global_buffer` | 调用方 |
| 7 | 56 | 4 | `by_value` | 调用方 |
| 8 | 60 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `gemv_w4a8<3,false,2>`

- mangled 符号：`_Z9gemv_w4a8ILi3ELb0ELi2EEvPKjPKtS1_S1_PKfPKiPfii`
- 签名：`void gemv_w4a8<3, false, 2>(unsigned int const*, unsigned short const*, unsigned int const*, unsigned int const*, float const*, int const*, float*, int, int)`
- kernarg：64 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 8 | `global_buffer` | 调用方 |
| 7 | 56 | 4 | `by_value` | 调用方 |
| 8 | 60 | 4 | `by_value` | 调用方 |

#### `gemv_w4a8<3,true,2>`

- mangled 符号：`_Z9gemv_w4a8ILi3ELb1ELi2EEvPKjPKtS1_S1_PKfPKiPfii`
- 签名：`void gemv_w4a8<3, true, 2>(unsigned int const*, unsigned short const*, unsigned int const*, unsigned int const*, float const*, int const*, float*, int, int)`
- kernarg：320 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 8 | `global_buffer` | 调用方 |
| 7 | 56 | 4 | `by_value` | 调用方 |
| 8 | 60 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `gemv_w4a8<4,false,2>`

- mangled 符号：`_Z9gemv_w4a8ILi4ELb0ELi2EEvPKjPKtS1_S1_PKfPKiPfii`
- 签名：`void gemv_w4a8<4, false, 2>(unsigned int const*, unsigned short const*, unsigned int const*, unsigned int const*, float const*, int const*, float*, int, int)`
- kernarg：64 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 8 | `global_buffer` | 调用方 |
| 7 | 56 | 4 | `by_value` | 调用方 |
| 8 | 60 | 4 | `by_value` | 调用方 |

#### `gemv_w4a8<4,true,2>`

- mangled 符号：`_Z9gemv_w4a8ILi4ELb1ELi2EEvPKjPKtS1_S1_PKfPKiPfii`
- 签名：`void gemv_w4a8<4, true, 2>(unsigned int const*, unsigned short const*, unsigned int const*, unsigned int const*, float const*, int const*, float*, int, int)`
- kernarg：320 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 8 | `global_buffer` | 调用方 |
| 7 | 56 | 4 | `by_value` | 调用方 |
| 8 | 60 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `hidden_block_count_x` | 运行时 |
| 10 | 68 | 4 | `hidden_block_count_y` | 运行时 |
| 11 | 72 | 4 | `hidden_block_count_z` | 运行时 |
| 12 | 76 | 2 | `hidden_group_size_x` | 运行时 |
| 13 | 78 | 2 | `hidden_group_size_y` | 运行时 |
| 14 | 80 | 2 | `hidden_group_size_z` | 运行时 |
| 15 | 82 | 2 | `hidden_remainder_x` | 运行时 |
| 16 | 84 | 2 | `hidden_remainder_y` | 运行时 |
| 17 | 86 | 2 | `hidden_remainder_z` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_x` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_y` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_z` | 运行时 |
| 21 | 128 | 2 | `hidden_grid_dims` | 运行时 |

#### `int4_dequant_k`

- mangled 符号：`int4_dequant_k`
- 签名：`int4_dequant_k`
- kernarg：32 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |

#### `int4_dot_k`

- mangled 符号：`int4_dot_k`
- 签名：`int4_dot_k`
- kernarg：80 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |
| 9 | 48 | 8 | `global_buffer` | 调用方 |
| 10 | 56 | 8 | `global_buffer` | 调用方 |
| 11 | 64 | 4 | `by_value` | 调用方 |
| 12 | 68 | 4 | `by_value` | 调用方 |
| 13 | 72 | 4 | `by_value` | 调用方 |

#### `quant_act`

- mangled 符号：`_Z9quant_actPKfPaS1_PfPiii`
- 签名：`quant_act(float const*, signed char*, signed char*, float*, int*, int, int)`
- kernarg：304 B，LDS：4 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 4 | `by_value` | 调用方 |
| 6 | 44 | 4 | `by_value` | 调用方 |
| 7 | 48 | 4 | `hidden_block_count_x` | 运行时 |
| 8 | 52 | 4 | `hidden_block_count_y` | 运行时 |
| 9 | 56 | 4 | `hidden_block_count_z` | 运行时 |
| 10 | 60 | 2 | `hidden_group_size_x` | 运行时 |
| 11 | 62 | 2 | `hidden_group_size_y` | 运行时 |
| 12 | 64 | 2 | `hidden_group_size_z` | 运行时 |
| 13 | 66 | 2 | `hidden_remainder_x` | 运行时 |
| 14 | 68 | 2 | `hidden_remainder_y` | 运行时 |
| 15 | 70 | 2 | `hidden_remainder_z` | 运行时 |
| 16 | 88 | 8 | `hidden_global_offset_x` | 运行时 |
| 17 | 96 | 8 | `hidden_global_offset_y` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_z` | 运行时 |
| 19 | 112 | 2 | `hidden_grid_dims` | 运行时 |

#### `quant_act4`

- mangled 符号：`_Z10quant_act4PKfPhPfii`
- 签名：`quant_act4(float const*, unsigned char*, float*, int, int)`
- kernarg：32 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |

#### `quant_rows_a8_k`

- mangled 符号：`_Z15quant_rows_a8_kPjS_PfS0_PKfiiiii`
- 签名：`quant_rows_a8_k(unsigned int*, unsigned int*, float*, float*, float const*, int, int, int, int, int)`
- kernarg：60 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 4 | `by_value` | 调用方 |
| 6 | 44 | 4 | `by_value` | 调用方 |
| 7 | 48 | 4 | `by_value` | 调用方 |
| 8 | 52 | 4 | `by_value` | 调用方 |
| 9 | 56 | 4 | `by_value` | 调用方 |

#### `quant_rows_fast_k`

- mangled 符号：`quant_rows_fast_k`
- 签名：`quant_rows_fast_k`
- kernarg：48 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |

#### `quant_rows_k`

- mangled 符号：`_Z12quant_rows_kPjPfPKfiiiii`
- 签名：`quant_rows_k(unsigned int*, float*, float const*, int, int, int, int, int)`
- kernarg：44 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |

### 融合点积（原生量化块 × 激活）

#### `iq2s_dot_k`

- mangled 符号：`iq2s_dot_k`
- 签名：`iq2s_dot_k`
- kernarg：80 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |
| 9 | 48 | 8 | `global_buffer` | 调用方 |
| 10 | 56 | 8 | `global_buffer` | 调用方 |
| 11 | 64 | 4 | `by_value` | 调用方 |
| 12 | 68 | 4 | `by_value` | 调用方 |
| 13 | 72 | 4 | `by_value` | 调用方 |

#### `iq3s_dot_k`

- mangled 符号：`iq3s_dot_k`
- 签名：`iq3s_dot_k`
- kernarg：80 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |
| 9 | 48 | 8 | `global_buffer` | 调用方 |
| 10 | 56 | 8 | `global_buffer` | 调用方 |
| 11 | 64 | 4 | `by_value` | 调用方 |
| 12 | 68 | 4 | `by_value` | 调用方 |
| 13 | 72 | 4 | `by_value` | 调用方 |

#### `iq3xxs_dot_k`

- mangled 符号：`iq3xxs_dot_k`
- 签名：`iq3xxs_dot_k`
- kernarg：88 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |
| 9 | 48 | 8 | `global_buffer` | 调用方 |
| 10 | 56 | 8 | `global_buffer` | 调用方 |
| 11 | 64 | 8 | `global_buffer` | 调用方 |
| 12 | 72 | 4 | `by_value` | 调用方 |
| 13 | 76 | 4 | `by_value` | 调用方 |
| 14 | 80 | 4 | `by_value` | 调用方 |

#### `iq4nl_dot_k`

- mangled 符号：`iq4nl_dot_k`
- 签名：`iq4nl_dot_k`
- kernarg：72 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |
| 9 | 48 | 8 | `global_buffer` | 调用方 |
| 10 | 56 | 4 | `by_value` | 调用方 |
| 11 | 60 | 4 | `by_value` | 调用方 |
| 12 | 64 | 4 | `by_value` | 调用方 |

#### `iq4xs_dot_k`

- mangled 符号：`iq4xs_dot_k`
- 签名：`iq4xs_dot_k`
- kernarg：72 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |
| 9 | 48 | 8 | `global_buffer` | 调用方 |
| 10 | 56 | 4 | `by_value` | 调用方 |
| 11 | 60 | 4 | `by_value` | 调用方 |
| 12 | 64 | 4 | `by_value` | 调用方 |

#### `q2_0_dot_k`

- mangled 符号：`q2_0_dot_k`
- 签名：`q2_0_dot_k`
- kernarg：72 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |
| 9 | 48 | 8 | `global_buffer` | 调用方 |
| 10 | 56 | 4 | `by_value` | 调用方 |
| 11 | 60 | 4 | `by_value` | 调用方 |
| 12 | 64 | 4 | `by_value` | 调用方 |

#### `q4_0_dot_k`

- mangled 符号：`q4_0_dot_k`
- 签名：`q4_0_dot_k`
- kernarg：72 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |
| 9 | 48 | 8 | `global_buffer` | 调用方 |
| 10 | 56 | 4 | `by_value` | 调用方 |
| 11 | 60 | 4 | `by_value` | 调用方 |
| 12 | 64 | 4 | `by_value` | 调用方 |

#### `q4k_dot_k`

- mangled 符号：`q4k_dot_k`
- 签名：`q4k_dot_k`
- kernarg：72 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |
| 9 | 48 | 8 | `global_buffer` | 调用方 |
| 10 | 56 | 4 | `by_value` | 调用方 |
| 11 | 60 | 4 | `by_value` | 调用方 |
| 12 | 64 | 4 | `by_value` | 调用方 |

#### `q5k_dot_k`

- mangled 符号：`q5k_dot_k`
- 签名：`q5k_dot_k`
- kernarg：72 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |
| 9 | 48 | 8 | `global_buffer` | 调用方 |
| 10 | 56 | 4 | `by_value` | 调用方 |
| 11 | 60 | 4 | `by_value` | 调用方 |
| 12 | 64 | 4 | `by_value` | 调用方 |

#### `q6k_dot_k`

- mangled 符号：`q6k_dot_k`
- 签名：`q6k_dot_k`
- kernarg：72 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |
| 9 | 48 | 8 | `global_buffer` | 调用方 |
| 10 | 56 | 4 | `by_value` | 调用方 |
| 11 | 60 | 4 | `by_value` | 调用方 |
| 12 | 64 | 4 | `by_value` | 调用方 |

#### `q8_0_dot_k`

- mangled 符号：`q8_0_dot_k`
- 签名：`q8_0_dot_k`
- kernarg：72 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |
| 9 | 48 | 8 | `global_buffer` | 调用方 |
| 10 | 56 | 4 | `by_value` | 调用方 |
| 11 | 60 | 4 | `by_value` | 调用方 |
| 12 | 64 | 4 | `by_value` | 调用方 |

#### `reduce_blocks_k`

- mangled 符号：`reduce_blocks_k`
- 签名：`reduce_blocks_k`
- kernarg：24 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 20 | 4 | `by_value` | 调用方 |

### 量化解码

#### `iq2s_dequant_k`

- mangled 符号：`iq2s_dequant_k`
- 签名：`iq2s_dequant_k`
- kernarg：32 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |

#### `iq3s_dequant_k`

- mangled 符号：`iq3s_dequant_k`
- 签名：`iq3s_dequant_k`
- kernarg：32 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |

#### `iq3xxs_dequant_k`

- mangled 符号：`iq3xxs_dequant_k`
- 签名：`iq3xxs_dequant_k`
- kernarg：40 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |

#### `iq4nl_dequant_k`

- mangled 符号：`iq4nl_dequant_k`
- 签名：`iq4nl_dequant_k`
- kernarg：24 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 20 | 4 | `by_value` | 调用方 |

#### `iq4nl_to_i8_k`

- mangled 符号：`iq4nl_to_i8_k`
- 签名：`iq4nl_to_i8_k`
- kernarg：32 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |

#### `iq4xs_dequant_k`

- mangled 符号：`iq4xs_dequant_k`
- 签名：`iq4xs_dequant_k`
- kernarg：24 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 20 | 4 | `by_value` | 调用方 |

#### `q2_0_dequant_k`

- mangled 符号：`q2_0_dequant_k`
- 签名：`q2_0_dequant_k`
- kernarg：24 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 20 | 4 | `by_value` | 调用方 |

#### `q4_0_dequant_k`

- mangled 符号：`q4_0_dequant_k`
- 签名：`q4_0_dequant_k`
- kernarg：24 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 20 | 4 | `by_value` | 调用方 |

#### `q4k_dequant`

- mangled 符号：`q4k_dequant`
- 签名：`q4k_dequant`
- kernarg：80 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 24 | 4 | `hidden_block_count_x` | 运行时 |
| 4 | 28 | 4 | `hidden_block_count_y` | 运行时 |
| 5 | 32 | 4 | `hidden_block_count_z` | 运行时 |
| 6 | 36 | 2 | `hidden_group_size_x` | 运行时 |
| 7 | 38 | 2 | `hidden_group_size_y` | 运行时 |
| 8 | 40 | 2 | `hidden_group_size_z` | 运行时 |
| 9 | 42 | 2 | `hidden_remainder_x` | 运行时 |
| 10 | 44 | 2 | `hidden_remainder_y` | 运行时 |
| 11 | 46 | 2 | `hidden_remainder_z` | 运行时 |
| 12 | 48 | 8 | `hidden_global_offset_x` | 运行时 |
| 13 | 56 | 8 | `hidden_global_offset_y` | 运行时 |
| 14 | 64 | 8 | `hidden_global_offset_z` | 运行时 |
| 15 | 72 | 2 | `hidden_grid_dims` | 运行时 |

#### `q5k_dequant`

- mangled 符号：`q5k_dequant`
- 签名：`q5k_dequant`
- kernarg：80 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 24 | 4 | `hidden_block_count_x` | 运行时 |
| 4 | 28 | 4 | `hidden_block_count_y` | 运行时 |
| 5 | 32 | 4 | `hidden_block_count_z` | 运行时 |
| 6 | 36 | 2 | `hidden_group_size_x` | 运行时 |
| 7 | 38 | 2 | `hidden_group_size_y` | 运行时 |
| 8 | 40 | 2 | `hidden_group_size_z` | 运行时 |
| 9 | 42 | 2 | `hidden_remainder_x` | 运行时 |
| 10 | 44 | 2 | `hidden_remainder_y` | 运行时 |
| 11 | 46 | 2 | `hidden_remainder_z` | 运行时 |
| 12 | 48 | 8 | `hidden_global_offset_x` | 运行时 |
| 13 | 56 | 8 | `hidden_global_offset_y` | 运行时 |
| 14 | 64 | 8 | `hidden_global_offset_z` | 运行时 |
| 15 | 72 | 2 | `hidden_grid_dims` | 运行时 |

#### `q6k_dequant`

- mangled 符号：`q6k_dequant`
- 签名：`q6k_dequant`
- kernarg：80 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 24 | 4 | `hidden_block_count_x` | 运行时 |
| 4 | 28 | 4 | `hidden_block_count_y` | 运行时 |
| 5 | 32 | 4 | `hidden_block_count_z` | 运行时 |
| 6 | 36 | 2 | `hidden_group_size_x` | 运行时 |
| 7 | 38 | 2 | `hidden_group_size_y` | 运行时 |
| 8 | 40 | 2 | `hidden_group_size_z` | 运行时 |
| 9 | 42 | 2 | `hidden_remainder_x` | 运行时 |
| 10 | 44 | 2 | `hidden_remainder_y` | 运行时 |
| 11 | 46 | 2 | `hidden_remainder_z` | 运行时 |
| 12 | 48 | 8 | `hidden_global_offset_x` | 运行时 |
| 13 | 56 | 8 | `hidden_global_offset_y` | 运行时 |
| 14 | 64 | 8 | `hidden_global_offset_z` | 运行时 |
| 15 | 72 | 2 | `hidden_grid_dims` | 运行时 |

#### `q8_0_dequant_k`

- mangled 符号：`q8_0_dequant_k`
- 签名：`q8_0_dequant_k`
- kernarg：24 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 20 | 4 | `by_value` | 调用方 |

### GEMV / GEMM

#### `gemv_f32_k`

- mangled 符号：`gemv_f32_k`
- 签名：`gemv_f32_k`
- kernarg：40 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |

#### `gemv_f32_rows8_k`

- mangled 符号：`gemv_f32_rows8_k`
- 签名：`gemv_f32_rows8_k`
- kernarg：40 B，LDS：256 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |

#### `gemv_f32_rows8_split_k`

- mangled 符号：`gemv_f32_rows8_split_k`
- 签名：`gemv_f32_rows8_split_k`
- kernarg：40 B，LDS：256 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |

#### `gemv_f32_warp_k`

- mangled 符号：`gemv_f32_warp_k`
- 签名：`gemv_f32_warp_k`
- kernarg：40 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |

#### `gemv_i8_k`

- mangled 符号：`gemv_i8_k`
- 签名：`gemv_i8_k`
- kernarg：56 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |
| 6 | 40 | 4 | `by_value` | 调用方 |
| 7 | 44 | 4 | `by_value` | 调用方 |
| 8 | 48 | 4 | `by_value` | 调用方 |

### Transformer 常用算子

#### `add_inplace_k`

- mangled 符号：`_Z13add_inplace_kPfPKfx`
- 签名：`add_inplace_k(float*, float const*, long long)`
- kernarg：280 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `by_value` | 调用方 |
| 3 | 24 | 4 | `hidden_block_count_x` | 运行时 |
| 4 | 28 | 4 | `hidden_block_count_y` | 运行时 |
| 5 | 32 | 4 | `hidden_block_count_z` | 运行时 |
| 6 | 36 | 2 | `hidden_group_size_x` | 运行时 |
| 7 | 38 | 2 | `hidden_group_size_y` | 运行时 |
| 8 | 40 | 2 | `hidden_group_size_z` | 运行时 |
| 9 | 42 | 2 | `hidden_remainder_x` | 运行时 |
| 10 | 44 | 2 | `hidden_remainder_y` | 运行时 |
| 11 | 46 | 2 | `hidden_remainder_z` | 运行时 |
| 12 | 64 | 8 | `hidden_global_offset_x` | 运行时 |
| 13 | 72 | 8 | `hidden_global_offset_y` | 运行时 |
| 14 | 80 | 8 | `hidden_global_offset_z` | 运行时 |
| 15 | 88 | 2 | `hidden_grid_dims` | 运行时 |

#### `argmax_comb_k`

- mangled 符号：`_Z13argmax_comb_kPKfPKiiPi`
- 签名：`argmax_comb_k(float const*, int const*, int, int*)`
- kernarg：288 B，LDS：256 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `hidden_block_count_x` | 运行时 |
| 5 | 36 | 4 | `hidden_block_count_y` | 运行时 |
| 6 | 40 | 4 | `hidden_block_count_z` | 运行时 |
| 7 | 44 | 2 | `hidden_group_size_x` | 运行时 |
| 8 | 46 | 2 | `hidden_group_size_y` | 运行时 |
| 9 | 48 | 2 | `hidden_group_size_z` | 运行时 |
| 10 | 50 | 2 | `hidden_remainder_x` | 运行时 |
| 11 | 52 | 2 | `hidden_remainder_y` | 运行时 |
| 12 | 54 | 2 | `hidden_remainder_z` | 运行时 |
| 13 | 72 | 8 | `hidden_global_offset_x` | 运行时 |
| 14 | 80 | 8 | `hidden_global_offset_y` | 运行时 |
| 15 | 88 | 8 | `hidden_global_offset_z` | 运行时 |
| 16 | 96 | 2 | `hidden_grid_dims` | 运行时 |

#### `argmax_k`

- mangled 符号：`_Z8argmax_kPKfxPi`
- 签名：`argmax_k(float const*, long long, int*)`
- kernarg：280 B，LDS：256 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `by_value` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `hidden_block_count_x` | 运行时 |
| 4 | 28 | 4 | `hidden_block_count_y` | 运行时 |
| 5 | 32 | 4 | `hidden_block_count_z` | 运行时 |
| 6 | 36 | 2 | `hidden_group_size_x` | 运行时 |
| 7 | 38 | 2 | `hidden_group_size_y` | 运行时 |
| 8 | 40 | 2 | `hidden_group_size_z` | 运行时 |
| 9 | 42 | 2 | `hidden_remainder_x` | 运行时 |
| 10 | 44 | 2 | `hidden_remainder_y` | 运行时 |
| 11 | 46 | 2 | `hidden_remainder_z` | 运行时 |
| 12 | 64 | 8 | `hidden_global_offset_x` | 运行时 |
| 13 | 72 | 8 | `hidden_global_offset_y` | 运行时 |
| 14 | 80 | 8 | `hidden_global_offset_z` | 运行时 |
| 15 | 88 | 2 | `hidden_grid_dims` | 运行时 |

#### `argmax_part_k`

- mangled 符号：`_Z13argmax_part_kPKfxiPfPi`
- 签名：`argmax_part_k(float const*, long long, int, float*, int*)`
- kernarg：296 B，LDS：256 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `by_value` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 4 | `hidden_block_count_x` | 运行时 |
| 6 | 44 | 4 | `hidden_block_count_y` | 运行时 |
| 7 | 48 | 4 | `hidden_block_count_z` | 运行时 |
| 8 | 52 | 2 | `hidden_group_size_x` | 运行时 |
| 9 | 54 | 2 | `hidden_group_size_y` | 运行时 |
| 10 | 56 | 2 | `hidden_group_size_z` | 运行时 |
| 11 | 58 | 2 | `hidden_remainder_x` | 运行时 |
| 12 | 60 | 2 | `hidden_remainder_y` | 运行时 |
| 13 | 62 | 2 | `hidden_remainder_z` | 运行时 |
| 14 | 80 | 8 | `hidden_global_offset_x` | 运行时 |
| 15 | 88 | 8 | `hidden_global_offset_y` | 运行时 |
| 16 | 96 | 8 | `hidden_global_offset_z` | 运行时 |
| 17 | 104 | 2 | `hidden_grid_dims` | 运行时 |

#### `concat2_k`

- mangled 符号：`_Z9concat2_kPfPKfS1_ix`
- 签名：`concat2_k(float*, float const*, float const*, int, long long)`
- kernarg：296 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 32 | 8 | `by_value` | 调用方 |
| 5 | 40 | 4 | `hidden_block_count_x` | 运行时 |
| 6 | 44 | 4 | `hidden_block_count_y` | 运行时 |
| 7 | 48 | 4 | `hidden_block_count_z` | 运行时 |
| 8 | 52 | 2 | `hidden_group_size_x` | 运行时 |
| 9 | 54 | 2 | `hidden_group_size_y` | 运行时 |
| 10 | 56 | 2 | `hidden_group_size_z` | 运行时 |
| 11 | 58 | 2 | `hidden_remainder_x` | 运行时 |
| 12 | 60 | 2 | `hidden_remainder_y` | 运行时 |
| 13 | 62 | 2 | `hidden_remainder_z` | 运行时 |
| 14 | 80 | 8 | `hidden_global_offset_x` | 运行时 |
| 15 | 88 | 8 | `hidden_global_offset_y` | 运行时 |
| 16 | 96 | 8 | `hidden_global_offset_z` | 运行时 |
| 17 | 104 | 2 | `hidden_grid_dims` | 运行时 |

#### `embed_k`

- mangled 符号：`_Z7embed_kPfPKjPKtPKiii`
- 签名：`embed_k(float*, unsigned int const*, unsigned short const*, int const*, int, int)`
- kernarg：296 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |
| 6 | 40 | 4 | `hidden_block_count_x` | 运行时 |
| 7 | 44 | 4 | `hidden_block_count_y` | 运行时 |
| 8 | 48 | 4 | `hidden_block_count_z` | 运行时 |
| 9 | 52 | 2 | `hidden_group_size_x` | 运行时 |
| 10 | 54 | 2 | `hidden_group_size_y` | 运行时 |
| 11 | 56 | 2 | `hidden_group_size_z` | 运行时 |
| 12 | 58 | 2 | `hidden_remainder_x` | 运行时 |
| 13 | 60 | 2 | `hidden_remainder_y` | 运行时 |
| 14 | 62 | 2 | `hidden_remainder_z` | 运行时 |
| 15 | 80 | 8 | `hidden_global_offset_x` | 运行时 |
| 16 | 88 | 8 | `hidden_global_offset_y` | 运行时 |
| 17 | 96 | 8 | `hidden_global_offset_z` | 运行时 |
| 18 | 104 | 2 | `hidden_grid_dims` | 运行时 |

#### `fill_k`

- mangled 符号：`_Z6fill_kPffx`
- 签名：`fill_k(float*, float, long long)`
- kernarg：280 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 4 | `by_value` | 调用方 |
| 2 | 16 | 8 | `by_value` | 调用方 |
| 3 | 24 | 4 | `hidden_block_count_x` | 运行时 |
| 4 | 28 | 4 | `hidden_block_count_y` | 运行时 |
| 5 | 32 | 4 | `hidden_block_count_z` | 运行时 |
| 6 | 36 | 2 | `hidden_group_size_x` | 运行时 |
| 7 | 38 | 2 | `hidden_group_size_y` | 运行时 |
| 8 | 40 | 2 | `hidden_group_size_z` | 运行时 |
| 9 | 42 | 2 | `hidden_remainder_x` | 运行时 |
| 10 | 44 | 2 | `hidden_remainder_y` | 运行时 |
| 11 | 46 | 2 | `hidden_remainder_z` | 运行时 |
| 12 | 64 | 8 | `hidden_global_offset_x` | 运行时 |
| 13 | 72 | 8 | `hidden_global_offset_y` | 运行时 |
| 14 | 80 | 8 | `hidden_global_offset_z` | 运行时 |
| 15 | 88 | 2 | `hidden_grid_dims` | 运行时 |

#### `gelu_mul_k`

- mangled 符号：`gelu_mul_k`
- 签名：`gelu_mul_k`
- kernarg：32 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |

#### `l2norm_k`

- mangled 符号：`_Z8l2norm_kPfif`
- 签名：`l2norm_k(float*, int, float)`
- kernarg：272 B，LDS：128 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 4 | `by_value` | 调用方 |
| 2 | 12 | 4 | `by_value` | 调用方 |
| 3 | 16 | 4 | `hidden_block_count_x` | 运行时 |
| 4 | 20 | 4 | `hidden_block_count_y` | 运行时 |
| 5 | 24 | 4 | `hidden_block_count_z` | 运行时 |
| 6 | 28 | 2 | `hidden_group_size_x` | 运行时 |
| 7 | 30 | 2 | `hidden_group_size_y` | 运行时 |
| 8 | 32 | 2 | `hidden_group_size_z` | 运行时 |
| 9 | 34 | 2 | `hidden_remainder_x` | 运行时 |
| 10 | 36 | 2 | `hidden_remainder_y` | 运行时 |
| 11 | 38 | 2 | `hidden_remainder_z` | 运行时 |
| 12 | 56 | 8 | `hidden_global_offset_x` | 运行时 |
| 13 | 64 | 8 | `hidden_global_offset_y` | 运行时 |
| 14 | 72 | 8 | `hidden_global_offset_z` | 运行时 |
| 15 | 80 | 2 | `hidden_grid_dims` | 运行时 |

#### `layernorm_k`

- mangled 符号：`layernorm_k`
- 签名：`layernorm_k`
- kernarg：48 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |
| 6 | 40 | 4 | `by_value` | 调用方 |
| 7 | 44 | 4 | `by_value` | 调用方 |

#### `moe_combine_gather_k`

- mangled 符号：`moe_combine_gather_k`
- 签名：`moe_combine_gather_k`
- kernarg：48 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |
| 6 | 40 | 4 | `by_value` | 调用方 |

#### `moe_combine_k`

- mangled 符号：`moe_combine_k`
- 签名：`moe_combine_k`
- kernarg：40 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |

#### `rmsnorm_gated_k`

- mangled 符号：`_Z15rmsnorm_gated_kPfPKfS1_S1_if`
- 签名：`rmsnorm_gated_k(float*, float const*, float const*, float const*, int, float)`
- kernarg：296 B，LDS：128 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |
| 6 | 40 | 4 | `hidden_block_count_x` | 运行时 |
| 7 | 44 | 4 | `hidden_block_count_y` | 运行时 |
| 8 | 48 | 4 | `hidden_block_count_z` | 运行时 |
| 9 | 52 | 2 | `hidden_group_size_x` | 运行时 |
| 10 | 54 | 2 | `hidden_group_size_y` | 运行时 |
| 11 | 56 | 2 | `hidden_group_size_z` | 运行时 |
| 12 | 58 | 2 | `hidden_remainder_x` | 运行时 |
| 13 | 60 | 2 | `hidden_remainder_y` | 运行时 |
| 14 | 62 | 2 | `hidden_remainder_z` | 运行时 |
| 15 | 80 | 8 | `hidden_global_offset_x` | 运行时 |
| 16 | 88 | 8 | `hidden_global_offset_y` | 运行时 |
| 17 | 96 | 8 | `hidden_global_offset_z` | 运行时 |
| 18 | 104 | 2 | `hidden_grid_dims` | 运行时 |

#### `rmsnorm_k`

- mangled 符号：`_Z9rmsnorm_kPfPKfS1_ifi`
- 签名：`rmsnorm_k(float*, float const*, float const*, int, float, int)`
- kernarg：296 B，LDS：128 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 40 | 4 | `hidden_block_count_x` | 运行时 |
| 7 | 44 | 4 | `hidden_block_count_y` | 运行时 |
| 8 | 48 | 4 | `hidden_block_count_z` | 运行时 |
| 9 | 52 | 2 | `hidden_group_size_x` | 运行时 |
| 10 | 54 | 2 | `hidden_group_size_y` | 运行时 |
| 11 | 56 | 2 | `hidden_group_size_z` | 运行时 |
| 12 | 58 | 2 | `hidden_remainder_x` | 运行时 |
| 13 | 60 | 2 | `hidden_remainder_y` | 运行时 |
| 14 | 62 | 2 | `hidden_remainder_z` | 运行时 |
| 15 | 80 | 8 | `hidden_global_offset_x` | 运行时 |
| 16 | 88 | 8 | `hidden_global_offset_y` | 运行时 |
| 17 | 96 | 8 | `hidden_global_offset_z` | 运行时 |
| 18 | 104 | 2 | `hidden_grid_dims` | 运行时 |

#### `rope_apply_k`

- mangled 符号：`rope_apply_k`
- 签名：`rope_apply_k`
- kernarg：96 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |
| 6 | 40 | 4 | `hidden_block_count_x` | 运行时 |
| 7 | 44 | 4 | `hidden_block_count_y` | 运行时 |
| 8 | 48 | 4 | `hidden_block_count_z` | 运行时 |
| 9 | 52 | 2 | `hidden_group_size_x` | 运行时 |
| 10 | 54 | 2 | `hidden_group_size_y` | 运行时 |
| 11 | 56 | 2 | `hidden_group_size_z` | 运行时 |
| 12 | 58 | 2 | `hidden_remainder_x` | 运行时 |
| 13 | 60 | 2 | `hidden_remainder_y` | 运行时 |
| 14 | 62 | 2 | `hidden_remainder_z` | 运行时 |
| 15 | 64 | 8 | `hidden_global_offset_x` | 运行时 |
| 16 | 72 | 8 | `hidden_global_offset_y` | 运行时 |
| 17 | 80 | 8 | `hidden_global_offset_z` | 运行时 |
| 18 | 88 | 2 | `hidden_grid_dims` | 运行时 |

#### `rope_k`

- mangled 符号：`_Z6rope_kPfS_PKiiiiiiiif`
- 签名：`rope_k(float*, float*, int const*, int, int, int, int, int, int, int, float)`
- kernarg：312 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |
| 9 | 48 | 4 | `by_value` | 调用方 |
| 10 | 52 | 4 | `by_value` | 调用方 |
| 11 | 56 | 4 | `hidden_block_count_x` | 运行时 |
| 12 | 60 | 4 | `hidden_block_count_y` | 运行时 |
| 13 | 64 | 4 | `hidden_block_count_z` | 运行时 |
| 14 | 68 | 2 | `hidden_group_size_x` | 运行时 |
| 15 | 70 | 2 | `hidden_group_size_y` | 运行时 |
| 16 | 72 | 2 | `hidden_group_size_z` | 运行时 |
| 17 | 74 | 2 | `hidden_remainder_x` | 运行时 |
| 18 | 76 | 2 | `hidden_remainder_y` | 运行时 |
| 19 | 78 | 2 | `hidden_remainder_z` | 运行时 |
| 20 | 96 | 8 | `hidden_global_offset_x` | 运行时 |
| 21 | 104 | 8 | `hidden_global_offset_y` | 运行时 |
| 22 | 112 | 8 | `hidden_global_offset_z` | 运行时 |
| 23 | 120 | 2 | `hidden_grid_dims` | 运行时 |

#### `router_top10_k`

- mangled 符号：`router_top10_k`
- 签名：`router_top10_k`
- kernarg：32 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |

#### `scale_mul_k`

- mangled 符号：`_Z11scale_mul_kPffx`
- 签名：`scale_mul_k(float*, float, long long)`
- kernarg：280 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 4 | `by_value` | 调用方 |
| 2 | 16 | 8 | `by_value` | 调用方 |
| 3 | 24 | 4 | `hidden_block_count_x` | 运行时 |
| 4 | 28 | 4 | `hidden_block_count_y` | 运行时 |
| 5 | 32 | 4 | `hidden_block_count_z` | 运行时 |
| 6 | 36 | 2 | `hidden_group_size_x` | 运行时 |
| 7 | 38 | 2 | `hidden_group_size_y` | 运行时 |
| 8 | 40 | 2 | `hidden_group_size_z` | 运行时 |
| 9 | 42 | 2 | `hidden_remainder_x` | 运行时 |
| 10 | 44 | 2 | `hidden_remainder_y` | 运行时 |
| 11 | 46 | 2 | `hidden_remainder_z` | 运行时 |
| 12 | 64 | 8 | `hidden_global_offset_x` | 运行时 |
| 13 | 72 | 8 | `hidden_global_offset_y` | 运行时 |
| 14 | 80 | 8 | `hidden_global_offset_z` | 运行时 |
| 15 | 88 | 2 | `hidden_grid_dims` | 运行时 |

#### `sigmoid_mul_k`

- mangled 符号：`_Z13sigmoid_mul_kPfPKfS1_x`
- 签名：`sigmoid_mul_k(float*, float const*, float const*, long long)`
- kernarg：288 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `by_value` | 调用方 |
| 4 | 32 | 4 | `hidden_block_count_x` | 运行时 |
| 5 | 36 | 4 | `hidden_block_count_y` | 运行时 |
| 6 | 40 | 4 | `hidden_block_count_z` | 运行时 |
| 7 | 44 | 2 | `hidden_group_size_x` | 运行时 |
| 8 | 46 | 2 | `hidden_group_size_y` | 运行时 |
| 9 | 48 | 2 | `hidden_group_size_z` | 运行时 |
| 10 | 50 | 2 | `hidden_remainder_x` | 运行时 |
| 11 | 52 | 2 | `hidden_remainder_y` | 运行时 |
| 12 | 54 | 2 | `hidden_remainder_z` | 运行时 |
| 13 | 72 | 8 | `hidden_global_offset_x` | 运行时 |
| 14 | 80 | 8 | `hidden_global_offset_y` | 运行时 |
| 15 | 88 | 8 | `hidden_global_offset_z` | 运行时 |
| 16 | 96 | 2 | `hidden_grid_dims` | 运行时 |

#### `silu_mul_k`

- mangled 符号：`_Z10silu_mul_kPfPKfS1_x`
- 签名：`silu_mul_k(float*, float const*, float const*, long long)`
- kernarg：288 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `by_value` | 调用方 |
| 4 | 32 | 4 | `hidden_block_count_x` | 运行时 |
| 5 | 36 | 4 | `hidden_block_count_y` | 运行时 |
| 6 | 40 | 4 | `hidden_block_count_z` | 运行时 |
| 7 | 44 | 2 | `hidden_group_size_x` | 运行时 |
| 8 | 46 | 2 | `hidden_group_size_y` | 运行时 |
| 9 | 48 | 2 | `hidden_group_size_z` | 运行时 |
| 10 | 50 | 2 | `hidden_remainder_x` | 运行时 |
| 11 | 52 | 2 | `hidden_remainder_y` | 运行时 |
| 12 | 54 | 2 | `hidden_remainder_z` | 运行时 |
| 13 | 72 | 8 | `hidden_global_offset_x` | 运行时 |
| 14 | 80 | 8 | `hidden_global_offset_y` | 运行时 |
| 15 | 88 | 8 | `hidden_global_offset_z` | 运行时 |
| 16 | 96 | 2 | `hidden_grid_dims` | 运行时 |

#### `softmax_k`

- mangled 符号：`softmax_k`
- 签名：`softmax_k`
- kernarg：32 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 20 | 4 | `by_value` | 调用方 |
| 4 | 24 | 4 | `by_value` | 调用方 |

#### `softmax_vec_k`

- mangled 符号：`softmax_vec_k`
- 签名：`softmax_vec_k`
- kernarg：32 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 20 | 4 | `by_value` | 调用方 |
| 4 | 24 | 4 | `by_value` | 调用方 |

#### `topk_k`

- mangled 符号：`topk_k`
- 签名：`topk_k`
- kernarg：40 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |

### Attention / KV / 视觉塔

#### `attn_pv_part`

- mangled 符号：`attn_pv_part`
- 签名：`attn_pv_part`
- kernarg：96 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `hidden_block_count_x` | 运行时 |
| 8 | 44 | 4 | `hidden_block_count_y` | 运行时 |
| 9 | 48 | 4 | `hidden_block_count_z` | 运行时 |
| 10 | 52 | 2 | `hidden_group_size_x` | 运行时 |
| 11 | 54 | 2 | `hidden_group_size_y` | 运行时 |
| 12 | 56 | 2 | `hidden_group_size_z` | 运行时 |
| 13 | 58 | 2 | `hidden_remainder_x` | 运行时 |
| 14 | 60 | 2 | `hidden_remainder_y` | 运行时 |
| 15 | 62 | 2 | `hidden_remainder_z` | 运行时 |
| 16 | 64 | 8 | `hidden_global_offset_x` | 运行时 |
| 17 | 72 | 8 | `hidden_global_offset_y` | 运行时 |
| 18 | 80 | 8 | `hidden_global_offset_z` | 运行时 |
| 19 | 88 | 2 | `hidden_grid_dims` | 运行时 |

#### `attn_q_quant_k`

- mangled 符号：`_Z14attn_q_quant_kPjPfPKfiiiii`
- 签名：`attn_q_quant_k(unsigned int*, float*, float const*, int, int, int, int, int)`
- kernarg：304 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 48 | 4 | `hidden_block_count_x` | 运行时 |
| 9 | 52 | 4 | `hidden_block_count_y` | 运行时 |
| 10 | 56 | 4 | `hidden_block_count_z` | 运行时 |
| 11 | 60 | 2 | `hidden_group_size_x` | 运行时 |
| 12 | 62 | 2 | `hidden_group_size_y` | 运行时 |
| 13 | 64 | 2 | `hidden_group_size_z` | 运行时 |
| 14 | 66 | 2 | `hidden_remainder_x` | 运行时 |
| 15 | 68 | 2 | `hidden_remainder_y` | 运行时 |
| 16 | 70 | 2 | `hidden_remainder_z` | 运行时 |
| 17 | 88 | 8 | `hidden_global_offset_x` | 运行时 |
| 18 | 96 | 8 | `hidden_global_offset_y` | 运行时 |
| 19 | 104 | 8 | `hidden_global_offset_z` | 运行时 |
| 20 | 112 | 2 | `hidden_grid_dims` | 运行时 |

#### `fa_decode_comb_k`

- mangled 符号：`_Z16fa_decode_comb_kPfPKfS1_S1_ii`
- 签名：`fa_decode_comb_k(float*, float const*, float const*, float const*, int, int)`
- kernarg：40 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |

#### `fa_decode_k`

- mangled 符号：`_Z11fa_decode_kPfS_S_PKjPKfS1_S3_S1_S3_iiiiiiii`
- 签名：`fa_decode_k(float*, float*, float*, unsigned int const*, float const*, unsigned int const*, float const*, unsigned int const*, float const*, int, int, int, int, int, int, int, int)`
- kernarg：104 B，LDS：648 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 8 | `global_buffer` | 调用方 |
| 7 | 56 | 8 | `global_buffer` | 调用方 |
| 8 | 64 | 8 | `global_buffer` | 调用方 |
| 9 | 72 | 4 | `by_value` | 调用方 |
| 10 | 76 | 4 | `by_value` | 调用方 |
| 11 | 80 | 4 | `by_value` | 调用方 |
| 12 | 84 | 4 | `by_value` | 调用方 |
| 13 | 88 | 4 | `by_value` | 调用方 |
| 14 | 92 | 4 | `by_value` | 调用方 |
| 15 | 96 | 4 | `by_value` | 调用方 |
| 16 | 100 | 4 | `by_value` | 调用方 |

#### `fa_decode_rows_comb_k`

- mangled 符号：`_Z21fa_decode_rows_comb_kPfPKfS1_S1_iiiiiim`
- 签名：`fa_decode_rows_comb_k(float*, float const*, float const*, float const*, int, int, int, int, int, int, unsigned long)`
- kernarg：64 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |
| 6 | 40 | 4 | `by_value` | 调用方 |
| 7 | 44 | 4 | `by_value` | 调用方 |
| 8 | 48 | 4 | `by_value` | 调用方 |
| 9 | 52 | 4 | `by_value` | 调用方 |
| 10 | 56 | 8 | `by_value` | 调用方 |

#### `fa_decode_rows_k`

- mangled 符号：`_Z16fa_decode_rows_kPfS_S_PKjPKfS1_S3_S1_S3_S3_miiiiiiiiiimmi`
- 签名：`fa_decode_rows_k(float*, float*, float*, unsigned int const*, float const*, unsigned int const*, float const*, unsigned int const*, float const*, float const*, unsigned long, int, int, int, int, int, int, int, int, int, int, unsigned long, unsigned long, int)`
- kernarg：148 B，LDS：648 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 8 | `global_buffer` | 调用方 |
| 7 | 56 | 8 | `global_buffer` | 调用方 |
| 8 | 64 | 8 | `global_buffer` | 调用方 |
| 9 | 72 | 8 | `global_buffer` | 调用方 |
| 10 | 80 | 8 | `by_value` | 调用方 |
| 11 | 88 | 4 | `by_value` | 调用方 |
| 12 | 92 | 4 | `by_value` | 调用方 |
| 13 | 96 | 4 | `by_value` | 调用方 |
| 14 | 100 | 4 | `by_value` | 调用方 |
| 15 | 104 | 4 | `by_value` | 调用方 |
| 16 | 108 | 4 | `by_value` | 调用方 |
| 17 | 112 | 4 | `by_value` | 调用方 |
| 18 | 116 | 4 | `by_value` | 调用方 |
| 19 | 120 | 4 | `by_value` | 调用方 |
| 20 | 124 | 4 | `by_value` | 调用方 |
| 21 | 128 | 8 | `by_value` | 调用方 |
| 22 | 136 | 8 | `by_value` | 调用方 |
| 23 | 144 | 4 | `by_value` | 调用方 |

#### `fa_int4`

- mangled 符号：`_Z7fa_int4PKjPKfS0_S2_S0_S2_Pfiiiiii`
- 签名：`fa_int4(unsigned int const*, float const*, unsigned int const*, float const*, unsigned int const*, float const*, float*, int, int, int, int, int, int)`
- kernarg：80 B，LDS：58880 B，private：116 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 8 | `global_buffer` | 调用方 |
| 7 | 56 | 4 | `by_value` | 调用方 |
| 8 | 60 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `by_value` | 调用方 |
| 10 | 68 | 4 | `by_value` | 调用方 |
| 11 | 72 | 4 | `by_value` | 调用方 |
| 12 | 76 | 4 | `by_value` | 调用方 |

#### `gather_heads_k`

- mangled 符号：`_Z14gather_heads_kPfPKfiiiiix`
- 签名：`gather_heads_k(float*, float const*, int, int, int, int, int, long long)`
- kernarg：304 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 20 | 4 | `by_value` | 调用方 |
| 4 | 24 | 4 | `by_value` | 调用方 |
| 5 | 28 | 4 | `by_value` | 调用方 |
| 6 | 32 | 4 | `by_value` | 调用方 |
| 7 | 40 | 8 | `by_value` | 调用方 |
| 8 | 48 | 4 | `hidden_block_count_x` | 运行时 |
| 9 | 52 | 4 | `hidden_block_count_y` | 运行时 |
| 10 | 56 | 4 | `hidden_block_count_z` | 运行时 |
| 11 | 60 | 2 | `hidden_group_size_x` | 运行时 |
| 12 | 62 | 2 | `hidden_group_size_y` | 运行时 |
| 13 | 64 | 2 | `hidden_group_size_z` | 运行时 |
| 14 | 66 | 2 | `hidden_remainder_x` | 运行时 |
| 15 | 68 | 2 | `hidden_remainder_y` | 运行时 |
| 16 | 70 | 2 | `hidden_remainder_z` | 运行时 |
| 17 | 88 | 8 | `hidden_global_offset_x` | 运行时 |
| 18 | 96 | 8 | `hidden_global_offset_y` | 运行时 |
| 19 | 104 | 8 | `hidden_global_offset_z` | 运行时 |
| 20 | 112 | 2 | `hidden_grid_dims` | 运行时 |

#### `gather_rows_k`

- mangled 符号：`gather_rows_k`
- 签名：`gather_rows_k`
- kernarg：32 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |

#### `kv_append_k_k`

- mangled 符号：`_Z13kv_append_k_kPjPfPKfiiiiii`
- 签名：`kv_append_k_k(unsigned int*, float*, float const*, int, int, int, int, int, int)`
- kernarg：48 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 44 | 4 | `by_value` | 调用方 |

#### `kv_append_v_k`

- mangled 符号：`_Z13kv_append_v_kPjPfS0_PKfiiiiii`
- 签名：`kv_append_v_k(unsigned int*, float*, float*, float const*, int, int, int, int, int, int)`
- kernarg：56 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |
| 6 | 40 | 4 | `by_value` | 调用方 |
| 7 | 44 | 4 | `by_value` | 调用方 |
| 8 | 48 | 4 | `by_value` | 调用方 |
| 9 | 52 | 4 | `by_value` | 调用方 |

#### `scatter_heads_k`

- mangled 符号：`_Z15scatter_heads_kPfPKfiiiiix`
- 签名：`scatter_heads_k(float*, float const*, int, int, int, int, int, long long)`
- kernarg：304 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 20 | 4 | `by_value` | 调用方 |
| 4 | 24 | 4 | `by_value` | 调用方 |
| 5 | 28 | 4 | `by_value` | 调用方 |
| 6 | 32 | 4 | `by_value` | 调用方 |
| 7 | 40 | 8 | `by_value` | 调用方 |
| 8 | 48 | 4 | `hidden_block_count_x` | 运行时 |
| 9 | 52 | 4 | `hidden_block_count_y` | 运行时 |
| 10 | 56 | 4 | `hidden_block_count_z` | 运行时 |
| 11 | 60 | 2 | `hidden_group_size_x` | 运行时 |
| 12 | 62 | 2 | `hidden_group_size_y` | 运行时 |
| 13 | 64 | 2 | `hidden_group_size_z` | 运行时 |
| 14 | 66 | 2 | `hidden_remainder_x` | 运行时 |
| 15 | 68 | 2 | `hidden_remainder_y` | 运行时 |
| 16 | 70 | 2 | `hidden_remainder_z` | 运行时 |
| 17 | 88 | 8 | `hidden_global_offset_x` | 运行时 |
| 18 | 96 | 8 | `hidden_global_offset_y` | 运行时 |
| 19 | 104 | 8 | `hidden_global_offset_z` | 运行时 |
| 20 | 112 | 2 | `hidden_grid_dims` | 运行时 |

#### `vit_attn_kernel`

- mangled 符号：`_Z15vit_attn_kernelPfPKfiiiif`
- 签名：`vit_attn_kernel(float*, float const*, int, int, int, int, float)`
- kernarg：296 B，LDS：37888 B，private：304 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 20 | 4 | `by_value` | 调用方 |
| 4 | 24 | 4 | `by_value` | 调用方 |
| 5 | 28 | 4 | `by_value` | 调用方 |
| 6 | 32 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `hidden_block_count_x` | 运行时 |
| 8 | 44 | 4 | `hidden_block_count_y` | 运行时 |
| 9 | 48 | 4 | `hidden_block_count_z` | 运行时 |
| 10 | 52 | 2 | `hidden_group_size_x` | 运行时 |
| 11 | 54 | 2 | `hidden_group_size_y` | 运行时 |
| 12 | 56 | 2 | `hidden_group_size_z` | 运行时 |
| 13 | 58 | 2 | `hidden_remainder_x` | 运行时 |
| 14 | 60 | 2 | `hidden_remainder_y` | 运行时 |
| 15 | 62 | 2 | `hidden_remainder_z` | 运行时 |
| 16 | 80 | 8 | `hidden_global_offset_x` | 运行时 |
| 17 | 88 | 8 | `hidden_global_offset_y` | 运行时 |
| 18 | 96 | 8 | `hidden_global_offset_z` | 运行时 |
| 19 | 104 | 2 | `hidden_grid_dims` | 运行时 |

#### `vit_bias_kernel`

- mangled 符号：`_Z15vit_bias_kernelPfPKfii`
- 签名：`vit_bias_kernel(float*, float const*, int, int)`
- kernarg：280 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 20 | 4 | `by_value` | 调用方 |
| 4 | 24 | 4 | `hidden_block_count_x` | 运行时 |
| 5 | 28 | 4 | `hidden_block_count_y` | 运行时 |
| 6 | 32 | 4 | `hidden_block_count_z` | 运行时 |
| 7 | 36 | 2 | `hidden_group_size_x` | 运行时 |
| 8 | 38 | 2 | `hidden_group_size_y` | 运行时 |
| 9 | 40 | 2 | `hidden_group_size_z` | 运行时 |
| 10 | 42 | 2 | `hidden_remainder_x` | 运行时 |
| 11 | 44 | 2 | `hidden_remainder_y` | 运行时 |
| 12 | 46 | 2 | `hidden_remainder_z` | 运行时 |
| 13 | 64 | 8 | `hidden_global_offset_x` | 运行时 |
| 14 | 72 | 8 | `hidden_global_offset_y` | 运行时 |
| 15 | 80 | 8 | `hidden_global_offset_z` | 运行时 |
| 16 | 88 | 2 | `hidden_grid_dims` | 运行时 |

#### `vit_bias_s_kernel`

- mangled 符号：`_Z17vit_bias_s_kernelPfPKfiii`
- 签名：`vit_bias_s_kernel(float*, float const*, int, int, int)`
- kernarg：288 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 4 | `by_value` | 调用方 |
| 3 | 20 | 4 | `by_value` | 调用方 |
| 4 | 24 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `hidden_block_count_x` | 运行时 |
| 6 | 36 | 4 | `hidden_block_count_y` | 运行时 |
| 7 | 40 | 4 | `hidden_block_count_z` | 运行时 |
| 8 | 44 | 2 | `hidden_group_size_x` | 运行时 |
| 9 | 46 | 2 | `hidden_group_size_y` | 运行时 |
| 10 | 48 | 2 | `hidden_group_size_z` | 运行时 |
| 11 | 50 | 2 | `hidden_remainder_x` | 运行时 |
| 12 | 52 | 2 | `hidden_remainder_y` | 运行时 |
| 13 | 54 | 2 | `hidden_remainder_z` | 运行时 |
| 14 | 72 | 8 | `hidden_global_offset_x` | 运行时 |
| 15 | 80 | 8 | `hidden_global_offset_y` | 运行时 |
| 16 | 88 | 8 | `hidden_global_offset_z` | 运行时 |
| 17 | 96 | 2 | `hidden_grid_dims` | 运行时 |

#### `vit_gelu_kernel`

- mangled 符号：`_Z15vit_gelu_kernelPfPKfxi`
- 签名：`vit_gelu_kernel(float*, float const*, long long, int)`
- kernarg：288 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `by_value` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 32 | 4 | `hidden_block_count_x` | 运行时 |
| 5 | 36 | 4 | `hidden_block_count_y` | 运行时 |
| 6 | 40 | 4 | `hidden_block_count_z` | 运行时 |
| 7 | 44 | 2 | `hidden_group_size_x` | 运行时 |
| 8 | 46 | 2 | `hidden_group_size_y` | 运行时 |
| 9 | 48 | 2 | `hidden_group_size_z` | 运行时 |
| 10 | 50 | 2 | `hidden_remainder_x` | 运行时 |
| 11 | 52 | 2 | `hidden_remainder_y` | 运行时 |
| 12 | 54 | 2 | `hidden_remainder_z` | 运行时 |
| 13 | 72 | 8 | `hidden_global_offset_x` | 运行时 |
| 14 | 80 | 8 | `hidden_global_offset_y` | 运行时 |
| 15 | 88 | 8 | `hidden_global_offset_z` | 运行时 |
| 16 | 96 | 2 | `hidden_grid_dims` | 运行时 |

#### `vit_linear_f16_kernel`

- mangled 符号：`_Z21vit_linear_f16_kernelPfPK6__halfPKfiiiii`
- 签名：`vit_linear_f16_kernel(float*, __half const*, float const*, int, int, int, int, int)`
- kernarg：304 B，LDS：8192 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `by_value` | 调用方 |
| 8 | 48 | 4 | `hidden_block_count_x` | 运行时 |
| 9 | 52 | 4 | `hidden_block_count_y` | 运行时 |
| 10 | 56 | 4 | `hidden_block_count_z` | 运行时 |
| 11 | 60 | 2 | `hidden_group_size_x` | 运行时 |
| 12 | 62 | 2 | `hidden_group_size_y` | 运行时 |
| 13 | 64 | 2 | `hidden_group_size_z` | 运行时 |
| 14 | 66 | 2 | `hidden_remainder_x` | 运行时 |
| 15 | 68 | 2 | `hidden_remainder_y` | 运行时 |
| 16 | 70 | 2 | `hidden_remainder_z` | 运行时 |
| 17 | 88 | 8 | `hidden_global_offset_x` | 运行时 |
| 18 | 96 | 8 | `hidden_global_offset_y` | 运行时 |
| 19 | 104 | 8 | `hidden_global_offset_z` | 运行时 |
| 20 | 112 | 2 | `hidden_grid_dims` | 运行时 |

#### `vit_ln_kernel`

- mangled 符号：`_Z13vit_ln_kernelPfPKfS1_S1_iif`
- 签名：`vit_ln_kernel(float*, float const*, float const*, float const*, int, int, float)`
- kernarg：304 B，LDS：1024 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |
| 6 | 40 | 4 | `by_value` | 调用方 |
| 7 | 48 | 4 | `hidden_block_count_x` | 运行时 |
| 8 | 52 | 4 | `hidden_block_count_y` | 运行时 |
| 9 | 56 | 4 | `hidden_block_count_z` | 运行时 |
| 10 | 60 | 2 | `hidden_group_size_x` | 运行时 |
| 11 | 62 | 2 | `hidden_group_size_y` | 运行时 |
| 12 | 64 | 2 | `hidden_group_size_z` | 运行时 |
| 13 | 66 | 2 | `hidden_remainder_x` | 运行时 |
| 14 | 68 | 2 | `hidden_remainder_y` | 运行时 |
| 15 | 70 | 2 | `hidden_remainder_z` | 运行时 |
| 16 | 88 | 8 | `hidden_global_offset_x` | 运行时 |
| 17 | 96 | 8 | `hidden_global_offset_y` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_z` | 运行时 |
| 19 | 112 | 2 | `hidden_grid_dims` | 运行时 |

#### `vit_rope_kernel`

- mangled 符号：`_Z15vit_rope_kernelPfPKfS1_iiii`
- 签名：`vit_rope_kernel(float*, float const*, float const*, int, int, int, int)`
- kernarg：296 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 4 | `by_value` | 调用方 |
| 4 | 28 | 4 | `by_value` | 调用方 |
| 5 | 32 | 4 | `by_value` | 调用方 |
| 6 | 36 | 4 | `by_value` | 调用方 |
| 7 | 40 | 4 | `hidden_block_count_x` | 运行时 |
| 8 | 44 | 4 | `hidden_block_count_y` | 运行时 |
| 9 | 48 | 4 | `hidden_block_count_z` | 运行时 |
| 10 | 52 | 2 | `hidden_group_size_x` | 运行时 |
| 11 | 54 | 2 | `hidden_group_size_y` | 运行时 |
| 12 | 56 | 2 | `hidden_group_size_z` | 运行时 |
| 13 | 58 | 2 | `hidden_remainder_x` | 运行时 |
| 14 | 60 | 2 | `hidden_remainder_y` | 运行时 |
| 15 | 62 | 2 | `hidden_remainder_z` | 运行时 |
| 16 | 80 | 8 | `hidden_global_offset_x` | 运行时 |
| 17 | 88 | 8 | `hidden_global_offset_y` | 运行时 |
| 18 | 96 | 8 | `hidden_global_offset_z` | 运行时 |
| 19 | 104 | 2 | `hidden_grid_dims` | 运行时 |

### 序列模型 / 卷积

#### `conv1d_silu_k`

- mangled 符号：`_Z13conv1d_silu_kPfPKfS1_S1_iii`
- 签名：`conv1d_silu_k(float*, float const*, float const*, float const*, int, int, int)`
- kernarg：304 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |
| 6 | 40 | 4 | `by_value` | 调用方 |
| 7 | 48 | 4 | `hidden_block_count_x` | 运行时 |
| 8 | 52 | 4 | `hidden_block_count_y` | 运行时 |
| 9 | 56 | 4 | `hidden_block_count_z` | 运行时 |
| 10 | 60 | 2 | `hidden_group_size_x` | 运行时 |
| 11 | 62 | 2 | `hidden_group_size_y` | 运行时 |
| 12 | 64 | 2 | `hidden_group_size_z` | 运行时 |
| 13 | 66 | 2 | `hidden_remainder_x` | 运行时 |
| 14 | 68 | 2 | `hidden_remainder_y` | 运行时 |
| 15 | 70 | 2 | `hidden_remainder_z` | 运行时 |
| 16 | 88 | 8 | `hidden_global_offset_x` | 运行时 |
| 17 | 96 | 8 | `hidden_global_offset_y` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_z` | 运行时 |
| 19 | 112 | 2 | `hidden_grid_dims` | 运行时 |

#### `conv_state_update_k`

- mangled 符号：`_Z19conv_state_update_kPfS_PKfS1_iii`
- 签名：`conv_state_update_k(float*, float*, float const*, float const*, int, int, int)`
- kernarg：304 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |
| 6 | 40 | 4 | `by_value` | 调用方 |
| 7 | 48 | 4 | `hidden_block_count_x` | 运行时 |
| 8 | 52 | 4 | `hidden_block_count_y` | 运行时 |
| 9 | 56 | 4 | `hidden_block_count_z` | 运行时 |
| 10 | 60 | 2 | `hidden_group_size_x` | 运行时 |
| 11 | 62 | 2 | `hidden_group_size_y` | 运行时 |
| 12 | 64 | 2 | `hidden_group_size_z` | 运行时 |
| 13 | 66 | 2 | `hidden_remainder_x` | 运行时 |
| 14 | 68 | 2 | `hidden_remainder_y` | 运行时 |
| 15 | 70 | 2 | `hidden_remainder_z` | 运行时 |
| 16 | 88 | 8 | `hidden_global_offset_x` | 运行时 |
| 17 | 96 | 8 | `hidden_global_offset_y` | 运行时 |
| 18 | 104 | 8 | `hidden_global_offset_z` | 运行时 |
| 19 | 112 | 2 | `hidden_grid_dims` | 运行时 |

#### `gdn_k`

- mangled 符号：`_Z5gdn_kPfPKfS1_S1_S1_S1_S_iiiiii`
- 签名：`gdn_k(float*, float const*, float const*, float const*, float const*, float const*, float*, int, int, int, int, int, int)`
- kernarg：336 B，LDS：0 B，private：528 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 8 | `global_buffer` | 调用方 |
| 7 | 56 | 4 | `by_value` | 调用方 |
| 8 | 60 | 4 | `by_value` | 调用方 |
| 9 | 64 | 4 | `by_value` | 调用方 |
| 10 | 68 | 4 | `by_value` | 调用方 |
| 11 | 72 | 4 | `by_value` | 调用方 |
| 12 | 76 | 4 | `by_value` | 调用方 |
| 13 | 80 | 4 | `hidden_block_count_x` | 运行时 |
| 14 | 84 | 4 | `hidden_block_count_y` | 运行时 |
| 15 | 88 | 4 | `hidden_block_count_z` | 运行时 |
| 16 | 92 | 2 | `hidden_group_size_x` | 运行时 |
| 17 | 94 | 2 | `hidden_group_size_y` | 运行时 |
| 18 | 96 | 2 | `hidden_group_size_z` | 运行时 |
| 19 | 98 | 2 | `hidden_remainder_x` | 运行时 |
| 20 | 100 | 2 | `hidden_remainder_y` | 运行时 |
| 21 | 102 | 2 | `hidden_remainder_z` | 运行时 |
| 22 | 120 | 8 | `hidden_global_offset_x` | 运行时 |
| 23 | 128 | 8 | `hidden_global_offset_y` | 运行时 |
| 24 | 136 | 8 | `hidden_global_offset_z` | 运行时 |
| 25 | 144 | 2 | `hidden_grid_dims` | 运行时 |

#### `gdn_k2<32>`

- mangled 符号：`_Z6gdn_k2ILi32EEvPfPKfS2_S2_S2_S2_S0_S0_iiiii`
- 签名：`void gdn_k2<32>(float*, float const*, float const*, float const*, float const*, float const*, float*, float*, int, int, int, int, int)`
- kernarg：84 B，LDS：512 B，private：144 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 8 | `global_buffer` | 调用方 |
| 7 | 56 | 8 | `global_buffer` | 调用方 |
| 8 | 64 | 4 | `by_value` | 调用方 |
| 9 | 68 | 4 | `by_value` | 调用方 |
| 10 | 72 | 4 | `by_value` | 调用方 |
| 11 | 76 | 4 | `by_value` | 调用方 |
| 12 | 80 | 4 | `by_value` | 调用方 |

#### `split_qkv_k`

- mangled 符号：`split_qkv_k`
- 签名：`split_qkv_k`
- kernarg：104 B，LDS：0 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 4 | `by_value` | 调用方 |
| 5 | 36 | 4 | `by_value` | 调用方 |
| 6 | 40 | 4 | `by_value` | 调用方 |
| 7 | 44 | 4 | `by_value` | 调用方 |
| 8 | 48 | 4 | `hidden_block_count_x` | 运行时 |
| 9 | 52 | 4 | `hidden_block_count_y` | 运行时 |
| 10 | 56 | 4 | `hidden_block_count_z` | 运行时 |
| 11 | 60 | 2 | `hidden_group_size_x` | 运行时 |
| 12 | 62 | 2 | `hidden_group_size_y` | 运行时 |
| 13 | 64 | 2 | `hidden_group_size_z` | 运行时 |
| 14 | 66 | 2 | `hidden_remainder_x` | 运行时 |
| 15 | 68 | 2 | `hidden_remainder_y` | 运行时 |
| 16 | 70 | 2 | `hidden_remainder_z` | 运行时 |
| 17 | 72 | 8 | `hidden_global_offset_x` | 运行时 |
| 18 | 80 | 8 | `hidden_global_offset_y` | 运行时 |
| 19 | 88 | 8 | `hidden_global_offset_z` | 运行时 |
| 20 | 96 | 2 | `hidden_grid_dims` | 运行时 |

#### `ssm_ab_gate_k`

- mangled 符号：`_Z13ssm_ab_gate_kPfS_S_PKfS1_S1_S1_S1_ii`
- 签名：`ssm_ab_gate_k(float*, float*, float*, float const*, float const*, float const*, float const*, float const*, int, int)`
- kernarg：328 B，LDS：64 B，private：0 B

| # | offset | size | kind | 由谁填 |
|---:|---:|---:|---|---|
| 0 | 0 | 8 | `global_buffer` | 调用方 |
| 1 | 8 | 8 | `global_buffer` | 调用方 |
| 2 | 16 | 8 | `global_buffer` | 调用方 |
| 3 | 24 | 8 | `global_buffer` | 调用方 |
| 4 | 32 | 8 | `global_buffer` | 调用方 |
| 5 | 40 | 8 | `global_buffer` | 调用方 |
| 6 | 48 | 8 | `global_buffer` | 调用方 |
| 7 | 56 | 8 | `global_buffer` | 调用方 |
| 8 | 64 | 4 | `by_value` | 调用方 |
| 9 | 68 | 4 | `by_value` | 调用方 |
| 10 | 72 | 4 | `hidden_block_count_x` | 运行时 |
| 11 | 76 | 4 | `hidden_block_count_y` | 运行时 |
| 12 | 80 | 4 | `hidden_block_count_z` | 运行时 |
| 13 | 84 | 2 | `hidden_group_size_x` | 运行时 |
| 14 | 86 | 2 | `hidden_group_size_y` | 运行时 |
| 15 | 88 | 2 | `hidden_group_size_z` | 运行时 |
| 16 | 90 | 2 | `hidden_remainder_x` | 运行时 |
| 17 | 92 | 2 | `hidden_remainder_y` | 运行时 |
| 18 | 94 | 2 | `hidden_remainder_z` | 运行时 |
| 19 | 112 | 8 | `hidden_global_offset_x` | 运行时 |
| 20 | 120 | 8 | `hidden_global_offset_y` | 运行时 |
| 21 | 128 | 8 | `hidden_global_offset_z` | 运行时 |
| 22 | 136 | 2 | `hidden_grid_dims` | 运行时 |

### 其他

- `block_exp_sum_k`：`buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4 val:40:4`
- `block_max_k`：`buf:0:8 buf:8:8 val:16:4 val:20:4`
- `div_scalar_k`：`buf:0:8 buf:8:8 val:16:4`
- `flash_dec_comb_k`：`buf:0:8 buf:8:8 buf:16:8 buf:24:8 val:32:4 val:36:4 val:40:4`
- `flash_dec_part_k`：`buf:0:8 buf:8:8 buf:16:8 buf:24:8 buf:32:8 buf:40:8 val:48:4 val:52:4 val:56:4 val:60:4 val:64:4 val:68:4 val:72:4`
- `reduce_max1_k`：`buf:0:8 buf:8:8 val:16:4`
- `reduce_sum1_k`：`buf:0:8 buf:8:8 val:16:4`
- `vt_scatter_k`：`buf:0:8 buf:8:8 val:16:4 val:20:4 val:24:4 val:28:4 val:32:4 val:36:4`

