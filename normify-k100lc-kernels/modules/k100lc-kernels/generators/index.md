---
uid: 6363094e
id: k100lc-kernels.generators
parent: k100lc-kernels
name: {zh: "内核生成器", en: "Kernel Generators"}
description:
  zh: >
      用 Python 直接发射 gfx926 汇编的生成器族：GEMV/GEMM 变体、量化权重原生解码点积、GGUF 各编码解码、归一化/激活/MoE 算子，每个生成器自带上卡自检。
      
  en: >
      Python generators that emit gfx926 assembly: GEMV/GEMM variants, native quantized-weight decode-and-dot kernels, GGUF decoder/quant codecs, and normalization/activation/MoE operators — each with a GPU self-test.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.875Z"
fingerprint: 64f4ed777d25c34695046aa6a2115cf0f6562f77791d057cfc84b73652995766
source:
  - path: "tools/gen_gemv_f32.py"
  - path: "tools/gen_gemv_f32_warp.py"
  - path: "tools/gen_gemv_i8.py"
  - path: "tools/gen_gemv_w4a4_r2.py"
  - path: "tools/gen_gemv_qdot.py"
  - path: "tools/gen_gemm_flat.py"
  - path: "tools/gen_int4_dot.py"
  - path: "tools/gen_layernorm.py"
  - path: "tools/gen_softmax.py"
  - path: "tools/gen_gelu_mul.py"
  - path: "tools/gen_quant_fast.py"
  - path: "tools/gen_moe_route.py"
  - path: "tools/gen_moe_combine.py"
  - path: "tools/gen_router_top10.py"
  - path: "tools/gen_topk.py"
  - path: "tools/gen_gdn_split_norm.py"
  - path: "tools/gen_q2_0_dequant.py"
  - path: "tools/gen_q4_0_dequant.py"
  - path: "tools/gen_q8_0_dequant.py"
  - path: "tools/gen_iq2s_dequant.py"
  - path: "tools/gen_iq3s_dequant.py"
  - path: "tools/gen_iq3xxs_dequant.py"
  - path: "tools/gen_iq4nl_dequant.py"
  - path: "tools/gen_iq4nl_gemv.py"
  - path: "tools/gen_iq4nl_to_i8.py"
  - path: "tools/gen_iq4xs_dequant.py"
---
