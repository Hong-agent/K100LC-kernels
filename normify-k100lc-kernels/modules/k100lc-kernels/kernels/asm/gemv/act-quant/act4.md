---
uid: f4dd2e9c
id: k100lc-kernels.kernels.asm.gemv.act-quant.act4
parent: k100lc-kernels.kernels.asm.gemv.act-quant
name: {zh: "INT4 激活量化", en: "INT4 Activation Quantization"}
description:
  zh: >
      quant_act4：f32 → int4 打包 + f32 子组尺度（SG=32），供 W4A4 GEMV/GEMM 使用。
      
  en: >
      quant_act4: f32 to packed int4 plus f32 sub-group scales (SG=32), consumed by W4A4 GEMV/GEMM.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.862Z"
fingerprint: 3b2b0e3a68a994094a9f95a359563d3e557ed7ef118b20127cda0805e617949e
source:
  - path: "kernels/asm/k_gemv/001__Z10quant_act4PKfPhPfii.s"
apis:
  - protocol: rpc
    path: "quant_act4"
    description:
      zh: >
          f32 激活 → int4 + 尺度
          
      en: >
          f32 activations to int4 with scales
          
---
