---
uid: b0c9eb47
id: k100lc-kernels.kernels.asm.gemv.act-quant.act
parent: k100lc-kernels.kernels.asm.gemv.act-quant
name: {zh: "INT8 激活量化", en: "INT8 Activation Quantization"}
description:
  zh: >
      quant_act：f32 → int8 打包 + 每 128 元素组尺度，供 W4A8 GEMV 使用。
      
  en: >
      quant_act: f32 to packed int8 plus per-128-element group scales, consumed by W4A8 GEMV.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.862Z"
fingerprint: 649c69c3471da8f7b8860ecad5b663d9c087082b0bfeca0d44948a08be9dc39d
source:
  - path: "kernels/asm/k_gemv/002__Z9quant_actPKfPaS1_PfPiii.s"
apis:
  - protocol: rpc
    path: "quant_act"
    description:
      zh: >
          f32 激活 → int8 + 尺度
          
      en: >
          f32 activations to int8 with scales
          
---
