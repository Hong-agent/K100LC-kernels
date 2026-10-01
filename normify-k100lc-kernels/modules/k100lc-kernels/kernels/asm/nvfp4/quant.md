---
uid: 44f79ba3
id: k100lc-kernels.kernels.asm.nvfp4.quant
parent: k100lc-kernels.kernels.asm.nvfp4
name: {zh: "NVFP4 激活量化", en: "NVFP4 Activation Quantization"}
description:
  zh: >
      nvfp4_quant_act：把 f32 激活量化成 NVFP4（每 16 元素一块，输出 4bit 码、块尺度与全局尺度）。
      
  en: >
      nvfp4_quant_act: quantizes f32 activations to NVFP4 (16-element blocks, emitting 4-bit codes, block scales and a global scale).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.864Z"
fingerprint: 276965127e006e23b17aa2ddfc54c3d804306e63d16a00ad621a402f9cc62d46
source:
  - path: "kernels/asm/k_nvfp4/001__Z15nvfp4_quant_actPKfPaS1_Pfii.s"
apis:
  - protocol: rpc
    path: "nvfp4_quant_act"
    description:
      zh: >
          f32 激活 → NVFP4 码 + 尺度
          
      en: >
          f32 activations to NVFP4 codes and scales
          
---
