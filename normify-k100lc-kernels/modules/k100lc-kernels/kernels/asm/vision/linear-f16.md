---
uid: 6b092c1b
id: k100lc-kernels.kernels.asm.vision.linear-f16
parent: k100lc-kernels.kernels.asm.vision
name: {zh: "ViT f16 线性层", en: "ViT f16 Linear"}
description:
  zh: >
      vit_linear_f16_kernel：权重 f16、激活 f32 的线性层（视觉塔投影）。
      
  en: >
      vit_linear_f16_kernel: linear layer with f16 weights and f32 activations (vision-tower projection).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.872Z"
fingerprint: a9d29b34a44561522e951bcc95cca60dc05fa19eb262e6b1e6fedc7b0a5ded3a
source:
  - path: "kernels/asm/k_vision/001__Z21vit_linear_f16_kernelPfPK6__halfPKfiiiii.s"
apis:
  - protocol: rpc
    path: "vit_linear_f16_kernel"
    description:
      zh: >
          ViT f16 线性层
          
      en: >
          ViT f16 linear layer
          
---
