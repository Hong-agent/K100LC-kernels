---
uid: 51f086ef
id: k100lc-kernels.kernels.asm.vision.attn
parent: k100lc-kernels.kernels.asm.vision
name: {zh: "ViT 自注意力", en: "ViT Self-Attention"}
description:
  zh: >
      vit_attn_kernel：视觉塔的自注意力前向，private_segment>0（当前运行时无 scratch backing）。
      
  en: >
      vit_attn_kernel: the vision-tower self-attention forward pass; private_segment > 0 (no scratch backing in the current runtime).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.873Z"
fingerprint: 2a65da01e3d96046a17a072f523954a10abb3255b8ea53ac97f5ecbbcc91593a
source:
  - path: "kernels/asm/k_vision/007__Z15vit_attn_kernelPfPKfiiiif.s"
apis:
  - protocol: rpc
    path: "vit_attn_kernel"
    description:
      zh: >
          ViT 自注意力
          
      en: >
          ViT self-attention
          
---
