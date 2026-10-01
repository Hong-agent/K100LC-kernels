---
uid: e5d309ad
id: k100lc-kernels.kernels.asm.vision
parent: k100lc-kernels.kernels.asm
name: {zh: "视觉塔内核", en: "Vision Tower Kernels"}
description:
  zh: >
      ViT 视觉塔专用内核：f16 线性层、LayerNorm、GELU、bias 加、RoPE 与自注意力，全部按行/warp 映射。
      
  en: >
      ViT vision-tower kernels: f16 linear layer, LayerNorm, GELU, bias add, RoPE and self-attention, all mapped per row/warp.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.872Z"
fingerprint: 7b0076f02825d5baed245d9f9b246e94fdccb18cec351ab95be79ef75699f90b
source:
  - path: "kernels/asm/k_vision/001__Z21vit_linear_f16_kernelPfPK6__halfPKfiiiii.s"
  - path: "kernels/asm/k_vision/007__Z15vit_attn_kernelPfPKfiiiif.s"
---
