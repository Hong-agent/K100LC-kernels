---
uid: 1ef931f1
id: k100lc-kernels.kernels.asm.vision.ln
parent: k100lc-kernels.kernels.asm.vision
name: {zh: "ViT LayerNorm", en: "ViT LayerNorm"}
description:
  zh: >
      vit_ln_kernel(y, x, w, b, rows, cols, eps)：视觉塔的 LayerNorm。
      
  en: >
      vit_ln_kernel(y, x, w, b, rows, cols, eps): LayerNorm for the vision tower.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.872Z"
fingerprint: bbd99e65f6f944829899aeaa4bef90996cf31b0df8583fd99b072fd73a6dbc0e
source:
  - path: "kernels/asm/k_vision/002__Z13vit_ln_kernelPfPKfS1_S1_iif.s"
apis:
  - protocol: rpc
    path: "vit_ln_kernel"
    description:
      zh: >
          ViT LayerNorm
          
      en: >
          ViT LayerNorm
          
---
