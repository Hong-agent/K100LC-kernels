---
uid: 0f6f30f0
id: k100lc-kernels.kernels.asm.new.elementwise.scale-mul
parent: k100lc-kernels.kernels.asm.new.elementwise
name: {zh: "缩放", en: "Scale Multiply"}
description:
  zh: >
      scale_mul_k(y, scale, n)：y[i] *= scale。
      
  en: >
      scale_mul_k(y, scale, n): y[i] *= scale.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.868Z"
fingerprint: 95995f0fedc76ad02ca9295805243e231dab72d5e7132b6018ed4cbd3039e9d2
source:
  - path: "kernels/asm/k_new/024__Z11scale_mul_kPffx.s"
apis:
  - protocol: rpc
    path: "scale_mul_k"
    description:
      zh: >
          逐元素乘以标量
          
      en: >
          Element-wise multiply by a scalar
          
---
