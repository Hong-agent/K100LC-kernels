---
uid: edf8e2f8
id: k100lc-kernels.kernels.asm.new.elementwise.silu-mul
parent: k100lc-kernels.kernels.asm.new.elementwise
name: {zh: "SiLU 门控乘", en: "SiLU Gated Multiply"}
description:
  zh: >
      silu_mul_k(y, a, b, n)：y[i] = silu(a[i]) * b[i]，MLP 的 SwiGLU 主体。
      
  en: >
      silu_mul_k(y, a, b, n): y[i] = silu(a[i]) * b[i], the SwiGLU body of the MLP.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.868Z"
fingerprint: 8adfaa706f8ad1292e97d71a7b7c829c88f1c88efadf9a7ffc5940286befad13
source:
  - path: "kernels/asm/k_new/004__Z10silu_mul_kPfPKfS1_x.s"
apis:
  - protocol: rpc
    path: "silu_mul_k"
    description:
      zh: >
          y = silu(a) * b
          
      en: >
          y = silu(a) * b
          
---
