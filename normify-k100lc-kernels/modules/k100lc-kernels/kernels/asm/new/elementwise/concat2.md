---
uid: c03ad113
id: k100lc-kernels.kernels.asm.new.elementwise.concat2
parent: k100lc-kernels.kernels.asm.new.elementwise
name: {zh: "两段拼接", en: "Two-Segment Concat"}
description:
  zh: >
      concat2_k(y, a, b, pre, n)：把 a 的前 pre 个元素与 b 的其余元素拼成 y。
      
  en: >
      concat2_k(y, a, b, pre, n): concatenates the first `pre` elements of a with the rest of b into y.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.868Z"
fingerprint: 2850f3a6c05a1b46a65bc6f5d6a5050634a1b97554c87b1835b79ad66f58ffb9
source:
  - path: "kernels/asm/k_new/003__Z9concat2_kPfPKfS1_ix.s"
apis:
  - protocol: rpc
    path: "concat2_k"
    description:
      zh: >
          两段缓冲区拼接
          
      en: >
          Concatenates two buffer segments
          
---
