---
uid: e4979a0e
id: k100lc-kernels.kernels.asm.new.elementwise.add-inplace
parent: k100lc-kernels.kernels.asm.new.elementwise
name: {zh: "原地累加", en: "In-place Add"}
description:
  zh: >
      add_inplace_k(y, x, n)：y[i] += x[i]，用于 partial 累加与残差合并。
      
  en: >
      add_inplace_k(y, x, n): y[i] += x[i], used for partial accumulation and residual merges.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.868Z"
fingerprint: 4c5b4aaa65059654d76707d366850a16186369b4f869a086079dfed70248e468
source:
  - path: "kernels/asm/k_new/002__Z13add_inplace_kPfPKfx.s"
apis:
  - protocol: rpc
    path: "add_inplace_k"
    description:
      zh: >
          y += x 逐元素
          
      en: >
          Element-wise y += x
          
---
