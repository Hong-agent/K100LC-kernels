---
uid: c92977cc
id: k100lc-kernels.kernels.asm.fa.int4
parent: k100lc-kernels.kernels.asm.fa
name: {zh: "INT4 分块注意力", en: "INT4 Tiled Attention"}
description:
  zh: >
      fa_int4：int4 权重的 FlashAttention 主体，分块做 QK^T、在线 softmax 与 PV 累加，LDS 58KB+、带 116B private 段。
      
  en: >
      fa_int4: the int4 FlashAttention body — tiled QK^T, online softmax and PV accumulation, with 58KB+ LDS and a 116B private segment.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.860Z"
fingerprint: 1b061116a166ed87bc2c4932efe05efef99a2872388ac2b909378d5638ff0b1a
source:
  - path: "kernels/asm/k_fa/001__Z7fa_int4PKjPKfS0_S2_S0_S2_Pfiiiiii.s"
apis:
  - protocol: rpc
    path: "fa_int4"
    description:
      zh: >
          int4 分块注意力内核
          
      en: >
          Int4 tiled attention kernel
          
---
