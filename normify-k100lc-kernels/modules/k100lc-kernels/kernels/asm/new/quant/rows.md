---
uid: 868a4533
id: k100lc-kernels.kernels.asm.new.quant.rows
parent: k100lc-kernels.kernels.asm.new.quant
name: {zh: "行式 INT4 量化", en: "Row-wise INT4 Quantization"}
description:
  zh: >
      quant_rows_k：每 128 元素一组把激活量化成 int4 打包（W4A4 GEMM 的激活侧）。
      
  en: >
      quant_rows_k: quantizes activations to packed int4 per 128-element group (the activation side of W4A4 GEMM).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.869Z"
fingerprint: 4b6415855c3ccac0028dfc5125aa98a6040ee908ccb68f01c166916465ccb779
source:
  - path: "kernels/asm/k_new/010__Z12quant_rows_kPjPfPKfiiiii.s"
apis:
  - protocol: rpc
    path: "quant_rows_k"
    description:
      zh: >
          激活 → int4（组 128）
          
      en: >
          Activations to int4, group 128
          
---
