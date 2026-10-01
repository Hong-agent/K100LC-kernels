---
uid: 8aed5447
id: k100lc-kernels.generators.gemm
parent: k100lc-kernels.generators
name: {zh: "GEMM 生成器", en: "GEMM Generators"}
description:
  zh: >
      预填充矩阵乘的生成器：从基线 gemm_w4a4 派生出绕开 2D grid 限制的 flat 变体。
      
  en: >
      Generators for prefill matrix multiplication: derives the flat variant that avoids the runtime 2D-grid limitation from the baseline gemm_w4a4.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.879Z"
fingerprint: 513ed7fec2a82efa2df303e0af9a929439ba7fc16b0fe54442ed60e34da992d1
source:
  - path: "tools/gen_gemm_flat.py"
---
