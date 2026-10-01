---
uid: 17d2ce74
id: k100lc-kernels.kernels.asm.gemm
parent: k100lc-kernels.kernels.asm
name: {zh: "W4A4 GEMM 内核", en: "W4A4 GEMM Kernels"}
description:
  zh: >
      预填充阶段的 int4×int4 矩阵乘：2D 分块版与绕开运行时 2D grid 限制的 flat 1D 版。
      
  en: >
      Prefill-stage int4×int4 matrix multiplication: the 2D tiled version and a flat 1D variant that avoids the runtime 2D-grid limitation.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.861Z"
fingerprint: 59c75a8bc3d50eebe2dd0e9f090d989904425d3ebd065c34b9d192619e0c3f80
source:
  - path: "kernels/asm/k_gemm/001__Z9gemm_w4a4PKjS0_PKfS2_Pfiii.s"
  - path: "kernels/asm/k_gemm/002__Z14gemm_w4a4_flatPKjS0_PKfS2_Pfiiiii.s"
---
