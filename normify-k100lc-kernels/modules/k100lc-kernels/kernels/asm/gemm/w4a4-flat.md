---
uid: 2aaa04d9
id: k100lc-kernels.kernels.asm.gemm.w4a4-flat
parent: k100lc-kernels.kernels.asm.gemm
name: {zh: "Flat W4A4 GEMM", en: "Flat W4A4 GEMM"}
description:
  zh: >
      gemm_w4a4_flat：把 2D grid 折叠成 1D 启动（blockIdx.y 在本运行时不可靠），每块拆多次 1D launch 以提高占用率。
      
  en: >
      gemm_w4a4_flat: folds the 2D grid into 1D launches (blockIdx.y is unreliable on this runtime), splitting each tile into several 1D launches for better occupancy.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.861Z"
fingerprint: c05ffd3434a6e206051dc748db62806bad524d74a444f33660a36d3b9df67871
source:
  - path: "kernels/asm/k_gemm/002__Z14gemm_w4a4_flatPKjS0_PKfS2_Pfiiiii.s"
apis:
  - protocol: rpc
    path: "gemm_w4a4_flat"
    description:
      zh: >
          1D 网格 W4A4 GEMM
          
      en: >
          Flat 1D-grid W4A4 GEMM
          
---
