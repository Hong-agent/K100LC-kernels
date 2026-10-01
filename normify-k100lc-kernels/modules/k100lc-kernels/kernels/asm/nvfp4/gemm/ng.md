---
uid: 81ff5ba5
id: k100lc-kernels.kernels.asm.nvfp4.gemm.ng
parent: k100lc-kernels.kernels.asm.nvfp4.gemm
name: {zh: "NVFP4 GEMM（非对齐）", en: "NVFP4 GEMM (Non-Aligned)"}
description:
  zh: >
      nvfp4_gemm_kernel_ng：M/N/K 不满足对齐要求时使用的 NVFP4 GEMM。
      
  en: >
      nvfp4_gemm_kernel_ng: the NVFP4 GEMM used when M/N/K do not satisfy the alignment requirement.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.864Z"
fingerprint: b46284bb88318f3f6a2176534579f11cad21a7ddf02fec2e446c06682ff6a7a2
source:
  - path: "kernels/asm/k_nvfp4/025__Z20nvfp4_gemm_kernel_ngPKjPKhPKaS4_PKfPfiiif.s"
apis:
  - protocol: rpc
    path: "nvfp4_gemm_kernel_ng"
    description:
      zh: >
          NVFP4 GEMM（非对齐形状）
          
      en: >
          NVFP4 GEMM for non-aligned shapes
          
---
