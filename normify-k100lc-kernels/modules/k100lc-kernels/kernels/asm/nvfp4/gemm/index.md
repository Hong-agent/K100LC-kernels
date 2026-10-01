---
uid: e7b118fd
id: k100lc-kernels.kernels.asm.nvfp4.gemm
parent: k100lc-kernels.kernels.asm.nvfp4
name: {zh: "NVFP4 GEMM", en: "NVFP4 GEMM"}
description:
  zh: >
      预填充用的 NVFP4 矩阵乘：标准版与不做行对齐（ng）版，LDS 10KB。
      
  en: >
      Prefill NVFP4 matrix multiplication: the standard version and a non-aligned (ng) variant, 10KB LDS.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.864Z"
fingerprint: ee3556fa502eca597daeb815fe021daa80d982a9e7de1c76572981991e3a513c
source:
  - path: "kernels/asm/k_nvfp4/002__Z17nvfp4_gemm_kernelPKjPKhPKaS4_PKfPfiiif.s"
  - path: "kernels/asm/k_nvfp4/025__Z20nvfp4_gemm_kernel_ngPKjPKhPKaS4_PKfPfiiif.s"
---
