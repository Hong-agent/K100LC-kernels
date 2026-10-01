---
uid: a677fb10
id: k100lc-kernels.kernels.asm.nvfp4
parent: k100lc-kernels.kernels.asm
name: {zh: "NVFP4 内核", en: "NVFP4 Kernels"}
description:
  zh: >
      NVFP4（E2M1 4bit + E4M3 逐块尺度 + f32 全局尺度，块 16 值）的激活量化、GEMV 与 GEMM，模板参数决定每 lane 行数与列宽。
      
  en: >
      NVFP4 (E2M1 4-bit with E4M3 block scales and an f32 global scale, block=16) activation quantization, GEMV and GEMM; template parameters set rows-per-lane and column width.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.864Z"
fingerprint: d251bcd99a83f54c45668f207b4f8aeae1f5c385c045b4ddb9a4234a8349013f
source:
  - path: "kernels/asm/k_nvfp4/001__Z15nvfp4_quant_actPKfPaS1_Pfii.s"
  - path: "kernels/asm/k_nvfp4/011__Z10nvfp4_gemvILi1ELi1EEvPKjPKhPKaS5_PKfPfiif.s"
---
