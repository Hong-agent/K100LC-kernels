---
uid: 70b266bf
id: k100lc-kernels.kernels.asm
parent: k100lc-kernels.kernels
name: {zh: "gfx926 汇编内核源", en: "gfx926 Assembly Kernel Sources"}
description:
  zh: >
      六组用自研汇编语法写成的 gfx926 内核源：FlashAttention、W4A4 GEMM、W4A4/W4A8 GEMV、NVFP4、通用算子与视觉塔，由 tools/build_kernels.py 汇编。
      
  en: >
      Six families of gfx926 kernel sources written in our own assembly syntax: FlashAttention, W4A4 GEMM, W4A4/W4A8 GEMV, NVFP4, general operators and the vision tower; assembled by tools/build_kernels.py.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.860Z"
fingerprint: 9232d6854859ee39524b909fe4ed4bf86bf26024aa5d1633e3242c5f2bc85e89
source:
  - path: "kernels/asm/k_fa/001__Z7fa_int4PKjPKfS0_S2_S0_S2_Pfiiiiii.s"
  - path: "kernels/asm/k_gemm/001__Z9gemm_w4a4PKjS0_PKfS2_Pfiii.s"
  - path: "kernels/asm/k_gemv/001__Z10quant_act4PKfPhPfii.s"
  - path: "kernels/asm/k_nvfp4/001__Z15nvfp4_quant_actPKfPaS1_Pfii.s"
  - path: "kernels/asm/k_new/001__Z6fill_kPffx.s"
  - path: "kernels/asm/k_vision/001__Z21vit_linear_f16_kernelPfPK6__halfPKfiiiii.s"
deps:
  - kind: call
    to: k100lc-kernels.asm.cli
    label: {zh: "汇编为机器码", en: "Assembles to machine code"}
---
