---
uid: a0dfffc5
id: k100lc-kernels.kernels.asm.nvfp4.gemv-wide
parent: k100lc-kernels.kernels.asm.nvfp4
name: {zh: "NVFP4 GEMV（宽）", en: "NVFP4 GEMV (Wide)"}
description:
  zh: >
      nvfp4_gemv_wide<M,B>：每 lane 覆盖更宽列范围的 NVFP4 GEMV 变体（M=1..4，B=1..2）。
      
  en: >
      nvfp4_gemv_wide<M,B>: NVFP4 GEMV variants with a wider per-lane column range (M=1..4, B=1..2).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.865Z"
fingerprint: d56879404a647020b273f064c3dda6f05320b3e01eb60a3d4419b11a5dfc8250
source:
  - path: "kernels/asm/k_nvfp4/003__Z15nvfp4_gemv_wideILi1ELi1EEvPKjPKhPKaS5_PKfPfiif.s"
  - path: "kernels/asm/k_nvfp4/004__Z15nvfp4_gemv_wideILi2ELi1EEvPKjPKhPKaS5_PKfPfiif.s"
  - path: "kernels/asm/k_nvfp4/005__Z15nvfp4_gemv_wideILi3ELi1EEvPKjPKhPKaS5_PKfPfiif.s"
  - path: "kernels/asm/k_nvfp4/006__Z15nvfp4_gemv_wideILi4ELi1EEvPKjPKhPKaS5_PKfPfiif.s"
  - path: "kernels/asm/k_nvfp4/007__Z15nvfp4_gemv_wideILi1ELi2EEvPKjPKhPKaS5_PKfPfiif.s"
  - path: "kernels/asm/k_nvfp4/008__Z15nvfp4_gemv_wideILi2ELi2EEvPKjPKhPKaS5_PKfPfiif.s"
  - path: "kernels/asm/k_nvfp4/009__Z15nvfp4_gemv_wideILi3ELi2EEvPKjPKhPKaS5_PKfPfiif.s"
  - path: "kernels/asm/k_nvfp4/010__Z15nvfp4_gemv_wideILi4ELi2EEvPKjPKhPKaS5_PKfPfiif.s"
---
