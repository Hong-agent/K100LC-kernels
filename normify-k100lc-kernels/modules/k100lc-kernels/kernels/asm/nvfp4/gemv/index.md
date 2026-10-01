---
uid: 7cdc93f0
id: k100lc-kernels.kernels.asm.nvfp4.gemv
parent: k100lc-kernels.kernels.asm.nvfp4
name: {zh: "NVFP4 GEMV 模板实例", en: "NVFP4 GEMV Template Instances"}
description:
  zh: >
      nvfp4_gemv<M,S>：M=1..4 行、S=1..4 份 K 拆分的 16 个模板实例，workgroup=256、每 workgroup 处理 4 行。
      
  en: >
      nvfp4_gemv<M,S>: 16 template instances for M=1..4 rows and S=1..4 K-splits, workgroup=256 with four rows per workgroup.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.866Z"
fingerprint: c597d139c139e689fce9e18ae4059ea60a67ef6d46310fc199a7e58b866b5422
source:
  - path: "kernels/asm/k_nvfp4/011__Z10nvfp4_gemvILi1ELi1EEvPKjPKhPKaS5_PKfPfiif.s"
  - path: "kernels/asm/k_nvfp4/026__Z10nvfp4_gemvILi4ELi4EEvPKjPKhPKaS5_PKfPfiif.s"
---
