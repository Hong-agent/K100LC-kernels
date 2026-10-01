---
uid: e662c892
id: k100lc-kernels.generators.gemm.w4a4-flat
parent: k100lc-kernels.generators.gemm
name: {zh: "Flat W4A4 GEMM 生成器", en: "Flat W4A4 GEMM Generator"}
description:
  zh: >
      gen_gemm_flat.py：把 gemm_w4a4 的 2D dim3 grid 折叠成 1D 启动，每块按 128 行多次 launch，恢复占用率。
      
  en: >
      gen_gemm_flat.py: folds the 2D dim3 grid of gemm_w4a4 into 1D launches, splitting each tile by 128 rows to restore occupancy.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.879Z"
fingerprint: 513ed7fec2a82efa2df303e0af9a929439ba7fc16b0fe54442ed60e34da992d1
source:
  - path: "tools/gen_gemm_flat.py"
apis:
  - protocol: file
    path: "tools/gen_gemm_flat.py"
    description:
      zh: >
          生成 gemm_w4a4_flat
          
      en: >
          Generates gemm_w4a4_flat
          
---
