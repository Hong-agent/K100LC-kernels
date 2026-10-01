---
uid: ca81d2fc
id: k100lc-kernels.generators.gemv.f32-warp
parent: k100lc-kernels.generators.gemv
name: {zh: "warp-per-row f32 GEMV 生成器", en: "Warp-per-Row f32 GEMV Generator"}
description:
  zh: >
      gen_gemv_f32_warp.py：让一个 warp 处理一行、lane 交错读，解决相邻 lane 地址相差 k*4 字节的访存灾难。
      
  en: >
      gen_gemv_f32_warp.py: assigns one row per warp with interleaved lane reads, fixing the k*4-byte stride access pattern of the naive version.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.875Z"
fingerprint: e2ae96e4ba86151a0c739280807d17dfd1649cbeddd35961d4e3069b692a78a9
source:
  - path: "tools/gen_gemv_f32_warp.py"
apis:
  - protocol: file
    path: "tools/gen_gemv_f32_warp.py"
    description:
      zh: >
          生成 gemv_f32_warp_k
          
      en: >
          Generates gemv_f32_warp_k
          
---
