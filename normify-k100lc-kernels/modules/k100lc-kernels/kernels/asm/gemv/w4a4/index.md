---
uid: 53f1016c
id: k100lc-kernels.kernels.asm.gemv.w4a4
parent: k100lc-kernels.kernels.asm.gemv
name: {zh: "W4A4 GEMV 变体", en: "W4A4 GEMV Variants"}
description:
  zh: >
      模板实例化 gemv_w4a4<M>（M=1..4）：每 warp 处理一行权重，激活 int4 × 权重 int4，用 v_dot8 累加。
      
  en: >
      Template instantiations gemv_w4a4<M> (M=1..4): one weight row per warp, int4 activations × int4 weights accumulated with v_dot8.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.862Z"
fingerprint: 0492aa07b4b1bd0d3eb9fb521fecfccd065038c5d4e6ab2e32bb15da0be13996
source:
  - path: "kernels/asm/k_gemv/003__Z9gemv_w4a4ILi1EEvPKjPKtS1_PKfPfii.s"
  - path: "kernels/asm/k_gemv/004__Z9gemv_w4a4ILi2EEvPKjPKtS1_PKfPfii.s"
  - path: "kernels/asm/k_gemv/005__Z9gemv_w4a4ILi3EEvPKjPKtS1_PKfPfii.s"
  - path: "kernels/asm/k_gemv/006__Z9gemv_w4a4ILi4EEvPKjPKtS1_PKfPfii.s"
---
