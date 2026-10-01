---
uid: ceba6921
id: k100lc-kernels.kernels.asm.new.norm
parent: k100lc-kernels.kernels.asm.new
name: {zh: "归一化内核", en: "Normalization Kernels"}
description:
  zh: >
      RMSNorm（含门控变体）与 L2 归一化：每行一个 workgroup，LDS 归约求平方和/范数。
      
  en: >
      RMSNorm (including the gated variant) and L2 normalization: one workgroup per row with LDS-tree reduction of the sum of squares.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.868Z"
fingerprint: 2598b7e8cb12ea47993cdc0e75df943532701a8c379f2890b9b78c595d029897
source:
  - path: "kernels/asm/k_new/006__Z9rmsnorm_kPfPKfS1_ifi.s"
  - path: "kernels/asm/k_new/007__Z8l2norm_kPfif.s"
  - path: "kernels/asm/k_new/025__Z15rmsnorm_gated_kPfPKfS1_S1_if.s"
---
