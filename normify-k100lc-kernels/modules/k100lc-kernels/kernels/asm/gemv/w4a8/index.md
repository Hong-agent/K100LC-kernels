---
uid: e7d6eb03
id: k100lc-kernels.kernels.asm.gemv.w4a8
parent: k100lc-kernels.kernels.asm.gemv
name: {zh: "W4A8 GEMV 变体", en: "W4A8 GEMV Variants"}
description:
  zh: >
      gemv_w4a8<M,aligned,split>：int4 权重 × int8 激活，模板第二参选择激活是否对齐、第三参选择 K 方向拆分份数。
      
  en: >
      gemv_w4a8<M,aligned,split>: int4 weights × int8 activations; the second template parameter selects activation alignment and the third the K-direction split count.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.863Z"
fingerprint: 7f00e468a911f9a60b5d62abb7434341a97980c709fa4b165ca17f64004cf28a
source:
  - path: "kernels/asm/k_gemv/007__Z9gemv_w4a8ILi1ELb1ELi1EEvPKjPKtS1_S1_PKfPKiPfii.s"
  - path: "kernels/asm/k_gemv/008__Z9gemv_w4a8ILi1ELb0ELi1EEvPKjPKtS1_S1_PKfPKiPfii.s"
  - path: "kernels/asm/k_gemv/008__Z9gemv_w4a8ILi2ELb1ELi2EEvPKjPKtS1_S1_PKfPKiPfii.s"
  - path: "kernels/asm/k_gemv/009__Z9gemv_w4a8ILi2ELb0ELi2EEvPKjPKtS1_S1_PKfPKiPfii.s"
  - path: "kernels/asm/k_gemv/009__Z9gemv_w4a8ILi3ELb1ELi2EEvPKjPKtS1_S1_PKfPKiPfii.s"
  - path: "kernels/asm/k_gemv/010__Z9gemv_w4a8ILi3ELb0ELi2EEvPKjPKtS1_S1_PKfPKiPfii.s"
  - path: "kernels/asm/k_gemv/010__Z9gemv_w4a8ILi4ELb1ELi2EEvPKjPKtS1_S1_PKfPKiPfii.s"
  - path: "kernels/asm/k_gemv/014__Z9gemv_w4a8ILi4ELb0ELi2EEvPKjPKtS1_S1_PKfPKiPfii.s"
---
