---
uid: 1b4dd29b
id: k100lc-kernels.kernels.asm.new.elementwise
parent: k100lc-kernels.kernels.asm.new
name: {zh: "逐元素算子", en: "Element-wise Operators"}
description:
  zh: >
      grid-stride 的逐元素内核：填充、原地加、缩放、SiLU/Sigmoid 门控乘与两段拼接。
      
  en: >
      Grid-stride element-wise kernels: fill, in-place add, scale, SiLU/Sigmoid gated multiply and two-segment concat.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.867Z"
fingerprint: 1435cfbd85a393ea0aea711613c9a01b181cf9e20e0dfb2aad4d3a4b55f39ea7
source:
  - path: "kernels/asm/k_new/001__Z6fill_kPffx.s"
  - path: "kernels/asm/k_new/002__Z13add_inplace_kPfPKfx.s"
  - path: "kernels/asm/k_new/003__Z9concat2_kPfPKfS1_ix.s"
  - path: "kernels/asm/k_new/004__Z10silu_mul_kPfPKfS1_x.s"
  - path: "kernels/asm/k_new/005__Z13sigmoid_mul_kPfPKfS1_x.s"
  - path: "kernels/asm/k_new/024__Z11scale_mul_kPffx.s"
---
