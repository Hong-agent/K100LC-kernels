---
uid: 584527ec
id: k100lc-kernels.kernels.asm.gemv
parent: k100lc-kernels.kernels.asm
name: {zh: "W4A4 / W4A8 GEMV 内核", en: "W4A4 / W4A8 GEMV Kernels"}
description:
  zh: >
      RT4 格式的 int4 权重 GEMV（解码期主力）：激活量化前置步骤 + 每 warp 处理一行/多行的 W4A4 与 W4A8 变体。
      
  en: >
      RT4-format int4 weight GEMV (the decode-time workhorse): an activation-quantization pre-step plus W4A4 and W4A8 variants processing one or more rows per warp.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.861Z"
fingerprint: c4e040beaf3e1eac7516f5f029ca4e779ed6ff60663d2db30b0dda7ad528daa3
source:
  - path: "kernels/asm/k_gemv/001__Z10quant_act4PKfPhPfii.s"
  - path: "kernels/asm/k_gemv/003__Z9gemv_w4a4ILi1EEvPKjPKtS1_PKfPfii.s"
---
