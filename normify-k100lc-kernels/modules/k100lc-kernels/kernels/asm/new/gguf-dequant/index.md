---
uid: 9fd1662a
id: k100lc-kernels.kernels.asm.new.gguf-dequant
parent: k100lc-kernels.kernels.asm.new
name: {zh: "K-量化解码内核", en: "K-Quant Dequant Kernels"}
description:
  zh: >
      GGUF K-系列编码（Q4_K/Q5_K/Q6_K）到 f32 的整块解码，作为参考解码器与编译器示例的对照实现。
      
  en: >
      Whole-block GGUF K-quant (Q4_K/Q5_K/Q6_K) to f32 dequantization, serving as the reference against the decoder and compiler examples.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.872Z"
fingerprint: 8f56147f684029df63108d49803c35cbf1714d297ff19080ecd58840f52951ce
source:
  - path: "kernels/asm/k_new/q4k_dequant.s"
  - path: "kernels/asm/k_new/q5k_dequant.s"
  - path: "kernels/asm/k_new/q6k_dequant.s"
---
