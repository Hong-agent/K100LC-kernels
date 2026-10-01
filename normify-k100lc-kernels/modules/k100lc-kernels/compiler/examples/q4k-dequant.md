---
uid: 70367edb
id: k100lc-kernels.compiler.examples.q4k-dequant
parent: k100lc-kernels.compiler.examples
name: {zh: "Q4_K 反量化示例", en: "Q4_K Dequant Example"}
description:
  zh: >
      q4k_dequant.kkl：用 DSL 复现手写汇编的 Q4_K 反量化（f16 尺度、6bit 子块尺度、半字节码）。
      
  en: >
      q4k_dequant.kkl: reproduces the hand-written Q4_K dequantization in the DSL (f16 scales, 6-bit sub-block scales, nibble codes).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.890Z"
fingerprint: 93f08e9415b522180cd2a7fc2ec80d497095c36e6e26fed66a1418e9eee41c42
source:
  - path: "compiler/examples/q4k_dequant.kkl"
apis:
  - protocol: file
    path: "compiler/examples/q4k_dequant.kkl:q4k_dequant"
    description:
      zh: >
          Q4_K 反量化示例
          
      en: >
          Q4_K dequant example
          
deps:
  - kind: reference
    to: k100lc-kernels.kernels.asm.new.gguf-dequant.q4k
    from_api: "file:compiler/examples/q4k_dequant.kkl:q4k_dequant"
    to_api: "rpc:q4k_dequant"
    label: {zh: "重写自手写汇编", en: "Rewritten from the"}
---
