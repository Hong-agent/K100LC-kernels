---
uid: c9e01f6b
id: k100lc-kernels.compiler.examples
parent: k100lc-kernels.compiler
name: {zh: "DSL 示例", en: "DSL Examples"}
description:
  zh: >
      编译器示例内核：vadd/axpy/silu 等最简算子，以及 q4k/q5k/q6k 反量化——用 DSL 重写已有的手写汇编内核。
      
  en: >
      Example compiler kernels: minimal operators such as vadd/axpy/silu, plus q4k/q5k/q6k dequantization — rewriting existing hand-written assembly kernels in the DSL.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.889Z"
fingerprint: b713d87a02d64a9517b67ab3f0dad10b94634fbd755f5509a1a03dd9401bb9ad
source:
  - path: "compiler/examples/vadd.kkl"
  - path: "compiler/examples/axpy.kkl"
  - path: "compiler/examples/silu.kkl"
  - path: "compiler/examples/q4k_dequant.kkl"
  - path: "compiler/examples/q5k_dequant.kkl"
  - path: "compiler/examples/q6k_dequant.kkl"
  - path: "compiler/examples/run_vadd.py"
---
