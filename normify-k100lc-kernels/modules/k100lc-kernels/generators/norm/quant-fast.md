---
uid: b999ffed
id: k100lc-kernels.generators.norm.quant-fast
parent: k100lc-kernels.generators.norm
name: {zh: "快速激活量化生成器", en: "Fast Activation Quantization Generator"}
description:
  zh: >
      gen_quant_fast.py：一个 warp 负责一个 128 元素组、lane 读连续 float2，替换访存完全不合并的 quant_rows_k。
      
  en: >
      gen_quant_fast.py: one warp per 128-element group with lanes reading contiguous float2, replacing the fully uncoalesced quant_rows_k.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.882Z"
fingerprint: 7e598b04e629ef687749428ba4094fca75392dcc4e6f7c3f6086b33796efbe52
source:
  - path: "tools/gen_quant_fast.py"
apis:
  - protocol: file
    path: "tools/gen_quant_fast.py"
    description:
      zh: >
          生成 quant_rows_fast_k
          
      en: >
          Generates quant_rows_fast_k
          
deps:
  - kind: dataflow
    to: k100lc-kernels.kernels.asm.new.quant.rows
    from_api: "file:tools/gen_quant_fast.py"
    label: {zh: "替代访存不合并的版本", en: "Replaces the uncoalesced"}
---
