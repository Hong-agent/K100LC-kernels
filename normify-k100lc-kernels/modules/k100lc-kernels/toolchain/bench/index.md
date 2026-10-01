---
uid: a3d502c9
id: k100lc-kernels.toolchain.bench
parent: k100lc-kernels.toolchain
name: {zh: "性能基准", en: "Performance Benchmarks"}
description:
  zh: >
      各条推理通路的实测脚本：按层类型的外推、模型级线性层五路对比、MoE 稠密与分桶对比。
      
  en: >
      Measurement scripts for each inference path: per-layer-type extrapolation, the five-way model-level linear comparison, and dense vs bucketed MoE.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.894Z"
fingerprint: 98cdc071907d5801245550651ff1a88af45118c1147a08606ce2788ca8dc8d4f
source:
  - path: "tools/bench_int4_layers.py"
  - path: "tools/bench_model_paths.py"
  - path: "tools/bench_moe.py"
---
