---
uid: 9b3810b3
id: k100lc-kernels.toolchain.bench.int4-layers
parent: k100lc-kernels.toolchain.bench
name: {zh: "INT4 分层基准", en: "INT4 Per-Layer Benchmark"}
description:
  zh: >
      bench_int4_layers.py：按 checkpoint 里出现的少数几种 (N,K) 形状实测 int4_dot_k，再推算整模型每 token 的权重流时间。
      
  en: >
      bench_int4_layers.py: benchmarks int4_dot_k over the few (N,K) shapes found in the checkpoint and extrapolates the whole-model per-token weight-stream time.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.894Z"
fingerprint: 27a891ea823ab1639a263b62601012b285fa7844afe8089479fce81697b96aa6
source:
  - path: "tools/bench_int4_layers.py"
apis:
  - protocol: file
    path: "tools/bench_int4_layers.py"
    description:
      zh: >
          按层形状外推整模型开销
          
      en: >
          Extrapolates whole-model cost from layer shapes
          
deps:
  - kind: call
    to: k100lc-kernels.toolchain.weights.ct-int4
    from_api: "file:tools/bench_int4_layers.py"
    to_api: "file:tools/ct_int4.py"
    label: {zh: "按层形状取权重", en: "Gets weights per layer shape"}
---
