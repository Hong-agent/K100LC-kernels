---
uid: 4ada6125
id: k100lc-kernels.toolchain.bench.moe
parent: k100lc-kernels.toolchain.bench
name: {zh: "MoE 路径基准", en: "MoE Path Benchmark"}
description:
  zh: >
      bench_moe.py：用确定性 round-robin 路由对比稠密全专家与 token→expert 分桶两条路径（均走 W4A4 GEMM）。
      
  en: >
      bench_moe.py: uses deterministic round-robin routing to compare dense all-experts against token-to-expert bucketing, both on W4A4 GEMM.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.894Z"
fingerprint: 2e3b472dacf9a2bbadb5e230df00fcec0493faff602093aa552d29b73737c8f3
source:
  - path: "tools/bench_moe.py"
apis:
  - protocol: file
    path: "tools/bench_moe.py"
    description:
      zh: >
          稠密 vs 分桶 MoE 对比
          
      en: >
          Dense vs bucketed MoE comparison
          
deps:
  - kind: call
    to: k100lc-kernels.python.model.moe
    from_api: "file:tools/bench_moe.py"
    to_api: "file:k100lc_kernels.model.MoEExperts.route"
    label: {zh: "对比稠密与分桶 MoE", en: "Compares dense and bucketed"}
---
