---
uid: e33ca302
id: k100lc-kernels.examples.w4a4-gemm-py
parent: k100lc-kernels.examples
name: {zh: "W4A4 GEMM 示例", en: "W4A4 GEMM Example"}
description:
  zh: >
      python_w4a4_gemm.py：RT4 int4 × int4 的预填充 GEMM，按 BM=128 拆 1D 启动，报 TMAC/s 与等效权重带宽。
      
  en: >
      python_w4a4_gemm.py: RT4 int4 × int4 prefill GEMM, split into 1D launches per BM=128 tile, reporting TMAC/s and effective weight bandwidth.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.895Z"
fingerprint: de5fbdf2cd5aaf2f992f632c01fd6a8182f40862bf5abf33afb7ca3a7191f617
source:
  - path: "examples/python_w4a4_gemm.py"
apis:
  - protocol: file
    path: "examples/python_w4a4_gemm.py"
    description:
      zh: >
          W4A4 预填充 GEMM 实测
          
      en: >
          Benchmarks the W4A4 prefill GEMM
          
deps:
  - kind: call
    to: k100lc-kernels.python.rt4.gemm
    from_api: "file:examples/python_w4a4_gemm.py"
    to_api: "file:k100lc_kernels.rt4.W4Runner.gemm"
    label: {zh: "跑 W4A4 预填充 GEMM", en: "Runs the W4A4 prefill GEMM"}
---
