---
uid: 7c069a77
id: k100lc-kernels.toolchain.bench.model-paths
parent: k100lc-kernels.toolchain.bench
name: {zh: "模型路径基准", en: "Model Path Benchmark"}
description:
  zh: >
      bench_model_paths.py：f32 / INT4 W4A16 / W4A8 / W4A4 / 预填充 GEMM 五路耗时与相对 f32 误差对比。
      
  en: >
      bench_model_paths.py: compares f32 / INT4 W4A16 / W4A8 / W4A4 / prefill GEMM in time and relative error against f32.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.894Z"
fingerprint: f92f4bf260e801feb3ab75eb36e72c9816db71776b964990f14c85270abd011b
source:
  - path: "tools/bench_model_paths.py"
apis:
  - protocol: file
    path: "tools/bench_model_paths.py"
    description:
      zh: >
          五条线性层通路对比
          
      en: >
          Compares five linear-layer paths
          
deps:
  - kind: call
    to: k100lc-kernels.python.model.linear.dot
    from_api: "file:tools/bench_model_paths.py"
    to_api: "file:k100lc_kernels.model.DotLinear.forward_device"
    label: {zh: "对比量化点积路径", en: "Compares the decode-and-dot"}
  - kind: call
    to: k100lc-kernels.python.model.linear.int4
    from_api: "file:tools/bench_model_paths.py"
    to_api: "file:k100lc_kernels.model.Int4Linear.forward_device"
    label: {zh: "对比 INT4 三通路", en: "Compares the three INT4 paths"}
---
