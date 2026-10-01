---
uid: bf261cda
id: k100lc-kernels.examples.int4-gemv-py
parent: k100lc-kernels.examples
name: {zh: "真模型 INT4 层示例", en: "Real-Model INT4 Layer Example"}
description:
  zh: >
      python_int4_gemv.py：直接把 compressed-tensors checkpoint 的 weight_packed/weight_scale 原字节搬进显存跑 int4_dot_k + reduce_blocks_k，并报等效带宽。
      
  en: >
      python_int4_gemv.py: uploads the raw weight_packed/weight_scale bytes of a compressed-tensors checkpoint and runs int4_dot_k plus reduce_blocks_k, reporting effective bandwidth.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.895Z"
fingerprint: 1b6956a318f1b4bd81c1995dd377425d5eec7dd3960be8286ee49174b40ce844
source:
  - path: "examples/python_int4_gemv.py"
apis:
  - protocol: file
    path: "examples/python_int4_gemv.py"
    description:
      zh: >
          INT4 checkpoint 原字节直跑
          
      en: >
          Runs raw INT4 checkpoint bytes directly
          
deps:
  - kind: call
    to: k100lc-kernels.generators.int4.dot
    from_api: "file:examples/python_int4_gemv.py"
    to_api: "file:tools/gen_int4_dot.py:gen_int4_dot_asm"
    label: {zh: "直跑 INT4 点积内核", en: "Runs the INT4 dot kernel"}
---
