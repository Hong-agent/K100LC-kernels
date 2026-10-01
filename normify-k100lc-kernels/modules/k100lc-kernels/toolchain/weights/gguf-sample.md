---
uid: e6fe4ef0
id: k100lc-kernels.toolchain.weights.gguf-sample
parent: k100lc-kernels.toolchain.weights
name: {zh: "真权重定位", en: "Real-Weight Locator"}
description:
  zh: >
      gguf_sample.py：读可选的真权重索引 JSON，定位 GGUF/量化张量在原始文件里的位置，供各生成器的自检使用。
      
  en: >
      gguf_sample.py: reads the optional real-weight index JSON and locates GGUF/quantized tensors inside the source files for the generators’ self-tests.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.893Z"
fingerprint: 53258f5d07eb1416e4b044ceab0d440b3a36bd1b5e995d52e434928bc67e35d3
source:
  - path: "tools/gguf_sample.py"
apis:
  - protocol: file
    path: "tools/gguf_sample.py"
    description:
      zh: >
          按索引取真权重块
          
      en: >
          Fetches real-weight blocks by index
          
deps:
  - kind: dataflow
    to: k100lc-kernels.toolchain.weights.iq-dequant.dispatch
    from_api: "file:tools/gguf_sample.py"
    to_api: "file:tools/iq_dequant.py:dequantize"
    label: {zh: "提供真权重块", en: "Supplies real-weight blocks"}
---
