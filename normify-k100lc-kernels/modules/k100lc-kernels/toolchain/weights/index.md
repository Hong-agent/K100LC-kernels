---
uid: 43f0843c
id: k100lc-kernels.toolchain.weights
parent: k100lc-kernels.toolchain
name: {zh: "权重格式与参考解码", en: "Weight Formats & Reference Decoders"}
description:
  zh: >
      量化权重的主机侧金标准：GGUF 全部编码的 NumPy 参考解码、IQ 数值表、可选的原始权重索引，以及 compressed-tensors INT4 的参考解码与引擎直跑清单。
      
  en: >
      Host-side golden references for quantized weights: NumPy reference decoders for every GGUF codec, the IQ lookup tables, the optional raw-weight index, plus the compressed-tensors INT4 reference decoder and the manifest for direct engine runs.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.892Z"
fingerprint: 9c64d1781975ecb622e0b8edf53f1b90e8dfacbe9b256bd28ed4983e107ea19e
source:
  - path: "tools/gguf_sample.py"
  - path: "tools/iq_tables.py"
  - path: "tools/iq_dequant.py"
  - path: "tools/ct_int4.py"
  - path: "tools/int4_engine_manifest.py"
---
