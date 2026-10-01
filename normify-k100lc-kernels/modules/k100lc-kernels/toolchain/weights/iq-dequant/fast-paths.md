---
uid: 1a4a6578
id: k100lc-kernels.toolchain.weights.iq-dequant.fast-paths
parent: k100lc-kernels.toolchain.weights.iq-dequant
name: {zh: "向量化快路解码", en: "Vectorized Fast Paths"}
description:
  zh: >
      各编码的 NumPy 向量化版本（*_fast）：批量位运算代替逐元素循环，用于大张量快速对账。
      
  en: >
      NumPy-vectorized variants (*_fast) for several codecs: batched bit operations instead of element loops, for fast reconciliation over large tensors.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.893Z"
fingerprint: e06a3daff752350ceab3660a1a4df6b12c1f1cf261e7fb1cfb8638b4744a7d81
source:
  - path: "tools/iq_dequant.py"
    line: 248
    end_line: 345
apis:
  - protocol: file
    path: "tools/iq_dequant.py:dequant_q4_k_fast"
    description:
      zh: >
          Q4_K 向量化解码
          
      en: >
          Vectorized Q4_K decode
          
---
