---
uid: bdf1a004
id: k100lc-kernels.toolchain.weights.iq-dequant.k-quants
parent: k100lc-kernels.toolchain.weights.iq-dequant
name: {zh: "K 系列参考解码", en: "K-Quant Reference"}
description:
  zh: >
      Q4_K/Q5_K/Q6_K 的参考解码：_scale_min_k4 解析 6bit 子块尺度，逐块还原 4/5/6bit 码。
      
  en: >
      Reference decoders for Q4_K/Q5_K/Q6_K: _scale_min_k4 unpacks the 6-bit sub-block scales and each block restores its 4/5/6-bit codes.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.893Z"
fingerprint: e06a3daff752350ceab3660a1a4df6b12c1f1cf261e7fb1cfb8638b4744a7d81
source:
  - path: "tools/iq_dequant.py"
    line: 137
    end_line: 247
apis:
  - protocol: file
    path: "tools/iq_dequant.py:dequant_q4_k"
    description:
      zh: >
          Q4_K 参考解码
          
      en: >
          Reference Q4_K decode
          
---
