---
uid: 1a5cae05
id: k100lc-kernels.toolchain.weights.iq-dequant.legacy-codecs
parent: k100lc-kernels.toolchain.weights.iq-dequant
name: {zh: "旧编码参考解码", en: "Legacy Codec Reference"}
description:
  zh: >
      Q2_0 / Q4_0 / Q8_0 / IQ4_NL / IQ4_XS 的参考解码，以及 f16/bf16/f32 直通解码函数。
      
  en: >
      Reference decoders for Q2_0 / Q4_0 / Q8_0 / IQ4_NL / IQ4_XS, plus the f16/bf16/f32 passthrough decoders.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.893Z"
fingerprint: e06a3daff752350ceab3660a1a4df6b12c1f1cf261e7fb1cfb8638b4744a7d81
source:
  - path: "tools/iq_dequant.py"
    line: 83
    end_line: 136
  - path: "tools/iq_dequant.py"
    line: 427
    end_line: 439
apis:
  - protocol: file
    path: "tools/iq_dequant.py:dequant_q4_0"
    description:
      zh: >
          Q4_0 参考解码
          
      en: >
          Reference Q4_0 decode
          
---
