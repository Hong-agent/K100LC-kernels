---
uid: dbe2d155
id: k100lc-kernels.toolchain.weights.iq-dequant.iq-codecs-ref
parent: k100lc-kernels.toolchain.weights.iq-dequant
name: {zh: "IQ 系列参考解码", en: "IQ Codec Reference"}
description:
  zh: >
      IQ3_XXS / IQ2_S / IQ3_S 的参考解码：网格查表 + 符号位还原。
      
  en: >
      Reference decoders for IQ3_XXS / IQ2_S / IQ3_S: grid lookup plus sign-bit restoration.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.893Z"
fingerprint: e06a3daff752350ceab3660a1a4df6b12c1f1cf261e7fb1cfb8638b4744a7d81
source:
  - path: "tools/iq_dequant.py"
    line: 346
    end_line: 426
apis:
  - protocol: file
    path: "tools/iq_dequant.py:dequant_iq2_s"
    description:
      zh: >
          IQ2_S 参考解码
          
      en: >
          Reference IQ2_S decode
          
---
