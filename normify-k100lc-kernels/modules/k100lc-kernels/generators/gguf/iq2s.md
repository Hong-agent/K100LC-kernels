---
uid: 6d31cee0
id: k100lc-kernels.generators.gguf.iq2s
parent: k100lc-kernels.generators.gguf
name: {zh: "IQ2_S 解码生成器", en: "IQ2_S Dequant Generator"}
description:
  zh: >
      gen_iq2s_dequant.py：256 元素 / 82 字节块，f16 d + qs[64] + qh[8] + scales[8]，1024×u64 网格查表。
      
  en: >
      gen_iq2s_dequant.py: 256 elements per 82-byte block — f16 d, qs[64], qh[8], scales[8] — decoding through a 1024×u64 grid.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.881Z"
fingerprint: f5e93db660e3fb128772ca6dfbd5191fa7d39e445a1a6487b4fb970f0d323462
source:
  - path: "tools/gen_iq2s_dequant.py"
apis:
  - protocol: file
    path: "tools/gen_iq2s_dequant.py"
    description:
      zh: >
          生成 iq2s_dequant_k
          
      en: >
          Generates iq2s_dequant_k
          
---
