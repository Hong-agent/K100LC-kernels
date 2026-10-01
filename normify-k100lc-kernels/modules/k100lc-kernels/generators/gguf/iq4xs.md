---
uid: 2d4cade5
id: k100lc-kernels.generators.gguf.iq4xs
parent: k100lc-kernels.generators.gguf
name: {zh: "IQ4_XS 解码生成器", en: "IQ4_XS Dequant Generator"}
description:
  zh: >
      gen_iq4xs_dequant.py：256 元素 / 136 字节块，6bit 尺度（scales_h + scales_l）与 IQ4_NL 码本组合解码。
      
  en: >
      gen_iq4xs_dequant.py: 256 elements per 136-byte block, decoding 6-bit scales (scales_h + scales_l) together with the IQ4_NL codebook.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.881Z"
fingerprint: 0099049d180238830e1dd56a54e962551dc179083ffb15888b2a0f0e2603a212
source:
  - path: "tools/gen_iq4xs_dequant.py"
apis:
  - protocol: file
    path: "tools/gen_iq4xs_dequant.py"
    description:
      zh: >
          生成 iq4xs_dequant_k
          
      en: >
          Generates iq4xs_dequant_k
          
---
