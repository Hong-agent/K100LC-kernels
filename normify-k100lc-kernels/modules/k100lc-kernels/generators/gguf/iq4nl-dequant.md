---
uid: 260c5550
id: k100lc-kernels.generators.gguf.iq4nl-dequant
parent: k100lc-kernels.generators.gguf
name: {zh: "IQ4_NL 解码生成器", en: "IQ4_NL Dequant Generator"}
description:
  zh: >
      gen_iq4nl_dequant.py：32 元素 / 18 字节块，f16 d + 16 字节 4bit 码查 16 项码本。
      
  en: >
      gen_iq4nl_dequant.py: 32 elements per 18-byte block — f16 d plus 16 bytes of 4-bit codes looked up in a 16-entry codebook.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.881Z"
fingerprint: 417d8e00451d2e05c28d0e6e2f4ab65454dd13685711511768f7cdbc52d2345d
source:
  - path: "tools/gen_iq4nl_dequant.py"
apis:
  - protocol: file
    path: "tools/gen_iq4nl_dequant.py"
    description:
      zh: >
          生成 iq4nl_dequant_k
          
      en: >
          Generates iq4nl_dequant_k
          
---
