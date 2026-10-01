---
uid: aa139998
id: k100lc-kernels.generators.gguf.iq3s
parent: k100lc-kernels.generators.gguf
name: {zh: "IQ3_S 解码生成器", en: "IQ3_S Dequant Generator"}
description:
  zh: >
      gen_iq3s_dequant.py：256 元素 / 110 字节块，两档尺度 + 512×u32 网格 + 符号字节。
      
  en: >
      gen_iq3s_dequant.py: 256 elements per 110-byte block, two scale tiers with a 512×u32 grid and a sign byte.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.881Z"
fingerprint: a3f3092d9bb1887d497971a8e2e7a56cc7e0a8a816855a47e20f706bf3c219f0
source:
  - path: "tools/gen_iq3s_dequant.py"
apis:
  - protocol: file
    path: "tools/gen_iq3s_dequant.py"
    description:
      zh: >
          生成 iq3s_dequant_k
          
      en: >
          Generates iq3s_dequant_k
          
---
