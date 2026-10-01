---
uid: ff95605f
id: k100lc-kernels.generators.gemv.qdot.iq-codecs.iq3s
parent: k100lc-kernels.generators.gemv.qdot.iq-codecs
name: {zh: "IQ3_S 点积生成器", en: "IQ3_S Dot Generator"}
description:
  zh: >
      gen_iq3s_dot_asm()：256 元素 / 110 字节块，两档尺度 + 512×u32 网格 + 独立符号字节。
      
  en: >
      gen_iq3s_dot_asm(): 256 elements per 110-byte block, two scale tiers with a 512×u32 grid and a separate sign byte.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.876Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 415
    end_line: 486
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:gen_iq3s_dot_asm"
    description:
      zh: >
          IQ3_S 点积内核生成
          
      en: >
          Generates the IQ3_S dot kernel
          
---
