---
uid: 6457648d
id: k100lc-kernels.generators.gemv.qdot.iq-codecs.iq2s
parent: k100lc-kernels.generators.gemv.qdot.iq-codecs
name: {zh: "IQ2_S 点积生成器", en: "IQ2_S Dot Generator"}
description:
  zh: >
      gen_iq2s_dot_asm()：256 元素 / 82 字节块，半字节两档尺度 + 1024×u64 网格与符号位还原。
      
  en: >
      gen_iq2s_dot_asm(): 256 elements per 82-byte block, two nibble scale tiers with a 1024×u64 grid and sign-bit restoration.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.876Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 348
    end_line: 414
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:gen_iq2s_dot_asm"
    description:
      zh: >
          IQ2_S 点积内核生成
          
      en: >
          Generates the IQ2_S dot kernel
          
---
