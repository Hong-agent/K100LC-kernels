---
uid: 95c7a65f
id: k100lc-kernels.generators.gemv.qdot.iq-codecs.iq3xxs
parent: k100lc-kernels.generators.gemv.qdot.iq-codecs
name: {zh: "IQ3_XXS 点积生成器", en: "IQ3_XXS Dot Generator"}
description:
  zh: >
      gen_iq3xxs_dot_asm()：256 元素 / 98 字节块，u32 尺度数组 + 256×u32 网格 + 7bit 符号表。
      
  en: >
      gen_iq3xxs_dot_asm(): 256 elements per 98-byte block with a u32 scale array, a 256×u32 grid and a 7-bit sign table.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.877Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 948
    end_line: 1036
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:gen_iq3xxs_dot_asm"
    description:
      zh: >
          IQ3_XXS 点积内核生成
          
      en: >
          Generates the IQ3_XXS dot kernel
          
---
