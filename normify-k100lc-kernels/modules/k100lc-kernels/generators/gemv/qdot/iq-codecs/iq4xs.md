---
uid: ce1ce065
id: k100lc-kernels.generators.gemv.qdot.iq-codecs.iq4xs
parent: k100lc-kernels.generators.gemv.qdot.iq-codecs
name: {zh: "IQ4_XS 点积生成器", en: "IQ4_XS Dot Generator"}
description:
  zh: >
      gen_iq4xs_dot_asm()：256 元素 / 136 字节块，6bit 尺度（scales_h + scales_l）配 IQ4_NL 码本。
      
  en: >
      gen_iq4xs_dot_asm(): 256 elements per 136-byte block, decoding 6-bit scales (scales_h + scales_l) with the IQ4_NL codebook.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.877Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 533
    end_line: 610
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:gen_iq4xs_dot_asm"
    description:
      zh: >
          IQ4_XS 点积内核生成
          
      en: >
          Generates the IQ4_XS dot kernel
          
---
