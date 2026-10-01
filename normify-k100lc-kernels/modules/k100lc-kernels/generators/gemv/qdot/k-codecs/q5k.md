---
uid: 25a33545
id: k100lc-kernels.generators.gemv.qdot.k-codecs.q5k
parent: k100lc-kernels.generators.gemv.qdot.k-codecs
name: {zh: "Q5_K 点积生成器", en: "Q5_K Dot Generator"}
description:
  zh: >
      gen_q5k_dot_asm()：256 元素 / 176 字节块，在 Q4_K 之上增加高位平面（5bit 码）。
      
  en: >
      gen_q5k_dot_asm(): 256 elements per 176-byte block, adding a high-bit plane on top of Q4_K (5-bit codes).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.878Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 887
    end_line: 947
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:gen_q5k_dot_asm"
    description:
      zh: >
          Q5_K 点积内核生成
          
      en: >
          Generates the Q5_K dot kernel
          
---
