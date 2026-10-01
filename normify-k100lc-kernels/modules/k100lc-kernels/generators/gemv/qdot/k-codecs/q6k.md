---
uid: 8c5d6501
id: k100lc-kernels.generators.gemv.qdot.k-codecs.q6k
parent: k100lc-kernels.generators.gemv.qdot.k-codecs
name: {zh: "Q6_K 点积生成器", en: "Q6_K Dot Generator"}
description:
  zh: >
      _gen_q6k_dot_asm/gen_q6k_dot_asm：256 元素 / 210 字节块，6bit 码由低 4 位与 2 位高位平面拼出，逐 16 元素尺度。
      
  en: >
      _gen_q6k_dot_asm/gen_q6k_dot_asm: 256 elements per 210-byte block, assembling 6-bit codes from a low nibble and a 2-bit plane with per-16-element scales.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.877Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 611
    end_line: 718
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:gen_q6k_dot_asm"
    description:
      zh: >
          Q6_K 点积内核生成
          
      en: >
          Generates the Q6_K dot kernel
          
---
