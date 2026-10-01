---
uid: b9c7d5b8
id: k100lc-kernels.generators.gemv.qdot.k-codecs.q4k
parent: k100lc-kernels.generators.gemv.qdot.k-codecs
name: {zh: "Q4_K 点积生成器", en: "Q4_K Dot Generator"}
description:
  zh: >
      gen_q4k_dot_asm()：256 元素 / 144 字节块，最小/尺度对与 6bit 子块尺度的 4bit 解码。
      
  en: >
      gen_q4k_dot_asm(): 256 elements per 144-byte block, decoding 4-bit codes with min/scale pairs and 6-bit sub-block scales.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.878Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 848
    end_line: 886
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:gen_q4k_dot_asm"
    description:
      zh: >
          Q4_K 点积内核生成
          
      en: >
          Generates the Q4_K dot kernel
          
deps:
  - kind: dataflow
    to: k100lc-kernels.kernels.asm.new.gguf-dequant
    from_api: "file:tools/gen_gemv_qdot.py:gen_q4k_dot_asm"
    label: {zh: "同族编码的解码实现", en: "Dequant implementations of the"}
---
