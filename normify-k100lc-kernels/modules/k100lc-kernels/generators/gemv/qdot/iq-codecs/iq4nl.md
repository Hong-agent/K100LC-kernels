---
uid: 6571f5f5
id: k100lc-kernels.generators.gemv.qdot.iq-codecs.iq4nl
parent: k100lc-kernels.generators.gemv.qdot.iq-codecs
name: {zh: "IQ4_NL 点积生成器", en: "IQ4_NL Dot Generator"}
description:
  zh: >
      gen_iq4nl_dot_asm()：32 元素 / 18 字节块，f16 d + 16 字节 4bit 码，查 16 项 kvalues_iq4nl 码本后与激活点积。
      
  en: >
      gen_iq4nl_dot_asm(): 32 elements per 18-byte block — f16 d plus 16 bytes of 4-bit codes — looking up the 16-entry kvalues_iq4nl codebook and dotting with activations.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.876Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 251
    end_line: 325
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:gen_iq4nl_dot_asm"
    description:
      zh: >
          IQ4_NL 点积内核生成
          
      en: >
          Generates the IQ4_NL dot kernel
          
deps:
  - kind: dataflow
    to: k100lc-kernels.kernels.asm.new.gguf-dequant
    from_api: "file:tools/gen_gemv_qdot.py:gen_iq4nl_dot_asm"
    label: {zh: "同族编码的解码实现", en: "Dequant implementations of the"}
---
