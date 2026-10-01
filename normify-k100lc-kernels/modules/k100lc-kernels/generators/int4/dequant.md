---
uid: e3fbe43d
id: k100lc-kernels.generators.int4.dequant
parent: k100lc-kernels.generators.int4
name: {zh: "INT4 反量化内核", en: "INT4 Dequant Kernel"}
description:
  zh: >
      gen_int4_dequant_asm()：int4_dequant_k —— 把 compressed-tensors INT4 整块解成 f32，作为点积路径的对照实现。
      
  en: >
      gen_int4_dequant_asm(): int4_dequant_k — dequantizes compressed-tensors INT4 blocks to f32 as the reference counterpart of the dot path.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.880Z"
fingerprint: 45593fc236bc9c4fdcc774bbad579c2177415e810ce503801255c2a432b25722
source:
  - path: "tools/gen_int4_dot.py"
    line: 146
    end_line: 218
apis:
  - protocol: file
    path: "tools/gen_int4_dot.py:gen_int4_dequant_asm"
    description:
      zh: >
          生成 int4_dequant_k
          
      en: >
          Generates int4_dequant_k
          
---
