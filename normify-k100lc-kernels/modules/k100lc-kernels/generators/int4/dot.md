---
uid: 04ad6aa5
id: k100lc-kernels.generators.int4.dot
parent: k100lc-kernels.generators.int4
name: {zh: "INT4 点积内核", en: "INT4 Dot Kernel"}
description:
  zh: >
      gen_int4_dot_asm()：int4_dot_k —— 每 (128 权重块 × 一行) 一个 work-item，解码 4bit 码乘激活写 partial。
      
  en: >
      gen_int4_dot_asm(): int4_dot_k — one work item per (128-weight block × row), decoding 4-bit codes, multiplying by activations and writing a partial.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.879Z"
fingerprint: 45593fc236bc9c4fdcc774bbad579c2177415e810ce503801255c2a432b25722
source:
  - path: "tools/gen_int4_dot.py"
    line: 88
    end_line: 145
apis:
  - protocol: file
    path: "tools/gen_int4_dot.py:gen_int4_dot_asm"
    description:
      zh: >
          生成 int4_dot_k
          
      en: >
          Generates int4_dot_k
          
---
