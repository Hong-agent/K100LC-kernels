---
uid: 304a5d94
id: k100lc-kernels.generators.int4.registry
parent: k100lc-kernels.generators.int4
name: {zh: "INT4 内核清单", en: "INT4 Kernel Registry"}
description:
  zh: >
      KERNELS 注册表：登记 int4_dot_k / int4_dequant_k / reduce_blocks_k 的名称、生成函数与参数布局。
      
  en: >
      The KERNELS registry: records the names, generator functions and argument layouts of int4_dot_k / int4_dequant_k / reduce_blocks_k.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.880Z"
fingerprint: 45593fc236bc9c4fdcc774bbad579c2177415e810ce503801255c2a432b25722
source:
  - path: "tools/gen_int4_dot.py"
    line: 498
    end_line: 505
apis:
  - protocol: file
    path: "tools/gen_int4_dot.py:KERNELS"
    description:
      zh: >
          INT4 内核注册表
          
      en: >
          INT4 kernel registry
          
---
