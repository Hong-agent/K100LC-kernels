---
uid: 20f5c33c
id: k100lc-kernels.generators.int4.verify
parent: k100lc-kernels.generators.int4
name: {zh: "INT4 参考与参数推导", en: "INT4 Reference & Parameter Derivation"}
description:
  zh: >
      ref_dot（f32 参考点积）、dense_params（按行数/nbpr 推导网格与共享内存需求）与 _launch_args（把参数序列化成 catalog 风格的 (off,size,kind) 列表）。
      
  en: >
      ref_dot (the f32 reference dot product), dense_params (derives grid and shared-memory needs from rows/nbpr) and _launch_args (serializes arguments into catalog-style (off,size,kind) lists).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.880Z"
fingerprint: 45593fc236bc9c4fdcc774bbad579c2177415e810ce503801255c2a432b25722
source:
  - path: "tools/gen_int4_dot.py"
    line: 219
    end_line: 280
apis:
  - protocol: file
    path: "tools/gen_int4_dot.py:ref_dot"
    description:
      zh: >
          INT4 参考点积
          
      en: >
          Reference INT4 dot product
          
  - protocol: file
    path: "tools/gen_int4_dot.py:dense_params"
    description:
      zh: >
          稠密启动参数推导
          
      en: >
          Derives dense launch parameters
          
---
