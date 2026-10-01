---
uid: 9acef23e
id: k100lc-kernels.generators.norm.gelu-mul
parent: k100lc-kernels.generators.norm
name: {zh: "GELU 门控乘生成器", en: "GELU Gated Multiply Generator"}
description:
  zh: >
      gen_gelu_mul.py：tanh 近似 GELU(gate)*up 的逐元素内核（gelu_mul_k）。
      
  en: >
      gen_gelu_mul.py: element-wise tanh-approximation GELU(gate)*up kernel (gelu_mul_k).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.882Z"
fingerprint: 4be1a5d583634736b63b467d8220a613480c8f8138afd53949172bd2471187fd
source:
  - path: "tools/gen_gelu_mul.py"
apis:
  - protocol: file
    path: "tools/gen_gelu_mul.py"
    description:
      zh: >
          生成 gelu_mul_k
          
      en: >
          Generates gelu_mul_k
          
---
