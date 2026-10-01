---
uid: 732f0cd0
id: k100lc-kernels.kernels.asm.new.elementwise.sigmoid-mul
parent: k100lc-kernels.kernels.asm.new.elementwise
name: {zh: "Sigmoid 门控乘", en: "Sigmoid Gated Multiply"}
description:
  zh: >
      sigmoid_mul_k(y, x, g, n)：y[i] = sigmoid(x[i]) * g[i]。
      
  en: >
      sigmoid_mul_k(y, x, g, n): y[i] = sigmoid(x[i]) * g[i].
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.868Z"
fingerprint: aa980964eea6ef9eb7a4d9cdc300277d3177b8a53c4eedb0dd89fed6d6b53844
source:
  - path: "kernels/asm/k_new/005__Z13sigmoid_mul_kPfPKfS1_x.s"
apis:
  - protocol: rpc
    path: "sigmoid_mul_k"
    description:
      zh: >
          y = sigmoid(x) * g
          
      en: >
          y = sigmoid(x) * g
          
---
