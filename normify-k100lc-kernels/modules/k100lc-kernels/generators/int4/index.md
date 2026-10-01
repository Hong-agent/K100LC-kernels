---
uid: e41e34ad
id: k100lc-kernels.generators.int4
parent: k100lc-kernels.generators
name: {zh: "INT4 W4A16 生成器", en: "INT4 W4A16 Generator"}
description:
  zh: >
      compressed-tensors INT4（group 128）权重的原生解码点积与反量化内核生成器，含主机侧参数推导与上卡对账。
      
  en: >
      Generator for native decode-and-dot and dequant kernels of compressed-tensors INT4 (group 128) weights, with host-side parameter derivation and on-device reconciliation.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.879Z"
fingerprint: 45593fc236bc9c4fdcc774bbad579c2177415e810ce503801255c2a432b25722
source:
  - path: "tools/gen_int4_dot.py"
---
