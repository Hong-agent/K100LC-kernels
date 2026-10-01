---
uid: 6fca4016
id: k100lc-kernels.generators.gguf.iq4nl-to-i8
parent: k100lc-kernels.generators.gguf
name: {zh: "IQ4_NL → INT8 生成器", en: "IQ4_NL to INT8 Generator"}
description:
  zh: >
      gen_iq4nl_to_i8.py：把 IQ4_NL 权重离线转成 int8 + 每块尺度（iq4nl_to_i8_k），让后续走更快的 int8 通路。
      
  en: >
      gen_iq4nl_to_i8.py: converts IQ4_NL weights offline to int8 plus per-block scales (iq4nl_to_i8_k) so later steps can use the faster int8 path.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.881Z"
fingerprint: 2ecb6732b9b11386eb0c836e7858963fc790b1db5f4f2e834d251376afc10f2e
source:
  - path: "tools/gen_iq4nl_to_i8.py"
apis:
  - protocol: file
    path: "tools/gen_iq4nl_to_i8.py"
    description:
      zh: >
          生成 iq4nl_to_i8_k
          
      en: >
          Generates iq4nl_to_i8_k
          
---
