---
uid: 0f9a0c79
id: k100lc-kernels.generators.gguf.iq4nl-gemv
parent: k100lc-kernels.generators.gguf
name: {zh: "IQ4_NL 原生解码 GEMV 生成器", en: "IQ4_NL Decode GEMV Generator"}
description:
  zh: >
      gen_iq4nl_gemv.py：IQ4_NL 权重不反量化，直接在 GEMV 内查表解码并乘激活（iq4nl_gemv_k）。
      
  en: >
      gen_iq4nl_gemv.py: GEMV that decodes IQ4_NL weights through the codebook on the fly and multiplies by activations (iq4nl_gemv_k).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.881Z"
fingerprint: fb04faaaa71a68181affc3895a5894916f038fe8b30e54bde0b2c763c9bfc21f
source:
  - path: "tools/gen_iq4nl_gemv.py"
apis:
  - protocol: file
    path: "tools/gen_iq4nl_gemv.py"
    description:
      zh: >
          生成 iq4nl_gemv_k
          
      en: >
          Generates iq4nl_gemv_k
          
---
