---
uid: 45f65d11
id: k100lc-kernels.toolchain.weights.iq-tables
parent: k100lc-kernels.toolchain.weights
name: {zh: "IQ 数值表", en: "IQ Lookup Tables"}
description:
  zh: >
      iq_tables.py：从 llama.cpp/ggml 的 ggml-common.h 整理的 kmask/ksigns 与各 IQ 网格码表，内核解码与参考解码共用。
      
  en: >
      iq_tables.py: the kmask/ksigns and per-IQ grid tables transcribed from llama.cpp/ggml’s ggml-common.h, shared by the kernels and the reference decoder.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.893Z"
fingerprint: eca881e2861d210fe9d21e1051fa071d436d77147a3430257532f6945b670aee
source:
  - path: "tools/iq_tables.py"
apis:
  - protocol: file
    path: "tools/iq_tables.py"
    description:
      zh: >
          IQ 码本与符号表
          
      en: >
          IQ codebooks and sign tables
          
---
