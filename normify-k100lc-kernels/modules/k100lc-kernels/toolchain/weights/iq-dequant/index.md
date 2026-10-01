---
uid: bc52d5dd
id: k100lc-kernels.toolchain.weights.iq-dequant
parent: k100lc-kernels.toolchain.weights
name: {zh: "GGUF 参考解码器", en: "GGUF Reference Decoder"}
description:
  zh: >
      iq_dequant.py：纯 NumPy 的 GGUF 全编码参考解码，是运行时内核的「金标准」——同一块喂两边应逐元素一致。
      
  en: >
      iq_dequant.py: the pure-NumPy reference decoder for every GGUF codec, the golden reference for the runtime kernels — the same block fed to both must match element-wise.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.893Z"
fingerprint: e06a3daff752350ceab3660a1a4df6b12c1f1cf261e7fb1cfb8638b4744a7d81
source:
  - path: "tools/iq_dequant.py"
---
