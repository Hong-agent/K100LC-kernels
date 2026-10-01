---
uid: 5c75bb1d
id: k100lc-kernels.generators.gemv.qdot
parent: k100lc-kernels.generators.gemv
name: {zh: "原生解码点积生成器", en: "Native Decode-and-Dot Generator"}
description:
  zh: >
      gen_gemv_qdot.py：一套「权重不反量化、内核直接按原编码解码并乘激活」的生成框架，覆盖 11 类 GGUF/量化编码，是所有 *_dot_k 内核的产地。
      
  en: >
      gen_gemv_qdot.py: the framework that decodes weights in their native encoding inside the kernel and multiplies by activations, covering 11 GGUF/quant codecs; it produces every *_dot_k kernel.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.876Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
---
