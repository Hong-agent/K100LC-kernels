---
uid: 48b997de
id: k100lc-kernels.python.model.linear
parent: k100lc-kernels.python.model
name: {zh: "线性层族", en: "Linear Layer Family"}
description:
  zh: >
      四种权重形态的线性层封装：f32、量化权重原生解码点积（DotLinear）、compressed-tensors INT4 三通路、RT4 INT4。
      
  en: >
      Linear layers for four weight forms: f32, native decode-and-dot quantized weights (DotLinear), three-path compressed-tensors INT4, and RT4 INT4.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.886Z"
fingerprint: 7f78e9c51e0530a21223674f1f32aac9e9e82a6a467fb358c04e6c3cee1a2dc1
source:
  - path: "python/k100lc_kernels/model.py"
    line: 150
    end_line: 424
---
