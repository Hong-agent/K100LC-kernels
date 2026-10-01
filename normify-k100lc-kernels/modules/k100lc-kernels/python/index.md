---
uid: 557308d5
id: k100lc-kernels.python
parent: k100lc-kernels
name: {zh: "Python 封装包", en: "Python Binding Package"}
description:
  zh: >
      k100lc_kernels 包：Runtime 驱动常驻引擎、catalog 暴露内核参数表、model 把内核组合成推理算子，另有量化打包与 RT4 权重读取工具。
      
  en: >
      The k100lc_kernels package: Runtime drives the resident engine, catalog exposes the kernel argument tables, model composes kernels into inference operators, plus quantization packing and RT4 weight loading utilities.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.885Z"
fingerprint: 8245551dba51e7ff487a21c3ca73414b96155b897ec62f40b27268a978751426
source:
  - path: "python/k100lc_kernels/__init__.py"
  - path: "python/k100lc_kernels/runtime.py"
  - path: "python/k100lc_kernels/model.py"
  - path: "python/k100lc_kernels/catalog.py"
  - path: "python/k100lc_kernels/quant.py"
  - path: "python/k100lc_kernels/rt4.py"
---
