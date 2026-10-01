---
uid: 14d26476
id: k100lc-kernels.python.quant
parent: k100lc-kernels.python
name: {zh: "量化打包工具", en: "Quantization Packing Utilities"}
description:
  zh: >
      quant.py：主机侧 bf16 位操作、INT4 group-128 打包/反量化，以及 compressed-tensors 的 offset-binary → 两补码/RT4 布局转换与尺度重排。
      
  en: >
      quant.py: host-side bf16 bit twiddling, INT4 group-128 pack/unpack, and the compressed-tensors offset-binary to two’s-complement/RT4 layout conversion with scale regrouping.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.887Z"
fingerprint: ba5c4b7622b8f6ca986027d789484d3c31a1e5ccab27f23977be9020da6357ea
source:
  - path: "python/k100lc_kernels/quant.py"
---
