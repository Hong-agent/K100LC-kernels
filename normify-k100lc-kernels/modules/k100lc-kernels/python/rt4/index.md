---
uid: db4f4d3e
id: k100lc-kernels.python.rt4
parent: k100lc-kernels.python
name: {zh: "RT4 权重读取", en: "RT4 Weight Loading"}
description:
  zh: >
      rt4.py：读取 .rt4 权重文件（RT4Tensor/RT4File）、按 GEMV/GEMM 两种布局上传，并用 W4Runner 提供 W4A8/W4A4 的 GEMV 与 GEMM 直跑入口。
      
  en: >
      rt4.py: reads .rt4 weight files (RT4Tensor/RT4File), uploads them in GEMV or GEMM layout, and exposes W4Runner for direct W4A8/W4A4 GEMV and GEMM runs.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.887Z"
fingerprint: 9c30915ad83bb227a417d704f682c99a5d6f67f22200a81824d911530e2d9359
source:
  - path: "python/k100lc_kernels/rt4.py"
---
