---
uid: 8d314148
id: k100lc-kernels.python.rt4.container
parent: k100lc-kernels.python.rt4
name: {zh: "RT4 文件与张量", en: "RT4 File & Tensor"}
description:
  zh: >
      RT4Tensor/RT4File：解析 .rt4 容器，按 kind 枚举张量名，取权重字节并提供 dequant 参考。
      
  en: >
      RT4Tensor/RT4File: parse the .rt4 container, enumerate tensor names by kind, read weight bytes and provide a reference dequant.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.887Z"
fingerprint: 9c30915ad83bb227a417d704f682c99a5d6f67f22200a81824d911530e2d9359
source:
  - path: "python/k100lc_kernels/rt4.py"
    line: 62
    end_line: 192
apis:
  - protocol: file
    path: "k100lc_kernels.rt4.RT4File.tensor"
    description:
      zh: >
          按名取 RT4 张量
          
      en: >
          Fetches an RT4 tensor by name
          
---
