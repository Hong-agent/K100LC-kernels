---
uid: 7e25b5a3
id: k100lc-kernels.kernels.headers.gemv-w4a8
parent: k100lc-kernels.kernels.headers
name: {zh: "W4A8 GEMV 块参数", en: "W4A8 GEMV Block Parameters"}
description:
  zh: >
      gemv_w4a8_core.h：量化组 GRP、线程数、dot4_i8 设备端点积，以及 W4A8 路径的行/列划分常量。
      
  en: >
      gemv_w4a8_core.h: quantization group GRP, thread count and the dot4_i8 device dot product, plus the row/column split constants of the W4A8 path.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.859Z"
fingerprint: 657a737526926b550b4d3d2aefd3eef098c4c6e3c6c92804e59fd8ca1afce324
source:
  - path: "kernels/gemv_w4a8_core.h"
    line: 1
    end_line: 219
apis:
  - protocol: file
    path: "kernels/gemv_w4a8_core.h:dot4_i8"
    description:
      zh: >
          int8 点积辅助函数
          
      en: >
          int8 dot-product helper
          
deps:
  - kind: reference
    to: k100lc-kernels.kernels.asm.gemv.w4a8.m1
    from_api: "file:kernels/gemv_w4a8_core.h:dot4_i8"
    to_api: "rpc:gemv_w4a8<1,false,1>"
    label: {zh: "块参数与点积辅助", en: "Block parameters and dot"}
---
