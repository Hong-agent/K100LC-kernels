---
uid: 2adbbfb3
id: k100lc-kernels.kernels.headers.gemv-w4a4
parent: k100lc-kernels.kernels.headers
name: {zh: "W4A4 GEMV 块参数", en: "W4A4 GEMV Block Parameters"}
description:
  zh: >
      gemv_w4a4_core.h：GRP/SG/NTHREADS/ROWS 与 dot8_i4 设备端点积，定义每 warp 处理的行数与激活子分组。
      
  en: >
      gemv_w4a4_core.h: GRP/SG/NTHREADS/ROWS plus the dot8_i4 device dot product, defining rows-per-warp and the activation sub-group.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.859Z"
fingerprint: 185ed5e60d11d1983ed0441177f7ba15aa26c359f52aee9519b54b9d403a55c2
source:
  - path: "kernels/gemv_w4a4_core.h"
    line: 1
    end_line: 134
apis:
  - protocol: file
    path: "kernels/gemv_w4a4_core.h:dot8_i4"
    description:
      zh: >
          int4 点积辅助函数
          
      en: >
          int4 dot-product helper
          
deps:
  - kind: reference
    to: k100lc-kernels.kernels.asm.gemv.w4a4.m1
    from_api: "file:kernels/gemv_w4a4_core.h:dot8_i4"
    to_api: "rpc:gemv_w4a4<1>"
    label: {zh: "块参数与点积辅助", en: "Block parameters and dot"}
---
