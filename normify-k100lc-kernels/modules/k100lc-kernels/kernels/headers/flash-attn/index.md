---
uid: 3dd57d05
id: k100lc-kernels.kernels.headers.flash-attn
parent: k100lc-kernels.kernels.headers
name: {zh: "FlashAttention 块参数", en: "FlashAttention Block Parameters"}
description:
  zh: >
      flash_attn_core.h：int4 打包格式说明、分块与线程映射参数、LDS 列交错技巧、dot4/dot8 设备函数，以及 int4 QK/PV 的 fa_int4 参考实现。
      
  en: >
      flash_attn_core.h: the int4 packing formats, tiling and thread-mapping parameters, the LDS column-swizzle trick, the dot4/dot8 device helpers, and the fa_int4 reference implementation for int4 QK/PV.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.858Z"
fingerprint: 159986b60b2c2b48f8ed9e95fdda58fef428936348812ffb3ac6896f9070924c
source:
  - path: "kernels/flash_attn_core.h"
    line: 1
    end_line: 96
---
