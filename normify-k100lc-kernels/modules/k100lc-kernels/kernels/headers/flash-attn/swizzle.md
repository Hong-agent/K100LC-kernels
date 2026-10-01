---
uid: 1c11e5c1
id: k100lc-kernels.kernels.headers.flash-attn.swizzle
parent: k100lc-kernels.kernels.headers.flash-attn
name: {zh: "FA LDS 交错与点积", en: "FA LDS Swizzle & Dot Helpers"}
description:
  zh: >
      xcol 列交错函数（把 8 列拆成两个 4 列半段，使 LDS.128 铺满 32 个 bank）与随 KV_BITS 切换的 dot4/dot8 收缩维点积。
      
  en: >
      The xcol column-swizzle (splitting 8 columns into two 4-column halves so LDS.128 fills all 32 banks) and the dot4/dot8 contraction dot products selected by KV_BITS.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.859Z"
fingerprint: 159986b60b2c2b48f8ed9e95fdda58fef428936348812ffb3ac6896f9070924c
source:
  - path: "kernels/flash_attn_core.h"
    line: 63
    end_line: 78
apis:
  - protocol: file
    path: "kernels/flash_attn_core.h:xcol"
    description:
      zh: >
          LDS 列号交错
          
      en: >
          LDS column swizzle
          
---
