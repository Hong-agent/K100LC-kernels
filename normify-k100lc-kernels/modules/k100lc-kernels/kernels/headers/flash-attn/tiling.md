---
uid: 71a2c1a5
id: k100lc-kernels.kernels.headers.flash-attn.tiling
parent: k100lc-kernels.kernels.headers.flash-attn
name: {zh: "FA 打包与分块参数", en: "FA Packing & Tiling Parameters"}
description:
  zh: >
      Q/K/V/P 的 u32 打包布局与 KV_BITS 选择，以及 HD/QG/BM/BN/NT/TM/TN/TMO/TNO 等分块、线程映射和 LDS padding 常量。
      
  en: >
      The u32 packing layout of Q/K/V/P with the KV_BITS choice, plus the HD/QG/BM/BN/NT/TM/TN/TMO/TNO tiling, thread-mapping and LDS padding constants.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.859Z"
fingerprint: 159986b60b2c2b48f8ed9e95fdda58fef428936348812ffb3ac6896f9070924c
source:
  - path: "kernels/flash_attn_core.h"
    line: 1
    end_line: 62
apis:
  - protocol: file
    path: "kernels/flash_attn_core.h:tiling"
    description:
      zh: >
          分块与打包参数宏
          
      en: >
          Tiling and packing parameter macros
          
---
