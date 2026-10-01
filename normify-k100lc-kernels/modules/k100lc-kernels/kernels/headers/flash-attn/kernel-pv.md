---
uid: f6871b0b
id: k100lc-kernels.kernels.headers.flash-attn.kernel-pv
parent: k100lc-kernels.kernels.headers.flash-attn
name: {zh: "fa_int4 PV 与写回", en: "fa_int4 PV & Output"}
description:
  zh: >
      主 tile 循环后半段与收尾：O = O·alpha + (P·V)·(beta/KVQMAX)·sv 的 PV 累加、V 的转置打包与最终按行/维度写回 Out。
      
  en: >
      The second half of the main tile loop and the epilogue: the PV accumulation O = O·alpha + (P·V)·(beta/KVQMAX)·sv, V transposed packing, and the final write-back of Out.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.859Z"
fingerprint: 159986b60b2c2b48f8ed9e95fdda58fef428936348812ffb3ac6896f9070924c
source:
  - path: "kernels/flash_attn_core.h"
    line: 384
    end_line: 463
apis:
  - protocol: file
    path: "kernels/flash_attn_core.h:pv"
    description:
      zh: >
          PV 累加与输出写回
          
      en: >
          PV accumulation and output write
          
---
