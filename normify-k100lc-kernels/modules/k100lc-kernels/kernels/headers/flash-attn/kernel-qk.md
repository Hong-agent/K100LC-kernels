---
uid: dc3de3f7
id: k100lc-kernels.kernels.headers.flash-attn.kernel-qk
parent: k100lc-kernels.kernels.headers.flash-attn
name: {zh: "fa_int4 QK 与在线 softmax", en: "fa_int4 QK & Online Softmax"}
description:
  zh: >
      主 tile 循环前半段：QK 收缩点积（按量化组还原尺度）、因果掩码、行最大、P 的量化打包与行和，随后按 exp2 递推在线 softmax 的 RowRef/RowAlpha/RowPB/RowL。
      
  en: >
      The first half of the main tile loop: the QK contraction (rescaling per quantization group), causal masking, row max, P quantization packing and row sums, then the exp2 online-softmax update of RowRef/RowAlpha/RowPB/RowL.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.859Z"
fingerprint: 159986b60b2c2b48f8ed9e95fdda58fef428936348812ffb3ac6896f9070924c
source:
  - path: "kernels/flash_attn_core.h"
    line: 222
    end_line: 383
apis:
  - protocol: file
    path: "kernels/flash_attn_core.h:qk_softmax"
    description:
      zh: >
          QK 与在线 softmax 主体
          
      en: >
          QK and online-softmax body
          
---
