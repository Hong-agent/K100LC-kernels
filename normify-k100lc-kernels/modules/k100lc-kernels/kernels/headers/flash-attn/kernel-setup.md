---
uid: "582713e4"
id: k100lc-kernels.kernels.headers.flash-attn.kernel-setup
parent: k100lc-kernels.kernels.headers.flash-attn
name: {zh: "fa_int4 内核入口与装载", en: "fa_int4 Setup & Staging"}
description:
  zh: >
      fa_int4 内核签名、LDS 布局（QsT/KsT/Vs/PsT 与行状态数组）、GQA 头映射与 QKV 行距约定，以及首个 tile 的 K/V 装载。
      
  en: >
      The fa_int4 signature, LDS layout (QsT/KsT/Vs/PsT plus row-state arrays), GQA head mapping and KV row-stride convention, and the first-tile K/V staging.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.859Z"
fingerprint: 159986b60b2c2b48f8ed9e95fdda58fef428936348812ffb3ac6896f9070924c
source:
  - path: "kernels/flash_attn_core.h"
    line: 97
    end_line: 221
apis:
  - protocol: file
    path: "kernels/flash_attn_core.h:fa_int4"
    description:
      zh: >
          fa_int4 内核入口（签名与 LDS 布局）
          
      en: >
          fa_int4 entry (signature and LDS layout)
          
deps:
  - kind: reference
    to: k100lc-kernels.kernels.asm.fa.int4
    from_api: "file:kernels/flash_attn_core.h:fa_int4"
    to_api: "rpc:fa_int4"
    label: {zh: "参考实现与 LDS 布局", en: "Reference implementation and"}
---
