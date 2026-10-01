---
uid: 113bf252
id: k100lc-kernels.kernels.headers.kv-pack
parent: k100lc-kernels.kernels.headers
name: {zh: "KV 打包参数", en: "KV Packing Parameters"}
description:
  zh: >
      kv_pack.h：KV cache 的位宽选择（int8/int4）、KVEL 每 dword 元素数、KVQMAX 量化满量程与掩码，供 KV 写入与注意力内核共用。
      
  en: >
      kv_pack.h: KV-cache bit-width selection (int8/int4), KVEL elements per dword, KVQMAX quantization range and masks, shared by KV-append and attention kernels.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.859Z"
fingerprint: 3442c5c0f15b04158694517d3148d213219e4ea1e9aba8e6b3f7004a56db800e
source:
  - path: "kernels/kv_pack.h"
    line: 1
    end_line: 25
apis:
  - protocol: file
    path: "kernels/kv_pack.h"
    description:
      zh: >
          KV 量化打包常量
          
      en: >
          KV quantization/packing constants
          
---
