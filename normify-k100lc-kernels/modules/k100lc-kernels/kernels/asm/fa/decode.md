---
uid: 0d681f20
id: k100lc-kernels.kernels.asm.fa.decode
parent: k100lc-kernels.kernels.asm.fa
name: {zh: "单行解码注意力", en: "Single-Row Decode Attention"}
description:
  zh: >
      fa_decode_k：解码期单个 query 行对整段 KV 的注意力，读 int8/int4 打包的 KV cache。
      
  en: >
      fa_decode_k: decode-time attention of one query row over the whole KV cache, reading int8/int4 packed KV.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.860Z"
fingerprint: 1c75ce383b217c797e0b98bfc54ea3894f9a7bb224e54295c0710e09faf3397b
source:
  - path: "kernels/asm/k_fa/002__Z11fa_decode_kPfS_S_PKjPKfS1_S3_S1_S3_iiiiiiii.s"
apis:
  - protocol: rpc
    path: "fa_decode_k"
    description:
      zh: >
          单行解码注意力
          
      en: >
          Single-row decode attention
          
---
