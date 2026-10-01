---
uid: deb61051
id: k100lc-kernels.kernels.asm.fa.decode-rows
parent: k100lc-kernels.kernels.asm.fa
name: {zh: "多行解码注意力", en: "Multi-Row Decode Attention"}
description:
  zh: >
      fa_decode_rows_k：一次处理多行 query 的解码注意力（MTP/验证批），行间共享 KV 读取。
      
  en: >
      fa_decode_rows_k: decode attention processing several query rows at once (MTP/verification batch), sharing KV reads across rows.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.861Z"
fingerprint: 3b0f212844e1cd85dc9a0b7709a5aec17b635759d497728cfc47bc04c6487cf5
source:
  - path: "kernels/asm/k_fa/003__Z16fa_decode_rows_kPfS_S_PKjPKfS1_S3_S1_S3_S3_miiiiiiiiiimmi.s"
apis:
  - protocol: rpc
    path: "fa_decode_rows_k"
    description:
      zh: >
          多行解码注意力
          
      en: >
          Multi-row decode attention
          
---
