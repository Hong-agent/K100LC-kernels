---
uid: "52048629"
id: k100lc-kernels.kernels.asm.new.position.embed
parent: k100lc-kernels.kernels.asm.new.position
name: {zh: "Token 嵌入", en: "Token Embedding"}
description:
  zh: >
      embed_k：按 token id 从量化嵌入表中取出并解码成 f32 行。
      
  en: >
      embed_k: gathers and decodes f32 rows from a quantized embedding table by token id.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.870Z"
fingerprint: fea753fe4fc8ab4f1b84049be934856ab5928f131dd2d26549aa623b35f85c48
source:
  - path: "kernels/asm/k_new/013__Z7embed_kPfPKjPKtPKiii.s"
apis:
  - protocol: rpc
    path: "embed_k"
    description:
      zh: >
          量化嵌入表查表
          
      en: >
          Lookup in a quantized embedding table
          
---
