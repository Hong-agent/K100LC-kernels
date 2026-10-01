---
uid: 9e5c8803
id: k100lc-kernels.kernels.asm.fa.decode-comb
parent: k100lc-kernels.kernels.asm.fa
name: {zh: "单行解码合并", en: "Single-Row Decode Combine"}
description:
  zh: >
      fa_decode_comb_k：把分块算出的注意力 partial 合并成一行输出（配合 fa_decode_k 的分块归约）。
      
  en: >
      fa_decode_comb_k: combines chunked attention partials into one output row (the reduction stage paired with fa_decode_k).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.861Z"
fingerprint: d6f2721b72bcc9eb51ebfc8c9f3186c9a64fdf8ac693fc8debb0cb054b81fa0a
source:
  - path: "kernels/asm/k_fa/004__Z16fa_decode_comb_kPfPKfS1_S1_ii.s"
apis:
  - protocol: rpc
    path: "fa_decode_comb_k"
    description:
      zh: >
          单行注意力 partial 合并
          
      en: >
          Combines single-row attention partials
          
---
