---
uid: a6eb0d52
id: k100lc-kernels.kernels.asm.fa.decode-rows-comb
parent: k100lc-kernels.kernels.asm.fa
name: {zh: "多行解码合并", en: "Multi-Row Decode Combine"}
description:
  zh: >
      fa_decode_rows_comb_k：多行解码注意力的 partial 合并阶段。
      
  en: >
      fa_decode_rows_comb_k: the partial-combine stage of multi-row decode attention.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.861Z"
fingerprint: 749677e397fb534abc6a9c50dbb74369a2e6f6c44de74e8a9e1122def1309946
source:
  - path: "kernels/asm/k_fa/005__Z21fa_decode_rows_comb_kPfPKfS1_S1_iiiiiim.s"
apis:
  - protocol: rpc
    path: "fa_decode_rows_comb_k"
    description:
      zh: >
          多行注意力 partial 合并
          
      en: >
          Combines multi-row attention partials
          
---
