---
uid: bb28346e
id: k100lc-kernels.kernels.asm.new.reduction.argmax-comb
parent: k100lc-kernels.kernels.asm.new.reduction
name: {zh: "argmax 合并", en: "Argmax Combine"}
description:
  zh: >
      argmax_comb_k(val, idx, nb, out)：把 nb 段 partial 合并成全局 argmax（贪心采样的收尾）。
      
  en: >
      argmax_comb_k(val, idx, nb, out): combines nb partials into the global argmax (the tail of greedy sampling).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.869Z"
fingerprint: 0c0cbeb668c506ce46c8c603490e8a2044995321e17e099a35e4544215033f70
source:
  - path: "kernels/asm/k_new/010__Z13argmax_comb_kPKfPKiiPi.s"
apis:
  - protocol: rpc
    path: "argmax_comb_k"
    description:
      zh: >
          合并 partial argmax
          
      en: >
          Combines partial argmax results
          
---
