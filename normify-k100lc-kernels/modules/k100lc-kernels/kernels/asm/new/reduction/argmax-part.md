---
uid: 3445f629
id: k100lc-kernels.kernels.asm.new.reduction.argmax-part
parent: k100lc-kernels.kernels.asm.new.reduction
name: {zh: "分块 partial argmax", en: "Chunked Partial Argmax"}
description:
  zh: >
      argmax_part_k(x, n, nb, val, idx)：每个 workgroup 算一段的部分最大值与下标。
      
  en: >
      argmax_part_k(x, n, nb, val, idx): each workgroup produces the partial max value and index of one chunk.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.869Z"
fingerprint: b209eccd2925ad02a126cae6a02300bf99523a723ea3f18a665834b0c956ddba
source:
  - path: "kernels/asm/k_new/009__Z13argmax_part_kPKfxiPfPi.s"
apis:
  - protocol: rpc
    path: "argmax_part_k"
    description:
      zh: >
          分块部分 argmax
          
      en: >
          Chunked partial argmax
          
---
