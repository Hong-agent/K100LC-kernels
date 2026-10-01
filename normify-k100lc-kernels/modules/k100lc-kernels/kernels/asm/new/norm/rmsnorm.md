---
uid: 54c8108a
id: k100lc-kernels.kernels.asm.new.norm.rmsnorm
parent: k100lc-kernels.kernels.asm.new.norm
name: {zh: "RMSNorm", en: "RMSNorm"}
description:
  zh: >
      rmsnorm_k(y, x, w, cols, eps, flag)：warp-per-row + LDS 归约；flag=0 用 w，非 0 用 1+w。
      
  en: >
      rmsnorm_k(y, x, w, cols, eps, flag): warp-per-row with LDS reduction; flag=0 uses w, otherwise 1+w.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.868Z"
fingerprint: 37f4886a3a1cd0f613900700ba4ff99a1143a06c1973cc1f1fc37daffc0bbc8e
source:
  - path: "kernels/asm/k_new/006__Z9rmsnorm_kPfPKfS1_ifi.s"
apis:
  - protocol: rpc
    path: "rmsnorm_k"
    description:
      zh: >
          行 RMSNorm
          
      en: >
          Row-wise RMSNorm
          
---
