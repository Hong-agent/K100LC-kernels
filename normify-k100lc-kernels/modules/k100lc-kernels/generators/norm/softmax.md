---
uid: b5e69131
id: k100lc-kernels.generators.norm.softmax
parent: k100lc-kernels.generators.norm
name: {zh: "Softmax 生成器", en: "Softmax Generator"}
description:
  zh: >
      gen_softmax.py：warp-per-row + LDS 树形归约的行 softmax（softmax_k）。
      
  en: >
      gen_softmax.py: row softmax with warp-per-row LDS tree reduction (softmax_k).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.882Z"
fingerprint: 2da801f964fee13741c26a04728112f424494952674891b77ac0291c4c983909
source:
  - path: "tools/gen_softmax.py"
apis:
  - protocol: file
    path: "tools/gen_softmax.py"
    description:
      zh: >
          生成 softmax_k
          
      en: >
          Generates softmax_k
          
---
