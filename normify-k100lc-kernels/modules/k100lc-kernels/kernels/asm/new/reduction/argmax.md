---
uid: 0958f47e
id: k100lc-kernels.kernels.asm.new.reduction.argmax
parent: k100lc-kernels.kernels.asm.new.reduction
name: {zh: "单行 argmax", en: "Single-Row Argmax"}
description:
  zh: >
      argmax_k(x, n, idx)：grid-stride 求一行最大值的下标。
      
  en: >
      argmax_k(x, n, idx): grid-stride computation of the argmax index of one row.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.869Z"
fingerprint: d1deb88e8f8baed49254454bcf993c51cb6f204cd00aa1661fd9f0adb8bd2da9
source:
  - path: "kernels/asm/k_new/008__Z8argmax_kPKfxPi.s"
apis:
  - protocol: rpc
    path: "argmax_k"
    description:
      zh: >
          一行 argmax
          
      en: >
          Argmax of one row
          
---
