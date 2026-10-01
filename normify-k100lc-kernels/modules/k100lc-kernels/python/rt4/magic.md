---
uid: 68e81ffa
id: k100lc-kernels.python.rt4.magic
parent: k100lc-kernels.python.rt4
name: {zh: "除魔数辅助", en: "Division Magic Helper"}
description:
  zh: >
      _div_magic 与 GRP/SG 常量：为内核参数生成免除法的乘法魔数，避免 dim 较大时魔法除不可精确表示。
      
  en: >
      _div_magic with the GRP/SG constants: generates multiply-magic constants for kernels, avoiding the imprecise magic division at large dim.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.887Z"
fingerprint: 9c30915ad83bb227a417d704f682c99a5d6f67f22200a81824d911530e2d9359
source:
  - path: "python/k100lc_kernels/rt4.py"
    line: 36
    end_line: 61
apis:
  - protocol: file
    path: "k100lc_kernels.rt4._div_magic"
    description:
      zh: >
          生成乘法魔数
          
      en: >
          Generates multiply-magic constants
          
---
