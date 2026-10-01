---
uid: a03df3aa
id: k100lc-kernels.generators.gemv.qdot.prologue
parent: k100lc-kernels.generators.gemv.qdot
name: {zh: "点积内核前导", en: "Dot Kernel Prologue"}
description:
  zh: >
      生成每个点积内核的公共前导：魔法除法参数（div_magic/_divmod）、dot_args/dot_kernarg 参数布局、寄存器初始化（dot_head）、收尾归约（dot_tail）与 x 地址推进（dot_xaddr）。
      
  en: >
      Emits the shared prologue of every dot kernel: magic-division parameters (div_magic/_divmod), dot_args/dot_kernarg argument layout, register setup (dot_head), tail reduction (dot_tail) and x-address stepping (dot_xaddr).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.876Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 58
    end_line: 190
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:dot_args"
    description:
      zh: >
          点积内核参数布局
          
      en: >
          Dot-kernel argument layout
          
---
