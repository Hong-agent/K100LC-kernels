---
uid: 255395ad
id: k100lc-kernels.generators.gemv.qdot.registry
parent: k100lc-kernels.generators.gemv.qdot
name: {zh: "点积内核清单", en: "Dot Kernel Registry"}
description:
  zh: >
      KERNELS 注册表：把每个 *_dot_k 的名称、生成函数、参数布局与 kernarg 大小登记成构建期内核清单（供 kernel_spec.json / HSACO 组装使用）。
      
  en: >
      The KERNELS registry: records each *_dot_k name, generator function, argument layout and kernarg size as a build-time kernel manifest (consumed by kernel_spec.json / HSACO assembly).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.879Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 1429
    end_line: 1446
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:KERNELS"
    description:
      zh: >
          点积内核注册表
          
      en: >
          Dot-kernel registry
          
---
