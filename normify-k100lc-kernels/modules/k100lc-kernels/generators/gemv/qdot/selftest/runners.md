---
uid: "75479152"
id: k100lc-kernels.generators.gemv.qdot.selftest.runners
parent: k100lc-kernels.generators.gemv.qdot.selftest
name: {zh: "逐编码自检与入口", en: "Per-Codec Self-Tests & Entry"}
description:
  zh: >
      逐编码 selftest_* 与 _selftest_grid/_selftest_notable 自检框架，main() 按参数跑全部 11 类编码并报差异。
      
  en: >
      The per-codec selftest_* functions with the _selftest_grid/_selftest_notable harness; main() runs all 11 codecs and reports differences.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.879Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 1177
    end_line: 1428
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:main"
    description:
      zh: >
          全部编码自检入口
          
      en: >
          Self-test entry point for all codecs
          
---
