---
uid: 3b2118de
id: k100lc-kernels.docs.kernels-table
parent: k100lc-kernels.docs
name: {zh: "内核参数总表", en: "Kernel Parameter Table"}
description:
  zh: >
      KERNELS.md：由 catalog 自动生成的 122 内核分类统计与逐内核速查表（显式参数、kernarg、LDS、private）。
      
  en: >
      KERNELS.md: auto-generated category counts and a per-kernel quick table for all 122 kernels (explicit args, kernarg, LDS, private).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.896Z"
fingerprint: 29ad0490bf0c23044f0ab900fbf39d3888337974a769e09c322132734ef853bf
source:
  - path: "docs/KERNELS.md"
apis:
  - protocol: file
    path: "docs/KERNELS.md"
    description:
      zh: >
          逐内核参数总表
          
      en: >
          Full per-kernel parameter table
          
deps:
  - kind: dataflow
    to: k100lc-kernels.python.catalog.data
    from_api: "file:docs/KERNELS.md"
    to_api: "file:python/k100lc_kernels/catalog.json"
    label: {zh: "由 catalog 生成", en: "Generated from the catalog"}
---
