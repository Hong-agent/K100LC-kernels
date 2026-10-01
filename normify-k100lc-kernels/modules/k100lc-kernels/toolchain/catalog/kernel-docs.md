---
uid: b2fce56f
id: k100lc-kernels.toolchain.catalog.kernel-docs
parent: k100lc-kernels.toolchain.catalog
name: {zh: "内核文档生成", en: "Kernel Docs Generation"}
description:
  zh: >
      gen_kernel_docs.py：把 catalog.json 渲染成分类统计 + 逐内核速查表（显式参数、kernarg、LDS、private）的 docs/KERNELS.md。
      
  en: >
      gen_kernel_docs.py: renders catalog.json into docs/KERNELS.md — category counts plus a per-kernel table of explicit args, kernarg, LDS and private sizes.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.892Z"
fingerprint: a909d2193a0a3826eb9d64ba00377987276b753af91af374b1fee6fedfceb086
source:
  - path: "tools/gen_kernel_docs.py"
apis:
  - protocol: file
    path: "tools/gen_kernel_docs.py:main"
    description:
      zh: >
          catalog.json → docs/KERNELS.md
          
      en: >
          catalog.json to docs/KERNELS.md
          
deps:
  - kind: dataflow
    to: k100lc-kernels.docs.kernels-table
    from_api: "file:tools/gen_kernel_docs.py:main"
    to_api: "file:docs/KERNELS.md"
    label: {zh: "生成内核参数总表", en: "Generates the kernel parameter"}
  - kind: dataflow
    to: k100lc-kernels.python.catalog.data
    from_api: "file:tools/gen_kernel_docs.py:main"
    to_api: "file:python/k100lc_kernels/catalog.json"
    label: {zh: "读取内核目录", en: "Reads the catalog"}
---
