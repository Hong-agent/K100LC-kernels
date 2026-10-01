---
uid: 09df3938
id: k100lc-kernels.toolchain.catalog.make
parent: k100lc-kernels.toolchain.catalog
name: {zh: "Catalog 生成", en: "Catalog Generation"}
description:
  zh: >
      make_catalog.py：读 HSACO metadata，输出 lookup 名、mangled 名、参数 (off,size,kind) 与段大小到 catalog.json。
      
  en: >
      make_catalog.py: reads HSACO metadata and writes lookup names, mangled names, argument (off,size,kind) and segment sizes into catalog.json.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.892Z"
fingerprint: 8a93d6410098be8d23f8b92adc811ec7c308ade62e9296be3b8bd31b74a656ed
source:
  - path: "tools/make_catalog.py"
apis:
  - protocol: file
    path: "tools/make_catalog.py:main"
    description:
      zh: >
          HSACO → catalog.json
          
      en: >
          HSACO to catalog.json
          
deps:
  - kind: dataflow
    to: k100lc-kernels.python.catalog.data
    from_api: "file:tools/make_catalog.py:main"
    to_api: "file:python/k100lc_kernels/catalog.json"
    label: {zh: "生成运行时内核目录", en: "Generates the runtime catalog"}
---
