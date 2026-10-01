---
uid: c836cd26
id: k100lc-kernels.python.catalog.accessor
parent: k100lc-kernels.python.catalog
name: {zh: "目录访问器", en: "Catalog Accessors"}
description:
  zh: >
      load/kernels/info：载入 catalog.json 并按名字或子串过滤内核，供 Runtime 与文档生成共用。
      
  en: >
      load/kernels/info: loads catalog.json and filters kernels by name or substring, shared by Runtime and the doc generator.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.885Z"
fingerprint: cb078eb0148e8906acd7e9efa534e82e8477c4f9c88ace9d331f1758ba64dc32
source:
  - path: "python/k100lc_kernels/catalog.py"
    line: 1
    end_line: 25
apis:
  - protocol: file
    path: "k100lc_kernels.catalog.info"
    description:
      zh: >
          按名字查内核参数表
          
      en: >
          Looks up a kernel argument table by name
          
deps:
  - kind: dataflow
    to: k100lc-kernels.python.catalog.data
    from_api: "file:k100lc_kernels.catalog.info"
    to_api: "file:python/k100lc_kernels/catalog.json"
    label: {zh: "读取 catalog.json", en: "Reads catalog.json"}
---
