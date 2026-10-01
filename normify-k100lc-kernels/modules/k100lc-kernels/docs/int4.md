---
uid: "20225809"
id: k100lc-kernels.docs.int4
parent: k100lc-kernels.docs
name: {zh: "INT4 格式文档", en: "INT4 Format Docs"}
description:
  zh: >
      INT4.md：compressed-tensors（W4A16）存储格式、对应内核、用法与对账方法、性能与限制。
      
  en: >
      INT4.md: the compressed-tensors (W4A16) storage format, the matching kernels, usage and reconciliation method, performance and limits.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.896Z"
fingerprint: b004892b68100ae52320d7960896f7455f31e456d17bf3a843b5d567ccc5f13b
source:
  - path: "docs/INT4.md"
apis:
  - protocol: file
    path: "docs/INT4.md"
    description:
      zh: >
          INT4 格式与用法
          
      en: >
          INT4 format and usage
          
deps:
  - kind: reference
    to: k100lc-kernels.python.quant.ct-bridge
    from_api: "file:docs/INT4.md"
    to_api: "file:k100lc_kernels.quant.ct_int4_to_rt4"
    label: {zh: "描述 INT4 格式与转换", en: "Documents the INT4 format and"}
---
