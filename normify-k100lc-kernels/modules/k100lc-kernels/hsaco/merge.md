---
uid: 867c1420
id: k100lc-kernels.hsaco.merge
parent: k100lc-kernels.hsaco
name: {zh: "HSACO 合并", en: "HSACO Merging"}
description:
  zh: >
      把若干个本工具链生成的 HSACO 合并成一个代码对象：合并 .text、符号与 metadata，重定位各自的 kernel 描述符。
      
  en: >
      Merges several toolchain-generated HSACOs into one code object: concatenating .text, symbols and metadata while fixing up each kernel descriptor.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.857Z"
fingerprint: 39e0506344db7609f84febaa36e660936e0dbc2dcda1f6e5dcd75504d8b2f962
source:
  - path: "tools/merge_hsacos.py"
apis:
  - protocol: file
    path: "tools/merge_hsacos.py"
    description:
      zh: >
          合并多个 HSACO
          
      en: >
          Merges multiple HSACOs
          
deps:
  - kind: dataflow
    to: k100lc-kernels.hsaco.multi-builder
    from_api: "file:tools/merge_hsacos.py"
    to_api: "file:tools/make_hsaco_multi.py"
    label: {zh: "合并多份 HSACO", en: "Merges code objects"}
  - kind: dataflow
    to: k100lc-kernels.prebuilt.kernels-hsaco
    from_api: "file:tools/merge_hsacos.py"
    to_api: "file:prebuilt/k100lc_kernels.hsaco"
    label: {zh: "合并出完整内核包", en: "Merges the full kernel package"}
---
