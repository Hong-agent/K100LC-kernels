---
uid: 14c7b2bd
id: k100lc-kernels.hsaco.metadata-read
parent: k100lc-kernels.hsaco
name: {zh: "AMDGPU 元数据读取", en: "AMDGPU Metadata Reading"}
description:
  zh: >
      只读解析 gfx926 code object 里的 AMDGPU msgpack 元数据，取出每个内核的参数表与 kernarg/段大小，构建期用它生成内核表。
      
  en: >
      Read-only parser for the AMDGPU msgpack metadata inside a gfx926 code object, extracting per-kernel argument tables and kernarg/segment sizes for the build.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.857Z"
fingerprint: 436535603d734677324728af1111f959a1cb05b9a9f46cc9a1582989d9258a60
source:
  - path: "tools/amdgpu_meta.py"
apis:
  - protocol: file
    path: "tools/amdgpu_meta.py"
    description:
      zh: >
          HSACO → 内核元数据
          
      en: >
          HSACO to kernel metadata
          
deps:
  - kind: reference
    to: k100lc-kernels.hsaco.symbol-names
    from_api: "file:tools/amdgpu_meta.py"
    to_api: "file:tools/hsaco_names.py"
    label: {zh: "隐藏参数 value_kind 映射", en: "Hidden-arg value_kind mapping"}
---
