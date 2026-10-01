---
uid: e2de949a
id: k100lc-kernels.hsaco
parent: k100lc-kernels
name: {zh: "HSACO 代码对象构建", en: "HSACO Code-Object Construction"}
description:
  zh: >
      不依赖 DTK 直接拼出 gfx926 的 ELF 代码对象：ELF 布局、metadata/symbol/hash/descriptor 段、多内核合并，以及从既有 code object 回读 AMDGPU msgpack 元数据。
      
  en: >
      Builds gfx926 ELF code objects without DTK: ELF layout, metadata/symbol/hash/descriptor sections, multi-kernel merging, and reading AMDGPU msgpack metadata back from an existing code object.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.857Z"
fingerprint: a4d59c00aa9bc4dfe2c3dadbeaef594a7e9f5cd9de63b1e12b13fe0c54e3c50e
source:
  - path: "tools/make_hsaco_full.py"
  - path: "tools/make_hsaco_multi.py"
  - path: "tools/make_hsaco_from_scratch.py"
  - path: "tools/merge_hsacos.py"
  - path: "tools/amdgpu_meta.py"
  - path: "tools/hsaco_names.py"
  - path: "tools/gen_kernel_table.py"
---
