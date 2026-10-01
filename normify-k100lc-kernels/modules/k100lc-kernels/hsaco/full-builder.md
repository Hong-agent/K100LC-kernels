---
uid: 636f6ea3
id: k100lc-kernels.hsaco.full-builder
parent: k100lc-kernels.hsaco
name: {zh: "完整 HSACO 生成", en: "Full HSACO Builder"}
description:
  zh: >
      完全生成一个最小 gfx926 code object：metadata（参数 kind/offset/size）、symbol 表、hash、descriptor 与 kernel descriptor 全部按规范拼装。
      
  en: >
      Fully generates a minimal gfx926 code object: metadata (arg kind/offset/size), symbol table, hashes, descriptor and kernel descriptor assembled per spec.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.857Z"
fingerprint: 6f5d8bd7d708b0b3a6db270de2ec9163b57a120c8bd46ef5cfaf2a61a7ba3f6d
source:
  - path: "tools/make_hsaco_full.py"
apis:
  - protocol: file
    path: "tools/make_hsaco_full.py"
    description:
      zh: >
          生成完整 HSACO（含 metadata）
          
      en: >
          Generates a full HSACO with metadata
          
---
