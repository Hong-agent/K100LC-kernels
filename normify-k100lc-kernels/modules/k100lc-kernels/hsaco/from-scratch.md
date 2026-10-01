---
uid: 7c75a87d
id: k100lc-kernels.hsaco.from-scratch
parent: k100lc-kernels.hsaco
name: {zh: "最小 HSACO 构造", en: "Minimal HSACO from Scratch"}
description:
  zh: >
      从零重建 ELF 布局并调整 .text 段大小，复用已知可用的 nop_k 元数据/符号/Hash 段，支持任意长度的无参内核。
      
  en: >
      Rebuilds the ELF layout from scratch with a resized .text section, reusing known-good metadata/symbol/hash sections of nop_k; supports arbitrarily sized no-argument kernels.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.857Z"
fingerprint: 53a241a564b8c2b00324bd8007f4de665b33394830625d0e50536ea18836facf
source:
  - path: "tools/make_hsaco_from_scratch.py"
apis:
  - protocol: file
    path: "tools/make_hsaco_from_scratch.py"
    description:
      zh: >
          生成最小 gfx926 HSACO
          
      en: >
          Generates a minimal gfx926 HSACO
          
---
