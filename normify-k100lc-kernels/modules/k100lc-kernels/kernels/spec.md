---
uid: a1f7809c
id: k100lc-kernels.kernels.spec
parent: k100lc-kernels.kernels
name: {zh: "内核元数据表", en: "Kernel Metadata Table"}
description:
  zh: >
      kernel_spec.json：85 个汇编内核的 name、参数表、kernarg 大小/对齐、SGPR/VGPR 计数、LDS/private 段与源文件路径，是 HSACO 组装与运行时参数填充的唯一依据。
      
  en: >
      kernel_spec.json: name, argument table, kernarg size/align, SGPR/VGPR counts, LDS/private segments and source paths for the 85 assembly kernels; the single source of truth for HSACO assembly and runtime argument filling.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.858Z"
fingerprint: 5da8e2c4a18ac1f88dc1ec1661838ba5c5fd84b60632d31f202b831dfd322674
source:
  - path: "kernels/kernel_spec.json"
apis:
  - protocol: file
    path: "kernels/kernel_spec.json"
    description:
      zh: >
          内核参数与寄存器/段元数据
          
      en: >
          Kernel argument and register/segment metadata
          
deps:
  - kind: dataflow
    to: k100lc-kernels.kernels.asm
    from_api: "file:kernels/kernel_spec.json"
    label: {zh: "登记内核元数据", en: "Registers kernel metadata"}
---
