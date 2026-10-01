---
uid: c0545e1c
id: k100lc-kernels.prebuilt.kernel-header
parent: k100lc-kernels.prebuilt
name: {zh: "内核表 C 头", en: "Kernel Table C Header"}
description:
  zh: >
      nodtk_kernels.h：从 HSACO metadata 生成的内核名与参数布局常量表，供 runtime/hsa_rt 在 C++ 侧按偏移填 kernarg。
      
  en: >
      nodtk_kernels.h: kernel names and argument-layout constants generated from HSACO metadata, used by runtime/hsa_rt to fill kernarg by offset.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.853Z"
fingerprint: 4c5868574b718ec6d4d4586118950b74bf4c4e7250f5b2374fcee0f6089dfb24
source:
  - path: "prebuilt/nodtk_kernels.h"
apis:
  - protocol: file
    path: "prebuilt/nodtk_kernels.h"
    description:
      zh: >
          内核名/参数偏移/段大小常量
          
      en: >
          Kernel names, arg offsets and segment sizes
          
---
