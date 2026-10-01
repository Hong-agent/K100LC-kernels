---
uid: fdbf74e7
id: k100lc-kernels.runtime.engine.abi
parent: k100lc-kernels.runtime.engine
name: {zh: "引擎头文件", en: "Engine Header"}
description:
  zh: >
      fm_engine.h：C ABI 声明（fm_init/fm_alloc/fm_free/fm_upload/fm_download/fm_copy/fm_memset/fm_sync/fm_launch/fm_launch2d/fm_launch_dyn）。
      
  en: >
      fm_engine.h: the C ABI declarations (fm_init/fm_alloc/fm_free/fm_upload/fm_download/fm_copy/fm_memset/fm_sync/fm_launch/fm_launch2d/fm_launch_dyn).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.884Z"
fingerprint: 6594662583df8b2ddcff21bb8f6fb546fe152886d6eb6858952a44cba21d7a8e
source:
  - path: "runtime/fm_engine.h"
    line: 1
    end_line: 32
apis:
  - protocol: file
    path: "runtime/fm_engine.h"
    description:
      zh: >
          引擎 C ABI 声明
          
      en: >
          Engine C ABI declarations
          
deps:
  - kind: reference
    to: k100lc-kernels.runtime.hsa.types
    from_api: "file:runtime/fm_engine.h"
    to_api: "file:runtime/hsa_rt.h:RtKernel"
    label: {zh: "复用 RtKernel 参数表", en: "Reuses the RtKernel argument"}
---
