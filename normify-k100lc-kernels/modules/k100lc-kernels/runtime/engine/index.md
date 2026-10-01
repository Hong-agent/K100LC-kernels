---
uid: aac94b27
id: k100lc-kernels.runtime.engine
parent: k100lc-kernels.runtime
name: {zh: "常驻引擎 C ABI", en: "Resident Engine C ABI"}
description:
  zh: >
      fm_engine：把 hsa_rt 包装成稳定的 C ABI（fm_init/alloc/upload/launch/sync），按内核表的显式参数个数校验并填 kernarg，是 libfm_engine.so 的实现。
      
  en: >
      fm_engine: wraps hsa_rt in a stable C ABI (fm_init/alloc/upload/launch/sync), validating against the kernel table’s explicit argument count and filling kernarg; it is the implementation of libfm_engine.so.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.884Z"
fingerprint: 337960b34f7aac9f9f95b9583bc9544b7e885dd1f63641f599bf7eef24b59abe
source:
  - path: "runtime/fm_engine.h"
  - path: "runtime/fm_engine.cpp"
---
