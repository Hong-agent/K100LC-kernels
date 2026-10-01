---
uid: 97ef794d
id: k100lc-kernels.runtime.hsa
parent: k100lc-kernels.runtime
name: {zh: "HSA 垫片", en: "HSA Shim"}
description:
  zh: >
      hsa_rt：直接调 libhsa-runtime64 的薄封装——枚举 agent/memory pool、加载 HSACO、建队列、拷贝与投递 dispatch packet，并对外装作 HIP。
      
  en: >
      hsa_rt: a thin wrapper over libhsa-runtime64 — enumerating agents/memory pools, loading HSACO, creating queues, copying memory and submitting dispatch packets, while presenting a HIP-like surface.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.883Z"
fingerprint: 84d9e3db3d2910553a5cb4fd5a2d226bd0cf0a9bbe9f96cf9fc5f45f2c683e3d
source:
  - path: "runtime/hsa_rt.h"
  - path: "runtime/hsa_rt.cpp"
---
