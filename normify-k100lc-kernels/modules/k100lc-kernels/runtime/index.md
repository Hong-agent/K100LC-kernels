---
uid: 79cd889f
id: k100lc-kernels.runtime
parent: k100lc-kernels
name: {zh: "无 DTK HSA 运行时", en: "No-DTK HSA Runtime"}
description:
  zh: >
      只依赖 /opt/hyhal 的常驻运行时：hsa_rt 提供 HSA 垫片与 HIP 兼容 API，fm_engine 在它之上暴露 C ABI，供 Python/C++ 直接驱动内核。
      
  en: >
      A resident runtime that only depends on /opt/hyhal: hsa_rt provides the HSA shim and a HIP-compatible API, and fm_engine exposes a C ABI on top for Python/C++.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.883Z"
fingerprint: 51c97f83cc8bcfd7d95a82aa7b4396daeaf32cc3ec3500d5912c5579ea14ffd7
source:
  - path: "runtime/hsa_rt.h"
  - path: "runtime/hsa_rt.cpp"
  - path: "runtime/fm_engine.h"
  - path: "runtime/fm_engine.cpp"
---
