---
uid: 8b0cd829
id: k100lc-kernels.runtime.hsa.hip-shim
parent: k100lc-kernels.runtime.hsa
name: {zh: "HIP 兼容层", en: "HIP Compatibility Layer"}
description:
  zh: >
      以 hip* 命名的兼容实现：hipMalloc/hipHostMalloc/hipMemcpy(Async/2D)/hipMemset、stream 与 event 的哑实现、hipDeviceSynchronize。
      
  en: >
      HIP-named compatible implementations: hipMalloc/hipHostMalloc/hipMemcpy(Async/2D)/hipMemset, dummy stream and event implementations, and hipDeviceSynchronize.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.884Z"
fingerprint: 12d13a092207d5da24b991721ce5aafd5fa3a94506ed44e0fa89cd4aaed25ea5
source:
  - path: "runtime/hsa_rt.cpp"
    line: 467
    end_line: 582
apis:
  - protocol: file
    path: "runtime/hsa_rt.cpp:hipMemcpy"
    description:
      zh: >
          HIP 风格拷贝入口
          
      en: >
          HIP-style copy entry point
          
deps:
  - kind: call
    to: k100lc-kernels.runtime.hsa.staging
    from_api: "file:runtime/hsa_rt.cpp:hipMemcpy"
    to_api: "file:runtime/hsa_rt.cpp:copy_h2d"
    label: {zh: "走暂存区拷贝", en: "Copies through the staging"}
---
