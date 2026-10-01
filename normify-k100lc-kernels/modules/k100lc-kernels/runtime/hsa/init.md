---
uid: 99196d40
id: k100lc-kernels.runtime.hsa.init
parent: k100lc-kernels.runtime.hsa
name: {zh: "运行时初始化与内核查找", en: "Runtime Init & Kernel Lookup"}
description:
  zh: >
      hsart_init（幂等初始化，默认读 $RT_HSACO）与 hsart_lookup（按名取 RtKernel），是引擎启动与每次 launch 的入口。
      
  en: >
      hsart_init (idempotent init, defaults to $RT_HSACO) and hsart_lookup (RtKernel by name): the entry points for engine startup and every launch.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.884Z"
fingerprint: 12d13a092207d5da24b991721ce5aafd5fa3a94506ed44e0fa89cd4aaed25ea5
source:
  - path: "runtime/hsa_rt.cpp"
    line: 325
    end_line: 350
apis:
  - protocol: file
    path: "runtime/hsa_rt.cpp:hsart_lookup"
    description:
      zh: >
          按名字查内核
          
      en: >
          Looks up a kernel by name
          
---
