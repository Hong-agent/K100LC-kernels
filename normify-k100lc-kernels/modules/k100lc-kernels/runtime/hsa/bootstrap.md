---
uid: 1a4b0b1a
id: k100lc-kernels.runtime.hsa.bootstrap
parent: k100lc-kernels.runtime.hsa
name: {zh: "设备与内存池发现", en: "Agent & Memory Pool Discovery"}
description:
  zh: >
      HSA 初始化回调：枚举 GPU/CPU agent、按 GLOBAL_FLAG 选择 kernarg/data/host（pinned）pool，并提供带 allow_access 的 pool_alloc 与拷贝暂存区。
      
  en: >
      HSA init callbacks: enumerate GPU/CPU agents, pick kernarg/data/host (pinned) pools by GLOBAL_FLAG, and provide pool_alloc with allow_access plus a copy staging area.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.883Z"
fingerprint: 12d13a092207d5da24b991721ce5aafd5fa3a94506ed44e0fa89cd4aaed25ea5
source:
  - path: "runtime/hsa_rt.cpp"
    line: 44
    end_line: 157
apis:
  - protocol: file
    path: "runtime/hsa_rt.cpp:cb_pool"
    description:
      zh: >
          按全局标志选择内存池
          
      en: >
          Selects memory pools by global flags
          
deps:
  - kind: reference
    to: k100lc-kernels.runtime.hsa.types
    from_api: "file:runtime/hsa_rt.cpp:cb_pool"
    label: {zh: "填充内核参数表", en: "Fills the kernel argument"}
---
