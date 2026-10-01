---
uid: a5f99cb6
id: k100lc-kernels.runtime.hsa.dispatch
parent: k100lc-kernels.runtime.hsa
name: {zh: "内核投递", en: "Kernel Dispatch"}
description:
  zh: >
      dispatch_packet：填 kernarg（含 hidden_*）、按 grid/block 算 AQL packet 并提交信号；hsart_dispatch / hsart_dispatch_dyn 分别处理静态与动态启动。
      
  en: >
      dispatch_packet: fills kernarg (including hidden_*), builds the AQL packet from grid/block and submits the signal; hsart_dispatch / hsart_dispatch_dyn cover static and dynamic launches.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.884Z"
fingerprint: 12d13a092207d5da24b991721ce5aafd5fa3a94506ed44e0fa89cd4aaed25ea5
source:
  - path: "runtime/hsa_rt.cpp"
    line: 351
    end_line: 466
apis:
  - protocol: file
    path: "runtime/hsa_rt.cpp:hsart_dispatch_dyn"
    description:
      zh: >
          动态内核投递
          
      en: >
          Dynamic kernel dispatch
          
deps:
  - kind: call
    to: k100lc-kernels.runtime.hsa.hsaco-load
    from_api: "file:runtime/hsa_rt.cpp:hsart_dispatch_dyn"
    to_api: "file:runtime/hsa_rt.cpp:load_hsaco"
    label: {zh: "按名取内核符号", en: "Looks up kernel symbols"}
---
