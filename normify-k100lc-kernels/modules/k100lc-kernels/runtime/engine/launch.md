---
uid: 6eeb5d2d
id: k100lc-kernels.runtime.engine.launch
parent: k100lc-kernels.runtime.engine
name: {zh: "引擎启动接口", en: "Engine Launch API"}
description:
  zh: >
      fm_launch / fm_launch2d / fm_launch_dyn：按内核表的参数偏移与大小填 kernarg，支持 1D/2D 与动态（编译期未知）启动。
      
  en: >
      fm_launch / fm_launch2d / fm_launch_dyn: fill kernarg per the kernel table’s offsets and sizes, supporting 1D/2D and dynamic (compile-time-unknown) launches.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.884Z"
fingerprint: 6bc4a856398ca20995322aeda5e4bccea896e1547be0d304f151737be9ff3762
source:
  - path: "runtime/fm_engine.cpp"
    line: 75
    end_line: 133
apis:
  - protocol: file
    path: "runtime/fm_engine.cpp:fm_launch_dyn"
    description:
      zh: >
          动态内核启动
          
      en: >
          Dynamic kernel launch
          
deps:
  - kind: call
    to: k100lc-kernels.runtime.hsa.dispatch
    from_api: "file:runtime/fm_engine.cpp:fm_launch_dyn"
    to_api: "file:runtime/hsa_rt.cpp:hsart_dispatch_dyn"
    label: {zh: "投递内核", en: "Dispatches the kernel"}
---
