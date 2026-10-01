---
uid: 086ca298
id: k100lc-kernels.runtime.engine.lifecycle
parent: k100lc-kernels.runtime.engine
name: {zh: "引擎初始化与内核探测", en: "Engine Init & Kernel Probe"}
description:
  zh: >
      fm_init（加载 HSACO 并初始化 hsa_rt）与 fm_has_kernel（探测内核是否存在），以及显式参数计数 explicit_args。
      
  en: >
      fm_init (loads the HSACO and initializes hsa_rt), fm_has_kernel (probes kernel existence) and the explicit_args counter.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.884Z"
fingerprint: 6bc4a856398ca20995322aeda5e4bccea896e1547be0d304f151737be9ff3762
source:
  - path: "runtime/fm_engine.cpp"
    line: 21
    end_line: 41
  - path: "runtime/fm_engine.cpp"
    line: 67
    end_line: 74
apis:
  - protocol: file
    path: "runtime/fm_engine.cpp:fm_init"
    description:
      zh: >
          初始化引擎
          
      en: >
          Initializes the engine
          
deps:
  - kind: call
    to: k100lc-kernels.runtime.hsa.init
    from_api: "file:runtime/fm_engine.cpp:fm_init"
    to_api: "file:runtime/hsa_rt.cpp:hsart_lookup"
    label: {zh: "初始化 HSA 运行时", en: "Initializes the HSA runtime"}
  - kind: call
    to: k100lc-kernels.runtime.hsa.hsaco-load
    from_api: "file:runtime/fm_engine.cpp:fm_init"
    label: {zh: "加载 HSACO", en: "Loads the HSACO"}
---
