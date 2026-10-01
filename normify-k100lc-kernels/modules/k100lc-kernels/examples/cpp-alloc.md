---
uid: 25a2c32c
id: k100lc-kernels.examples.cpp-alloc
parent: k100lc-kernels.examples
name: {zh: "C++ 分配示例", en: "C++ Allocation Example"}
description:
  zh: >
      cpp_alloc.cpp：链接 libfm_engine.so 的最小 C++ 程序（fm_init/fm_alloc/fm_upload/fm_sync/fm_free）。
      
  en: >
      cpp_alloc.cpp: the smallest C++ program linking libfm_engine.so (fm_init/fm_alloc/fm_upload/fm_sync/fm_free).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.895Z"
fingerprint: 728a284ec00204e33ef8d4f1d72bb072944996ce5f6882e5a0726ee9bd5ab213
source:
  - path: "examples/cpp_alloc.cpp"
apis:
  - protocol: file
    path: "examples/cpp_alloc.cpp"
    description:
      zh: >
          最小 C++ 引擎调用
          
      en: >
          Minimal C++ engine usage
          
deps:
  - kind: call
    to: k100lc-kernels.runtime.engine.abi
    from_api: "file:examples/cpp_alloc.cpp"
    to_api: "file:runtime/fm_engine.h"
    label: {zh: "调引擎 C ABI", en: "Calls the engine C ABI"}
---
