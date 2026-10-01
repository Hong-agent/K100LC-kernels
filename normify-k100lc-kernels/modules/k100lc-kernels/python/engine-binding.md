---
uid: 17c849c1
id: k100lc-kernels.python.engine-binding
parent: k100lc-kernels.python
name: {zh: "引擎 ctypes 绑定", en: "Engine ctypes Binding"}
description:
  zh: >
      tools/fm_engine.py：直接 ctypes 调 libfm_engine.so 的 Engine 类（alloc/upload/launch/sync），与 Rust/C++/脚本环境共享同一份 C ABI，不经过 k100lc_kernels 包。
      
  en: >
      tools/fm_engine.py: the Engine class calling libfm_engine.so directly through ctypes (alloc/upload/launch/sync), sharing the same C ABI with Rust/C++/script environments without going through the k100lc_kernels package.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.894Z"
fingerprint: 6fa99e5d4164e616386b867073e8fecae27476bf617c886459a7280e3c4bb900
source:
  - path: "tools/fm_engine.py"
apis:
  - protocol: file
    path: "tools/fm_engine.py:Engine.launch"
    description:
      zh: >
          直接调 C ABI 启动内核
          
      en: >
          Launches a kernel straight through the C ABI
          
deps:
  - kind: call
    to: k100lc-kernels.prebuilt.engine-lib
    from_api: "file:tools/fm_engine.py:Engine.launch"
    to_api: "file:prebuilt/libfm_engine.so"
    label: {zh: "ctypes 调 C ABI", en: "Calls the C ABI via ctypes"}
---
