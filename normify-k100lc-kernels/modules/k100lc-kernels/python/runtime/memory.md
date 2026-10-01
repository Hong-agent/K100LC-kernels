---
uid: 60c5205e
id: k100lc-kernels.python.runtime.memory
parent: k100lc-kernels.python.runtime
name: {zh: "设备内存操作", en: "Device Memory Operations"}
description:
  zh: >
      alloc/free/upload/download/copy_dev/memset：把 numpy 数组与设备地址互转的薄封装。
      
  en: >
      alloc/free/upload/download/copy_dev/memset: thin wrappers converting between numpy arrays and device addresses.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.885Z"
fingerprint: 37dc62de478453fe6f5be11d36f90cce42729691db79297c865828fff2c3274d
source:
  - path: "python/k100lc_kernels/runtime.py"
    line: 104
    end_line: 138
apis:
  - protocol: file
    path: "k100lc_kernels.Runtime.upload"
    description:
      zh: >
          numpy → 设备
          
      en: >
          numpy to device
          
  - protocol: file
    path: "k100lc_kernels.Runtime.download"
    description:
      zh: >
          设备 → numpy
          
      en: >
          device to numpy
          
deps:
  - kind: call
    to: k100lc-kernels.runtime.engine.memory
    from_api: "file:k100lc_kernels.Runtime.upload"
    to_api: "file:runtime/fm_engine.cpp:fm_upload"
    label: {zh: "调 fm_upload/fm_download", en: "Calls fm_upload/fm_download"}
---
