---
uid: "92369921"
id: k100lc-kernels.runtime.engine.memory
parent: k100lc-kernels.runtime.engine
name: {zh: "引擎内存接口", en: "Engine Memory API"}
description:
  zh: >
      fm_alloc/fm_free/fm_upload/fm_download/fm_copy/fm_memset/fm_sync：设备内存生命周期与同步，是 Python Runtime 的底层。
      
  en: >
      fm_alloc/fm_free/fm_upload/fm_download/fm_copy/fm_memset/fm_sync: device-memory lifecycle and synchronization, the layer beneath the Python Runtime.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.884Z"
fingerprint: 6bc4a856398ca20995322aeda5e4bccea896e1547be0d304f151737be9ff3762
source:
  - path: "runtime/fm_engine.cpp"
    line: 42
    end_line: 66
apis:
  - protocol: file
    path: "runtime/fm_engine.cpp:fm_upload"
    description:
      zh: >
          主机 → 设备上传
          
      en: >
          Host-to-device upload
          
deps:
  - kind: call
    to: k100lc-kernels.runtime.hsa.hip-shim
    from_api: "file:runtime/fm_engine.cpp:fm_upload"
    to_api: "file:runtime/hsa_rt.cpp:hipMemcpy"
    label: {zh: "HIP 拷贝实现", en: "HIP copy implementations"}
---
