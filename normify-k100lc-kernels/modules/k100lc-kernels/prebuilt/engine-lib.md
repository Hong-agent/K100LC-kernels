---
uid: 0776a4a9
id: k100lc-kernels.prebuilt.engine-lib
parent: k100lc-kernels.prebuilt
name: {zh: "常驻引擎共享库", en: "Resident Engine Library"}
description:
  zh: >
      libfm_engine.so：C ABI 的常驻 HSA 引擎，提供 alloc/upload/launch/sync 等接口，Python 与 C++ 都通过它驱动内核。
      
  en: >
      libfm_engine.so: the C-ABI resident HSA engine exposing alloc/upload/launch/sync; both the Python and C++ front-ends drive kernels through it.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.852Z"
fingerprint: a8468ff095518517b19ef4e443f5a27fba3d233a5547f002d593395d38ea6f04
source:
  - path: "prebuilt/libfm_engine.so"
apis:
  - protocol: file
    path: "prebuilt/libfm_engine.so"
    description:
      zh: >
          引擎 C ABI（fm_init/fm_launch/…）
          
      en: >
          Engine C ABI (fm_init/fm_launch/…)
          
deps:
  - kind: reference
    to: k100lc-kernels.runtime.engine
    from_api: "file:prebuilt/libfm_engine.so"
    label: {zh: "由引擎 C ABI 实现", en: "Implemented by the engine C"}
---
