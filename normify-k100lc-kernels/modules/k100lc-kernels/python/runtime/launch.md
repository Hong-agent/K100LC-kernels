---
uid: 55db8518
id: k100lc-kernels.python.runtime.launch
parent: k100lc-kernels.python.runtime
name: {zh: "内核启动", en: "Kernel Launch"}
description:
  zh: >
      launch/launch2d/launch_dyn/launch_any/has/sync：按 catalog 的参数偏移填 kernarg，支持 1D/2D 与动态内核；sync 统一同步设备。
      
  en: >
      launch/launch2d/launch_dyn/launch_any/has/sync: fills kernarg from catalog offsets, supports 1D/2D and dynamic kernels, and syncs the device.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.886Z"
fingerprint: 37dc62de478453fe6f5be11d36f90cce42729691db79297c865828fff2c3274d
source:
  - path: "python/k100lc_kernels/runtime.py"
    line: 139
    end_line: 191
apis:
  - protocol: file
    path: "k100lc_kernels.Runtime.launch"
    description:
      zh: >
          启动内核（grid = workgroup 数）
          
      en: >
          Launches a kernel (grid = workgroup count)
          
  - protocol: file
    path: "k100lc_kernels.Runtime.launch_dyn"
    description:
      zh: >
          动态内核启动
          
      en: >
          Dynamic kernel launch
          
deps:
  - kind: call
    to: k100lc-kernels.runtime.engine.launch
    from_api: "file:k100lc_kernels.Runtime.launch"
    to_api: "file:runtime/fm_engine.cpp:fm_launch_dyn"
    label: {zh: "调 fm_launch/fm_launch2d", en: "Calls fm_launch/fm_launch2d"}
---
