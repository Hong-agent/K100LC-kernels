---
uid: 65b29e26
id: k100lc-kernels.python.runtime.lifecycle
parent: k100lc-kernels.python.runtime
name: {zh: "Runtime 生命周期", en: "Runtime Lifecycle"}
description:
  zh: >
      构造函数加载 .so 与 HSACO、建立 ctypes 原型、读入 catalog；_cname/_find/_pack_argv 负责名字解析与参数打包。
      
  en: >
      The constructor loads the .so and HSACO, sets up ctypes prototypes and reads the catalog; _cname/_find/_pack_argv handle name resolution and argument packing.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.885Z"
fingerprint: 37dc62de478453fe6f5be11d36f90cce42729691db79297c865828fff2c3274d
source:
  - path: "python/k100lc_kernels/runtime.py"
    line: 10
    end_line: 103
  - path: "python/k100lc_kernels/runtime.py"
    line: 149
    end_line: 154
apis:
  - protocol: file
    path: "k100lc_kernels.Runtime.__init__"
    description:
      zh: >
          构造 Runtime（加载引擎与目录）
          
      en: >
          Constructs Runtime (loads engine and catalog)
          
deps:
  - kind: call
    to: k100lc-kernels.prebuilt.engine-lib
    from_api: "file:k100lc_kernels.Runtime.__init__"
    to_api: "file:prebuilt/libfm_engine.so"
    label: {zh: "加载引擎共享库", en: "Loads the engine shared"}
  - kind: dataflow
    to: k100lc-kernels.python.catalog.data
    from_api: "file:k100lc_kernels.Runtime.__init__"
    to_api: "file:python/k100lc_kernels/catalog.json"
    label: {zh: "读取内核目录", en: "Reads the kernel catalog"}
---
