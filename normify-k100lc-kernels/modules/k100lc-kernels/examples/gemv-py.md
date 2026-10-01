---
uid: eb5fd140
id: k100lc-kernels.examples.gemv-py
parent: k100lc-kernels.examples
name: {zh: "最小 GEMV 示例", en: "Minimal GEMV Example"}
description:
  zh: >
      python_gemv.py：分配缓冲、上传权重与激活、启动 gemv_f32_warp_k 并与 numpy 对账的最短路径。
      
  en: >
      python_gemv.py: the shortest path — allocate buffers, upload weights and activations, launch gemv_f32_warp_k and reconcile with numpy.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.894Z"
fingerprint: 9bede5aa70290cb98dfd84d7a84268db984784526ff7dffa6a9d6ef5dab05648
source:
  - path: "examples/python_gemv.py"
apis:
  - protocol: file
    path: "examples/python_gemv.py"
    description:
      zh: >
          最小 GEMV 调用示例
          
      en: >
          Minimal GEMV usage example
          
deps:
  - kind: call
    to: k100lc-kernels.python.runtime.launch
    from_api: "file:examples/python_gemv.py"
    to_api: "file:k100lc_kernels.Runtime.launch"
    label: {zh: "启动 GEMV 内核", en: "Launches the GEMV kernel"}
---
