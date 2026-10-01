---
uid: bf502aeb
id: k100lc-kernels.python.model.linear.rt4
parent: k100lc-kernels.python.model.linear
name: {zh: "RT4 线性层", en: "RT4 Linear"}
description:
  zh: >
      RT4Linear：RT4 格式 INT4 权重的 W4A8 / W4A4 通路选择和激活量化前置。
      
  en: >
      RT4Linear: path selection between W4A8 and W4A4 for RT4-format INT4 weights, with the activation-quantization pre-step.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.886Z"
fingerprint: 7f78e9c51e0530a21223674f1f32aac9e9e82a6a467fb358c04e6c3cee1a2dc1
source:
  - path: "python/k100lc_kernels/model.py"
    line: 303
    end_line: 330
apis:
  - protocol: file
    path: "k100lc_kernels.model.RT4Linear.forward"
    description:
      zh: >
          RT4 线性层前向
          
      en: >
          RT4 linear forward
          
deps:
  - kind: call
    to: k100lc-kernels.kernels.asm.gemv.w4a8.m1
    from_api: "file:k100lc_kernels.model.RT4Linear.forward"
    to_api: "rpc:gemv_w4a8<1,false,1>"
    label: {zh: "W4A8 GEMV 通路", en: "W4A8 GEMV path"}
  - kind: call
    to: k100lc-kernels.kernels.asm.gemv.w4a4.m1
    from_api: "file:k100lc_kernels.model.RT4Linear.forward"
    to_api: "rpc:gemv_w4a4<1>"
    label: {zh: "W4A4 GEMV 通路", en: "W4A4 GEMV path"}
---
