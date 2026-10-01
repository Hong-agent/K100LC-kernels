---
uid: f5309aaf
id: k100lc-kernels.python.rt4.gemv
parent: k100lc-kernels.python.rt4
name: {zh: "W4 GEMV 通路", en: "W4 GEMV Path"}
description:
  zh: >
      W4Runner.gemv_device/gemv：按 kind（w4a8/w4a4）选择 quant_act 前置与对应 GEMV 内核，处理 M=1..4 的启动参数。
      
  en: >
      W4Runner.gemv_device/gemv: selects the quant_act pre-step and matching GEMV kernel by kind (w4a8/w4a4), handling launch parameters for M=1..4.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.887Z"
fingerprint: 9c30915ad83bb227a417d704f682c99a5d6f67f22200a81824d911530e2d9359
source:
  - path: "python/k100lc_kernels/rt4.py"
    line: 216
    end_line: 287
apis:
  - protocol: file
    path: "k100lc_kernels.rt4.W4Runner.gemv"
    description:
      zh: >
          RT4 GEMV（W4A8/W4A4）
          
      en: >
          RT4 GEMV (W4A8/W4A4)
          
deps:
  - kind: call
    to: k100lc-kernels.kernels.asm.gemv.act-quant.act4
    from_api: "file:k100lc_kernels.rt4.W4Runner.gemv"
    to_api: "rpc:quant_act4"
    label: {zh: "激活量化前置", en: "Activation quantization"}
  - kind: call
    to: k100lc-kernels.kernels.asm.gemv.w4a8.m1
    from_api: "file:k100lc_kernels.rt4.W4Runner.gemv"
    to_api: "rpc:gemv_w4a8<1,false,1>"
    label: {zh: "W4A8 GEMV", en: "W4A8 GEMV"}
---
