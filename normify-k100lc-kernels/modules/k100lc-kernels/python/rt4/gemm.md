---
uid: a7d0ee6c
id: k100lc-kernels.python.rt4.gemm
parent: k100lc-kernels.python.rt4
name: {zh: "W4 GEMM 通路", en: "W4 GEMM Path"}
description:
  zh: >
      W4Runner.gemm_device/gemm：预填充走 quant_rows_k + gemm_w4a4，按 BM=128 分块拆成多次 1D 启动。
      
  en: >
      W4Runner.gemm_device/gemm: prefill goes through quant_rows_k plus gemm_w4a4, split into repeated 1D launches per BM=128 tile.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.888Z"
fingerprint: 9c30915ad83bb227a417d704f682c99a5d6f67f22200a81824d911530e2d9359
source:
  - path: "python/k100lc_kernels/rt4.py"
    line: 288
    end_line: 359
apis:
  - protocol: file
    path: "k100lc_kernels.rt4.W4Runner.gemm"
    description:
      zh: >
          RT4 GEMM 预填充
          
      en: >
          RT4 GEMM prefill
          
deps:
  - kind: call
    to: k100lc-kernels.kernels.asm.gemv.act-quant.act4
    from_api: "file:k100lc_kernels.rt4.W4Runner.gemm"
    to_api: "rpc:quant_act4"
    label: {zh: "激活量化前置", en: "Activation quantization"}
  - kind: call
    to: k100lc-kernels.kernels.asm.gemm.w4a4
    from_api: "file:k100lc_kernels.rt4.W4Runner.gemm"
    to_api: "rpc:gemm_w4a4"
    label: {zh: "W4A4 GEMM", en: "W4A4 GEMM"}
---
