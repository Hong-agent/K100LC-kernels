---
uid: 5540177f
id: k100lc-kernels.python.model.linear.int4
parent: k100lc-kernels.python.model.linear
name: {zh: "compressed-tensors INT4 线性层", en: "compressed-tensors INT4 Linear"}
description:
  zh: >
      Int4Linear：同一份 INT4 权重可选 w4a16（原字节）/w4a8/w4a4 三条通路，预填充自动走 W4A4 GEMM，也可强制 GEMV 换取更高精度。
      
  en: >
      Int4Linear: the same INT4 weights can run w4a16 (raw bytes), w4a8 or w4a4; prefill automatically uses the W4A4 GEMM, or GEMV can be forced for higher precision.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.886Z"
fingerprint: 7f78e9c51e0530a21223674f1f32aac9e9e82a6a467fb358c04e6c3cee1a2dc1
source:
  - path: "python/k100lc_kernels/model.py"
    line: 331
    end_line: 424
apis:
  - protocol: file
    path: "k100lc_kernels.model.Int4Linear.forward_device"
    description:
      zh: >
          INT4 线性层设备侧前向
          
      en: >
          INT4 linear device-side forward
          
deps:
  - kind: call
    to: k100lc-kernels.kernels.asm.new.quant.rows
    from_api: "file:k100lc_kernels.model.Int4Linear.forward_device"
    to_api: "rpc:quant_rows_k"
    label: {zh: "量化激活到 int4", en: "Quantizes activations to int4"}
  - kind: call
    to: k100lc-kernels.kernels.asm.gemm.w4a4-flat
    from_api: "file:k100lc_kernels.model.Int4Linear.forward_device"
    to_api: "rpc:gemm_w4a4_flat"
    label: {zh: "预填充走 flat GEMM", en: "Prefill uses the flat GEMM"}
---
