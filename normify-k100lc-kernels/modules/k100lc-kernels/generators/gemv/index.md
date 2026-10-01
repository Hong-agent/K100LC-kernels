---
uid: 5ef32ffc
id: k100lc-kernels.generators.gemv
parent: k100lc-kernels.generators
name: {zh: "GEMV 生成器", en: "GEMV Generators"}
description:
  zh: >
      解码期主力算子的生成器：f32/int8 通用 GEMV、warp-per-row 合并访存变体、W4A4 双行快路，以及量化权重原生解码点积（qdot）。
      
  en: >
      Generators for the decode-time workhorses: generic f32/int8 GEMV, the warp-per-row coalesced variant, the two-row W4A4 fast path, and the native quantized-weight decode-and-dot family (qdot).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.875Z"
fingerprint: 9812ab6d46276cb823aa371b5ae3be4363745a5dbc02eedec3e67cb0cacac743
source:
  - path: "tools/gen_gemv_f32.py"
  - path: "tools/gen_gemv_f32_warp.py"
  - path: "tools/gen_gemv_i8.py"
  - path: "tools/gen_gemv_w4a4_r2.py"
  - path: "tools/gen_gemv_qdot.py"
---
