---
uid: 1d764b9d
id: k100lc-kernels.kernels.headers
parent: k100lc-kernels.kernels
name: {zh: "内核块参数头", en: "Kernel Block-Parameter Headers"}
description:
  zh: >
      各组内核的编译期参数：分块尺寸、量化组、打包粒度与设备端点积/LDS 辅助函数，是汇编源与内核语义的对照文档。
      
  en: >
      Compile-time parameters for each kernel family: tile sizes, quantization groups, packing granularity and device-side dot-product/LDS helpers that document the assembly sources.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.858Z"
fingerprint: 496a9bad87c1222ccee7d386608c8b54ae9b117462f4b3a51b3c35af2282996c
source:
  - path: "kernels/flash_attn_core.h"
  - path: "kernels/gemm_core.h"
  - path: "kernels/gemv_w4a4_core.h"
  - path: "kernels/gemv_w4a8_core.h"
  - path: "kernels/kv_pack.h"
---
