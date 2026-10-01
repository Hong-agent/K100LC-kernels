---
uid: 7640f2f6
id: k100lc-kernels.kernels
parent: k100lc-kernels
name: {zh: "内核源码与元数据", en: "Kernel Sources & Metadata"}
description:
  zh: >
      122 个预编译内核的来源：自研汇编语法的手写/生成 .s 源、块参数头（分块、量化组、打包宏），以及驱动构建的 kernel_spec.json 元数据。
      
  en: >
      Sources of the 122 prebuilt kernels: hand-written/generated .s files in the home-grown assembly syntax, block-parameter headers (tiling, quant groups, packing macros), and the kernel_spec.json metadata driving the build.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.858Z"
fingerprint: 05938d5d80849e87ff4a5a5cf6f545ca97b3df9faddf206fc737df96355aa6b2
source:
  - path: "kernels/kernel_spec.json"
  - path: "kernels/flash_attn_core.h"
  - path: "kernels/gemm_core.h"
  - path: "kernels/gemv_w4a4_core.h"
  - path: "kernels/gemv_w4a8_core.h"
  - path: "kernels/kv_pack.h"
---
