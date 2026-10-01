---
uid: 0777d671
id: k100lc-kernels.kernels.headers.gemm
parent: k100lc-kernels.kernels.headers
name: {zh: "W4A4 GEMM 块参数", en: "W4A4 GEMM Block Parameters"}
description:
  zh: >
      gemm_core.h：BM/BN/GD/QG/TM/TN 分块与流水步长、激活量化 quant_group、打包 pack8 与 LDS 装载 ldv，以及 gemm_w4a4 参考实现。
      
  en: >
      gemm_core.h: BM/BN/GD/QG/TM/TN tiling and pipeline steps, the quant_group activation quantizer, pack8 packing and ldv LDS loading, plus the gemm_w4a4 reference implementation.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.859Z"
fingerprint: 62238dc2a6cc42e60d230c7b68e1208ee21ede90a12010beb408a134fe3a297c
source:
  - path: "kernels/gemm_core.h"
    line: 1
    end_line: 206
apis:
  - protocol: file
    path: "kernels/gemm_core.h:gemm_w4a4"
    description:
      zh: >
          W4A4 GEMM 参考实现
          
      en: >
          Reference W4A4 GEMM implementation
          
deps:
  - kind: reference
    to: k100lc-kernels.kernels.asm.gemm.w4a4
    from_api: "file:kernels/gemm_core.h:gemm_w4a4"
    to_api: "rpc:gemm_w4a4"
    label: {zh: "块参数与参考实现", en: "Block parameters and reference"}
---
