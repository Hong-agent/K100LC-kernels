---
uid: 17a28a8f
id: k100lc-kernels.python.quant.int4-pack
parent: k100lc-kernels.python.quant
name: {zh: "INT4 打包/解包", en: "INT4 Pack/Unpack"}
description:
  zh: >
      pack_int4_group128 / dequant_int4_group128：每 8 个 4bit 码打包进一个 I32、每 128 元素一个尺度的对称打包与参考反量化。
      
  en: >
      pack_int4_group128 / dequant_int4_group128: pack 8 four-bit codes into one I32 with one scale per 128 elements, and the reference dequantizer.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.887Z"
fingerprint: ba5c4b7622b8f6ca986027d789484d3c31a1e5ccab27f23977be9020da6357ea
source:
  - path: "python/k100lc_kernels/quant.py"
    line: 33
    end_line: 69
apis:
  - protocol: file
    path: "k100lc_kernels.quant.pack_int4_group128"
    description:
      zh: >
          INT4 打包（组 128）
          
      en: >
          Packs INT4, group 128
          
---
