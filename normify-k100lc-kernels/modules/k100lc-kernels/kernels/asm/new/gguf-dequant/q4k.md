---
uid: 42e324e7
id: k100lc-kernels.kernels.asm.new.gguf-dequant.q4k
parent: k100lc-kernels.kernels.asm.new.gguf-dequant
name: {zh: "Q4_K 解码", en: "Q4_K Dequant"}
description:
  zh: >
      q4k_dequant：每块 144 字节 / 256 元素，含两组 f16 尺度与 6bit 子块尺度的 4bit 解码。
      
  en: >
      q4k_dequant: 144 bytes / 256 elements per block, decoding 4-bit codes with two f16 scales and 6-bit sub-block scales.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.872Z"
fingerprint: 685e7c868e922e1589da51f966c17f5f39f597aea25da9bb8dcf593b27e640dc
source:
  - path: "kernels/asm/k_new/q4k_dequant.s"
apis:
  - protocol: rpc
    path: "q4k_dequant"
    description:
      zh: >
          Q4_K → f32 解码
          
      en: >
          Q4_K to f32 dequant
          
---
