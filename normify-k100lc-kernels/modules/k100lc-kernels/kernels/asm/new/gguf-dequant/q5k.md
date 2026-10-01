---
uid: 28893c40
id: k100lc-kernels.kernels.asm.new.gguf-dequant.q5k
parent: k100lc-kernels.kernels.asm.new.gguf-dequant
name: {zh: "Q5_K 解码", en: "Q5_K Dequant"}
description:
  zh: >
      q5k_dequant：每块 176 字节 / 256 元素，在 Q4_K 基础上加了高位平面（5bit 码）。
      
  en: >
      q5k_dequant: 176 bytes / 256 elements per block, adding a high-bit plane on top of Q4_K (5-bit codes).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.872Z"
fingerprint: 968a85552726df6433ce4fb51d83e428194a88ba342e662fdd07b007361d1181
source:
  - path: "kernels/asm/k_new/q5k_dequant.s"
apis:
  - protocol: rpc
    path: "q5k_dequant"
    description:
      zh: >
          Q5_K → f32 解码
          
      en: >
          Q5_K to f32 dequant
          
---
