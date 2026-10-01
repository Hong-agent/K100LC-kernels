---
uid: 042180b1
id: k100lc-kernels.kernels.asm.new.gguf-dequant.q6k
parent: k100lc-kernels.kernels.asm.new.gguf-dequant
name: {zh: "Q6_K 解码", en: "Q6_K Dequant"}
description:
  zh: >
      q6k_dequant：每块 210 字节 / 256 元素，6bit 码由低 4 位与 2 位高位平面拼出。
      
  en: >
      q6k_dequant: 210 bytes / 256 elements per block, assembling 6-bit codes from a low nibble and a 2-bit plane.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.872Z"
fingerprint: 67d817391d01c050497202d1b514f0516e2900fe50666710bd439af61caf2cea
source:
  - path: "kernels/asm/k_new/q6k_dequant.s"
apis:
  - protocol: rpc
    path: "q6k_dequant"
    description:
      zh: >
          Q6_K → f32 解码
          
      en: >
          Q6_K to f32 dequant
          
---
