---
uid: 1d5150d5
id: k100lc-kernels.kernels.asm.new.position.rope
parent: k100lc-kernels.kernels.asm.new.position
name: {zh: "RoPE", en: "RoPE"}
description:
  zh: >
      rope_k：按位置与 θ 基做旋转位置编码，支持部分维度与交错/半旋转布局。
      
  en: >
      rope_k: applies rotary position encoding from position and theta base, supporting partial dimensions and interleaved/half-rotation layouts.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.870Z"
fingerprint: 7b1d31cd31e5601300668193f812cc5d968ed0d9f2113dee9c85edbb25da32f9
source:
  - path: "kernels/asm/k_new/009__Z6rope_kPfS_PKiiiiiiiif.s"
apis:
  - protocol: rpc
    path: "rope_k"
    description:
      zh: >
          旋转位置编码
          
      en: >
          Rotary position encoding
          
---
