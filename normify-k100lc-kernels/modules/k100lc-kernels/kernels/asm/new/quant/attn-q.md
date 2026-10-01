---
uid: 6f964983
id: k100lc-kernels.kernels.asm.new.quant.attn-q
parent: k100lc-kernels.kernels.asm.new.quant
name: {zh: "注意力 Q 量化", en: "Attention Q Quantization"}
description:
  zh: >
      attn_q_quant_k：把 attention 的 Q 量化成 int8 并写尺度，供 FA 的 QK^T 路径使用。
      
  en: >
      attn_q_quant_k: quantizes attention Q to int8 with scales for the QK^T path of FlashAttention.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.869Z"
fingerprint: 297992002c7b3861fc4d97978603c0d4056f12935f58d335b3d38de1cbd89a18
source:
  - path: "kernels/asm/k_new/011__Z14attn_q_quant_kPjPfPKfiiiii.s"
apis:
  - protocol: rpc
    path: "attn_q_quant_k"
    description:
      zh: >
          attention Q → int8
          
      en: >
          Attention Q to int8
          
---
