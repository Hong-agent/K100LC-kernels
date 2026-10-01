---
uid: d716e3e9
id: k100lc-kernels.kernels.asm.new.quant
parent: k100lc-kernels.kernels.asm.new
name: {zh: "激活量化内核", en: "Activation Quantization Kernels"}
description:
  zh: >
      把 f32 激活量化成 int4/int8 打包表示并输出组尺度：通用行式量化与 attention 专用的 Q 量化。
      
  en: >
      Quantizes f32 activations into packed int4/int8 with group scales: the general row-wise quantizer and the attention-specific Q quantizer.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.869Z"
fingerprint: fc1e93c317eef98153a53f0018850c534d608a238ed9d75b96b45359c945f3a5
source:
  - path: "kernels/asm/k_new/010__Z12quant_rows_kPjPfPKfiiiii.s"
  - path: "kernels/asm/k_new/012__Z15quant_rows_a8_kPjS_PfS0_PKfiiiii.s"
  - path: "kernels/asm/k_new/011__Z14attn_q_quant_kPjPfPKfiiiii.s"
---
