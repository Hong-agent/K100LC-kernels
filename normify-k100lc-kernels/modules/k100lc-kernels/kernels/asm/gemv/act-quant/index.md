---
uid: f31bf86a
id: k100lc-kernels.kernels.asm.gemv.act-quant
parent: k100lc-kernels.kernels.asm.gemv
name: {zh: "激活量化（GEMV 前置）", en: "Activation Quantization (GEMV Pre-step)"}
description:
  zh: >
      把 f32 激活量化成 int4（quant_act4）/ int8（quant_act），并输出逐组尺度，供同目录的 W4A4/W4A8 GEMV 直接消费。
      
  en: >
      Quantizes f32 activations to int4 (quant_act4) / int8 (quant_act) and emits per-group scales consumed directly by the sibling W4A4/W4A8 GEMV kernels.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.861Z"
fingerprint: c806e3eb8de7486328ea96cdacd534e610af0c9217a8ee470fb1f9fe17005230
source:
  - path: "kernels/asm/k_gemv/001__Z10quant_act4PKfPhPfii.s"
  - path: "kernels/asm/k_gemv/002__Z9quant_actPKfPaS1_PfPiii.s"
---
