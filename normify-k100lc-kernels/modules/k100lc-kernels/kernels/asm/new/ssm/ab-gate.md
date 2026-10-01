---
uid: ee0a169a
id: k100lc-kernels.kernels.asm.new.ssm.ab-gate
parent: k100lc-kernels.kernels.asm.new.ssm
name: {zh: "SSM α/β 门控", en: "SSM Alpha/Beta Gate"}
description:
  zh: >
      ssm_ab_gate_k：由输入与状态算出 SSM 的 A/B 门控系数。
      
  en: >
      ssm_ab_gate_k: computes the SSM A/B gate coefficients from the input and state.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.871Z"
fingerprint: dd60d68b969cee57594771cfcd71fa6b3df81d8334491d8b29c579e5edc7a0e9
source:
  - path: "kernels/asm/k_new/020__Z13ssm_ab_gate_kPfS_S_PKfS1_S1_S1_S1_ii.s"
apis:
  - protocol: rpc
    path: "ssm_ab_gate_k"
    description:
      zh: >
          SSM 门控系数计算
          
      en: >
          Computes SSM gate coefficients
          
---
