---
uid: 0374679d
id: k100lc-kernels.kernels.asm.new.norm.rmsnorm-gated
parent: k100lc-kernels.kernels.asm.new.norm
name: {zh: "门控 RMSNorm", en: "Gated RMSNorm"}
description:
  zh: >
      rmsnorm_gated_k：RMSNorm 后乘门控向量，用于 SSM/GDN 层。
      
  en: >
      rmsnorm_gated_k: RMSNorm followed by a gating vector multiply, used by SSM/GDN layers.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.868Z"
fingerprint: 8a40afa59ae7bd2cdfe1694e746525da70edd8f3f48b73fca42d07499b8e7779
source:
  - path: "kernels/asm/k_new/025__Z15rmsnorm_gated_kPfPKfS1_S1_if.s"
apis:
  - protocol: rpc
    path: "rmsnorm_gated_k"
    description:
      zh: >
          RMSNorm + 门控
          
      en: >
          RMSNorm with gating
          
---
