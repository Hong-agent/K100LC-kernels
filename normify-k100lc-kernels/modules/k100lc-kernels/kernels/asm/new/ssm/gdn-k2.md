---
uid: 6071dc24
id: k100lc-kernels.kernels.asm.new.ssm.gdn-k2
parent: k100lc-kernels.kernels.asm.new.ssm
name: {zh: "GDN 小批量变体", en: "GDN Small-Batch Variant"}
description:
  zh: >
      gdn_k2<32>：模板化的 GDN 小批量版本，workgroup 内协作处理 32 个通道块。
      
  en: >
      gdn_k2<32>: the templated small-batch GDN variant with in-workgroup cooperation over 32 channel blocks.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.871Z"
fingerprint: c2c9f7dcb19bb1c9bc41c7c44a4e34f5ed0ed4c3b3ccefedd8d38b46ccf9f206
source:
  - path: "kernels/asm/k_new/027__Z6gdn_k2ILi32EEvPfPKfS2_S2_S2_S2_S0_S0_iiiii.s"
apis:
  - protocol: rpc
    path: "gdn_k2<32>"
    description:
      zh: >
          GDN 小批量变体（块 32）
          
      en: >
          GDN small-batch variant, block 32
          
---
