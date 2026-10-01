---
uid: 02c9cff7
id: k100lc-kernels.kernels.asm.new.ssm.gdn
parent: k100lc-kernels.kernels.asm.new.ssm
name: {zh: "GDN 主体", en: "GDN Core"}
description:
  zh: >
      gdn_k：门控 delta 网络的状态更新与输出，private_segment>0（当前运行时无 scratch backing）。
      
  en: >
      gdn_k: state update and output of the gated delta network; private_segment > 0 (no scratch backing in the current runtime).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.871Z"
fingerprint: 79e358bb4aaed6609d644c70b334b27dfeaf6a32df85147bb52c9e6731bcd956
source:
  - path: "kernels/asm/k_new/019__Z5gdn_kPfPKfS1_S1_S1_S1_S_iiiiii.s"
apis:
  - protocol: rpc
    path: "gdn_k"
    description:
      zh: >
          GDN 状态更新与输出
          
      en: >
          GDN state update and output
          
---
