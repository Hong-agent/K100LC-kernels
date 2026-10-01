---
uid: f7bb4050
id: k100lc-kernels.kernels.asm.new.ssm
parent: k100lc-kernels.kernels.asm.new
name: {zh: "SSM / 卷积内核", en: "SSM / Convolution Kernels"}
description:
  zh: >
      状态空间与卷积路径：卷积状态更新、conv1d+SiLU 融合、GDN 门控 delta 网络（含 k2 小批量变体）与 SSM 的 α/β 门控。
      
  en: >
      State-space and convolution path: conv state update, fused conv1d+SiLU, the gated delta network (GDN, including the k2 small-batch variant) and the SSM alpha/beta gate.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.870Z"
fingerprint: 41a3eaff4a88ba4022dbd336d49c29888beaa25f51a1bca25a5f8a70762c6f51
source:
  - path: "kernels/asm/k_new/017__Z19conv_state_update_kPfS_PKfS1_iii.s"
  - path: "kernels/asm/k_new/018__Z13conv1d_silu_kPfPKfS1_S1_iii.s"
  - path: "kernels/asm/k_new/019__Z5gdn_kPfPKfS1_S1_S1_S1_S_iiiiii.s"
  - path: "kernels/asm/k_new/020__Z13ssm_ab_gate_kPfS_S_PKfS1_S1_S1_S1_ii.s"
  - path: "kernels/asm/k_new/027__Z6gdn_k2ILi32EEvPfPKfS2_S2_S2_S2_S0_S0_iiiii.s"
---
