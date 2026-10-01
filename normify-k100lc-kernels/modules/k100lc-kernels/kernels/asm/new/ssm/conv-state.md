---
uid: 27162c50
id: k100lc-kernels.kernels.asm.new.ssm.conv-state
parent: k100lc-kernels.kernels.asm.new.ssm
name: {zh: "卷积状态更新", en: "Conv State Update"}
description:
  zh: >
      conv_state_update_k：滑动窗口卷积状态的滚动更新（保留最近 T 个 token 的输入）。
      
  en: >
      conv_state_update_k: rolling update of the sliding-window convolution state (keeps the last T token inputs).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.871Z"
fingerprint: 6689f1bb12e11ce0cad52254126bf89a1393fbaa784eda54ac4295ecb79efe44
source:
  - path: "kernels/asm/k_new/017__Z19conv_state_update_kPfS_PKfS1_iii.s"
apis:
  - protocol: rpc
    path: "conv_state_update_k"
    description:
      zh: >
          卷积状态滚动
          
      en: >
          Rolls the convolution state
          
---
