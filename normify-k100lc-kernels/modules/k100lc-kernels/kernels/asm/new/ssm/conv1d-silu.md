---
uid: ae9fa68d
id: k100lc-kernels.kernels.asm.new.ssm.conv1d-silu
parent: k100lc-kernels.kernels.asm.new.ssm
name: {zh: "Conv1D + SiLU", en: "Conv1D + SiLU"}
description:
  zh: >
      conv1d_silu_k：深度可分离 conv1d 与 SiLU 激活融合成一个内核，减少一次启动与一趟访存。
      
  en: >
      conv1d_silu_k: fuses depthwise conv1d with the SiLU activation into one kernel, saving a launch and a memory pass.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.871Z"
fingerprint: 633308debcd35b2730bb9c01a1ac1c0a5d7c5ad3098ce90bdfd014bc8a0f7f7b
source:
  - path: "kernels/asm/k_new/018__Z13conv1d_silu_kPfPKfS1_S1_iii.s"
apis:
  - protocol: rpc
    path: "conv1d_silu_k"
    description:
      zh: >
          conv1d 融合 SiLU
          
      en: >
          Conv1d fused with SiLU
          
---
