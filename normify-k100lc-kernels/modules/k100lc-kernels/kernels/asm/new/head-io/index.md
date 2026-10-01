---
uid: f5f4c095
id: k100lc-kernels.kernels.asm.new.head-io
parent: k100lc-kernels.kernels.asm.new
name: {zh: "Head 重排与 QKV 切分", en: "Head Reshuffle & QKV Split"}
description:
  zh: >
      把交错存储的多 head 数据重排成按 head 连续（gather/scatter），以及从融合的 QKV 缓冲区切出 q/k/v 三段。
      
  en: >
      Reshuffles interleaved multi-head data into head-major order (gather/scatter) and splits q/k/v out of a fused QKV buffer.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.871Z"
fingerprint: 85b1c04f534f28ca844c70fada8ff41ec8efced78cf2f77d063c46692bdc9f09
source:
  - path: "kernels/asm/k_new/020__Z14gather_heads_kPfPKfiiiiix.s"
  - path: "kernels/asm/k_new/021__Z15scatter_heads_kPfPKfiiiiix.s"
  - path: "kernels/asm/k_new/026__Z11split_qkv_kPfS_S_PKfiiii.s"
---
