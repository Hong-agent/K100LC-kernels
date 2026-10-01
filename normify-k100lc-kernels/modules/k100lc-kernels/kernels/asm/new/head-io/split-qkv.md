---
uid: 2acbb915
id: k100lc-kernels.kernels.asm.new.head-io.split-qkv
parent: k100lc-kernels.kernels.asm.new.head-io
name: {zh: "QKV 切分", en: "QKV Split"}
description:
  zh: >
      split_qkv_k：从融合的 QKV 缓冲区切出 Q/K/V 三个视图，供后续 QK-Norm 与注意力使用。
      
  en: >
      split_qkv_k: splits the fused QKV buffer into Q/K/V views for QK-Norm and attention.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.871Z"
fingerprint: 9a834ffcf772e3deb98379f8bf2ecaf32a3f7b9019b9815f38e4eb91de8c74d2
source:
  - path: "kernels/asm/k_new/026__Z11split_qkv_kPfS_S_PKfiiii.s"
apis:
  - protocol: rpc
    path: "split_qkv_k"
    description:
      zh: >
          融合 QKV → q/k/v
          
      en: >
          Fused QKV to q/k/v
          
---
