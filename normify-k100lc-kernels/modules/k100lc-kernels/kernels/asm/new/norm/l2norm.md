---
uid: 0ab5a52d
id: k100lc-kernels.kernels.asm.new.norm.l2norm
parent: k100lc-kernels.kernels.asm.new.norm
name: {zh: "L2 归一化", en: "L2 Normalization"}
description:
  zh: >
      l2norm_k(x, S, eps)：对 x 的每行原地做 L2 归一化，用于 QK-Norm。
      
  en: >
      l2norm_k(x, S, eps): in-place L2 normalization of each row of x, used for QK-Norm.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.868Z"
fingerprint: 088f480eeafb7093664b97aeacc3cd3c3106662245365c2b1d6dd1687674efff
source:
  - path: "kernels/asm/k_new/007__Z8l2norm_kPfif.s"
apis:
  - protocol: rpc
    path: "l2norm_k"
    description:
      zh: >
          行 L2 归一化（原地）
          
      en: >
          In-place row L2 normalization
          
---
