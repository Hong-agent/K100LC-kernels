---
uid: 492b6896
id: k100lc-kernels.kernels.asm.new.head-io.scatter-heads
parent: k100lc-kernels.kernels.asm.new.head-io
name: {zh: "Head 散开", en: "Scatter Heads"}
description:
  zh: >
      scatter_heads_k：gather 的逆操作，把连续布局按 head 索引散回原布局。
      
  en: >
      scatter_heads_k: the inverse of gather, scattering a contiguous layout back by head index.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.871Z"
fingerprint: 133fb11e1e15454f80b3d3b470bf90a7d1078f48aa3618d8298f46cbf70af07a
source:
  - path: "kernels/asm/k_new/021__Z15scatter_heads_kPfPKfiiiiix.s"
apis:
  - protocol: rpc
    path: "scatter_heads_k"
    description:
      zh: >
          按 head 散开行
          
      en: >
          Scatters rows by head
          
---
