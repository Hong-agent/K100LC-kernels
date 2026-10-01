---
uid: cb79c944
id: k100lc-kernels.kernels.asm.new.head-io.gather-heads
parent: k100lc-kernels.kernels.asm.new.head-io
name: {zh: "Head 收集", en: "Gather Heads"}
description:
  zh: >
      gather_heads_k：按 head 索引把源行收集成连续布局。
      
  en: >
      gather_heads_k: gathers source rows into a contiguous layout by head index.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.871Z"
fingerprint: a10ba545ffe4bc34f44215067bf07ec90bd850265cf34715070a8aaef8922e7a
source:
  - path: "kernels/asm/k_new/020__Z14gather_heads_kPfPKfiiiiix.s"
apis:
  - protocol: rpc
    path: "gather_heads_k"
    description:
      zh: >
          按 head 收集行
          
      en: >
          Gathers rows by head
          
---
