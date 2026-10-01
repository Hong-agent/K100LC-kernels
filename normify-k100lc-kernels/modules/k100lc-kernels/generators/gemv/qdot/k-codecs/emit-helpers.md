---
uid: 554ea95a
id: k100lc-kernels.generators.gemv.qdot.k-codecs.emit-helpers
parent: k100lc-kernels.generators.gemv.qdot.k-codecs
name: {zh: "K 系列公共发射器", en: "K-Family Emitters"}
description:
  zh: >
      _emit_waddr/_emit_nibble/_emit_k_scale：权重块地址推进、半字节拆取与 6bit 子块尺度提取的公共汇编发射器。
      
  en: >
      _emit_waddr/_emit_nibble/_emit_k_scale: the shared emitters for weight-block addressing, nibble extraction and 6-bit sub-block scale extraction.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.877Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 719
    end_line: 782
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:_emit_k_scale"
    description:
      zh: >
          子块尺度提取发射器
          
      en: >
          Sub-block scale emitter
          
---
