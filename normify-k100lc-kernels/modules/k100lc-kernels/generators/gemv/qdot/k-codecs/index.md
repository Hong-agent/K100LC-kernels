---
uid: 7ab168e5
id: k100lc-kernels.generators.gemv.qdot.k-codecs
parent: k100lc-kernels.generators.gemv.qdot
name: {zh: "K/Q 系列编码", en: "K-Quant & Legacy Codec Family"}
description:
  zh: >
      Q2_0/Q4_0/Q8_0 与 K-系列（Q4_K/Q5_K/Q6_K）的解码点积生成器族，共用权重地址推进、半字节提取与子块尺度提取三个公共发射器。
      
  en: >
      The decode-and-dot generator family for Q2_0/Q4_0/Q8_0 and the K-family (Q4_K/Q5_K/Q6_K), sharing the weight-addressing, nibble-extraction and sub-block-scale emitters.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.877Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 487
    end_line: 947
---
