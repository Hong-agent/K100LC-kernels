---
uid: c50bfae1
id: k100lc-kernels.generators.gdn
parent: k100lc-kernels.generators
name: {zh: "GDN 小算子融合生成器", en: "GDN Operator Fusion Generator"}
description:
  zh: >
      把 `split_qkv + 2×l2norm` 三次启动融合成一个内核，省掉每层两次约 20µs 的启动开销。
      
  en: >
      Fuses `split_qkv + 2×l2norm` into a single kernel, removing two ~20µs launches per layer.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.883Z"
fingerprint: e03966fd6e008468f81f0ddf9be22709d8ea197c68a36b33eba3b46d29e004d3
source:
  - path: "tools/gen_gdn_split_norm.py"
---
