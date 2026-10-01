---
uid: 901c4c6f
id: k100lc-kernels.generators.moe
parent: k100lc-kernels.generators
name: {zh: "MoE 与采样生成器", en: "MoE & Sampling Generators"}
description:
  zh: >
      MoE 路由/合并与采样路径的生成器：按索引 gather 行 + 分桶合并、router top-10、行 top-k。
      
  en: >
      Generators for the MoE routing/combine and sampling paths: index-based row gather plus bucketed combine, router top-10 and row top-k.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.882Z"
fingerprint: f6fcaa9a19b06ddbb9d83314c32de74d0962c70f56dd9a3ed3ba786452f66c25
source:
  - path: "tools/gen_moe_route.py"
  - path: "tools/gen_moe_combine.py"
  - path: "tools/gen_router_top10.py"
  - path: "tools/gen_topk.py"
---
