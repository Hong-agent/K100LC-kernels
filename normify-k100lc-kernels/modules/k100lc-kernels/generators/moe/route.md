---
uid: 9a3e4a92
id: k100lc-kernels.generators.moe.route
parent: k100lc-kernels.generators.moe
name: {zh: "MoE 分桶路由生成器", en: "MoE Bucketed Routing Generator"}
description:
  zh: >
      gen_moe_route.py：生成 gather_rows_k（按索引把行 gather 成桶）与 moe_combine_gather_k（按 base 偏移合并专家输出）。
      
  en: >
      gen_moe_route.py: produces gather_rows_k (index-based row bucketing) and moe_combine_gather_k (combining expert outputs by base offsets).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.882Z"
fingerprint: 739385540ca757c3e6d643c382a5491e4202acb35ee0ffb879b98a60a8bc126f
source:
  - path: "tools/gen_moe_route.py"
apis:
  - protocol: file
    path: "tools/gen_moe_route.py"
    description:
      zh: >
          生成 gather_rows_k / moe_combine_gather_k
          
      en: >
          Generates gather_rows_k / moe_combine_gather_k
          
---
