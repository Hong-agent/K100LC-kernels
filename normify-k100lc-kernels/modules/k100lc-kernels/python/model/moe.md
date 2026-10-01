---
uid: 2b938d53
id: k100lc-kernels.python.model.moe
parent: k100lc-kernels.python.model
name: {zh: "MoE 算子", en: "MoE Operators"}
description:
  zh: >
      MoECombine（稠密专家输出加权合并）与 MoEExperts（token→expert 分桶：NumPy 向量化路由 + gather_rows_k + 专家 GEMV + moe_combine_gather_k）。
      
  en: >
      MoECombine (dense weighted merge of expert outputs) and MoEExperts (token-to-expert bucketing: vectorized NumPy routing, gather_rows_k, expert GEMV, moe_combine_gather_k).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.886Z"
fingerprint: 7f78e9c51e0530a21223674f1f32aac9e9e82a6a467fb358c04e6c3cee1a2dc1
source:
  - path: "python/k100lc_kernels/model.py"
    line: 514
    end_line: 654
apis:
  - protocol: file
    path: "k100lc_kernels.model.MoEExperts.route"
    description:
      zh: >
          token → 专家分桶（向量化）
          
      en: >
          Token-to-expert bucketing (vectorized)
          
  - protocol: file
    path: "k100lc_kernels.model.MoECombine.forward"
    description:
      zh: >
          专家输出加权合并
          
      en: >
          Weighted merge of expert outputs
          
deps:
  - kind: call
    to: k100lc-kernels.generators.moe.route
    from_api: "file:k100lc_kernels.model.MoEExperts.route"
    to_api: "file:tools/gen_moe_route.py"
    label: {zh: "gather_rows_k 分桶", en: "Buckets rows with"}
  - kind: call
    to: k100lc-kernels.generators.moe.combine
    from_api: "file:k100lc_kernels.model.MoEExperts.route"
    to_api: "file:tools/gen_moe_combine.py"
    label: {zh: "专家输出合并", en: "Merges expert outputs"}
---
