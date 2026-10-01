---
uid: 1e8cdfbd
id: k100lc-kernels.generators.moe.router-top10
parent: k100lc-kernels.generators.moe
name: {zh: "Router Top-10 生成器", en: "Router Top-10 Generator"}
description:
  zh: >
      gen_router_top10.py：每行取 logits 最大的 10 个专家并做 softmax/renormalize（router_top10_k）。
      
  en: >
      gen_router_top10.py: picks the top-10 experts per row and softmax/renormalizes the weights (router_top10_k).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.882Z"
fingerprint: e9ffa953b530c6aa837d6329e23b99c8931cdc8788edb553ca03a062afc6b30f
source:
  - path: "tools/gen_router_top10.py"
apis:
  - protocol: file
    path: "tools/gen_router_top10.py"
    description:
      zh: >
          生成 router_top10_k
          
      en: >
          Generates router_top10_k
          
---
