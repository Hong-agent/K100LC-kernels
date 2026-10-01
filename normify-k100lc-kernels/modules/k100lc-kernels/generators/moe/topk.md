---
uid: 8ad1fc4a
id: k100lc-kernels.generators.moe.topk
parent: k100lc-kernels.generators.moe
name: {zh: "行 Top-K 生成器", en: "Row Top-K Generator"}
description:
  zh: >
      gen_topk.py：一行一个线程的插入式 top-k（KMAX=16），给出下标与值（topk_k）。
      
  en: >
      gen_topk.py: insertion-based top-k with one thread per row (KMAX=16), returning indices and values (topk_k).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.882Z"
fingerprint: 336a0eef2371e288a3e7ddf9dccd3064321d7111e59916ff0552da42f1ec9e23
source:
  - path: "tools/gen_topk.py"
apis:
  - protocol: file
    path: "tools/gen_topk.py"
    description:
      zh: >
          生成 topk_k
          
      en: >
          Generates topk_k
          
---
