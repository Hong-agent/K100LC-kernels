---
uid: 345ff27d
id: k100lc-kernels.generators.gemv.qdot.reduce
parent: k100lc-kernels.generators.gemv.qdot
name: {zh: "行归约内核", en: "Row Reduction Kernel"}
description:
  zh: >
      gen_reduce_asm()：生成 reduce_blocks_k，把每行 nbpr 个 partial 求和成 y，是所有 *_dot_k 路径的固定收尾。
      
  en: >
      gen_reduce_asm(): emits reduce_blocks_k, summing the nbpr partials of each row into y — the fixed epilogue of every *_dot_k path.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.876Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 191
    end_line: 250
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:gen_reduce_asm"
    description:
      zh: >
          生成 reduce_blocks_k
          
      en: >
          Generates reduce_blocks_k
          
---
