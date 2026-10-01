---
uid: a765c4ea
id: k100lc-kernels.compiler.core.buffers
parent: k100lc-kernels.compiler.core
name: {zh: "临时寄存器管理", en: "Temporary Register Management"}
description:
  zh: >
      alloc_tmp_v/free_tmp_v/alloc_addr_pair/free_addr_pair：VGPR 与地址对（s 对）的临时分配与回收。
      
  en: >
      alloc_tmp_v/free_tmp_v/alloc_addr_pair/free_addr_pair: allocation and recycling of temporary VGPRs and address pairs (s-pairs).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.888Z"
fingerprint: 1d3f98afdd7d4104296ee0551a8b906dea518792e66d5bedd57a444d92b4af51
source:
  - path: "compiler/k100lc_compiler/core.py"
    line: 101
    end_line: 131
apis:
  - protocol: file
    path: "k100lc_compiler.core.CodeGen.alloc_tmp_v"
    description:
      zh: >
          分配临时 VGPR
          
      en: >
          Allocates a temporary VGPR
          
---
