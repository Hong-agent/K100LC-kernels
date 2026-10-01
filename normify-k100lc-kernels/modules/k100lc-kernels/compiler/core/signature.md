---
uid: 8d8a507d
id: k100lc-kernels.compiler.core.signature
parent: k100lc-kernels.compiler.core
name: {zh: "内核签名与资源计数", en: "Kernel Signature & Resource Counts"}
description:
  zh: >
      hidden_args/arg_meta/vgpr_count/sgpr_count：算出隐藏参数列表、kernarg 布局与所需 SGPR/VGPR 数量，写入 HSACO 的 kernel descriptor。
      
  en: >
      hidden_args/arg_meta/vgpr_count/sgpr_count: compute the hidden argument list, kernarg layout and required SGPR/VGPR counts written into the HSACO kernel descriptor.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.889Z"
fingerprint: 1d3f98afdd7d4104296ee0551a8b906dea518792e66d5bedd57a444d92b4af51
source:
  - path: "compiler/k100lc_compiler/core.py"
    line: 689
    end_line: 722
apis:
  - protocol: file
    path: "k100lc_compiler.core.CodeGen.arg_meta"
    description:
      zh: >
          Kernarg 布局与段需求
          
      en: >
          Kernarg layout and segment needs
          
---
