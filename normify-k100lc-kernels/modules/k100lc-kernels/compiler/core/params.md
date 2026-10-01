---
uid: cc7e6e22
id: k100lc-kernels.compiler.core.params
parent: k100lc-kernels.compiler.core
name: {zh: "参数与函数编译", en: "Parameters & Function Compilation"}
description:
  zh: >
      Param 与 _parse_params/compile：按声明顺序分配 kernarg 偏移与寄存器，生成函数序言并组织整体编译流程。
      
  en: >
      Param with _parse_params/compile: assign kernarg offsets and registers in declaration order, emit the prologue and orchestrate the overall compilation.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.888Z"
fingerprint: 1d3f98afdd7d4104296ee0551a8b906dea518792e66d5bedd57a444d92b4af51
source:
  - path: "compiler/k100lc_compiler/core.py"
    line: 65
    end_line: 79
  - path: "compiler/k100lc_compiler/core.py"
    line: 132
    end_line: 202
apis:
  - protocol: file
    path: "k100lc_compiler.core.CodeGen.compile"
    description:
      zh: >
          编译一个内核函数
          
      en: >
          Compiles one kernel function
          
---
