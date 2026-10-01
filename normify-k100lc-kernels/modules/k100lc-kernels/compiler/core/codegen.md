---
uid: e48895d3
id: k100lc-kernels.compiler.core.codegen
parent: k100lc-kernels.compiler.core
name: {zh: "CodeGen 驱动与发射", en: "CodeGen Driver & Emission"}
description:
  zh: >
      CodeGen.__init__/emit/label/new_label：维护汇编输出缓冲与标签、驱动整个编译流程。
      
  en: >
      CodeGen.__init__/emit/label/new_label: maintain the assembly output buffer and labels and drive the whole compilation.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.888Z"
fingerprint: 1d3f98afdd7d4104296ee0551a8b906dea518792e66d5bedd57a444d92b4af51
source:
  - path: "compiler/k100lc_compiler/core.py"
    line: 80
    end_line: 100
apis:
  - protocol: file
    path: "k100lc_compiler.core.CodeGen.emit"
    description:
      zh: >
          发射一行汇编
          
      en: >
          Emits one assembly line
          
---
