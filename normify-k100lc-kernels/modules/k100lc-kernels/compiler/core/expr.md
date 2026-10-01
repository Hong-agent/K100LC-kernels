---
uid: 02ded2ad
id: k100lc-kernels.compiler.core.expr
parent: k100lc-kernels.compiler.core
name: {zh: "表达式求值", en: "Expression Evaluation"}
description:
  zh: >
      is_uniform/as_vreg/as_sreg/type_of/expr：判断值是否均匀、把 Val 落到 VGPR/SGPR，并按 AST 节点推出表达式类型与寄存器表示。
      
  en: >
      is_uniform/as_vreg/as_sreg/type_of/expr: decide value uniformity, lower a Val into VGPR/SGPR and infer expression type and register form from the AST node.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.888Z"
fingerprint: 1d3f98afdd7d4104296ee0551a8b906dea518792e66d5bedd57a444d92b4af51
source:
  - path: "compiler/k100lc_compiler/core.py"
    line: 203
    end_line: 257
apis:
  - protocol: file
    path: "k100lc_compiler.core.CodeGen.expr"
    description:
      zh: >
          AST 表达式 → Val
          
      en: >
          AST expression to Val
          
---
