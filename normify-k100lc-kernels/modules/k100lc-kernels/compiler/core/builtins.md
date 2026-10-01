---
uid: 38c7c985
id: k100lc-kernels.compiler.core.builtins
parent: k100lc-kernels.compiler.core
name: {zh: "运算与内建函数", en: "Operators & Builtins"}
description:
  zh: >
      binop/compare/call：算术与比较的类型提升与指令选择，以及 gid()/tid()/bid()/lane()、exp/sqrt/rsqrt/fma/fabs/max、load16/位操作等内建的代码生成；cdiv 等标量辅助也在这里。
      
  en: >
      binop/compare/call: type promotion and instruction selection for arithmetic and comparison, plus code generation for builtins such as gid()/tid()/bid()/lane(), exp/sqrt/rsqrt/fma/fabs/max, load16 and bit operations; scalar helpers like cdiv live here too.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.888Z"
fingerprint: 1d3f98afdd7d4104296ee0551a8b906dea518792e66d5bedd57a444d92b4af51
source:
  - path: "compiler/k100lc_compiler/core.py"
    line: 258
    end_line: 478
apis:
  - protocol: file
    path: "k100lc_compiler.core.CodeGen.call"
    description:
      zh: >
          内建函数调用降级
          
      en: >
          Lowers builtin function calls
          
---
