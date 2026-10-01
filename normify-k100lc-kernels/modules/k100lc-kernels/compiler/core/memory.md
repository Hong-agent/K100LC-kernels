---
uid: db33f80a
id: k100lc-kernels.compiler.core.memory
parent: k100lc-kernels.compiler.core
name: {zh: "指针与访存", en: "Pointers & Memory Access"}
description:
  zh: >
      _gid/_addr/load/store：算 work-item 全局 id、把 `ptr[ty]` 下标换算成设备地址并生成 load/store 指令。
      
  en: >
      _gid/_addr/load/store: compute the work-item global id, turn `ptr[ty]` indexing into device addresses and emit load/store instructions.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.888Z"
fingerprint: 1d3f98afdd7d4104296ee0551a8b906dea518792e66d5bedd57a444d92b4af51
source:
  - path: "compiler/k100lc_compiler/core.py"
    line: 479
    end_line: 551
apis:
  - protocol: file
    path: "k100lc_compiler.core.CodeGen.load"
    description:
      zh: >
          指针取值
          
      en: >
          Pointer load
          
---
