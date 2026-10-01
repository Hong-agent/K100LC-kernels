---
uid: 6e35ca25
id: k100lc-kernels.compiler.core.stmt
parent: k100lc-kernels.compiler.core
name: {zh: "语句降级", en: "Statement Lowering"}
description:
  zh: >
      stmt/_stmt/_assign_name/_if/_for：赋值（含类型推断）、uniform if（无 else 或均匀 else）、for range、break/continue 的降级与标签生成。
      
  en: >
      stmt/_stmt/_assign_name/_if/_for: lowering of assignment (with type inference), uniform if (no else or uniform else), for-range, break/continue, with label generation.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.889Z"
fingerprint: 1d3f98afdd7d4104296ee0551a8b906dea518792e66d5bedd57a444d92b4af51
source:
  - path: "compiler/k100lc_compiler/core.py"
    line: 552
    end_line: 688
apis:
  - protocol: file
    path: "k100lc_compiler.core.CodeGen._for"
    description:
      zh: >
          for range 降级
          
      en: >
          Lowers a for-range loop
          
---
