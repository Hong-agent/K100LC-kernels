---
uid: a15c3eaa
id: k100lc-kernels.compiler.core.types
parent: k100lc-kernels.compiler.core
name: {zh: "类型与注解解析", en: "Type & Annotation Parsing"}
description:
  zh: >
      ELEM_SIZE/_align/_ty_name/parse_type：把 `ptr[f32]`、字符串与 Name 注解解析成 (类型, 是否指针)，并给出元素大小与对齐。
      
  en: >
      ELEM_SIZE/_align/_ty_name/parse_type: parse `ptr[f32]`, string and Name annotations into (type, is-pointer), with element size and alignment.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.888Z"
fingerprint: 1d3f98afdd7d4104296ee0551a8b906dea518792e66d5bedd57a444d92b4af51
source:
  - path: "compiler/k100lc_compiler/core.py"
    line: 35
    end_line: 64
apis:
  - protocol: file
    path: "k100lc_compiler.core.parse_type"
    description:
      zh: >
          解析 DSL 注解类型
          
      en: >
          Parses a DSL annotation type
          
---
