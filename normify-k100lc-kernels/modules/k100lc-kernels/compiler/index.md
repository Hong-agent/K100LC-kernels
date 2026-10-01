---
uid: fbe77b5c
id: k100lc-kernels.compiler
parent: k100lc-kernels
name: {zh: "DSL 编译器", en: "DSL Compiler"}
description:
  zh: >
      受限 Python DSL → gfx926 汇编 → HSACO 的编译器：内核/参数注解、内建函数、赋值/if/for/指针 load/store，编译产物为可动态启动的 HSACO。
      
  en: >
      Compiler from a restricted Python DSL to gfx926 assembly and HSACO: kernel/parameter annotations, builtins, assignment/if/for/pointer load-store, emitting a dynamically launchable HSACO.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.888Z"
fingerprint: a3386cca3f25e81c5b05432e39f0d46be1ff922566e895ce365afe7792216151
source:
  - path: "compiler/k100lc_compiler/core.py"
  - path: "compiler/k100lc_compiler/__main__.py"
  - path: "compiler/k100lc_compiler/__init__.py"
  - path: "compiler/README.md"
---
