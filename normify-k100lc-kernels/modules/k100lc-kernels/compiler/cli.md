---
uid: 451cdcef
id: k100lc-kernels.compiler.cli
parent: k100lc-kernels.compiler
name: {zh: "编译器 CLI", en: "Compiler CLI"}
description:
  zh: >
      __main__.py：`build`（编译 .kkl，可选 --emit-asm 打印汇编）与 `list`（列出文件里的内核）两个子命令。
      
  en: >
      __main__.py: the `build` subcommand (compiles .kkl, optionally printing assembly with --emit-asm) and `list` (lists the kernels in a file).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.889Z"
fingerprint: 7c43b1ed5bd1d0ad40920af06c0111602b33db81c60f1e23af5e926fa92ae177
source:
  - path: "compiler/k100lc_compiler/__main__.py"
apis:
  - protocol: file
    path: "k100lc_compiler.__main__.main"
    description:
      zh: >
          build / list 子命令
          
      en: >
          build / list subcommands
          
deps:
  - kind: call
    to: k100lc-kernels.compiler.core.api
    from_api: "file:k100lc_compiler.__main__.main"
    to_api: "file:k100lc_compiler.compile_file"
    label: {zh: "编译 .kkl", en: "Compiles .kkl"}
---
