---
uid: 9c0dba05
id: k100lc-kernels.compiler.core.api
parent: k100lc-kernels.compiler.core
name: {zh: "编译入口", en: "Compilation Entry Points"}
description:
  zh: >
      CompileError 与 compile_source/compile_file：把 DSL 源码编译成 .s 与 HSACO 并返回产物路径（可只编 one 指定的内核）。
      
  en: >
      CompileError with compile_source/compile_file: compile DSL source into .s and HSACO and return artifact paths (optionally compiling only the kernel named by `only`).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.889Z"
fingerprint: 1d3f98afdd7d4104296ee0551a8b906dea518792e66d5bedd57a444d92b4af51
source:
  - path: "compiler/k100lc_compiler/core.py"
    line: 31
    end_line: 34
  - path: "compiler/k100lc_compiler/core.py"
    line: 723
    end_line: 759
apis:
  - protocol: file
    path: "k100lc_compiler.compile_file"
    description:
      zh: >
          编译 .kkl 文件
          
      en: >
          Compiles a .kkl file
          
deps:
  - kind: call
    to: k100lc-kernels.asm.cli
    from_api: "file:k100lc_compiler.compile_file"
    to_api: "file:asm.py:assemble"
    label: {zh: "发射 gfx926 汇编", en: "Emits gfx926 assembly"}
  - kind: call
    to: k100lc-kernels.hsaco.multi-builder
    from_api: "file:k100lc_compiler.compile_file"
    to_api: "file:tools/make_hsaco_multi.py"
    label: {zh: "产出 HSACO", en: "Produces a HSACO"}
  - kind: call
    to: k100lc-kernels.toolchain.hsa-probe.lab
    from_api: "file:k100lc_compiler.compile_file"
    to_api: "file:tools/kernel_lab.py:build_one"
    label: {zh: "编译产物上卡自检", en: "Runs compiled kernels on"}
---
