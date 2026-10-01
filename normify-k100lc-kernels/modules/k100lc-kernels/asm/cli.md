---
uid: 0d4afda3
id: k100lc-kernels.asm.cli
parent: k100lc-kernels.asm
name: {zh: "汇编驱动与 CLI", en: "Assemble Driver & CLI"}
description:
  zh: >
      把源文件汇编为机器码并落地：逐指令编码、按标签计算分支偏移、输出二进制与统计，同时提供命令行入口。
      
  en: >
      Turns a source file into machine code on disk: encodes each instruction, resolves label-relative branch offsets, writes the binary and stats, and provides the CLI entry point.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.855Z"
fingerprint: 7b44c04358b0e8348da08968d285782ab64c9eed1e942b149a189946bc97aff2
source:
  - path: "asm.py"
    line: 380
    end_line: 499
apis:
  - protocol: file
    path: "asm.py:assemble"
    description:
      zh: >
          汇编单个 .s 文件
          
      en: >
          Assembles one .s file
          
deps:
  - kind: call
    to: k100lc-kernels.asm.encodings
    from_api: "file:asm.py:assemble"
    to_api: "file:asm.py:load_entries"
    label: {zh: "载入编码表", en: "Loads the encoding table"}
  - kind: call
    to: k100lc-kernels.asm.instr-match
    from_api: "file:asm.py:assemble"
    to_api: "file:asm.py:match_entry"
    label: {zh: "逐指令匹配", en: "Matches each instruction"}
  - kind: call
    to: k100lc-kernels.asm.sdwa
    from_api: "file:asm.py:assemble"
    to_api: "file:asm.py:encode_sdwa"
    label: {zh: "SDWA 指令编码", en: "Encodes SDWA instructions"}
  - kind: call
    to: k100lc-kernels.asm.source-parse
    from_api: "file:asm.py:assemble"
    to_api: "file:asm.py:parse_source"
    label: {zh: "解析 .s 源文件", en: "Parses .s sources"}
---
