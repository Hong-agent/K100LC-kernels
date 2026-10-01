---
uid: 9e6cd169
id: k100lc-kernels.asm.source-parse
parent: k100lc-kernels.asm
name: {zh: "汇编源解析", en: "Assembly Source Parsing"}
description:
  zh: >
      逐行读取 .s 源：跳过注释/空白，识别 `.text` 段与 `name:` 标签，把助记符与操作数切成 Instruction 列表。
      
  en: >
      Reads .s sources line by line: skips comments/blanks, recognizes .text sections and `name:` labels, and splits mnemonics/operands into Instruction lists.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.855Z"
fingerprint: 7b44c04358b0e8348da08968d285782ab64c9eed1e942b149a189946bc97aff2
source:
  - path: "asm.py"
    line: 299
    end_line: 321
apis:
  - protocol: file
    path: "asm.py:parse_source"
    description:
      zh: >
          解析 .s 为指令序列
          
      en: >
          Parses .s into an instruction list
          
---
