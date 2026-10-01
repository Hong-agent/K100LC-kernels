---
uid: ef16b3f2
id: k100lc-kernels.asm.instr-match
parent: k100lc-kernels.asm
name: {zh: "指令匹配", en: "Instruction Matching"}
description:
  zh: >
      把 (mnemonic, 操作数列表) 匹配到唯一 Entry：按域逐位比较操作数类型并抽取字段值，匹配失败时报行号与候选。
      
  en: >
      Matches a (mnemonic, operand list) pair to a unique Entry by comparing operand kinds per domain and extracting field values; reports line number and candidates on failure.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.854Z"
fingerprint: 7b44c04358b0e8348da08968d285782ab64c9eed1e942b149a189946bc97aff2
source:
  - path: "asm.py"
    line: 244
    end_line: 283
  - path: "asm.py"
    line: 370
    end_line: 379
apis:
  - protocol: file
    path: "asm.py:match_entry"
    description:
      zh: >
          操作数模式匹配
          
      en: >
          Operand-pattern matching
          
deps:
  - kind: call
    to: k100lc-kernels.asm.operands
    from_api: "file:asm.py:match_entry"
    to_api: "file:asm.py:parse_domain"
    label: {zh: "解析操作数域", en: "Parses operand domains"}
  - kind: call
    to: k100lc-kernels.asm.encodings
    from_api: "file:asm.py:match_entry"
    to_api: "file:asm.py:load_entries"
    label: {zh: "按 mnemonic 取候选", en: "Gets candidates by mnemonic"}
---
