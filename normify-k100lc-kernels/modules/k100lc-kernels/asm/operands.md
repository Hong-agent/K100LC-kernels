---
uid: 0b417cf1
id: k100lc-kernels.asm.operands
parent: k100lc-kernels.asm
name: {zh: "操作数域解析", en: "Operand Domain Parsing"}
description:
  zh: >
      解析 vsrc0/ssrc/ssrc_pair/vreg/sreg/spair 等操作数域：识别 v/s 寄存器、FLOAT_CONST 浮点常量与 inline const，返回可直接编码的整数。
      
  en: >
      Parses operand domains (vsrc0/ssrc/ssrc_pair/vreg/sreg/spair): recognizes v/s registers, FLOAT_CONST literals and inline constants, returning encodable integers.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.854Z"
fingerprint: 7b44c04358b0e8348da08968d285782ab64c9eed1e942b149a189946bc97aff2
source:
  - path: "asm.py"
    line: 98
    end_line: 243
apis:
  - protocol: file
    path: "asm.py:parse_domain"
    description:
      zh: >
          按域解析寄存器/立即数文本
          
      en: >
          Resolves register/immediate text per domain
          
---
