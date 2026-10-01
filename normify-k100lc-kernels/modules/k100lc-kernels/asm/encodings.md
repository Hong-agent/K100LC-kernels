---
uid: 6a93d43c
id: k100lc-kernels.asm.encodings
parent: k100lc-kernels.asm
name: {zh: "编码表与指令表", en: "Encoding Table & Entry Loading"}
description:
  zh: >
      从 encodings.json 载入 gfx926 指令编码表，构造 Entry（mnemonic、操作数域、字段、位型、宽度），并按 mnemonic 建索引缓存供汇编复用。
      
  en: >
      Loads the gfx926 instruction encoding table from encodings.json, builds Entry records (mnemonic, operand domains, fields, bit pattern, width) and caches them indexed by mnemonic.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.854Z"
fingerprint: 38dff88ec85b29104db381cca9965b6212b12ecb83eb3f88e21762f060e9009d
source:
  - path: "asm.py"
    line: 34
    end_line: 95
  - path: "encodings.json"
apis:
  - protocol: file
    path: "asm.py:load_entries"
    description:
      zh: >
          载入并按 mnemonic 缓存编码表
          
      en: >
          Loads and caches the encoding table by mnemonic
          
---
