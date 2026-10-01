---
uid: d72bb0bb
id: k100lc-kernels.asm.encoding
parent: k100lc-kernels.asm
name: {zh: "字段编码", en: "Field Encoding"}
description:
  zh: >
      把匹配得到的字段值按 encodings.json 的位宽/偏移拼成 32/64 位指令字（encode_values），含 fixed 位与域掩码。
      
  en: >
      Packs matched field values into 32/64-bit instruction words by bit width/offset (encode_values), honoring fixed bits and domain masks.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.854Z"
fingerprint: 7b44c04358b0e8348da08968d285782ab64c9eed1e942b149a189946bc97aff2
source:
  - path: "asm.py"
    line: 284
    end_line: 298
apis:
  - protocol: file
    path: "asm.py:encode_values"
    description:
      zh: >
          字段值 → 指令字
          
      en: >
          Field values to instruction word
          
deps:
  - kind: reference
    to: k100lc-kernels.asm.encodings
    from_api: "file:asm.py:encode_values"
    to_api: "file:asm.py:load_entries"
    label: {zh: "使用域/字段定义", en: "Uses domain/field definitions"}
---
