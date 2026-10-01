---
uid: c588c197
id: k100lc-kernels.asm.sdwa
parent: k100lc-kernels.asm
name: {zh: "SDWA 编码", en: "SDWA Encoding"}
description:
  zh: >
      处理带 SDWA（sub-dword addressing）修饰的指令：把 src0/src1 的选取与合并字段写进 sdwa 扩展字节。
      
  en: >
      Handles SDWA (sub-dword addressing) instructions: writes src0/src1 select and merge fields into the sdwa extension bytes.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.855Z"
fingerprint: 7b44c04358b0e8348da08968d285782ab64c9eed1e942b149a189946bc97aff2
source:
  - path: "asm.py"
    line: 322
    end_line: 369
apis:
  - protocol: file
    path: "asm.py:encode_sdwa"
    description:
      zh: >
          SDWA 指令扩展编码
          
      en: >
          Encodes SDWA instruction extensions
          
---
