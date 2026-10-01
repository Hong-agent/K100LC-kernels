---
uid: a1b2c3d4
id: k100lc-kernels.asm
parent: k100lc-kernels
name: {zh: "自研汇编器", en: "Assembler"}
description:
  zh: >
      把 gfx926 汇编子集汇编成机器码：由 encodings.json 编码表驱动，匹配 mnemonic 与操作数域、编码立即数/SDWA，产出 .text 二进制。
      
  en: >
      Assembles the gfx926 assembly subset into machine code, driven by the encodings.json table: mnemonic and operand-domain matching, immediate/SDWA encoding, .text output.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.896Z"
fingerprint: 7b44c04358b0e8348da08968d285782ab64c9eed1e942b149a189946bc97aff2
source:
  - path: "asm.py"
---
