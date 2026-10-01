---
uid: 4715e84e
id: k100lc-kernels.toolchain.weights.iq-dequant.dispatch
parent: k100lc-kernels.toolchain.weights.iq-dequant
name: {zh: "参考解码分发", en: "Reference Dispatch"}
description:
  zh: >
      DEQUANT 表、dequantize() 按类型名分发与自检 main()，是脚本与外部工具的统一入口。
      
  en: >
      The DEQUANT table, dequantize() dispatch by type name and the self-test main(): the unified entry for scripts and external tools.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.893Z"
fingerprint: e06a3daff752350ceab3660a1a4df6b12c1f1cf261e7fb1cfb8638b4744a7d81
source:
  - path: "tools/iq_dequant.py"
    line: 440
    end_line: 488
apis:
  - protocol: file
    path: "tools/iq_dequant.py:dequantize"
    description:
      zh: >
          按编码名解码任意块
          
      en: >
          Decodes any block by codec name
          
deps:
  - kind: reference
    to: k100lc-kernels.generators.gemv.qdot.iq-codecs.iq4nl
    from_api: "file:tools/iq_dequant.py:dequantize"
    to_api: "file:tools/gen_gemv_qdot.py:gen_iq4nl_dot_asm"
    label: {zh: "作为 IQ 内核金标准", en: "Golden reference for IQ"}
  - kind: reference
    to: k100lc-kernels.generators.gemv.qdot.k-codecs.q4k
    from_api: "file:tools/iq_dequant.py:dequantize"
    to_api: "file:tools/gen_gemv_qdot.py:gen_q4k_dot_asm"
    label: {zh: "作为 K/Q 内核金标准", en: "Golden reference for K/Q"}
  - kind: call
    to: k100lc-kernels.toolchain.weights.iq-tables
    from_api: "file:tools/iq_dequant.py:dequantize"
    to_api: "file:tools/iq_tables.py"
    label: {zh: "查码本与符号表", en: "Looks up codebooks and sign"}
---
