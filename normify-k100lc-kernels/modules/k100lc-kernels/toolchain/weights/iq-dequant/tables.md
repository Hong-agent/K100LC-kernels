---
uid: a4e272c1
id: k100lc-kernels.toolchain.weights.iq-dequant.tables
parent: k100lc-kernels.toolchain.weights.iq-dequant
name: {zh: "参考解码公共部件", en: "Reference Decoder Helpers"}
description:
  zh: >
      模块常量 kmask/ksigns 与 _grid_bytes/_f16/_blocks 辅助：把码表展成网格、读 f16 尺度并把原始字节切成块。
      
  en: >
      Module constants kmask/ksigns with the _grid_bytes/_f16/_blocks helpers: expand code tables into grids, read f16 scales and slice raw bytes into blocks.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.893Z"
fingerprint: e06a3daff752350ceab3660a1a4df6b12c1f1cf261e7fb1cfb8638b4744a7d81
source:
  - path: "tools/iq_dequant.py"
    line: 53
    end_line: 82
apis:
  - protocol: file
    path: "tools/iq_dequant.py:_blocks"
    description:
      zh: >
          原始字节 → 块数组
          
      en: >
          Raw bytes to block array
          
---
