---
uid: 761691f4
id: k100lc-kernels.generators.gemv.qdot.iq-codecs
parent: k100lc-kernels.generators.gemv.qdot
name: {zh: "IQ 系列编码", en: "IQ Codec Family"}
description:
  zh: >
      IQ 编码的解码点积生成器族：IQ4_NL、IQ2_S、IQ3_S、IQ4_XS、IQ3_XXS，靠查表（LUT）还原码值与符号，各自维护独立的块布局与解码序列。
      
  en: >
      The decode-and-dot generator family for the IQ codecs: IQ4_NL, IQ2_S, IQ3_S, IQ4_XS and IQ3_XXS, restoring values and signs through lookup tables, each keeping its own block layout and decode sequence.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.876Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 251
    end_line: 610
  - path: "tools/gen_gemv_qdot.py"
    line: 948
    end_line: 1036
---
