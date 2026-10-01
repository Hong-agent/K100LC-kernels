---
uid: 7ca75b63
id: k100lc-kernels.generators.gemv.qdot.iq-codecs.helpers
parent: k100lc-kernels.generators.gemv.qdot.iq-codecs
name: {zh: "IQ 解码公共发射器", en: "IQ Decode Emitters"}
description:
  zh: >
      _iq23_value 与 _dot_fma4：把查表值按位移拼成码值、以及 4 元素一组的 FMA 累加发射器，供 IQ2_S/IQ3_S/IQ4_XS 复用。
      
  en: >
      _iq23_value and _dot_fma4: assemble table values into codes by shifting and emit 4-element FMA accumulation, reused by IQ2_S/IQ3_S/IQ4_XS.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.876Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 326
    end_line: 347
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:_iq23_value"
    description:
      zh: >
          IQ 码值拼装
          
      en: >
          Assembles IQ code values
          
---
