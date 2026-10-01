---
uid: 0b25b6fa
id: k100lc-kernels.generators.gemv.qdot.selftest.reference
parent: k100lc-kernels.generators.gemv.qdot.selftest
name: {zh: "参考点积与块加载", en: "Reference Dot & Block Loading"}
description:
  zh: >
      ref_dot 参考实现、_load_iq4nl_blocks/_synth_blocks/_load_blocks：从真权重文件或合成数据准备块，为各编码自检提供输入与期望值。
      
  en: >
      The ref_dot reference with _load_iq4nl_blocks/_synth_blocks/_load_blocks: prepares blocks from real weight files or synthetic data, providing inputs and expectations for each codec self-test.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.879Z"
fingerprint: cd2c28437a5fe91a91bbc48984f7d3475adfbea310f3533141877e2eefbba804
source:
  - path: "tools/gen_gemv_qdot.py"
    line: 1037
    end_line: 1176
apis:
  - protocol: file
    path: "tools/gen_gemv_qdot.py:ref_dot"
    description:
      zh: >
          参考点积实现
          
      en: >
          Reference dot-product implementation
          
---
