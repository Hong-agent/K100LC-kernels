---
uid: c378e2c8
id: k100lc-kernels.generators.int4.selftest
parent: k100lc-kernels.generators.int4
name: {zh: "INT4 上卡自检", en: "INT4 On-Device Self-Test"}
description:
  zh: >
      build_int4_hsaco 把生成的内核打成单核 HSACO，再用合成权重与真 checkpoint 逐行对账（selftest_synthetic/_selftest_dot_case/selftest_real）。
      
  en: >
      build_int4_hsaco packs the generated kernels into a single-kernel HSACO, then reconciles row by row against synthetic weights and a real checkpoint (selftest_synthetic/_selftest_dot_case/selftest_real).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.880Z"
fingerprint: 45593fc236bc9c4fdcc774bbad579c2177415e810ce503801255c2a432b25722
source:
  - path: "tools/gen_int4_dot.py"
    line: 281
    end_line: 497
apis:
  - protocol: file
    path: "tools/gen_int4_dot.py:selftest_synthetic"
    description:
      zh: >
          合成权重自检
          
      en: >
          Synthetic-weight self-test
          
  - protocol: file
    path: "tools/gen_int4_dot.py:build_int4_hsaco"
    description:
      zh: >
          构建单核 HSACO 并上卡
          
      en: >
          Builds a single-kernel HSACO and runs it
          
---
