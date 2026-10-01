---
uid: "714e9653"
id: k100lc-kernels.examples.w4a4-w4a8-gemv-py
parent: k100lc-kernels.examples
name: {zh: "W4A8/W4A4 GEMV 示例", en: "W4A8/W4A4 GEMV Example"}
description:
  zh: >
      python_w4a4_w4a8_gemv.py：真权重或合成权重下实跑 W4A8（v_dot4）与 W4A4（v_dot8）两条最强解码通路。
      
  en: >
      python_w4a4_w4a8_gemv.py: runs the two strongest decode paths, W4A8 (v_dot4) and W4A4 (v_dot8), on real or synthetic weights.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.895Z"
fingerprint: 8215ffb40dd4e445f78c2326461c98e0a06cc75ad780774a14b76dd5ac41c930
source:
  - path: "examples/python_w4a4_w4a8_gemv.py"
apis:
  - protocol: file
    path: "examples/python_w4a4_w4a8_gemv.py"
    description:
      zh: >
          W4A8 / W4A4 GEMV 实测
          
      en: >
          Benchmarks W4A8 / W4A4 GEMV
          
deps:
  - kind: call
    to: k100lc-kernels.python.rt4.gemv
    from_api: "file:examples/python_w4a4_w4a8_gemv.py"
    to_api: "file:k100lc_kernels.rt4.W4Runner.gemv"
    label: {zh: "跑 W4A8/W4A4 GEMV", en: "Runs W4A8/W4A4 GEMV"}
---
