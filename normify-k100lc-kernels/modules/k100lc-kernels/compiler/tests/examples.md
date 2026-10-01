---
uid: 0e004e6e
id: k100lc-kernels.compiler.tests.examples
parent: k100lc-kernels.compiler.tests
name: {zh: "示例对账测试", en: "Example Reconciliation Test"}
description:
  zh: >
      test_examples.py：逐例编译 vadd/silu/axpy/scaleloop 并核对结果，覆盖 if 与 for 降级。
      
  en: >
      test_examples.py: compiles vadd/silu/axpy/scaleloop case by case and checks the results, covering if and for lowering.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.890Z"
fingerprint: fbbcd1b41529a57e7bd16970f633bce66750ea6ddf00f777648dfbb1f40dfc56
source:
  - path: "compiler/tests/test_examples.py"
apis:
  - protocol: file
    path: "compiler/tests/test_examples.py:main"
    description:
      zh: >
          示例内核正确性测试
          
      en: >
          Correctness test for the example kernels
          
deps:
  - kind: call
    to: k100lc-kernels.compiler.core.api
    from_api: "file:compiler/tests/test_examples.py:main"
    to_api: "file:k100lc_compiler.compile_file"
    label: {zh: "编译示例内核", en: "Compiles the example kernels"}
---
