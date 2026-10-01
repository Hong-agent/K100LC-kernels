---
uid: f7af4a01
id: k100lc-kernels.compiler.examples.runner
parent: k100lc-kernels.compiler.examples
name: {zh: "示例运行脚本", en: "Example Runner"}
description:
  zh: >
      run_vadd.py：编译 vadd.kkl、上卡执行并与 numpy 对账的最小端到端脚本。
      
  en: >
      run_vadd.py: the minimal end-to-end script that compiles vadd.kkl, runs it on device and reconciles against numpy.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.890Z"
fingerprint: 689cba248e9d62caac6735152d50319b67aa35f416fa1fe9793ed33be3c80018
source:
  - path: "compiler/examples/run_vadd.py"
apis:
  - protocol: file
    path: "compiler/examples/run_vadd.py"
    description:
      zh: >
          编译+上卡+对账
          
      en: >
          Compile, run on device, reconcile
          
deps:
  - kind: call
    to: k100lc-kernels.compiler.core.api
    from_api: "file:compiler/examples/run_vadd.py"
    to_api: "file:k100lc_compiler.compile_file"
    label: {zh: "编译并运行 vadd", en: "Compiles and runs vadd"}
---
