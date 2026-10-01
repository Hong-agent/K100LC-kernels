---
uid: 0e1f3a90
id: k100lc-kernels.compiler.tests.dynamic
parent: k100lc-kernels.compiler.tests
name: {zh: "动态启动测试", en: "Dynamic Launch Test"}
description:
  zh: >
      test_dynamic.py：编译一个内核并用 launch_dyn 走动态 kernarg/网格路径，验证编译期未知布局下的启动。
      
  en: >
      test_dynamic.py: compiles a kernel and exercises the dynamic kernarg/grid path through launch_dyn, validating launches with a compile-time-unknown layout.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.890Z"
fingerprint: 0afd6e9619fd22ec5088fecc0c1e8aab80d4ac554e909e28bd8fe68a0c6e15de
source:
  - path: "compiler/tests/test_dynamic.py"
apis:
  - protocol: file
    path: "compiler/tests/test_dynamic.py:main"
    description:
      zh: >
          动态内核启动测试
          
      en: >
          Dynamic kernel launch test
          
deps:
  - kind: call
    to: k100lc-kernels.python.runtime.launch
    from_api: "file:compiler/tests/test_dynamic.py:main"
    to_api: "file:k100lc_kernels.Runtime.launch_dyn"
    label: {zh: "走动态启动路径", en: "Uses the dynamic launch path"}
---
