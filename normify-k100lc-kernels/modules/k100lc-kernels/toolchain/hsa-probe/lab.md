---
uid: ea58b869
id: k100lc-kernels.toolchain.hsa-probe.lab
parent: k100lc-kernels.toolchain.hsa-probe
name: {zh: "新内核实验台", en: "New-Kernel Lab"}
description:
  zh: >
      kernel_lab.py：build_one（.s → 单核 HSACO）与 run_one（上卡执行并回读）两步组成的快速迭代环路。
      
  en: >
      kernel_lab.py: the fast iteration loop made of build_one (.s to single-kernel HSACO) and run_one (run on device and read back).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.892Z"
fingerprint: af92dfe8ab7f195e892cfc9acf2a9220fad5f5b7616f43762f9d8348936b50a6
source:
  - path: "tools/kernel_lab.py"
apis:
  - protocol: file
    path: "tools/kernel_lab.py:build_one"
    description:
      zh: >
          汇编单核 HSACO
          
      en: >
          Assembles a single-kernel HSACO
          
  - protocol: file
    path: "tools/kernel_lab.py:run_one"
    description:
      zh: >
          上卡执行并回读
          
      en: >
          Runs on device and reads back
          
deps:
  - kind: call
    to: k100lc-kernels.asm.cli
    from_api: "file:tools/kernel_lab.py:build_one"
    to_api: "file:asm.py:assemble"
    label: {zh: "汇编 .s 源", en: "Assembles .s sources"}
  - kind: call
    to: k100lc-kernels.hsaco.from-scratch
    from_api: "file:tools/kernel_lab.py:build_one"
    to_api: "file:tools/make_hsaco_from_scratch.py"
    label: {zh: "生成最小 HSACO", en: "Generates a minimal HSACO"}
---
