---
uid: 2cb9ffdd
id: k100lc-kernels.toolchain.hsa-probe
parent: k100lc-kernels.toolchain
name: {zh: "上卡探针与实验台", en: "On-Device Probes & Lab"}
description:
  zh: >
      不依赖 DTK 的内核执行脚手架：为单个内核生成 C host 程序并运行、在同一 HSA 程序里连续跑多内核保留缓冲，以及新内核「写 .s → 汇编 → 单核 HSACO → 上卡 → 回读」的一键环路。
      
  en: >
      DTK-free kernel execution scaffolding: generate and run a C host program for one kernel, run several kernels in one HSA program with buffers preserved, and the one-shot write-assemble-run-read loop for new kernels.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.892Z"
fingerprint: 6e3ec73c540411f1297fbcec0f0a06e068e5ba25cb6b4539bfefc507ec5154d9
source:
  - path: "tools/hsa_job.py"
  - path: "tools/hsa_seq.py"
  - path: "tools/kernel_lab.py"
---
