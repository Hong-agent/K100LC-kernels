---
uid: 34ccc604
id: k100lc-kernels.toolchain.hsa-probe.job
parent: k100lc-kernels.toolchain.hsa-probe
name: {zh: "单核执行器", en: "Single-Kernel Runner"}
description:
  zh: >
      hsa_job.py：按内核元数据生成一个只依赖 driver 侧 HSA 的最小 host 程序，编译运行后回读结果缓冲。
      
  en: >
      hsa_job.py: generates a minimal host program depending only on the driver-side HSA runtime from kernel metadata, builds and runs it, then reads back the result buffers.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.892Z"
fingerprint: c8552118e51150017225dc6d703d0f0fe2652e25796326c7907aef6d99e9b2f7
source:
  - path: "tools/hsa_job.py"
apis:
  - protocol: file
    path: "tools/hsa_job.py"
    description:
      zh: >
          生成并运行单核 host 程序
          
      en: >
          Generates and runs a single-kernel host program
          
deps:
  - kind: call
    to: k100lc-kernels.hsaco.full-builder
    from_api: "file:tools/hsa_job.py"
    to_api: "file:tools/make_hsaco_full.py"
    label: {zh: "拼单核 HSACO", en: "Builds a single-kernel HSACO"}
---
