---
uid: b3097f77
id: k100lc-kernels.toolchain.hsa-probe.sequence
parent: k100lc-kernels.toolchain.hsa-probe
name: {zh: "多核连续执行器", en: "Multi-Kernel Sequence Runner"}
description:
  zh: >
      hsa_seq.py：在同一个 HSA 程序里连续跑多个内核、缓冲跨内核保留，是「解码 → scratch → GEMV」这类多内核数据流的底座。
      
  en: >
      hsa_seq.py: runs several kernels inside one HSA program with buffers surviving across kernels — the base for multi-kernel flows such as decode → scratch → GEMV.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.892Z"
fingerprint: 5684e371c15fa416f807b10cf43c00e011b08ede5b81de1392bae2922239dd48
source:
  - path: "tools/hsa_seq.py"
apis:
  - protocol: file
    path: "tools/hsa_seq.py:run_seq"
    description:
      zh: >
          同一程序内连续跑多内核
          
      en: >
          Runs multiple kernels in one program
          
deps:
  - kind: call
    to: k100lc-kernels.toolchain.hsa-probe.job
    from_api: "file:tools/hsa_seq.py:run_seq"
    to_api: "file:tools/hsa_job.py"
    label: {zh: "复用单核 host 生成", en: "Reuses the single-kernel host"}
---
