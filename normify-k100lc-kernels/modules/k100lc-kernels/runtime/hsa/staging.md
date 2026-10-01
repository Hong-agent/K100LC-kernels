---
uid: "19049731"
id: k100lc-kernels.runtime.hsa.staging
parent: k100lc-kernels.runtime.hsa
name: {zh: "拷贝暂存与 H2D/D2H", en: "Staging & H2D/D2H Copies"}
description:
  zh: >
      ensure_registered + copy_h2d/copy_d2h：用 pinned 暂存区把主机内存注册进 HSA，实现无 DTK 的双向拷贝。
      
  en: >
      ensure_registered plus copy_h2d/copy_d2h: registers host memory into HSA through the pinned staging area, enabling DTK-free bidirectional copies.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.883Z"
fingerprint: 12d13a092207d5da24b991721ce5aafd5fa3a94506ed44e0fa89cd4aaed25ea5
source:
  - path: "runtime/hsa_rt.cpp"
    line: 265
    end_line: 324
apis:
  - protocol: file
    path: "runtime/hsa_rt.cpp:copy_h2d"
    description:
      zh: >
          主机 → 设备拷贝
          
      en: >
          Host-to-device copy
          
---
