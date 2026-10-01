---
uid: dc99633a
id: k100lc-kernels.hsaco.multi-builder
parent: k100lc-kernels.hsaco
name: {zh: "多内核 HSACO 生成", en: "Multi-Kernel HSACO Builder"}
description:
  zh: >
      把多个内核的机器码与符号放进同一个 code object，按 kernel_spec.json 的 kernarg/SGPR/VGPR/LDS 生成每个 kernel descriptor。
      
  en: >
      Packs multiple kernels’ machine code and symbols into one code object, emitting a kernel descriptor per kernel from kernel_spec.json (kernarg/SGPR/VGPR/LDS).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.857Z"
fingerprint: f8c684412f0554b5bf4cfad130196a95065c12b57a6f39518d0e86f1dfbb592a
source:
  - path: "tools/make_hsaco_multi.py"
apis:
  - protocol: file
    path: "tools/make_hsaco_multi.py"
    description:
      zh: >
          多内核 → 单个 HSACO
          
      en: >
          Multiple kernels to one HSACO
          
deps:
  - kind: call
    to: k100lc-kernels.hsaco.full-builder
    from_api: "file:tools/make_hsaco_multi.py"
    to_api: "file:tools/make_hsaco_full.py"
    label: {zh: "复用单核组装逻辑", en: "Reuses the single-kernel"}
  - kind: dataflow
    to: k100lc-kernels.prebuilt.base-hsaco
    from_api: "file:tools/make_hsaco_multi.py"
    to_api: "file:prebuilt/k100lc_base.hsaco"
    label: {zh: "产出基线 HSACO", en: "Produces the baseline HSACO"}
---
