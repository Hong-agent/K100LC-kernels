---
uid: fbfe5007
id: k100lc-kernels.toolchain.catalog
parent: k100lc-kernels.toolchain
name: {zh: "目录与文档生成", en: "Catalog & Documentation Generation"}
description:
  zh: >
      从 HSACO metadata 生成运行时使用的 catalog.json，并把它展开成可读的逐内核参数表 docs/KERNELS.md，避免文档与预编译产物不一致。
      
  en: >
      Generates the runtime catalog.json from HSACO metadata and expands it into the readable per-kernel parameter table docs/KERNELS.md, keeping docs consistent with the prebuilt artifacts.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.892Z"
fingerprint: f4d584c0bb097fa17c35cf4bbb520955fdbdc23887974760920952fa9cb91dd2
source:
  - path: "tools/make_catalog.py"
  - path: "tools/gen_kernel_docs.py"
---
