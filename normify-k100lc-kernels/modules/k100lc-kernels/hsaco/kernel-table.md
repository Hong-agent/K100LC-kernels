---
uid: 05667b22
id: k100lc-kernels.hsaco.kernel-table
parent: k100lc-kernels.hsaco
name: {zh: "内核表头生成", en: "Kernel Table Header Generation"}
description:
  zh: >
      从自研 HSACO 生成 runtime 侧的内核表头 nodtk_kernels.h：名字、每个显式参数的 (offset,size)、kernarg/group/private 段大小。
      
  en: >
      Generates the runtime-side kernel table header nodtk_kernels.h from our HSACO: names, per-argument (offset,size), and kernarg/group/private sizes.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.857Z"
fingerprint: d4959945d7c4acdf1ab27bf294be4ed82d245767b2d30c7b6b693a93ce78997f
source:
  - path: "tools/gen_kernel_table.py"
apis:
  - protocol: file
    path: "tools/gen_kernel_table.py"
    description:
      zh: >
          HSACO → nodtk_kernels.h
          
      en: >
          HSACO to nodtk_kernels.h
          
deps:
  - kind: call
    to: k100lc-kernels.hsaco.metadata-read
    from_api: "file:tools/gen_kernel_table.py"
    to_api: "file:tools/amdgpu_meta.py"
    label: {zh: "读 AMDGPU 元数据", en: "Reads AMDGPU metadata"}
  - kind: reference
    to: k100lc-kernels.hsaco.symbol-names
    from_api: "file:tools/gen_kernel_table.py"
    to_api: "file:tools/hsaco_names.py"
    label: {zh: "复用符号/隐藏参数定义", en: "Reuses symbol/hidden-arg"}
  - kind: dataflow
    to: k100lc-kernels.prebuilt.kernel-header
    from_api: "file:tools/gen_kernel_table.py"
    to_api: "file:prebuilt/nodtk_kernels.h"
    label: {zh: "生成内核表头", en: "Generates the kernel table"}
---
