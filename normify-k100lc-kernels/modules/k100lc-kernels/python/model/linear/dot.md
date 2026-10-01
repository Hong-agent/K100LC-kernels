---
uid: 6e1fc3da
id: k100lc-kernels.python.model.linear.dot
parent: k100lc-kernels.python.model.linear
name: {zh: "量化点积线性层", en: "Decode-and-Dot Linear"}
description:
  zh: >
      DotLinear：任意 `*_dot_k` 编码的线性层，一次启动写 partial、再一次 reduce_blocks_k 归约成 y；支持专家行切片与 out_dev 直写。
      
  en: >
      DotLinear: a linear layer over any `*_dot_k` codec, writing partials in one launch and reducing with reduce_blocks_k into y; supports expert row slicing and direct out_dev writes.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.886Z"
fingerprint: 7f78e9c51e0530a21223674f1f32aac9e9e82a6a467fb358c04e6c3cee1a2dc1
source:
  - path: "python/k100lc_kernels/model.py"
    line: 150
    end_line: 256
apis:
  - protocol: file
    path: "k100lc_kernels.model.DotLinear.forward_device"
    description:
      zh: >
          设备侧前向（解码+点积+归约）
          
      en: >
          Device-side forward (decode, dot, reduce)
          
deps:
  - kind: call
    to: k100lc-kernels.generators.gemv.qdot.reduce
    from_api: "file:k100lc_kernels.model.DotLinear.forward_device"
    to_api: "file:tools/gen_gemv_qdot.py:gen_reduce_asm"
    label: {zh: "partial 行归约", en: "Reduces partials per row"}
  - kind: call
    to: k100lc-kernels.generators.int4.dot
    from_api: "file:k100lc_kernels.model.DotLinear.forward_device"
    to_api: "file:tools/gen_int4_dot.py:gen_int4_dot_asm"
    label: {zh: "INT4 编码点积", en: "INT4 decode-and-dot"}
---
