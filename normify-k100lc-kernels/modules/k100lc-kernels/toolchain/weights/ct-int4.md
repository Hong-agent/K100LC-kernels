---
uid: e237a545
id: k100lc-kernels.toolchain.weights.ct-int4
parent: k100lc-kernels.toolchain.weights
name: {zh: "INT4 参考解码与加载", en: "INT4 Reference Decoder & Loader"}
description:
  zh: >
      ct_int4.py：compressed-tensors INT4/pack-quantized（W4A16）的主机侧参考解码器与 checkpoint 加载器，枚举量化线性层并按 (N,K) 分组。
      
  en: >
      ct_int4.py: the host-side reference decoder and checkpoint loader for compressed-tensors INT4/pack-quantized (W4A16), enumerating quantized linear layers grouped by (N,K).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.893Z"
fingerprint: 60482534afd8464d17a9a2613318f3e88317e9111db174571fae0263e1010a71
source:
  - path: "tools/ct_int4.py"
apis:
  - protocol: file
    path: "tools/ct_int4.py"
    description:
      zh: >
          INT4 checkpoint 读取与参考解码
          
      en: >
          INT4 checkpoint loading and reference decode
          
deps:
  - kind: reference
    to: k100lc-kernels.generators.int4.dot
    from_api: "file:tools/ct_int4.py"
    to_api: "file:tools/gen_int4_dot.py:gen_int4_dot_asm"
    label: {zh: "作为 INT4 内核参考", en: "Reference for the INT4 kernels"}
---
