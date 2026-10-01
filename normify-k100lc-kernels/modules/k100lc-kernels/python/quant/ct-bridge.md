---
uid: 803e9bbf
id: k100lc-kernels.python.quant.ct-bridge
parent: k100lc-kernels.python.quant
name: {zh: "compressed-tensors 转换", en: "compressed-tensors Bridge"}
description:
  zh: >
      ct_int4_to_twos_complement / ct_int4_to_rt4 / int4_scale_group_first_f32：把 checkpoint 的 offset-binary INT4 与 f16 尺度转成运行时需要的两补码/RT4 布局。
      
  en: >
      ct_int4_to_twos_complement / ct_int4_to_rt4 / int4_scale_group_first_f32: convert the checkpoint’s offset-binary INT4 and f16 scales into the two’s-complement/RT4 layout the runtime expects.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.887Z"
fingerprint: ba5c4b7622b8f6ca986027d789484d3c31a1e5ccab27f23977be9020da6357ea
source:
  - path: "python/k100lc_kernels/quant.py"
    line: 70
    end_line: 101
apis:
  - protocol: file
    path: "k100lc_kernels.quant.ct_int4_to_rt4"
    description:
      zh: >
          INT4 → RT4 布局转换
          
      en: >
          Converts INT4 to the RT4 layout
          
deps:
  - kind: dataflow
    to: k100lc-kernels.generators.int4.dot
    from_api: "file:k100lc_kernels.quant.ct_int4_to_rt4"
    to_api: "file:tools/gen_int4_dot.py:gen_int4_dot_asm"
    label: {zh: "转换到内核要求的布局", en: "Converts to the layout the"}
---
