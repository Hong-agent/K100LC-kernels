---
uid: 0001f46b
id: k100lc-kernels.toolchain.weights.int4-manifest
parent: k100lc-kernels.toolchain.weights
name: {zh: "INT4 权重清单", en: "INT4 Weight Manifest"}
description:
  zh: >
      int4_engine_manifest.py：为「引擎直跑原生 INT4」生成 TSV 清单，逐层给出 weight_packed/weight_scale 在 checkpoint 里的绝对字节偏移。
      
  en: >
      int4_engine_manifest.py: emits a TSV manifest for running native INT4 directly from the engine, giving the absolute byte offsets of each layer’s weight_packed/weight_scale pair.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.893Z"
fingerprint: 4f1281ea970d06e2344d2f7ffa1c04cc0008656ea6b33d1f8957f6d0c23ca428
source:
  - path: "tools/int4_engine_manifest.py"
apis:
  - protocol: file
    path: "tools/int4_engine_manifest.py"
    description:
      zh: >
          生成 INT4 权重偏移清单
          
      en: >
          Emits the INT4 weight-offset manifest
          
deps:
  - kind: call
    to: k100lc-kernels.toolchain.weights.ct-int4
    from_api: "file:tools/int4_engine_manifest.py"
    to_api: "file:tools/ct_int4.py"
    label: {zh: "枚举量化层", en: "Enumerates the quantized"}
---
