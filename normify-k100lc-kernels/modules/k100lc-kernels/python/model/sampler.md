---
uid: 9303242e
id: k100lc-kernels.python.model.sampler
parent: k100lc-kernels.python.model
name: {zh: "采样算子", en: "Sampler Operator"}
description:
  zh: >
      Sampler：贪心采样路径（argmax_part_k → argmax_comb_k），覆盖大词表的跨块归约。
      
  en: >
      Sampler: the greedy sampling path (argmax_part_k to argmax_comb_k), covering cross-chunk reduction over large vocabularies.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.887Z"
fingerprint: 7f78e9c51e0530a21223674f1f32aac9e9e82a6a467fb358c04e6c3cee1a2dc1
source:
  - path: "python/k100lc_kernels/model.py"
    line: 697
    end_line: 750
apis:
  - protocol: file
    path: "k100lc_kernels.model.Sampler.argmax_device"
    description:
      zh: >
          设备侧 argmax 采样
          
      en: >
          Device-side argmax sampling
          
deps:
  - kind: call
    to: k100lc-kernels.kernels.asm.new.reduction.argmax-part
    from_api: "file:k100lc_kernels.model.Sampler.argmax_device"
    to_api: "rpc:argmax_part_k"
    label: {zh: "分块 argmax", en: "Chunked argmax"}
  - kind: call
    to: k100lc-kernels.kernels.asm.new.reduction.argmax-comb
    from_api: "file:k100lc_kernels.model.Sampler.argmax_device"
    to_api: "rpc:argmax_comb_k"
    label: {zh: "argmax 合并", en: "Combines argmax partials"}
---
