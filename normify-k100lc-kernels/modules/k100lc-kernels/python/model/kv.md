---
uid: 4acfae18
id: k100lc-kernels.python.model.kv
parent: k100lc-kernels.python.model
name: {zh: "KV Cache 算子", en: "KV Cache Operator"}
description:
  zh: >
      KVCache：预分配 KV 缓冲并按 token 追加（append_device/append/reset），底层用 kv_append_k_k / kv_append_v_k。
      
  en: >
      KVCache: pre-allocates the KV buffers and appends per token (append_device/append/reset) through kv_append_k_k / kv_append_v_k.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.887Z"
fingerprint: 7f78e9c51e0530a21223674f1f32aac9e9e82a6a467fb358c04e6c3cee1a2dc1
source:
  - path: "python/k100lc_kernels/model.py"
    line: 655
    end_line: 696
apis:
  - protocol: file
    path: "k100lc_kernels.model.KVCache.append_device"
    description:
      zh: >
          追加 KV（设备侧）
          
      en: >
          Appends KV on device
          
deps:
  - kind: call
    to: k100lc-kernels.kernels.asm.new.kv.append-k
    from_api: "file:k100lc_kernels.model.KVCache.append_device"
    to_api: "rpc:kv_append_k_k"
    label: {zh: "追加 K", en: "Appends K"}
  - kind: call
    to: k100lc-kernels.kernels.asm.new.kv.append-v
    from_api: "file:k100lc_kernels.model.KVCache.append_device"
    to_api: "rpc:kv_append_v_k"
    label: {zh: "追加 V", en: "Appends V"}
  - kind: reference
    to: k100lc-kernels.kernels.headers.kv-pack
    from_api: "file:k100lc_kernels.model.KVCache.append_device"
    to_api: "file:kernels/kv_pack.h"
    label: {zh: "KV 打包位宽约定", en: "KV packing bit-width"}
---
