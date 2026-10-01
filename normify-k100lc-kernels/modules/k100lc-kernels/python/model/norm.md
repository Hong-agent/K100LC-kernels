---
uid: c18e6d35
id: k100lc-kernels.python.model.norm
parent: k100lc-kernels.python.model
name: {zh: "RMSNorm 算子", en: "RMSNorm Operator"}
description:
  zh: >
      RMSNorm：调用 rmsnorm_k 的行归一化封装（W 与 1+W 两种模式）。
      
  en: >
      RMSNorm: a row-normalization wrapper over rmsnorm_k (both W and 1+W modes).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.886Z"
fingerprint: 7f78e9c51e0530a21223674f1f32aac9e9e82a6a467fb358c04e6c3cee1a2dc1
source:
  - path: "python/k100lc_kernels/model.py"
    line: 425
    end_line: 458
apis:
  - protocol: file
    path: "k100lc_kernels.model.RMSNorm.forward"
    description:
      zh: >
          RMSNorm 前向
          
      en: >
          RMSNorm forward
          
deps:
  - kind: call
    to: k100lc-kernels.kernels.asm.new.norm.rmsnorm
    from_api: "file:k100lc_kernels.model.RMSNorm.forward"
    to_api: "rpc:rmsnorm_k"
    label: {zh: "调用 rmsnorm_k", en: "Calls rmsnorm_k"}
---
