---
uid: 8f6f7e86
id: k100lc-kernels.python.model.mlp
parent: k100lc-kernels.python.model
name: {zh: "门控 MLP", en: "Gated MLP"}
description:
  zh: >
      SwiGLU 与 MLP：gate/up 两条线性层 + silu_mul_k 门控 + down 投影，多次 kernel 连续入队只 sync 一次。
      
  en: >
      SwiGLU and MLP: gate/up linear layers, the silu_mul_k gate and the down projection, queued back to back with a single sync.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.886Z"
fingerprint: 7f78e9c51e0530a21223674f1f32aac9e9e82a6a467fb358c04e6c3cee1a2dc1
source:
  - path: "python/k100lc_kernels/model.py"
    line: 459
    end_line: 513
apis:
  - protocol: file
    path: "k100lc_kernels.model.MLP.forward"
    description:
      zh: >
          MLP 前向
          
      en: >
          MLP forward
          
deps:
  - kind: call
    to: k100lc-kernels.kernels.asm.new.elementwise.silu-mul
    from_api: "file:k100lc_kernels.model.MLP.forward"
    to_api: "rpc:silu_mul_k"
    label: {zh: "SwiGLU 门控", en: "SwiGLU gating"}
---
