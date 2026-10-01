---
uid: "144967e4"
id: k100lc-kernels.python.model
parent: k100lc-kernels.python
name: {zh: "模型级算子", en: "Model-Level Operators"}
description:
  zh: >
      model.py：把内核组合成可直接使用的推理算子（四种线性层、RMSNorm、SwiGLU/MLP、MoE、KV cache、采样），目标是权重只上传一次、工作缓冲复用、每次 forward 只 sync 一次。
      
  en: >
      model.py: composes kernels into ready-to-use inference operators (four linear-layer kinds, RMSNorm, SwiGLU/MLP, MoE, KV cache, sampling) aiming at upload-once weights, reused workspaces and a single sync per forward.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.886Z"
fingerprint: 7f78e9c51e0530a21223674f1f32aac9e9e82a6a467fb358c04e6c3cee1a2dc1
source:
  - path: "python/k100lc_kernels/model.py"
---
