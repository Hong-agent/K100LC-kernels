---
uid: 047a9c43
id: k100lc-kernels.python.model.workspace
parent: k100lc-kernels.python.model
name: {zh: "工作区与连续入队", en: "Workspace & Queued Sequences"}
description:
  zh: >
      Workspace 按 key 复用设备缓冲（避免逐层 alloc/free），run_sequence 把整段前向连续入队只 sync 一次，另有 cdiv/div_magic 数学辅助。
      
  en: >
      Workspace reuses device buffers by key (avoiding per-layer alloc/free); run_sequence queues a whole forward and syncs once; cdiv/div_magic provide math helpers.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.886Z"
fingerprint: 7f78e9c51e0530a21223674f1f32aac9e9e82a6a467fb358c04e6c3cee1a2dc1
source:
  - path: "python/k100lc_kernels/model.py"
    line: 47
    end_line: 149
apis:
  - protocol: file
    path: "k100lc_kernels.model.Workspace.buffer"
    description:
      zh: >
          按 key 复用缓冲
          
      en: >
          Reuses a buffer by key
          
  - protocol: file
    path: "k100lc_kernels.model.run_sequence"
    description:
      zh: >
          连续入队多次启动
          
      en: >
          Queues multiple launches back to back
          
deps:
  - kind: call
    to: k100lc-kernels.python.runtime.memory
    from_api: "file:k100lc_kernels.model.Workspace.buffer"
    to_api: "file:k100lc_kernels.Runtime.upload"
    label: {zh: "复用设备缓冲", en: "Reuses device buffers"}
---
