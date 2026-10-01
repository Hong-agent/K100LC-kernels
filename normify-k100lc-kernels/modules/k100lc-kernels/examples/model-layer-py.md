---
uid: 7c9fdb16
id: k100lc-kernels.examples.model-layer-py
parent: k100lc-kernels.examples
name: {zh: "模型级对账示例", en: "Model-Level Reconciliation Example"}
description:
  zh: >
      python_model_layer.py：合成权重跑 RMSNorm→MLP，并验证 MoE 合并与分桶两条路径，逐项与 NumPy 对账。
      
  en: >
      python_model_layer.py: runs RMSNorm to MLP on synthetic weights and validates both the dense and bucketed MoE paths, reconciling each against NumPy.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.895Z"
fingerprint: 73b679a8b26a11e1ef53fd87ce6524eb9d58b18638bdb4933f806bbafa229035
source:
  - path: "examples/python_model_layer.py"
apis:
  - protocol: file
    path: "examples/python_model_layer.py"
    description:
      zh: >
          模型级端到端对账
          
      en: >
          Model-level end-to-end reconciliation
          
deps:
  - kind: call
    to: k100lc-kernels.python.model.mlp
    from_api: "file:examples/python_model_layer.py"
    to_api: "file:k100lc_kernels.model.MLP.forward"
    label: {zh: "验证门控 MLP", en: "Validates the gated MLP"}
  - kind: call
    to: k100lc-kernels.python.model.moe
    from_api: "file:examples/python_model_layer.py"
    to_api: "file:k100lc_kernels.model.MoEExperts.route"
    label: {zh: "验证 MoE 两条路径", en: "Validates both MoE paths"}
---
