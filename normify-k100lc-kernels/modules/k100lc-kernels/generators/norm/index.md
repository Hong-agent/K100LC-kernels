---
uid: 31ba20b9
id: k100lc-kernels.generators.norm
parent: k100lc-kernels.generators
name: {zh: "归一化与激活生成器", en: "Normalization & Activation Generators"}
description:
  zh: >
      归一化/激活算子的生成器：LayerNorm、行 softmax、GELU 门控乘，以及访存合并的多线程激活量化。
      
  en: >
      Generators for normalization/activation operators: LayerNorm, row softmax, gated GELU multiply, and the coalesced multi-threaded activation quantizer.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.882Z"
fingerprint: 94d0611addd18f451a7d8efe01485b1b6d2df13605fa358d2ab80f47c45a8f2e
source:
  - path: "tools/gen_layernorm.py"
  - path: "tools/gen_softmax.py"
  - path: "tools/gen_gelu_mul.py"
  - path: "tools/gen_quant_fast.py"
---
