---
uid: 36d587a0
id: k100lc-kernels.kernels.asm.new
parent: k100lc-kernels.kernels.asm
name: {zh: "Transformer 与序列算子内核", en: "Transformer & Sequence Operator Kernels"}
description:
  zh: >
      模型解码/预填充需要的通用算子：逐元素、归一化、归约、激活量化、位置编码、KV 写入、SSM/卷积、head 重排与 GGUF K-量化解码。
      
  en: >
      General operators for model decode/prefill: element-wise, normalization, reduction, activation quantization, positional encoding, KV append, SSM/convolution, head reshuffling and GGUF K-quant decoding.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.867Z"
fingerprint: 7ac6d35d7382976fa039b93ccc4211a5fe3bcd360cf264844b78abc22fc78a74
source:
  - path: "kernels/asm/k_new/001__Z6fill_kPffx.s"
  - path: "kernels/asm/k_new/019__Z5gdn_kPfPKfS1_S1_S1_S1_S_iiiiii.s"
---
