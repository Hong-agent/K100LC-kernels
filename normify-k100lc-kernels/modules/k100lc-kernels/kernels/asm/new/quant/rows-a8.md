---
uid: a1abee7a
id: k100lc-kernels.kernels.asm.new.quant.rows-a8
parent: k100lc-kernels.kernels.asm.new.quant
name: {zh: "行式 W4A8 量化", en: "Row-wise W4A8 Quantization"}
description:
  zh: >
      quant_rows_a8_k：输出 int8 码与两组尺度，供 W4A8 GEMV/GEMM 使用。
      
  en: >
      quant_rows_a8_k: emits int8 codes plus two sets of scales for the W4A8 GEMV/GEMM path.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.869Z"
fingerprint: d7b570a0df951cf789780e9ad08785b407bff075f302ab8e2beae9f06f0f0043
source:
  - path: "kernels/asm/k_new/012__Z15quant_rows_a8_kPjS_PfS0_PKfiiiii.s"
apis:
  - protocol: rpc
    path: "quant_rows_a8_k"
    description:
      zh: >
          激活 → int8（组 128）
          
      en: >
          Activations to int8, group 128
          
---
