---
uid: b1c4b42f
id: k100lc-kernels.generators.gguf.iq3xxs
parent: k100lc-kernels.generators.gguf
name: {zh: "IQ3_XXS 解码生成器", en: "IQ3_XXS Dequant Generator"}
description:
  zh: >
      gen_iq3xxs_dequant.py：256 元素 / 98 字节块，u32 尺度数组 + 256×u32 网格 + 7bit 符号表。
      
  en: >
      gen_iq3xxs_dequant.py: 256 elements per 98-byte block with a u32 scale array, a 256×u32 grid and a 7-bit sign table.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.881Z"
fingerprint: fd95f58c527635d3a506c17d093902c72dc7fa11017b0ced7f74944e0e107079
source:
  - path: "tools/gen_iq3xxs_dequant.py"
apis:
  - protocol: file
    path: "tools/gen_iq3xxs_dequant.py"
    description:
      zh: >
          生成 iq3xxs_dequant_k
          
      en: >
          Generates iq3xxs_dequant_k
          
---
