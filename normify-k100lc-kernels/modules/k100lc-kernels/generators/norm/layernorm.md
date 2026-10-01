---
uid: 2e54340e
id: k100lc-kernels.generators.norm.layernorm
parent: k100lc-kernels.generators.norm
name: {zh: "LayerNorm 生成器", en: "LayerNorm Generator"}
description:
  zh: >
      gen_layernorm.py：warp-per-row + LDS 归约的 LayerNorm（layernorm_k），要求 cols%64==0。
      
  en: >
      gen_layernorm.py: warp-per-row LayerNorm with LDS reduction (layernorm_k), requiring cols%64==0.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.882Z"
fingerprint: 80979d77c0b63b64a848f6d20e02a5af379a2d2d3afe0f9fc3b2eeacd532ead2
source:
  - path: "tools/gen_layernorm.py"
apis:
  - protocol: file
    path: "tools/gen_layernorm.py"
    description:
      zh: >
          生成 layernorm_k
          
      en: >
          Generates layernorm_k
          
---
