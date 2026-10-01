---
uid: "56856839"
id: k100lc-kernels.kernels.asm.vision.bias
parent: k100lc-kernels.kernels.asm.vision
name: {zh: "ViT Bias 加", en: "ViT Bias Add"}
description:
  zh: >
      vit_bias_kernel：逐行加 bias 广播。
      
  en: >
      vit_bias_kernel: row-wise bias broadcast add.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.873Z"
fingerprint: 23df657a4364c8383e43795c2bc10aee53eedb06e492d4f3ee12911e8d16ea72
source:
  - path: "kernels/asm/k_vision/004__Z15vit_bias_kernelPfPKfii.s"
apis:
  - protocol: rpc
    path: "vit_bias_kernel"
    description:
      zh: >
          ViT bias 广播加
          
      en: >
          ViT bias broadcast add
          
---
