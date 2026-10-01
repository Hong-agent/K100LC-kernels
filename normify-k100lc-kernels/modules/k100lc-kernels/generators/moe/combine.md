---
uid: "087755e9"
id: k100lc-kernels.generators.moe.combine
parent: k100lc-kernels.generators.moe
name: {zh: "MoE 稠密合并生成器", en: "MoE Dense Combine Generator"}
description:
  zh: >
      gen_moe_combine.py：y[r,:] = Σ_e weights[r,e]·exp_out[e,r,:] 的稠密专家输出加权合并（moe_combine_k）。
      
  en: >
      gen_moe_combine.py: dense weighted merge of expert outputs, y[r,:] = Σ_e weights[r,e]·exp_out[e,r,:] (moe_combine_k).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.882Z"
fingerprint: eb1c04f729cedcd83b70f30284c7c257f5df9e4769515f1759081462a5bb49ed
source:
  - path: "tools/gen_moe_combine.py"
apis:
  - protocol: file
    path: "tools/gen_moe_combine.py"
    description:
      zh: >
          生成 moe_combine_k
          
      en: >
          Generates moe_combine_k
          
---
