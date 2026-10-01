---
uid: ae5db052
id: k100lc-kernels.generators.gdn.split-norm
parent: k100lc-kernels.generators.gdn
name: {zh: "QKV 切分+归一化融合", en: "Fused QKV Split + Norm"}
description:
  zh: >
      gen_gdn_split_norm.py：生成 k_gdn_split_norm_k，一次完成 QKV 切分与 q/k 的 L2 归一化。
      
  en: >
      gen_gdn_split_norm.py: generates k_gdn_split_norm_k, doing QKV split and q/k L2 normalization in one pass.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.883Z"
fingerprint: e03966fd6e008468f81f0ddf9be22709d8ea197c68a36b33eba3b46d29e004d3
source:
  - path: "tools/gen_gdn_split_norm.py"
    line: 38
    end_line: 177
apis:
  - protocol: file
    path: "tools/gen_gdn_split_norm.py:gen_asm"
    description:
      zh: >
          生成 k_gdn_split_norm_k
          
      en: >
          Generates k_gdn_split_norm_k
          
deps:
  - kind: dataflow
    to: k100lc-kernels.kernels.asm.new.ssm
    from_api: "file:tools/gen_gdn_split_norm.py:gen_asm"
    label: {zh: "生成融合小算子", en: "Generates the fused operator"}
---
