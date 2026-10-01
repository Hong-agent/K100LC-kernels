---
uid: 7b49d033
id: k100lc-kernels.generators.gguf
parent: k100lc-kernels.generators
name: {zh: "GGUF 编码生成器", en: "GGUF Codec Generators"}
description:
  zh: >
      GGUF/llama.cpp 各量化编码的独立解码（与部分点积）内核生成器，每个都是单编码、单内核、带真权重自检的脚本。
      
  en: >
      Standalone dequant (and some dot) kernel generators for the GGUF/llama.cpp quantization codecs — one codec, one kernel, one script with a real-weight self-test each.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.880Z"
fingerprint: b51c453504caf7b2165148471292629613e393ca238a447b63c6c976cdf0cdb7
source:
  - path: "tools/gen_q2_0_dequant.py"
  - path: "tools/gen_q4_0_dequant.py"
  - path: "tools/gen_q8_0_dequant.py"
  - path: "tools/gen_iq2s_dequant.py"
  - path: "tools/gen_iq3s_dequant.py"
  - path: "tools/gen_iq3xxs_dequant.py"
  - path: "tools/gen_iq4nl_dequant.py"
  - path: "tools/gen_iq4nl_gemv.py"
  - path: "tools/gen_iq4nl_to_i8.py"
  - path: "tools/gen_iq4xs_dequant.py"
---
