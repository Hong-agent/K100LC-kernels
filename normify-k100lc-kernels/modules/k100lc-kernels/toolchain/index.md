---
uid: 96d2fcb0
id: k100lc-kernels.toolchain
parent: k100lc-kernels
name: {zh: "构建与验证工具链", en: "Build & Verification Toolchain"}
description:
  zh: >
      围绕内核包的工程脚本：全量构建、内核目录与文档生成、单核上卡探针、GGUF/INT4 权重参考解码与清单，以及各路径性能基准。
      
  en: >
      Engineering scripts around the kernel package: full builds, kernel catalog and doc generation, single-kernel on-device probes, GGUF/INT4 reference decoders and manifests, plus performance benchmarks for each path.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.891Z"
fingerprint: 327afdc996da61e65e958c100aab71d308807927fb15fd8e557ba77031bcf559
source:
  - path: "tools/build_all.sh"
  - path: "tools/build_kernels.py"
  - path: "tools/build_native_kernels.py"
  - path: "tools/make_catalog.py"
  - path: "tools/gen_kernel_docs.py"
  - path: "tools/hsa_job.py"
  - path: "tools/hsa_seq.py"
  - path: "tools/kernel_lab.py"
  - path: "tools/gguf_sample.py"
  - path: "tools/iq_tables.py"
  - path: "tools/iq_dequant.py"
  - path: "tools/ct_int4.py"
  - path: "tools/int4_engine_manifest.py"
  - path: "tools/bench_int4_layers.py"
  - path: "tools/bench_model_paths.py"
  - path: "tools/bench_moe.py"
---
