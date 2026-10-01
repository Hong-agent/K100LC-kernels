---
uid: 0b23f6b9
id: k100lc-kernels.kernels.asm.new.kv
parent: k100lc-kernels.kernels.asm.new
name: {zh: "KV Cache 写入", en: "KV Cache Append"}
description:
  zh: >
      kv_append_k_k / kv_append_v_k：把新 token 的 K/V 量化打包后追加进 KV cache（int8 或 int4 由 kv_pack.h 决定）。
      
  en: >
      kv_append_k_k / kv_append_v_k: quantize and pack the new tokens’ K/V and append them to the KV cache (int8 or int4 per kv_pack.h).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.870Z"
fingerprint: 040773b565520cd774732101cd9b63d328ddae5d844e788536238f763bb961c0
source:
  - path: "kernels/asm/k_new/015__Z13kv_append_v_kPjPfS0_PKfiiiiii.s"
  - path: "kernels/asm/k_new/016__Z13kv_append_k_kPjPfPKfiiiiii.s"
---
