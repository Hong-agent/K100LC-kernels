---
uid: 878d7be5
id: k100lc-kernels.kernels.asm.new.kv.append-k
parent: k100lc-kernels.kernels.asm.new.kv
name: {zh: "K 追加", en: "K Append"}
description:
  zh: >
      kv_append_k_k：把 K 量化打包写入 KV cache 的 K 区。
      
  en: >
      kv_append_k_k: quantizes and packs K into the K region of the KV cache.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.870Z"
fingerprint: 151a8c8cabc9f6f8d4ea447dd3e1df76b0549932c1b620ea13dddcbc61301f1d
source:
  - path: "kernels/asm/k_new/016__Z13kv_append_k_kPjPfPKfiiiiii.s"
apis:
  - protocol: rpc
    path: "kv_append_k_k"
    description:
      zh: >
          KV cache 追加 K
          
      en: >
          Append K to the KV cache
          
---
