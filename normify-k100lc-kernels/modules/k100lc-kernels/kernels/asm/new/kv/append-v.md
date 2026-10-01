---
uid: b9382950
id: k100lc-kernels.kernels.asm.new.kv.append-v
parent: k100lc-kernels.kernels.asm.new.kv
name: {zh: "V 追加", en: "V Append"}
description:
  zh: >
      kv_append_v_k：把 V 量化打包写入 KV cache 的 V 区。
      
  en: >
      kv_append_v_k: quantizes and packs V into the V region of the KV cache.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.870Z"
fingerprint: d17450239df45dbf2aba214c0272c387ed1019f14d41ce5d4efb9f49e522b345
source:
  - path: "kernels/asm/k_new/015__Z13kv_append_v_kPjPfS0_PKfiiiiii.s"
apis:
  - protocol: rpc
    path: "kv_append_v_k"
    description:
      zh: >
          KV cache 追加 V
          
      en: >
          Append V to the KV cache
          
---
