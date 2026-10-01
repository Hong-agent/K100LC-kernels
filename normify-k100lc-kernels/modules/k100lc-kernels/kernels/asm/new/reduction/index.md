---
uid: b9ff3957
id: k100lc-kernels.kernels.asm.new.reduction
parent: k100lc-kernels.kernels.asm.new
name: {zh: "归约内核", en: "Reduction Kernels"}
description:
  zh: >
      argmax 的两阶段实现：分块部分 argmax（argmax_part_k）与跨块合并（argmax_comb_k），以及单行小规模 argmax_k。
      
  en: >
      Two-stage argmax: per-chunk partial argmax (argmax_part_k) and cross-chunk combine (argmax_comb_k), plus the single-row argmax_k for small inputs.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.869Z"
fingerprint: 61c255d5687ea9b747632bb156ee4336f140a04b8a95fb1a2f73512c099fb3ac
source:
  - path: "kernels/asm/k_new/008__Z8argmax_kPKfxPi.s"
  - path: "kernels/asm/k_new/009__Z13argmax_part_kPKfxiPfPi.s"
  - path: "kernels/asm/k_new/010__Z13argmax_comb_kPKfPKiiPi.s"
---
