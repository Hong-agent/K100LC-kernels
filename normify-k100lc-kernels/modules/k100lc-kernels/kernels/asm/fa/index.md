---
uid: ff077060
id: k100lc-kernels.kernels.asm.fa
parent: k100lc-kernels.kernels.asm
name: {zh: "FlashAttention 内核", en: "FlashAttention Kernels"}
description:
  zh: >
      int4 QK/PV 分块注意力与解码期（M=1 / 多行）注意力，含行式与合并式两个变体；LDS 占用高、参数最多。
      
  en: >
      Int4 QK/PV tiled attention plus decode-time attention (M=1 and multi-row), with row-wise and combined variants; the highest LDS usage and argument count.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.860Z"
fingerprint: 7536f7c994b05ff55fbcf84e4f7be3792a827e0b91b9c646286b72fee82d8627
source:
  - path: "kernels/asm/k_fa/001__Z7fa_int4PKjPKfS0_S2_S0_S2_Pfiiiiii.s"
  - path: "kernels/asm/k_fa/003__Z16fa_decode_rows_kPfS_S_PKjPKfS1_S3_S1_S3_S3_miiiiiiiiiimmi.s"
---
