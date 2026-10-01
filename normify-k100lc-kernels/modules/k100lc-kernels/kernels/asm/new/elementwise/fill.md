---
uid: aaa09186
id: k100lc-kernels.kernels.asm.new.elementwise.fill
parent: k100lc-kernels.kernels.asm.new.elementwise
name: {zh: "常量填充", en: "Fill with Constant"}
description:
  zh: >
      fill_k(y, value, n)：把 n 个 float 写成常量 value。
      
  en: >
      fill_k(y, value, n): writes n floats set to the constant value.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.868Z"
fingerprint: 4ded9dfb61802656ca5f357eacd83c1bea5870f2af5b0400a7394308dcca05dd
source:
  - path: "kernels/asm/k_new/001__Z6fill_kPffx.s"
apis:
  - protocol: rpc
    path: "fill_k"
    description:
      zh: >
          填充缓冲区为常量
          
      en: >
          Fills a buffer with a constant
          
---
