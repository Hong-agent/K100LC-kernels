---
uid: 5ad5449c
id: k100lc-kernels.runtime.hsa.hsaco-load
parent: k100lc-kernels.runtime.hsa
name: {zh: "HSACO 加载与符号查找", en: "HSACO Loading & Symbol Lookup"}
description:
  zh: >
      把 HSACO 读成 executable 并遍历符号表，取出每个内核的 kernel object 与显式参数布局（get_symbol/cb_symbol/load_hsaco）。
      
  en: >
      Loads the HSACO as an executable and walks its symbol table to collect each kernel’s kernel object and explicit argument layout (get_symbol/cb_symbol/load_hsaco).
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.883Z"
fingerprint: 12d13a092207d5da24b991721ce5aafd5fa3a94506ed44e0fa89cd4aaed25ea5
source:
  - path: "runtime/hsa_rt.cpp"
    line: 158
    end_line: 249
apis:
  - protocol: file
    path: "runtime/hsa_rt.cpp:load_hsaco"
    description:
      zh: >
          加载代码对象并建内核表
          
      en: >
          Loads the code object and builds the kernel table
          
---
