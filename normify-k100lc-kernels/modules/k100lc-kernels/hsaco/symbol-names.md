---
uid: e044e5ec
id: k100lc-kernels.hsaco.symbol-names
parent: k100lc-kernels.hsaco
name: {zh: "符号名与隐藏参数", en: "Symbol Names & Hidden Args"}
description:
  zh: >
      维护 HSACO 内核符号名（mangled 名）与隐藏参数（hidden_*）的映射定义，供运行时查找与参数填充共用。
      
  en: >
      Maintains the shared definitions mapping HSACO kernel symbol (mangled names) and hidden parameters (hidden_*), used by both lookup and argument filling.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.857Z"
fingerprint: 6aca8ccf8db3c555de8ae6558a54d2fed85fb2345c16aa65c2ba5daca512339f
source:
  - path: "tools/hsaco_names.py"
apis:
  - protocol: file
    path: "tools/hsaco_names.py"
    description:
      zh: >
          mangled 名 ↔ 查找名、隐藏参数编号
          
      en: >
          Mangled name to lookup name; hidden-arg codes
          
---
