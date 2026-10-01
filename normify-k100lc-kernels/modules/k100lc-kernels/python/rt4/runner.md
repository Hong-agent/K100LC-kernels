---
uid: 5237f968
id: k100lc-kernels.python.rt4.runner
parent: k100lc-kernels.python.rt4
name: {zh: "W4 运行器", en: "W4 Runner"}
description:
  zh: >
      W4Runner：复用设备缓冲的 W4A8/W4A4 执行器，_buf 管理按名复用的缓冲区，close 释放。
      
  en: >
      W4Runner: a W4A8/W4A4 executor reusing device buffers, with _buf managing name-keyed buffers and close releasing them.
      
revision: d8724c9d122531123b90d1dbb6d222c19d8a443a
updated_at: "2026-10-01T15:09:07.887Z"
fingerprint: 9c30915ad83bb227a417d704f682c99a5d6f67f22200a81824d911530e2d9359
source:
  - path: "python/k100lc_kernels/rt4.py"
    line: 44
    end_line: 60
  - path: "python/k100lc_kernels/rt4.py"
    line: 193
    end_line: 215
apis:
  - protocol: file
    path: "k100lc_kernels.rt4.W4Runner._buf"
    description:
      zh: >
          按名复用设备缓冲
          
      en: >
          Reuses a device buffer by name
          
---
