#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GDN 小算子融合：`k_split_qkv + 2×k_l2norm` → 一个内核 + 自检。

原路径（每层 3 次启动、每个 ~20µs）：
    k_split_qkv(gq,gk,gv,conv,T,qn,kn,vn)
    k_l2norm(gq, T*Hq, D, eps)
    k_l2norm(gk, T*Hk, D, eps)

融合后（1 次启动）：一个 warp 负责「一个 token 的一个头」——
读 2 个元素（lane, lane+64；要求 D=128）、wave 内 `ds_bpermute` 归约平方和、
`v_rsq_f32(ss+eps)` 后写回；同时把 v 段按块切块拷过去。

签名：
  k_gdn_split_norm_k(float* q, float* k, float* v, const float* c,
                     u32 T, u32 qn, u32 kn, u32 vn, u32 D, u32 wbits,
                     u32 hq, float eps)
grid = T << wbits（每块一个 warp，wbits = log2(hq + hk)），workgroup = 64。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "k_gdn_split_norm_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16), buffer_arg(24),
        scalar_arg(32, 4), scalar_arg(36, 4), scalar_arg(40, 4), scalar_arg(44, 4),
        scalar_arg(48, 4), scalar_arg(52, 4), scalar_arg(56, 4), scalar_arg(60, 4)]
KERNARG_SIZE = 64


def gen_asm() -> str:
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{NAME}:")
    em("s_load_dwordx2 s[16:17], s[4:5], 0x0")     # q
    em("s_load_dwordx2 s[18:19], s[4:5], 0x8")     # k
    em("s_load_dwordx2 s[20:21], s[4:5], 0x10")    # v
    em("s_load_dwordx2 s[22:23], s[4:5], 0x18")    # c
    em("s_load_dword s24, s[4:5], 0x20")           # T
    em("s_load_dword s25, s[4:5], 0x24")           # qn
    em("s_load_dword s26, s[4:5], 0x28")           # kn
    em("s_load_dword s27, s[4:5], 0x2c")           # vn
    em("s_load_dword s28, s[4:5], 0x30")           # D
    em("s_load_dword s29, s[4:5], 0x34")           # wbits
    em("s_load_dword s30, s[4:5], 0x38")           # hq
    em("s_load_dword s31, s[4:5], 0x3c")           # eps (f32)
    em("s_waitcnt lgkmcnt(0)")
    em("v_mov_b32_e32 v4, 0")
    em("v_mov_b32_e32 v50, s25")                   # qn
    em("v_mov_b32_e32 v51, s26")                   # kn
    em("v_mov_b32_e32 v52, s27")                   # vn
    em("v_mov_b32_e32 v54, s28")                   # D
    em("v_add_u32_e32 v55, v50, v50")              # 2*qn
    em("v_add_u32_e32 v53, v55, v52")              # C = 2*qn + vn
    em("v_mov_b32_e32 v57, s31")                   # eps
    em("v_and_b32_e32 v1, 63, v0")                 # lane
    em("v_mov_b32_e32 v6, s29")                    # wbits
    em("v_mov_b32_e32 v7, s6")                     # bid
    em("v_lshrrev_b32_e32 v3, v6, v7")             # t = bid >> wbits
    em("v_mov_b32_e32 v5, 1")
    em("v_lshlrev_b32_e32 v5, v6, v5")
    em("v_add_u32_e32 v5, -1, v5")                 # mask = (1<<wbits)-1
    em("v_and_b32_e32 v2, v5, v7")                 # w = bid & mask
    # 源/目的：w < hq → q 头，否则 k 头
    em("v_cmp_gt_u32_e32 vcc, s30, v2")            # hq > w ⇔ w<hq
    em("v_mov_b32_e32 v5, s30")
    em("v_sub_u32_e32 v5, v2, v5")                 # w-hq（k 用）
    em("v_cndmask_b32_e32 v5, v5, v2, vcc")        # 真→第二源：w<hq 取 v2
    # C = 2*qn + vn ; src 行基址 = t*C + (w<hq ? w : qn + w-hq)*D
    em("v_mov_b32_e32 v8, v53")                    # C
    em("v_mul_lo_u32 v8, v3, v8")                  # t*C
    em("v_mul_lo_u32 v9, v5, v54")                 # head*D
    em("v_add_u32_e32 v8, v8, v9")
    em("v_cmp_gt_u32_e32 vcc, s30, v2")
    em("v_add_u32_e32 v9, v50, v9")                # qn + head*D（k 用）
    em("v_cndmask_b32_e32 v9, v9, v8, vcc")        # 真→第二源
    em("v_lshlrev_b32_e32 v9, 2, v9")
    em("v_mov_b32_e32 v11, s23")
    em("v_add_co_u32_e32 v10, vcc, s22, v9")
    em("v_addc_co_u32_e32 v11, vcc, v11, v4, vcc")  # c + src_off*4
    # dst = (w<hq ? q + t*qn + head*D : k + t*kn + head*D)
    em("v_mul_lo_u32 v12, v3, v50")                # t*qn
    em("v_mul_lo_u32 v13, v3, v51")                # t*kn
    em("v_mul_lo_u32 v14, v5, v54")                # head*D
    em("v_add_u32_e32 v12, v12, v14")
    em("v_add_u32_e32 v13, v13, v14")
    em("v_lshlrev_b32_e32 v12, 2, v12")
    em("v_lshlrev_b32_e32 v13, 2, v13")
    em("v_mov_b32_e32 v15, s17")
    em("v_add_co_u32_e32 v14, vcc, s16, v12")
    em("v_addc_co_u32_e32 v15, vcc, v15, v4, vcc")   # q 目的
    em("v_mov_b32_e32 v17, s19")
    em("v_add_co_u32_e32 v16, vcc, s18, v13")
    em("v_addc_co_u32_e32 v17, vcc, v17, v4, vcc")   # k 目的
    em("v_cmp_gt_u32_e32 vcc, s30, v2")
    em("v_cndmask_b32_e32 v18, v16, v14, vcc")
    em("v_cndmask_b32_e32 v19, v17, v15, vcc")
    # 读两个元素
    em("v_lshlrev_b32_e32 v8, 2, v1")              # lane*4
    em("v_add_u32_e32 v9, 256, v8")                # (lane+64)*4
    em("v_mov_b32_e32 v21, v11")
    em("v_add_co_u32_e32 v20, vcc, v10, v8")
    em("v_addc_co_u32_e32 v21, vcc, v21, v4, vcc")
    em("global_load_dword v22, v[20:21], off")
    em("v_mov_b32_e32 v21, v11")
    em("v_add_co_u32_e32 v20, vcc, v10, v9")
    em("v_addc_co_u32_e32 v21, vcc, v21, v4, vcc")
    em("global_load_dword v23, v[20:21], off")
    em("s_waitcnt vmcnt(0)")
    em("v_mul_f32_e32 v24, v22, v22")
    em("v_fma_f32 v24, v23, v23, v24")             # ss
    # wave 内蝶形归约
    for off in (32, 16, 8, 4, 2, 1):
        em(f"v_xor_b32_e32 v25, {off}, v1")
        em("v_lshlrev_b32_e32 v25, 2, v25")
        em("ds_bpermute_b32 v26, v25, v24")
        em("s_waitcnt lgkmcnt(0)")
        em("v_add_f32_e32 v24, v24, v26")
    em("v_add_f32_e32 v24, v24, v57")              # + eps
    em("v_rsq_f32_e32 v24, v24")
    em("v_mul_f32_e32 v22, v22, v24")
    em("v_mul_f32_e32 v23, v23, v24")
    em("v_mov_b32_e32 v21, v19")
    em("v_add_co_u32_e32 v20, vcc, v18, v8")
    em("v_addc_co_u32_e32 v21, vcc, v21, v4, vcc")
    em("global_store_dword v[20:21], v22, off")
    em("v_mov_b32_e32 v21, v19")
    em("v_add_co_u32_e32 v20, vcc, v18, v9")
    em("v_addc_co_u32_e32 v21, vcc, v21, v4, vcc")
    em("global_store_dword v[20:21], v23, off")
    # v 段拷贝：每块 vn>>wbits 个元素，每 lane 1/64
    em("v_lshrrev_b32_e32 v27, v6, v52")           # nv = vn >> wbits
    em("v_lshrrev_b32_e32 v28, 6, v27")            # nv/64（假设整除）
    em("v_mov_b32_e32 v29, 0")                     # i
    em("L_vloop:")
    em("v_cmp_lt_u32_e32 vcc, v29, v28")
    em("s_cbranch_vccz L_vend")
    em("v_lshlrev_b32_e32 v30, 6, v29")            # i*64
    em("v_add_u32_e32 v30, v30, v1")               # + lane
    em("v_mul_lo_u32 v31, v2, v27")                # w*nv
    em("v_add_u32_e32 v30, v30, v31")              # j（v 段内偏移）
    # src 号 = t*C + 2*qn + j
    em("v_mov_b32_e32 v40, v55")                   # 2*qn
    em("v_add_u32_e32 v40, v40, v52")              # + vn = C
    em("v_mul_lo_u32 v40, v3, v40")                # t*C
    em("v_mov_b32_e32 v41, v55")
    em("v_add_u32_e32 v41, v41, v30")              # + j
    em("v_add_u32_e32 v41, v41, v40")
    em("v_lshlrev_b32_e32 v41, 2, v41")
    em("v_mov_b32_e32 v42, s23")
    em("v_add_co_u32_e32 v41, vcc, s22, v41")
    em("v_addc_co_u32_e32 v42, vcc, v42, v4, vcc")
    em("global_load_dword v43, v[41:42], off")
    em("s_waitcnt vmcnt(0)")
    # dst 号 = t*vn + j
    em("v_mul_lo_u32 v40, v3, v52")
    em("v_add_u32_e32 v40, v40, v30")
    em("v_lshlrev_b32_e32 v40, 2, v40")
    em("v_mov_b32_e32 v45, s21")
    em("v_add_co_u32_e32 v44, vcc, s20, v40")
    em("v_addc_co_u32_e32 v45, vcc, v45, v4, vcc")
    em("global_store_dword v[44:45], v43, off")
    em("v_add_u32_e32 v29, 1, v29")
    em("s_branch L_vloop")
    em("L_vend:")
    em("s_endpgm")
    return "\n".join(L) + "\n"


def main() -> int:
    # 数值自检：与「split + 逐头 l2norm」的参考实现对比
    rng = np.random.default_rng(5)
    T, Hq, Hk, D = 2, 16, 16, 128      # 与 27B 模型一致（v 段 48 头 = 6144 元素）
    HB = Hq + Hk
    wbits = int(np.log2(HB))
    qn, kn, vn = Hq * D, Hk * D, 48 * D
    C = 2 * qn + vn
    c = rng.standard_normal((T, C)).astype(np.float32)
    eps = np.float32(1e-6)
    q = c[:, :qn].copy()
    k = c[:, qn:qn + kn].copy()
    v = c[:, 2 * qn:2 * qn + vn].copy()
    for arr, n in ((q, Hq), (k, Hk)):
        for t in range(T):
            for h in range(n):
                sl = slice(h * D, (h + 1) * D)
                ss = np.float32((arr[t, sl].astype(np.float32) ** 2).sum())
                inv = np.float32(1.0 / np.sqrt(np.float32(ss + eps)))
                arr[t, sl] = arr[t, sl] * inv
    hs = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="gdnpre")
    out = run_one(hs, NAME,
                  [{"buffer": "q"}, {"buffer": "k"}, {"buffer": "v"}, {"buffer": "c"},
                   {"scalar": {"dtype": "u32", "value": T}},
                   {"scalar": {"dtype": "u32", "value": qn}},
                   {"scalar": {"dtype": "u32", "value": kn}},
                   {"scalar": {"dtype": "u32", "value": vn}},
                   {"scalar": {"dtype": "u32", "value": D}},
                   {"scalar": {"dtype": "u32", "value": wbits}},
                   {"scalar": {"dtype": "u32", "value": Hq}},
                   {"scalar": {"dtype": "f32", "value": 1e-6}}],
                  {"q": {"dtype": "f32", "values": [0.0] * (T * qn)},
                   "k": {"dtype": "f32", "values": [0.0] * (T * kn)},
                   "v": {"dtype": "f32", "values": [0.0] * (T * vn)},
                   "c": {"dtype": "f32", "values": c.reshape(-1).tolist()}},
                  grid=(T * HB) * 64, workgroup=64)
    gq = np.array(out["q"], dtype=np.float32).reshape(T, qn)
    gk = np.array(out["k"], dtype=np.float32).reshape(T, kn)
    gv = np.array(out["v"], dtype=np.float32).reshape(T, vn)
    eq = float(np.abs(gq - q).max() / max(1e-9, float(np.abs(q).max())))
    ek = float(np.abs(gk - k).max() / max(1e-9, float(np.abs(k).max())))
    ev = float(np.abs(gv - v).max() / max(1e-9, float(np.abs(v).max())))
    ok = max(eq, ek, ev) < 1e-4
    print(f"gdn_split_norm: q {eq:.2e} k {ek:.2e} v {ev:.2e} {'OK ✔' if ok else 'FAIL ✘'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
