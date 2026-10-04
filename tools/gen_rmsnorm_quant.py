#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RMSNorm + W4A8 激活量化融合内核：`rmsnorm_quant_k`。

为什么合并：本项目（以及消费方 FASTASM）在每一层 decoder 里都是

    k_rmsnorm(xb, x, in_ln, 1, Hd, eps, ...);   # 一个 dispatch
    k_w4a8_quant(xb, Hd);                       # 又一个 dispatch

两步。两个内核各自都已经贴着「一次 dispatch ~7.3 us」的硬件地板（见
ROADMAP「每次 dispatch ~7.3 us，而且是硬件的」），所以唯一能省的就是
**少派发一次**：归一化后的 f32 向量（dim*4 字节）既不写也不读，直接在同一
颗内核里量化成 W4A8 的激活布局。这与参考项目 stratum/MI50 那套
「norm 的同时输出量化图像、中间 f32 不落地」是同一个思路。

语义（量化约定逐位对齐 `quant_act`）：

    rs     = rsqrt(mean(x^2) + eps)           # 整行一个，归约顺序同 rmsnorm_fast_k
    y[i]   = x[i] * rs * w[i]
    g      = i // 128
    amax_g = max(|y| over group g)
    s[g]   = amax_g > 0 ? amax_g / 127 : 1.0  # 同 quant_act 的 IEEE 除法序列
    q[i]   = clamp(rint(y[i] / s[g]), -128, 127)
    ae[j]  = q[2j]                            # 偶下表一字节流
    ao[j]  = q[2j+1]                          # 奇下表一字节流
    asum[g]= Σ_{i∈g} q[i]

签名：
    rmsnorm_quant_k(int8_t* ae, int8_t* ao, float* sc, float* asum,
                    const float* x, const float* w, uint32_t dim, float eps)

grid = rows、workgroup = 64；要求 `dim % 128 == 0`。
LDS：归约槽 64×4 B + 广播槽 4 B。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "rmsnorm_quant_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16), buffer_arg(24),
        buffer_arg(32), buffer_arg(40), scalar_arg(48, 4), scalar_arg(52, 4)]
KERNARG_SIZE = 56
GROUP_SEGMENT = 1024
NB = 4                      # 第一趟一次发几条 load
STRIDE = 64 * 4             # 相邻抽样元素的字节间隔
GRP = 128                   # 量化组


def _fdiv(L: list[str], dst: str, a: str, b: str) -> None:
    """IEEE f32 除法序列 (a/b)，与本机编译器 / `quant_act` 的展开一致。"""
    t0, t1, t2, t3, t4 = "v46", "v47", "v48", "v49", "v50"
    L += [
        f"v_div_scale_f32 {t0}, s[6:7], {b}, {b}, {a}",
        f"v_div_scale_f32 {t1}, vcc, {a}, {b}, {a}",
        f"v_rcp_f32_e32 {t2}, {t0}",
        f"v_fma_f32 {t3}, -{t0}, {t2}, 1.0",
        f"v_fmac_f32_e32 {t2}, {t3}, {t2}",
        f"v_fma_f32 {t4}, -{t0}, {t3}, {t1}",
        f"v_fmac_f32_e32 {t3}, {t4}, {t2}",
        f"v_fma_f32 {t0}, -{t0}, {t3}, {t1}",
        f"v_div_fmas_f32 {t4}, {t0}, {t2}, {t3}",
        f"v_div_fixup_f32 {dst}, {t4}, {b}, {a}",
    ]


def _tree(L: list[str], reg: str, off: int) -> None:
    """64 lane 的 LDS 树形归约（槽位 = lane id）。"""
    L += [
        f"v_cmp_gt_u32_e32 vcc, {off}, v0",
        "s_and_saveexec_b64 s[2:3], vcc",
        f"v_mov_b32_e32 v9, {off * 4}",
        "v_add_u32_e32 v9, v9, v8",
        "ds_read_b32 v19, v9",
        "s_waitcnt lgkmcnt(0)",
        f"v_add_f32_e32 {reg}, {reg}, v19",
        f"ds_write_b32 v8, {reg}",
        "s_or_b64 exec, exec, s[2:3]",
        "s_barrier",
    ]


def _shfl(L: list[str], reg: str, off: int, op: str) -> None:
    """wave 内 64 lane 蝶形归约（ds_bpermute，无 barrier）。"""
    L += [
        f"v_xor_b32_e32 v27, {off}, v0",
        "v_lshlrev_b32_e32 v27, 2, v27",
        f"ds_bpermute_b32 v28, v27, {reg}",
        "s_waitcnt lgkmcnt(0)",
        f"{op} {reg}, {reg}, v28",
    ]


def gen_asm() -> str:
    S = STRIDE
    L: list[str] = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",     # ae
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",     # ao
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",    # sc
        "s_load_dwordx2 s[22:23], s[4:5], 0x18",    # asum
        "s_load_dwordx2 s[26:27], s[4:5], 0x20",    # x
        "s_load_dwordx2 s[28:29], s[4:5], 0x28",    # w
        "s_load_dwordx2 s[24:25], s[4:5], 0x30",    # dim, eps
        "s_waitcnt lgkmcnt(0)",
        "s_lshr_b32 s30, s24, 6",                   # n64 = dim/64
        "s_lshr_b32 s31, s24, 7",                   # ng  = dim/128
        "v_mov_b32_e32 v1, 0",
        "v_mov_b32_e32 v2, s6",                     # row = bid
        "v_mul_lo_u32 v3, v2, s24",
        "v_lshlrev_b32_e32 v3, 2, v3",              # row 的字节偏移
        "v_mov_b32_e32 v4, s26",
        "v_mov_b32_e32 v5, s27",
        "v_add_co_u32_e32 v4, vcc, v4, v3",
        "v_addc_co_u32_e32 v5, vcc, v5, v1, vcc",   # x 行指针 v4:5
        "v_lshlrev_b32_e32 v8, 2, v0",              # lane*4（LDS 槽地址）
        "v_mov_b32_e32 v16, 0",                     # sumsq
        "s_mov_b32 s0, 0",                          # j（第几组 64 个元素）
        # ---- 第一趟：Σx²，一次发 NB 条 load ----
        "L_sb:",
        f"s_add_i32 s1, s0, {NB - 1}",
        "s_cmp_lt_u32 s1, s30",
        "s_cbranch_scc0 L_s1",
        "v_mov_b32_e32 v9, s0",
        "v_lshlrev_b32_e32 v9, 8, v9",              # j*256
        "v_add_u32_e32 v9, v9, v8",
        "v_mov_b32_e32 v11, v5",
        "v_add_co_u32_e32 v10, vcc, v4, v9",
        "v_addc_co_u32_e32 v11, vcc, v11, v1, vcc",
        *[f"global_load_dword v{12 + k}, v[10:11], off offset:{S * k}"
          for k in range(NB)],
        "s_waitcnt vmcnt(0)",
        *[f"v_fma_f32 v16, v{12 + k}, v{12 + k}, v16" for k in range(NB)],
        f"s_add_i32 s0, s0, {NB}",
        "s_branch L_sb",
        "L_s1:",
        "s_cmp_lt_u32 s0, s30",
        "s_cbranch_scc0 L_s1_done",
        "v_mov_b32_e32 v9, s0",
        "v_lshlrev_b32_e32 v9, 8, v9",
        "v_add_u32_e32 v9, v9, v8",
        "v_mov_b32_e32 v11, v5",
        "v_add_co_u32_e32 v10, vcc, v4, v9",
        "v_addc_co_u32_e32 v11, vcc, v11, v1, vcc",
        "global_load_dword v12, v[10:11], off",
        "s_waitcnt vmcnt(0)",
        "v_fma_f32 v16, v12, v12, v16",
        "s_add_i32 s0, s0, 1",
        "s_branch L_s1",
        "L_s1_done:",
        # ---- 归约 + 广播 rs = rsqrt(mean+eps) ----
        "ds_write_b32 v8, v16",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
    ]
    for off in (32, 16, 8, 4, 2, 1):
        _tree(L, "v16", off)
    L += [
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[2:3], vcc",
        "v_mov_b32_e32 v9, 256",
        "ds_write_b32 v9, v16",
        "s_or_b64 exec, exec, s[2:3]",
        "s_barrier",
        "v_mov_b32_e32 v9, 256",
        "ds_read_b32 v17, v9",
        "s_waitcnt lgkmcnt(0)",
        "v_cvt_f32_u32_e32 v18, s24",
        "s_nop 0",
        "v_rcp_f32_e32 v18, v18",                   # 1/dim
        "s_nop 0",
        "v_mul_f32_e32 v17, v17, v18",              # mean(x²)
        "v_mov_b32_e32 v19, s25",
        "v_add_f32_e32 v17, v17, v19",              # + eps
        "s_nop 0",
        "v_rsq_f32_e32 v17, v17",                   # rs = 1/sqrt(mean+eps)
        "s_nop 0",
        "v_mov_b32_e32 v29, 0x3F800000",            # 1.0f
        "v_mov_b32_e32 v45, 0x42FE0000",            # 127.0f
        "v_mov_b32_e32 v51, 0xFFFFFF80",            # -128
        "v_mov_b32_e32 v52, 127",                   # 127
        "v_mul_lo_u32 v53, v2, s31",                # row*ng
        "v_lshlrev_b32_e32 v53, 6, v53",            # ae / ao 的行偏移
        "s_mov_b32 s0, 0",                          # g（第几个 128 组）
        # ---- 第二趟：逐组 归一化 -> amax -> scale -> 量化 ----
        "L_g:",
        "s_cmp_lt_u32 s0, s31",
        "s_cbranch_scc0 L_g_done",
        "v_mov_b32_e32 v20, s0",
        "v_lshlrev_b32_e32 v20, 7, v20",            # 128*g
        "v_add_u32_e32 v20, v20, v0",               # j0 = 128g + lane
        "v_lshlrev_b32_e32 v21, 2, v20",            # j0*4
        "v_mov_b32_e32 v11, v5",
        "v_add_co_u32_e32 v10, vcc, v4, v21",
        "v_addc_co_u32_e32 v11, vcc, v11, v1, vcc",
        "global_load_dword v22, v[10:11], off offset:0",       # x[j0]
        "global_load_dword v23, v[10:11], off offset:256",     # x[j0+64]
        "v_mov_b32_e32 v11, s29",
        "v_add_co_u32_e32 v10, vcc, s28, v21",
        "v_addc_co_u32_e32 v11, vcc, v11, v1, vcc",
        "global_load_dword v24, v[10:11], off offset:0",       # w[j0]
        "global_load_dword v25, v[10:11], off offset:256",     # w[j0+64]
        "s_waitcnt vmcnt(0)",
        "v_mul_f32_e32 v22, v22, v17",              # x*rs
        "v_mul_f32_e32 v23, v23, v17",
        "v_mul_f32_e32 v22, v22, v24",              # y0
        "v_mul_f32_e32 v23, v23, v25",              # y1
        "v_max_f32_e64 v26, |v22|, |v23|",          # 组内局部 max
    ]
    for off in (32, 16, 8, 4, 2, 1):
        _shfl(L, "v26", off, "v_max_f32_e32")
    L += [
        # s = amax>0 ? amax/127 : 1.0
        *_fdiv_seq("v30", "v26", "v45"),
        "v_cmp_lt_f32_e32 vcc, 0, v26",
        "v_cndmask_b32_e32 v30, v29, v30, vcc",
        # q0 / q1 = clamp(rint(y / s), -128, 127)
        *_fdiv_seq("v31", "v22", "v30"),
        "v_rndne_f32_e32 v31, v31",
        "v_cvt_i32_f32_e32 v31, v31",
        "v_med3_i32 v31, v31, v51, v52",
        *_fdiv_seq("v32", "v23", "v30"),
        "v_rndne_f32_e32 v32, v32",
        "v_cvt_i32_f32_e32 v32, v32",
        "v_med3_i32 v32, v32, v51, v52",
        # ---- 写 ae / ao：偶 lane 写 ae，奇 lane 写 ao ----
        "v_lshrrev_b32_e32 v33, 1, v0",             # lane/2
        "v_mov_b32_e32 v34, s0",
        "v_lshlrev_b32_e32 v34, 6, v34",            # 64g
        "v_add_u32_e32 v33, v33, v34",              # idx0 = 64g + lane/2
        "v_add_u32_e32 v33, v33, v53",              # + 行偏移
        "v_add_u32_e32 v34, 32, v33",               # idx1 = idx0 + 32
        "v_and_b32_e32 v35, 1, v0",
        "v_cmp_eq_u32_e32 vcc, 0, v35",             # vcc 真 = lane 偶
        "v_mov_b32_e32 v36, s16",                   # ae lo
        "v_mov_b32_e32 v37, s18",                   # ao lo
        "v_mov_b32_e32 v38, s17",                   # ae hi
        "v_mov_b32_e32 v39, s19",                   # ao hi
        "v_cndmask_b32_e32 v36, v37, v36, vcc",     # base lo
        "v_cndmask_b32_e32 v37, v39, v38, vcc",     # base hi
        "v_add_co_u32_e32 v38, vcc, v36, v33",
        "v_addc_co_u32_e32 v39, vcc, v37, v1, vcc",
        "global_store_byte v[38:39], v31, off",
        "v_add_co_u32_e32 v38, vcc, v36, v34",
        "v_addc_co_u32_e32 v39, vcc, v37, v1, vcc",
        "global_store_byte v[38:39], v32, off",
        # ---- asum[g] = Σ q（64 lane 蝶形求和）----
        "v_add_u32_e32 v35, v31, v32",
    ]
    for off in (32, 16, 8, 4, 2, 1):
        _shfl(L, "v35", off, "v_add_u32_e32")
    L += [
        # lane 0：写 sc[g] 与 asum[g]
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[2:3], vcc",
        "v_mul_lo_u32 v36, v2, s31",
        "v_add_u32_e32 v36, s0, v36",
        "v_lshlrev_b32_e32 v36, 2, v36",            # (row*ng + g)*4
        "v_mov_b32_e32 v37, s21",
        "v_add_co_u32_e32 v38, vcc, s20, v36",
        "v_addc_co_u32_e32 v39, vcc, v37, v1, vcc",
        "global_store_dword v[38:39], v30, off",    # sc[g]
        "v_mov_b32_e32 v37, s23",
        "v_add_co_u32_e32 v38, vcc, s22, v36",
        "v_addc_co_u32_e32 v39, vcc, v37, v1, vcc",
        "global_store_dword v[38:39], v35, off",    # asum[g]
        "s_or_b64 exec, exec, s[2:3]",
        "s_add_i32 s0, s0, 1",
        "s_branch L_g",
        "L_g_done:",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


def _fdiv_seq(dst: str, a: str, b: str) -> list[str]:
    L: list[str] = []
    _fdiv(L, dst, a, b)
    return L


def _ref(x: np.ndarray, w: np.ndarray, eps: float):
    """主机参考：rmsnorm（同内核的 rs）+ quant_act 的量化约定。"""
    x = np.asarray(x, np.float32)
    w = np.asarray(w, np.float32)
    ms = (x.astype(np.float64) ** 2).mean(axis=1, keepdims=True)
    rs = (1.0 / np.sqrt(ms + eps)).astype(np.float32)
    y = (x * rs).astype(np.float32) * w[None, :]
    m, k = y.shape
    yg = y.reshape(m, k // GRP, GRP)
    amax = np.abs(yg).max(axis=2)
    s = np.where(amax > 0, (amax.astype(np.float32) / np.float32(127.0)),
                 np.float32(1.0)).astype(np.float32)
    q = np.clip(np.rint(yg / s[:, :, None]), -128, 127).astype(np.int32)
    q = q.reshape(m, k)
    asum = q.reshape(m, k // GRP, GRP).sum(axis=2).astype(np.int32)
    return q, s, asum


def selftest() -> int:
    hsaco = build_one(NAME, gen_asm(), ARGS, KERNARG_SIZE,
                      group_segment=GROUP_SEGMENT, tag="rnq")
    rng = np.random.default_rng(20261012)
    rc = 0
    for rows, dim in ((1, 128), (2, 256), (1, 512), (3, 1280), (1, 5120)):
        ng = dim // GRP
        x = rng.standard_normal((rows, dim)).astype(np.float32) * 2.0
        w = (rng.random(dim).astype(np.float32) + 0.5)
        eps = 1e-6
        out = run_one(hsaco, NAME,
                      [{"buffer": "ae"}, {"buffer": "ao"},
                       {"buffer": "sc"}, {"buffer": "asum"},
                       {"buffer": "x"}, {"buffer": "w"},
                       {"scalar": {"dtype": "u32", "value": dim}},
                       {"scalar": {"dtype": "f32", "value": eps}}],
                      {"ae": {"dtype": "i8", "values": [0] * (rows * dim // 2)},
                       "ao": {"dtype": "i8", "values": [0] * (rows * dim // 2)},
                       "sc": {"dtype": "f32", "values": [0.0] * (rows * ng)},
                       "asum": {"dtype": "i32", "values": [0] * (rows * ng)},
                       "x": {"dtype": "f32", "values": x.reshape(-1).tolist()},
                       "w": {"dtype": "f32", "values": w.tolist()}},
                      grid=rows * 64, workgroup=64)
        qref, sref, asumref = _ref(x, w, eps)
        ae = np.array(out["ae"], np.int8).reshape(rows, dim // 2)
        ao = np.array(out["ao"], np.int8).reshape(rows, dim // 2)
        q = np.empty((rows, dim), np.int32)
        q[:, 0::2] = ae
        q[:, 1::2] = ao
        sc = np.array(out["sc"], np.float32).reshape(rows, ng)
        asum = np.array(out["asum"], np.int32).reshape(rows, ng)
        dq = np.abs(q - qref)
        bad = int((dq != 0).sum())
        srel = float(np.abs(sc - sref).max() / max(1e-9, float(np.abs(sref).max())))
        arel = float(np.abs(asum.astype(np.int64) - asumref.astype(np.int64)).max()
                     / max(1.0, float(np.abs(asumref).max())))
        ok = (dq.max() <= 1 and bad <= q.size // 1000
              and srel < 1e-6 and arel < 1e-5)
        rc |= 0 if ok else 1
        print(f"rows={rows} dim={dim}: q差={int(dq.max())} 不一致={bad}/{q.size} "
              f"sc_rel={srel:.2e} asum_rel={arel:.2e} {'✔' if ok else '✘'}")
    return rc


def main() -> int:
    rc = selftest()
    print("rmsnorm_quant_k:", "全部一致 ✔" if rc == 0 else "有不一致 ✘")
    return rc


KERNELS = [(NAME, gen_asm, ARGS, KERNARG_SIZE)]

if __name__ == "__main__":
    raise SystemExit(main())
