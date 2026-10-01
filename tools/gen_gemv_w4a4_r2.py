#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W4A4 GEMV（每 warp 2 行权重，M=4）—— MTP 验证批的专用快路 + 自检。

背景：现役 `gemv_w4a4<4>` 每个 warp 只处理 1 行权重，每个 lane 每轮读
4×16B 激活 + 1×16B 权重，M=4 时实测只有 ~275 GB/s（M=1 有 465 GB/s），
是解码的最大的瓶颈。W4A8 那边用「每 warp 2 行权重、两份权重共享同一份
激活加载」拿到了 +22%。W4A4 没有这个变体，这里补上。

算法（与 gemv_w4a4_core.h 逐位一致：int4 权重 × int4 激活，`v_dot8_i32_i4`）：

    out[m][row] = Σ_k (qw[row,k] * ws[row,k/128]) * (qa[m,k] * as[m,k/32])

一个 lane 每轮拿 4 个 dword（32 个权重 / 32 个激活）；64 lane 一轮覆盖 2048
个元素。ROWS=2 时同一份 a4 被两行权重复用，激活的全局读减半。

签名：gemv_w4a4_r2_k(const u32* wq, const u16* ws, const u32* aq,
                     const float* as, float* out, u32 N, u32 K)
前提：M=4、K%8==0；grid = ceil(N/8)，workgroup = 256（4 warp × 2 行）。
"""
from __future__ import annotations

import pathlib
import os
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "gemv_w4a4_r2_k"
M3_NAME = "gemv_w4a4_r2_m3_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16), buffer_arg(24),
        buffer_arg(32), scalar_arg(40, 4), scalar_arg(44, 4)]
KERNARG_SIZE = 48
M = 4           # token 数（MTP 验证批）默认值
ROWS = 2        # 每 warp 权重行数
NWARP = 4       # 每 block 的 warp 数
_NORED = bool(os.environ.get("R2_NORED"))   # 调试用：跳过错 lane 归约


def _cvt_f32(L, dst, src):
    L.append(f"v_cvt_f32_i32_e32 {dst}, {src}")


def gen_asm(M: int = M, name: str = NAME) -> str:
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{name}:")
    em("s_load_dwordx2 s[16:17], s[4:5], 0x0")     # wq
    em("s_load_dwordx2 s[18:19], s[4:5], 0x8")     # ws
    em("s_load_dwordx2 s[20:21], s[4:5], 0x10")    # aq
    em("s_load_dwordx2 s[22:23], s[4:5], 0x18")    # as
    em("s_load_dwordx2 s[24:25], s[4:5], 0x20")    # out
    em("s_load_dword s26, s[4:5], 0x28")           # N
    em("s_load_dword s27, s[4:5], 0x2c")           # K
    em("s_waitcnt lgkmcnt(0)")
    em("s_lshr_b32 s28, s27, 3")                   # KD = K/8
    em("s_lshr_b32 s29, s27, 7")                   # NG = K/128
    em("s_lshl_b32 s30, s28, 2")                   # KD*4 字节 / 权重行
    em("s_lshl_b32 s31, s29, 1")                   # NG*2 字节 / 尺度行
    em("s_lshr_b32 s32, s27, 5")                   # K/32
    em("s_lshl_b32 s33, s32, 2")                   # (K/32)*4 字节 / 激活尺度行
    em("v_mov_b32_e32 v4, 0")
    em("v_and_b32_e32 v1, 63, v0")                 # lane
    em("v_lshrrev_b32_e32 v2, 6, v0")              # warp
    em("v_lshl_or_b32 v2, s6, 2, v2")              # blockIdx.x*4 + warp
    em("v_lshlrev_b32_e32 v2, 1, v2")              # row0 = *ROWS
    em("v_cmp_gt_i32_e32 vcc, s26, v2")            # N > row0 ?
    em("s_and_saveexec_b64 s[0:1], vcc")
    em("s_cbranch_execz L_end")
    # nr = (N-row0 > 1) ? 2 : 1
    em("v_mov_b32_e32 v4, s26")
    em("v_sub_u32_e32 v3, v4, v2")
    em("v_cmp_gt_u32_e32 vcc, 1, v3")              # 1 > N-row0 → nr=1
    em("v_mov_b32_e32 v4, 2")
    em("v_mov_b32_e32 v5, 1")
    em("v_cndmask_b32_e32 v3, v4, v5, vcc")        # 真 → 第二源（本机语义）
    em("v_mov_b32_e32 v4, 0")
    # ---- 基址 ----
    em("v_mul_lo_u32 v5, v2, s30")
    em("v_mov_b32_e32 v6, s16")
    em("v_add_co_u32_e32 v6, vcc, v6, v5")
    em("v_mov_b32_e32 v7, s17")
    em("v_addc_co_u32_e32 v7, vcc, v7, v4, vcc")   # wq + row0*KD*4
    em("v_mul_lo_u32 v5, v2, s31")
    em("v_mov_b32_e32 v8, s18")
    em("v_add_co_u32_e32 v8, vcc, v8, v5")
    em("v_mov_b32_e32 v9, s19")
    em("v_addc_co_u32_e32 v9, vcc, v9, v4, vcc")   # ws + row0*NG*2
    em("v_mov_b32_e32 v10, s20")
    em("v_mov_b32_e32 v11, s21")                   # aq
    em("v_mov_b32_e32 v12, s22")
    em("v_mov_b32_e32 v13, s23")                   # as
    em("v_lshlrev_b32_e32 v5, 2, v2")
    em("v_mov_b32_e32 v14, s24")
    em("v_add_co_u32_e32 v14, vcc, v14, v5")
    em("v_mov_b32_e32 v15, s25")
    em("v_addc_co_u32_e32 v15, vcc, v15, v4, vcc")  # out + row0*4
    em("v_lshlrev_b32_e32 v19, 4, v1")             # lane*16 字节
    em("v_lshlrev_b32_e32 v20, 2, v1")             # lane*4 字节（as）
    em("v_lshrrev_b32_e32 v21, 2, v1")
    em("v_lshlrev_b32_e32 v21, 1, v21")            # (lane>>2)*2（ws）
    # ---- 累加器清零 acc[2][4] = v40..v47 ----
    for r in range(ROWS):
        for m in range(M):
            em(f"v_mov_b32_e32 v{40 + r * M + m}, 0")
    em("v_mov_b32_e32 v16, 0")                     # base（dword）
    em("v_mov_b32_e32 v22, s28")                   # KD
    em("L_loop:")
    em("v_cmp_lt_u32_e32 vcc, v16, v22")           # base < KD ?
    em("s_cbranch_vccz L_done")
    # ---- 本轮的 dword 起点 d0 = base + lane*4；ok = d0+4 <= KD ----
    em("v_lshlrev_b32_e32 v23, 2, v1")
    em("v_add_u32_e32 v23, v16, v23")              # d0
    # d0、KD 都是 4 的倍数 ⇒ (d0 < KD) 等价于 (d0+4 <= KD)
    em("v_cmp_lt_u32_e32 vcc, v23, v22")
    em("s_and_saveexec_b64 s[2:3], vcc")
    em("s_cbranch_execz L_skip")
    # base 字节偏移
    em("v_lshlrev_b32_e32 v17, 2, v16")            # base*4
    em("v_lshrrev_b32_e32 v18, 4, v16")
    em("v_lshlrev_b32_e32 v18, 1, v18")            # (base>>4)*2
    # w4[0] = wq + base*4 + lane*16
    em("v_add_u32_e32 v57, v17, v19")            # 注意：不能用 v24..v27（权重寄存器）
    em("v_mov_b32_e32 v50, v6")
    em("v_add_co_u32_e32 v50, vcc, v50, v57")
    em("v_mov_b32_e32 v51, v7")
    em("v_addc_co_u32_e32 v51, vcc, v51, v4, vcc")
    em("global_load_dwordx4 v[24:27], v[50:51], off")
    # w4[1] = + KD*4
    em("v_mov_b32_e32 v52, s30")
    em("v_add_u32_e32 v56, v57, v52")
    em("v_mov_b32_e32 v50, v6")
    em("v_add_co_u32_e32 v50, vcc, v50, v56")
    em("v_mov_b32_e32 v51, v7")
    em("v_addc_co_u32_e32 v51, vcc, v51, v4, vcc")
    em("global_load_dwordx4 v[28:31], v[50:51], off")
    # sw0/sw1 = ws + (base>>4)*2 + (lane>>2)*2 [+ NG*2]
    em("v_add_u32_e32 v53, v18, v21")
    em("v_mov_b32_e32 v50, v8")
    em("v_add_co_u32_e32 v50, vcc, v50, v53")
    em("v_mov_b32_e32 v51, v9")
    em("v_addc_co_u32_e32 v51, vcc, v51, v4, vcc")
    em("global_load_ushort v36, v[50:51], off")
    em("v_mov_b32_e32 v54, s31")
    em("v_add_u32_e32 v53, v53, v54")
    em("v_mov_b32_e32 v50, v8")
    em("v_add_co_u32_e32 v50, vcc, v50, v53")
    em("v_mov_b32_e32 v51, v9")
    em("v_addc_co_u32_e32 v51, vcc, v51, v4, vcc")
    em("global_load_ushort v37, v[50:51], off")
    em("s_waitcnt vmcnt(0)")
    em("v_cvt_f32_f16_e32 v36, v36")
    em("v_cvt_f32_f16_e32 v37, v37")
    # ---- 逐 token ----
    for m in range(M):
        # a4 = aq + m*KD*4 + base*4 + lane*16
        if m:
            em("v_mov_b32_e32 v52, s30")
            em(f"v_mov_b32_e32 v54, {m}")
            em("v_mul_lo_u32 v52, v52, v54")
            em("v_add_u32_e32 v56, v17, v19")
            em("v_add_u32_e32 v56, v56, v52")
        else:
            em("v_add_u32_e32 v56, v17, v19")
        em("v_mov_b32_e32 v50, v10")
        em("v_add_co_u32_e32 v50, vcc, v50, v56")
        em("v_mov_b32_e32 v51, v11")
        em("v_addc_co_u32_e32 v51, vcc, v51, v4, vcc")
        em("global_load_dwordx4 v[32:35], v[50:51], off")
        # sa = as + m*(K/32)*4 + base + lane*4
        if m:
            em("v_mov_b32_e32 v52, s33")
            em(f"v_mov_b32_e32 v54, {m}")
            em("v_mul_lo_u32 v52, v52, v54")
            em("v_add_u32_e32 v53, v16, v20")
            em("v_add_u32_e32 v53, v53, v52")
        else:
            em("v_add_u32_e32 v53, v16, v20")
        em("v_mov_b32_e32 v50, v12")
        em("v_add_co_u32_e32 v50, vcc, v50, v53")
        em("v_mov_b32_e32 v51, v13")
        em("v_addc_co_u32_e32 v51, vcc, v51, v4, vcc")
        em("global_load_dword v38, v[50:51], off")
        em("s_waitcnt vmcnt(0)")
        for r in range(ROWS):
            base_w = 24 + 4 * r
            # t0 = dot8(w0,a0,0); t1 = dot8(w1,a1,0); t0 += w2*a2; t1 += w3*a3
            em(f"v_dot8_i32_i4 v48, v{base_w}, v32, v4")
            em(f"v_dot8_i32_i4 v49, v{base_w + 1}, v33, v4")
            em(f"v_dot8_i32_i4 v48, v{base_w + 2}, v34, v48")
            em(f"v_dot8_i32_i4 v49, v{base_w + 3}, v35, v49")
            em("v_add_u32_e32 v48, v48, v49")
            _cvt_f32(L, "v49", "v48")
            sw = 36 + r
            em(f"v_mul_f32_e32 v52, v{sw}, v38")     # sw*sa
            em(f"v_fma_f32 v{40 + r * M + m}, v52, v49, v{40 + r * M + m}")
    em("L_skip:")
    em("s_or_b64 exec, exec, s[2:3]")
    em("v_add_u32_e32 v16, 256, v16")
    em("s_branch L_loop")
    em("L_done:")
    # ---- warp 内归约（ds_bpermute 蝶形）并写 out ----
    for r in range(ROWS):
        for m in range(M):
            acc = 40 + r * M + m
            if not _NORED:
                for off in (32, 16, 8, 4, 2, 1):
                    em(f"v_xor_b32_e32 v53, {off}, v1")
                    em("v_lshlrev_b32_e32 v53, 2, v53")
                    em(f"ds_bpermute_b32 v54, v53, v{acc}")
                    em("s_waitcnt lgkmcnt(0)")
                    em(f"v_add_f32_e32 v{acc}, v{acc}, v54")
            # 只有 lane0、且 r < nr 时写
            em("v_cmp_eq_u32_e32 vcc, 0, v1")
            em("s_and_saveexec_b64 s[2:3], vcc")
            em(f"v_cmp_lt_u32_e32 vcc, {r}, v3")
            em("s_and_saveexec_b64 s[8:9], vcc")
            # out + m*N*4 + r*4（v14 已含 row0*4）
            em("v_mov_b32_e32 v52, s26")
            em(f"v_mov_b32_e32 v54, {m}")
            em("v_mul_lo_u32 v52, v52, v54")
            em("v_lshlrev_b32_e32 v52, 2, v52")
            em(f"v_add_u32_e32 v52, {4 * r}, v52")
            em("v_mov_b32_e32 v50, v14")
            em("v_add_co_u32_e32 v50, vcc, v50, v52")
            em("v_mov_b32_e32 v51, v15")
            em("v_addc_co_u32_e32 v51, vcc, v51, v4, vcc")
            em(f"global_store_dword v[50:51], v{acc}, off")
            em("s_or_b64 exec, exec, s[8:9]")
            em("s_or_b64 exec, exec, s[2:3]")
    em("L_end:")
    em("s_or_b64 exec, exec, s[0:1]")
    em("s_endpgm")
    return "\n".join(L) + "\n"


def _ref(wq_i4, ws_f16, aq_i4, as_f32, N, K):
    M_ = aq_i4.shape[0]
    w = (wq_i4.reshape(N, K // 128, 128).astype(np.float32)
         * ws_f16.astype(np.float32)[:, :, None]).reshape(N, K)
    a = (aq_i4.reshape(M_, K // 32, 32).astype(np.float32)
         * as_f32.astype(np.float32)[:, :, None]).reshape(M_, K)
    return a @ w.T


def _pack(q):
    return ((q & 0xF)[:, 0::2] | ((q & 0xF)[:, 1::2] << 4)).astype(np.uint8)


def _selftest(N: int = 64, K: int = 2048, m: int = 4, name: str = NAME) -> int:
    rng = np.random.default_rng(11)
    wq_i4 = rng.integers(-8, 8, size=(N, K), dtype=np.int32)
    aq_i4 = rng.integers(-8, 8, size=(m, K), dtype=np.int32)
    ws = rng.uniform(0.001, 0.02, size=(N, K // 128)).astype(np.float16)
    as_ = rng.uniform(0.001, 0.02, size=(m, K // 32)).astype(np.float32)
    wqb = _pack(wq_i4).tobytes()
    aqb = _pack(aq_i4).tobytes()
    ref = _ref(wq_i4, ws, aq_i4, as_, N, K)
    hsaco = build_one(name, gen_asm(m, name), ARGS, kernarg_size=KERNARG_SIZE, tag="r2")
    grid = ((N + NWARP * ROWS - 1) // (NWARP * ROWS)) * 256
    out = run_one(hsaco, name,
                  [{"buffer": "wq"}, {"buffer": "ws"}, {"buffer": "aq"},
                   {"buffer": "as"}, {"buffer": "o"},
                   {"scalar": {"dtype": "u32", "value": N}},
                   {"scalar": {"dtype": "u32", "value": K}}],
                  {"wq": {"dtype": "u8", "values": list(wqb)},
                   "ws": {"dtype": "f16", "values": list(ws.astype(np.float16).reshape(-1))},
                   "aq": {"dtype": "u8", "values": list(aqb)},
                   "as": {"dtype": "f32", "values": list(as_.reshape(-1))},
                   "o": {"dtype": "f32", "values": [0.0] * (m * N)}},
                  grid=grid, workgroup=256)
    got = np.array(out["o"], dtype=np.float32).reshape(m, N)
    err = float(np.abs(got - ref).max() / max(1e-9, float(np.abs(ref).max())))
    ok = err < 1e-4
    print(f"{name} M={m} ROWS={ROWS} N={N} K={K}: relerr {err:.3e} "
          f"{'OK ✔' if ok else 'FAIL ✘'}")
    return 0 if ok else 1


def main() -> int:
    rc = _selftest(m=4, name=NAME)
    rc |= _selftest(N=128, K=5120, m=4, name=NAME)
    rc |= _selftest(N=128, K=5120, m=3, name=M3_NAME)
    return rc


# 供 build_native_kernels.py 收集；M=3/4（小批 GEMV 的两行变体）
KERNELS = [
    (NAME, lambda: gen_asm(4, NAME), ARGS, KERNARG_SIZE),
    (M3_NAME, lambda: gen_asm(3, M3_NAME), ARGS, KERNARG_SIZE),
]


if __name__ == "__main__":
    raise SystemExit(main())
