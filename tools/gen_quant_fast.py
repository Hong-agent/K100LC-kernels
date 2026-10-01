#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多线程/向量化激活量化：`quant_rows_fast_k` + 自检。

替代 `quant_rows_k`（每个线程串行处理一整个 128 元素组，且相邻 lane 的组
相隔 512B → 访存完全不合并）。新内核：

* grid = rows，workgroup = 256（4 warp）；**一个 warp 负责一个 128 元素组**，
  lane 读连续 float2（64 lane × 8B = 512B 连续），访存完全合并；
* 组内 amax 用 wave 内 `ds_bpermute_b32` 蝶形归约（无 barrier）；
* 缩放/量化用与编译器一致的 IEEE 除法序列（bit-exact 于 `x/s`），
  `v_rndne_f32` 取整、`v_med3_i32` 夹到 [-8,7]；
* 每个 lane 出 1 字节（2 个 int4），lanes 0..15 再用 `ds_bpermute_b32`
  每 4 个 lane 拼成 1 个 u32 写回 —— 输出布局与 `quant_rows_k` 完全一致
  （q[row][K/8] 低半字节=偶数 k；sc[g][row] 组优先）。

签名（比 quant_rows_k 多一个 NG = K/G）：
    quant_rows_fast_k(u32* q, float* sc, const float* x,
                      uint32_t K, uint32_t G, uint32_t rows,
                      uint32_t in_stride, uint32_t in_off, uint32_t NG)
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "quant_rows_fast_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
        scalar_arg(24, 4), scalar_arg(28, 4), scalar_arg(32, 4),
        scalar_arg(36, 4), scalar_arg(40, 4), scalar_arg(44, 4)]
KERNARG_SIZE = 48


def _fdiv(L: list[str], dst: str, a: str, b: str) -> None:
    """IEEE float 除法序列（与本机编译器展开一致，逐位等于 a/b）。

    与 probe 验证过的写法一致：t0=scale(b,b,a)、t1=scale(a,b,a)，
    最后 fixup(t, b, a)。
    """
    t0, t1, t2, t3, t4 = "v10", "v11", "v12", "v13", "v14"
    L += [
        f"v_div_scale_f32 {t0}, vcc, {b}, {b}, {a}",
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


def gen_asm() -> str:
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{NAME}:")
    em("s_load_dwordx2 s[16:17], s[4:5], 0x0")     # q
    em("s_load_dwordx2 s[18:19], s[4:5], 0x8")     # sc
    em("s_load_dwordx2 s[20:21], s[4:5], 0x10")    # x
    em("s_load_dword s22, s[4:5], 0x18")           # K
    em("s_load_dword s23, s[4:5], 0x1c")           # G
    em("s_load_dword s24, s[4:5], 0x20")           # rows
    em("s_load_dword s25, s[4:5], 0x24")           # in_stride
    em("s_load_dword s26, s[4:5], 0x28")           # in_off
    em("s_load_dword s27, s[4:5], 0x2c")           # NG = K/G
    em("s_waitcnt lgkmcnt(0)")
    em("v_mov_b32_e32 v30, 0")
    em("v_and_b32_e32 v1, 63, v0")                 # lane
    em("v_lshrrev_b32_e32 v2, 6, v0")              # warp（= 起始组号）
    em("v_lshlrev_b32_e32 v4, 3, v1")              # lane*8 字节
    em("s_mov_b32 s28, s6")                        # row = blockIdx.x
    em("s_mul_i32 s29, s28, s25")
    em("s_add_i32 s29, s29, s26")
    em("s_lshl_b32 s29, s29, 2")                   # x 行基址（字节）
    em("s_lshl_b32 s30, s28, 2")                   # row*4
    em("s_lshr_b32 s31, s22, 1")
    em("s_mul_i32 s31, s31, s28")                  # row*(K/2) 字节
    em("v_mov_b32_e32 v19, 0x40e00000")            # 7.0f
    em("L_loop:")
    em("v_cmp_gt_u32_e32 vcc, s27, v2")            # NG > g ?
    em("s_cbranch_vccz L_end")
    # ---- 读该组的 128 个元素：lane 读连续 float2 ----
    em("v_mul_lo_u32 v28, v2, s23")
    em("v_lshlrev_b32_e32 v28, 2, v28")            # g*G*4
    em("v_add_u32_e32 v28, v28, v4")
    em("v_mov_b32_e32 v7, s29")
    em("v_add_u32_e32 v28, v28, v7")
    em("v_mov_b32_e32 v5, s20")
    em("v_add_co_u32_e32 v5, vcc, v5, v28")
    em("v_mov_b32_e32 v6, s21")
    em("v_addc_co_u32_e32 v6, vcc, v6, v30, vcc")
    em("global_load_dwordx2 v[8:9], v[5:6], off")
    em("s_waitcnt vmcnt(0)")
    em("v_max_f32_e64 v15, |v8|, |v9|")
    # ---- wave 内蝶形归约求 amax（ds_bpermute，无 barrier）----
    for off in (32, 16, 8, 4, 2, 1):
        em(f"v_xor_b32_e32 v16, {off}, v1")
        em("v_lshlrev_b32_e32 v16, 2, v16")
        em("ds_bpermute_b32 v17, v16, v15")
        em("s_waitcnt lgkmcnt(0)")
        em("v_max_f32_e32 v15, v15, v17")
    # ---- s = amax>0 ? amax/7 : 1.0 ----
    _fdiv(L, "v18", "v15", "v19")
    em("v_cmp_lt_f32_e32 vcc, 0, v15")
    em("v_mov_b32_e32 v7, 0x3f800000")            # 1.0f
    # 本机实测：vcc 为真时取第二个源操作数（与编译器的 v_cndmask 用法一致）
    em("v_cndmask_b32_e32 v20, v7, v18, vcc")
    # ---- lane0 写 sc[g*rows + row] ----
    em("v_cmp_eq_u32_e32 vcc, 0, v1")
    em("s_and_saveexec_b64 s[0:1], vcc")
    em("v_mul_lo_u32 v21, v2, s24")
    em("v_lshlrev_b32_e32 v21, 2, v21")
    em("v_mov_b32_e32 v7, s30")
    em("v_add_u32_e32 v21, v21, v7")
    em("v_mov_b32_e32 v22, s18")
    em("v_add_co_u32_e32 v21, vcc, v22, v21")
    em("v_mov_b32_e32 v22, s19")
    em("v_addc_co_u32_e32 v22, vcc, v22, v30, vcc")
    em("global_store_dword v[21:22], v20, off")
    em("s_or_b64 exec, exec, s[0:1]")
    # ---- 量化 + 打包：一个 lane 出 1 字节 ----
    _fdiv(L, "v21", "v8", "v20")
    em("v_rndne_f32_e32 v21, v21")
    em("v_cvt_i32_f32_e32 v21, v21")
    em("v_med3_i32 v21, v21, -8, 7")
    em("v_and_b32_e32 v21, 15, v21")
    _fdiv(L, "v22", "v9", "v20")
    em("v_rndne_f32_e32 v22, v22")
    em("v_cvt_i32_f32_e32 v22, v22")
    em("v_med3_i32 v22, v22, -8, 7")
    em("v_and_b32_e32 v22, 15, v22")
    em("v_lshlrev_b32_e32 v22, 4, v22")
    em("v_or_b32_e32 v22, v21, v22")
    # ---- 拼 u32：lanes 0..15 各要 4 个 lane 的字节 ----
    # 注意：ds_bpermute 只能读「活跃」lane，所以 gather 阶段必须全 wave 参与，
    # 只在最后 store 时把 exec 掩到 lanes 0..15。
    em("v_lshlrev_b32_e32 v23, 4, v1")
    em("ds_bpermute_b32 v24, v23, v22")
    em("v_add_u32_e32 v23, 4, v23")
    em("ds_bpermute_b32 v25, v23, v22")
    em("v_add_u32_e32 v23, 4, v23")
    em("ds_bpermute_b32 v26, v23, v22")
    em("v_add_u32_e32 v23, 4, v23")
    em("ds_bpermute_b32 v27, v23, v22")
    em("s_waitcnt lgkmcnt(0)")
    em("v_and_b32_e32 v24, 0xff, v24")
    em("v_and_b32_e32 v25, 0xff, v25")
    em("v_lshlrev_b32_e32 v25, 8, v25")
    em("v_or_b32_e32 v24, v24, v25")
    em("v_and_b32_e32 v26, 0xff, v26")
    em("v_lshlrev_b32_e32 v26, 16, v26")
    em("v_and_b32_e32 v27, 0xff, v27")
    em("v_lshlrev_b32_e32 v27, 24, v27")
    em("v_or_b32_e32 v26, v26, v27")
    em("v_or_b32_e32 v24, v24, v26")
    em("v_mul_lo_u32 v28, v2, s23")
    em("v_lshrrev_b32_e32 v28, 3, v28")            # g*G/8（u32 个数）
    em("v_lshlrev_b32_e32 v28, 2, v28")            # 字节
    em("v_mov_b32_e32 v7, s31")
    em("v_add_u32_e32 v28, v28, v7")
    em("v_lshlrev_b32_e32 v29, 2, v1")
    em("v_add_u32_e32 v28, v28, v29")
    em("v_cmp_gt_u32_e32 vcc, 16, v1")
    em("s_and_saveexec_b64 s[2:3], vcc")
    em("v_mov_b32_e32 v26, s16")
    em("v_add_co_u32_e32 v26, vcc, v26, v28")
    em("v_mov_b32_e32 v27, s17")
    em("v_addc_co_u32_e32 v27, vcc, v27, v30, vcc")
    em("global_store_dword v[26:27], v24, off")
    em("s_or_b64 exec, exec, s[2:3]")
    em("v_add_u32_e32 v2, 4, v2")
    em("s_branch L_loop")
    em("L_end:")
    em("s_endpgm")
    return "\n".join(L) + "\n"


def _ref(x: np.ndarray, k: int, g: int) -> tuple[np.ndarray, np.ndarray]:
    """主机参考：与 quant_rows_k 一致（scale=amax/7，round-to-nearest-even）。"""
    rows = x.shape[0]
    ng = k // g
    q = np.zeros((rows, k // 8), dtype=np.uint32)
    sc = np.zeros((ng, rows), dtype=np.float32)
    for r in range(rows):
        for gg in range(ng):
            blk = x[r, gg * g:(gg + 1) * g].astype(np.float32)
            amax = np.float32(np.abs(blk).max())
            s = np.float32(amax / np.float32(7.0)) if amax > 0 else np.float32(1.0)
            sc[gg, r] = s
            codes = np.clip(np.rint(blk / s), -8, 7).astype(np.int64)
            for e in range(0, g, 8):
                w = 0
                for j in range(8):
                    w |= int(codes[e + j] & 0xF) << (4 * j)
                q[r, (gg * g + e) // 8] = w
    return q, sc


def main() -> int:
    rng = np.random.default_rng(7)
    rows, k, g = 64, 512, 128
    x = rng.standard_normal((rows, k)).astype(np.float32)
    ng = k // g
    qref, scref = _ref(x, k, g)
    hsaco = build_one(NAME, gen_asm(), ARGS, kernarg_size=KERNARG_SIZE, tag="qfast")
    out = run_one(hsaco, NAME,
                  [{"buffer": "q"}, {"buffer": "sc"}, {"buffer": "x"},
                   {"scalar": {"dtype": "u32", "value": k}},
                   {"scalar": {"dtype": "u32", "value": g}},
                   {"scalar": {"dtype": "u32", "value": rows}},
                   {"scalar": {"dtype": "u32", "value": k}},
                   {"scalar": {"dtype": "u32", "value": 0}},
                   {"scalar": {"dtype": "u32", "value": ng}}],
                  {"q": {"dtype": "u8", "values": [0] * (rows * k // 8 * 4)},
                   "sc": {"dtype": "f32", "values": [0.0] * (ng * rows)},
                   "x": {"dtype": "f32", "values": x.reshape(-1).tolist()}},
                  grid=rows * 256, workgroup=256)   # kernel_lab 的 grid 是总 work-item 数
    q = np.frombuffer(bytes(out["q"]), dtype=np.uint32).reshape(rows, k // 8)
    sc = np.array(out["sc"], dtype=np.float32).reshape(ng, rows)
    qok = bool(np.array_equal(q, qref))
    sok = bool(np.array_equal(sc.view(np.uint32), scref.view(np.uint32)))
    print(f"rows={rows} K={k} G={g}: q {'bit-exact ✔' if qok else '不一致 ✘'}  "
          f"sc {'bit-exact ✔' if sok else '不一致 ✘'}")
    return 0 if (qok and sok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
