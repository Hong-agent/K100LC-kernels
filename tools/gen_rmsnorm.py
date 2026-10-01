#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RMSNorm（`rmsnorm_fast_k`）：`y[r] = x[r]/sqrt(mean(x[r]^2)+eps) * w`。

为什么要重写：包里那个 `rmsnorm_k` 是 HIP 编出来的，两趟循环**每个元素一条
load + 一条 `s_waitcnt vmcnt(0)`**——dim=512、64 lane 时是 8 次完整访存往返，
实测单次 15.1 us（其中投递地板 ~7.3 us，真正卡在往返延迟上）。这里是同一套
算法，但把 load 按 4 个一批发（同一 lane 的相邻元素差 64 个 float = 256 字节，
正好落在 `offset:imm13` 里），往返次数降 4 倍。

    rmsnorm_fast_k(float* y, const float* x, const float* w, uint32_t dim,
                   float eps)

grid = rows、workgroup = 64；要求 `dim % 64 == 0`。LDS：归约槽 64×4 B +
广播槽 4 B。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

NAME = "rmsnorm_fast_k"
ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
        scalar_arg(24, 4), scalar_arg(28, 4)]
KERNARG_SIZE = 32
GROUP_SEGMENT = 1024          # 归约 256 B + 广播 4 B，跟着 layernorm_k 取 1024
NB = 4                        # 一趟发几条 load（相邻 256 字节）
STRIDE = 64 * 4               # 相邻元素的字节间隔（64 个 float）


def _tree(L: list[str], reg: str, off: int) -> None:
    """64 lane 的 LDS 树形归约（槽位 = lane id，在 v8 里）。"""
    L += [
        f"v_cmp_gt_u32_e32 vcc, {off}, v0",
        "s_and_saveexec_b64 s[2:3], vcc",
        f"v_mov_b32_e32 v9, {off * 4}",
        "v_add_u32_e32 v9, v9, v8",
        "ds_read_b32 v26, v9",
        "s_waitcnt lgkmcnt(0)",
        f"v_add_f32_e32 {reg}, {reg}, v26",
        f"ds_write_b32 v8, {reg}",
        "s_or_b64 exec, exec, s[2:3]",
        "s_barrier",
    ]


def _broadcast(L: list[str], src: str, slot: int) -> None:
    """lane 0 写槽 `slot`，然后所有 lane 读它（读到 v17）。"""
    L += [
        "v_cmp_eq_u32_e32 vcc, 0, v0",
        "s_and_saveexec_b64 s[2:3], vcc",
        f"v_mov_b32_e32 v9, {slot}",
        f"ds_write_b32 v9, {src}",
        "s_or_b64 exec, exec, s[2:3]",
        "s_barrier",
        f"v_mov_b32_e32 v9, {slot}",
        "ds_read_b32 v17, v9",
        "s_waitcnt lgkmcnt(0)",
    ]


def gen_asm() -> str:
    S = STRIDE
    L: list[str] = [
        ".text", f"k_{NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",   # y
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",   # x
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",  # w
        "s_load_dword s22, s[4:5], 0x18",         # dim
        "s_load_dword s23, s[4:5], 0x1c",         # eps（f32 位模式）
        "s_waitcnt lgkmcnt(0)",
        "s_lshr_b32 s24, s22, 6",                 # n64 = dim/64
        "v_mov_b32_e32 v1, 0",
        "v_mov_b32_e32 v2, s6",                   # row = bid
        "v_mul_lo_u32 v3, v2, s22",
        "v_lshlrev_b32_e32 v3, 2, v3",            # row 的字节偏移
        # x 行指针 / y 行指针
        "v_mov_b32_e32 v4, s18",
        "v_mov_b32_e32 v5, s19",
        "v_add_co_u32_e32 v4, vcc, v4, v3",
        "v_addc_co_u32_e32 v5, vcc, v5, v1, vcc",
        "v_mov_b32_e32 v6, s16",
        "v_mov_b32_e32 v7, s17",
        "v_add_co_u32_e32 v6, vcc, v6, v3",
        "v_addc_co_u32_e32 v7, vcc, v7, v1, vcc",
        "v_lshlrev_b32_e32 v8, 2, v0",            # lane*4（也是 LDS 槽地址）
        "v_mov_b32_e32 v16, 0",                   # sumsq
        "s_mov_b32 s25, 0",                       # j（第几组 64 个元素）
        # ---- 第一趟：Σx²，一次发 NB 条 load ----
        "L_sb:",
        f"s_add_i32 s26, s25, {NB - 1}",
        "s_cmp_lt_u32 s26, s24",
        "s_cbranch_scc0 L_s1",
        "v_mov_b32_e32 v9, s25",
        "v_lshlrev_b32_e32 v9, 8, v9",            # j*256
        "v_add_u32_e32 v9, v9, v8",
        "v_mov_b32_e32 v11, v5",
        "v_add_co_u32_e32 v10, vcc, v4, v9",
        "v_addc_co_u32_e32 v11, vcc, v11, v1, vcc",
        *[f"global_load_dword v{12 + k}, v[10:11], off offset:{S * k}"
          for k in range(NB)],
        "s_waitcnt vmcnt(0)",
        *[f"v_fma_f32 v16, v{12 + k}, v{12 + k}, v16" for k in range(NB)],
        f"s_add_i32 s25, s25, {NB}",
        "s_branch L_sb",
        # 收尾：剩下不到 NB 个
        "L_s1:",
        "s_cmp_lt_u32 s25, s24",
        "s_cbranch_scc0 L_s1_done",
        "v_mov_b32_e32 v9, s25",
        "v_lshlrev_b32_e32 v9, 8, v9",            # j*256
        "v_add_u32_e32 v9, v9, v8",
        "v_mov_b32_e32 v11, v5",
        "v_add_co_u32_e32 v10, vcc, v4, v9",
        "v_addc_co_u32_e32 v11, vcc, v11, v1, vcc",
        "global_load_dword v12, v[10:11], off",
        "s_waitcnt vmcnt(0)",
        "v_fma_f32 v16, v12, v12, v16",
        "s_add_i32 s25, s25, 1",
        "s_branch L_s1",
        "L_s1_done:",
        # ---- 归约 + 广播 ----
        "ds_write_b32 v8, v16",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
    ]
    for off in (32, 16, 8, 4, 2, 1):
        _tree(L, "v16", off)
    _broadcast(L, "v16", 256)
    L += [
        # inv = rsqrt(sumsq/dim + eps)
        "v_cvt_f32_u32_e32 v18, s22",
        "s_nop 0",
        "v_rcp_f32_e32 v18, v18",
        "s_nop 0",
        "v_mul_f32_e32 v17, v17, v18",            # mean(x²)
        "v_mov_b32_e32 v19, s23",
        "v_add_f32_e32 v17, v17, v19",            # + eps
        "s_nop 0",
        "v_rsq_f32_e32 v17, v17",
        "s_nop 0",
        # ---- 第二趟：y = x * inv * w，同样 NB 条一批 ----
        "s_mov_b32 s25, 0",
        "L_nb:",
        f"s_add_i32 s26, s25, {NB - 1}",
        "s_cmp_lt_u32 s26, s24",
        "s_cbranch_scc0 L_n1",
        "v_mov_b32_e32 v9, s25",
        "v_lshlrev_b32_e32 v9, 8, v9",            # j*256
        "v_add_u32_e32 v9, v9, v8",
        "v_mov_b32_e32 v11, v5",
        "v_add_co_u32_e32 v10, vcc, v4, v9",
        "v_addc_co_u32_e32 v11, vcc, v11, v1, vcc",
        "v_mov_b32_e32 v19, s21",
        "v_add_co_u32_e32 v18, vcc, s20, v9",
        "v_addc_co_u32_e32 v19, vcc, v19, v1, vcc",
        "v_mov_b32_e32 v25, v7",
        "v_add_co_u32_e32 v24, vcc, v6, v9",
        "v_addc_co_u32_e32 v25, vcc, v25, v1, vcc",
        *[f"global_load_dword v{12 + k}, v[10:11], off offset:{S * k}"
          for k in range(NB)],
        *[f"global_load_dword v{20 + k}, v[18:19], off offset:{S * k}"
          for k in range(NB)],
        "s_waitcnt vmcnt(0)",
        *[f"v_mul_f32_e32 v{12 + k}, v{12 + k}, v17" for k in range(NB)],
        *[f"v_mul_f32_e32 v{12 + k}, v{12 + k}, v{20 + k}" for k in range(NB)],
        *[f"global_store_dword v[24:25], v{12 + k}, off offset:{S * k}"
          for k in range(NB)],
        f"s_add_i32 s25, s25, {NB}",
        "s_branch L_nb",
        "L_n1:",
        "s_cmp_lt_u32 s25, s24",
        "s_cbranch_scc0 L_n1_done",
        "v_mov_b32_e32 v9, s25",
        "v_lshlrev_b32_e32 v9, 8, v9",            # j*256
        "v_add_u32_e32 v9, v9, v8",
        "v_mov_b32_e32 v11, v5",
        "v_add_co_u32_e32 v10, vcc, v4, v9",
        "v_addc_co_u32_e32 v11, vcc, v11, v1, vcc",
        "v_mov_b32_e32 v19, s21",
        "v_add_co_u32_e32 v18, vcc, s20, v9",
        "v_addc_co_u32_e32 v19, vcc, v19, v1, vcc",
        "v_mov_b32_e32 v25, v7",
        "v_add_co_u32_e32 v24, vcc, v6, v9",
        "v_addc_co_u32_e32 v25, vcc, v25, v1, vcc",
        "global_load_dword v12, v[10:11], off",
        "global_load_dword v20, v[18:19], off",
        "s_waitcnt vmcnt(0)",
        "v_mul_f32_e32 v12, v12, v17",
        "v_mul_f32_e32 v12, v12, v20",
        "global_store_dword v[24:25], v12, off",
        "s_add_i32 s25, s25, 1",
        "s_branch L_n1",
        "L_n1_done:",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


def _ref(x, w, eps):
    x = np.asarray(x, np.float32)
    ms = (x ** 2).mean(axis=1, keepdims=True)        # 每个**行**各自的均方
    return x / np.sqrt(ms + eps) * np.asarray(w, np.float32)


def selftest() -> int:
    hsaco = build_one(NAME, gen_asm(), ARGS, KERNARG_SIZE,
                      group_segment=GROUP_SEGMENT, tag="rmsn")
    rng = np.random.default_rng(20261010)
    rc = 0
    for rows, dim in ((1, 64), (3, 128), (4, 512), (2, 1024), (5, 256), (1, 64 * 5)):
        x = rng.standard_normal((rows, dim)).astype(np.float32) * 2.0
        w = (rng.random(dim).astype(np.float32) + 0.5)
        eps = 1e-6
        o = run_one(hsaco, NAME,
                    [{"buffer": "y"}, {"buffer": "x"}, {"buffer": "w"},
                     {"scalar": {"dtype": "u32", "value": dim}},
                     {"scalar": {"dtype": "f32", "value": eps}}],
                    {"x": {"dtype": "f32", "values": x.reshape(-1).tolist()},
                     "w": {"dtype": "f32", "values": w.tolist()},
                     "y": {"dtype": "f32", "values": [0.0] * (rows * dim)}},
                    grid=rows * 64, workgroup=64)      # `run_one` 的 grid 是总 work-item 数
        got = np.array(o["y"], np.float32).reshape(rows, dim)
        ref = _ref(x, w, eps)
        rel = float(np.abs(got - ref).max() / max(1e-9, float(np.abs(ref).max())))
        ok = rel < 1e-5
        rc |= 0 if ok else 1
        print(f"rows={rows} dim={dim}: max_rel={rel:.2e} {'✔' if ok else '✘'}")
    return rc


def main() -> int:
    rc = selftest()
    print("rmsnorm_fast_k:", "全部一致 ✔" if rc == 0 else "有不一致 ✘")
    return rc


KERNELS = [(NAME, gen_asm, ARGS, KERNARG_SIZE)]

if __name__ == "__main__":
    raise SystemExit(main())
