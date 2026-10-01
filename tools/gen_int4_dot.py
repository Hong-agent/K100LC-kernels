#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compressed-tensors **INT4（W4A16, group 128）**权重「原生解码 + 点积」融合内核。

目标：满足下列布局的 compressed-tensors INT4 checkpoint（目录用
`--model-dir` / `K100LC_INT4_MODEL` 指定）。打包语义见 `tools/ct_int4.py`：

    weight_packed I32 [N, K/8]    8 个 4bit 码 / 字，低半字节 = 更小的 k
    weight_scale  BF16[N, K/128]  每 128 个 k 一个尺度，无 global scale
    w[n, k] = (码 - 8) * bf16(scale[n, k // 128])      码 ∈ [0,15]，offset-binary

产出两个内核：

| 内核 | 作用 | 显式参数 |
|---|---|---|
| `int4_dequant_k` | 码字 → f32（逐位对账/回退路径） | `(w, s, y, nwords, 64)`，`nwords = 块数*16` |
| `int4_dot_k` | 块原生解码 + 点积，写 `partial[块号]`（主路径） | 同其它 `*_dot_k`，1 张表 = 尺度数组 |

## 为什么直接吃 checkpoint 的字节

checkpoint 里 int4 权重是 0.5 字节/权重，先解成 f32 scratch 再 GEMV 等于把权重
访存放大 8 倍（本包实测：融合路径比「解成 f32 再 `gemv_f32_warp_k`」快 8.7 倍，
其它编码是 12~106 倍，见 `CHANGELOG.md`）。`int4_dot_k` 只读原编码，边解边
点积，和已有的 `iq4nl_dot_k` / `q6k_dot_k` 一个路子。

## 块 × 行 的索引（与其它 `*_dot_k` 完全一致）

一个 work-item 负责「一个 128 权重块 × 一行」，块号
`i = blockIdx.x * workgroup + tid`：

    b    = i % nbpr          # 该行内第几个 128 块（nbpr = K/128）
    row  = i / nbpr
    ew   = row / rows_per_w  # 第几个「权重专家组」（dense 时 rows_per_w = 总行数）
    e    = row / rows_per_exp
    xblk = e * nbpr + b      # 激活里对应的块

设 `local = (row - ew*rows_per_w)*nbpr + b`（专家组内的线性块号）：

    权重地址 = w + ids[ew]*stride      + local*64
    尺度地址 = s + ids[ew]*stride/32   + local*2

尺度的 stride 是权重的 1/32（一个 64 字节权重块配 2 字节 BF16 尺度），所以
**不需要额外的 stride 标量**：尺度数组按同样的块序排列就行。于是 `int4_dot_k`
的参数表就是其它 `*_dot_k` 那份，加上「尺度数组」这一张表（`dot_args(1)`）。
dense 单矩阵传 `ids = [0..N-1]`、`stride = nbpr*64`、`rows_per_exp = rows_per_w = N`；
MoE 传专家表（每组 rows_per_w 行、stride = rows_per_w*nbpr*64）。

## 精度

尺度在整块内是常数，内核在 128 权重块内先累加 `Σ(码-8)*x`、最后才乘一次尺度，
与参考 `Σ((码-8)*尺度)*x` 只差 f32 舍入（实测相对误差 ~1e-7）。

    python3 tools/gen_int4_dot.py                # 合成权重 + 真 checkpoint 双重对账
    python3 tools/gen_int4_dot.py --syn-only     # 只跑合成权重（不需要 checkpoint）
    python3 tools/gen_int4_dot.py --rows 512     # 真权重取多少行
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from ct_int4 import GROUP, Int4Linear, bf16_to_f32, unpack_int4  # noqa: E402
from gen_gemv_qdot import (  # noqa: E402
    REDUCE_ARGS, REDUCE_KERNARG, REDUCE_NAME, DOT_ACC, div_magic,
    dot_args, dot_head, dot_kernarg, dot_tail, dot_xaddr, gen_reduce_asm,
)
from kernel_lab import buffer_arg, build_one, run_one, scalar_arg  # noqa: E402

BLOCK = GROUP                  # 一个块 = 128 个权重
BLOCK_WORDS = BLOCK // 8       # 16 个 I32
BLOCK_BYTES = BLOCK // 2       # 64 字节
BLOCK_SHIFT = 9                # x 侧一个块 = 128 * 4B = 512B → log2


# --------------------------------------------------------------------------
# int4_dot_k：块原生解码 + 点积
# --------------------------------------------------------------------------
INT4_NAME = "int4_dot_k"
INT4_ARGS = dot_args(1)        # 1 张表 = int4 组的 bf16 尺度数组
INT4_KERNARG = dot_kernarg(1)  # 80


def gen_int4_dot_asm() -> str:
    L: list[str] = []
    em = L.append
    em(".text")
    em(f"k_{INT4_NAME}:")
    dot_head(L, 1)                       # s28:29 = 尺度数组基址
    dot_xaddr(L, BLOCK_SHIFT)            # v6:7 = x + xblk*512

    # 权重地址 = w + ids[ew]*stride + 局部块号*64（v120/v124 由 dot_head 算好）
    em("v_mul_lo_u32 v2, v124, 64")
    em("v_add_u32_e32 v2, v2, v120")
    em("v_mov_b32_e32 v4, s16")
    em("v_mov_b32_e32 v5, s17")
    em("v_add_co_u32_e32 v4, vcc, v4, v2")
    em("v_addc_co_u32_e32 v5, vcc, v5, v3, vcc")
    for i in range(4):                   # 16 个 dwordx4 = 64 字节
        em(f"global_load_dwordx4 v[{8 + 4 * i}:{11 + 4 * i}], v[4:5], off "
           f"offset:{16 * i}")

    # 尺度地址 = s + ids[ew]*stride/32 + 局部块号*2（v120 = ids[ew]*stride）
    em("v_lshrrev_b32_e32 v2, 5, v120")
    em("v_lshlrev_b32_e32 v44, 1, v124")
    em("v_add_u32_e32 v2, v2, v44")
    em("v_mov_b32_e32 v44, s28")
    em("v_mov_b32_e32 v45, s29")
    em("v_add_co_u32_e32 v44, vcc, v44, v2")
    em("v_addc_co_u32_e32 v45, vcc, v45, v3, vcc")
    em("global_load_ushort v24, v[44:45], off")
    em("s_waitcnt vmcnt(0)")
    em("v_lshlrev_b32_e32 v25, 16, v24")
    em("v_mov_b32_e32 v26, 0xc1000000")   # -8.0f（offset-binary 的零点）

    # 16 个字 × 8 个 4bit：先 Σ(码-8)*x，再整块乘一次尺度
    #
    # 激活按 4 个字一组批量预载：原来每个字 2 条 dwordx4 后面各跟一条
    # `s_waitcnt vmcnt(0)`（整块 16 次全排空），改成每 4 个字发 8 条载入、
    # 只等一次，访存并行度提高一个量级。XREG 起 v128 这一段没有别的用途。
    XREG = 128
    for g in range(BLOCK_WORDS // 4):
        for i in range(8):
            em(f"global_load_dwordx4 v[{XREG + 4 * i}:{XREG + 4 * i + 3}], v[6:7], "
               f"off offset:{16 * (8 * g + i)}")
        em("s_waitcnt vmcnt(0)")
        for d in range(4 * g, 4 * g + 4):
            lo = XREG + 4 * (2 * d - 8 * g)
            hi = lo + 4
            for j in range(8):
                if j == 0:
                    em(f"v_and_b32_e32 v27, 0x0f, v{8 + d}")
                else:
                    em(f"v_lshrrev_b32_e32 v27, {4 * j}, v{8 + d}")
                    em("v_and_b32_e32 v27, 0x0f, v27")
                em("v_cvt_f32_u32_e32 v27, v27")
                em("v_add_f32_e32 v27, v26, v27")
                xr = lo + j if j < 4 else hi + j - 4
                em(f"v_fma_f32 v{DOT_ACC[j % 4]}, v27, v{xr}, v{DOT_ACC[j % 4]}")
    for r in DOT_ACC:
        em(f"v_mul_f32_e32 v{r}, v{r}, v25")
    dot_tail(L)
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# int4_dequant_k：码字 → f32（一个 work-item 解一个 I32 码字 = 8 个权重）
# --------------------------------------------------------------------------
# 按「码字」而不是「块」切分：第 g 个码字的 8 个 f32 输出正好 32 字节，
# 相邻 work-item 的输入/输出地址都连续 → 访存完全合并（按块切会让一个
# work-item 写 512 字节、相邻线程隔 512 字节，真权重上实测慢 5.4 倍）。
DEQ_NAME = "int4_dequant_k"
DEQ_ARGS = [buffer_arg(0), buffer_arg(8), buffer_arg(16),
            scalar_arg(24, 4), scalar_arg(28, 4)]
DEQ_KERNARG = 32


def gen_int4_dequant_asm() -> str:
    """`(const uint32_t* w, const uint16_t* s, float* y, uint32_t nwords, uint32_t group_size)`

    `nwords = 总块数 * 16`；第 g 个码字属于第 `g >> 4` 个块，写 `y[8g .. 8g+8)`。
    """
    L = [
        ".text", f"k_{DEQ_NAME}:",
        "s_load_dwordx2 s[16:17], s[4:5], 0x0",     # w（packed I32）
        "s_load_dwordx2 s[18:19], s[4:5], 0x8",     # s（bf16 尺度）
        "s_load_dwordx2 s[20:21], s[4:5], 0x10",    # y（f32）
        "s_load_dword s22, s[4:5], 0x18",           # nwords
        "s_load_dword s23, s[4:5], 0x1c",           # group_size(=workgroup)
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v3, 0",
        "v_mov_b32_e32 v1, s6",
        "v_mul_lo_u32 v1, v1, s23",
        "v_add_u32_e32 v1, v0, v1",                 # gid = 码字号
        "v_cmp_gt_u32_e32 vcc, s22, v1",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz L_end",
        # 尺度：块号 = gid >> 4，字节偏移 = 块号 * 2
        "v_lshrrev_b32_e32 v2, 4, v1",
        "v_lshlrev_b32_e32 v2, 1, v2",
        "v_mov_b32_e32 v8, s18",
        "v_mov_b32_e32 v9, s19",
        "v_add_co_u32_e32 v8, vcc, v8, v2",
        "v_addc_co_u32_e32 v9, vcc, v9, v3, vcc",
        # w 指针 = w + gid*4
        "v_lshlrev_b32_e32 v2, 2, v1",
        "v_mov_b32_e32 v4, s16",
        "v_mov_b32_e32 v5, s17",
        "v_add_co_u32_e32 v4, vcc, v4, v2",
        "v_addc_co_u32_e32 v5, vcc, v5, v3, vcc",
        # y 指针 = y + gid*32
        "v_lshlrev_b32_e32 v2, 5, v1",
        "v_mov_b32_e32 v6, s20",
        "v_mov_b32_e32 v7, s21",
        "v_add_co_u32_e32 v6, vcc, v6, v2",
        "v_addc_co_u32_e32 v7, vcc, v7, v3, vcc",
        "global_load_dword v11, v[4:5], off",
        "global_load_ushort v10, v[8:9], off",
        "s_waitcnt vmcnt(0)",
        "v_lshlrev_b32_e32 v10, 16, v10",           # bf16 → f32
        "v_mov_b32_e32 v20, 0xc1000000",            # -8.0f
    ]
    for j in range(8):
        if j == 0:
            L.append("v_and_b32_e32 v21, 0x0f, v11")
        else:
            L.append(f"v_lshrrev_b32_e32 v21, {4 * j}, v11")
            L.append("v_and_b32_e32 v21, 0x0f, v21")
        L.append("v_cvt_f32_u32_e32 v21, v21")
        L.append("v_add_f32_e32 v21, v20, v21")
        L.append(f"v_mul_f32_e32 v{12 + j}, v21, v10")
    L += [
        "global_store_dwordx4 v[6:7], v[12:15], off",
        "global_store_dwordx4 v[6:7], v[16:19], off offset:16",
        "L_end:",
        "s_or_b64 exec, exec, s[0:1]",
        "s_endpgm",
    ]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# 主机侧参考 + 启动参数
# --------------------------------------------------------------------------
def ref_dot(wbuf: np.ndarray, sbuf: np.ndarray, x: np.ndarray, nrows: int,
            nbpr: int, ids: list[int] | None = None,
            rows_per_exp: int | None = None,
            rows_per_w: int | None = None) -> np.ndarray:
    """参考 `partial[nrows*nbpr]`（逐字复刻内核的索引：见模块 docstring）。

    `wbuf [bufrows, nbpr*16] u32`、`sbuf [bufrows, nbpr] u16`、`x [xrows, nbpr*128]`。
    dense 时传 `ids=None`（等价于 ids[0]=0、rows_per_w=nrows）。
    """
    rows_per_exp = nrows if rows_per_exp is None else rows_per_exp
    rows_per_w = nrows if rows_per_w is None else rows_per_w
    vals = unpack_int4(wbuf).astype(np.float32).reshape(wbuf.shape[0], nbpr, BLOCK)
    sc = bf16_to_f32(sbuf).astype(np.float32).reshape(sbuf.shape[0], nbpr)
    xb = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, nbpr, BLOCK)
    out = np.empty(nrows * nbpr, dtype=np.float32)
    for row in range(nrows):
        ew, er = divmod(row, rows_per_w)
        e = row // rows_per_exp
        src = row if ids is None else ids[ew] * rows_per_w + er
        for b in range(nbpr):
            term = (vals[src, b] * sc[src, b]).astype(np.float32) * xb[e, b]
            out[row * nbpr + b] = np.float32(term.sum(dtype=np.float32))
    return out


def dense_params(nrows: int, nbpr: int, workgroup: int = 64) -> dict:
    """dense 单矩阵的启动参数（ids = 0..N-1、stride = 每行字节数）。"""
    nblocks = nrows * nbpr
    return {
        "nblocks": nblocks,
        "workgroup": workgroup,
        "nbpr": nbpr,
        "magic_nbpr": div_magic(nbpr, nblocks),
        "rows_per_exp": nrows,
        "magic_rpe": div_magic(nrows, nrows),
        "ids": list(range(nrows)),
        "stride": nbpr * BLOCK_BYTES,
        "rows_per_w": nrows,
        "magic_rpw": div_magic(nrows, nrows),
    }


def _launch_args(p: dict) -> list[dict]:
    return [
        {"buffer": "w"}, {"buffer": "x"}, {"buffer": "partial"},
        {"scalar": {"dtype": "u32", "value": p["nblocks"]}},
        {"scalar": {"dtype": "u32", "value": p["workgroup"]}},
        {"scalar": {"dtype": "u32", "value": p["nbpr"]}},
        {"scalar": {"dtype": "u32", "value": p["magic_nbpr"]}},
        {"scalar": {"dtype": "u32", "value": p["rows_per_exp"]}},
        {"scalar": {"dtype": "u32", "value": p["magic_rpe"]}},
        {"buffer": "scale"},
        {"buffer": "ids"},
        {"scalar": {"dtype": "u32", "value": p["stride"]}},
        {"scalar": {"dtype": "u32", "value": p["rows_per_w"]}},
        {"scalar": {"dtype": "u32", "value": p["magic_rpw"]}},
    ]


_HSACO_CACHE: dict[str, tuple[pathlib.Path, pathlib.Path, pathlib.Path]] = {}


def build_int4_hsaco(tag: str = "int4"):
    """把两个新内核 + `reduce_blocks_k` 打到 build/lab 下。

    返回 `(dot_hsaco, dequant_hsaco, reduce_hsaco)`；同一个 tag 只构建一次。
    """
    if tag in _HSACO_CACHE:
        return _HSACO_CACHE[tag]
    dot = build_one(INT4_NAME, gen_int4_dot_asm(), INT4_ARGS,
                    kernarg_size=INT4_KERNARG, tag=tag)
    deq = build_one(DEQ_NAME, gen_int4_dequant_asm(), DEQ_ARGS,
                    kernarg_size=DEQ_KERNARG, tag=tag)
    red = build_one(REDUCE_NAME, gen_reduce_asm(), REDUCE_ARGS,
                    kernarg_size=REDUCE_KERNARG, tag=tag)
    _HSACO_CACHE[tag] = (dot, deq, red)
    return dot, deq, red


# --------------------------------------------------------------------------
# 自检 1：合成权重
# --------------------------------------------------------------------------
def _random_bf16(rng, shape, lo=1e-3, hi=5e-2) -> np.ndarray:
    """随机但可精确表示成 bf16 的尺度（截断到 bf16）。"""
    f = rng.uniform(lo, hi, size=shape).astype(np.float32)
    bits = f.view(np.uint32) & np.uint32(0xFFFF0000)
    return (bits >> 16).astype(np.uint16)


def selftest_synthetic() -> int:
    rc = 0
    rng = np.random.default_rng(20261001)

    # ---- int4_dequant_k ----
    nblocks = 512
    packed = rng.integers(0, 1 << 32, size=nblocks * BLOCK_WORDS, dtype=np.uint64)
    packed = packed.astype(np.uint32)
    scales = _random_bf16(rng, (nblocks,))
    ref = (unpack_int4(packed.reshape(nblocks, BLOCK_WORDS)).astype(np.float32)
           * bf16_to_f32(scales).astype(np.float32)[:, None].repeat(BLOCK, axis=1))
    _dot, deq, _red = build_int4_hsaco()
    nwords = nblocks * BLOCK_WORDS
    out = run_one(deq, DEQ_NAME,
                  [{"buffer": "w"}, {"buffer": "s"}, {"buffer": "y"},
                   {"scalar": {"dtype": "u32", "value": nwords}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"w": {"dtype": "u8", "values": list(packed.tobytes())},
                   "s": {"dtype": "u8", "values": list(scales.tobytes())},
                   "y": {"dtype": "f32", "values": [0.0] * (nblocks * BLOCK)}},
                  grid=nwords, workgroup=64)
    got = np.array(out["y"], dtype=np.float32).reshape(nblocks, BLOCK)
    dmax = float(np.abs(got - ref).max())
    print(f"int4_dequant_k: nblocks={nblocks} max_abs={dmax:.3e}")
    rc |= 0 if dmax == 0.0 else 1
    print("INT4 原生解码 vs 参考：", "bit-exact ✔" if dmax == 0.0 else "不一致 ✘")

    # ---- int4_dot_k：dense + 专家组两种索引 ----
    rc |= _selftest_dot_case(rng, "dense", nrows=16, k=640,
                             rows_per_exp=None, rows_per_w=None, n_ids=16)
    rc |= _selftest_dot_case(rng, "专家", nrows=6, k=512, rows_per_exp=3,
                             rows_per_w=3, n_ids=2, extra_rows=4)
    return rc


def _selftest_dot_case(rng, label: str, nrows: int, k: int,
                       rows_per_exp: int | None, rows_per_w: int | None,
                       n_ids: int, extra_rows: int = 0) -> int:
    """`extra_rows` 用来造「权重缓冲里比 x 多的行」（专家表只在其中挑 nrows 行）。"""
    nbpr = k // BLOCK
    buf_rows = nrows + extra_rows
    packed = rng.integers(0, 1 << 32, size=buf_rows * nbpr * BLOCK_WORDS,
                          dtype=np.uint64).astype(np.uint32)
    packed = packed.reshape(buf_rows, nbpr * BLOCK_WORDS)
    scales = _random_bf16(rng, (buf_rows, nbpr))
    xrows = (nrows // rows_per_exp) if rows_per_exp else nrows
    x = rng.standard_normal((xrows, k)).astype(np.float32)

    if n_ids == buf_rows:                       # dense：一行一个组
        ids, stride_blocks = list(range(buf_rows)), nbpr
        p = dense_params(nrows, nbpr)
        p.update(ids=ids, stride=stride_blocks * BLOCK_BYTES,
                 rows_per_exp=nrows, magic_rpe=div_magic(nrows, nrows),
                 rows_per_w=nrows, magic_rpw=div_magic(nrows, nrows))
    else:                                       # 专家：每组 rows_per_w 行
        ids = list(range(n_ids))
        stride_blocks = rows_per_w * nbpr
        p = {
            "nblocks": nrows * nbpr, "workgroup": 64, "nbpr": nbpr,
            "magic_nbpr": div_magic(nbpr, nrows * nbpr),
            "rows_per_exp": rows_per_exp,
            "magic_rpe": div_magic(rows_per_exp, nrows),
            "ids": ids, "stride": stride_blocks * BLOCK_BYTES,
            "rows_per_w": rows_per_w,
            "magic_rpw": div_magic(rows_per_w, nrows),
        }
    ref = ref_dot(packed, scales, x, nrows, nbpr, ids=ids,
                  rows_per_exp=p["rows_per_exp"], rows_per_w=p["rows_per_w"])

    dot, _deq, red = build_int4_hsaco()
    out = run_one(dot, INT4_NAME, _launch_args(p),
                  {"w": {"dtype": "u8", "values": list(packed.tobytes())},
                   "x": {"dtype": "f32", "values": x.reshape(-1).tolist()},
                   "scale": {"dtype": "u8", "values": list(scales.tobytes())},
                   "partial": {"dtype": "f32", "values": [0.0] * p["nblocks"]},
                   "ids": {"dtype": "u32", "values": ids}},
                  grid=p["nblocks"], workgroup=64)
    got = np.array(out["partial"], dtype=np.float32)
    scale = max(1.0, float(np.abs(ref).max()))
    dmax = float(np.abs(got - ref).max())
    rel = dmax / scale
    print(f"int4_dot_k[{label}]: nblocks={p['nblocks']} nbpr={nbpr} "
          f"max_abs={dmax:.3e} (相对 {rel:.2e})")
    ok = rel < 1e-5
    print(f"  融合点积[{label}]：", "一致 ✔" if ok else "不一致 ✘")

    # 顺带把 reduce_blocks_k 也过一遍
    ref_y = ref.reshape(nrows, nbpr).sum(axis=1)
    out2 = run_one(red, REDUCE_NAME,
                   [{"buffer": "partial"}, {"buffer": "y"},
                    {"scalar": {"dtype": "u32", "value": nrows}},
                    {"scalar": {"dtype": "u32", "value": nbpr}}],
                   {"partial": {"dtype": "f32", "values": list(got)},
                    "y": {"dtype": "f32", "values": [0.0] * nrows}},
                   grid=nrows, workgroup=64)
    y = np.array(out2["y"], dtype=np.float32)
    ymax = float(np.abs(y - ref_y).max())
    yscale = max(1.0, float(np.abs(ref_y).max()))
    print(f"  reduce: max_abs={ymax:.3e} (相对 {ymax / yscale:.2e})")
    ok &= ymax / yscale < 1e-5
    return 0 if ok else 1


# --------------------------------------------------------------------------
# 自检 2：真 checkpoint（可选，由 --model-dir 指定）
# --------------------------------------------------------------------------
def selftest_real(prefix: str, nrows: int, model_dir) -> int:
    lin = Int4Linear(prefix, model_dir)
    nrows = min(nrows, lin.n)
    nbpr = lin.nbpr
    print(f"{prefix}: N={lin.n} K={lin.k} nbpr={nbpr}（取前 {nrows} 行）")
    packed = np.frombuffer(lin.packed_rows(0, nrows), dtype=np.uint32).reshape(nrows, -1)
    scales = np.frombuffer(lin.scale_rows(0, nrows), dtype=np.uint16).reshape(nrows, nbpr)
    x = np.random.default_rng(7).standard_normal(lin.k).astype(np.float32)
    ref_w = lin.dequant_rows(0, nrows)
    ref_y = ref_w @ x
    ref_partial = ref_dot(packed, scales, x.reshape(1, -1), nrows, nbpr)

    dot, deq, red = build_int4_hsaco()
    nblocks = nrows * nbpr

    # (a) 整块解码：和 checkpoint 参考解码逐位比
    nwords = nblocks * BLOCK_WORDS
    out = run_one(deq, DEQ_NAME,
                  [{"buffer": "w"}, {"buffer": "s"}, {"buffer": "y"},
                   {"scalar": {"dtype": "u32", "value": nwords}},
                   {"scalar": {"dtype": "u32", "value": 64}}],
                  {"w": {"dtype": "u8", "values": list(packed.tobytes())},
                   "s": {"dtype": "u8", "values": list(scales.tobytes())},
                   "y": {"dtype": "f32", "values": [0.0] * nblocks * BLOCK}},
                  grid=nwords, workgroup=64)
    got_w = np.array(out["y"], dtype=np.float32).reshape(nrows, lin.k)
    dmax = float(np.abs(got_w - ref_w).max())
    print(f"int4_dequant_k 真权重：max_abs={dmax:.3e}",
          "bit-exact ✔" if dmax == 0.0 else "✘")

    # (b) 融合点积 + 归约：和 f32 参考 GEMV 比
    p = dense_params(nrows, nbpr)
    out2 = run_one(dot, INT4_NAME, _launch_args(p),
                   {"w": {"dtype": "u8", "values": list(packed.tobytes())},
                    "x": {"dtype": "f32", "values": x.tolist()},
                    "scale": {"dtype": "u8", "values": list(scales.tobytes())},
                    "partial": {"dtype": "f32", "values": [0.0] * nblocks},
                    "ids": {"dtype": "u32", "values": p["ids"]}},
                   grid=nblocks, workgroup=64)
    partial = np.array(out2["partial"], dtype=np.float32)
    pmax = float(np.abs(partial - ref_partial).max())
    pscale = max(1.0, float(np.abs(ref_partial).max()))
    out3 = run_one(red, REDUCE_NAME,
                   [{"buffer": "partial"}, {"buffer": "y"},
                    {"scalar": {"dtype": "u32", "value": nrows}},
                    {"scalar": {"dtype": "u32", "value": nbpr}}],
                   {"partial": {"dtype": "f32", "values": list(partial)},
                    "y": {"dtype": "f32", "values": [0.0] * nrows}},
                   grid=nrows, workgroup=64)
    y = np.array(out3["y"], dtype=np.float32)
    ymax = float(np.abs(y - ref_partial.reshape(nrows, nbpr).sum(axis=1)).max())
    yscale = max(1.0, float(np.abs(ref_partial).max()))
    blas_max = float(np.abs(y - ref_y).max())
    print(f"int4_dot_k  真权重：partial max_abs={pmax:.3e} (相对 {pmax/pscale:.2e})")
    print(f"reduce_blocks_k    ：y max_abs={ymax:.3e} (相对 {ymax/yscale:.2e})")
    print(f"对比 f32 矩阵乘参考  ：max_abs={blas_max:.3e} "
          f"(相对 {blas_max/max(1.0, float(np.abs(ref_y).max())):.2e})")
    ok = (dmax == 0.0 and pmax / pscale < 1e-5 and ymax / yscale < 1e-5
          and blas_max / max(1.0, float(np.abs(ref_y).max())) < 1e-4)
    print("真 checkpoint 对账：", "一致 ✔" if ok else "不一致 ✘")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="model.language_model.layers.3.mlp.gate_proj")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--syn-only", action="store_true")
    args = ap.parse_args()

    rc = selftest_synthetic()
    if args.syn_only:
        return rc
    from ct_int4 import DEFAULT_MODEL_DIR
    model_dir = pathlib.Path(args.model_dir) if args.model_dir else DEFAULT_MODEL_DIR
    if not (model_dir / "model.safetensors").is_file():
        print(f"（跳过真 checkpoint：{model_dir} 不在）")
        return rc
    rc |= selftest_real(args.prefix, args.rows, model_dir)
    return rc


# 供 build_native_kernels.py 收集（name, gen_asm, args, kernarg_size）
KERNELS = [
    (INT4_NAME, gen_int4_dot_asm, INT4_ARGS, INT4_KERNARG),
    (DEQ_NAME, gen_int4_dequant_asm, DEQ_ARGS, DEQ_KERNARG),
]


if __name__ == "__main__":
    raise SystemExit(main())
