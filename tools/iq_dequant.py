#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GGUF 全部原生编码的**参考解码器**（纯 NumPy，无第三方依赖）。

定位：运行时（gfx926 内核）按**原编码原生解码**、不做任何重量化；本文件是那些
内核的"金标准"——同一块喂两边应逐元素一致（只允许激活量化的差异）。块布局与
反量化过程照抄 llama.cpp / ggml 的 `ggml-common.h` + `ggml-quants.c`（MIT），
码本由 `tools/iq_tables.py` 提供（`tools/extract_iq_tables.py` 生成）。

覆盖本包已支持的全部 GGUF 编码：

| 编码 | 块（元素/字节） | 布局要点 |
|---|---|---|
| `Q2_0` | 64 / 18 | 2 bit 线性码，值 `(码-1)×d`，4 码/字节低位在前 |
| `IQ4_NL` | 32 / 18 | 16 项非线性码本，低半字节 = 元素 0..15 |
| `IQ4_XS` | 256 / 136 | 4 bit + 每 32 元素 6 bit 尺度（`ls-32`） |
| `Q6_K` | 256 / 210 | `ql` 4 bit + `qh` 2 bit，减 32；16 元素一组 int8 尺度 |
| `Q5_K` | 256 / 176 | 4 bit + 第 5 位（`qh`）+ 6 bit 尺度/偏移 |
| `Q4_K` | 256 / 144 | 4 bit + 6 bit 尺度/偏移 |
| `IQ3_XXS` | 256 / 98 | 4 元素网格（256×u32）+ 7 bit 符号 + 每 32 元素 4 bit 尺度 |
| `IQ2_S` | 256 / 82 | 8 元素网格（1024×u64）+ 符号字节 + 4 bit 尺度 |
| `IQ3_S` | 256 / 110 | 4 元素网格（512×u32）+ 符号 + 每 64 元素 2×4 bit 尺度 |
| `Q8_0` | 32 / 34 | int8 + f16 尺度 |
| `Q4_0` | 32 / 18 | 4 bit + f16 尺度，值 `(码-8)×d` |
| `BF16` / `F16` / `F32` | 1 / 2,2,4 | 直读 |

    python3 tools/iq_dequant.py --selftest     # 每种编码的结构自检
    python3 tools/iq_dequant.py --info         # 支持的编码与位宽

注意：`IQ3_XXS/IQ2_S/IQ3_S` 的网格值是**无符号幅值**，符号来自单独的符号位；
`IQ2_S` 的符号字节就嵌在 `qs[64]` 的后 32 字节里（ggml 源码里 `signs = qs + 32`）。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from iq_tables import (  # noqa: E402
    iq2s_grid, iq3s_grid, iq3xxs_grid, kmask_iq2xs, ksigns_iq2xs, kvalues_iq4nl,
)

GEOMETRY: dict[str, tuple[int, int]] = {
    "Q2_0": (64, 18), "IQ4_NL": (32, 18), "IQ4_XS": (256, 136),
    "Q6_K": (256, 210), "Q5_K": (256, 176), "Q4_K": (256, 144),
    "IQ3_XXS": (256, 98), "IQ2_S": (256, 82), "IQ3_S": (256, 110),
    "Q8_0": (32, 34), "Q4_0": (32, 18),
    "BF16": (1, 2), "F16": (1, 2), "F32": (1, 4),
}

_KMASK = [int(v) for v in kmask_iq2xs]                  # 1,2,4,…,128
_KSIGNS = [int(v) for v in ksigns_iq2xs]
_IQ4NL = np.array(kvalues_iq4nl, dtype=np.int8)


def _grid_bytes(table, item_bytes: int) -> np.ndarray:
    """整型网格表（u32/u64）→ (n, item_bytes) 的无符号幅值字节表（小端）。"""
    arr = np.array(table, dtype=np.uint64).reshape(-1, 1)
    shifts = (np.arange(item_bytes, dtype=np.uint64) * 8).reshape(1, -1)
    return ((arr >> shifts) & 0xFF).astype(np.uint8)


_G_IQ3XXS = _grid_bytes(iq3xxs_grid, 4)
_G_IQ3S = _grid_bytes(iq3s_grid, 4)
_G_IQ2S = _grid_bytes(iq2s_grid, 8)


def _f16(buf: np.ndarray, off: int) -> np.ndarray:
    raw = np.ascontiguousarray(buf[:, off:off + 2])
    return raw.view(np.float16).reshape(-1).astype(np.float32)


def _blocks(raw: bytes, type_name: str) -> np.ndarray:
    qk, bb = GEOMETRY[type_name]
    if len(raw) % bb:
        raise ValueError(f"{type_name}: {len(raw)} 字节不是块大小 {bb} 的整数倍")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, bb)


# ------------------------------------------------------------------ 线性/码本类
def dequant_q2_0(raw: bytes) -> np.ndarray:
    buf = _blocks(raw, "Q2_0")
    d = _f16(buf, 0)[:, None]
    qs = buf[:, 2:18].reshape(-1, 16, 1)
    shifts = (np.arange(4) * 2).reshape(1, 1, 4)
    codes = ((qs >> shifts) & 3).reshape(-1, 64).astype(np.float32)
    return ((codes - 1.0) * d).reshape(-1)


def dequant_iq4_nl(raw: bytes) -> np.ndarray:
    buf = _blocks(raw, "IQ4_NL")
    d = _f16(buf, 0)[:, None]
    qs = buf[:, 2:18]
    codes = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.uint8)
    return (_IQ4NL[codes].astype(np.float32) * d).reshape(-1)


def dequant_iq4_xs(raw: bytes) -> np.ndarray:
    buf = _blocks(raw, "IQ4_XS")
    out = np.empty((buf.shape[0], 256), dtype=np.float32)
    for i in range(buf.shape[0]):
        b = buf[i]
        d = float(np.frombuffer(b[0:2].tobytes(), dtype=np.float16)[0])
        scales_h = int(np.frombuffer(b[2:4].tobytes(), dtype=np.uint16)[0])
        scales_l = b[4:8]
        qs = b[8:136]
        for ib in range(8):
            ls = int((scales_l[ib // 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4)
            dl = d * (ls - 32)
            q = qs[ib * 16:(ib + 1) * 16]
            out[i, ib * 32:ib * 32 + 16] = dl * _IQ4NL[(q & 0x0F).astype(np.uint8)]
            out[i, ib * 32 + 16:ib * 32 + 32] = dl * _IQ4NL[(q >> 4).astype(np.uint8)]
    return out.reshape(-1)


def dequant_q8_0(raw: bytes) -> np.ndarray:
    buf = _blocks(raw, "Q8_0")
    d = _f16(buf, 0)[:, None]
    q = np.ascontiguousarray(buf[:, 2:34]).view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)


def dequant_q4_0(raw: bytes) -> np.ndarray:
    """Q4_0：32 元素 / 18 字节；低半字节 = 元素 0..15，高半字节 = 16..31，值 = d*(q-8)。"""
    buf = _blocks(raw, "Q4_0")
    d = _f16(buf, 0)[:, None]
    qs = buf[:, 2:18].astype(np.int32)
    out = np.empty((buf.shape[0], 32), dtype=np.float32)
    out[:, 0:16] = (qs & 0x0F) - 8
    out[:, 16:32] = (qs >> 4) - 8
    return (out * d).reshape(-1)


# ------------------------------------------------------------------ K-quant
def _scale_min_k4(j: int, scales: np.ndarray) -> tuple[int, int]:
    if j < 4:
        return int(scales[j] & 63), int(scales[j + 4] & 63)
    return (int((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)),
            int((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)))


def dequant_q4_k(raw: bytes) -> np.ndarray:
    """Q4_K：256 元素 = 4×64；每 64 元素两组 `d*sc` 与 `dmin*m`（低半字节在前）。"""
    buf = _blocks(raw, "Q4_K")
    out = np.empty((buf.shape[0], 256), dtype=np.float32)
    for i in range(buf.shape[0]):
        b = buf[i]
        d = float(np.frombuffer(b[0:2].tobytes(), dtype=np.float16)[0])
        dmin = float(np.frombuffer(b[2:4].tobytes(), dtype=np.float16)[0])
        scales = b[4:16]
        qs = b[16:144]
        for s in range(4):                                  # 每 64 元素
            sc0, m0 = _scale_min_k4(2 * s, scales)
            sc1, m1 = _scale_min_k4(2 * s + 1, scales)
            q = qs[s * 32:(s + 1) * 32]
            out[i, s * 64:s * 64 + 32] = d * sc0 * (q & 0x0F) - dmin * m0
            out[i, s * 64 + 32:s * 64 + 64] = d * sc1 * (q >> 4) - dmin * m1
    return out.reshape(-1)


def dequant_q5_k(raw: bytes) -> np.ndarray:
    """Q5_K：同 Q4_K 再加第 5 位。

    注意 `qh` 是 32 字节、**整块共用**：第 s 个 64 元素组的第 5 位取
    `qh[l]` 的 bit(2s) / bit(2s+1)，不是每组换 8 字节（对照 ggml
    `dequantize_row_q5_K` 的 `u1/u2` 左移）。
    """
    buf = _blocks(raw, "Q5_K")
    out = np.empty((buf.shape[0], 256), dtype=np.float32)
    for i in range(buf.shape[0]):
        b = buf[i]
        d = float(np.frombuffer(b[0:2].tobytes(), dtype=np.float16)[0])
        dmin = float(np.frombuffer(b[2:4].tobytes(), dtype=np.float16)[0])
        scales = b[4:16]
        qh = b[16:48]
        qs = b[48:176]
        for s in range(4):
            sc0, m0 = _scale_min_k4(2 * s, scales)
            sc1, m1 = _scale_min_k4(2 * s + 1, scales)
            q = qs[s * 32:(s + 1) * 32]
            u1, u2 = 1 << (2 * s), 2 << (2 * s)
            lo = (q & 0x0F).astype(np.int32) + np.where(qh & u1, 16, 0)
            hi = (q >> 4).astype(np.int32) + np.where(qh & u2, 16, 0)
            out[i, s * 64:s * 64 + 32] = d * sc0 * lo - dmin * m0
            out[i, s * 64 + 32:s * 64 + 64] = d * sc1 * hi - dmin * m1
    return out.reshape(-1)


def dequant_q6_k(raw: bytes) -> np.ndarray:
    """Q6_K：256 元素 = 2×128；`ql` 低 4 位 + `qh` 高 2 位，减 32；16 元素一组 int8 尺度。"""
    buf = _blocks(raw, "Q6_K")
    out = np.empty((buf.shape[0], 256), dtype=np.float32)
    for i in range(buf.shape[0]):
        b = buf[i]
        d = float(np.frombuffer(b[208:210].tobytes(), dtype=np.float16)[0])
        sc = np.frombuffer(b[192:208].tobytes(), dtype=np.int8).astype(np.float32)
        for half in range(2):
            ql = b[half * 64:half * 64 + 64].astype(np.int32)
            qh = b[128 + half * 32:128 + half * 32 + 32].astype(np.int32)
            s = sc[half * 8:(half + 1) * 8]
            base = half * 128
            for l in range(32):
                is_ = l // 16
                q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
                out[i, base + l] = d * s[is_ + 0] * q1
                out[i, base + 32 + l] = d * s[is_ + 2] * q2
                out[i, base + 64 + l] = d * s[is_ + 4] * q3
                out[i, base + 96 + l] = d * s[is_ + 6] * q4
    return out.reshape(-1)


def dequant_q6_k_fast(raw: bytes) -> np.ndarray:
    """Q6_K 的向量化版本（语义与 dequant_q6_k 完全一致，供整矩阵解码用）。

    lm_head 是 2560×248320 ≈ 6.4 亿个元素，逐块 Python 循环要几分钟；
    这里把「按块循环」换成按 l 维一次性向量化。
    """
    buf = _blocks(raw, "Q6_K")                     # [B, 210]
    b = buf.astype(np.int32)
    d = _f16(buf, 208)[:, None]                    # [B,1]
    sc = np.ascontiguousarray(buf[:, 192:208]).view(np.int8).astype(np.int32)  # [B,16]
    out = np.empty((buf.shape[0], 256), dtype=np.float32)
    for half in range(2):
        ql = b[:, half * 64:half * 64 + 64]        # [B,64]
        qh = b[:, 128 + half * 32:128 + half * 32 + 32]  # [B,32]
        s = sc[:, half * 8:(half + 1) * 8]         # [B,8]
        q1 = ((ql[:, :32] & 0xF) | (((qh >> 0) & 3) << 4)) - 32
        q2 = ((ql[:, 32:] & 0xF) | (((qh >> 2) & 3) << 4)) - 32
        q3 = ((ql[:, :32] >> 4) | (((qh >> 4) & 3) << 4)) - 32
        q4 = ((ql[:, 32:] >> 4) | (((qh >> 6) & 3) << 4)) - 32
        base = half * 128
        # l 0..15 -> s[0], 16..31 -> s[1]（其余同理，步长 2）
        for lo, qq, idx in ((0, q1, 0), (32, q2, 2), (64, q3, 4), (96, q4, 6)):
            sc_lo = s[:, idx][:, None].astype(np.float32)
            sc_hi = s[:, idx + 1][:, None].astype(np.float32)
            out[:, base + lo:base + lo + 16] = (
                qq[:, :16].astype(np.float32) * (d * sc_lo))
            out[:, base + lo + 16:base + lo + 32] = (
                qq[:, 16:].astype(np.float32) * (d * sc_hi))
    return out.reshape(-1)


def dequant_iq4_nl_fast(raw: bytes) -> np.ndarray:
    """IQ4_NL 向量化版（与 dequant_iq4_nl 逐位一致）。"""
    buf = _blocks(raw, "IQ4_NL")
    d = _f16(buf, 0)[:, None]
    qs = buf[:, 2:18]
    lo = _IQ4NL[qs & 0x0F].astype(np.float32)
    hi = _IQ4NL[qs >> 4].astype(np.float32)
    return (np.concatenate([lo, hi], axis=1) * d).reshape(-1)


def dequant_q8_0_fast(raw: bytes) -> np.ndarray:
    buf = _blocks(raw, "Q8_0")
    d = _f16(buf, 0)[:, None]
    q = np.ascontiguousarray(buf[:, 2:34]).view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)


def dequant_q4_0_fast(raw: bytes) -> np.ndarray:
    buf = _blocks(raw, "Q4_0")
    d = _f16(buf, 0)[:, None]
    qs = buf[:, 2:18].astype(np.int32)
    out = np.empty((buf.shape[0], 32), dtype=np.float32)
    out[:, 0:16] = (qs & 0x0F) - 8
    out[:, 16:32] = (qs >> 4) - 8
    return (out * d).reshape(-1)


def _scale_min_k4_vec(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[B,12] → (sc[B,8], m[B,8])，与 _scale_min_k4 等价。"""
    B = scales.shape[0]
    sc = np.empty((B, 8), dtype=np.int32)
    m = np.empty((B, 8), dtype=np.int32)
    for j in range(8):
        if j < 4:
            sc[:, j] = scales[:, j] & 63
            m[:, j] = scales[:, j + 4] & 63
        else:
            sc[:, j] = (scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4)
            m[:, j] = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    return sc, m


def dequant_q4_k_fast(raw: bytes) -> np.ndarray:
    buf = _blocks(raw, "Q4_K")
    B = buf.shape[0]
    d = _f16(buf, 0)[:, None]
    dmin = _f16(buf, 2)[:, None]
    sc, m = _scale_min_k4_vec(buf[:, 4:16].astype(np.int32))
    qs = buf[:, 16:144].astype(np.int32)
    out = np.empty((B, 256), dtype=np.float32)
    for s in range(4):
        q = qs[:, s * 32:(s + 1) * 32]
        out[:, s * 64:s * 64 + 32] = (d * sc[:, 2 * s][:, None] * (q & 0xF)
                                      - dmin * m[:, 2 * s][:, None])
        out[:, s * 64 + 32:s * 64 + 64] = (d * sc[:, 2 * s + 1][:, None] * (q >> 4)
                                           - dmin * m[:, 2 * s + 1][:, None])
    return out.reshape(-1)


def dequant_q5_k_fast(raw: bytes) -> np.ndarray:
    buf = _blocks(raw, "Q5_K")
    B = buf.shape[0]
    d = _f16(buf, 0)[:, None]
    dmin = _f16(buf, 2)[:, None]
    sc, m = _scale_min_k4_vec(buf[:, 4:16].astype(np.int32))
    qh = buf[:, 16:48].astype(np.int32)
    qs = buf[:, 48:176].astype(np.int32)
    out = np.empty((B, 256), dtype=np.float32)
    for s in range(4):
        q = qs[:, s * 32:(s + 1) * 32]
        u1, u2 = 1 << (2 * s), 2 << (2 * s)
        lo = (q & 0xF) + np.where(qh & u1, 16, 0)
        hi = (q >> 4) + np.where(qh & u2, 16, 0)
        out[:, s * 64:s * 64 + 32] = (d * sc[:, 2 * s][:, None] * lo
                                      - dmin * m[:, 2 * s][:, None])
        out[:, s * 64 + 32:s * 64 + 64] = (d * sc[:, 2 * s + 1][:, None] * hi
                                           - dmin * m[:, 2 * s + 1][:, None])
    return out.reshape(-1)


def dequant_iq4_xs_fast(raw: bytes) -> np.ndarray:
    buf = _blocks(raw, "IQ4_XS")
    B = buf.shape[0]
    d = _f16(buf, 0)[:, None]
    scales_h = np.ascontiguousarray(buf[:, 2:4]).view(np.uint16).reshape(-1).astype(np.int32)[:, None]
    scales_l = np.ascontiguousarray(buf[:, 4:8]).view(np.uint32).reshape(-1).astype(np.int32)[:, None]
    qs = buf[:, 8:136]
    out = np.empty((B, 256), dtype=np.float32)
    for ib in range(8):
        ls = ((scales_l >> (4 * ib)) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4)
        dl = (d * (ls - 32)).astype(np.float32)
        q = qs[:, ib * 16:(ib + 1) * 16]
        out[:, ib * 32:ib * 32 + 16] = _IQ4NL[q & 0x0F].astype(np.float32) * dl
        out[:, ib * 32 + 16:ib * 32 + 32] = _IQ4NL[q >> 4].astype(np.float32) * dl
    return out.reshape(-1)


# ------------------------------------------------------------------ 网格类（i-quant）
def dequant_iq3_xxs(raw: bytes) -> np.ndarray:
    """IQ3_XXS：8×32；每 32 元素 = 4 bit 尺度 + 4 组「4 元素网格 + 7 bit 符号」。"""
    buf = _blocks(raw, "IQ3_XXS")
    out = np.empty((buf.shape[0], 256), dtype=np.float32)
    for i in range(buf.shape[0]):
        b = buf[i]
        d = float(np.frombuffer(b[0:2].tobytes(), dtype=np.float16)[0])
        qs = b[2:66]
        ss = np.frombuffer(b[66:98].tobytes(), dtype=np.uint32)
        y = 0
        for ib in range(8):
            aux = int(ss[ib])
            db = d * (0.5 + (aux >> 28)) * 0.5
            for l in range(4):
                signs = _KSIGNS[(aux >> (7 * l)) & 127]
                g1 = _G_IQ3XXS[qs[8 * ib + 2 * l]]
                g2 = _G_IQ3XXS[qs[8 * ib + 2 * l + 1]]
                for j in range(4):
                    out[i, y + j] = db * g1[j] * (-1.0 if signs & _KMASK[j] else 1.0)
                    out[i, y + 4 + j] = db * g2[j] * (-1.0 if signs & _KMASK[j + 4] else 1.0)
                y += 8
    return out.reshape(-1)


def dequant_iq2_s(raw: bytes) -> np.ndarray:
    """IQ2_S：8×32；8 元素网格（1024 项），索引 4 字节 + 符号 4 字节交替。"""
    buf = _blocks(raw, "IQ2_S")
    out = np.empty((buf.shape[0], 256), dtype=np.float32)
    for i in range(buf.shape[0]):
        b = buf[i]
        d = float(np.frombuffer(b[0:2].tobytes(), dtype=np.float16)[0])
        qs = b[2:66]
        indices = qs[:32]
        signs = qs[32:]
        qh = b[66:74]
        scales = b[74:82]
        y = 0
        for ib in range(8):
            db = [d * (0.5 + (scales[ib] & 0xF)) * 0.25,
                  d * (0.5 + (scales[ib] >> 4)) * 0.25]
            for l in range(4):
                dl = db[l // 2]
                idx = int(indices[4 * ib + l]) | ((int(qh[ib]) << (8 - 2 * l)) & 0x300)
                grid = _G_IQ2S[idx]
                sg = int(signs[4 * ib + l])
                for j in range(8):
                    out[i, y + j] = dl * grid[j] * (-1.0 if sg & _KMASK[j] else 1.0)
                y += 8
    return out.reshape(-1)


def dequant_iq3_s(raw: bytes) -> np.ndarray:
    """IQ3_S：4×64；每 64 元素 = 2 个 32 元素组（同一尺度字节的低/高半字节）。"""
    buf = _blocks(raw, "IQ3_S")
    out = np.empty((buf.shape[0], 256), dtype=np.float32)
    for i in range(buf.shape[0]):
        b = buf[i]
        d = float(np.frombuffer(b[0:2].tobytes(), dtype=np.float16)[0])
        qs_all = b[2:66]
        qh = b[66:74]
        signs_all = b[74:106]
        scales = b[106:110]
        y = 0
        for pair in range(4):                       # 每对 = 64 元素
            db = [d * (1 + 2 * (scales[pair] & 0xF)), d * (1 + 2 * (scales[pair] >> 4))]
            for half in range(2):
                qs = qs_all[pair * 16 + half * 8: pair * 16 + half * 8 + 8]
                sg = signs_all[pair * 8 + half * 4: pair * 8 + half * 4 + 4]
                qhb = int(qh[pair * 2 + half])
                for l in range(4):
                    idx1 = int(qs[2 * l]) | ((qhb << (8 - 2 * l)) & 256)
                    idx2 = int(qs[2 * l + 1]) | ((qhb << (7 - 2 * l)) & 256)
                    g1 = _G_IQ3S[idx1]
                    g2 = _G_IQ3S[idx2]
                    for j in range(4):
                        out[i, y + j] = db[half] * g1[j] * (-1.0 if sg[l] & _KMASK[j] else 1.0)
                        out[i, y + 4 + j] = db[half] * g2[j] * (-1.0 if sg[l] & _KMASK[j + 4] else 1.0)
                    y += 8
    return out.reshape(-1)


def dequant_bf16(raw: bytes) -> np.ndarray:
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def dequant_f16(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)


def dequant_f32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32)


DEQUANT = {
    "Q2_0": dequant_q2_0, "IQ4_NL": dequant_iq4_nl, "IQ4_XS": dequant_iq4_xs,
    "Q6_K": dequant_q6_k, "Q5_K": dequant_q5_k, "Q4_K": dequant_q4_k,
    "IQ3_XXS": dequant_iq3_xxs, "IQ2_S": dequant_iq2_s, "IQ3_S": dequant_iq3_s,
    "Q8_0": dequant_q8_0, "Q4_0": dequant_q4_0,
    "BF16": dequant_bf16, "F16": dequant_f16, "F32": dequant_f32,
}


def dequantize(raw: bytes, type_name: str) -> np.ndarray:
    """按原编码解码（未知编码直接报错，绝不静默重量化）。"""
    if type_name not in DEQUANT:
        raise KeyError(f"未实现的原生解码：{type_name}")
    return DEQUANT[type_name](raw)


# ------------------------------------------------------------------ 自检
def _selftest() -> int:
    rng = np.random.default_rng(7)
    bad = 0
    print(f"{'编码':10s} {'块字节':>7s} {'每块元素':>8s} {'幅值均值':>10s}  {'全零块':>6s} {'有限':>5s}")
    for name, (qk, bb) in GEOMETRY.items():
        raw = rng.integers(0, 256, size=bb, dtype=np.uint8).tobytes()
        out = dequantize(raw, name)
        zero = dequantize(bytes(bb), name)
        zmax = float(np.abs(zero).max())
        finite = bool(np.isfinite(out).all())
        ok = (out.size == qk) and finite and zmax == 0.0
        bad += 0 if ok else 1
        print(f"{name:10s} {bb:7d} {out.size:8d} {float(np.abs(out).mean()):10.4f}  "
              f"{'ok' if zmax == 0 else f'{zmax:g}':>6s} {str(finite):>5s}")
    print("自检", "全部通过 ✔" if bad == 0 else f"{bad} 项失败 ✘")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="GGUF 原生编码参考解码")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--info", action="store_true")
    args = ap.parse_args()
    if args.info:
        for k, (qk, bb) in GEOMETRY.items():
            print(f"{k:10s} {qk:5d} 元素/块 {bb:5d} 字节/块 {bb * 8 / qk:5.2f} bit/元素")
        return 0
    return _selftest()


if __name__ == "__main__":
    sys.exit(main())
