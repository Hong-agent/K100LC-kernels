"""主机侧量化打包 / 参考解码工具（NumPy，无 torch 依赖）。

当前提供 compressed-tensors 风格的 **INT4 group=128 / offset-binary** 打包，
与 `tools/ct_int4.py`、GPU 侧 `int4_dot_k` 的语义一致：

    code = round(w / scale) + 8        # code ∈ [0,15]
    w    = (code - 8) * scale          # 尺度每 128 个 k 一个（BF16）

打包字节布局：每个 I32 放 8 个码，低半字节 = 更小的 k；尺度数组按
`[N, K/128]` 紧凑排列，与 GPU 内核对块号 → 地址的映射一致。
"""
from __future__ import annotations

import numpy as np

GROUP = 128


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 → bf16 位模式（uint16），round-to-nearest-even。"""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return (u >> 16).astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """bf16 位模式（uint16）→ float32。"""
    u = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32)
    return (u << 16).view(np.float32)


def pack_int4_group128(w: np.ndarray) -> tuple[bytes, bytes]:
    """把 f32 权重 `[N,K]` 打包成 `(packed_bytes, scale_bf16_bytes)`。

    要求 `K % 128 == 0`。尺度取每 128 个 k 的 `max|w| / 7`，保证码落在
    `[0,15]`；全零组尺度记 1.0。
    """
    w = np.ascontiguousarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("pack_int4_group128 需要 [N,K] 的二维权重")
    n, k = w.shape
    if k % GROUP:
        raise ValueError(f"K={k} 不是 {GROUP} 的整数倍")
    g = w.reshape(n, k // GROUP, GROUP)
    amax = np.abs(g).max(axis=-1)
    scale = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float32)
    q = np.rint(g / scale[..., None]) + 8.0
    q = np.clip(q, 0.0, 15.0).astype(np.uint32)
    q = q.reshape(n, k // 8, 8)
    words = np.zeros((n, k // 8), dtype=np.uint32)
    for j in range(8):
        words |= q[..., j] << np.uint32(4 * j)
    packed = np.ascontiguousarray(words).tobytes()
    scales = f32_to_bf16_bits(scale.reshape(n, k // GROUP)).tobytes()
    return packed, scales


def dequant_int4_group128(packed: bytes, scales: bytes,
                          n: int, k: int) -> np.ndarray:
    """打包字节 → f32 `[N,K]`（与 GPU 内核的数学定义一致，用于对账）。"""
    words = np.frombuffer(packed, dtype=np.uint32).reshape(n, k // 8)
    codes = np.empty((n, k), dtype=np.int32)
    for j in range(8):
        codes[:, j::8] = ((words >> np.uint32(4 * j)) & np.uint32(0xF)).astype(np.int32)
    s = bf16_bits_to_f32(np.frombuffer(scales, dtype=np.uint16)).reshape(n, k // GROUP)
    return ((codes - 8).astype(np.float32) * np.repeat(s, GROUP, axis=1))


def ct_int4_to_twos_complement(packed: bytes) -> bytes:
    """compressed-tensors 的 offset-binary 码 → 两补码 int4 字节。

    两者在字节内的位置完全一致（低半字节 = 偶数 k），只差一个常量：

        code(c) = value + 8        # c ∈ [0,15]，value = c-8 ∈ [-8,7]
        nibble(v) = v & 0xF        # 两补码编码

    所以逐半字节做 `(c+8) & 0xF` 即可，等价于 `value & 0xF`。
    转换后的字节可以直接喂给 RT4 的 `gemv_w4a8` / `gemv_w4a4`。
    """
    raw = np.frombuffer(packed, dtype=np.uint8)
    lo = ((raw & 0x0F) + 8) & 0x0F
    hi = (((raw >> 4) & 0x0F) + 8) & 0x0F
    return (lo | (hi << 4)).astype(np.uint8).tobytes()


def ct_int4_to_rt4(packed: bytes, scales_bf16: bytes) -> tuple[bytes, bytes]:
    """compressed-tensors INT4 → RT4 兼容的 `(q 字节, f16 尺度字节)`。

    q 区就是两补码 int4、低半字节 = 偶数 k；尺度由 BF16 转成 f16
    （RT4 的 W4A8/W4A4 内核吃 f16 尺度）。这一步每层只做一次。
    """
    q = ct_int4_to_twos_complement(packed)
    s = bf16_bits_to_f32(np.frombuffer(scales_bf16, dtype=np.uint16))
    return q, np.ascontiguousarray(s.astype(np.float16)).tobytes()


def int4_scale_group_first_f32(scales_bf16: bytes, n: int, k: int) -> np.ndarray:
    """`[N, K/128]` BF16 尺度 → `[K/128, N]` f32（W4A4 GEMM 需要组优先）。"""
    s = bf16_bits_to_f32(np.frombuffer(scales_bf16, dtype=np.uint16)) \
        .reshape(n, k // GROUP)
    return np.ascontiguousarray(s.T, dtype=np.float32)
