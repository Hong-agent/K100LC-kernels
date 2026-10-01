#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compressed-tensors **INT4 / pack-quantized（W4A16）**的主机侧参考解码器 + 加载器。

对应满足下列布局的任意 checkpoint（默认目录见 `DEFAULT_MODEL_DIR`，
可用环境变量 `K100LC_INT4_MODEL` 覆盖）：

## 存储格式

`config.json` 的 `quantization_config`：`format = pack-quantized`、
`group_size = 128`、`num_bits = 4`、`type = int`、`symmetric = true`、
`actorder = static`、没有 zero point、也没有 g_idx。于是每个线性层是三个张量：

| 张量 | dtype | shape | 内容 |
|---|---|---|---|
| `weight_packed` | I32 | `[N, K/8]` | 8 个 4bit 码 / 字，低半字节 = 更小的 k |
| `weight_scale` | BF16 | `[N, K/128]` | 每 128 个 k 一个尺度，无 global scale |
| `weight_shape` | I64 | `[2]` | `[N, K]`（行数、列数） |

反量化：

    码 = (word >> (4 * (k % 8))) & 0xF          # 低半字节 = 更小的 k
    w[n, k] = (码 - 8) * bf16(scale[n, k // 128])   # 码是 offset-binary ∈ [-8, 7]

也就是说 **scale 数组的线性下标就是「128 权重块」的下标**：
`scale_index = n * (K/128) + k // 128`，和 GPU 内核里 `block_id` 一一对应
（块 n 行内第 b 块 → `block_id = n * nbpr + b`，`nbpr = K/128`）。这一点让
`int4_dot_k` 只用「块号」就能算出权重和尺度两个地址。

本模块**不做任何重量化、不重排**：按上面的式子直接读 checkpoint 原始字节。
它的定位和 `tools/iq_dequant.py` 一样，是 GPU 内核（`tools/gen_int4_dot.py`
产出的 `int4_dequant_k` / `int4_dot_k`）的「金标准」，也是运行时直接喂设备的
加载器（`Runtime.launch` 的 buffer 可以直接拿 `packed_rows()` / `scale_rows()`）。

    python3 tools/ct_int4.py --model-dir /path/to/checkpoint --info
    python3 tools/ct_int4.py --model-dir /path/to/checkpoint --selftest

没有任何第三方依赖，只需要 numpy。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import struct

import numpy as np

DEFAULT_MODEL_DIR = pathlib.Path(
    os.environ.get("K100LC_INT4_MODEL", "models/int4-checkpoint"))

GROUP = 128                     # 每 128 个 k 一组（checkpoint 的 group_size）
WORDS_PER_GROUP = GROUP // 8    # 一组 = 16 个 I32（每个字 8 个 4bit 码）
BLOCK_BYTES = GROUP // 2        # 一组 = 64 字节（内核侧一个 work-item 读这么多）


# --------------------------------------------------------------------------
# safetensors：只读头 + 按需 seek，18 GB 也不用整份读进内存
# --------------------------------------------------------------------------
def read_header(path: pathlib.Path) -> dict:
    """返回 safetensors 头部 dict（张量名 → {dtype, shape, data_offsets}）。"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def tensor_bytes(path: pathlib.Path, hdr: dict, name: str) -> bytes:
    if name not in hdr:
        raise KeyError(f"{path.name} 里没有张量 {name}")
    e = hdr[name]
    off, end = e["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + _header_len(path) + off)
        raw = f.read(end - off)
    if len(raw) != end - off:
        raise IOError(f"{name}: 期望 {end - off} 字节，读到 {len(raw)}")
    return raw


_HDR_LEN_CACHE: dict[str, int] = {}


def _header_len(path: pathlib.Path) -> int:
    key = str(path)
    if key not in _HDR_LEN_CACHE:
        with open(path, "rb") as f:
            _HDR_LEN_CACHE[key] = struct.unpack("<Q", f.read(8))[0]
    return _HDR_LEN_CACHE[key]


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    """bf16 → f32：bf16 就是 f32 的高 16 位，左移 16 位即精确转换。"""
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)


def unpack_int4(packed: np.ndarray, k: int | None = None) -> np.ndarray:
    """`[..., K/8]` 的 I32 码字 → `[..., K]` 的 int8 值（已减 8）。"""
    w = np.asarray(packed, dtype=np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * 4)
    codes = ((w[..., None] >> shifts) & 0xF).astype(np.int8).reshape(*w.shape[:-1], -1)
    if k is not None:
        codes = codes[..., :k]
    return (codes - np.int8(8)).astype(np.int8)


# --------------------------------------------------------------------------
# 线性层加载器
# --------------------------------------------------------------------------
class Int4Linear:
    """checkpoint 里一个 pack-quantized 线性层（懒加载，按行读）。"""

    def __init__(self, prefix: str, model_dir: pathlib.Path = DEFAULT_MODEL_DIR,
                 files: list[pathlib.Path] | None = None):
        self.prefix = prefix
        self.model_dir = pathlib.Path(model_dir)
        files = files or [self.model_dir / "model.safetensors",
                          self.model_dir / "model_mtp.safetensors"]
        self.files = [p for p in files if p.is_file()]
        self._headers = [(p, read_header(p)) for p in self.files]
        self.n, self.k = self._shape()
        if self.k % GROUP:
            raise ValueError(f"{prefix}: K={self.k} 不是 {GROUP} 的整数倍")
        self.nbpr = self.k // GROUP

    def _entry(self, suffix: str):
        name = self.prefix + "." + suffix
        for p, hdr in self._headers:
            if name in hdr:
                return p, hdr, hdr[name]
        raise KeyError(f"找不到 {name}（{self.model_dir}）")

    def _shape(self) -> tuple[int, int]:
        _p, _h, e = self._entry("weight_shape")
        raws = tensor_bytes(_p, _h, self.prefix + ".weight_shape")
        n, k = np.frombuffer(raws, dtype=np.int64)
        # 交叉校验：packed / scale 的形状必须和 weight_shape 自洽
        _p2, _h2, pe = self._entry("weight_packed")
        _p3, _h3, se = self._entry("weight_scale")
        if list(pe["shape"]) != [int(n), int(k) // 8]:
            raise ValueError(f"{self.prefix}: packed 形状 {pe['shape']} 与 [{n},{k}] 不符")
        if list(se["shape"]) != [int(n), int(k) // GROUP]:
            raise ValueError(f"{self.prefix}: scale 形状 {se['shape']} 与 [{n},{k}] 不符")
        return int(n), int(k)

    def packed_rows(self, row0: int = 0, nrows: int | None = None) -> bytes:
        """`weight_packed` 里连续若干行（字节流，可直接当内核的 `w` 缓冲用）。"""
        return self._rows("weight_packed", row0, nrows, self.k // 8 * 4)

    def scale_rows(self, row0: int = 0, nrows: int | None = None) -> bytes:
        """`weight_scale` 里连续若干行（bf16 字节流 → 内核的 `s` 缓冲）。"""
        return self._rows("weight_scale", row0, nrows, self.k // GROUP * 2)

    def _rows(self, suffix: str, row0: int, nrows: int | None, row_bytes: int) -> bytes:
        nrows = self.n - row0 if nrows is None else nrows
        if row0 < 0 or nrows <= 0 or row0 + nrows > self.n:
            raise ValueError(f"{self.prefix}: 行范围 [{row0},{row0+nrows}) 越界（N={self.n}）")
        path, hdr, _e = self._entry(suffix)
        off0 = hdr[self.prefix + "." + suffix]["data_offsets"][0] + row0 * row_bytes
        with open(path, "rb") as f:
            f.seek(8 + _header_len(path) + off0)
            raw = f.read(nrows * row_bytes)
        if len(raw) != nrows * row_bytes:
            raise IOError(f"{self.prefix}.{suffix}: 期望 {nrows*row_bytes} 字节")
        return raw

    def dequant_rows(self, row0: int = 0, nrows: int | None = None) -> np.ndarray:
        """参考解码：返回 `[nrows, K]` 的 f32（式子和内核逐字对应）。"""
        nrows = self.n - row0 if nrows is None else nrows
        packed = np.frombuffer(self.packed_rows(row0, nrows), dtype=np.uint32)
        packed = packed.reshape(nrows, self.k // 8)
        scales = np.frombuffer(self.scale_rows(row0, nrows), dtype=np.uint16)
        scales = scales.reshape(nrows, self.nbpr)
        vals = unpack_int4(packed).astype(np.float32)
        return vals * bf16_to_f32(scales).repeat(GROUP, axis=1)

    def dequant(self) -> np.ndarray:
        """整个矩阵（小心内存：17408×5120 的 f32 是 356 MB）。"""
        return self.dequant_rows(0, self.n)

    def ref_gemv(self, x: np.ndarray, row0: int = 0, nrows: int | None = None) -> np.ndarray:
        """参考 `y = W x`：用 f32 矩阵乘（`x` 是 `[K]` 或 `[tokens, K]`）。"""
        w = self.dequant_rows(row0, nrows)
        return x @ w.T

    def __repr__(self) -> str:
        return (f"Int4Linear({self.prefix}, N={self.n}, K={self.k}, "
                f"nbpr={self.nbpr}, 权重 {self.n * self.k * 4 / 8 / 1e6:.1f} MB)")


def list_linears(model_dir: pathlib.Path = DEFAULT_MODEL_DIR) -> list[str]:
    """列出 checkpoint 里所有 pack-quantized 的线性层前缀。"""
    hdr = read_header(pathlib.Path(model_dir) / "model.safetensors")
    return sorted(k[: -len(".weight_packed")] for k in hdr
                  if k.endswith(".weight_packed"))


# --------------------------------------------------------------------------
# 自检：码分布 + 与同工程 RT4 转换结果对账
# --------------------------------------------------------------------------
RT4_MANIFEST = pathlib.Path(os.environ["K100LC_RT4_MANIFEST"]) \
    if os.environ.get("K100LC_RT4_MANIFEST") else None


def code_histogram(prefix: str, rows: int = 256, model_dir=DEFAULT_MODEL_DIR) -> np.ndarray:
    """取若干行统计 16 个码的出现次数。

    offset-binary（值 = 码 − 8）时长尾对称落在码 8 两侧、峰值在 8；
    若是二补码存档，峰值会落在 0 和 15 两端 —— 这是两者最直接的判别。
    """
    lin = Int4Linear(prefix, model_dir)
    raw = np.frombuffer(lin.packed_rows(0, min(rows, lin.n)), dtype=np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * 4)
    codes = ((raw[:, None] >> shifts) & 0xF).reshape(-1)
    return np.bincount(codes, minlength=16).astype(np.int64)


def _rt4_tensor(prefix: str, row0: int, nrows: int) -> np.ndarray:
    """从另一份 `.rt4`（可选交叉验证格式）里解同样几行，f16 尺度。"""
    if RT4_MANIFEST is None:
        raise SystemExit("未设置 K100LC_RT4_MANIFEST，跳过 RT4 交叉验证")
    man = json.loads(RT4_MANIFEST.read_text(encoding="utf-8"))
    ent = next((t for t in man["tensors"] if t["name"] == prefix + ".weight"), None)
    if ent is None:
        raise KeyError(f"RT4 manifest 里没有 {prefix}.weight")
    n, k = ent["shape"][0], ent["shape"][1]
    grp = ent["group"]
    path = RT4_MANIFEST.with_suffix("")            # 同名的 .rt4 权重文件
    with open(path, "rb") as f:
        f.seek(ent["q_off"] + row0 * (k // 2))
        q = np.frombuffer(f.read(nrows * (k // 2)), dtype=np.uint8).reshape(nrows, k // 2)
        f.seek(ent["s_off"] + row0 * (k // grp) * 2)
        s = np.frombuffer(f.read(nrows * (k // grp) * 2), dtype=np.float16).reshape(nrows, k // grp)
    lo = (q & 0xF).astype(np.int8)
    hi = ((q >> 4) & 0xF).astype(np.int8)
    vals = np.empty((nrows, k), dtype=np.int8)
    vals[:, 0::2] = lo                             # 低半字节 = 更小的 k
    vals[:, 1::2] = hi
    vals = np.where(vals > 7, vals - 16, vals)     # RT4 是二补码 int4
    return vals.astype(np.float32) * s.astype(np.float32).repeat(grp, axis=1)


def selftest(prefix: str = "model.language_model.layers.3.mlp.gate_proj",
             model_dir=DEFAULT_MODEL_DIR, rows: int = 64) -> int:
    lin = Int4Linear(prefix, model_dir)
    print(repr(lin))

    hist = code_histogram(prefix, rows=rows, model_dir=model_dir)
    tot = hist.sum()
    peak = int(hist.argmax())
    pct = 100.0 * hist / tot
    print("  码直方图（%，16 个码）：")
    print("   " + " ".join(f"{v:5.2f}" for v in pct))
    print(f"  峰值码 = {peak}（offset-binary 应为 8；两端峰值 = 二补码）")
    codes_ok = peak == 8

    cov = 0.0
    if RT4_MANIFEST is not None and RT4_MANIFEST.is_file():
        r0, nr = 0, min(rows, lin.n)
        mine = lin.dequant_rows(r0, nr)
        theirs = _rt4_tensor(prefix, r0, nr)
        a = mine.reshape(-1).astype(np.float64)
        b = theirs.reshape(-1).astype(np.float64)
        cov = float(np.corrcoef(a, b)[0, 1])
        rel = float(np.linalg.norm(a - b) / max(1e-12, np.linalg.norm(b)))
        print(f"  与 RT4（独立转换实现）逐元素相关 = {cov:.6f}，相对偏差 = {rel:.3%}")
        cov_ok = cov > 0.995
    else:
        print("  （跳过 .rt4 交叉验证：未设置 K100LC_RT4_MANIFEST 或文件不存在）")
        cov_ok = True
    ok = codes_ok and cov_ok
    print("INT4 打包语义：", "确认 ✔" if ok else "存疑 ✘")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="compressed-tensors INT4 主机侧参考解码")
    ap.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--prefix", default="model.language_model.layers.3.mlp.gate_proj")
    ap.add_argument("--info", action="store_true", help="列出所有量化线性层")
    ap.add_argument("--selftest", action="store_true", help="码分布 + RT4 对账")
    ap.add_argument("--rows", type=int, default=64)
    args = ap.parse_args()

    if args.info:
        names = list_linears(args.model_dir)
        print(f"{args.model_dir}: {len(names)} 个 pack-quantized 线性层")
        for nm in names[:8]:
            print("  " + repr(Int4Linear(nm, args.model_dir)))
        if len(names) > 8:
            print(f"  ... 其余 {len(names) - 8} 个")
        return 0
    if args.selftest:
        return selftest(args.prefix, args.model_dir, args.rows)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
