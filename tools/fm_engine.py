#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""常驻引擎的 Python 封装（ctypes → build/libfm_engine.so）。

    from fm_engine import Engine
    e = Engine("build/flashmoe.hsaco")
    p = e.alloc(1024)
    e.upload(p, x)                 # x: numpy 数组或 bytes
    e.launch("gemv_i8_k", grid, wg, [p, ...])
    y = e.download(p, 256, np.float32)
    e.free(p)

这是逐 token 跑模型用的底座：显存缓冲常驻，权重只上传一次。
"""
from __future__ import annotations

import ctypes
import pathlib
import struct

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LIB = ROOT / "build" / "libfm_engine.so"


class Engine:
    def __init__(self, hsaco: str | pathlib.Path, lib: str | pathlib.Path = DEFAULT_LIB):
        self.lib = ctypes.CDLL(str(lib))
        self.lib.fm_init.restype = ctypes.c_int
        self.lib.fm_init.argtypes = [ctypes.c_char_p]
        self.lib.fm_alloc.restype = ctypes.c_void_p
        self.lib.fm_alloc.argtypes = [ctypes.c_size_t]
        self.lib.fm_free.argtypes = [ctypes.c_void_p]
        self.lib.fm_upload.restype = ctypes.c_int
        self.lib.fm_upload.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.fm_download.restype = ctypes.c_int
        self.lib.fm_download.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.fm_copy.restype = ctypes.c_int
        self.lib.fm_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.fm_memset.restype = ctypes.c_int
        self.lib.fm_memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.lib.fm_sync.restype = ctypes.c_int
        self.lib.fm_launch.restype = ctypes.c_int
        self.lib.fm_launch.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32,
                                       ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
        self.lib.fm_launch2d.restype = ctypes.c_int
        self.lib.fm_launch2d.argtypes = [
            ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
        rc = self.lib.fm_init(str(hsaco).encode())
        if rc != 0:
            raise RuntimeError(f"fm_init 失败 rc={rc}")

    # ---------------- 缓冲 ----------------
    def alloc(self, nbytes: int) -> int:
        p = self.lib.fm_alloc(nbytes)
        if not p:
            raise MemoryError(f"fm_alloc({nbytes}) 失败")
        return int(p)

    def free(self, ptr: int) -> None:
        self.lib.fm_free(ctypes.c_void_p(ptr))

    def upload(self, ptr: int, data) -> None:
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data)
            buf = data.ctypes.data_as(ctypes.c_void_p)
            n = data.nbytes
        else:
            buf = ctypes.c_char_p(bytes(data))
            n = len(data)
        if self.lib.fm_upload(ctypes.c_void_p(ptr), buf, n) != 0:
            raise RuntimeError("fm_upload 失败")

    def download(self, ptr: int, count: int, dtype) -> np.ndarray:
        dt = np.dtype(dtype)
        out = np.empty(count, dtype=dt)
        if self.lib.fm_download(out.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_void_p(ptr), count * dt.itemsize) != 0:
            raise RuntimeError("fm_download 失败")
        return out

    def copy_dev(self, dst: int, src: int, nbytes: int) -> None:
        if self.lib.fm_copy(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes) != 0:
            raise RuntimeError("fm_copy 失败")

    def memset(self, ptr: int, value: int, nbytes: int) -> None:
        if self.lib.fm_memset(ctypes.c_void_p(ptr), value, nbytes) != 0:
            raise RuntimeError("fm_memset 失败")

    # ---------------- 启动 ----------------
    def launch(self, kernel: str, grid: int, workgroup: int, argv: list[int]) -> None:
        arr = (ctypes.c_uint64 * len(argv))(*[_pack_arg(v) for v in argv])
        rc = self.lib.fm_launch(kernel.encode(), grid, workgroup, arr, len(argv))
        if rc != 0:
            raise RuntimeError(f"fm_launch({kernel}) 失败 rc={rc}")

    def launch2d(self, kernel: str, gx: int, gy: int, wx: int, wy: int, argv: list[int]) -> None:
        arr = (ctypes.c_uint64 * len(argv))(*[_pack_arg(v) for v in argv])
        rc = self.lib.fm_launch2d(kernel.encode(), gx, gy, wx, wy, arr, len(argv))
        if rc != 0:
            raise RuntimeError(f"fm_launch2d({kernel}) 失败 rc={rc}")

    def sync(self) -> None:
        self.lib.fm_sync()


def _pack_arg(v) -> int:
    if isinstance(v, float):
        return struct.unpack("<I", struct.pack("<f", v))[0]
    return int(v)
