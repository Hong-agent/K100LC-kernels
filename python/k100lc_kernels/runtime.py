from __future__ import annotations

import ctypes
import pathlib
import struct

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_HSACO = ROOT / "prebuilt" / "k100lc_kernels.hsaco"
DEFAULT_LIB = ROOT / "prebuilt" / "libfm_engine.so"


def _pack(v) -> int:
    if isinstance(v, float):
        return struct.unpack("<I", struct.pack("<f", v))[0]
    return int(v)


class Runtime:
    """常驻 HSA 引擎：显存缓冲只分配一次，反复 launch。"""

    def __init__(self, hsaco=DEFAULT_HSACO, lib=DEFAULT_LIB):
        lib = ctypes.CDLL(str(lib))
        self._lib = lib
        lib.fm_init.restype = ctypes.c_int
        lib.fm_init.argtypes = [ctypes.c_char_p]
        lib.fm_alloc.restype = ctypes.c_void_p
        lib.fm_alloc.argtypes = [ctypes.c_size_t]
        lib.fm_free.argtypes = [ctypes.c_void_p]
        lib.fm_upload.restype = ctypes.c_int
        lib.fm_upload.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.fm_download.restype = ctypes.c_int
        lib.fm_download.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.fm_copy.restype = ctypes.c_int
        lib.fm_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.fm_memset.restype = ctypes.c_int
        lib.fm_memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        lib.fm_sync.restype = ctypes.c_int
        lib.fm_launch.restype = ctypes.c_int
        lib.fm_launch.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
        lib.fm_launch2d.restype = ctypes.c_int
        lib.fm_launch2d.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32,
                                    ctypes.c_uint32, ctypes.c_uint32,
                                    ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
        if lib.fm_init(str(hsaco).encode()) != 0:
            raise RuntimeError(f"fm_init 失败: {hsaco}")

    def alloc(self, nbytes: int) -> int:
        p = self._lib.fm_alloc(nbytes)
        if not p:
            raise MemoryError(f"fm_alloc({nbytes}) 失败")
        return int(p)

    def free(self, ptr: int) -> None:
        self._lib.fm_free(ctypes.c_void_p(ptr))

    def upload(self, ptr: int, data) -> None:
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data)
            buf, n = data.ctypes.data_as(ctypes.c_void_p), data.nbytes
        else:
            data = bytes(data)
            buf, n = ctypes.c_char_p(data), len(data)
        if self._lib.fm_upload(ctypes.c_void_p(ptr), buf, n) != 0:
            raise RuntimeError("fm_upload 失败")

    def download(self, ptr: int, count: int, dtype) -> np.ndarray:
        dt = np.dtype(dtype)
        out = np.empty(count, dtype=dt)
        if self._lib.fm_download(out.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_void_p(ptr), count * dt.itemsize) != 0:
            raise RuntimeError("fm_download 失败")
        return out

    def copy_dev(self, dst: int, src: int, nbytes: int) -> None:
        if self._lib.fm_copy(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes) != 0:
            raise RuntimeError("fm_copy 失败")

    def memset(self, ptr: int, value: int, nbytes: int) -> None:
        if self._lib.fm_memset(ctypes.c_void_p(ptr), value, nbytes) != 0:
            raise RuntimeError("fm_memset 失败")

    def launch(self, kernel: str, grid: int, workgroup: int, argv: list) -> None:
        a = (ctypes.c_uint64 * len(argv))(*[_pack(v) for v in argv])
        if self._lib.fm_launch(kernel.encode(), grid, workgroup, a, len(argv)) != 0:
            raise RuntimeError(f"fm_launch({kernel}) 失败")

    def launch2d(self, kernel: str, gx: int, gy: int, wx: int, wy: int, argv: list) -> None:
        a = (ctypes.c_uint64 * len(argv))(*[_pack(v) for v in argv])
        if self._lib.fm_launch2d(kernel.encode(), gx, gy, wx, wy, a, len(argv)) != 0:
            raise RuntimeError(f"fm_launch2d({kernel}) 失败")

    def sync(self) -> None:
        self._lib.fm_sync()
