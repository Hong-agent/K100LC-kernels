from __future__ import annotations

import ctypes
import json
import pathlib
import struct

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_HSACO = ROOT / "prebuilt" / "k100lc_kernels.hsaco"
DEFAULT_LIB = ROOT / "prebuilt" / "libfm_engine.so"

KINDS = {
    "by_value": 0, "global_buffer": 0,
    "hidden_block_count_x": 1, "hidden_block_count_y": 2, "hidden_block_count_z": 3,
    "hidden_group_size_x": 4, "hidden_group_size_y": 5, "hidden_group_size_z": 6,
    "hidden_remainder_x": 7, "hidden_remainder_y": 8, "hidden_remainder_z": 9,
    "hidden_global_offset_x": 10, "hidden_global_offset_y": 11, "hidden_global_offset_z": 12,
    "hidden_grid_dims": 13,
}


def _pack(v) -> int:
    if isinstance(v, float):
        return struct.unpack("<I", struct.pack("<f", v))[0]
    return int(v)


class _Batch:
    """`with rt.batch():` 的上下文管理器（见 `Runtime.batch`）。"""

    def __init__(self, rt: "Runtime"):
        self.rt = rt
        self.recs: list = []

    def __enter__(self) -> "_Batch":
        if self.rt._batch is not None:
            raise RuntimeError("batch 不能嵌套")
        self.rt._batch = self.recs
        return self

    def __exit__(self, *exc) -> None:
        self.rt._batch = None
        if exc[0] is None and self.recs:
            self.rt._flush_batch(self.recs)
        self.recs = []


class LaunchPlan:
    """把一串 launch **录一次、重放很多次**（类似 CUDA graph 的简化版）。

    一层 decoder 有十几个内核，逐 token 重新走一遍 Python 路径（打包参数 +
    ctypes 调用 + 引擎查找）实测每 token 要 ~0.27 ms；而这一层里除了一两个值
    （RoPE 的位置、KV 追加的列偏移、注意力的长度/分块）以外全是常量。这里把
    记录摊平成一个 uint64 数组，重放时只打几个补丁再一次性投递。

    用法：

    ```python
    plan = LaunchPlan(rt, calls)      # calls: [(kernel, grid, wg, argv), ...]
    plan.set_arg(3, 7, pos)           # 第 4 条记录的第 8 个参数 = pos
    plan.run()                        # 一次调用全部入队
    ```
    """

    def __init__(self, rt: "Runtime", calls: list):
        self.rt = rt
        self.n = len(calls)
        total = sum(4 + len(argv) for _k, _g, _w, argv in calls)
        self.buf = (ctypes.c_uint64 * total)()
        self.starts: list[int] = []       # 每条记录 argv 区在 buf 里的起点
        i = 0
        for kernel, grid, wg, argv in calls:
            kid = rt._kid(kernel)
            if kid < 0:
                raise RuntimeError(f"fm_kernel_id({kernel}) 失败")
            n = len(argv)
            self.buf[i] = kid
            self.buf[i + 1] = int(grid)
            self.buf[i + 2] = int(wg)
            self.buf[i + 3] = n
            for j, v in enumerate(argv):
                self.buf[i + 4 + j] = rt._pack_one(v)
            self.starts.append(i + 4)
            i += 4 + n

    def set_arg(self, rec: int, idx: int, value) -> None:
        self.buf[self.starts[rec] + idx] = self.rt._pack_one(value)

    def set_grid(self, rec: int, grid: int) -> None:
        self.buf[self.starts[rec] - 3] = int(grid)

    @property
    def kernels(self) -> list:
        return [self.rt._kid_cache_rev.get(int(self.buf[self.starts[r] - 4]), "?")
                for r in range(self.n)]

    def run(self) -> int:
        got = self.rt._lib.fm_launch_batch(self.buf, self.n)
        if got != self.n:
            raise RuntimeError(f"fm_launch_batch 只投递了 {got}/{self.n} 条")
        return got


class Runtime:
    """常驻 HSA 引擎：显存缓冲只分配一次，反复 launch。"""

    def __init__(self, hsaco=DEFAULT_HSACO, lib=DEFAULT_LIB, catalog=None):
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
        lib.fm_kernel_id.restype = ctypes.c_int
        lib.fm_kernel_id.argtypes = [ctypes.c_char_p]
        lib.fm_launch_batch.restype = ctypes.c_int
        lib.fm_launch_batch.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
        lib.fm_launch2d.restype = ctypes.c_int
        lib.fm_launch2d.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32,
                                    ctypes.c_uint32, ctypes.c_uint32,
                                    ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
        lib.fm_launch_dyn.restype = ctypes.c_int
        lib.fm_launch_dyn.argtypes = [
            ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
        if lib.fm_init(str(hsaco).encode()) != 0:
            raise RuntimeError(f"fm_init 失败: {hsaco}")
        self.hsaco = pathlib.Path(hsaco)
        self.catalog = self._load_catalog(catalog)
        # 每次 launch 的固定开销优化：缓存编码后的内核名，复用 argv 缓冲
        # （fm_launch 在返回前就把 argv 拷进 kernarg，所以复用是安全的）。
        self._name_cache: dict[str, bytes] = {}
        self._kid_cache: dict[str, int] = {}
        self._kid_cache_rev: dict[int, str] = {}
        self._batch: list | None = None
        self._argv_buf = (ctypes.c_uint64 * 64)()

    def _cname(self, kernel: str) -> bytes:
        b = self._name_cache.get(kernel)
        if b is None:
            b = kernel.encode()
            self._name_cache[kernel] = b
        return b

    def _pack_one(self, v) -> int:
        if isinstance(v, (float, np.floating)):
            return struct.unpack("<I", struct.pack("<f", float(v)))[0]
        if isinstance(v, np.integer):
            return int(v)
        return int(v)

    def _pack_argv(self, argv) -> tuple:
        n = len(argv)
        buf = self._argv_buf
        if n > len(buf):
            buf = self._argv_buf = (ctypes.c_uint64 * n)()
        for i, v in enumerate(argv):
            # 允许传 numpy 标量（np.float32 / np.int32 ...）：用户从数组里
            # 取出来的标量很常见，之前会直接 TypeError。
            if isinstance(v, (float, np.floating)):
                buf[i] = struct.unpack("<I", struct.pack("<f", float(v)))[0]
            elif isinstance(v, np.integer):
                buf[i] = int(v)
            else:
                buf[i] = v
        return buf

    def _load_catalog(self, catalog):
        if catalog is not None:
            return json.loads(pathlib.Path(catalog).read_text(encoding="utf-8"))
        for p in (self.hsaco.with_suffix(".catalog.json"),
                  pathlib.Path(str(self.hsaco) + ".catalog.json")):
            if p.is_file():
                return json.loads(p.read_text(encoding="utf-8"))
        if self.hsaco.resolve() == DEFAULT_HSACO.resolve():
            return json.loads((ROOT / "python" / "k100lc_kernels" / "catalog.json")
                              .read_text(encoding="utf-8"))
        return {"kernels": []}

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
        if self._batch is not None:
            # 注意：`_pack_argv` 用的是共享缓冲，下一次 launch 会把它覆盖掉，
            # 所以批量模式下必须**当场**把值拷下来（否则所有记录都指向最后一次的值）
            # 逐个打包（不要用 _pack_argv：它复用共享缓冲，长度是历史上最大的
            # 那次，直接迭代会多出尾巴）
            self._batch.append((kernel, int(grid), int(workgroup),
                                [self._pack_one(v) for v in argv], len(argv)))
            return
        a = self._pack_argv(argv)
        if self._lib.fm_launch(self._cname(kernel), grid, workgroup, a, len(argv)) != 0:
            raise RuntimeError(f"fm_launch({kernel}) 失败")

    # ---------------- 批量投递 ----------------
    def batch(self) -> "Runtime":
        """`with rt.batch():` —— 区间内的 `launch()` 只记录，退出时**一次性**入队。

        单次 launch 的主机侧开销实测 8~16 us（ctypes + 参数打包 + 引擎调用），
        逐条发时这段全在关键路径上；一层 decoder 有十几个内核，攒起来发能把
        这段砍掉大半。语义与逐条完全相同（同一条 in-order 队列、同样顺序）。
        """
        return _Batch(self)

    def _flush_batch(self, recs: list) -> None:
        """把记录链打成一个 uint64 数组，一次 `fm_launch_batch` 发出去。"""
        total = sum(4 + n for _k, _g, _w, _a, n in recs)
        buf = (ctypes.c_uint64 * total)()
        i = 0
        for kernel, grid, wg, a, n in recs:
            kid = self._kid(kernel)
            if kid < 0:
                raise RuntimeError(f"fm_kernel_id({kernel}) 失败")
            buf[i] = kid
            buf[i + 1] = grid
            buf[i + 2] = wg
            buf[i + 3] = n
            for j in range(n):
                buf[i + 4 + j] = a[j]
            i += 4 + n
        got = self._lib.fm_launch_batch(buf, len(recs))
        if got != len(recs):
            raise RuntimeError(f"fm_launch_batch 只投递了 {got}/{len(recs)} 条")

    def _kid(self, kernel: str) -> int:
        kid = self._kid_cache.get(kernel)
        if kid is None:
            kid = int(self._lib.fm_kernel_id(self._cname(kernel)))
            self._kid_cache[kernel] = kid
            self._kid_cache_rev[kid] = kernel
        return kid

    def launch2d(self, kernel: str, gx: int, gy: int, wx: int, wy: int, argv: list) -> None:
        a = self._pack_argv(argv)
        if self._lib.fm_launch2d(self._cname(kernel), gx, gy, wx, wy, a, len(argv)) != 0:
            raise RuntimeError(f"fm_launch2d({kernel}) 失败")

    def _find(self, kernel: str) -> dict:
        for k in self.catalog.get("kernels", []):
            if k.get("lookup") == kernel or k.get("name") == kernel:
                return k
        raise KeyError(f"catalog 里没有内核 {kernel}")

    def launch_dyn(self, kernel: str, gx: int, gy: int, wx: int, wy: int, argv: list) -> None:
        """用 HSACO 自带 metadata 动态启动（编译出的新内核走这条）。"""
        k = self._find(kernel)
        args = k["args"]
        layout = (ctypes.c_uint32 * (3 * len(args)))()
        for i, a in enumerate(args):
            vk = str(a.get("value_kind") or a.get("kind") or a.get(".value_kind")
                     or "by_value")
            layout[3 * i + 0] = int(a.get("off", a.get(".offset", 0)))
            layout[3 * i + 1] = int(a.get("size", a.get(".size", 0)))
            layout[3 * i + 2] = KINDS.get(vk, 0)
        a = self._pack_argv(argv)
        rc = self._lib.fm_launch_dyn(
            str(k.get("name", kernel)).encode(), gx, gy, wx, wy, layout, len(args),
            int(k.get("group_segment", 0)), int(k.get("private_segment", 0)),
            int(k.get("kernarg_size", 0)), a, len(argv))
        if rc != 0:
            raise RuntimeError(f"fm_launch_dyn({kernel}) 失败 rc={rc}")

    def launch_any(self, kernel: str, grid: int, workgroup: int, argv: list) -> None:
        """优先用动态 metadata；catalog 里没有时退回编译期内核表。"""
        try:
            self.launch_dyn(kernel, grid, 1, workgroup, 1, argv)
        except KeyError:
            self.launch(kernel, grid, workgroup, argv)

    def has(self, kernel: str) -> bool:
        """编译期内核表里有没有这个内核。"""
        try:
            self._lib.fm_has_kernel.restype = ctypes.c_int
            self._lib.fm_has_kernel.argtypes = [ctypes.c_char_p]
            return bool(self._lib.fm_has_kernel(kernel.encode()))
        except AttributeError:
            return False

    def sync(self) -> None:
        self._lib.fm_sync()
