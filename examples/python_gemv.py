#!/usr/bin/env python3
"""最小 Python 调用示例：gemv_f32_warp_k = W[n,k]·x[k]。"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from k100lc_kernels import Runtime  # noqa: E402


def main() -> int:
    n, k = 256, 640
    rng = np.random.default_rng(0)
    w = rng.standard_normal((n, k), dtype=np.float32)
    x = rng.standard_normal(k, dtype=np.float32)
    rt = Runtime()
    pw, px, py = rt.alloc(w.nbytes), rt.alloc(x.nbytes), rt.alloc(n * 4)
    rt.upload(pw, w)
    rt.upload(px, x)
    # 注意：Engine/fm_launch 的 grid 是 workgroup 个数，不是总 work-item 数
    rt.launch("gemv_f32_warp_k", n, 64, [pw, px, py, n, k, 64])
    rt.sync()
    got = rt.download(py, n, np.float32)
    ref = w @ x
    print("max_abs", float(np.abs(got - ref).max()))
    for p in (pw, px, py):
        rt.free(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
