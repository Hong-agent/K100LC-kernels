#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为「引擎直跑原生 int4」生成权重清单（TSV）。

清单里的每条 = 一个 `*.weight_packed` / `*.weight_scale` 对在 checkpoint 文件里的
**绝对字节偏移**。调用方可以按这份清单把权重按层预取到显存，然后交给本包的
`int4_dot_k` 直接吃 —— 不转换、不重量化。

格式（与引擎 `src/model.cpp` 里 `read_int4_rows()` 对应，列用空白分隔）：

    # name  packed_off  packed_bytes  scale_off  scale_bytes  N  K

`name` 去掉 `.weight_packed` 后缀，方便调用方复用同一套「名字 → (层号, 槽位)」
解析。首行 `# source <文件字节数>` 是自检：偏移是**文件绝对偏移**，换了
checkpoint 必须整份作废。

    python3 tools/int4_engine_manifest.py -o /tmp/int4_manifest.tsv
    python3 tools/int4_engine_manifest.py --model-dir /path/to/checkpoint
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from ct_int4 import DEFAULT_MODEL_DIR, GROUP, read_header  # noqa: E402


def build_rows(model_dir: pathlib.Path) -> list[str]:
    """返回清单行（不含表头）。偏移是**文件绝对偏移**（含 safetensors 头）。"""
    path = model_dir / "model.safetensors"
    hdr = read_header(path)
    hlen = 8 + _header_len(path)
    out: list[str] = []
    for name in sorted(hdr):
        if not name.endswith(".weight_packed"):
            continue
        base = name[: -len(".weight_packed")]
        scale = base + ".weight_scale"
        if scale not in hdr:
            raise KeyError(f"{base} 缺少 weight_scale")
        p, s = hdr[name], hdr[scale]
        n, k8 = (int(v) for v in p["shape"])
        nbpr = int(s["shape"][1])
        k = k8 * 8
        if nbpr * GROUP != k:
            raise ValueError(f"{base}: scale 形状与 K 不符（{s['shape']} vs K={k}）")
        for e, what in ((p, "packed"), (s, "scale")):
            e["data_offsets"] = list(e["data_offsets"])
        out.append(" ".join(str(v) for v in (
            base,
            hlen + p["data_offsets"][0], p["data_offsets"][1] - p["data_offsets"][0],
            hlen + s["data_offsets"][0], s["data_offsets"][1] - s["data_offsets"][0],
            n, k)))
    return out


_HDR_LEN: dict[str, int] = {}


def _header_len(path: pathlib.Path) -> int:
    key = str(path)
    if key not in _HDR_LEN:
        import struct
        with open(path, "rb") as f:
            _HDR_LEN[key] = struct.unpack("<Q", f.read(8))[0]
    return _HDR_LEN[key]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("-o", "--out", type=pathlib.Path, default=pathlib.Path("int4_manifest.tsv"))
    args = ap.parse_args()

    rows = build_rows(args.model_dir)
    size = (args.model_dir / "model.safetensors").stat().st_size
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# source {size}\n")
        f.write("# name\tpacked_off\tpacked_bytes\tscale_off\tscale_bytes\tN\tK\n")
        for r in rows:
            f.write(r + "\n")
    tot_p = sum(int(r.split()[2]) for r in rows)
    tot_s = sum(int(r.split()[4]) for r in rows)
    print(f"-> {args.out}: {len(rows)} 个 int4 线性层，"
          f"packed {tot_p/1e9:.3f} GB + scale {tot_s/1e9:.3f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
