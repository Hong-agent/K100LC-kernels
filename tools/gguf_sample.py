#!/usr/bin/env python3
"""定位可选的真权重索引，供各生成器的自检使用。

索引是一个 JSON 文件，描述若干 GGUF/量化张量在原始权重文件里的位置：

    {
      "sources": {"0": {"path": "/path/to/weights.gguf"}},
      "tensors": [
        {"name": "...", "type": "Q4_K", "shape": [K, N],
         "bytes": 123, "source": "0", "offset": 456}
      ]
    }

优先级：

1. 环境变量 `K100LC_GGUF_INDEX` 指向的 JSON；
2. 仓库内 `build/gguf_index.json`。

没有索引时，生成器自身的 `main()` 自检会给出明确提示；这不影响
`tools/build_all.sh` 的构建流程，因为构建只使用生成器里的合成权重。
"""
from __future__ import annotations

import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "build" / "gguf_index.json"
ENV_VAR = "K100LC_GGUF_INDEX"


def index_path() -> pathlib.Path:
    p = pathlib.Path(os.environ.get(ENV_VAR) or DEFAULT_INDEX)
    if not p.is_file():
        raise SystemExit(
            f"找不到权重索引 {p}。\n"
            f"用 {ENV_VAR}=/path/to/index.json 指定，或把索引放在 {DEFAULT_INDEX}。")
    return p


def load_index() -> dict:
    return json.loads(index_path().read_text(encoding="utf-8"))
