from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
CATALOG = ROOT / "python" / "k100lc_kernels" / "catalog.json"


def load() -> list[dict]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))["kernels"]


def kernels(pattern: str = "") -> list[dict]:
    items = load()
    if pattern:
        items = [k for k in items if pattern in k["lookup"]]
    return items


def info(name: str) -> dict:
    for k in load():
        if k["lookup"] == name or k["name"] == name:
            return k
    raise KeyError(name)
