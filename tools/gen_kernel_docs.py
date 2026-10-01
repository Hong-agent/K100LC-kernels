#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 `python/k100lc_kernels/catalog.json` 生成 `docs/KERNELS.md`。

    python3 tools/gen_kernel_docs.py

`catalog.json` 由 `tools/make_catalog.py` 从 HSACO metadata 生成，是运行时
启动内核时使用的唯一参数来源。这个脚本把它展开成可读的逐内核参数表，
避免文档与预编译产物不一致。
"""
from __future__ import annotations

import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG = ROOT / "python" / "k100lc_kernels" / "catalog.json"
OUT = ROOT / "docs" / "KERNELS.md"
HSACO = ROOT / "prebuilt" / "k100lc_kernels.hsaco"

CATEGORIES = [
    ("NVFP4", lambda n: n.startswith("nvfp4_")),
    ("INT4（RT4 / compressed-tensors）",
     lambda n: n.startswith(("int4_", "gemv_w4a4", "gemv_w4a8",
                             "gemm_w4a4", "quant_act", "quant_rows"))),
    ("融合点积（原生量化块 × 激活）",
     lambda n: n.endswith("_dot_k") or n == "reduce_blocks_k"),
    ("量化解码", lambda n: ("dequant" in n) or n.endswith("_to_i8_k")),
    ("GEMV / GEMM", lambda n: n.startswith(("gemv_", "gemm_"))),
    ("Transformer 常用算子",
     lambda n: n.startswith(("rmsnorm", "layernorm", "softmax", "topk",
                             "router_", "moe_", "rope", "embed", "silu", "gelu",
                             "sigmoid", "l2norm", "scale_", "fill_", "add_",
                             "concat", "argmax"))),
    ("Attention / KV / 视觉塔",
     lambda n: n.startswith(("fa_", "kv_", "attn_", "gather_", "scatter_",
                             "vit_"))),
    ("序列模型 / 卷积",
     lambda n: n.startswith(("gdn", "conv", "ssm", "split_qkv"))),
]


def category(name: str) -> str:
    for label, pred in CATEGORIES:
        if pred(name):
            return label
    return "其他"


def arg_short(a: dict) -> str:
    off = int(a.get("off", a.get(".offset", 0)))
    size = int(a.get("size", a.get(".size", 0)))
    kind = str(a.get("value_kind") or a.get("kind") or a.get(".value_kind")
               or "by_value")
    tag = {"global_buffer": "buf", "by_value": "val"}.get(kind, kind)
    return f"{tag}:{off}:{size}"


def kind_of(a: dict) -> str:
    return str(a.get("value_kind") or a.get("kind") or a.get(".value_kind")
               or "by_value")


def explicit_args(k: dict) -> list[dict]:
    return [a for a in k["args"] if not kind_of(a).startswith("hidden_")]


def main() -> int:
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    kernels = sorted(cat["kernels"], key=lambda k: k["lookup"])
    try:
        demangled = subprocess.run(
            ["c++filt", *[k["name"] for k in kernels]],
            capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        demangled = [k["name"] for k in kernels]
    signatures = {k["name"]: s for k, s in zip(kernels, demangled)}
    groups: dict[str, list[dict]] = {}
    for k in kernels:
        groups.setdefault(category(k["lookup"]), []).append(k)

    lines: list[str] = []
    lines.append("# 内核参数总表")
    lines.append("")
    lines.append("本文件由 `tools/gen_kernel_docs.py` 从 "
                 "[`python/k100lc_kernels/catalog.json`](../python/k100lc_kernels/catalog.json) "
                 "自动生成，请勿手工修改。")
    lines.append("")
    lines.append(f"预编译 HSACO：`prebuilt/{HSACO.name}`；"
                 f"共 **{len(kernels)} 个内核**。")
    lines.append("")
    lines.append("调用方法、grid 语义和完整示例先看 "
                 "[`docs/KERNEL_CALLING.md`](KERNEL_CALLING.md)。"
                 "本页只回答「某个 lookup 到底要传哪些参数」。")
    lines.append("")
    lines.append("## 分类统计")
    lines.append("")
    lines.append("| 分类 | 内核数 |")
    lines.append("|---|---:|")
    for label, _ in CATEGORIES:
        if label in groups:
            lines.append(f"| {label} | {len(groups[label])} |")
    if "其他" in groups:
        lines.append(f"| 其他 | {len(groups['其他'])} |")
    lines.append(f"| **合计** | **{len(kernels)}** |")
    lines.append("")
    lines.append("## 速查表")
    lines.append("")
    lines.append("`argv` 只包含下表列出的**显式参数**，按顺序传给 "
                 "`Runtime.launch(...)` / `fm_launch(...)`。")
    lines.append("`buf:偏移:字节` = 设备指针（u64），`val:偏移:字节` = 标量")
    lines.append("（参数类型由内核签名决定；`Runtime` 对 Python `float` 按 f32 打包）。")
    lines.append("")
    lines.append("| lookup | 显式参数 | kernarg | LDS | private |")
    lines.append("|---|---|---:|---:|---:|")
    for k in kernels:
        args = " ".join(arg_short(a) for a in explicit_args(k))
        lines.append(f"| `{k['lookup']}` | `{args or '-'}` | "
                     f"{k['kernarg_size']} | {k.get('group_segment', 0)} | "
                     f"{k.get('private_segment', 0)} |")
    lines.append("")
    lines.append("## 逐内核参数")
    lines.append("")
    for label, _ in CATEGORIES:
        if label not in groups:
            continue
        lines.append(f"### {label}")
        lines.append("")
        for k in groups[label]:
            lines.append(f"#### `{k['lookup']}`")
            lines.append("")
            lines.append(f"- mangled 符号：`{k['name']}`")
            lines.append(f"- 签名：`{signatures.get(k['name'], k['name'])}`")
            lines.append(f"- kernarg：{k['kernarg_size']} B，"
                         f"LDS：{k.get('group_segment', 0)} B，"
                         f"private：{k.get('private_segment', 0)} B")
            lines.append("")
            lines.append("| # | offset | size | kind | 由谁填 |")
            lines.append("|---:|---:|---:|---|---|")
            for i, a in enumerate(k["args"]):
                kind = kind_of(a)
                off = int(a.get("off", a.get(".offset", 0)))
                size = int(a.get("size", a.get(".size", 0)))
                who = "运行时" if kind.startswith("hidden_") else "调用方"
                lines.append(f"| {i} | {off} | {size} | `{kind}` | {who} |")
            lines.append("")
    if "其他" in groups:
        lines.append("### 其他")
        lines.append("")
        for k in groups["其他"]:
            args = " ".join(arg_short(a) for a in explicit_args(k))
            lines.append(f"- `{k['lookup']}`：`{args}`")
        lines.append("")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {OUT}（{len(kernels)} 个内核）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
