#!/usr/bin/env bash
# 从已有项目同步内核/生成器，然后重建并刷新预编译产物。
#   bash tools/update_from.sh /home/t/桌面/k100lc-flashmoe
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:?用法: update_from.sh <project-root>}"

if [ -d "$SRC/kernels/asm" ]; then
    cp -a "$SRC/kernels/asm/." "$ROOT/kernels/asm/"
fi
if [ -f "$SRC/kernels/kernel_spec.json" ]; then
    cp -f "$SRC/kernels/kernel_spec.json" "$ROOT/kernels/kernel_spec.json"
fi
for f in "$SRC"/tools/gen_*.py; do
    [ -f "$f" ] && cp -f "$f" "$ROOT/tools/"
done
for f in build_flashmoe.py fm_engine.py amdgpu_meta.py kernel_lab.py hsa_job.py; do
    [ -f "$SRC/tools/$f" ] && cp -f "$SRC/tools/$f" "$ROOT/tools/"
done
bash "$ROOT/tools/build_all.sh"
echo "已从 $SRC 同步并重建；请更新 CHANGELOG.md"
