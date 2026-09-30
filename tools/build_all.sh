#!/usr/bin/env bash
# 重建 81 个基线内核 + 10 个自研内核 → 91 内核 HSACO + 常驻引擎 + Python 目录。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p build prebuilt python/k100lc_kernels

python3 tools/build_kernels.py --out-dir build
python3 tools/make_hsaco_multi.py build/all_kernels_spec.json build/k100lc_base.hsaco
python3 tools/build_flashmoe.py
python3 tools/merge_hsacos.py build/k100lc_base.hsaco build/flashmoe.hsaco \
    -o build/k100lc_kernels.hsaco
python3 tools/gen_kernel_table.py build/k100lc_kernels.hsaco build/nodtk_kernels.h
python3 tools/make_catalog.py build/k100lc_kernels.hsaco python/k100lc_kernels/catalog.json

g++ -O2 -std=c++17 -fPIC -shared \
    -I runtime -I build -I /opt/hyhal/include \
    runtime/fm_engine.cpp runtime/hsa_rt.cpp \
    -o build/libfm_engine.so \
    -L/opt/hyhal/lib -lhsa-runtime64 -lpthread

cp -f build/k100lc_kernels.hsaco prebuilt/k100lc_kernels.hsaco
cp -f build/nodtk_kernels.h prebuilt/nodtk_kernels.h
cp -f build/libfm_engine.so prebuilt/libfm_engine.so
echo "-> prebuilt/k100lc_kernels.hsaco + prebuilt/libfm_engine.so"
