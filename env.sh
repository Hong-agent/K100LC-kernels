#!/usr/bin/env bash
# source env.sh 后即可直接 import k100lc_kernels / 运行示例
export K100LC_KERNELS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PYTHONPATH="$K100LC_KERNELS_ROOT/python:$K100LC_KERNELS_ROOT/compiler${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/hyhal/lib:/opt/hyhal/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
