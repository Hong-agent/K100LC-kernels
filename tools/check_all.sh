#!/usr/bin/env bash
# 一条命令跑完本包的全部验证。改动内核 / 运行时 / 编译器之后都应该先过这里。
#
#   bash tools/check_all.sh          # 完整（含重建 + 产物一致性）
#   bash tools/check_all.sh --quick  # 跳过重建，只跑对账与测试
#
# 检查项：
#   1. 重新构建（baseline asm + 生成器内核 + 合并 + catalog + 引擎）
#   2. 产物一致性：重建后的 prebuilt/ + catalog + docs/KERNELS.md 必须和仓库里
#      提交的一致（防止「改了生成器忘了重建 / 忘了提交产物」）
#   3. prebuilt 内核包的全内核对账（tools/selftest_all.py，53 个用例）
#   4. 编译器回归（compiler/tests/）
#   5. 模型级端到端对账（examples/python_model_layer.py）
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

FAILED=0
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
ok()   { printf '  \033[32m%s\033[0m\n' "$1"; }
bad()  { printf '  \033[31m%s\033[0m\n' "$1"; FAILED=$((FAILED + 1)); }

# shellcheck disable=SC1091
source "$ROOT/env.sh"

if [ "$QUICK" = 0 ]; then
    step "1/5 重建内核包"
    if bash tools/build_all.sh >/tmp/k100lc_build.log 2>&1; then
        ok "$(tail -1 /tmp/k100lc_build.log)"
    else
        bad "构建失败，日志见 /tmp/k100lc_build.log"
        tail -20 /tmp/k100lc_build.log
    fi

    step "2/5 产物一致性（重建结果 == 仓库里提交的产物）"
    if git diff --quiet -- prebuilt/ python/k100lc_kernels/catalog.json docs/KERNELS.md; then
        ok "prebuilt/ + catalog + docs/KERNELS.md 与生成器一致"
    else
        bad "以下产物和生成器不一致（改了生成器但没重建/提交？）"
        git diff --stat -- prebuilt/ python/k100lc_kernels/catalog.json docs/KERNELS.md
    fi
else
    step "1-2/5 跳过重建与产物一致性（--quick）"
fi

step "3/5 prebuilt 全内核对账（重复 2 遍，顺便查非确定性）"
if python3 tools/selftest_all.py --repeat 2 --json build/selftest.json 2>&1 | tail -3 | grep -q "0 失败"; then
    ok "$(python3 -c "
import json; d = json.load(open('build/selftest.json'))
print(f\"selftest {d['total']} 个用例全部通过（{d['seconds']}s）\")")"
else
    bad "selftest 有失败项"
    python3 tools/selftest_all.py 2>&1 | grep -E "^FAIL" | head -20
fi

step "4/5 编译器回归"
if python3 compiler/tests/test_examples.py > /tmp/k100lc_cc.log 2>&1; then
    ok "$(tail -1 /tmp/k100lc_cc.log)"
else
    bad "compiler/tests/test_examples.py 失败"
    tail -10 /tmp/k100lc_cc.log
fi
if python3 compiler/tests/test_dynamic.py > /tmp/k100lc_ccd.log 2>&1; then
    ok "动态启动编译产物 ok"
else
    bad "compiler/tests/test_dynamic.py 失败"
    tail -10 /tmp/k100lc_ccd.log
fi

step "5/5 模型级端到端"
if python3 examples/python_model_layer.py --rows 4 --dim 512 --ffn 1024 \
        > /tmp/k100lc_model.log 2>&1; then
    ok "$(tail -1 /tmp/k100lc_model.log)"
else
    bad "examples/python_model_layer.py 失败"
    tail -10 /tmp/k100lc_model.log
fi

if [ "$FAILED" = 0 ]; then
    printf '\n\033[32m全部通过\033[0m\n'
    exit 0
fi
printf '\n\033[31m%d 项失败\033[0m\n' "$FAILED"
exit 1
