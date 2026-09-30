# K100LC-kernels

海光 **K100_LC (gfx926, wave64)** 的可复用 GPU 内核包：自研汇编内核 +
FASTASM 基线内核 + 无 DTK 的常驻 HSA 运行时。后续项目直接引用本目录即可，
不需要再复制 `asm.py`、HSACO 管线或 HSA 垫片。

当前包含 **96 个内核**（81 个 FASTASM 基线 + 15 个自研），预编译产物在
`prebuilt/`；内核目录在 `python/k100lc_kernels/catalog.json`。

## 目录

| 路径 | 内容 |
|---|---|
| `asm.py` / `encodings.json` | 自研表驱动 gfx926 汇编器 |
| `kernels/asm/` | 81 个基线 `.s`（FASTASM） |
| `tools/gen_*.py` | 10 个自研内核生成器（GEMV / dequant / IQ→i8） |
| `runtime/hsa_rt.{h,cpp}` | 无 DTK HSA 运行时垫片 |
| `runtime/fm_engine.{h,cpp}` | 常驻引擎 C ABI |
| `python/k100lc_kernels/` | Python 封装 + 内核目录 |
| `compiler/` | 受限 Python DSL → gfx926 汇编 → HSACO 的编译器 |
| `prebuilt/` | `k100lc_base.hsaco`（81 基线）、`k100lc_kernels.hsaco`（91 全量）、`libfm_engine.so`、`nodtk_kernels.h` |
| `docs/ABI.md` | 参数布局、grid 语义、加内核/更新流程 |

## 快速调用（Python）

```bash
source env.sh
python3 -m k100lc_kernels list gemv
python3 examples/python_gemv.py
```

```python
from k100lc_kernels import Runtime, kernels, info

rt = Runtime()                    # 默认加载 prebuilt/k100lc_kernels.hsaco
p = rt.alloc(1024)
rt.upload(p, bytes(1024))
rt.launch("fill_k", 1, 64, [p, 1.0, 256])   # grid=workgroup 个数！
rt.sync()
print(info("gemv_f32_warp_k"))
```

任意 HSACO（包括编译器新产出的内核）可以用动态 metadata 启动，不必重建
`libfm_engine.so`：

```python
rt = Runtime(hsaco="build/my.hsaco", catalog="build/my.catalog.json")
rt.launch_dyn("my_kernel", grid_x, 1, workgroup_x, 1, [arg0, arg1, ...])
```

`Runtime.launch_dyn` 的 catalog 也可以放在 HSACO 旁边
（`<name>.catalog.json`，编译器会自动生成）。当前 HSA 路径**一个进程只加载
一个 HSACO**；多个 HSACO 先用 `tools/merge_hsacos.py` 合并，或分进程使用。

## 快速调用（C++）

```cpp
#include "fm_engine.h"
fm_init("prebuilt/k100lc_kernels.hsaco");
void* p = fm_alloc(1024);
fm_free(p);
```

编译：

```bash
g++ -O2 -std=c++17 -I runtime examples/cpp_alloc.cpp \
    -L prebuilt -lfm_engine -L/opt/hyhal/lib -lhsa-runtime64 \
    -Wl,-rpath,$PWD/prebuilt -Wl,-rpath,/opt/hyhal/lib -o /tmp/cpp_alloc
LD_LIBRARY_PATH=/opt/hyhal/lib:$LD_LIBRARY_PATH /tmp/cpp_alloc prebuilt/k100lc_kernels.hsaco
```

## 两种 grid 语义（重要）

| 入口 | `grid` 含义 |
|---|---|
| `Runtime.launch` / `fm_launch` / `Engine.launch` | **workgroup 个数**（块数） |
| `tools/kernel_lab.run_one` / `hsa_job` | **总 work-item 数**，内部自己除 workgroup |

内核里 `v0 = tid`、`s6 = blockIdx.x`（2D 时 `blockIdx.y` 来自 `launch2d`
的 `gy`）。写内核测试时最容易在这里踩坑。

## 重建

```bash
bash tools/build_all.sh
```

产物：`build/k100lc_kernels.hsaco`（91 内核）、`build/libfm_engine.so`、
`python/k100lc_kernels/catalog.json`，并同步到 `prebuilt/`。

## 内核编译器

```bash
python3 -m k100lc_compiler build compiler/examples/silu.kkl \
    -o /tmp/kbuild --emit-asm
python3 compiler/tests/test_examples.py
```

语言子集、ABI、已知 hazard 规则见 `compiler/README.md`。

## 更新（从其他项目吸收新内核）

```bash
bash tools/update_from.sh /home/t/桌面/k100lc-flashmoe
```

脚本会同步 `kernels/asm/`、`kernels/kernel_spec.json`、`tools/gen_*.py` 和
关键运行时文件，然后重建。新增/修改内核后请更新 `CHANGELOG.md` 和
`python/k100lc_kernels/catalog.json`（`build_all.sh` 会重生成 catalog）。

## 已知限制

- 基线 C 内核里 **private_segment > 0** 的（例如 `gdn_k2<32>`、`gdn_k`）
  需要 HSA queue scratch backing；当前运行时没有接通 scratch，直接调用会
  fault。调用前请用 `kernels()` 里的 `private_segment` 字段判断。
- 当前引擎的 `grid` 是块数；不要照搬 CUDA 的“总 work-item”写法。
- 只支持 `/opt/hyhal` 的 HSA 运行时；主机需能访问 `/dev/kfd` 与 DCU。

## 许可

本包代码 Apache-2.0（见 `LICENSE`）；i-quant 码本/掩码/块布局来自
llama.cpp / ggml（MIT），FASTASM 基线内核来自本机 `K100LC-FASTASM-NVFP4`
项目，第三方说明见 `NOTICE`。
