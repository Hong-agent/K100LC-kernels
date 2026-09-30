// fm_engine.cpp —— k100lc-flashmoe 的常驻引擎（C ABI，Python 用 ctypes 驱动）。
//
// 与 `hsa_seq.py`（每次编译一个临时 C 程序）不同，这里是**一个长驻进程**：
// 显存缓冲只分配一次，51 GiB 专家权重可以常驻，逐 token 反复 launch。
//
//   fm_init(hsaco)                     初始化 HSA + 加载 HSACO
//   fm_alloc(n) -> ptr                 显存分配
//   fm_free(ptr)
//   fm_upload(ptr, host, n) / fm_download
//   fm_launch(kernel, grid, workgroup, argv, nargs)
//       argv[i] 是第 i 个**显式**参数的值：指针参数给设备地址，
//       标量参数给数值（按 metadata 里的 size 取低字节）。
//   fm_sync()
#include "hsa_rt.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int explicit_args(const RtKernel* k) {
    int n = 0;
    for (uint32_t i = 0; i < k->nargs; i++) {
        if (k->args[i].kind != 0) break;  // kind!=0：隐藏参数从这里开始
        n++;
    }
    return n;
}

}  // namespace

extern "C" {

int fm_init(const char* hsaco) {
    hsart_init(hsaco);
    return 0;
}

void* fm_alloc(size_t n) {
    void* p = nullptr;
    if (hipMalloc(&p, n) != hipSuccess) return nullptr;
    return p;
}

void fm_free(void* p) { hipFree(p); }

int fm_upload(void* dev, const void* host, size_t n) {
    return hipMemcpy(dev, host, n, hipMemcpyHostToDevice) == hipSuccess ? 0 : -1;
}

int fm_download(void* host, const void* dev, size_t n) {
    return hipMemcpy(host, dev, n, hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}

int fm_copy(void* dst, const void* src, size_t n) {
    return hipMemcpy(dst, src, n, hipMemcpyDeviceToDevice) == hipSuccess ? 0 : -1;
}

int fm_memset(void* dev, int v, size_t n) {
    return hipMemset(dev, v, n) == hipSuccess ? 0 : -1;
}

int fm_sync(void) { return hipDeviceSynchronize() == hipSuccess ? 0 : -1; }

int fm_launch2d(const char* kernel, uint32_t gx, uint32_t gy,
                uint32_t wx, uint32_t wy, const uint64_t* argv, int nargs);

int fm_launch(const char* kernel, uint32_t grid, uint32_t workgroup,
              const uint64_t* argv, int nargs) {
    return fm_launch2d(kernel, grid, 1, workgroup, 1, argv, nargs);
}

int fm_launch2d(const char* kernel, uint32_t gx, uint32_t gy,
                uint32_t wx, uint32_t wy, const uint64_t* argv, int nargs) {
    const RtKernel* k = hsart_lookup(kernel);
    if (!k) {
        fprintf(stderr, "fm_engine: 没有内核 %s\n", kernel);
        return -1;
    }
    const int n = explicit_args(k);
    if (n != nargs) {
        fprintf(stderr, "fm_engine: %s 显式参数 %d 个，调用给了 %d 个\n", kernel, n, nargs);
        return -2;
    }
    char kernarg[MAX_KERNARG];
    memset(kernarg, 0, k->kernarg_size);
    for (int i = 0; i < n; i++) {
        size_t sz = k->args[i].size;
        if (sz > 8) sz = 8;
        memcpy(kernarg + k->args[i].off, &argv[i], sz);
    }
    hsart_dispatch(k, dim3(gx, gy), dim3(wx, wy), 0, kernarg, k->kernarg_size);
    return 0;
}

int fm_launch_dyn(const char* kernel, uint32_t gx, uint32_t gy,
                  uint32_t wx, uint32_t wy,
                  const uint32_t* layout, uint32_t nargs,
                  uint32_t group_size, uint32_t private_size,
                  uint32_t kernarg_size, const uint64_t* argv, int nargv) {
    if (kernarg_size > MAX_KERNARG) return -3;
    std::vector<RtArg> args(nargs);
    int explicit_n = 0;
    for (uint32_t i = 0; i < nargs; i++) {
        args[i].off = layout[3 * i + 0];
        args[i].size = layout[3 * i + 1];
        args[i].kind = (uint8_t)layout[3 * i + 2];
        if (args[i].kind == 0) explicit_n++;
    }
    if (explicit_n != nargv) return -2;
    char kernarg[MAX_KERNARG];
    memset(kernarg, 0, kernarg_size);
    int ai = 0;
    for (uint32_t i = 0; i < nargs; i++) {
        if (args[i].kind != 0) continue;
        size_t sz = args[i].size;
        if (sz > 8) sz = 8;
        memcpy(kernarg + args[i].off, &argv[ai++], sz);
    }
    hsart_dispatch_dyn(kernel, dim3(gx, gy), dim3(wx, wy), 0,
                       kernarg, kernarg_size, args.data(), nargs,
                       group_size, private_size);
    return 0;
}

}  // extern "C"
