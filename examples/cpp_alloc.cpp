// 最小 C++ 调用示例（链接 prebuilt/libfm_engine.so）。
#include "fm_engine.h"
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    const char* hsaco = argc > 1 ? argv[1] : "prebuilt/k100lc_kernels.hsaco";
    if (fm_init(hsaco) != 0) return 1;
    void* p = fm_alloc(64 * sizeof(float));
    if (!p) return 2;
    std::vector<float> h(64, 1.0f);
    fm_upload(p, h.data(), h.size() * sizeof(float));
    fm_sync();
    fm_free(p);
    std::printf("K100LC fm_engine ok\n");
    return 0;
}
