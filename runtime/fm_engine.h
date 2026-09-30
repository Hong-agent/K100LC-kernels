// 常驻引擎 C ABI（libfm_engine.so）
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int  fm_init(const char* hsaco);
void* fm_alloc(size_t n);
void  fm_free(void* p);
int  fm_upload(void* dev, const void* host, size_t n);
int  fm_download(void* host, const void* dev, size_t n);
int  fm_copy(void* dst, const void* src, size_t n);
int  fm_memset(void* dev, int v, size_t n);
int  fm_sync(void);
// grid/wg 都是 workgroup 的个数/线程数（不是总 work-item 数）
int  fm_launch(const char* kernel, uint32_t grid, uint32_t workgroup,
               const uint64_t* argv, int nargs);
int  fm_launch2d(const char* kernel, uint32_t gx, uint32_t gy,
                 uint32_t wx, uint32_t wy, const uint64_t* argv, int nargs);

#ifdef __cplusplus
}
#endif
