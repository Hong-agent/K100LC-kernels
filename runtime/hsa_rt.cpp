// hsa_rt.cpp —— 用 /opt/hyhal 的 HSA 实现本包需要的 HIP 子集。
// 内核来自自研 HSACO：优先 $RT_HSACO，其次 build/k100lc_kernels.hsaco，
// 再退回可执行文件旁边的 prebuilt/（源码包里带了预编译版本，解压即可跑）。
//
// 语义取舍（都对「跑通」友好，性能上标明代价）：
//   * 拷贝：H2D 走 hsa_amd_memory_async_copy（源先 hsa_memory_register，否则退回
//     hsa_memory_copy）；D2H/D2D 因为要读「可能刚被内核写过」的设备内存，先做一次
//     全设备同步再拷 —— 比 HIP 的流序保守，但绝对安全。
//   * 事件：本实现里拷贝在调用线程上就是同步完成的，所以「记录」= 置一个序号，
//     「等事件」= 自旋等序号到位（预取线程与主线程之间靠它排序）。
//     hipEventElapsedTime 因此返回 0（本包不支持事件计时）。
//   * 内核投递：单队列、in-order；kernarg 用 64 个槽轮转，复用前等该槽上一次完成。
#include "hsa_rt.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/amd_hsa_queue.h>      // amd_queue_t：看 scratch 描述符是否被挂上

#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <ctime>
#include <unordered_map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef RT_HSACO_DEFAULT
#define RT_HSACO_DEFAULT "build/k100lc_kernels.hsaco"
#endif

#define HSA_CHECK(expr) do { hsa_status_t st_ = (expr); if (st_ != HSA_STATUS_SUCCESS) { \
    const char* s_ = nullptr; hsa_status_string(st_, &s_); \
    fprintf(stderr, "hsa_rt: %s:%d %s -> %s\n", __FILE__, __LINE__, #expr, s_ ? s_ : "?"); \
    exit(1); } } while (0)

// 头里只前置声明；定义放全局，保证 hipEvent_t/hipStream_t 的指针语义与 HIP 一致
struct EventImpl {
    std::atomic<uint64_t> seq{0};
};
struct StreamImpl { int dummy = 0; };

namespace {

// kernarg 槽数 / 队列深度。投递路径的门槛在**硬件**：这批内核每条要 ~7.3 us
// 才 retire（v1.9.9 量过：把槽加到 1024、把每条的完成信号去掉、换队列类型，
// 都不动它）。所以这里维持 64 就好——加大只多占 pinned 内存。
// 注意 `make_queue` 里 `qsize = min(设备上限, N_SLOT)`：只有两者相等时
// 「槽号 == 环上位置」这个前提才成立，改一个就得改另一个。
constexpr int N_SLOT = 64;

// 诊断用：`RT_HSART_PROF=1` 时把批量投递的内部分段耗时打到 stderr。
// （一度以为「每次 dispatch 要 7 us」是 GPU 的固定开销，量下来其实是
// 主机侧 `hsart_launch_batch` 里每条记录的填包/等待，见 CHANGELOG 1.9.9。）
double now_us() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e6 + (double)t.tv_nsec / 1e3;
}
const bool g_prof = getenv("RT_HSART_PROF") != nullptr;
double g_pf_wait_us = 0, g_pf_fill_us = 0, g_pub_us = 0;
long   g_pf_wait_n = 0;

hsa_agent_t g_gpu{};
std::vector<hsa_agent_t> g_cpus;
hsa_amd_memory_pool_t g_data_pool{};
hsa_amd_memory_pool_t g_karg_pool{};
hsa_amd_memory_pool_t g_host_pool{};        // CPU 侧（pinned）——拷贝的暂存区
hsa_status_t g_host_pool_st = HSA_STATUS_ERROR;
hsa_queue_t* g_queue = nullptr;
hsa_executable_t g_exec{};
uint64_t g_kobj[k_table_n];
std::unordered_map<std::string, uint64_t> g_dyn_kobj;
std::unordered_map<std::string, uint64_t> g_sym_kobj;

void*     g_karg[N_SLOT];
hsa_signal_t g_sig[N_SLOT];
bool      g_slot_used[N_SLOT];
int       g_next = 0;
// 自上次「全同步」以来投递过、还没等过的槽数。同步只需要等这几只，
// 不必每次扫满 N_SLOT 个信号（之前每次 sync 都要过一遍 64 个信号，
// 实测单次 sync 因此要 ~10 us）。
int       g_pending = 0;

// 内核名 → 表项 的哈希缓存：hsart_lookup 被每次 launch 调用，原来是对
// k_table（122 项）做线性扫描 + std::string 比较，实测吃掉每次 launch
// 好几微秒。
std::unordered_map<std::string, const RtKernel*> g_lookup;

std::mutex g_mtx;

std::vector<std::pair<const void*, size_t>> g_reg;      // 已注册的主机区间

void check_hsa(hsa_status_t st, const char* what) {
    if (st != HSA_STATUS_SUCCESS) {
        const char* s = nullptr;
        hsa_status_string(st, &s);
        fprintf(stderr, "hsa_rt: %s -> %s\n", what, s ? s : "?");
        exit(1);
    }
}

hsa_status_t cb_agent(hsa_agent_t agent, void*) {
    hsa_device_type_t type{};
    HSA_CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type));
    if (type == HSA_DEVICE_TYPE_GPU) {
        g_gpu = agent;
    } else if (type == HSA_DEVICE_TYPE_CPU) {
        g_cpus.push_back(agent);
    }
    return HSA_STATUS_SUCCESS;
}

hsa_status_t cb_pool(hsa_amd_memory_pool_t pool, void*) {
    hsa_amd_segment_t seg{};
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    uint32_t flags = 0;
    bool alloc_ok = false;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_ok);
    if (!alloc_ok) return HSA_STATUS_SUCCESS;
    static bool have_karg = false, have_data = false;
    if (!have_karg && (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)) {
        g_karg_pool = pool; have_karg = true;
    }
    if (!have_karg && (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
        g_karg_pool = pool; have_karg = true;
    }
    if (!have_data && (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)) {
        g_data_pool = pool; have_data = true;
    }
    return HSA_STATUS_SUCCESS;
}

void* pool_alloc(hsa_amd_memory_pool_t pool, size_t n) {
    void* p = nullptr;
    check_hsa(hsa_amd_memory_pool_allocate(pool, n, 0, &p), "pool_allocate");
    std::vector<hsa_agent_t> agents;
    agents.push_back(g_gpu);
    for (hsa_agent_t c : g_cpus) agents.push_back(c);
    check_hsa(hsa_amd_agents_allow_access((uint32_t)agents.size(), agents.data(), nullptr, p),
              "allow_access");
    return p;
}

hsa_status_t cb_host_pool(hsa_amd_memory_pool_t pool, void*) {
    if (g_host_pool_st == HSA_STATUS_SUCCESS) return HSA_STATUS_SUCCESS;
    hsa_amd_segment_t seg{};
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    bool alloc_ok = false;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_ok);
    if (!alloc_ok) return HSA_STATUS_SUCCESS;
    g_host_pool = pool;
    g_host_pool_st = HSA_STATUS_SUCCESS;
    return HSA_STATUS_SUCCESS;
}

hsa_status_t cb_pool_ok(hsa_status_t st) {
    return st == HSA_STATUS_INFO_BREAK ? HSA_STATUS_SUCCESS : st;
}

// 拷贝暂存区：主机侧 pinned 内存。hsa_amd_memory_async_copy 的源/目的必须
// 是 agent 可访问的（页锁定）内存 —— 直接拿 malloc/栈地址去拷会报
// "Invalid address access"，所以统一过一次暂存区。
void*  g_stage = nullptr;
size_t g_stage_sz = 0;
std::mutex g_stage_mtx;

// 单次暂存上限。之前 stage(n) 会跟着最大一次上传（例如 2.5 GB 的
// output.weight f32）永久膨胀，把 7.5 GB 主机内存挤到 swap；改成固定
// 8 MB 分块后，主机侧 pinned 内存与模型大小无关。
constexpr size_t STAGE_CHUNK = 8u << 20;

void* stage(size_t n) {
    if (n > g_stage_sz) {
        if (g_stage) hsa_amd_memory_pool_free(g_stage);
        g_stage = pool_alloc(g_host_pool, n);
        g_stage_sz = n;
    }
    return g_stage;
}

hsa_status_t get_symbol(const char* name, hsa_agent_t agent, hsa_executable_symbol_t* out) {
    hsa_status_t st = hsa_executable_get_symbol_by_name(g_exec, name, &agent, out);
    if (st == HSA_STATUS_SUCCESS) return st;
    std::string alt = std::string("&") + name;
    st = hsa_executable_get_symbol_by_name(g_exec, alt.c_str(), &agent, out);
    if (st == HSA_STATUS_SUCCESS) return st;
    alt = std::string(name) + ".kd";
    return hsa_executable_get_symbol_by_name(g_exec, alt.c_str(), &agent, out);
}

hsa_status_t cb_symbol(hsa_executable_t, hsa_executable_symbol_t sym, void*) {
    hsa_symbol_kind_t kind{};
    if (hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind)
        != HSA_STATUS_SUCCESS || kind != HSA_SYMBOL_KIND_KERNEL) {
        return HSA_STATUS_SUCCESS;
    }
    uint32_t len = 0;
    if (hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &len)
        != HSA_STATUS_SUCCESS) {
        return HSA_STATUS_SUCCESS;
    }
    std::string name(len ? len - 1 : 0, '\0');
    if (len && hsa_executable_symbol_get_info(
            sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data()) != HSA_STATUS_SUCCESS) {
        return HSA_STATUS_SUCCESS;
    }
    uint64_t obj = 0;
    if (hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &obj)
        == HSA_STATUS_SUCCESS) {
        g_sym_kobj[name] = obj;
    }
    return HSA_STATUS_SUCCESS;
}

void load_hsaco(const char* path) {
    static std::string alt_path;              // 命中备选路径时给它一个长生命周期
    FILE* f = fopen(path, "rb");
    if (!f) {
        // 再试几个常见位置（相对当前目录 + 相对可执行文件所在目录）。
        char exe_dir[4096] = "";
        if (ssize_t n = readlink("/proc/self/exe", exe_dir, sizeof(exe_dir) - 1); n > 0) {
            exe_dir[n] = '\0';
            if (char* s = strrchr(exe_dir, '/')) *s = '\0'; else exe_dir[0] = '\0';
        }
        std::vector<std::string> alts = {
            "build/k100lc_kernels.hsaco", "prebuilt/k100lc_kernels.hsaco",
            "./k100lc_kernels.hsaco"};
        if (exe_dir[0]) {
            alts.push_back(std::string(exe_dir) + "/k100lc_kernels.hsaco");
            alts.push_back(std::string(exe_dir) + "/../build/k100lc_kernels.hsaco");
            alts.push_back(std::string(exe_dir) + "/../prebuilt/k100lc_kernels.hsaco");
        }
        for (const std::string& a : alts)
            if ((f = fopen(a.c_str(), "rb"))) { alt_path = a; path = alt_path.c_str(); break; }
    }
    if (!f) { fprintf(stderr, "hsa_rt: 打不开自研 HSACO（%s）\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* image = malloc((size_t)sz);
    if (fread(image, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "hsa_rt: HSACO 读失败\n"); exit(1); }
    fclose(f);
    check_hsa(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                        nullptr, &g_exec), "executable_create");
    hsa_code_object_reader_t reader;
    check_hsa(hsa_code_object_reader_create_from_memory(image, (size_t)sz, &reader),
              "code_object_reader");
    check_hsa(hsa_executable_load_agent_code_object(g_exec, g_gpu, reader, nullptr, nullptr),
              "load_agent_code_object");
    check_hsa(hsa_executable_freeze(g_exec, nullptr), "executable_freeze");
    g_sym_kobj.clear();
    HSA_CHECK(hsa_executable_iterate_symbols(g_exec, cb_symbol, nullptr));
    if (getenv("RT_HSART_DEBUG")) {
        fprintf(stderr, "hsa_rt: symbols:");
        for (const auto& kv : g_sym_kobj) fprintf(stderr, " %s", kv.first.c_str());
        fprintf(stderr, "\n");
    }
    int resolved = 0;
    for (int i = 0; i < k_table_n; i++) {
        hsa_executable_symbol_t sym;
        if (get_symbol(k_table[i].name, g_gpu, &sym) != HSA_STATUS_SUCCESS) {
            g_kobj[i] = 0;          // HSACO 里没有这个编译期内核：动态路径仍可用
            continue;
        }
        HSA_CHECK(hsa_executable_symbol_get_info(
            sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &g_kobj[i]));
        resolved++;
    }
    fprintf(stderr, "hsa_rt: 自研 HSACO %s 已加载（编译期内核 %d/%d，动态内核走 lookup）\n",
            path, resolved, k_table_n);
}

static void report_scratch(const char* when);      // 定义见下方「scratch 诊断」

void make_queue() {
    uint32_t qsize = 0;
    HSA_CHECK(hsa_agent_get_info(g_gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &qsize));
    if (qsize > (uint32_t)N_SLOT) qsize = N_SLOT;
    const char* qenv = getenv("RT_HSART_QMULTI");
    const hsa_queue_type_t qtype = (qenv && atoi(qenv)) ? HSA_QUEUE_TYPE_MULTI : HSA_QUEUE_TYPE_SINGLE;
    HSA_CHECK(hsa_queue_create(g_gpu, qsize, qtype, nullptr, nullptr,
                               UINT32_MAX, UINT32_MAX, &g_queue));
    if (getenv("RT_HSART_SCRATCH_INFO")) report_scratch("queue_create 之后");
    for (int i = 0; i < N_SLOT; i++) {
        g_karg[i] = pool_alloc(g_karg_pool, MAX_KERNARG);
        HSA_CHECK(hsa_signal_create(1, 0, nullptr, &g_sig[i]));
        g_slot_used[i] = false;
    }
}

// ============================ scratch 诊断 ============================
// private_segment > 0 的内核（gdn_k / gdn_k2<32> / fa_int4 / vit_attn_kernel）
// 需要 HSA 队列给它们准备 scratch backing。hyhal 这份 libhsa-runtime64 既没有
// 导出 `hsa_amd_queue_set_scratch_allocator`，头文件里也没有它的声明，队列创建
// 后 `amd_queue_t` 里的 scratch 描述符是全零——也就是说这套运行时没接 scratch。
// 这里把这几个字段读出来，方便判断「内核能不能跑」而不是让它直接 fault。
static amd_queue_t* amdq() { return reinterpret_cast<amd_queue_t*>(g_queue); }

static bool scratch_ready() {
    if (!g_queue) return false;
    return amdq()->scratch_backing_memory_byte_size != 0;
}

static void report_scratch(const char* when) {
    if (!g_queue) return;
    amd_queue_t* q = amdq();
    fprintf(stderr, "hsa_rt: [scratch] %s: base=%#llx size=%llu lane_bytes=%u "
                    "desc=[%#x %#x %#x %#x] priv_ap_hi=%#x\n",
            when, (unsigned long long)q->scratch_backing_memory_location,
            (unsigned long long)q->scratch_backing_memory_byte_size,
            q->scratch_wave64_lane_byte_size,
            q->scratch_resource_descriptor[0], q->scratch_resource_descriptor[1],
            q->scratch_resource_descriptor[2], q->scratch_resource_descriptor[3],
            q->private_segment_aperture_base_hi);
}

void ensure_registered(const void* p, size_t n) {
    for (auto& r : g_reg) {
        const char* a = (const char*)r.first;
        if ((const char*)p >= a && (const char*)p + n <= a + r.second) return;
    }
    if (hsa_memory_register(const_cast<void*>(p), n) == HSA_STATUS_SUCCESS) g_reg.push_back({p, n});
}

// H2D：源在主机内存（可能是 mmap 出来的文件页 / 栈上的小数组）
void copy_h2d(void* dst, const void* src, size_t n) {
    if (g_cpus.empty() || g_host_pool_st != HSA_STATUS_SUCCESS) {
        HSA_CHECK(hsa_memory_copy(dst, src, n));
        return;
    }
    std::lock_guard<std::mutex> lk(g_stage_mtx);
    for (size_t off = 0; off < n; off += STAGE_CHUNK) {
        size_t m = n - off < STAGE_CHUNK ? n - off : STAGE_CHUNK;
        void* st = stage(m);
        memcpy(st, (const char*)src + off, m);
        hsa_signal_t sig;
        HSA_CHECK(hsa_signal_create(1, 0, nullptr, &sig));
        hsa_status_t stc = hsa_amd_memory_async_copy(
            (char*)dst + off, g_gpu, st, g_cpus[0], m, 0, nullptr, sig);
        if (stc == HSA_STATUS_SUCCESS)
            hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                      HSA_WAIT_STATE_ACTIVE);
        else
            HSA_CHECK(hsa_memory_copy((char*)dst + off, (const char*)src + off, m));
        hsa_signal_destroy(sig);
    }
}

// D2H：同样过一次 pinned 暂存区
void copy_d2h(void* dst, const void* src, size_t n) {
    if (g_cpus.empty() || g_host_pool_st != HSA_STATUS_SUCCESS) {
        HSA_CHECK(hsa_memory_copy(dst, src, n));
        return;
    }
    std::lock_guard<std::mutex> lk(g_stage_mtx);
    for (size_t off = 0; off < n; off += STAGE_CHUNK) {
        size_t m = n - off < STAGE_CHUNK ? n - off : STAGE_CHUNK;
        void* st = stage(m);
        hsa_signal_t sig;
        HSA_CHECK(hsa_signal_create(1, 0, nullptr, &sig));
        hsa_status_t stc = hsa_amd_memory_async_copy(
            st, g_cpus[0], (const char*)src + off, g_gpu, m, 0, nullptr, sig);
        if (stc == HSA_STATUS_SUCCESS) {
            hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                      HSA_WAIT_STATE_ACTIVE);
            memcpy((char*)dst + off, st, m);
        } else {
            HSA_CHECK(hsa_memory_copy((char*)dst + off, (const char*)src + off, m));
        }
        hsa_signal_destroy(sig);
    }
}

}  // namespace

// ============================ 初始化 ============================
void hsart_init(const char* hsaco_path) {
    static bool done = false;
    static std::mutex init_mtx;
    std::lock_guard<std::mutex> lk(init_mtx);
    if (done) return;
    done = true;
    HSA_CHECK(hsa_init());
    HSA_CHECK(hsa_iterate_agents(cb_agent, nullptr));
    HSA_CHECK(hsa_amd_agent_iterate_memory_pools(g_gpu, cb_pool, nullptr));
    if (!g_cpus.empty()) hsa_amd_agent_iterate_memory_pools(g_cpus[0], cb_host_pool, nullptr);
    const char* env = getenv("RT_HSACO");
    load_hsaco((env && *env) ? env : (hsaco_path && *hsaco_path ? hsaco_path : RT_HSACO_DEFAULT));
    make_queue();
}

const RtKernel* hsart_lookup(const char* name) {
    static std::string norm_buf;
    norm_buf.clear();
    for (const char* p = name; *p; p++)
        if (*p != ' ' && *p != '\t' && *p != '\n') norm_buf.push_back(*p);
    auto it = g_lookup.find(norm_buf);
    if (it != g_lookup.end()) return it->second;
    // 未命中才做线性扫描（首次调用 / 名字真不存在），并把结果缓存下来
    for (int i = 0; i < k_table_n; i++)
        if (norm_buf == k_table[i].lookup) {
            g_lookup.emplace(norm_buf, &k_table[i]);
            return &k_table[i];
        }
    return nullptr;
}

// ============================ 内核投递 ============================
static uint64_t dynamic_kobject(const char* name) {
    auto it = g_dyn_kobj.find(name);
    if (it != g_dyn_kobj.end()) return it->second;
    auto sit = g_sym_kobj.find(name);
    if (sit != g_sym_kobj.end()) {
        g_dyn_kobj[name] = sit->second;
        return sit->second;
    }
    std::string kname = std::string(name) + ".k";
    auto kit = g_sym_kobj.find(kname);
    if (kit != g_sym_kobj.end()) {
        g_dyn_kobj[name] = kit->second;
        return kit->second;
    }
    hsa_executable_symbol_t sym;
    HSA_CHECK(get_symbol(name, g_gpu, &sym));
    uint64_t obj = 0;
    HSA_CHECK(hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &obj));
    g_dyn_kobj[name] = obj;
    return obj;
}

static void dispatch_packet(uint64_t kobj, const char* dbg_name, dim3 grid, dim3 block,
                            int smem, const void* kernarg, size_t kernarg_size,
                            const RtArg* args, uint32_t nargs,
                            uint32_t group_size, uint32_t private_size) {
    if (!kobj) {
        fprintf(stderr, "hsa_rt: HSACO 里没有内核 %s\n", dbg_name);
        return;
    }
    if (private_size > 0) {
        // scratch 是**按内核**由 ROCR 自己备的（见下面 report_scratch 的说明），
        // 队列里的描述符本来就一直是 0，所以这里不能拿它当「能不能跑」的依据。
        // 只在显式要求时才拦（调试/排查用）。
        if (getenv("RT_HSART_NO_SCRATCH")) {
            fprintf(stderr, "hsa_rt: %s 需要 %u 字节 scratch，"
                            "RT_HSART_NO_SCRATCH 已设置，拒绝投递。\n",
                    dbg_name, private_size);
            return;
        }
        if (getenv("RT_HSART_SCRATCH_INFO")) report_scratch("dispatch 之前");
    }
    std::lock_guard<std::mutex> lk(g_mtx);
    const int slot = g_next;
    if (g_slot_used[slot])
        hsa_signal_wait_scacquire(g_sig[slot], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                  HSA_WAIT_STATE_ACTIVE);
    char* ka = (char*)g_karg[slot];
    memcpy(ka, kernarg, kernarg_size);

    const uint64_t gx = (uint64_t)grid.x * block.x, gy = (uint64_t)grid.y * block.y,
                   gz = (uint64_t)grid.z * block.z;
    const uint32_t nbx = (uint32_t)((gx + block.x - 1) / block.x);
    const uint32_t nby = (uint32_t)((gy + block.y - 1) / block.y);
    const uint32_t nbz = (uint32_t)((gz + block.z - 1) / block.z);
    int dims = gz > 1 ? 3 : (gy > 1 ? 2 : 1);
    for (uint32_t i = 0; i < nargs; i++) {
        const RtArg& a = args[i];
        if (a.kind == 0) continue;
        uint64_t v = 0;
        switch (a.kind) {
            case 1: v = nbx; break;
            case 2: v = nby; break;
            case 3: v = nbz; break;
            case 4: v = block.x; break;
            case 5: v = block.y; break;
            case 6: v = block.z; break;
            case 7: case 8: case 9: v = 0; break;
            case 10: case 11: case 12: v = 0; break;
            case 13: v = (uint64_t)dims; break;
            default: v = 0; break;
        }
        if (a.size == 2) { uint16_t t = (uint16_t)v; memcpy(ka + a.off, &t, 2); }
        else if (a.size == 4) { uint32_t t = (uint32_t)v; memcpy(ka + a.off, &t, 4); }
        else { memcpy(ka + a.off, &v, 8); }
    }

    hsa_kernel_dispatch_packet_t* pkt = (hsa_kernel_dispatch_packet_t*)(
        (char*)g_queue->base_address +
        (hsa_queue_load_write_index_relaxed(g_queue) % g_queue->size) * sizeof(*pkt));
    memset(pkt, 0, sizeof(*pkt));
    pkt->setup = (uint16_t)dims;
    pkt->workgroup_size_x = (uint16_t)block.x;
    pkt->workgroup_size_y = (uint16_t)block.y;
    pkt->workgroup_size_z = (uint16_t)block.z;
    pkt->grid_size_x = gx;
    pkt->grid_size_y = gy;
    pkt->grid_size_z = gz;
    pkt->private_segment_size = private_size;
    pkt->group_segment_size = group_size + (uint32_t)smem;
    pkt->kernel_object = kobj;
    pkt->kernarg_address = ka;
    hsa_signal_store_screlease(g_sig[slot], 1);
    pkt->completion_signal = g_sig[slot];
    const uint16_t header = (uint16_t)(
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
    static const int dbg = getenv("RT_HSART_DEBUG") ? atoi(getenv("RT_HSART_DEBUG")) : 0;
    if (dbg)
        fprintf(stderr, "hsa_rt: %-28s grid=(%u,%u,%u) wg=(%u,%u,%u) kernarg=%p kobj=%llu "
                        "group=%u(+%d) priv=%u ksize=%zu\n",
                dbg_name, (unsigned)gx, (unsigned)gy, (unsigned)gz,
                (unsigned)block.x, (unsigned)block.y, (unsigned)block.z, (void*)ka,
                (unsigned long long)kobj, group_size, smem, private_size, kernarg_size);
    __atomic_store_n(&pkt->header, header, __ATOMIC_RELEASE);
    const uint64_t index = hsa_queue_load_write_index_relaxed(g_queue);
    hsa_queue_store_write_index_screlease(g_queue, index + 1);
    hsa_signal_store_screlease(g_queue->doorbell_signal, index);
    g_slot_used[slot] = true;
    g_next = (g_next + 1) % N_SLOT;
    if (g_pending < N_SLOT) g_pending++;
}

// ---------------- 批量投递 ----------------
// 关键点：**门铃（doorbell）只敲一次**。实测一次 doorbell 的 MMIO 写就要几微秒，
// 逐条 launch 时它是每次的大头；批量时把 N 个包填进队列、最后 store 一次 write
// index + 敲一次门铃，硬件会把索引以内的包全部取走。
//
// `packet_fill` 只填包（并推进 *widx），`packet_publish` 提交并敲门铃。
static bool packet_fill(uint64_t kobj, const char* dbg_name, uint32_t grid_wg,
                        uint32_t wg, const void* kernarg, size_t kernarg_size,
                        const RtArg* args, uint32_t nargs,
                        uint32_t group_size, uint32_t private_size,
                        uint64_t* widx) {
    if (!kobj) {
        fprintf(stderr, "hsa_rt: HSACO 里没有内核 %s\n", dbg_name);
        return false;
    }
    const int slot = g_next;
    const double t_ent = g_prof ? now_us() : 0;
    if (g_slot_used[slot])
        hsa_signal_wait_scacquire(g_sig[slot], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                  HSA_WAIT_STATE_ACTIVE);
    const double t_w = g_prof ? now_us() : 0;
    char* ka = (char*)g_karg[slot];
    memcpy(ka, kernarg, kernarg_size);
    const uint64_t gx = (uint64_t)grid_wg * wg;
    const uint32_t nbx = grid_wg;
    for (uint32_t i = 0; i < nargs; i++) {
        const RtArg& a = args[i];
        if (a.kind == 0) continue;
        uint64_t v = 0;
        switch (a.kind) {
            case 1: v = nbx; break;         // hidden_block_count_x
            case 4: v = wg; break;          // hidden_group_size_x
            case 13: v = 1; break;          // hidden_grid_dims（只用 1D）
            default: v = 0; break;
        }
        if (a.size == 2) { uint16_t x = (uint16_t)v; memcpy(ka + a.off, &x, 2); }
        else if (a.size == 4) { uint32_t x = (uint32_t)v; memcpy(ka + a.off, &x, 4); }
        else { memcpy(ka + a.off, &v, 8); }
    }
    hsa_kernel_dispatch_packet_t* pkt = (hsa_kernel_dispatch_packet_t*)(
        (char*)g_queue->base_address + (*widx % g_queue->size) * sizeof(*pkt));
    (*widx)++;
    memset(pkt, 0, sizeof(*pkt));
    pkt->setup = 1;
    pkt->workgroup_size_x = (uint16_t)wg;
    pkt->workgroup_size_y = 1;
    pkt->workgroup_size_z = 1;
    pkt->grid_size_x = gx;
    pkt->grid_size_y = 1;
    pkt->grid_size_z = 1;
    pkt->private_segment_size = private_size;
    pkt->group_segment_size = group_size;
    pkt->kernel_object = kobj;
    pkt->kernarg_address = ka;
    hsa_signal_store_screlease(g_sig[slot], 1);
    pkt->completion_signal = g_sig[slot];
    const uint16_t header = (uint16_t)(
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
    __atomic_store_n(&pkt->header, header, __ATOMIC_RELEASE);
    g_slot_used[slot] = true;
    g_next = (g_next + 1) % N_SLOT;
    if (g_pending < N_SLOT) g_pending++;
    if (g_prof) {
        g_pf_wait_us += t_w - t_ent;
        g_pf_fill_us += now_us() - t_w;
        g_pf_wait_n++;
    }
    return true;
}

static void packet_publish(uint64_t widx) {
    hsa_queue_store_write_index_screlease(g_queue, widx);
    hsa_signal_store_screlease(g_queue->doorbell_signal, widx - 1);
}

// 批量投递：`plan` 每条记录 = [kernel_id, grid, workgroup, nargs, argv...]。
// 一次调用把整段前向（十几个内核）全部入队，省掉逐条 launch 的主机侧开销。
static bool packet_fill(uint64_t kobj, const char* dbg_name, uint32_t grid_wg,
                        uint32_t wg, const void* kernarg, size_t kernarg_size,
                        const RtArg* args, uint32_t nargs,
                        uint32_t group_size, uint32_t private_size, uint64_t* widx);
static void packet_publish(uint64_t widx);

int hsart_kernel_id(const char* name) {
    const RtKernel* k = hsart_lookup(name);
    return k ? (int)(k - k_table) : -1;
}

int hsart_launch_batch(const uint64_t* plan, int nrec) {
    std::lock_guard<std::mutex> lk(g_mtx);
    const double t0p = g_prof ? now_us() : 0;
    g_pf_wait_us = g_pf_fill_us = g_pub_us = 0;
    g_pf_wait_n = 0;
    uint64_t widx = hsa_queue_load_write_index_relaxed(g_queue);
    int done = 0;
    int chunk = 0;
    const uint64_t* p = plan;
    for (int r = 0; r < nrec; r++) {
        const uint64_t id = *p++;
        const uint64_t grid_wg = *p++;
        const uint64_t wg = *p++;
        const int nargs = (int)*p++;
        if (id >= (uint64_t)k_table_n || nargs < 0) break;
        const RtKernel* k = &k_table[id];
        int n = 0;
        for (uint32_t i = 0; i < k->nargs; i++) {
            if (k->args[i].kind != 0) break;
            n++;
        }
        if (n != nargs) {
            fprintf(stderr, "hsa_rt: 批量投递 %s 参数个数不符（要 %d 给 %d）\n",
                    k->lookup, n, nargs);
            break;
        }
        char kernarg[MAX_KERNARG];
        memset(kernarg, 0, k->kernarg_size);
        for (int i = 0; i < n; i++) {
            size_t sz = k->args[i].size;
            if (sz > 8) sz = 8;
            memcpy(kernarg + k->args[i].off, p + i, sz);
        }
        p += nargs;
        if (!packet_fill(g_kobj[id], k->lookup, (uint32_t)grid_wg, (uint32_t)wg,
                         kernarg, k->kernarg_size, k->args, k->nargs,
                         k->group_size, k->private_size, &widx))
            break;
        done++;
        // kernarg 只有 N_SLOT 个槽：填满一段就得先把门铃敲了，否则下一条
        // `packet_fill` 会去等「同一个槽的上一次」——而那次还在**本批**里、
        // 门铃没敲、GPU 根本看不到它 → 死锁（v1.9.7 之前一次投 >64 条就挂住）。
        if (++chunk >= N_SLOT) {
            const double tp = g_prof ? now_us() : 0;
            packet_publish(widx);
            if (g_prof) g_pub_us += now_us() - tp;
            chunk = 0;
        }
    }
    if (chunk > 0) {                       // 不满一段的尾巴
        const double tp = g_prof ? now_us() : 0;
        packet_publish(widx);
        if (g_prof) g_pub_us += now_us() - tp;
    }
    if (g_prof && nrec > 0) {
        const double tot = now_us() - t0p;
        fprintf(stderr, "hsa_rt: batch %d 条: 总 %.1f us = 等槽 %.1f (%.2f/条, %ld 次)"
                        " + 填包 %.1f (%.2f/条) + 敲门铃 %.1f\n",
                nrec, tot, g_pf_wait_us, g_pf_wait_us / nrec, g_pf_wait_n,
                g_pf_fill_us, g_pf_fill_us / nrec, g_pub_us);
    }
    return done;
}

void hsart_dispatch(const RtKernel* k, dim3 grid, dim3 block, int smem,
                    const void* kernarg, size_t kernarg_size) {
    dispatch_packet(g_kobj[k - k_table], k->lookup, grid, block, smem, kernarg, kernarg_size,
                    k->args, k->nargs, k->group_size, k->private_size);
}

void hsart_dispatch_dyn(const char* name, dim3 grid, dim3 block, int smem,
                        const void* kernarg, size_t kernarg_size,
                        const RtArg* args, uint32_t nargs,
                        uint32_t group_size, uint32_t private_size) {
    dispatch_packet(dynamic_kobject(name), name, grid, block, smem, kernarg, kernarg_size,
                    args, nargs, group_size, private_size);
}

// ============================ HIP API ============================
hipError_t hipMalloc(void** p, size_t n) {
    hsart_init(nullptr);
    if (!n) n = 1;
    // 非致命分配：显存不足时返回错误而不是 exit —— 上层要能按可用显存
    // 决定「专家权重常驻」还是「回退逐 token 读盘」。
    void* q = nullptr;
    hsa_status_t st = hsa_amd_memory_pool_allocate(g_data_pool, n, 0, &q);
    if (st != HSA_STATUS_SUCCESS || !q) {
        *p = nullptr;
        return hipErrorOutOfMemory;
    }
    std::vector<hsa_agent_t> agents;
    agents.push_back(g_gpu);
    for (hsa_agent_t c : g_cpus) agents.push_back(c);
    st = hsa_amd_agents_allow_access((uint32_t)agents.size(), agents.data(), nullptr, q);
    if (st != HSA_STATUS_SUCCESS) {
        hsa_amd_memory_pool_free(q);
        *p = nullptr;
        return hipErrorOutOfMemory;
    }
    *p = q;
    return hipSuccess;
}

hipError_t hipHostMalloc(void** p, size_t n) {
    if (posix_memalign(p, 4096, n < 4096 ? 4096 : n) != 0) return 1;   // 4K 对齐（O_DIRECT 用）
    return hipSuccess;
}

hipError_t hipFree(void* p) {
    if (p) hsa_amd_memory_pool_free(p);
    return hipSuccess;
}

hipError_t hipDeviceSynchronize() {
    if (!g_queue) return hipSuccess;
    std::lock_guard<std::mutex> lk(g_mtx);
    // 只等自上次同步以来投递过的槽（单队列 in-order，槽按投递顺序轮转）。
    // 原来每次都把 64 个信号扫一遍，空转等待开销 ~10 us/次。
    const int start = (g_next + N_SLOT - g_pending) % N_SLOT;
    for (int k = 0; k < g_pending; k++) {
        const int i = (start + k) % N_SLOT;
        if (g_slot_used[i])
            hsa_signal_wait_scacquire(g_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                      HSA_WAIT_STATE_ACTIVE);
    }
    g_pending = 0;
    if (getenv("RT_HSART_SCRATCH_INFO")) report_scratch("sync 之后");
    return hipSuccess;
}

hipError_t hipMemcpy(void* dst, const void* src, size_t n, int kind) {
    hsart_init(nullptr);
    if (kind == hipMemcpyHostToDevice) {
        copy_h2d(dst, src, n);
    } else if (kind == hipMemcpyDeviceToHost) {
        hipDeviceSynchronize();          // 源可能刚被内核写过
        copy_d2h(dst, src, n);
    } else {
        hipDeviceSynchronize();          // 源可能刚被内核写过
        HSA_CHECK(hsa_memory_copy(dst, src, n));
    }
    return hipSuccess;
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t n, int kind, hipStream_t) {
    return hipMemcpy(dst, src, n, kind);
}

// 二维拷贝：height 行、每行 width 字节，两侧行距各自为 pitch。
// 垫片里的 hipMemcpy 本来就是同步的（hipMemcpyAsync 只是转发），所以这里只在
// 「行距 != 行宽」时按行拆开搬；整块连续时退化成一次拷贝。
// KV 槽池（Model::kv_copy_slot）走的就是这条路径：live 侧按 max_ctx 的步长、
// 槽内紧排，设备到设备。
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                            size_t width, size_t height, int kind, hipStream_t) {
    hsart_init(nullptr);
    if (width == 0 || height == 0) return hipSuccess;
    if (dpitch == width && spitch == width)          // 两侧都紧排：一次搬完
        return hipMemcpy(dst, src, width * height, kind);
    if (kind != hipMemcpyHostToDevice) hipDeviceSynchronize();   // 源可能刚被内核写过
    char* d = static_cast<char*>(dst);
    const char* s = static_cast<const char*>(src);
    for (size_t r = 0; r < height; r++) {
        void* dr = d + r * dpitch;
        const void* sr = s + r * spitch;
        if (kind == hipMemcpyHostToDevice)        copy_h2d(dr, sr, width);
        else if (kind == hipMemcpyDeviceToHost)   copy_d2h(dr, sr, width);
        else                                      HSA_CHECK(hsa_memory_copy(dr, sr, width));
    }
    return hipSuccess;
}

hipError_t hipMemset(void* p, int v, size_t n) {
    hipDeviceSynchronize();
    memset(p, v, n);
    return hipSuccess;
}

hipError_t hipStreamCreate(hipStream_t* s) { *s = new StreamImpl(); return hipSuccess; }
hipError_t hipStreamSynchronize(hipStream_t) { return hipDeviceSynchronize(); }
hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t e, unsigned) {
    return hipEventSynchronize(e);
}

hipError_t hipEventCreate(hipEvent_t* e) { *e = new EventImpl(); return hipSuccess; }
hipError_t hipEventCreateWithFlags(hipEvent_t* e, unsigned) { return hipEventCreate(e); }
hipError_t hipEventRecord(hipEvent_t e, hipStream_t) {
    if (!e) return hipSuccess;
    e->seq.fetch_add(1, std::memory_order_release);
    return hipSuccess;
}
hipError_t hipEventSynchronize(hipEvent_t e) {
    if (!e) return hipSuccess;
    const uint64_t want = e->seq.load(std::memory_order_acquire);
    if (want == 0) return hipSuccess;            // 没记录过：按 HIP 语义是空操作
    while (e->seq.load(std::memory_order_acquire) < want) std::this_thread::yield();
    return hipSuccess;
}
hipError_t hipEventElapsedTime(float* ms, hipEvent_t, hipEvent_t) { *ms = 0.f; return hipSuccess; }

hipError_t hipGetLastError() { return hipSuccess; }
const char* hipGetErrorString(hipError_t) { return "hsa_rt"; }
