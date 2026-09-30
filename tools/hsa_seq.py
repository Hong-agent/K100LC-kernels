#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在**同一个 HSA 程序**里连续跑多个内核，缓冲在 kernel 之间保留。

    from hsa_seq import run_seq
    out = run_seq(hsaco, jobs, buffers)

这是「解码 → scratch → GEMV」这类多 kernel 数据流的底座：`hsa_job` 每次
只跑一个 kernel、进程退出后显存就没了，无法把 scratch 交给下一个 kernel。

`jobs` 里每项：{"kernel": 名字, "args": [...], "grid": 总 work-item 数,
                 "workgroup": 64}
`buffers`：{名字: {"dtype": "f32", "values": [...]}} 或
          {名字: {"dtype": "f32", "n": 元素数, "readback": False}}（零初始化）
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from hsa_job import DTYPES, encode_buffer, decode_buffer, kernel_args  # noqa: E402

_BOILER = r"""
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do {{ hsa_status_t st_ = (expr); if (st_ != HSA_STATUS_SUCCESS) {{ \
    const char* s_ = NULL; hsa_status_string(st_, &s_); \
    fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr, s_ ? s_ : "?"); \
    exit(1); }} }} while (0)

static hsa_agent_t gpu;
static hsa_amd_memory_pool_t data_pool, kernarg_pool;

static hsa_status_t find_gpu(hsa_agent_t agent, void* data) {{
    hsa_device_type_t type;
    CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type));
    if (type == HSA_DEVICE_TYPE_GPU) {{ gpu = agent; return HSA_STATUS_INFO_BREAK; }}
    return HSA_STATUS_SUCCESS;
}}

static hsa_status_t find_pools(hsa_amd_memory_pool_t pool, void* data) {{
    hsa_amd_segment_t segment;
    CHECK(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment));
    if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    uint32_t flags = 0; bool alloc_ok = false;
    CHECK(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags));
    CHECK(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_ok));
    if (!alloc_ok) return HSA_STATUS_SUCCESS;
    static int have_kernarg = 0, have_data = 0;
    if (!have_kernarg && (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)) {{
        kernarg_pool = pool; have_kernarg = 1;
    }}
    if (!have_kernarg && (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {{
        kernarg_pool = pool; have_kernarg = 1;
    }}
    if (!have_data && (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)) {{
        data_pool = pool; have_data = 1;
    }}
    if (have_kernarg && have_data) return HSA_STATUS_INFO_BREAK;
    return HSA_STATUS_SUCCESS;
}}

static void* alloc_pool(hsa_amd_memory_pool_t pool, hsa_agent_t agent, size_t size) {{
    void* ptr = NULL;
    CHECK(hsa_amd_memory_pool_allocate(pool, size, 0, &ptr));
    CHECK(hsa_amd_agents_allow_access(1, &agent, NULL, ptr));
    return ptr;
}}

static hsa_status_t no_break(hsa_status_t st) {{
    return st == HSA_STATUS_INFO_BREAK ? HSA_STATUS_SUCCESS : st;
}}

static hsa_executable_t executable;

static uint64_t get_kernel(const char* base, uint32_t* ksize, uint32_t* gsize,
                           uint32_t* psize) {{
    hsa_executable_symbol_t symbol;
    char name[512];
    hsa_status_t st = hsa_executable_get_symbol_by_name(executable, base, &gpu, &symbol);
    if (st != HSA_STATUS_SUCCESS) {{
        snprintf(name, sizeof(name), "&%s", base);
        st = hsa_executable_get_symbol_by_name(executable, name, &gpu, &symbol);
    }}
    if (st != HSA_STATUS_SUCCESS) {{
        snprintf(name, sizeof(name), "%s.kd", base);
        st = hsa_executable_get_symbol_by_name(executable, name, &gpu, &symbol);
    }}
    CHECK(st);
    uint64_t obj = 0;
    CHECK(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &obj));
    CHECK(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, ksize));
    CHECK(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, gsize));
    CHECK(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, psize));
    return obj;
}}

static hsa_queue_t* queue;

static void launch(uint64_t kernel_object, uint32_t kernarg_size, uint32_t group_size,
                   uint32_t private_size, void* kernarg, uint32_t grid,
                   uint32_t workgroup, uint32_t grid_y, uint32_t extra_group) {{
    hsa_signal_t completion;
    CHECK(hsa_signal_create(1, 0, NULL, &completion));
    void* ka = alloc_pool(kernarg_pool, gpu, kernarg_size);
    CHECK(hsa_memory_copy(ka, kernarg, kernarg_size));
    uint64_t index = hsa_queue_load_write_index_relaxed(queue);
    hsa_kernel_dispatch_packet_t* packet =
        (hsa_kernel_dispatch_packet_t*)((char*)queue->base_address +
                                        (index % queue->size) * sizeof(*packet));
    memset(packet, 0, sizeof(*packet));
    packet->setup = (grid_y > 1) ? 2 : 1;
    packet->workgroup_size_x = (uint16_t)workgroup;
    packet->workgroup_size_y = 1;
    packet->workgroup_size_z = 1;
    packet->grid_size_x = grid;
    packet->grid_size_y = grid_y;
    packet->grid_size_z = 1;
    packet->private_segment_size = private_size;
    packet->group_segment_size = group_size + extra_group;
    packet->kernel_object = kernel_object;
    packet->kernarg_address = ka;
    packet->completion_signal = completion;
    uint16_t header = (uint16_t)(
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
    __atomic_store_n(&packet->header, header, __ATOMIC_RELEASE);
    hsa_queue_store_write_index_screlease(queue, index + 1);
    hsa_signal_store_screlease(queue->doorbell_signal, index);
    hsa_signal_wait_scacquire(completion, HSA_SIGNAL_CONDITION_LT, 1,
                              10ull * 1000 * 1000 * 1000, HSA_WAIT_STATE_ACTIVE);
    CHECK(hsa_signal_destroy(completion));
}}
"""


def _arg_fill(spec: dict, meta: dict, buffers: dict, decls: list[str]) -> str:
    off = int(meta[".offset"])
    size = int(meta[".size"])
    kind = meta.get(".value_kind", "by_value")
    if "buffer" in spec:
        name = spec["buffer"]
        return f"  {{ void* p = {name}; memcpy((char*)kernarg + {off}, &p, 8); }}"
    if spec.get("null"):
        return f"  {{ void* p = NULL; memcpy((char*)kernarg + {off}, &p, 8); }}"
    if "scalar" in spec:
        dtype = spec["scalar"]["dtype"]
        ctype, _ = DTYPES[dtype]
        value = spec["scalar"]["value"]
        if dtype == "f32":
            return f"  {{ {ctype} v = {value!r}f; memcpy((char*)kernarg + {off}, &v, {size}); }}"
        return f"  {{ {ctype} v = {int(value)}; memcpy((char*)kernarg + {off}, &v, {size}); }}"
    raise SystemExit(f"unsupported arg {meta}")


def run_seq(hsaco: pathlib.Path, jobs: list[dict], buffers: dict) -> dict:
    os.environ["RT_HSACO"] = str(hsaco)
    source = build_c_seq_full(hsaco, jobs, buffers)
    td = pathlib.Path(tempfile.mkdtemp(prefix="hsa_seq_"))
    cpath = td / "seq.c"
    cpath.write_text(source, encoding="utf-8")
    exe = td / "seq"
    subprocess.run(["gcc", "-O2", "-I/opt/hyhal/include", str(cpath),
                    "-L/opt/hyhal/lib", "-Wl,-rpath,/opt/hyhal/lib",
                    "-lhsa-runtime64", "-o", str(exe)], check=True)
    proc = subprocess.run([str(exe)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"seq job failed:\n{proc.stdout}\n{proc.stderr}")
    out: dict = {}
    for line in proc.stdout.splitlines():
        if line.startswith("BUF "):
            _, name, hexdata = line.split()
            dtype = buffers[name]["dtype"]
            out[name] = decode_buffer(dtype, bytes.fromhex(hexdata))
    return out


def build_c_seq_full(hsaco: pathlib.Path, jobs: list[dict], buffers: dict) -> str:
    """把 {boiler + 缓冲分配 + 各 job 的 kernarg/launch + 回读} 拼成完整 C 源。"""
    decls: list[str] = []
    allocs: list[str] = []
    for name, buf in buffers.items():
        dtype, esize = DTYPES[buf["dtype"]]
        if "values" in buf:
            blob = encode_buffer(buf["dtype"], buf["values"])
            hexes = ",".join(f"0x{b:02x}" for b in blob)
            decls.append(f"static const unsigned char init_{name}[{len(blob)}] = {{{hexes}}};")
            allocs.append(
                f"  void* {name} = alloc_pool(data_pool, gpu, {len(blob)});\n"
                f"  CHECK(hsa_memory_copy({name}, init_{name}, {len(blob)}));"
            )
        elif "file" in buf:
            # 大权重块从文件读，避免把几 MB 塞进 C 字面量
            fpath = str(buf["file"])
            blob_len = os.path.getsize(fpath)
            allocs.append(
                f"  void* {name} = alloc_pool(data_pool, gpu, {blob_len});\n"
                f"  {{ FILE* fp = fopen(\"{fpath}\", \"rb\"); if (!fp) {{ perror(\"{fpath}\"); exit(1); }}\n"
                f"    void* hb = malloc({blob_len}); if (fread(hb, 1, {blob_len}, fp) != {blob_len}) {{ fprintf(stderr, \"short read\\n\"); exit(1); }}\n"
                f"    fclose(fp); CHECK(hsa_memory_copy({name}, hb, {blob_len})); free(hb); }}"
            )
        else:
            blob_len = int(buf["n"]) * esize
            allocs.append(
                f"  void* {name} = alloc_pool(data_pool, gpu, {blob_len});\n"
                f"  {{ void* z = calloc(1, {blob_len}); CHECK(hsa_memory_copy({name}, z, {blob_len})); free(z); }}"
            )

    launches: list[str] = []
    for job in jobs:
        kernel = job["kernel"]
        meta_args = kernel_args(kernel)
        explicit = [a for a in meta_args
                    if not str(a.get("value_kind", "by_value")).startswith("hidden_")]
        if len(explicit) != len(job["args"]):
            raise SystemExit(f"{kernel}: metadata has {len(explicit)} explicit args, job has {len(job['args'])}")
        ksize = max(int(a[".offset"]) + int(a[".size"]) for a in meta_args)
        grid = job.get("grid", 64)
        wg = job.get("workgroup", 64)
        grid_y = job.get("grid_y", 1)
        nblocks = max(1, (grid + wg - 1) // wg)
        fills = [f"  {{ uint64_t obj; uint32_t ksz, gssz, pssz; obj = get_kernel(\"{kernel}\", &ksz, &gssz, &pssz);\n"
                 f"    static unsigned char kernarg[4096]; memset(kernarg, 0, sizeof(kernarg));"]
        for spec, meta in zip(job["args"], explicit):
            fills.append(_arg_fill(spec, meta, buffers, decls))
        hidden_values = {
            "hidden_block_count_x": ("uint32_t", nblocks),
            "hidden_block_count_y": ("uint32_t", max(1, grid_y)),
            "hidden_block_count_z": ("uint32_t", 1),
            "hidden_group_size_x": ("uint16_t", wg),
            "hidden_group_size_y": ("uint16_t", 1),
            "hidden_group_size_z": ("uint16_t", 1),
            "hidden_remainder_x": ("uint16_t", 0),
            "hidden_remainder_y": ("uint16_t", 0),
            "hidden_remainder_z": ("uint16_t", 0),
            "hidden_global_offset_x": ("uint64_t", 0),
            "hidden_global_offset_y": ("uint64_t", 0),
            "hidden_global_offset_z": ("uint64_t", 0),
            "hidden_grid_dims": ("uint16_t", 2 if grid_y > 1 else 1),
        }
        for meta in meta_args:
            kind = str(meta.get(".value_kind", ""))
            if not kind.startswith("hidden_") or kind not in hidden_values:
                continue
            ctype, value = hidden_values[kind]
            off, size = int(meta[".offset"]), int(meta[".size"])
            fills.append(f"    {{ {ctype} v = {value}; memcpy((char*)kernarg + {off}, &v, {size}); }}")
        fills.append(f"    launch(obj, ksz, gssz, pssz, kernarg, {grid}, {wg}, {grid_y}, {job.get('extra_group', 0)}); }}")
        launches.append("\n".join(fills))

    outputs = []
    for name, buf in buffers.items():
        if buf.get("readback", True) is False:
            continue
        dtype = buf["dtype"]
        if "values" in buf:
            blob_len = len(encode_buffer(dtype, buf["values"]))
        elif "file" in buf:
            blob_len = os.path.getsize(buf["file"])
        else:
            blob_len = int(buf["n"]) * DTYPES[dtype][1]
        outputs.append(
            f'  {{ unsigned char* h = (unsigned char*)malloc({blob_len});\n'
            f'    CHECK(hsa_memory_copy(h, {name}, {blob_len}));\n'
            f'    printf("BUF {name} ");\n'
            f'    for (size_t i = 0; i < {blob_len}; i++) printf("%02x", h[i]);\n'
            f'    printf("\\n"); free(h); }}'
        )

    main_body = "\n".join(allocs) + "\n\n" + "\n".join(launches) + "\n\n" + "\n".join(outputs)
    return (_BOILER
            + "\n"
            + "\n".join(decls)
            + "\n\nint main(void) {\n"
            + "  CHECK(hsa_init());\n"
            + "  CHECK(no_break(hsa_iterate_agents(find_gpu, NULL)));\n"
            + "  CHECK(no_break(hsa_amd_agent_iterate_memory_pools(gpu, find_pools, NULL)));\n"
            + "  {\n"
            + f"    FILE* f = fopen(\"{hsaco}\", \"rb\");\n"
            + "    if (!f) { perror(\"hsaco\"); return 1; }\n"
            + "    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);\n"
            + "    void* image = malloc(sz); if (fread(image, 1, sz, f) != (size_t)sz) return 1; fclose(f);\n"
            + "    hsa_code_object_reader_t reader;\n"
            + "    CHECK(hsa_code_object_reader_create_from_memory(image, sz, &reader));\n"
            + "    CHECK(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL, &executable));\n"
            + "    CHECK(hsa_executable_load_agent_code_object(executable, gpu, reader, NULL, NULL));\n"
            + "    CHECK(hsa_executable_freeze(executable, NULL));\n"
            + "    uint32_t qsize = 0; CHECK(hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &qsize));\n"
            + "    if (qsize > 1024) qsize = 1024;\n"
            + "    CHECK(hsa_queue_create(gpu, qsize, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue));\n"
            + "  }\n"
            + main_body
            + "\n  CHECK(hsa_queue_destroy(queue));\n  CHECK(hsa_shut_down());\n  return 0;\n}\n")
