// 自动生成：tools/gen_nodtk.py（数据来自自研 HSACO 的 metadata）
#pragma once
#include <cstdint>

#define MAX_KERNARG 1024
struct RtArg { uint32_t off; uint32_t size; uint8_t kind; };
struct RtKernel {
    const char* name;      // mangled 名（HSACO 里的符号）
    const char* lookup;    // 源码里的写法（已去掉空白）
    uint32_t nargs;        // 参数个数（含隐藏）
    const RtArg* args;
    uint32_t kernarg_size, group_size, private_size;
};

static const RtArg k_args_0[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_1[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_2[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_3[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_4[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_5[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_6[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_7[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_8[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_9[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_10[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_11[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_12[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_13[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_14[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_15[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_16[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
};
static const RtArg k_args_17[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 1},   // hidden_block_count_x
    {36u, 4u, 2},   // hidden_block_count_y
    {40u, 4u, 3},   // hidden_block_count_z
    {44u, 2u, 4},   // hidden_group_size_x
    {46u, 2u, 5},   // hidden_group_size_y
    {48u, 2u, 6},   // hidden_group_size_z
    {50u, 2u, 7},   // hidden_remainder_x
    {52u, 2u, 8},   // hidden_remainder_y
    {54u, 2u, 9},   // hidden_remainder_z
    {72u, 8u, 10},   // hidden_global_offset_x
    {80u, 8u, 11},   // hidden_global_offset_y
    {88u, 8u, 12},   // hidden_global_offset_z
    {96u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_18[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 8u, 0},   // by_value/global
    {56u, 8u, 0},   // by_value/global
    {64u, 8u, 0},   // by_value/global
    {72u, 4u, 0},   // by_value/global
    {76u, 4u, 0},   // by_value/global
    {80u, 4u, 0},   // by_value/global
    {84u, 4u, 0},   // by_value/global
    {88u, 4u, 0},   // by_value/global
    {92u, 4u, 0},   // by_value/global
    {96u, 4u, 0},   // by_value/global
    {100u, 4u, 0},   // by_value/global
};
static const RtArg k_args_19[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 4u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 1},   // hidden_block_count_x
    {28u, 4u, 2},   // hidden_block_count_y
    {32u, 4u, 3},   // hidden_block_count_z
    {36u, 2u, 4},   // hidden_group_size_x
    {38u, 2u, 5},   // hidden_group_size_y
    {40u, 2u, 6},   // hidden_group_size_z
    {42u, 2u, 7},   // hidden_remainder_x
    {44u, 2u, 8},   // hidden_remainder_y
    {46u, 2u, 9},   // hidden_remainder_z
    {64u, 8u, 10},   // hidden_global_offset_x
    {72u, 8u, 11},   // hidden_global_offset_y
    {80u, 8u, 12},   // hidden_global_offset_z
    {88u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_20[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {44u, 4u, 0},   // by_value/global
    {48u, 4u, 1},   // hidden_block_count_x
    {52u, 4u, 2},   // hidden_block_count_y
    {56u, 4u, 3},   // hidden_block_count_z
    {60u, 2u, 4},   // hidden_group_size_x
    {62u, 2u, 5},   // hidden_group_size_y
    {64u, 2u, 6},   // hidden_group_size_z
    {66u, 2u, 7},   // hidden_remainder_x
    {68u, 2u, 8},   // hidden_remainder_y
    {70u, 2u, 9},   // hidden_remainder_z
    {88u, 8u, 10},   // hidden_global_offset_x
    {96u, 8u, 11},   // hidden_global_offset_y
    {104u, 8u, 12},   // hidden_global_offset_z
    {112u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_21[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
};
static const RtArg k_args_22[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 1},   // hidden_block_count_x
    {28u, 4u, 2},   // hidden_block_count_y
    {32u, 4u, 3},   // hidden_block_count_z
    {36u, 2u, 4},   // hidden_group_size_x
    {38u, 2u, 5},   // hidden_group_size_y
    {40u, 2u, 6},   // hidden_group_size_z
    {42u, 2u, 7},   // hidden_remainder_x
    {44u, 2u, 8},   // hidden_remainder_y
    {46u, 2u, 9},   // hidden_remainder_z
    {64u, 8u, 10},   // hidden_global_offset_x
    {72u, 8u, 11},   // hidden_global_offset_y
    {80u, 8u, 12},   // hidden_global_offset_z
    {88u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_23[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 4u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 1},   // hidden_block_count_x
    {36u, 4u, 2},   // hidden_block_count_y
    {40u, 4u, 3},   // hidden_block_count_z
    {44u, 2u, 4},   // hidden_group_size_x
    {46u, 2u, 5},   // hidden_group_size_y
    {48u, 2u, 6},   // hidden_group_size_z
    {50u, 2u, 7},   // hidden_remainder_x
    {52u, 2u, 8},   // hidden_remainder_y
    {54u, 2u, 9},   // hidden_remainder_z
    {72u, 8u, 10},   // hidden_global_offset_x
    {80u, 8u, 11},   // hidden_global_offset_y
    {88u, 8u, 12},   // hidden_global_offset_z
    {96u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_24[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 4u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 4u, 1},   // hidden_block_count_x
    {44u, 4u, 2},   // hidden_block_count_y
    {48u, 4u, 3},   // hidden_block_count_z
    {52u, 2u, 4},   // hidden_group_size_x
    {54u, 2u, 5},   // hidden_group_size_y
    {56u, 2u, 6},   // hidden_group_size_z
    {58u, 2u, 7},   // hidden_remainder_x
    {60u, 2u, 8},   // hidden_remainder_y
    {62u, 2u, 9},   // hidden_remainder_z
    {80u, 8u, 10},   // hidden_global_offset_x
    {88u, 8u, 11},   // hidden_global_offset_y
    {96u, 8u, 12},   // hidden_global_offset_z
    {104u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_25[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {48u, 4u, 1},   // hidden_block_count_x
    {52u, 4u, 2},   // hidden_block_count_y
    {56u, 4u, 3},   // hidden_block_count_z
    {60u, 2u, 4},   // hidden_group_size_x
    {62u, 2u, 5},   // hidden_group_size_y
    {64u, 2u, 6},   // hidden_group_size_z
    {66u, 2u, 7},   // hidden_remainder_x
    {68u, 2u, 8},   // hidden_remainder_y
    {70u, 2u, 9},   // hidden_remainder_z
    {88u, 8u, 10},   // hidden_global_offset_x
    {96u, 8u, 11},   // hidden_global_offset_y
    {104u, 8u, 12},   // hidden_global_offset_z
    {112u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_26[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {44u, 4u, 0},   // by_value/global
};
static const RtArg k_args_27[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {44u, 4u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
};
static const RtArg k_args_28[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 1},   // hidden_block_count_x
    {36u, 4u, 2},   // hidden_block_count_y
    {40u, 4u, 3},   // hidden_block_count_z
    {44u, 2u, 4},   // hidden_group_size_x
    {46u, 2u, 5},   // hidden_group_size_y
    {48u, 2u, 6},   // hidden_group_size_z
    {50u, 2u, 7},   // hidden_remainder_x
    {52u, 2u, 8},   // hidden_remainder_y
    {54u, 2u, 9},   // hidden_remainder_z
    {72u, 8u, 10},   // hidden_global_offset_x
    {80u, 8u, 11},   // hidden_global_offset_y
    {88u, 8u, 12},   // hidden_global_offset_z
    {96u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_29[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 8u, 0},   // by_value/global
    {56u, 8u, 0},   // by_value/global
    {64u, 4u, 0},   // by_value/global
    {68u, 4u, 0},   // by_value/global
    {72u, 4u, 1},   // hidden_block_count_x
    {76u, 4u, 2},   // hidden_block_count_y
    {80u, 4u, 3},   // hidden_block_count_z
    {84u, 2u, 4},   // hidden_group_size_x
    {86u, 2u, 5},   // hidden_group_size_y
    {88u, 2u, 6},   // hidden_group_size_z
    {90u, 2u, 7},   // hidden_remainder_x
    {92u, 2u, 8},   // hidden_remainder_y
    {94u, 2u, 9},   // hidden_remainder_z
    {112u, 8u, 10},   // hidden_global_offset_x
    {120u, 8u, 11},   // hidden_global_offset_y
    {128u, 8u, 12},   // hidden_global_offset_z
    {136u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_30[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {48u, 4u, 1},   // hidden_block_count_x
    {52u, 4u, 2},   // hidden_block_count_y
    {56u, 4u, 3},   // hidden_block_count_z
    {60u, 2u, 4},   // hidden_group_size_x
    {62u, 2u, 5},   // hidden_group_size_y
    {64u, 2u, 6},   // hidden_group_size_z
    {66u, 2u, 7},   // hidden_remainder_x
    {68u, 2u, 8},   // hidden_remainder_y
    {70u, 2u, 9},   // hidden_remainder_z
    {88u, 8u, 10},   // hidden_global_offset_x
    {96u, 8u, 11},   // hidden_global_offset_y
    {104u, 8u, 12},   // hidden_global_offset_z
    {112u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_31[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {48u, 4u, 1},   // hidden_block_count_x
    {52u, 4u, 2},   // hidden_block_count_y
    {56u, 4u, 3},   // hidden_block_count_z
    {60u, 2u, 4},   // hidden_group_size_x
    {62u, 2u, 5},   // hidden_group_size_y
    {64u, 2u, 6},   // hidden_group_size_z
    {66u, 2u, 7},   // hidden_remainder_x
    {68u, 2u, 8},   // hidden_remainder_y
    {70u, 2u, 9},   // hidden_remainder_z
    {88u, 8u, 10},   // hidden_global_offset_x
    {96u, 8u, 11},   // hidden_global_offset_y
    {104u, 8u, 12},   // hidden_global_offset_z
    {112u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_32[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 4u, 0},   // by_value/global
    {20u, 4u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 1},   // hidden_block_count_x
    {52u, 4u, 2},   // hidden_block_count_y
    {56u, 4u, 3},   // hidden_block_count_z
    {60u, 2u, 4},   // hidden_group_size_x
    {62u, 2u, 5},   // hidden_group_size_y
    {64u, 2u, 6},   // hidden_group_size_z
    {66u, 2u, 7},   // hidden_remainder_x
    {68u, 2u, 8},   // hidden_remainder_y
    {70u, 2u, 9},   // hidden_remainder_z
    {88u, 8u, 10},   // hidden_global_offset_x
    {96u, 8u, 11},   // hidden_global_offset_y
    {104u, 8u, 12},   // hidden_global_offset_z
    {112u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_33[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_34[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_35[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_36[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_37[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_38[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_39[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_40[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_41[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 1},   // hidden_block_count_x
    {44u, 4u, 2},   // hidden_block_count_y
    {48u, 4u, 3},   // hidden_block_count_z
    {52u, 2u, 4},   // hidden_group_size_x
    {54u, 2u, 5},   // hidden_group_size_y
    {56u, 2u, 6},   // hidden_group_size_z
    {58u, 2u, 7},   // hidden_remainder_x
    {60u, 2u, 8},   // hidden_remainder_y
    {62u, 2u, 9},   // hidden_remainder_z
    {80u, 8u, 10},   // hidden_global_offset_x
    {88u, 8u, 11},   // hidden_global_offset_y
    {96u, 8u, 12},   // hidden_global_offset_z
    {104u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_42[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {44u, 4u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
};
static const RtArg k_args_43[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 1},   // hidden_block_count_x
    {44u, 4u, 2},   // hidden_block_count_y
    {48u, 4u, 3},   // hidden_block_count_z
    {52u, 2u, 4},   // hidden_group_size_x
    {54u, 2u, 5},   // hidden_group_size_y
    {56u, 2u, 6},   // hidden_group_size_z
    {58u, 2u, 7},   // hidden_remainder_x
    {60u, 2u, 8},   // hidden_remainder_y
    {62u, 2u, 9},   // hidden_remainder_z
    {80u, 8u, 10},   // hidden_global_offset_x
    {88u, 8u, 11},   // hidden_global_offset_y
    {96u, 8u, 12},   // hidden_global_offset_z
    {104u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_44[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 4u, 0},   // by_value/global
    {20u, 4u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 1},   // hidden_block_count_x
    {52u, 4u, 2},   // hidden_block_count_y
    {56u, 4u, 3},   // hidden_block_count_z
    {60u, 2u, 4},   // hidden_group_size_x
    {62u, 2u, 5},   // hidden_group_size_y
    {64u, 2u, 6},   // hidden_group_size_z
    {66u, 2u, 7},   // hidden_remainder_x
    {68u, 2u, 8},   // hidden_remainder_y
    {70u, 2u, 9},   // hidden_remainder_z
    {88u, 8u, 10},   // hidden_global_offset_x
    {96u, 8u, 11},   // hidden_global_offset_y
    {104u, 8u, 12},   // hidden_global_offset_z
    {112u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_45[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 4u, 0},   // by_value/global
    {20u, 4u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {40u, 4u, 1},   // hidden_block_count_x
    {44u, 4u, 2},   // hidden_block_count_y
    {48u, 4u, 3},   // hidden_block_count_z
    {52u, 2u, 4},   // hidden_group_size_x
    {54u, 2u, 5},   // hidden_group_size_y
    {56u, 2u, 6},   // hidden_group_size_z
    {58u, 2u, 7},   // hidden_remainder_x
    {60u, 2u, 8},   // hidden_remainder_y
    {62u, 2u, 9},   // hidden_remainder_z
    {80u, 8u, 10},   // hidden_global_offset_x
    {88u, 8u, 11},   // hidden_global_offset_y
    {96u, 8u, 12},   // hidden_global_offset_z
    {104u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_46[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 4u, 0},   // by_value/global
    {20u, 4u, 0},   // by_value/global
    {24u, 4u, 1},   // hidden_block_count_x
    {28u, 4u, 2},   // hidden_block_count_y
    {32u, 4u, 3},   // hidden_block_count_z
    {36u, 2u, 4},   // hidden_group_size_x
    {38u, 2u, 5},   // hidden_group_size_y
    {40u, 2u, 6},   // hidden_group_size_z
    {42u, 2u, 7},   // hidden_remainder_x
    {44u, 2u, 8},   // hidden_remainder_y
    {46u, 2u, 9},   // hidden_remainder_z
    {64u, 8u, 10},   // hidden_global_offset_x
    {72u, 8u, 11},   // hidden_global_offset_y
    {80u, 8u, 12},   // hidden_global_offset_z
    {88u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_47[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {32u, 4u, 1},   // hidden_block_count_x
    {36u, 4u, 2},   // hidden_block_count_y
    {40u, 4u, 3},   // hidden_block_count_z
    {44u, 2u, 4},   // hidden_group_size_x
    {46u, 2u, 5},   // hidden_group_size_y
    {48u, 2u, 6},   // hidden_group_size_z
    {50u, 2u, 7},   // hidden_remainder_x
    {52u, 2u, 8},   // hidden_remainder_y
    {54u, 2u, 9},   // hidden_remainder_z
    {72u, 8u, 10},   // hidden_global_offset_x
    {80u, 8u, 11},   // hidden_global_offset_y
    {88u, 8u, 12},   // hidden_global_offset_z
    {96u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_48[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 1},   // hidden_block_count_x
    {44u, 4u, 2},   // hidden_block_count_y
    {48u, 4u, 3},   // hidden_block_count_z
    {52u, 2u, 4},   // hidden_group_size_x
    {54u, 2u, 5},   // hidden_group_size_y
    {56u, 2u, 6},   // hidden_group_size_z
    {58u, 2u, 7},   // hidden_remainder_x
    {60u, 2u, 8},   // hidden_remainder_y
    {62u, 2u, 9},   // hidden_remainder_z
    {80u, 8u, 10},   // hidden_global_offset_x
    {88u, 8u, 11},   // hidden_global_offset_y
    {96u, 8u, 12},   // hidden_global_offset_z
    {104u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_49[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
};
static const RtArg k_args_50[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 8u, 0},   // by_value/global
    {56u, 8u, 0},   // by_value/global
    {64u, 8u, 0},   // by_value/global
    {72u, 8u, 0},   // by_value/global
    {80u, 8u, 0},   // by_value/global
    {88u, 4u, 0},   // by_value/global
    {92u, 4u, 0},   // by_value/global
    {96u, 4u, 0},   // by_value/global
    {100u, 4u, 0},   // by_value/global
    {104u, 4u, 0},   // by_value/global
    {108u, 4u, 0},   // by_value/global
    {112u, 4u, 0},   // by_value/global
    {116u, 4u, 0},   // by_value/global
    {120u, 4u, 0},   // by_value/global
    {124u, 4u, 0},   // by_value/global
    {128u, 8u, 0},   // by_value/global
    {136u, 8u, 0},   // by_value/global
    {144u, 4u, 0},   // by_value/global
};
static const RtArg k_args_51[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {60u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_52[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 4u, 0},   // by_value/global
    {20u, 4u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {32u, 4u, 1},   // hidden_block_count_x
    {36u, 4u, 2},   // hidden_block_count_y
    {40u, 4u, 3},   // hidden_block_count_z
    {44u, 2u, 4},   // hidden_group_size_x
    {46u, 2u, 5},   // hidden_group_size_y
    {48u, 2u, 6},   // hidden_group_size_z
    {50u, 2u, 7},   // hidden_remainder_x
    {52u, 2u, 8},   // hidden_remainder_y
    {54u, 2u, 9},   // hidden_remainder_z
    {72u, 8u, 10},   // hidden_global_offset_x
    {80u, 8u, 11},   // hidden_global_offset_y
    {88u, 8u, 12},   // hidden_global_offset_z
    {96u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_53[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {48u, 4u, 1},   // hidden_block_count_x
    {52u, 4u, 2},   // hidden_block_count_y
    {56u, 4u, 3},   // hidden_block_count_z
    {60u, 2u, 4},   // hidden_group_size_x
    {62u, 2u, 5},   // hidden_group_size_y
    {64u, 2u, 6},   // hidden_group_size_z
    {66u, 2u, 7},   // hidden_remainder_x
    {68u, 2u, 8},   // hidden_remainder_y
    {70u, 2u, 9},   // hidden_remainder_z
    {88u, 8u, 10},   // hidden_global_offset_x
    {96u, 8u, 11},   // hidden_global_offset_y
    {104u, 8u, 12},   // hidden_global_offset_z
    {112u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_54[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {60u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_55[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {44u, 4u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 8u, 0},   // by_value/global
};
static const RtArg k_args_56[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {48u, 4u, 1},   // hidden_block_count_x
    {52u, 4u, 2},   // hidden_block_count_y
    {56u, 4u, 3},   // hidden_block_count_z
    {60u, 2u, 4},   // hidden_group_size_x
    {62u, 2u, 5},   // hidden_group_size_y
    {64u, 2u, 6},   // hidden_group_size_z
    {66u, 2u, 7},   // hidden_remainder_x
    {68u, 2u, 8},   // hidden_remainder_y
    {70u, 2u, 9},   // hidden_remainder_z
    {88u, 8u, 10},   // hidden_global_offset_x
    {96u, 8u, 11},   // hidden_global_offset_y
    {104u, 8u, 12},   // hidden_global_offset_z
    {112u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_57[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 8u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {60u, 4u, 0},   // by_value/global
    {64u, 4u, 0},   // by_value/global
    {68u, 4u, 0},   // by_value/global
    {72u, 4u, 0},   // by_value/global
    {76u, 4u, 0},   // by_value/global
    {80u, 4u, 1},   // hidden_block_count_x
    {84u, 4u, 2},   // hidden_block_count_y
    {88u, 4u, 3},   // hidden_block_count_z
    {92u, 2u, 4},   // hidden_group_size_x
    {94u, 2u, 5},   // hidden_group_size_y
    {96u, 2u, 6},   // hidden_group_size_z
    {98u, 2u, 7},   // hidden_remainder_x
    {100u, 2u, 8},   // hidden_remainder_y
    {102u, 2u, 9},   // hidden_remainder_z
    {120u, 8u, 10},   // hidden_global_offset_x
    {128u, 8u, 11},   // hidden_global_offset_y
    {136u, 8u, 12},   // hidden_global_offset_z
    {144u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_58[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 4u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 1},   // hidden_block_count_x
    {28u, 4u, 2},   // hidden_block_count_y
    {32u, 4u, 3},   // hidden_block_count_z
    {36u, 2u, 4},   // hidden_group_size_x
    {38u, 2u, 5},   // hidden_group_size_y
    {40u, 2u, 6},   // hidden_group_size_z
    {42u, 2u, 7},   // hidden_remainder_x
    {44u, 2u, 8},   // hidden_remainder_y
    {46u, 2u, 9},   // hidden_remainder_z
    {64u, 8u, 10},   // hidden_global_offset_x
    {72u, 8u, 11},   // hidden_global_offset_y
    {80u, 8u, 12},   // hidden_global_offset_z
    {88u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_59[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 8u, 0},   // by_value/global
    {56u, 8u, 0},   // by_value/global
    {64u, 4u, 0},   // by_value/global
    {68u, 4u, 0},   // by_value/global
    {72u, 4u, 0},   // by_value/global
    {76u, 4u, 0},   // by_value/global
    {80u, 4u, 0},   // by_value/global
};
static const RtArg k_args_60[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {44u, 4u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
    {52u, 4u, 0},   // by_value/global
    {56u, 4u, 1},   // hidden_block_count_x
    {60u, 4u, 2},   // hidden_block_count_y
    {64u, 4u, 3},   // hidden_block_count_z
    {68u, 2u, 4},   // hidden_group_size_x
    {70u, 2u, 5},   // hidden_group_size_y
    {72u, 2u, 6},   // hidden_group_size_z
    {74u, 2u, 7},   // hidden_remainder_x
    {76u, 2u, 8},   // hidden_remainder_y
    {78u, 2u, 9},   // hidden_remainder_z
    {96u, 8u, 10},   // hidden_global_offset_x
    {104u, 8u, 11},   // hidden_global_offset_y
    {112u, 8u, 12},   // hidden_global_offset_z
    {120u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_61[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 1},   // hidden_block_count_x
    {44u, 4u, 2},   // hidden_block_count_y
    {48u, 4u, 3},   // hidden_block_count_z
    {52u, 2u, 4},   // hidden_group_size_x
    {54u, 2u, 5},   // hidden_group_size_y
    {56u, 2u, 6},   // hidden_group_size_z
    {58u, 2u, 7},   // hidden_remainder_x
    {60u, 2u, 8},   // hidden_remainder_y
    {62u, 2u, 9},   // hidden_remainder_z
    {80u, 8u, 10},   // hidden_global_offset_x
    {88u, 8u, 11},   // hidden_global_offset_y
    {96u, 8u, 12},   // hidden_global_offset_z
    {104u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_62[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 8u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {60u, 4u, 0},   // by_value/global
    {64u, 4u, 0},   // by_value/global
    {68u, 4u, 0},   // by_value/global
    {72u, 4u, 0},   // by_value/global
    {76u, 4u, 0},   // by_value/global
};
static const RtArg k_args_63[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 1},   // hidden_block_count_x
    {28u, 4u, 2},   // hidden_block_count_y
    {32u, 4u, 3},   // hidden_block_count_z
    {36u, 2u, 4},   // hidden_group_size_x
    {38u, 2u, 5},   // hidden_group_size_y
    {40u, 2u, 6},   // hidden_group_size_z
    {42u, 2u, 7},   // hidden_remainder_x
    {44u, 2u, 8},   // hidden_remainder_y
    {46u, 2u, 9},   // hidden_remainder_z
    {64u, 8u, 10},   // hidden_global_offset_x
    {72u, 8u, 11},   // hidden_global_offset_y
    {80u, 8u, 12},   // hidden_global_offset_z
    {88u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_64[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 4u, 0},   // by_value/global
    {12u, 4u, 0},   // by_value/global
    {16u, 4u, 1},   // hidden_block_count_x
    {20u, 4u, 2},   // hidden_block_count_y
    {24u, 4u, 3},   // hidden_block_count_z
    {28u, 2u, 4},   // hidden_group_size_x
    {30u, 2u, 5},   // hidden_group_size_y
    {32u, 2u, 6},   // hidden_group_size_z
    {34u, 2u, 7},   // hidden_remainder_x
    {36u, 2u, 8},   // hidden_remainder_y
    {38u, 2u, 9},   // hidden_remainder_z
    {56u, 8u, 10},   // hidden_global_offset_x
    {64u, 8u, 11},   // hidden_global_offset_y
    {72u, 8u, 12},   // hidden_global_offset_z
    {80u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_65[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 4u, 1},   // hidden_block_count_x
    {44u, 4u, 2},   // hidden_block_count_y
    {48u, 4u, 3},   // hidden_block_count_z
    {52u, 2u, 4},   // hidden_group_size_x
    {54u, 2u, 5},   // hidden_group_size_y
    {56u, 2u, 6},   // hidden_group_size_z
    {58u, 2u, 7},   // hidden_remainder_x
    {60u, 2u, 8},   // hidden_remainder_y
    {62u, 2u, 9},   // hidden_remainder_z
    {80u, 8u, 10},   // hidden_global_offset_x
    {88u, 8u, 11},   // hidden_global_offset_y
    {96u, 8u, 12},   // hidden_global_offset_z
    {104u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_66[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {44u, 4u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
};
static const RtArg k_args_67[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {44u, 4u, 0},   // by_value/global
};
static const RtArg k_args_68[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {44u, 4u, 0},   // by_value/global
};
static const RtArg k_args_69[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {44u, 4u, 0},   // by_value/global
};
static const RtArg k_args_70[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {44u, 4u, 0},   // by_value/global
};
static const RtArg k_args_71[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 8u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {60u, 4u, 0},   // by_value/global
};
static const RtArg k_args_72[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 8u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {60u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_73[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 8u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {60u, 4u, 0},   // by_value/global
};
static const RtArg k_args_74[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 8u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {60u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_75[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 8u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {60u, 4u, 0},   // by_value/global
};
static const RtArg k_args_76[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 8u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {60u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_77[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 8u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {60u, 4u, 0},   // by_value/global
};
static const RtArg k_args_78[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 8u, 0},   // by_value/global
    {48u, 8u, 0},   // by_value/global
    {56u, 4u, 0},   // by_value/global
    {60u, 4u, 0},   // by_value/global
    {64u, 4u, 1},   // hidden_block_count_x
    {68u, 4u, 2},   // hidden_block_count_y
    {72u, 4u, 3},   // hidden_block_count_z
    {76u, 2u, 4},   // hidden_group_size_x
    {78u, 2u, 5},   // hidden_group_size_y
    {80u, 2u, 6},   // hidden_group_size_z
    {82u, 2u, 7},   // hidden_remainder_x
    {84u, 2u, 8},   // hidden_remainder_y
    {86u, 2u, 9},   // hidden_remainder_z
    {104u, 8u, 10},   // hidden_global_offset_x
    {112u, 8u, 11},   // hidden_global_offset_y
    {120u, 8u, 12},   // hidden_global_offset_z
    {128u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_79[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 8u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {44u, 4u, 0},   // by_value/global
    {48u, 4u, 1},   // hidden_block_count_x
    {52u, 4u, 2},   // hidden_block_count_y
    {56u, 4u, 3},   // hidden_block_count_z
    {60u, 2u, 4},   // hidden_group_size_x
    {62u, 2u, 5},   // hidden_group_size_y
    {64u, 2u, 6},   // hidden_group_size_z
    {66u, 2u, 7},   // hidden_remainder_x
    {68u, 2u, 8},   // hidden_remainder_y
    {70u, 2u, 9},   // hidden_remainder_z
    {88u, 8u, 10},   // hidden_global_offset_x
    {96u, 8u, 11},   // hidden_global_offset_y
    {104u, 8u, 12},   // hidden_global_offset_z
    {112u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_80[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {40u, 4u, 1},   // hidden_block_count_x
    {44u, 4u, 2},   // hidden_block_count_y
    {48u, 4u, 3},   // hidden_block_count_z
    {52u, 2u, 4},   // hidden_group_size_x
    {54u, 2u, 5},   // hidden_group_size_y
    {56u, 2u, 6},   // hidden_group_size_z
    {58u, 2u, 7},   // hidden_remainder_x
    {60u, 2u, 8},   // hidden_remainder_y
    {62u, 2u, 9},   // hidden_remainder_z
    {80u, 8u, 10},   // hidden_global_offset_x
    {88u, 8u, 11},   // hidden_global_offset_y
    {96u, 8u, 12},   // hidden_global_offset_z
    {104u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_81[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
};
static const RtArg k_args_82[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
};
static const RtArg k_args_83[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
};
static const RtArg k_args_84[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
    {40u, 4u, 0},   // by_value/global
    {44u, 4u, 0},   // by_value/global
    {48u, 4u, 0},   // by_value/global
};
static const RtArg k_args_85[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
};
static const RtArg k_args_86[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
};
static const RtArg k_args_87[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 8u, 0},   // by_value/global
    {32u, 4u, 0},   // by_value/global
    {36u, 4u, 0},   // by_value/global
};
static const RtArg k_args_88[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 4u, 0},   // by_value/global
    {20u, 4u, 0},   // by_value/global
};
static const RtArg k_args_89[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 8u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
    {28u, 4u, 0},   // by_value/global
};
static const RtArg k_args_90[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 4u, 0},   // by_value/global
    {20u, 4u, 0},   // by_value/global
};
static const RtArg k_args_91[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 4u, 0},   // by_value/global
    {20u, 4u, 0},   // by_value/global
};
static const RtArg k_args_92[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 4u, 0},   // by_value/global
    {20u, 4u, 0},   // by_value/global
};
static const RtArg k_args_93[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 4u, 0},   // by_value/global
    {24u, 4u, 1},   // hidden_block_count_x
    {28u, 4u, 2},   // hidden_block_count_y
    {32u, 4u, 3},   // hidden_block_count_z
    {36u, 2u, 4},   // hidden_group_size_x
    {38u, 2u, 5},   // hidden_group_size_y
    {40u, 2u, 6},   // hidden_group_size_z
    {42u, 2u, 7},   // hidden_remainder_x
    {44u, 2u, 8},   // hidden_remainder_y
    {46u, 2u, 9},   // hidden_remainder_z
    {48u, 8u, 10},   // hidden_global_offset_x
    {56u, 8u, 11},   // hidden_global_offset_y
    {64u, 8u, 12},   // hidden_global_offset_z
    {72u, 2u, 13},   // hidden_grid_dims
};
static const RtArg k_args_94[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 4u, 0},   // by_value/global
    {20u, 4u, 0},   // by_value/global
};
static const RtArg k_args_95[] = {
    {0u, 8u, 0},   // by_value/global
    {8u, 8u, 0},   // by_value/global
    {16u, 4u, 0},   // by_value/global
    {20u, 4u, 0},   // by_value/global
    {24u, 4u, 0},   // by_value/global
};

static const RtKernel k_table[] = {
    {"_Z10nvfp4_gemvILi1ELi1EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<1,1>", 22u, k_args_0, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi1ELi2EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<1,2>", 22u, k_args_1, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi1ELi3EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<1,3>", 22u, k_args_2, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi1ELi4EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<1,4>", 22u, k_args_3, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi2ELi1EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<2,1>", 22u, k_args_4, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi2ELi2EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<2,2>", 22u, k_args_5, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi2ELi3EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<2,3>", 22u, k_args_6, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi2ELi4EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<2,4>", 22u, k_args_7, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi3ELi1EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<3,1>", 22u, k_args_8, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi3ELi2EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<3,2>", 22u, k_args_9, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi3ELi3EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<3,3>", 22u, k_args_10, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi3ELi4EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<3,4>", 22u, k_args_11, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi4ELi1EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<4,1>", 22u, k_args_12, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi4ELi2EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<4,2>", 22u, k_args_13, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi4ELi3EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<4,3>", 22u, k_args_14, 320u, 1024u, 0u},
    {"_Z10nvfp4_gemvILi4ELi4EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv<4,4>", 22u, k_args_15, 320u, 1024u, 0u},
    {"_Z10quant_act4PKfPhPfii", "quant_act4", 5u, k_args_16, 32u, 0u, 0u},
    {"_Z10silu_mul_kPfPKfS1_x", "silu_mul_k", 17u, k_args_17, 288u, 0u, 0u},
    {"_Z11fa_decode_kPfS_S_PKjPKfS1_S3_S1_S3_iiiiiiii", "fa_decode_k", 17u, k_args_18, 104u, 648u, 0u},
    {"_Z11scale_mul_kPffx", "scale_mul_k", 16u, k_args_19, 280u, 0u, 0u},
    {"_Z11split_qkv_kPfS_S_PKfiiii", "split_qkv_k", 21u, k_args_20, 304u, 0u, 0u},
    {"_Z12quant_rows_kPjPfPKfiiiii", "quant_rows_k", 8u, k_args_21, 44u, 0u, 0u},
    {"_Z13add_inplace_kPfPKfx", "add_inplace_k", 16u, k_args_22, 280u, 0u, 0u},
    {"_Z13argmax_comb_kPKfPKiiPi", "argmax_comb_k", 17u, k_args_23, 288u, 256u, 0u},
    {"_Z13argmax_part_kPKfxiPfPi", "argmax_part_k", 18u, k_args_24, 296u, 256u, 0u},
    {"_Z13conv1d_silu_kPfPKfS1_S1_iii", "conv1d_silu_k", 20u, k_args_25, 304u, 0u, 0u},
    {"_Z13kv_append_k_kPjPfPKfiiiiii", "kv_append_k_k", 9u, k_args_26, 48u, 0u, 0u},
    {"_Z13kv_append_v_kPjPfS0_PKfiiiiii", "kv_append_v_k", 10u, k_args_27, 56u, 0u, 0u},
    {"_Z13sigmoid_mul_kPfPKfS1_x", "sigmoid_mul_k", 17u, k_args_28, 288u, 0u, 0u},
    {"_Z13ssm_ab_gate_kPfS_S_PKfS1_S1_S1_S1_ii", "ssm_ab_gate_k", 23u, k_args_29, 328u, 64u, 0u},
    {"_Z13vit_ln_kernelPfPKfS1_S1_iif", "vit_ln_kernel", 20u, k_args_30, 304u, 1024u, 0u},
    {"_Z14attn_q_quant_kPjPfPKfiiiii", "attn_q_quant_k", 21u, k_args_31, 304u, 0u, 0u},
    {"_Z14gather_heads_kPfPKfiiiiix", "gather_heads_k", 21u, k_args_32, 304u, 0u, 0u},
    {"_Z15nvfp4_gemv_wideILi1ELi1EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv_wide<1,1>", 22u, k_args_33, 320u, 1024u, 0u},
    {"_Z15nvfp4_gemv_wideILi1ELi2EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv_wide<1,2>", 22u, k_args_34, 320u, 1024u, 0u},
    {"_Z15nvfp4_gemv_wideILi2ELi1EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv_wide<2,1>", 22u, k_args_35, 320u, 1024u, 0u},
    {"_Z15nvfp4_gemv_wideILi2ELi2EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv_wide<2,2>", 22u, k_args_36, 320u, 1024u, 0u},
    {"_Z15nvfp4_gemv_wideILi3ELi1EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv_wide<3,1>", 22u, k_args_37, 320u, 1024u, 0u},
    {"_Z15nvfp4_gemv_wideILi3ELi2EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv_wide<3,2>", 22u, k_args_38, 320u, 1024u, 0u},
    {"_Z15nvfp4_gemv_wideILi4ELi1EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv_wide<4,1>", 22u, k_args_39, 320u, 1024u, 0u},
    {"_Z15nvfp4_gemv_wideILi4ELi2EEvPKjPKhPKaS5_PKfPfiif", "nvfp4_gemv_wide<4,2>", 22u, k_args_40, 320u, 1024u, 0u},
    {"_Z15nvfp4_quant_actPKfPaS1_Pfii", "nvfp4_quant_act", 19u, k_args_41, 296u, 4u, 0u},
    {"_Z15quant_rows_a8_kPjS_PfS0_PKfiiiii", "quant_rows_a8_k", 10u, k_args_42, 60u, 0u, 0u},
    {"_Z15rmsnorm_gated_kPfPKfS1_S1_if", "rmsnorm_gated_k", 19u, k_args_43, 296u, 128u, 0u},
    {"_Z15scatter_heads_kPfPKfiiiiix", "scatter_heads_k", 21u, k_args_44, 304u, 0u, 0u},
    {"_Z15vit_attn_kernelPfPKfiiiif", "vit_attn_kernel", 20u, k_args_45, 296u, 37888u, 304u},
    {"_Z15vit_bias_kernelPfPKfii", "vit_bias_kernel", 17u, k_args_46, 280u, 0u, 0u},
    {"_Z15vit_gelu_kernelPfPKfxi", "vit_gelu_kernel", 17u, k_args_47, 288u, 0u, 0u},
    {"_Z15vit_rope_kernelPfPKfS1_iiii", "vit_rope_kernel", 20u, k_args_48, 296u, 0u, 0u},
    {"_Z16fa_decode_comb_kPfPKfS1_S1_ii", "fa_decode_comb_k", 6u, k_args_49, 40u, 0u, 0u},
    {"_Z16fa_decode_rows_kPfS_S_PKjPKfS1_S3_S1_S3_S3_miiiiiiiiiimmi", "fa_decode_rows_k", 24u, k_args_50, 148u, 648u, 0u},
    {"_Z17nvfp4_gemm_kernelPKjPKhPKaS4_PKfPfiiif", "nvfp4_gemm_kernel", 23u, k_args_51, 320u, 10240u, 0u},
    {"_Z17vit_bias_s_kernelPfPKfiii", "vit_bias_s_kernel", 18u, k_args_52, 288u, 0u, 0u},
    {"_Z19conv_state_update_kPfS_PKfS1_iii", "conv_state_update_k", 20u, k_args_53, 304u, 0u, 0u},
    {"_Z20nvfp4_gemm_kernel_ngPKjPKhPKaS4_PKfPfiiif", "nvfp4_gemm_kernel_ng", 23u, k_args_54, 320u, 10240u, 0u},
    {"_Z21fa_decode_rows_comb_kPfPKfS1_S1_iiiiiim", "fa_decode_rows_comb_k", 11u, k_args_55, 64u, 0u, 0u},
    {"_Z21vit_linear_f16_kernelPfPK6__halfPKfiiiii", "vit_linear_f16_kernel", 21u, k_args_56, 304u, 8192u, 0u},
    {"_Z5gdn_kPfPKfS1_S1_S1_S1_S_iiiiii", "gdn_k", 26u, k_args_57, 336u, 0u, 528u},
    {"_Z6fill_kPffx", "fill_k", 16u, k_args_58, 280u, 0u, 0u},
    {"_Z6gdn_k2ILi32EEvPfPKfS2_S2_S2_S2_S0_S0_iiiii", "gdn_k2<32>", 13u, k_args_59, 84u, 512u, 144u},
    {"_Z6rope_kPfS_PKiiiiiiiif", "rope_k", 24u, k_args_60, 312u, 0u, 0u},
    {"_Z7embed_kPfPKjPKtPKiii", "embed_k", 19u, k_args_61, 296u, 0u, 0u},
    {"_Z7fa_int4PKjPKfS0_S2_S0_S2_Pfiiiiii", "fa_int4", 13u, k_args_62, 80u, 58880u, 116u},
    {"_Z8argmax_kPKfxPi", "argmax_k", 16u, k_args_63, 280u, 256u, 0u},
    {"_Z8l2norm_kPfif", "l2norm_k", 16u, k_args_64, 272u, 128u, 0u},
    {"_Z9concat2_kPfPKfS1_ix", "concat2_k", 18u, k_args_65, 296u, 0u, 0u},
    {"_Z9gemm_w4a4PKjS0_PKfS2_Pfiii", "gemm_w4a4", 8u, k_args_66, 52u, 27136u, 0u},
    {"_Z9gemv_w4a4ILi1EEvPKjPKtS1_PKfPfii", "gemv_w4a4<1>", 7u, k_args_67, 48u, 0u, 0u},
    {"_Z9gemv_w4a4ILi2EEvPKjPKtS1_PKfPfii", "gemv_w4a4<2>", 7u, k_args_68, 48u, 0u, 0u},
    {"_Z9gemv_w4a4ILi3EEvPKjPKtS1_PKfPfii", "gemv_w4a4<3>", 7u, k_args_69, 48u, 0u, 0u},
    {"_Z9gemv_w4a4ILi4EEvPKjPKtS1_PKfPfii", "gemv_w4a4<4>", 7u, k_args_70, 48u, 0u, 0u},
    {"_Z9gemv_w4a8ILi1ELb0ELi1EEvPKjPKtS1_S1_PKfPKiPfii", "gemv_w4a8<1,false,1>", 9u, k_args_71, 64u, 0u, 0u},
    {"_Z9gemv_w4a8ILi1ELb1ELi1EEvPKjPKtS1_S1_PKfPKiPfii", "gemv_w4a8<1,true,1>", 22u, k_args_72, 320u, 0u, 0u},
    {"_Z9gemv_w4a8ILi2ELb0ELi2EEvPKjPKtS1_S1_PKfPKiPfii", "gemv_w4a8<2,false,2>", 9u, k_args_73, 64u, 0u, 0u},
    {"_Z9gemv_w4a8ILi2ELb1ELi2EEvPKjPKtS1_S1_PKfPKiPfii", "gemv_w4a8<2,true,2>", 22u, k_args_74, 320u, 0u, 0u},
    {"_Z9gemv_w4a8ILi3ELb0ELi2EEvPKjPKtS1_S1_PKfPKiPfii", "gemv_w4a8<3,false,2>", 9u, k_args_75, 64u, 0u, 0u},
    {"_Z9gemv_w4a8ILi3ELb1ELi2EEvPKjPKtS1_S1_PKfPKiPfii", "gemv_w4a8<3,true,2>", 22u, k_args_76, 320u, 0u, 0u},
    {"_Z9gemv_w4a8ILi4ELb0ELi2EEvPKjPKtS1_S1_PKfPKiPfii", "gemv_w4a8<4,false,2>", 9u, k_args_77, 64u, 0u, 0u},
    {"_Z9gemv_w4a8ILi4ELb1ELi2EEvPKjPKtS1_S1_PKfPKiPfii", "gemv_w4a8<4,true,2>", 22u, k_args_78, 320u, 0u, 0u},
    {"_Z9quant_actPKfPaS1_PfPiii", "quant_act", 20u, k_args_79, 304u, 4u, 0u},
    {"_Z9rmsnorm_kPfPKfS1_ifi", "rmsnorm_k", 19u, k_args_80, 296u, 128u, 0u},
    {"gelu_mul_k", "gelu_mul_k", 5u, k_args_81, 32u, 0u, 0u},
    {"gemv_f32_k", "gemv_f32_k", 6u, k_args_82, 40u, 0u, 0u},
    {"gemv_f32_warp_k", "gemv_f32_warp_k", 6u, k_args_83, 40u, 1024u, 0u},
    {"gemv_i8_k", "gemv_i8_k", 9u, k_args_84, 56u, 0u, 0u},
    {"iq2s_dequant_k", "iq2s_dequant_k", 5u, k_args_85, 32u, 0u, 0u},
    {"iq3s_dequant_k", "iq3s_dequant_k", 5u, k_args_86, 32u, 0u, 0u},
    {"iq3xxs_dequant_k", "iq3xxs_dequant_k", 6u, k_args_87, 40u, 0u, 0u},
    {"iq4nl_dequant_k", "iq4nl_dequant_k", 4u, k_args_88, 24u, 0u, 0u},
    {"iq4nl_to_i8_k", "iq4nl_to_i8_k", 5u, k_args_89, 32u, 0u, 0u},
    {"iq4xs_dequant_k", "iq4xs_dequant_k", 4u, k_args_90, 24u, 0u, 0u},
    {"q2_0_dequant_k", "q2_0_dequant_k", 4u, k_args_91, 24u, 0u, 0u},
    {"q4_0_dequant_k", "q4_0_dequant_k", 4u, k_args_92, 24u, 0u, 0u},
    {"q6k_dequant", "q6k_dequant", 16u, k_args_93, 80u, 0u, 0u},
    {"q8_0_dequant_k", "q8_0_dequant_k", 4u, k_args_94, 24u, 0u, 0u},
    {"softmax_k", "softmax_k", 5u, k_args_95, 32u, 1024u, 0u},
};
static const int k_table_n = 96;
