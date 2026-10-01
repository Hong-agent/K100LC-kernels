.text
k_q6k_dequant:
v_mov_b32_e32 v1, 0
s_load_dwordx2 s[16:17], s[4:5], 0x0
s_load_dwordx2 s[18:19], s[4:5], 0x8
s_load_dword s20, s[4:5], 0x10
s_waitcnt lgkmcnt(0)
s_load_dword s64, s[4:5], 0x24
s_waitcnt lgkmcnt(0)
s_and_b32 s64, s64, 0xffff
v_mov_b32_e32 v23, s64
v_mov_b32_e32 v22, s6
v_mul_lo_u32 v22, v22, v23
v_add_u32_e32 v22, v22, v0
v_mov_b32_e32 v2, v22
v_mov_b32_e32 v64, s20
v_cmp_lt_u32_e64 vcc, v2, v64
s_and_saveexec_b64 s[48:49], vcc
s_cbranch_execz if_end_1
v_mov_b32_e32 v64, 210
v_mul_lo_u32 v65, v2, v64
v_mov_b32_e32 v3, v65
v_mov_b32_e32 v64, 256
v_mul_lo_u32 v65, v2, v64
v_mov_b32_e32 v4, v65
v_mov_b32_e32 v64, 208
v_add_u32_e32 v65, v3, v64
v_mov_b32_e32 v64, v65
v_mov_b32_e32 v24, s16
v_mov_b32_e32 v25, s17
v_add_co_u32_e32 v24, vcc, v24, v64
v_addc_co_u32_e32 v25, vcc, v25, v1, vcc
global_load_ushort v65, v[24:25], off
s_waitcnt vmcnt(0)
v_cvt_f32_f16_e32 v65, v65
v_mov_b32_e32 v5, v65
s_mov_b32 s65, 2
s_mov_b32 s21, 0
for_loop_2:
s_cmp_lt_u32 s21, s65
s_cbranch_scc0 for_end_4
v_mov_b32_e32 v64, 192
v_add_u32_e32 v65, v3, v64
s_mov_b32 s66, 8
s_mul_i32 s67, s21, s66
v_mov_b32_e32 v64, s67
v_add_u32_e32 v66, v65, v64
v_mov_b32_e32 v7, v66
s_mov_b32 s68, 32
s_mov_b32 s22, 0
for_loop_5:
s_cmp_lt_u32 s22, s68
s_cbranch_scc0 for_end_7
s_mov_b32 s69, 64
s_mul_i32 s70, s21, s69
v_mov_b32_e32 v64, s70
v_add_u32_e32 v65, v3, v64
v_mov_b32_e32 v64, s22
v_add_u32_e32 v66, v65, v64
v_mov_b32_e32 v64, v66
v_mov_b32_e32 v24, s16
v_mov_b32_e32 v25, s17
v_add_co_u32_e32 v24, vcc, v24, v64
v_addc_co_u32_e32 v25, vcc, v25, v1, vcc
global_load_ubyte v66, v[24:25], off
s_waitcnt vmcnt(0)
v_mov_b32_e32 v9, v66
s_mov_b32 s71, 64
s_mul_i32 s72, s21, s71
v_mov_b32_e32 v64, s72
v_add_u32_e32 v65, v3, v64
v_mov_b32_e32 v64, s22
v_add_u32_e32 v66, v65, v64
v_mov_b32_e32 v64, 32
v_add_u32_e32 v65, v66, v64
v_mov_b32_e32 v64, v65
v_mov_b32_e32 v24, s16
v_mov_b32_e32 v25, s17
v_add_co_u32_e32 v24, vcc, v24, v64
v_addc_co_u32_e32 v25, vcc, v25, v1, vcc
global_load_ubyte v65, v[24:25], off
s_waitcnt vmcnt(0)
v_mov_b32_e32 v10, v65
v_mov_b32_e32 v64, 128
v_add_u32_e32 v65, v3, v64
s_mov_b32 s73, 32
s_mul_i32 s74, s21, s73
v_mov_b32_e32 v64, s74
v_add_u32_e32 v66, v65, v64
v_mov_b32_e32 v64, s22
v_add_u32_e32 v65, v66, v64
v_mov_b32_e32 v64, v65
v_mov_b32_e32 v24, s16
v_mov_b32_e32 v25, s17
v_add_co_u32_e32 v24, vcc, v24, v64
v_addc_co_u32_e32 v25, vcc, v25, v1, vcc
global_load_ubyte v65, v[24:25], off
s_waitcnt vmcnt(0)
v_mov_b32_e32 v11, v65
v_mov_b32_e32 v64, 15
v_and_b32_e32 v65, v9, v64
v_mov_b32_e32 v64, 3
v_and_b32_e32 v66, v11, v64
v_mov_b32_e32 v64, 4
v_lshlrev_b32_e32 v67, v64, v66
v_or_b32_e32 v64, v65, v67
v_mov_b32_e32 v67, 32
v_sub_u32_e32 v65, v64, v67
v_mov_b32_e32 v12, v65
v_mov_b32_e32 v64, 15
v_and_b32_e32 v65, v10, v64
v_mov_b32_e32 v64, 2
v_lshrrev_b32_e32 v66, v64, v11
v_mov_b32_e32 v64, 3
v_and_b32_e32 v67, v66, v64
v_mov_b32_e32 v64, 4
v_lshlrev_b32_e32 v66, v64, v67
v_or_b32_e32 v64, v65, v66
v_mov_b32_e32 v66, 32
v_sub_u32_e32 v65, v64, v66
v_mov_b32_e32 v13, v65
v_mov_b32_e32 v64, 4
v_lshrrev_b32_e32 v65, v64, v9
v_mov_b32_e32 v64, 4
v_lshrrev_b32_e32 v66, v64, v11
v_mov_b32_e32 v64, 3
v_and_b32_e32 v67, v66, v64
v_mov_b32_e32 v64, 4
v_lshlrev_b32_e32 v66, v64, v67
v_or_b32_e32 v64, v65, v66
v_mov_b32_e32 v66, 32
v_sub_u32_e32 v65, v64, v66
v_mov_b32_e32 v14, v65
v_mov_b32_e32 v64, 4
v_lshrrev_b32_e32 v65, v64, v10
v_mov_b32_e32 v64, 6
v_lshrrev_b32_e32 v66, v64, v11
v_mov_b32_e32 v64, 3
v_and_b32_e32 v67, v66, v64
v_mov_b32_e32 v64, 4
v_lshlrev_b32_e32 v66, v64, v67
v_or_b32_e32 v64, v65, v66
v_mov_b32_e32 v66, 32
v_sub_u32_e32 v65, v64, v66
v_mov_b32_e32 v15, v65
s_mov_b32 s75, 4
s_lshr_b32 s76, s22, s75
s_mov_b32 s23, s76
v_mov_b32_e32 v64, s23
v_add_u32_e32 v65, v7, v64
v_mov_b32_e32 v64, 0
v_add_u32_e32 v66, v65, v64
v_mov_b32_e32 v64, v66
v_mov_b32_e32 v24, s16
v_mov_b32_e32 v25, s17
v_add_co_u32_e32 v24, vcc, v24, v64
v_addc_co_u32_e32 v25, vcc, v25, v1, vcc
global_load_ubyte v66, v[24:25], off
s_waitcnt vmcnt(0)
v_lshlrev_b32_e32 v64, 24, v66
v_ashrrev_i32_e32 v64, 24, v64
v_mov_b32_e32 v17, v64
v_mov_b32_e32 v64, s23
v_add_u32_e32 v65, v7, v64
v_mov_b32_e32 v64, 2
v_add_u32_e32 v66, v65, v64
v_mov_b32_e32 v64, v66
v_mov_b32_e32 v24, s16
v_mov_b32_e32 v25, s17
v_add_co_u32_e32 v24, vcc, v24, v64
v_addc_co_u32_e32 v25, vcc, v25, v1, vcc
global_load_ubyte v66, v[24:25], off
s_waitcnt vmcnt(0)
v_lshlrev_b32_e32 v64, 24, v66
v_ashrrev_i32_e32 v64, 24, v64
v_mov_b32_e32 v18, v64
v_mov_b32_e32 v64, s23
v_add_u32_e32 v65, v7, v64
v_mov_b32_e32 v64, 4
v_add_u32_e32 v66, v65, v64
v_mov_b32_e32 v64, v66
v_mov_b32_e32 v24, s16
v_mov_b32_e32 v25, s17
v_add_co_u32_e32 v24, vcc, v24, v64
v_addc_co_u32_e32 v25, vcc, v25, v1, vcc
global_load_ubyte v66, v[24:25], off
s_waitcnt vmcnt(0)
v_lshlrev_b32_e32 v64, 24, v66
v_ashrrev_i32_e32 v64, 24, v64
v_mov_b32_e32 v19, v64
v_mov_b32_e32 v64, s23
v_add_u32_e32 v65, v7, v64
v_mov_b32_e32 v64, 6
v_add_u32_e32 v66, v65, v64
v_mov_b32_e32 v64, v66
v_mov_b32_e32 v24, s16
v_mov_b32_e32 v25, s17
v_add_co_u32_e32 v24, vcc, v24, v64
v_addc_co_u32_e32 v25, vcc, v25, v1, vcc
global_load_ubyte v66, v[24:25], off
s_waitcnt vmcnt(0)
v_lshlrev_b32_e32 v64, 24, v66
v_ashrrev_i32_e32 v64, 24, v64
v_mov_b32_e32 v20, v64
s_mov_b32 s77, 128
s_mul_i32 s78, s21, s77
v_mov_b32_e32 v64, s78
v_add_u32_e32 v65, v4, v64
v_mov_b32_e32 v21, v65
v_cvt_f32_i32_e32 v64, v17
v_mul_f32_e32 v65, v5, v64
v_cvt_f32_i32_e32 v64, v12
v_mul_f32_e32 v66, v65, v64
v_mov_b32_e32 v64, s22
v_add_u32_e32 v65, v21, v64
v_lshlrev_b32_e32 v64, 2, v65
v_mov_b32_e32 v24, s18
v_mov_b32_e32 v25, s19
v_add_co_u32_e32 v24, vcc, v24, v64
v_addc_co_u32_e32 v25, vcc, v25, v1, vcc
global_store_dword v[24:25], v66, off
v_cvt_f32_i32_e32 v64, v18
v_mul_f32_e32 v65, v5, v64
v_cvt_f32_i32_e32 v64, v13
v_mul_f32_e32 v66, v65, v64
v_mov_b32_e32 v64, 32
v_add_u32_e32 v65, v21, v64
v_mov_b32_e32 v64, s22
v_add_u32_e32 v67, v65, v64
v_lshlrev_b32_e32 v64, 2, v67
v_mov_b32_e32 v24, s18
v_mov_b32_e32 v25, s19
v_add_co_u32_e32 v24, vcc, v24, v64
v_addc_co_u32_e32 v25, vcc, v25, v1, vcc
global_store_dword v[24:25], v66, off
v_cvt_f32_i32_e32 v64, v19
v_mul_f32_e32 v65, v5, v64
v_cvt_f32_i32_e32 v64, v14
v_mul_f32_e32 v66, v65, v64
v_mov_b32_e32 v64, 64
v_add_u32_e32 v65, v21, v64
v_mov_b32_e32 v64, s22
v_add_u32_e32 v67, v65, v64
v_lshlrev_b32_e32 v64, 2, v67
v_mov_b32_e32 v24, s18
v_mov_b32_e32 v25, s19
v_add_co_u32_e32 v24, vcc, v24, v64
v_addc_co_u32_e32 v25, vcc, v25, v1, vcc
global_store_dword v[24:25], v66, off
v_cvt_f32_i32_e32 v64, v20
v_mul_f32_e32 v65, v5, v64
v_cvt_f32_i32_e32 v64, v15
v_mul_f32_e32 v66, v65, v64
v_mov_b32_e32 v64, 96
v_add_u32_e32 v65, v21, v64
v_mov_b32_e32 v64, s22
v_add_u32_e32 v67, v65, v64
v_lshlrev_b32_e32 v64, 2, v67
v_mov_b32_e32 v24, s18
v_mov_b32_e32 v25, s19
v_add_co_u32_e32 v24, vcc, v24, v64
v_addc_co_u32_e32 v25, vcc, v25, v1, vcc
global_store_dword v[24:25], v66, off
for_inc_6:
s_add_i32 s22, s22, 1
s_branch for_loop_5
for_end_7:
for_inc_3:
s_add_i32 s21, s21, 1
s_branch for_loop_2
for_end_4:
s_or_b64 exec, exec, s[48:49]
if_end_1:
s_endpgm
