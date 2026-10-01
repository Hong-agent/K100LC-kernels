.text
k_rope_apply_k:
v_mov_b32_e32 v1, 0
s_load_dwordx2 s[16:17], s[4:5], 0x0
s_load_dwordx2 s[18:19], s[4:5], 0x8
s_load_dwordx2 s[20:21], s[4:5], 0x10
s_load_dwordx2 s[22:23], s[4:5], 0x18
s_load_dword s24, s[4:5], 0x20
s_load_dword s25, s[4:5], 0x24
s_waitcnt lgkmcnt(0)
s_mov_b32 s26, s6
s_mov_b32 s27, s26
v_mov_b32_e32 v12, v0
v_and_b32_e32 v12, 63, v12
v_mov_b32_e32 v3, v12
s_mov_b32 s64, 1
s_lshr_b32 s65, s25, s64
s_mov_b32 s28, s65
s_mov_b32 s66, 63
s_add_u32 s67, s28, s66
s_mov_b32 s68, 6
s_lshr_b32 s69, s67, s68
s_mov_b32 s29, s69
s_cmp_lt_u32 s27, s24
s_cbranch_scc0 if_end_1
s_mov_b32 s30, 0
for_loop_3:
s_cmp_lt_u32 s30, s29
s_cbranch_scc0 for_end_5
s_mov_b32 s70, 6
s_lshl_b32 s71, s30, s70
v_mov_b32_e32 v64, s71
v_add_u32_e32 v65, v64, v3
v_mov_b32_e32 v7, v65
v_mov_b32_e32 v64, s28
v_cmp_lt_u32_e64 vcc, v7, v64
s_and_saveexec_b64 s[48:49], vcc
s_cbranch_execz if_end_6
s_mul_i32 s72, s27, s28
v_mov_b32_e32 v64, s72
v_add_u32_e32 v65, v64, v7
v_lshlrev_b32_e32 v64, 2, v65
v_mov_b32_e32 v13, s20
v_mov_b32_e32 v14, s21
v_add_co_u32_e32 v13, vcc, v13, v64
v_addc_co_u32_e32 v14, vcc, v14, v1, vcc
global_load_dword v65, v[13:14], off
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, v65
s_mul_i32 s73, s27, s28
v_mov_b32_e32 v64, s73
v_add_u32_e32 v65, v64, v7
v_lshlrev_b32_e32 v64, 2, v65
v_mov_b32_e32 v13, s22
v_mov_b32_e32 v14, s23
v_add_co_u32_e32 v13, vcc, v13, v64
v_addc_co_u32_e32 v14, vcc, v14, v1, vcc
global_load_dword v65, v[13:14], off
s_waitcnt vmcnt(0)
v_mov_b32_e32 v9, v65
s_mul_i32 s74, s27, s25
v_mov_b32_e32 v64, s74
v_add_u32_e32 v65, v64, v7
v_lshlrev_b32_e32 v64, 2, v65
v_mov_b32_e32 v13, s18
v_mov_b32_e32 v14, s19
v_add_co_u32_e32 v13, vcc, v13, v64
v_addc_co_u32_e32 v14, vcc, v14, v1, vcc
global_load_dword v65, v[13:14], off
s_waitcnt vmcnt(0)
v_mov_b32_e32 v10, v65
s_mul_i32 s75, s27, s25
v_mov_b32_e32 v64, s75
v_add_u32_e32 v65, v64, v7
v_mov_b32_e32 v64, s28
v_add_u32_e32 v66, v65, v64
v_lshlrev_b32_e32 v64, 2, v66
v_mov_b32_e32 v13, s18
v_mov_b32_e32 v14, s19
v_add_co_u32_e32 v13, vcc, v13, v64
v_addc_co_u32_e32 v14, vcc, v14, v1, vcc
global_load_dword v66, v[13:14], off
s_waitcnt vmcnt(0)
v_mov_b32_e32 v11, v66
v_mul_f32_e32 v64, v10, v8
v_mul_f32_e32 v65, v11, v9
v_sub_f32_e32 v66, v64, v65
s_mul_i32 s76, s27, s25
v_mov_b32_e32 v65, s76
v_add_u32_e32 v64, v65, v7
v_lshlrev_b32_e32 v65, 2, v64
v_mov_b32_e32 v13, s16
v_mov_b32_e32 v14, s17
v_add_co_u32_e32 v13, vcc, v13, v65
v_addc_co_u32_e32 v14, vcc, v14, v1, vcc
global_store_dword v[13:14], v66, off
s_waitcnt vmcnt(0)
v_mul_f32_e32 v64, v10, v9
v_mul_f32_e32 v65, v11, v8
v_add_f32_e32 v66, v64, v65
s_mul_i32 s77, s27, s25
v_mov_b32_e32 v65, s77
v_add_u32_e32 v64, v65, v7
v_mov_b32_e32 v65, s28
v_add_u32_e32 v67, v64, v65
v_lshlrev_b32_e32 v65, 2, v67
v_mov_b32_e32 v13, s16
v_mov_b32_e32 v14, s17
v_add_co_u32_e32 v13, vcc, v13, v65
v_addc_co_u32_e32 v14, vcc, v14, v1, vcc
global_store_dword v[13:14], v66, off
s_waitcnt vmcnt(0)
s_or_b64 exec, exec, s[48:49]
if_end_6:
for_inc_4:
s_add_i32 s30, s30, 1
s_branch for_loop_3
for_end_5:
if_end_1:
s_endpgm
