//! DXC 1.9's normalised output for the fixtures of intrinsics_oracle.rs. Generated: run that
//! test with XCL_DXIL_DXC set and XCL_DXIL_BLESS=1 to rewrite it.
#![allow(dead_code)]

pub const GOLDEN: &str = r#"
=== cf_exp2
  out0_0 = float 0x3F10000000000000
=== cf_exp2_0
  out0_0 = float 1.000000e+00
=== cf_exp2_sub
  out0_0 = float 0x3730000000000000
=== cf_log2
  out0_0 = float 3.000000e+00
=== cf_sqrt
  out0_0 = float 0x3FF6A09E60000000
=== cf_rsqrt
  out0_0 = float 5.000000e-01
=== cf_floor
  out0_0 = float -2.000000e+00
=== cf_ceil
  out0_0 = float -1.000000e+00
=== cf_round
  out0_0 = float 2.000000e+00
=== cf_round_odd
  out0_0 = float 4.000000e+00
=== cf_trunc
  out0_0 = float -1.000000e+00
=== cf_frac
  out0_0 = float 2.500000e-01
=== cf_abs
  out0_0 = float 1.500000e+00
=== cf_saturate
  out0_0 = float 1.000000e+00
=== cf_max_f
  out0_0 = float 2.000000e+00
=== cf_min_f
  out0_0 = float 1.000000e+00
=== cf_min_u
  out0_0 = i32 11
=== cf_max_u
  out0_0 = i32 -1
=== cf_clamp_f
  out0_0 = float 1.000000e+00
=== cf_clamp_u
  out0_0 = i32 7
=== cf_mad_f
  out0_0 = float 0x4018666660000000
=== cf_mad_u
  out0_0 = i32 17
=== cf_reversebits
  out0_0 = i32 335544320
=== cf_countbits
  out0_0 = i32 8
=== cf_firstbitlow
  out0_0 = i32 8
=== cf_firstbitlow_0
  out0_0 = i32 -1
=== cf_firstbithigh
  out0_0 = i32 8
=== cf_firstbithigh_0
  out0_0 = i32 -1
=== cf_isnan
  out0_0 = i32 0
=== asuint_f
  %1 = bitcast float in0_0 to i32
  out0_0 = i32 %1
=== asuint_i
  out0_0 = i32 in0_0
=== asuint_u
  out0_0 = i32 in0_0
=== asuint_f3
  %1 = bitcast float in0_0 to i32
  %2 = bitcast float in0_1 to i32
  %3 = bitcast float in0_2 to i32
  out0_0 = i32 %1
  out0_1 = i32 %2
  out0_2 = i32 %3
=== asuint_lit
  out0_0 = i32 1069547520
=== asuint_ilit
  out0_0 = i32 -1
=== asint_f
  %1 = bitcast float in0_0 to i32
  out0_0 = i32 %1
=== asint_u
  out0_0 = i32 in0_0
=== asint_f2
  %1 = bitcast float in0_0 to i32
  %2 = bitcast float in0_1 to i32
  out0_0 = i32 %1
  out0_1 = i32 %2
=== asfloat_u
  %1 = bitcast i32 in0_0 to float
  out0_0 = float %1
=== asfloat_i
  %1 = bitcast i32 in0_0 to float
  out0_0 = float %1
=== asfloat_f
  out0_0 = float in0_0
=== asfloat_u4
  %1 = bitcast i32 in0_0 to float
  %2 = bitcast i32 in0_1 to float
  %3 = bitcast i32 in0_2 to float
  %4 = bitcast i32 in0_3 to float
  out0_0 = float %1
  out0_1 = float %2
  out0_2 = float %3
  out0_3 = float %4
=== asfloat_inf
  out0_0 = float 0x7FF0000000000000
=== asfloat_ilit
  out0_0 = float 1.000000e+00
=== asdouble_u
  %1 = call double @dx.op.makeDouble.f64(i32 101, i32 in0_0, i32 in1_0) !precise
  out0_0 = double %1
=== asdouble_lit
  %1 = call double @dx.op.makeDouble.f64(i32 101, i32 in0_0, i32 1071644672) !precise
  out0_0 = double %1
=== asdouble_lits
  out0_0 = double 0x7FE0000000000000
=== asdouble_u2
  %1 = call double @dx.op.makeDouble.f64(i32 101, i32 in0_0, i32 in1_0) !precise
  %2 = call double @dx.op.makeDouble.f64(i32 101, i32 in0_1, i32 in1_1) !precise
  out0_0 = double %1
  out0_1 = double %2
=== asdouble_i
  %1 = call double @dx.op.makeDouble.f64(i32 101, i32 in0_0, i32 in1_0) !precise
  out0_0 = double %1
=== asuint_d
  %1 = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double in0_0)
  %2 = extractvalue %dx.types.splitdouble %1, 0
  %3 = extractvalue %dx.types.splitdouble %1, 1
  out1_0 = i32 %2
  out2_0 = i32 %3
=== asuint_d2
  %1 = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double in0_0)
  %2 = extractvalue %dx.types.splitdouble %1, 0
  %3 = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double in0_1)
  %4 = extractvalue %dx.types.splitdouble %3, 0
  %5 = extractvalue %dx.types.splitdouble %1, 1
  %6 = extractvalue %dx.types.splitdouble %3, 1
  out1_0 = i32 %2
  out1_1 = i32 %4
  out2_0 = i32 %5
  out2_1 = i32 %6
=== f16tof32_u
  %1 = call float @dx.op.legacyF16ToF32(i32 131, i32 in0_0) !precise
  out0_0 = float %1
=== f16tof32_i
  %1 = call float @dx.op.legacyF16ToF32(i32 131, i32 in0_0) !precise
  out0_0 = float %1
=== f16tof32_u3
  %1 = call float @dx.op.legacyF16ToF32(i32 131, i32 in0_0) !precise
  %2 = call float @dx.op.legacyF16ToF32(i32 131, i32 in0_1) !precise
  %3 = call float @dx.op.legacyF16ToF32(i32 131, i32 in0_2) !precise
  out0_0 = float %1
  out0_1 = float %2
  out0_2 = float %3
=== f16tof32_lit
  %1 = call float @dx.op.legacyF16ToF32(i32 131, i32 15360) !precise
  out0_0 = float %1
=== f16tof32_f
  %1 = fptoui float in0_0 to i32
  %2 = call float @dx.op.legacyF16ToF32(i32 131, i32 %1) !precise
  out0_0 = float %2
=== f16tof32_u64
  %1 = trunc i64 in0_0 to i32
  %2 = call float @dx.op.legacyF16ToF32(i32 131, i32 %1) !precise
  out0_0 = float %2
=== f32tof16_f
  %1 = call i32 @dx.op.legacyF32ToF16(i32 130, float in0_0)
  out0_0 = i32 %1
=== f32tof16_d
  %1 = fptrunc double in0_0 to float
  %2 = call i32 @dx.op.legacyF32ToF16(i32 130, float %1)
  out0_0 = i32 %2
=== f32tof16_f2
  %1 = call i32 @dx.op.legacyF32ToF16(i32 130, float in0_0)
  %2 = call i32 @dx.op.legacyF32ToF16(i32 130, float in0_1)
  out0_0 = i32 %1
  out0_1 = i32 %2
=== f32tof16_u
  %1 = uitofp i32 in0_0 to float
  %2 = call i32 @dx.op.legacyF32ToF16(i32 130, float %1)
  out0_0 = i32 %2
=== select_s
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = select i1 %1, float in1_0, float in2_0
  out0_0 = float %2
=== select_v_s
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = select i1 %1, float in1_0, float in2_0
  %3 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_1)
  %4 = select i1 %3, float in1_0, float in2_1
  %5 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_2)
  %6 = select i1 %5, float in1_0, float in2_2
  out0_0 = float %2
  out0_1 = float %4
  out0_2 = float %6
=== select_v_v
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = select i1 %1, i32 in1_0, i32 in2_0
  %3 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_1)
  %4 = select i1 %3, i32 in1_1, i32 in2_1
  %5 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_2)
  %6 = select i1 %5, i32 in1_2, i32 in2_2
  out0_0 = i32 %2
  out0_1 = i32 %4
  out0_2 = i32 %6
=== select_s_v
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = select i1 %1, i32 in1_0, i32 in2_0
  %3 = select i1 %1, i32 in1_1, i32 in2_1
  %4 = select i1 %1, i32 in1_2, i32 in2_2
  out0_0 = i32 %2
  out0_1 = i32 %3
  out0_2 = i32 %4
=== select_v3_s_s
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = select i1 %1, float in1_0, float in2_0
  %3 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_1)
  %4 = select i1 %3, float in1_0, float in2_0
  %5 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_2)
  %6 = select i1 %5, float in1_0, float in2_0
  out0_0 = float %2
  out0_1 = float %4
  out0_2 = float %6
=== select_s_v_s
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = select i1 %1, float in1_0, float in2_0
  %3 = select i1 %1, float in1_1, float in2_0
  %4 = select i1 %1, float in1_2, float in2_0
  out0_0 = float %2
  out0_1 = float %3
  out0_2 = float %4
=== select_c2_v3
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = select i1 %1, float in1_0, float in2_0
  %3 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_1)
  %4 = select i1 %3, float in1_1, float in2_1
  out0_0 = float %2
  out0_1 = float %4
=== select_lit
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = select i1 %1, i32 in1_0, i32 -1
  out0_0 = i32 %2
=== select_ilit
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = select i1 %1, i32 in1_0, i32 5
  out0_0 = i32 %2
=== select_i_u
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = select i1 %1, i32 in1_0, i32 in2_0
  out0_0 = i32 %2
=== select_u_i
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = select i1 %1, i32 in1_0, i32 in2_0
  out0_0 = i32 %2
=== select_f_i
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = sitofp i32 in2_0 to float
  %3 = select i1 %1, float in1_0, float %2
  out0_0 = float %3
=== select_i_f
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = sitofp i32 in1_0 to float
  %3 = select i1 %1, float %2, float in2_0
  out0_0 = float %3
=== select_flit
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = select i1 %1, float 0.000000e+00, float in1_0
  out0_0 = float %2
=== select_f_lit
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = select i1 %1, float in1_0, float 1.500000e+00
  out0_0 = float %2
=== select_d_lit
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = select i1 %1, double in1_0, double 1.100000e+00
  out0_0 = double %2
=== select_u_cond
  %1 = icmp ne i32 0, in0_0
  %2 = select i1 %1, float in1_0, float in2_0
  out0_0 = float %2
=== select_f_cond
  %1 = fcmp une float 0.000000e+00, in0_0
  %2 = select i1 %1, i32 in1_0, i32 in2_0
  out0_0 = i32 %2
=== select_f_d
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = fpext float in1_0 to double
  %3 = select i1 %1, double %2, double in2_0
  out0_0 = double %3
=== select_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_0)
  %3 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in2_0)
  %4 = select i1 %1, i1 %2, i1 %3
  %5 = zext i1 %4 to i32
  out0_0 = i32 %5
=== select_u64
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = zext i32 in2_0 to i64
  %3 = select i1 %1, i64 in1_0, i64 %2
  out0_0 = i64 %3
=== select_same
  out0_0 = float in1_0
=== select_true
  out0_0 = i32 in0_0
=== and_s
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_0)
  %3 = and i1 %1, %2
  %4 = zext i1 %3 to i32
  out0_0 = i32 %4
=== and_v
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_0)
  %3 = and i1 %1, %2
  %4 = zext i1 %3 to i32
  %5 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_1)
  %6 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_1)
  %7 = and i1 %5, %6
  %8 = zext i1 %7 to i32
  %9 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_2)
  %10 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_2)
  %11 = and i1 %9, %10
  %12 = zext i1 %11 to i32
  out0_0 = i32 %4
  out0_1 = i32 %8
  out0_2 = i32 %12
=== and_v_s
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_0)
  %3 = and i1 %1, %2
  %4 = zext i1 %3 to i32
  %5 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_1)
  %6 = and i1 %5, %2
  %7 = zext i1 %6 to i32
  %8 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_2)
  %9 = and i1 %8, %2
  %10 = zext i1 %9 to i32
  out0_0 = i32 %4
  out0_1 = i32 %7
  out0_2 = i32 %10
=== and_u
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_0)
  %2 = icmp ne i32 0, in0_0
  %3 = and i1 %1, %2
  %4 = zext i1 %3 to i32
  out0_0 = i32 %4
=== and_v3_v2
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_0)
  %3 = and i1 %1, %2
  %4 = zext i1 %3 to i32
  %5 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_1)
  %6 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_1)
  %7 = and i1 %5, %6
  %8 = zext i1 %7 to i32
  out0_0 = i32 %4
  out0_1 = i32 %8
=== or_s
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_0)
  %3 = or i1 %1, %2
  %4 = zext i1 %3 to i32
  out0_0 = i32 %4
=== or_v_s
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_0)
  %3 = or i1 %1, %2
  %4 = zext i1 %3 to i32
  %5 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_1)
  %6 = or i1 %1, %5
  %7 = zext i1 %6 to i32
  %8 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_2)
  %9 = or i1 %1, %8
  %10 = zext i1 %9 to i32
  out0_0 = i32 %4
  out0_1 = i32 %7
  out0_2 = i32 %10
=== or_f
  %1 = fcmp une float 0.000000e+00, in0_0
  %2 = fcmp une float 0.000000e+00, in1_0
  %3 = or i1 %1, %2
  %4 = zext i1 %3 to i32
  out0_0 = i32 %4
=== any_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_2)
  %2 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %3 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_1)
  %4 = or i1 %2, %3
  %5 = or i1 %1, %4
  %6 = zext i1 %5 to i32
  out0_0 = i32 %6
=== any_u
  %1 = icmp ne i32 0, in0_2
  %2 = icmp ne i32 0, in0_0
  %3 = icmp ne i32 0, in0_1
  %4 = or i1 %2, %3
  %5 = or i1 %1, %4
  %6 = zext i1 %5 to i32
  out0_0 = i32 %6
=== any_f
  %1 = fcmp une float 0.000000e+00, in0_3
  %2 = fcmp une float 0.000000e+00, in0_2
  %3 = fcmp une float 0.000000e+00, in0_0
  %4 = fcmp une float 0.000000e+00, in0_1
  %5 = or i1 %3, %4
  %6 = or i1 %2, %5
  %7 = or i1 %1, %6
  %8 = zext i1 %7 to i32
  out0_0 = i32 %8
=== any_s
  %1 = fcmp une float 0.000000e+00, in0_0
  %2 = zext i1 %1 to i32
  out0_0 = i32 %2
=== any_i2
  %1 = icmp ne i32 0, in0_0
  %2 = icmp ne i32 0, in0_1
  %3 = or i1 %1, %2
  %4 = zext i1 %3 to i32
  out0_0 = i32 %4
=== all_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_1)
  %3 = and i1 %1, %2
  %4 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_2)
  %5 = and i1 %3, %4
  %6 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_3)
  %7 = and i1 %5, %6
  %8 = zext i1 %7 to i32
  out0_0 = i32 %8
=== all_f
  %1 = fcmp une float 0.000000e+00, in0_0
  %2 = fcmp une float 0.000000e+00, in0_1
  %3 = and i1 %1, %2
  %4 = zext i1 %3 to i32
  out0_0 = i32 %4
=== all_u
  %1 = icmp ne i32 0, in0_0
  %2 = zext i1 %1 to i32
  out0_0 = i32 %2
=== all_d
  %1 = fcmp une double 0.000000e+00, in0_0
  %2 = fcmp une double 0.000000e+00, in0_1
  %3 = and i1 %1, %2
  %4 = zext i1 %3 to i32
  out0_0 = i32 %4
=== all_u64
  %1 = icmp ne i64 0, in0_0
  %2 = icmp ne i64 0, in0_1
  %3 = and i1 %1, %2
  %4 = zext i1 %3 to i32
  out0_0 = i32 %4
=== isnan_f
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = zext i1 %1 to i32
  out0_0 = i32 %2
=== isnan_f3
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = zext i1 %1 to i32
  %3 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_1)
  %4 = zext i1 %3 to i32
  %5 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_2)
  %6 = zext i1 %5 to i32
  out0_0 = i32 %2
  out0_1 = i32 %4
  out0_2 = i32 %6
=== isnan_d
  %1 = fptrunc double in0_0 to float
  %2 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float %1)
  %3 = zext i1 %2 to i32
  out0_0 = i32 %3
=== isnan_i
  %1 = sitofp i32 in0_0 to float
  %2 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float %1)
  %3 = zext i1 %2 to i32
  out0_0 = i32 %3
=== isinf_f
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 9, float in0_0)
  %2 = zext i1 %1 to i32
  out0_0 = i32 %2
=== isfinite_f
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 10, float in0_0)
  %2 = zext i1 %1 to i32
  out0_0 = i32 %2
=== isnormal_f2
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 11, float in0_0)
  %2 = zext i1 %1 to i32
  %3 = call i1 @dx.op.isSpecialFloat.f32(i32 11, float in0_1)
  %4 = zext i1 %3 to i32
  out0_0 = i32 %2
  out0_1 = i32 %4
=== abs_f
  %1 = call float @dx.op.unary.f32(i32 6, float in0_0) !precise
  out0_0 = float %1
=== abs_f3
  %1 = call float @dx.op.unary.f32(i32 6, float in0_0) !precise
  %2 = call float @dx.op.unary.f32(i32 6, float in0_1) !precise
  %3 = call float @dx.op.unary.f32(i32 6, float in0_2) !precise
  out0_0 = float %1
  out0_1 = float %2
  out0_2 = float %3
=== abs_d
  %1 = call double @dx.op.unary.f64(i32 6, double in0_0) !precise
  out0_0 = double %1
=== abs_i
  %1 = sub i32 0, in0_0
  %2 = call i32 @dx.op.binary.i32(i32 37, i32 in0_0, i32 %1)
  out0_0 = i32 %2
=== abs_i3
  %1 = sub i32 0, in0_0
  %2 = call i32 @dx.op.binary.i32(i32 37, i32 in0_0, i32 %1)
  %3 = sub i32 0, in0_1
  %4 = call i32 @dx.op.binary.i32(i32 37, i32 in0_1, i32 %3)
  %5 = sub i32 0, in0_2
  %6 = call i32 @dx.op.binary.i32(i32 37, i32 in0_2, i32 %5)
  out0_0 = i32 %2
  out0_1 = i32 %4
  out0_2 = i32 %6
=== abs_i64
  %1 = sub i64 0, in0_0
  %2 = call i64 @dx.op.binary.i64(i32 37, i64 in0_0, i64 %1)
  out0_0 = i64 %2
=== abs_u
  out0_0 = i32 in0_0
=== abs_u64
  out0_0 = i64 in0_0
=== abs_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = uitofp i1 %1 to float
  %3 = call float @dx.op.unary.f32(i32 6, float %2) !precise
  out0_0 = float %3
=== sign_f
  %1 = fcmp olt float 0.000000e+00, in0_0
  %2 = zext i1 %1 to i32
  %3 = fcmp olt float in0_0, 0.000000e+00
  %4 = zext i1 %3 to i32
  %5 = sub i32 %2, %4
  out0_0 = i32 %5
=== sign_f3
  %1 = fcmp olt float 0.000000e+00, in0_0
  %2 = zext i1 %1 to i32
  %3 = fcmp olt float in0_0, 0.000000e+00
  %4 = zext i1 %3 to i32
  %5 = sub i32 %2, %4
  %6 = fcmp olt float 0.000000e+00, in0_1
  %7 = zext i1 %6 to i32
  %8 = fcmp olt float in0_1, 0.000000e+00
  %9 = zext i1 %8 to i32
  %10 = sub i32 %7, %9
  %11 = fcmp olt float 0.000000e+00, in0_2
  %12 = zext i1 %11 to i32
  %13 = fcmp olt float in0_2, 0.000000e+00
  %14 = zext i1 %13 to i32
  %15 = sub i32 %12, %14
  out0_0 = i32 %5
  out0_1 = i32 %10
  out0_2 = i32 %15
=== sign_d
  %1 = fcmp olt double 0.000000e+00, in0_0
  %2 = zext i1 %1 to i32
  %3 = fcmp olt double in0_0, 0.000000e+00
  %4 = zext i1 %3 to i32
  %5 = sub i32 %2, %4
  out0_0 = i32 %5
=== sign_i
  %1 = icmp slt i32 0, in0_0
  %2 = zext i1 %1 to i32
  %3 = icmp slt i32 in0_0, 0
  %4 = zext i1 %3 to i32
  %5 = sub i32 %2, %4
  out0_0 = i32 %5
=== sign_i64
  %1 = icmp slt i64 0, in0_0
  %2 = zext i1 %1 to i32
  %3 = icmp slt i64 in0_0, 0
  %4 = zext i1 %3 to i32
  %5 = sub i32 %2, %4
  out0_0 = i32 %5
=== sign_u
  %1 = icmp ne i32 0, in0_0
  %2 = zext i1 %1 to i32
  out0_0 = i32 %2
=== sign_u64
  %1 = icmp ne i64 0, in0_0
  %2 = zext i1 %1 to i32
  out0_0 = i32 %2
=== min_f
  %1 = call float @dx.op.binary.f32(i32 36, float in0_0, float in1_0) !precise
  out0_0 = float %1
=== min_i
  %1 = call i32 @dx.op.binary.i32(i32 38, i32 in0_0, i32 in1_0)
  out0_0 = i32 %1
=== min_u
  %1 = call i32 @dx.op.binary.i32(i32 40, i32 in0_0, i32 in1_0)
  out0_0 = i32 %1
=== min_d
  %1 = call double @dx.op.binary.f64(i32 36, double in0_0, double in1_0) !precise
  out0_0 = double %1
=== min_u64
  %1 = call i64 @dx.op.binary.i64(i32 40, i64 in0_0, i64 in1_0)
  out0_0 = i64 %1
=== max_f
  %1 = call float @dx.op.binary.f32(i32 35, float in0_0, float in1_0) !precise
  out0_0 = float %1
=== max_i
  %1 = call i32 @dx.op.binary.i32(i32 37, i32 in0_0, i32 in1_0)
  out0_0 = i32 %1
=== max_u
  %1 = call i32 @dx.op.binary.i32(i32 39, i32 in0_0, i32 in1_0)
  out0_0 = i32 %1
=== max_i64
  %1 = call i64 @dx.op.binary.i64(i32 37, i64 in0_0, i64 in1_0)
  out0_0 = i64 %1
=== min_u_lit
  %1 = call i32 @dx.op.binary.i32(i32 40, i32 in0_0, i32 5)
  out0_0 = i32 %1
=== min_i_lit
  %1 = call i32 @dx.op.binary.i32(i32 38, i32 in0_0, i32 5)
  out0_0 = i32 %1
=== min_u_i
  %1 = call i32 @dx.op.binary.i32(i32 40, i32 in0_0, i32 in1_0)
  out0_0 = i32 %1
=== min_i_u
  %1 = call i32 @dx.op.binary.i32(i32 40, i32 in0_0, i32 in1_0)
  out0_0 = i32 %1
=== min_f_i
  %1 = sitofp i32 in1_0 to float
  %2 = call float @dx.op.binary.f32(i32 36, float in0_0, float %1) !precise
  out0_0 = float %2
=== min_i_f
  %1 = sitofp i32 in0_0 to float
  %2 = call float @dx.op.binary.f32(i32 36, float %1, float in1_0) !precise
  out0_0 = float %2
=== min_f_lit
  %1 = call float @dx.op.binary.f32(i32 36, float in0_0, float 1.000000e+00) !precise
  out0_0 = float %1
=== min_f_flit
  %1 = call float @dx.op.binary.f32(i32 36, float in0_0, float 1.500000e+00) !precise
  out0_0 = float %1
=== min_d_flit
  %1 = call double @dx.op.binary.f64(i32 36, double in0_0, double 1.100000e+00) !precise
  out0_0 = double %1
=== min_f_d
  %1 = fpext float in0_0 to double
  %2 = call double @dx.op.binary.f64(i32 36, double %1, double in1_0) !precise
  out0_0 = double %2
=== min_f3_f
  %1 = call float @dx.op.binary.f32(i32 36, float in0_0, float in1_0) !precise
  %2 = call float @dx.op.binary.f32(i32 36, float in0_1, float in1_0) !precise
  %3 = call float @dx.op.binary.f32(i32 36, float in0_2, float in1_0) !precise
  out0_0 = float %1
  out0_1 = float %2
  out0_2 = float %3
=== min_u3_u
  %1 = call i32 @dx.op.binary.i32(i32 40, i32 in0_0, i32 in1_0)
  %2 = call i32 @dx.op.binary.i32(i32 40, i32 in0_1, i32 in1_0)
  %3 = call i32 @dx.op.binary.i32(i32 40, i32 in0_2, i32 in1_0)
  out0_0 = i32 %1
  out0_1 = i32 %2
  out0_2 = i32 %3
=== min_u3_u2
  %1 = call i32 @dx.op.binary.i32(i32 40, i32 in0_0, i32 in1_0)
  %2 = call i32 @dx.op.binary.i32(i32 40, i32 in0_1, i32 in1_1)
  out0_0 = i32 %1
  out0_1 = i32 %2
=== min_i64_i
  %1 = sext i32 in1_0 to i64
  %2 = call i64 @dx.op.binary.i64(i32 38, i64 in0_0, i64 %1)
  out0_0 = i64 %2
=== min_u64_u
  %1 = zext i32 in1_0 to i64
  %2 = call i64 @dx.op.binary.i64(i32 40, i64 in0_0, i64 %1)
  out0_0 = i64 %2
=== max_i_u64
  %1 = sext i32 in0_0 to i64
  %2 = call i64 @dx.op.binary.i64(i32 39, i64 %1, i64 in1_0)
  out0_0 = i64 %2
=== min_b_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = uitofp i1 %1 to float
  %3 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_0)
  %4 = uitofp i1 %3 to float
  %5 = call float @dx.op.binary.f32(i32 36, float %2, float %4) !precise
  out0_0 = float %5
=== min_b_i
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = zext i1 %1 to i32
  %3 = call i32 @dx.op.binary.i32(i32 38, i32 %2, i32 in1_0)
  out0_0 = i32 %3
=== min_b_u
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = zext i1 %1 to i32
  %3 = call i32 @dx.op.binary.i32(i32 40, i32 %2, i32 in1_0)
  out0_0 = i32 %3
=== max_f_lit0
  %1 = call float @dx.op.binary.f32(i32 35, float in0_0, float 0.000000e+00) !precise
  out0_0 = float %1
=== clamp_f
  %1 = call float @dx.op.binary.f32(i32 35, float in0_0, float in1_0) !precise
  %2 = call float @dx.op.binary.f32(i32 36, float %1, float in2_0) !precise
  out0_0 = float %2
=== clamp_u
  %1 = call i32 @dx.op.binary.i32(i32 39, i32 in0_0, i32 in1_0)
  %2 = call i32 @dx.op.binary.i32(i32 40, i32 %1, i32 in2_0)
  out0_0 = i32 %2
=== clamp_i
  %1 = call i32 @dx.op.binary.i32(i32 37, i32 in0_0, i32 0)
  %2 = call i32 @dx.op.binary.i32(i32 38, i32 %1, i32 255)
  out0_0 = i32 %2
=== clamp_f_lit
  %1 = call float @dx.op.binary.f32(i32 35, float in0_0, float 0.000000e+00) !precise
  %2 = call float @dx.op.binary.f32(i32 36, float %1, float 1.000000e+00) !precise
  out0_0 = float %2
=== clamp_u_lit
  %1 = call i32 @dx.op.binary.i32(i32 39, i32 in0_0, i32 1)
  %2 = call i32 @dx.op.binary.i32(i32 40, i32 %1, i32 7)
  out0_0 = i32 %2
=== clamp_i_u
  %1 = call i32 @dx.op.binary.i32(i32 39, i32 in0_0, i32 0)
  %2 = call i32 @dx.op.binary.i32(i32 40, i32 %1, i32 in1_0)
  out0_0 = i32 %2
=== clamp_d_i32
  %1 = call double @dx.op.binary.f64(i32 35, double in0_0, double 0xC1E0000000000000) !precise
  %2 = call double @dx.op.binary.f64(i32 36, double %1, double 0x41DFFFFFFFC00000) !precise
  out0_0 = double %2
=== clamp_d_u32
  %1 = call double @dx.op.binary.f64(i32 35, double in0_0, double 0.000000e+00) !precise
  %2 = call double @dx.op.binary.f64(i32 36, double %1, double 0x41EFFFFFFFE00000) !precise
  out0_0 = double %2
=== clamp_f3
  %1 = call float @dx.op.binary.f32(i32 35, float in0_0, float in1_0) !precise
  %2 = call float @dx.op.binary.f32(i32 36, float %1, float in2_0) !precise
  %3 = call float @dx.op.binary.f32(i32 35, float in0_1, float in1_0) !precise
  %4 = call float @dx.op.binary.f32(i32 36, float %3, float in2_1) !precise
  %5 = call float @dx.op.binary.f32(i32 35, float in0_2, float in1_0) !precise
  %6 = call float @dx.op.binary.f32(i32 36, float %5, float in2_2) !precise
  out0_0 = float %2
  out0_1 = float %4
  out0_2 = float %6
=== clamp_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = uitofp i1 %1 to float
  %3 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_0)
  %4 = uitofp i1 %3 to float
  %5 = call float @dx.op.binary.f32(i32 35, float %2, float %4) !precise
  %6 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in2_0)
  %7 = uitofp i1 %6 to float
  %8 = call float @dx.op.binary.f32(i32 36, float %5, float %7) !precise
  out0_0 = float %8
=== saturate_f
  %1 = call float @dx.op.unary.f32(i32 7, float in0_0) !precise
  out0_0 = float %1
=== saturate_f4
  %1 = call float @dx.op.unary.f32(i32 7, float in0_0) !precise
  %2 = call float @dx.op.unary.f32(i32 7, float in0_1) !precise
  %3 = call float @dx.op.unary.f32(i32 7, float in0_2) !precise
  %4 = call float @dx.op.unary.f32(i32 7, float in0_3) !precise
  out0_0 = float %1
  out0_1 = float %2
  out0_2 = float %3
  out0_3 = float %4
=== saturate_d
  %1 = call double @dx.op.unary.f64(i32 7, double in0_0) !precise
  out0_0 = double %1
=== saturate_i
  %1 = sitofp i32 in0_0 to float
  %2 = call float @dx.op.unary.f32(i32 7, float %1) !precise
  out0_0 = float %2
=== mad_f
  %1 = call float @dx.op.tertiary.f32(i32 46, float in0_0, float in1_0, float in2_0) !precise
  out0_0 = float %1
=== mad_f3
  %1 = call float @dx.op.tertiary.f32(i32 46, float in0_0, float in1_0, float in2_0) !precise
  %2 = call float @dx.op.tertiary.f32(i32 46, float in0_1, float in1_1, float in2_1) !precise
  %3 = call float @dx.op.tertiary.f32(i32 46, float in0_2, float in1_2, float in2_2) !precise
  out0_0 = float %1
  out0_1 = float %2
  out0_2 = float %3
=== mad_d
  %1 = call double @dx.op.tertiary.f64(i32 46, double in0_0, double in1_0, double in2_0) !precise
  out0_0 = double %1
=== mad_i
  %1 = call i32 @dx.op.tertiary.i32(i32 48, i32 in0_0, i32 in1_0, i32 in2_0)
  out0_0 = i32 %1
=== mad_u
  %1 = call i32 @dx.op.tertiary.i32(i32 49, i32 in0_0, i32 in1_0, i32 in2_0)
  out0_0 = i32 %1
=== mad_i64
  %1 = call i64 @dx.op.tertiary.i64(i32 48, i64 in0_0, i64 in1_0, i64 in2_0)
  out0_0 = i64 %1
=== mad_u64
  %1 = call i64 @dx.op.tertiary.i64(i32 49, i64 in0_0, i64 in1_0, i64 in2_0)
  out0_0 = i64 %1
=== mad_f_lit
  %1 = call float @dx.op.tertiary.f32(i32 46, float in0_0, float 2.000000e+00, float in1_0) !precise
  out0_0 = float %1
=== mad_i_u
  %1 = call i32 @dx.op.tertiary.i32(i32 49, i32 in0_0, i32 in1_0, i32 in2_0)
  out0_0 = i32 %1
=== mad_u_lit
  %1 = call i32 @dx.op.tertiary.i32(i32 49, i32 in0_0, i32 3, i32 in1_0)
  out0_0 = i32 %1
=== mad_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = uitofp i1 %1 to float
  %3 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_0)
  %4 = uitofp i1 %3 to float
  %5 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in2_0)
  %6 = uitofp i1 %5 to float
  %7 = call float @dx.op.tertiary.f32(i32 46, float %2, float %4, float %6) !precise
  out0_0 = float %7
=== fma_d
  %1 = call double @dx.op.tertiary.f64(i32 47, double in0_0, double in1_0, double in2_0) !precise
  out0_0 = double %1
=== fma_flit
  %1 = call double @dx.op.tertiary.f64(i32 47, double in0_0, double 2.000000e+00, double 1.000000e+00) !precise
  out0_0 = double %1
=== rcp_f
  %1 = fdiv float 1.000000e+00, in0_0
  out0_0 = float %1
=== rcp_d
  %1 = fdiv double 1.000000e+00, in0_0
  out0_0 = double %1
=== rcp_i
  %1 = sitofp i32 in0_0 to float
  %2 = fdiv float 1.000000e+00, %1
  out0_0 = float %2
=== rcp_f3
  %1 = fdiv float 1.000000e+00, in0_0
  %2 = fdiv float 1.000000e+00, in0_1
  %3 = fdiv float 1.000000e+00, in0_2
  out0_0 = float %1
  out0_1 = float %2
  out0_2 = float %3
=== sin_f
  %1 = call float @dx.op.unary.f32(i32 13, float in0_0) !precise
  out0_0 = float %1
=== sin_f3
  %1 = call float @dx.op.unary.f32(i32 13, float in0_0) !precise
  %2 = call float @dx.op.unary.f32(i32 13, float in0_1) !precise
  %3 = call float @dx.op.unary.f32(i32 13, float in0_2) !precise
  out0_0 = float %1
  out0_1 = float %2
  out0_2 = float %3
=== sin_i
  %1 = sitofp i32 in0_0 to float
  %2 = call float @dx.op.unary.f32(i32 13, float %1) !precise
  out0_0 = float %2
=== sin_d
  %1 = fptrunc double in0_0 to float
  %2 = call float @dx.op.unary.f32(i32 13, float %1) !precise
  out0_0 = float %2
=== cos_f
  %1 = call float @dx.op.unary.f32(i32 12, float in0_0) !precise
  out0_0 = float %1
=== tan_f
  %1 = call float @dx.op.unary.f32(i32 14, float in0_0) !precise
  out0_0 = float %1
=== asin_f
  %1 = call float @dx.op.unary.f32(i32 16, float in0_0) !precise
  out0_0 = float %1
=== acos_f
  %1 = call float @dx.op.unary.f32(i32 15, float in0_0) !precise
  out0_0 = float %1
=== atan_f
  %1 = call float @dx.op.unary.f32(i32 17, float in0_0) !precise
  out0_0 = float %1
=== sinh_f
  %1 = call float @dx.op.unary.f32(i32 19, float in0_0) !precise
  out0_0 = float %1
=== cosh_f
  %1 = call float @dx.op.unary.f32(i32 18, float in0_0) !precise
  out0_0 = float %1
=== tanh_f
  %1 = call float @dx.op.unary.f32(i32 20, float in0_0) !precise
  out0_0 = float %1
=== sqrt_f
  %1 = call float @dx.op.unary.f32(i32 24, float in0_0) !precise
  out0_0 = float %1
=== sqrt_d
  %1 = fptrunc double in0_0 to float
  %2 = call float @dx.op.unary.f32(i32 24, float %1) !precise
  out0_0 = float %2
=== rsqrt_f
  %1 = call float @dx.op.unary.f32(i32 25, float in0_0) !precise
  out0_0 = float %1
=== exp_f
  %1 = fmul float 0x3FF7154760000000, in0_0
  %2 = call float @dx.op.unary.f32(i32 21, float %1) !precise
  out0_0 = float %2
=== exp_f2
  %1 = fmul float 0x3FF7154760000000, in0_0
  %2 = call float @dx.op.unary.f32(i32 21, float %1) !precise
  %3 = fmul float 0x3FF7154760000000, in0_1
  %4 = call float @dx.op.unary.f32(i32 21, float %3) !precise
  out0_0 = float %2
  out0_1 = float %4
=== exp2_f
  %1 = call float @dx.op.unary.f32(i32 21, float in0_0) !precise
  out0_0 = float %1
=== log_f
  %1 = call float @dx.op.unary.f32(i32 23, float in0_0) !precise
  %2 = fmul float 0x3FE62E4300000000, %1
  out0_0 = float %2
=== log2_f
  %1 = call float @dx.op.unary.f32(i32 23, float in0_0) !precise
  out0_0 = float %1
=== log10_f
  %1 = call float @dx.op.unary.f32(i32 23, float in0_0) !precise
  %2 = fmul float 0x3FD3441360000000, %1
  out0_0 = float %2
=== degrees_f
  %1 = fmul float 0x404CA5DC20000000, in0_0
  out0_0 = float %1
=== radians_f
  %1 = fmul float 0x3F91DF46A0000000, in0_0
  out0_0 = float %1
=== floor_f
  %1 = call float @dx.op.unary.f32(i32 27, float in0_0) !precise
  out0_0 = float %1
=== floor_d
  %1 = fptrunc double in0_0 to float
  %2 = call float @dx.op.unary.f32(i32 27, float %1) !precise
  out0_0 = float %2
=== ceil_f
  %1 = call float @dx.op.unary.f32(i32 28, float in0_0) !precise
  out0_0 = float %1
=== round_f
  %1 = call float @dx.op.unary.f32(i32 26, float in0_0) !precise
  out0_0 = float %1
=== trunc_f
  %1 = call float @dx.op.unary.f32(i32 29, float in0_0) !precise
  out0_0 = float %1
=== frac_f
  %1 = call float @dx.op.unary.f32(i32 22, float in0_0) !precise
  out0_0 = float %1
=== pow_f
  %1 = call float @dx.op.unary.f32(i32 23, float in0_0) !precise
  %2 = fmul float %1, in1_0
  %3 = call float @dx.op.unary.f32(i32 21, float %2) !precise
  out0_0 = float %3
=== pow_f3_f
  %1 = call float @dx.op.unary.f32(i32 23, float in0_0) !precise
  %2 = fmul float %1, in1_0
  %3 = call float @dx.op.unary.f32(i32 21, float %2) !precise
  %4 = call float @dx.op.unary.f32(i32 23, float in0_1) !precise
  %5 = fmul float %4, in1_0
  %6 = call float @dx.op.unary.f32(i32 21, float %5) !precise
  %7 = call float @dx.op.unary.f32(i32 23, float in0_2) !precise
  %8 = fmul float %7, in1_0
  %9 = call float @dx.op.unary.f32(i32 21, float %8) !precise
  out0_0 = float %3
  out0_1 = float %6
  out0_2 = float %9
=== pow_f_2
  %1 = fmul float in0_0, in0_0
  out0_0 = float %1
=== pow_f_2i
  %1 = fmul float in0_0, in0_0
  out0_0 = float %1
=== pow_f3_2
  %1 = fmul float in0_0, in0_0
  %2 = fmul float in0_1, in0_1
  %3 = fmul float in0_2, in0_2
  out0_0 = float %1
  out0_1 = float %2
  out0_2 = float %3
=== pow_f_3
  %1 = call float @dx.op.unary.f32(i32 23, float in0_0) !precise
  %2 = fmul float 3.000000e+00, %1
  %3 = call float @dx.op.unary.f32(i32 21, float %2) !precise
  out0_0 = float %3
=== pow_f_m2
  %1 = call float @dx.op.unary.f32(i32 23, float in0_0) !precise
  %2 = fmul float -2.000000e+00, %1
  %3 = call float @dx.op.unary.f32(i32 21, float %2) !precise
  out0_0 = float %3
=== pow_f_half
  %1 = call float @dx.op.unary.f32(i32 23, float in0_0) !precise
  %2 = fmul float 5.000000e-01, %1
  %3 = call float @dx.op.unary.f32(i32 21, float %2) !precise
  out0_0 = float %3
=== atan2_f
  %1 = fcmp oeq float 0.000000e+00, in1_0
  %2 = fcmp ole float 0.000000e+00, in0_0
  %3 = and i1 %1, %2
  %4 = fcmp olt float in0_0, 0.000000e+00
  %5 = and i1 %1, %4
  %6 = fcmp olt float in1_0, 0.000000e+00
  %7 = and i1 %4, %6
  %8 = fdiv float in0_0, in1_0
  %9 = call float @dx.op.unary.f32(i32 17, float %8) !precise
  %10 = fsub float %9, 0x400921FB60000000
  %11 = and i1 %2, %6
  %12 = fadd float 0x400921FB60000000, %9
  %13 = select i1 %11, float %12, float %9
  %14 = select i1 %7, float %10, float %13
  %15 = select i1 %5, float 0xBFF921FB60000000, float %14
  %16 = select i1 %3, float 0x3FF921FB60000000, float %15
  out0_0 = float %16
=== atan2_f2
  %1 = fcmp oeq float 0.000000e+00, in1_0
  %2 = fcmp ole float 0.000000e+00, in0_0
  %3 = and i1 %1, %2
  %4 = fcmp olt float in0_0, 0.000000e+00
  %5 = and i1 %1, %4
  %6 = fcmp olt float in1_0, 0.000000e+00
  %7 = and i1 %4, %6
  %8 = fdiv float in0_0, in1_0
  %9 = call float @dx.op.unary.f32(i32 17, float %8) !precise
  %10 = fsub float %9, 0x400921FB60000000
  %11 = and i1 %2, %6
  %12 = fadd float 0x400921FB60000000, %9
  %13 = select i1 %11, float %12, float %9
  %14 = select i1 %7, float %10, float %13
  %15 = select i1 %5, float 0xBFF921FB60000000, float %14
  %16 = select i1 %3, float 0x3FF921FB60000000, float %15
  %17 = fcmp oeq float 0.000000e+00, in1_1
  %18 = fcmp ole float 0.000000e+00, in0_1
  %19 = and i1 %17, %18
  %20 = fcmp olt float in0_1, 0.000000e+00
  %21 = and i1 %17, %20
  %22 = fcmp olt float in1_1, 0.000000e+00
  %23 = and i1 %20, %22
  %24 = fdiv float in0_1, in1_1
  %25 = call float @dx.op.unary.f32(i32 17, float %24) !precise
  %26 = fsub float %25, 0x400921FB60000000
  %27 = and i1 %18, %22
  %28 = fadd float 0x400921FB60000000, %25
  %29 = select i1 %27, float %28, float %25
  %30 = select i1 %23, float %26, float %29
  %31 = select i1 %21, float 0xBFF921FB60000000, float %30
  %32 = select i1 %19, float 0x3FF921FB60000000, float %31
  out0_0 = float %16
  out0_1 = float %32
=== sincos_f
  %1 = call float @dx.op.unary.f32(i32 13, float in0_0) !precise
  %2 = call float @dx.op.unary.f32(i32 12, float in0_0) !precise
  out1_0 = float %1
  out2_0 = float %2
=== fmod_f
  %1 = fdiv float in0_0, in1_0
  %2 = fsub float -0.000000e+00, %1
  %3 = fcmp ole float %2, %1
  %4 = call float @dx.op.unary.f32(i32 6, float %1) !precise
  %5 = call float @dx.op.unary.f32(i32 22, float %4) !precise
  %6 = fsub float -0.000000e+00, %5
  %7 = select i1 %3, float %5, float %6
  %8 = fmul float in1_0, %7
  out0_0 = float %8
=== fmod_f3_f
  %1 = fdiv float in0_0, in1_0
  %2 = fsub float -0.000000e+00, %1
  %3 = fcmp ole float %2, %1
  %4 = call float @dx.op.unary.f32(i32 6, float %1) !precise
  %5 = call float @dx.op.unary.f32(i32 22, float %4) !precise
  %6 = fsub float -0.000000e+00, %5
  %7 = select i1 %3, float %5, float %6
  %8 = fmul float in1_0, %7
  %9 = fdiv float in0_1, in1_0
  %10 = fsub float -0.000000e+00, %9
  %11 = fcmp ole float %10, %9
  %12 = call float @dx.op.unary.f32(i32 6, float %9) !precise
  %13 = call float @dx.op.unary.f32(i32 22, float %12) !precise
  %14 = fsub float -0.000000e+00, %13
  %15 = select i1 %11, float %13, float %14
  %16 = fmul float in1_0, %15
  %17 = fdiv float in0_2, in1_0
  %18 = fsub float -0.000000e+00, %17
  %19 = fcmp ole float %18, %17
  %20 = call float @dx.op.unary.f32(i32 6, float %17) !precise
  %21 = call float @dx.op.unary.f32(i32 22, float %20) !precise
  %22 = fsub float -0.000000e+00, %21
  %23 = select i1 %19, float %21, float %22
  %24 = fmul float in1_0, %23
  out0_0 = float %8
  out0_1 = float %16
  out0_2 = float %24
=== modf_f
  %1 = call float @dx.op.unary.f32(i32 29, float in0_0) !precise
  %2 = fsub float in0_0, %1
  out0_0 = float %2
  out1_0 = float %1
=== modf_f3
  %1 = call float @dx.op.unary.f32(i32 29, float in0_0) !precise
  %2 = fsub float in0_0, %1
  %3 = call float @dx.op.unary.f32(i32 29, float in0_1) !precise
  %4 = fsub float in0_1, %3
  %5 = call float @dx.op.unary.f32(i32 29, float in0_2) !precise
  %6 = fsub float in0_2, %5
  out0_0 = float %2
  out0_1 = float %4
  out0_2 = float %6
  out1_0 = float %1
  out1_1 = float %3
  out1_2 = float %5
=== modf_d
  %1 = fptrunc double in0_0 to float
  %2 = call float @dx.op.unary.f32(i32 29, float %1) !precise
  %3 = fsub float %1, %2
  out0_0 = float %3
  out1_0 = float %2
=== frexp_f
  %1 = bitcast float in0_0 to i32
  %2 = and i32 8388607, %1
  %3 = or i32 1056964608, %2
  %4 = fcmp une float 0.000000e+00, in0_0
  %5 = sext i1 %4 to i32
  %6 = and i32 %3, %5
  %7 = bitcast i32 %6 to float
  %8 = and i32 2139095040, %1
  %9 = add i32 -1056964608, %8
  %10 = and i32 %9, %5
  %11 = ashr i32 %10, 23
  %12 = sitofp i32 %11 to float
  out0_0 = float %7
  out1_0 = float %12
=== ldexp_f
  %1 = call float @dx.op.unary.f32(i32 21, float in1_0) !precise
  %2 = fmul float %1, in0_0
  out0_0 = float %2
=== ldexp_f_i
  %1 = sitofp i32 in1_0 to float
  %2 = call float @dx.op.unary.f32(i32 21, float %1) !precise
  %3 = fmul float %2, in0_0
  out0_0 = float %3
=== lerp_f
  %1 = fsub float in1_0, in0_0
  %2 = fmul float %1, in2_0
  %3 = fadd float %2, in0_0
  out0_0 = float %3
=== lerp_f3_f
  %1 = fsub float in1_0, in0_0
  %2 = fmul float %1, in2_0
  %3 = fadd float %2, in0_0
  %4 = fsub float in1_1, in0_1
  %5 = fmul float %4, in2_0
  %6 = fadd float %5, in0_1
  %7 = fsub float in1_2, in0_2
  %8 = fmul float %7, in2_0
  %9 = fadd float %8, in0_2
  out0_0 = float %3
  out0_1 = float %6
  out0_2 = float %9
=== lerp_i
  %1 = sitofp i32 in1_0 to float
  %2 = sitofp i32 in0_0 to float
  %3 = fsub float %1, %2
  %4 = sitofp i32 in2_0 to float
  %5 = fmul float %3, %4
  %6 = fadd float %5, %2
  out0_0 = float %6
=== lerp_d
  %1 = fptrunc double in2_0 to float
  %2 = fptrunc double in1_0 to float
  %3 = fptrunc double in0_0 to float
  %4 = fsub float %2, %3
  %5 = fmul float %1, %4
  %6 = fadd float %5, %3
  out0_0 = float %6
=== step_f
  %1 = fcmp olt float in1_0, in0_0
  %2 = select i1 %1, float 0.000000e+00, float 1.000000e+00
  out0_0 = float %2
=== step_lit
  %1 = fcmp olt float in0_0, 5.000000e-01
  %2 = select i1 %1, float 0.000000e+00, float 1.000000e+00
  out0_0 = float %2
=== smoothstep_f
  %1 = fsub float in2_0, in0_0
  %2 = fsub float in1_0, in0_0
  %3 = fdiv float %1, %2
  %4 = call float @dx.op.unary.f32(i32 7, float %3) !precise
  %5 = fmul float 2.000000e+00, %4
  %6 = fsub float 3.000000e+00, %5
  %7 = fmul float %4, %6
  %8 = fmul float %4, %7
  out0_0 = float %8
=== ddx_f
  %1 = call float @dx.op.unary.f32(i32 83, float in0_0) !precise
  out0_0 = float %1
=== ddx_f2
  %1 = call float @dx.op.unary.f32(i32 83, float in0_0) !precise
  %2 = call float @dx.op.unary.f32(i32 83, float in0_1) !precise
  out0_0 = float %1
  out0_1 = float %2
=== ddy_f
  %1 = call float @dx.op.unary.f32(i32 84, float in0_0) !precise
  out0_0 = float %1
=== ddx_coarse_f
  %1 = call float @dx.op.unary.f32(i32 83, float in0_0) !precise
  out0_0 = float %1
=== ddy_coarse_f
  %1 = call float @dx.op.unary.f32(i32 84, float in0_0) !precise
  out0_0 = float %1
=== ddx_fine_f
  %1 = call float @dx.op.unary.f32(i32 85, float in0_0) !precise
  out0_0 = float %1
=== ddy_fine_f
  %1 = call float @dx.op.unary.f32(i32 86, float in0_0) !precise
  out0_0 = float %1
=== ddx_i
  %1 = sitofp i32 in0_0 to float
  %2 = call float @dx.op.unary.f32(i32 83, float %1) !precise
  out0_0 = float %2
=== fwidth_f
  %1 = call float @dx.op.unary.f32(i32 83, float in0_0) !precise
  %2 = call float @dx.op.unary.f32(i32 6, float %1) !precise
  %3 = call float @dx.op.unary.f32(i32 84, float in0_0) !precise
  %4 = call float @dx.op.unary.f32(i32 6, float %3) !precise
  %5 = fadd float %2, %4
  out0_0 = float %5
=== dot_f2
  %1 = call float @dx.op.dot2.f32(i32 54, float in0_0, float in0_1, float in1_0, float in1_1) !precise
  out0_0 = float %1
=== dot_f3
  %1 = call float @dx.op.dot3.f32(i32 55, float in0_0, float in0_1, float in0_2, float in1_0, float in1_1, float in1_2) !precise
  out0_0 = float %1
=== dot_f4
  %1 = call float @dx.op.dot4.f32(i32 56, float in0_0, float in0_1, float in0_2, float in0_3, float in1_0, float in1_1, float in1_2, float in1_3) !precise
  out0_0 = float %1
=== dot_f
  %1 = fmul float in0_0, in1_0
  out0_0 = float %1
=== dot_f3_f
  %1 = call float @dx.op.dot3.f32(i32 55, float in0_0, float in0_1, float in0_2, float in1_0, float in1_0, float in1_0) !precise
  out0_0 = float %1
=== dot_i3
  %1 = mul i32 in0_0, in1_0
  %2 = call i32 @dx.op.tertiary.i32(i32 48, i32 in0_1, i32 in1_1, i32 %1)
  %3 = call i32 @dx.op.tertiary.i32(i32 48, i32 in0_2, i32 in1_2, i32 %2)
  out0_0 = i32 %3
=== dot_u2
  %1 = mul i32 in0_0, in1_0
  %2 = call i32 @dx.op.tertiary.i32(i32 49, i32 in0_1, i32 in1_1, i32 %1)
  out0_0 = i32 %2
=== dot_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = uitofp i1 %1 to float
  %3 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_1)
  %4 = uitofp i1 %3 to float
  %5 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_0)
  %6 = uitofp i1 %5 to float
  %7 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in1_1)
  %8 = uitofp i1 %7 to float
  %9 = call float @dx.op.dot2.f32(i32 54, float %2, float %4, float %6, float %8) !precise
  out0_0 = float %9
=== cross_f
  %1 = fmul float in0_1, in1_2
  %2 = fmul float in0_2, in1_1
  %3 = fsub float %1, %2
  %4 = fmul float in0_2, in1_0
  %5 = fmul float in0_0, in1_2
  %6 = fsub float %4, %5
  %7 = fmul float in0_0, in1_1
  %8 = fmul float in0_1, in1_0
  %9 = fsub float %7, %8
  out0_0 = float %3
  out0_1 = float %6
  out0_2 = float %9
=== length_f3
  %1 = fmul float in0_0, in0_0
  %2 = fmul float in0_1, in0_1
  %3 = fadd float %1, %2
  %4 = fmul float in0_2, in0_2
  %5 = fadd float %3, %4
  %6 = call float @dx.op.unary.f32(i32 24, float %5) !precise
  out0_0 = float %6
=== length_f2
  %1 = fmul float in0_0, in0_0
  %2 = fmul float in0_1, in0_1
  %3 = fadd float %1, %2
  %4 = call float @dx.op.unary.f32(i32 24, float %3) !precise
  out0_0 = float %4
=== length_f4
  %1 = fmul float in0_0, in0_0
  %2 = fmul float in0_1, in0_1
  %3 = fadd float %1, %2
  %4 = fmul float in0_2, in0_2
  %5 = fadd float %3, %4
  %6 = fmul float in0_3, in0_3
  %7 = fadd float %5, %6
  %8 = call float @dx.op.unary.f32(i32 24, float %7) !precise
  out0_0 = float %8
=== length_f
  %1 = call float @dx.op.unary.f32(i32 6, float in0_0) !precise
  out0_0 = float %1
=== distance_f3
  %1 = fsub float in0_0, in1_0
  %2 = fmul float %1, %1
  %3 = fsub float in0_1, in1_1
  %4 = fmul float %3, %3
  %5 = fadd float %2, %4
  %6 = fsub float in0_2, in1_2
  %7 = fmul float %6, %6
  %8 = fadd float %5, %7
  %9 = call float @dx.op.unary.f32(i32 24, float %8) !precise
  out0_0 = float %9
=== distance_f
  %1 = fsub float in0_0, in1_0
  %2 = call float @dx.op.unary.f32(i32 6, float %1) !precise
  out0_0 = float %2
=== normalize_f3
  %1 = call float @dx.op.dot3.f32(i32 55, float in0_0, float in0_1, float in0_2, float in0_0, float in0_1, float in0_2) !precise
  %2 = call float @dx.op.unary.f32(i32 25, float %1) !precise
  %3 = fmul float %2, in0_0
  %4 = fmul float %2, in0_1
  %5 = fmul float %2, in0_2
  out0_0 = float %3
  out0_1 = float %4
  out0_2 = float %5
=== normalize_f2
  %1 = call float @dx.op.dot2.f32(i32 54, float in0_0, float in0_1, float in0_0, float in0_1) !precise
  %2 = call float @dx.op.unary.f32(i32 25, float %1) !precise
  %3 = fmul float %2, in0_0
  %4 = fmul float %2, in0_1
  out0_0 = float %3
  out0_1 = float %4
=== normalize_f
  %1 = fmul float in0_0, in0_0
  %2 = call float @dx.op.unary.f32(i32 25, float %1) !precise
  %3 = fmul float %2, in0_0
  out0_0 = float %3
=== reflect_f3
  %1 = call float @dx.op.dot3.f32(i32 55, float in0_0, float in0_1, float in0_2, float in1_0, float in1_1, float in1_2) !precise
  %2 = fmul float 2.000000e+00, %1
  %3 = fmul float %2, in1_0
  %4 = fsub float in0_0, %3
  %5 = fmul float %2, in1_1
  %6 = fsub float in0_1, %5
  %7 = fmul float %2, in1_2
  %8 = fsub float in0_2, %7
  out0_0 = float %4
  out0_1 = float %6
  out0_2 = float %8
=== refract_f3
  %1 = fmul float in2_0, in2_0
  %2 = call float @dx.op.dot3.f32(i32 55, float in0_0, float in0_1, float in0_2, float in1_0, float in1_1, float in1_2) !precise
  %3 = fmul float %2, %2
  %4 = fsub float 1.000000e+00, %3
  %5 = fmul float %1, %4
  %6 = fsub float 1.000000e+00, %5
  %7 = fcmp ole float 0.000000e+00, %6
  %8 = fmul float in0_0, in2_0
  %9 = call float @dx.op.unary.f32(i32 24, float %6) !precise
  %10 = fmul float %2, in2_0
  %11 = fadd float %9, %10
  %12 = fmul float %11, in1_0
  %13 = fsub float %8, %12
  %14 = select i1 %7, float %13, float 0.000000e+00
  %15 = fmul float in0_1, in2_0
  %16 = fmul float %11, in1_1
  %17 = fsub float %15, %16
  %18 = select i1 %7, float %17, float 0.000000e+00
  %19 = fmul float in0_2, in2_0
  %20 = fmul float %11, in1_2
  %21 = fsub float %19, %20
  %22 = select i1 %7, float %21, float 0.000000e+00
  out0_0 = float %14
  out0_1 = float %18
  out0_2 = float %22
=== faceforward_f3
  %1 = call float @dx.op.dot3.f32(i32 55, float in1_0, float in1_1, float in1_2, float in2_0, float in2_1, float in2_2) !precise
  %2 = fcmp olt float %1, 0.000000e+00
  %3 = fsub float -0.000000e+00, in0_0
  %4 = select i1 %2, float in0_0, float %3
  %5 = fsub float -0.000000e+00, in0_1
  %6 = select i1 %2, float in0_1, float %5
  %7 = fsub float -0.000000e+00, in0_2
  %8 = select i1 %2, float in0_2, float %7
  out0_0 = float %4
  out0_1 = float %6
  out0_2 = float %8
=== mul_ff
  %1 = fmul float in0_0, in1_0
  out0_0 = float %1
=== mul_vv
  %1 = call float @dx.op.dot3.f32(i32 55, float in0_0, float in0_1, float in0_2, float in1_0, float in1_1, float in1_2) !precise
  out0_0 = float %1
=== mul_sv
  %1 = fmul float in0_0, in1_0
  %2 = fmul float in0_0, in1_1
  %3 = fmul float in0_0, in1_2
  out0_0 = float %1
  out0_1 = float %2
  out0_2 = float %3
=== mul_vs
  %1 = fmul float in0_0, in1_0
  %2 = fmul float in0_1, in1_0
  %3 = fmul float in0_2, in1_0
  out0_0 = float %1
  out0_1 = float %2
  out0_2 = float %3
=== mul_iu
  %1 = mul i32 in0_0, in1_0
  out0_0 = i32 %1
=== mul_uu3
  %1 = mul i32 in0_0, in1_0
  %2 = call i32 @dx.op.tertiary.i32(i32 49, i32 in0_1, i32 in1_1, i32 %1)
  %3 = call i32 @dx.op.tertiary.i32(i32 49, i32 in0_2, i32 in1_2, i32 %2)
  out0_0 = i32 %3
=== mul_mv
  %1 = fmul float in0_0, in1_0
  %2 = call float @dx.op.tertiary.f32(i32 46, float in0_1, float in1_1, float %1) !precise
  %3 = fmul float in0_2, in1_0
  %4 = call float @dx.op.tertiary.f32(i32 46, float in0_3, float in1_1, float %3) !precise
  out0_0 = float %2
  out0_1 = float %4
=== mul_vm
  %1 = fmul float in0_0, in1_0
  %2 = call float @dx.op.tertiary.f32(i32 46, float in0_1, float in1_2, float %1) !precise
  %3 = fmul float in0_0, in1_1
  %4 = call float @dx.op.tertiary.f32(i32 46, float in0_1, float in1_3, float %3) !precise
  out0_0 = float %2
  out0_1 = float %4
=== mul_mm
  %1 = fmul float in0_0, in1_0
  %2 = call float @dx.op.tertiary.f32(i32 46, float in0_1, float in1_2, float %1) !precise
  %3 = fmul float in0_0, in1_1
  %4 = call float @dx.op.tertiary.f32(i32 46, float in0_1, float in1_3, float %3) !precise
  %5 = fmul float in0_2, in1_0
  %6 = call float @dx.op.tertiary.f32(i32 46, float in0_3, float in1_2, float %5) !precise
  %7 = fmul float in0_2, in1_1
  %8 = call float @dx.op.tertiary.f32(i32 46, float in0_3, float in1_3, float %7) !precise
  out0_0 = float %2
  out0_1 = float %4
  out0_2 = float %6
  out0_3 = float %8
=== mul_sm
  %1 = fmul float in0_0, in1_0
  %2 = fmul float in0_0, in1_1
  %3 = fmul float in0_0, in1_2
  %4 = fmul float in0_0, in1_3
  out0_0 = float %1
  out0_1 = float %2
  out0_2 = float %3
  out0_3 = float %4
=== mul_ms
  %1 = fmul float in0_0, in1_0
  %2 = fmul float in0_1, in1_0
  %3 = fmul float in0_2, in1_0
  %4 = fmul float in0_3, in1_0
  out0_0 = float %1
  out0_1 = float %2
  out0_2 = float %3
  out0_3 = float %4
=== mul_umv
  %1 = mul i32 in0_0, in1_0
  %2 = call i32 @dx.op.tertiary.i32(i32 49, i32 in0_1, i32 in1_1, i32 %1)
  %3 = mul i32 in0_2, in1_0
  %4 = call i32 @dx.op.tertiary.i32(i32 49, i32 in0_3, i32 in1_1, i32 %3)
  out0_0 = i32 %2
  out0_1 = i32 %4
=== dot4add_u
  %1 = call i32 @dx.op.dot4AddPacked.i32(i32 164, i32 in2_0, i32 in0_0, i32 in1_0)
  out0_0 = i32 %1
=== dot4add_i
  %1 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 in2_0, i32 in0_0, i32 in1_0)
  out0_0 = i32 %1
=== countbits_u
  %1 = call i32 @dx.op.unaryBits.i32(i32 31, i32 in0_0)
  out0_0 = i32 %1
=== countbits_i
  %1 = call i32 @dx.op.unaryBits.i32(i32 31, i32 in0_0)
  out0_0 = i32 %1
=== countbits_u64
  %1 = call i32 @dx.op.unaryBits.i64(i32 31, i64 in0_0)
  out0_0 = i32 %1
=== countbits_i64
  %1 = call i32 @dx.op.unaryBits.i64(i32 31, i64 in0_0)
  out0_0 = i32 %1
=== countbits_u2
  %1 = call i32 @dx.op.unaryBits.i32(i32 31, i32 in0_0)
  %2 = call i32 @dx.op.unaryBits.i32(i32 31, i32 in0_1)
  out0_0 = i32 %1
  out0_1 = i32 %2
=== countbits_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = zext i1 %1 to i32
  %3 = call i32 @dx.op.unaryBits.i32(i32 31, i32 %2)
  out0_0 = i32 %3
=== countbits_f
  %1 = fptosi float in0_0 to i32
  %2 = call i32 @dx.op.unaryBits.i32(i32 31, i32 %1)
  out0_0 = i32 %2
=== reversebits_u
  %1 = call i32 @dx.op.unary.i32(i32 30, i32 in0_0)
  out0_0 = i32 %1
=== reversebits_i
  %1 = call i32 @dx.op.unary.i32(i32 30, i32 in0_0)
  out0_0 = i32 %1
=== reversebits_u64
  %1 = call i64 @dx.op.unary.i64(i32 30, i64 in0_0)
  out0_0 = i64 %1
=== firstbitlow_u
  %1 = call i32 @dx.op.unaryBits.i32(i32 32, i32 in0_0)
  out0_0 = i32 %1
=== firstbitlow_i
  %1 = call i32 @dx.op.unaryBits.i32(i32 32, i32 in0_0)
  out0_0 = i32 %1
=== firstbitlow_u64
  %1 = call i32 @dx.op.unaryBits.i64(i32 32, i64 in0_0)
  out0_0 = i32 %1
=== firstbithigh_u
  %1 = call i32 @dx.op.unaryBits.i32(i32 33, i32 in0_0)
  %2 = icmp eq i32 -1, %1
  %3 = sub i32 31, %1
  %4 = select i1 %2, i32 -1, i32 %3
  out0_0 = i32 %4
=== firstbithigh_i
  %1 = call i32 @dx.op.unaryBits.i32(i32 34, i32 in0_0)
  %2 = icmp eq i32 -1, %1
  %3 = sub i32 31, %1
  %4 = select i1 %2, i32 -1, i32 %3
  out0_0 = i32 %4
=== firstbithigh_u64
  %1 = call i32 @dx.op.unaryBits.i64(i32 33, i64 in0_0)
  %2 = icmp eq i32 -1, %1
  %3 = sub i32 63, %1
  %4 = select i1 %2, i32 -1, i32 %3
  out0_0 = i32 %4
=== firstbithigh_i64
  %1 = call i32 @dx.op.unaryBits.i64(i32 34, i64 in0_0)
  %2 = icmp eq i32 -1, %1
  %3 = sub i32 63, %1
  %4 = select i1 %2, i32 -1, i32 %3
  out0_0 = i32 %4
=== firstbithigh_u3
  %1 = call i32 @dx.op.unaryBits.i32(i32 33, i32 in0_0)
  %2 = icmp eq i32 -1, %1
  %3 = sub i32 31, %1
  %4 = select i1 %2, i32 -1, i32 %3
  %5 = call i32 @dx.op.unaryBits.i32(i32 33, i32 in0_1)
  %6 = icmp eq i32 -1, %5
  %7 = sub i32 31, %5
  %8 = select i1 %6, i32 -1, i32 %7
  %9 = call i32 @dx.op.unaryBits.i32(i32 33, i32 in0_2)
  %10 = icmp eq i32 -1, %9
  %11 = sub i32 31, %9
  %12 = select i1 %10, i32 -1, i32 %11
  out0_0 = i32 %4
  out0_1 = i32 %8
  out0_2 = i32 %12
=== firstbithigh_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = zext i1 %1 to i32
  %3 = call i32 @dx.op.unaryBits.i32(i32 34, i32 %2)
  %4 = icmp eq i32 -1, %3
  %5 = sub i32 31, %3
  %6 = select i1 %4, i32 -1, i32 %5
  out0_0 = i32 %6
=== wave_lane_index
  %1 = call i32 @dx.op.waveGetLaneIndex(i32 111)
  out0_0 = i32 %1
=== wave_lane_count
  %1 = call i32 @dx.op.waveGetLaneCount(i32 112)
  out0_0 = i32 %1
=== wave_is_first
  %1 = call i1 @dx.op.waveIsFirstLane(i32 110)
  %2 = zext i1 %1 to i32
  out0_0 = i32 %2
=== wave_ballot
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call %dx.types.fouri32 @dx.op.waveActiveBallot(i32 116, i1 %1)
  %3 = extractvalue %dx.types.fouri32 %2, 0
  %4 = extractvalue %dx.types.fouri32 %2, 1
  %5 = extractvalue %dx.types.fouri32 %2, 2
  %6 = extractvalue %dx.types.fouri32 %2, 3
  out0_0 = i32 %3
  out0_1 = i32 %4
  out0_2 = i32 %5
  out0_3 = i32 %6
=== wave_ballot_true
  %1 = call %dx.types.fouri32 @dx.op.waveActiveBallot(i32 116, i1 true)
  %2 = extractvalue %dx.types.fouri32 %1, 0
  %3 = extractvalue %dx.types.fouri32 %1, 1
  %4 = extractvalue %dx.types.fouri32 %1, 2
  %5 = extractvalue %dx.types.fouri32 %1, 3
  out0_0 = i32 %2
  out0_1 = i32 %3
  out0_2 = i32 %4
  out0_3 = i32 %5
=== wave_ballot_u
  %1 = icmp ne i32 0, in0_0
  %2 = call %dx.types.fouri32 @dx.op.waveActiveBallot(i32 116, i1 %1)
  %3 = extractvalue %dx.types.fouri32 %2, 0
  %4 = extractvalue %dx.types.fouri32 %2, 1
  %5 = extractvalue %dx.types.fouri32 %2, 2
  %6 = extractvalue %dx.types.fouri32 %2, 3
  out0_0 = i32 %3
  out0_1 = i32 %4
  out0_2 = i32 %5
  out0_3 = i32 %6
=== wave_read_at_u
  %1 = call i32 @dx.op.waveReadLaneAt.i32(i32 117, i32 in0_0, i32 in1_0)
  out0_0 = i32 %1
=== wave_read_at_f3
  %1 = call float @dx.op.waveReadLaneAt.f32(i32 117, float in0_0, i32 in1_0) !precise
  %2 = call float @dx.op.waveReadLaneAt.f32(i32 117, float in0_1, i32 in1_0) !precise
  %3 = call float @dx.op.waveReadLaneAt.f32(i32 117, float in0_2, i32 in1_0) !precise
  out0_0 = float %1
  out0_1 = float %2
  out0_2 = float %3
=== wave_read_at_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i1 @dx.op.waveReadLaneAt.i1(i32 117, i1 %1, i32 in1_0)
  %3 = zext i1 %2 to i32
  out0_0 = i32 %3
=== wave_read_at_i
  %1 = call i32 @dx.op.waveReadLaneAt.i32(i32 117, i32 in0_0, i32 in1_0)
  out0_0 = i32 %1
=== wave_read_at_d
  %1 = call double @dx.op.waveReadLaneAt.f64(i32 117, double in0_0, i32 in1_0) !precise
  out0_0 = double %1
=== wave_read_at_u64
  %1 = call i64 @dx.op.waveReadLaneAt.i64(i32 117, i64 in0_0, i32 in1_0)
  out0_0 = i64 %1
=== wave_read_at_lit
  %1 = call i32 @dx.op.waveReadLaneAt.i32(i32 117, i32 in0_0, i32 3)
  out0_0 = i32 %1
=== wave_read_at_flane
  %1 = fptoui float in1_0 to i32
  %2 = call i32 @dx.op.waveReadLaneAt.i32(i32 117, i32 in0_0, i32 %1)
  out0_0 = i32 %2
=== wave_read_first_u
  %1 = call i32 @dx.op.waveReadLaneFirst.i32(i32 118, i32 in0_0)
  out0_0 = i32 %1
=== wave_read_first_f2
  %1 = call float @dx.op.waveReadLaneFirst.f32(i32 118, float in0_0) !precise
  %2 = call float @dx.op.waveReadLaneFirst.f32(i32 118, float in0_1) !precise
  out0_0 = float %1
  out0_1 = float %2
=== wave_read_first_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i1 @dx.op.waveReadLaneFirst.i1(i32 118, i1 %1)
  %3 = zext i1 %2 to i32
  out0_0 = i32 %3
=== wave_any
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i1 @dx.op.waveAnyTrue(i32 113, i1 %1)
  %3 = zext i1 %2 to i32
  out0_0 = i32 %3
=== wave_all
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i1 @dx.op.waveAllTrue(i32 114, i1 %1)
  %3 = zext i1 %2 to i32
  out0_0 = i32 %3
=== wave_alleq_u
  %1 = call i1 @dx.op.waveActiveAllEqual.i32(i32 115, i32 in0_0)
  %2 = zext i1 %1 to i32
  out0_0 = i32 %2
=== wave_alleq_f3
  %1 = call i1 @dx.op.waveActiveAllEqual.f32(i32 115, float in0_0)
  %2 = zext i1 %1 to i32
  %3 = call i1 @dx.op.waveActiveAllEqual.f32(i32 115, float in0_1)
  %4 = zext i1 %3 to i32
  %5 = call i1 @dx.op.waveActiveAllEqual.f32(i32 115, float in0_2)
  %6 = zext i1 %5 to i32
  out0_0 = i32 %2
  out0_1 = i32 %4
  out0_2 = i32 %6
=== wave_alleq_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i1 @dx.op.waveActiveAllEqual.i1(i32 115, i1 %1)
  %3 = zext i1 %2 to i32
  out0_0 = i32 %3
=== wave_alleq_d
  %1 = call i1 @dx.op.waveActiveAllEqual.f64(i32 115, double in0_0)
  %2 = zext i1 %1 to i32
  out0_0 = i32 %2
=== wave_countbits
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i32 @dx.op.waveAllOp(i32 135, i1 %1)
  out0_0 = i32 %2
=== wave_countbits_true
  %1 = call i32 @dx.op.waveAllOp(i32 135, i1 true)
  out0_0 = i32 %1
=== wave_prefix_countbits
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = call i32 @dx.op.wavePrefixOp(i32 136, i1 %1)
  out0_0 = i32 %2
=== wave_prefix_countbits_true
  %1 = call i32 @dx.op.wavePrefixOp(i32 136, i1 true)
  out0_0 = i32 %1
=== wave_sum_u
  %1 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 in0_0, i8 0, i8 1)
  out0_0 = i32 %1
=== wave_sum_i
  %1 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 in0_0, i8 0, i8 0)
  out0_0 = i32 %1
=== wave_sum_f
  %1 = call float @dx.op.waveActiveOp.f32(i32 119, float in0_0, i8 0, i8 0) !precise
  out0_0 = float %1
=== wave_sum_u2
  %1 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 in0_0, i8 0, i8 1)
  %2 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 in0_1, i8 0, i8 1)
  out0_0 = i32 %1
  out0_1 = i32 %2
=== wave_sum_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = uitofp i1 %1 to float
  %3 = call float @dx.op.waveActiveOp.f32(i32 119, float %2, i8 0, i8 0) !precise
  out0_0 = float %3
=== wave_sum_d
  %1 = call double @dx.op.waveActiveOp.f64(i32 119, double in0_0, i8 0, i8 0) !precise
  out0_0 = double %1
=== wave_sum_u64
  %1 = call i64 @dx.op.waveActiveOp.i64(i32 119, i64 in0_0, i8 0, i8 1)
  out0_0 = i64 %1
=== wave_sum_i64
  %1 = call i64 @dx.op.waveActiveOp.i64(i32 119, i64 in0_0, i8 0, i8 0)
  out0_0 = i64 %1
=== wave_product_u
  %1 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 in0_0, i8 1, i8 1)
  out0_0 = i32 %1
=== wave_product_f
  %1 = call float @dx.op.waveActiveOp.f32(i32 119, float in0_0, i8 1, i8 0) !precise
  out0_0 = float %1
=== wave_min_u
  %1 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 in0_0, i8 2, i8 1)
  out0_0 = i32 %1
=== wave_min_i
  %1 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 in0_0, i8 2, i8 0)
  out0_0 = i32 %1
=== wave_min_f
  %1 = call float @dx.op.waveActiveOp.f32(i32 119, float in0_0, i8 2, i8 0) !precise
  out0_0 = float %1
=== wave_max_u
  %1 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 in0_0, i8 3, i8 1)
  out0_0 = i32 %1
=== wave_max_i
  %1 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 in0_0, i8 3, i8 0)
  out0_0 = i32 %1
=== wave_max_f
  %1 = call float @dx.op.waveActiveOp.f32(i32 119, float in0_0, i8 3, i8 0) !precise
  out0_0 = float %1
=== wave_bitand
  %1 = call i32 @dx.op.waveActiveBit.i32(i32 120, i32 in0_0, i8 0)
  out0_0 = i32 %1
=== wave_bitor
  %1 = call i32 @dx.op.waveActiveBit.i32(i32 120, i32 in0_0, i8 1)
  out0_0 = i32 %1
=== wave_bitxor
  %1 = call i32 @dx.op.waveActiveBit.i32(i32 120, i32 in0_0, i8 2)
  out0_0 = i32 %1
=== wave_bitor_u64
  %1 = call i64 @dx.op.waveActiveBit.i64(i32 120, i64 in0_0, i8 1)
  out0_0 = i64 %1
=== wave_prefix_sum_u
  %1 = call i32 @dx.op.wavePrefixOp.i32(i32 121, i32 in0_0, i8 0, i8 1)
  out0_0 = i32 %1
=== wave_prefix_sum_i
  %1 = call i32 @dx.op.wavePrefixOp.i32(i32 121, i32 in0_0, i8 0, i8 0)
  out0_0 = i32 %1
=== wave_prefix_sum_f
  %1 = call float @dx.op.wavePrefixOp.f32(i32 121, float in0_0, i8 0, i8 0) !precise
  out0_0 = float %1
=== wave_prefix_product_u
  %1 = call i32 @dx.op.wavePrefixOp.i32(i32 121, i32 in0_0, i8 1, i8 1)
  out0_0 = i32 %1
=== wave_prefix_product_i
  %1 = call i32 @dx.op.wavePrefixOp.i32(i32 121, i32 in0_0, i8 1, i8 0)
  out0_0 = i32 %1
=== quad_read_at
  %1 = call i32 @dx.op.quadReadLaneAt.i32(i32 122, i32 in0_0, i32 in1_0)
  out0_0 = i32 %1
=== quad_read_at_f2
  %1 = call float @dx.op.quadReadLaneAt.f32(i32 122, float in0_0, i32 in1_0) !precise
  %2 = call float @dx.op.quadReadLaneAt.f32(i32 122, float in0_1, i32 in1_0) !precise
  out0_0 = float %1
  out0_1 = float %2
=== quad_read_at_lit
  %1 = call i32 @dx.op.quadReadLaneAt.i32(i32 122, i32 in0_0, i32 2)
  out0_0 = i32 %1
=== quad_read_at_b
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = uitofp i1 %1 to float
  %3 = call float @dx.op.quadReadLaneAt.f32(i32 122, float %2, i32 in1_0) !precise
  out0_0 = float %3
=== quad_x
  %1 = call float @dx.op.quadOp.f32(i32 123, float in0_0, i8 0) !precise
  out0_0 = float %1
=== quad_y
  %1 = call i32 @dx.op.quadOp.i32(i32 123, i32 in0_0, i8 1)
  out0_0 = i32 %1
=== quad_diag
  %1 = call i32 @dx.op.quadOp.i32(i32 123, i32 in0_0, i8 2)
  out0_0 = i32 %1
=== quad_any
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = sext i1 %1 to i32
  %3 = call i32 @dx.op.quadOp.i32(i32 123, i32 %2, i8 2)
  %4 = call i32 @dx.op.quadOp.i32(i32 123, i32 %2, i8 0)
  %5 = call i32 @dx.op.quadOp.i32(i32 123, i32 %2, i8 1)
  %6 = or i32 %4, %5
  %7 = or i32 %3, %6
  %8 = or i32 %7, %2
  %9 = trunc i32 %8 to i1
  %10 = zext i1 %9 to i32
  out0_0 = i32 %10
=== quad_all
  %1 = call i1 @dx.op.isSpecialFloat.f32(i32 8, float in0_0)
  %2 = sext i1 %1 to i32
  %3 = call i32 @dx.op.quadOp.i32(i32 123, i32 %2, i8 0)
  %4 = call i32 @dx.op.quadOp.i32(i32 123, i32 %2, i8 1)
  %5 = and i32 %3, %4
  %6 = call i32 @dx.op.quadOp.i32(i32 123, i32 %2, i8 2)
  %7 = and i32 %5, %6
  %8 = and i32 %7, %2
  %9 = trunc i32 %8 to i1
  %10 = zext i1 %9 to i32
  out0_0 = i32 %10
=== barrier_g
  call void @dx.op.barrier(i32 80, i32 8)
=== barrier_gs
  call void @dx.op.barrier(i32 80, i32 9)
=== barrier_d
  call void @dx.op.barrier(i32 80, i32 2)
=== barrier_ds
  call void @dx.op.barrier(i32 80, i32 3)
=== barrier_a
  call void @dx.op.barrier(i32 80, i32 10)
=== barrier_as
  call void @dx.op.barrier(i32 80, i32 11)
=== clip_f
  %1 = fcmp olt float in0_0, 0.000000e+00
  call void @dx.op.discard(i32 82, i1 %1)
=== clip_f3
  %1 = fcmp olt float in0_2, 0.000000e+00
  %2 = fcmp olt float in0_0, 0.000000e+00
  %3 = fcmp olt float in0_1, 0.000000e+00
  %4 = or i1 %2, %3
  %5 = or i1 %1, %4
  call void @dx.op.discard(i32 82, i1 %5)
=== clip_i
  %1 = sitofp i32 in0_0 to float
  %2 = fcmp olt float %1, 0.000000e+00
  call void @dx.op.discard(i32 82, i1 %2)
=== rt_sample_count
  %1 = call i32 @dx.op.renderTargetGetSampleCount(i32 77)
  out0_0 = i32 %1
=== rt_sample_pos
  %1 = call %dx.types.SamplePos @dx.op.renderTargetGetSamplePosition(i32 76, i32 in0_0)
  %2 = extractvalue %dx.types.SamplePos %1, 0
  %3 = extractvalue %dx.types.SamplePos %1, 1
  out0_0 = float %2
  out0_1 = float %3
=== is_helper
  %1 = call i1 @dx.op.isHelperLane.i1(i32 221)
  %2 = zext i1 %1 to i32
  out0_0 = i32 %2
=== set_mesh
  call void @dx.op.setMeshOutputCounts(i32 168, i32 in0_0, i32 in1_0)
"#;
