//! The intrinsic lowerings against DXC, one fixture per overload.
//!
//! A fixture names typed inputs and one intrinsic call. Two things are built from it:
//! * a shader for DXC (`dxc -HV 2021 -Gis`, the options the emulator compiles with): inputs
//!   loaded from a raw buffer, so nothing folds, and the result stored back through a
//!   template, so the store shows the type DXC gave the call,
//! * the same call through `hlsl::intrinsics::call` on an `Emit`, inputs loaded from globals.
//!
//! Both IR texts (DXC's `-Fc` listing and `irprint::function`) go through one normaliser,
//! `canonical`, which reduces a function to the expression DAG behind each stored value: the
//! result is the same text whenever the same operations are applied to the same inputs in
//! the same operand order, whatever the instruction order or value numbering.
//!
//! `intrinsics_golden.rs` holds DXC's normalised output for every fixture, so the comparison
//! runs on any machine. With `XCL_DXIL_DXC` set (the `dxc-linux` folder) the goldens are
//! checked against DXC itself, and `XCL_DXIL_BLESS=1` rewrites them from it.

mod intrinsics_golden;

use std::collections::HashMap;
use std::fmt::Write as _;
use xcl_dxil::consts::OpCode;
use xcl_dxil::hlsl::emit::Emit;
use xcl_dxil::hlsl::intrinsics;
use xcl_dxil::hlsl::types::{HTy, Sk, Val};
use xcl_dxil::ir::{CastOp, Ty};
use xcl_dxil::{irprint, Error};

/// `name | inputs | result type | call | out types | flags`. Arguments of the call are input
/// names, `o0`.. for `out` parameters, or literals. Flags: `ps` / `ms` (the stage, compute
/// otherwise) and `O0`: the golden is DXC's `-O0` output, for expansions `-O1` rewrites into
/// an equivalent form (`frexp`'s masks become selects, `sign`'s compare becomes a shift).
const FIXTURES: &str = "
# ---- constant arguments: DXC evaluates these, and so does the fold (exact results only)
cf_exp2 | | float | exp2(-14.0f)
cf_exp2_0 | | float | exp2(0.0f)
cf_exp2_sub | | float | exp2(-140.0f)
cf_log2 | | float | log2(8.0f)
cf_sqrt | | float | sqrt(2.0f)
cf_rsqrt | | float | rsqrt(4.0f)
cf_floor | | float | floor(-1.5f)
cf_ceil | | float | ceil(-1.5f)
cf_round | | float | round(2.5f)
cf_round_odd | | float | round(3.5f)
cf_trunc | | float | trunc(-1.5f)
cf_frac | | float | frac(1.25f)
cf_abs | | float | abs(-1.5f)
cf_saturate | | float | saturate(1.5f)
cf_max_f | | float | max(1.0f, 2.0f)
cf_min_f | | float | min(1.0f, 2.0f)
cf_min_u | | uint | min(11u, 77u)
cf_max_u | | uint | max(0xFFFFFFFFu, 5u)
cf_clamp_f | | float | clamp(5.0f, 0.0f, 1.0f)
cf_clamp_u | | uint | clamp(9u, 1u, 7u)
cf_mad_f | | float | mad(2.0f, 3.0f, 0.1f)
cf_mad_u | | uint | mad(3u, 4u, 5u)
cf_reversebits | | uint | reversebits(40u)
cf_countbits | | uint | countbits(0xF0F0u)
cf_firstbitlow | | uint | firstbitlow(0x00000100u)
cf_firstbitlow_0 | | uint | firstbitlow(0u)
cf_firstbithigh | | uint | firstbithigh(0x00000100u)
cf_firstbithigh_0 | | uint | firstbithigh(0u)
cf_isnan | | bool | isnan(1.0f)
# ---- reinterpretation
asuint_f | float a | uint | asuint(a)
asuint_i | int a | uint | asuint(a)
asuint_u | uint a | uint | asuint(a)
asuint_f3 | float3 a | uint3 | asuint(a)
asuint_lit | | uint | asuint(1.5)
asuint_ilit | | uint | asuint(-1)
asint_f | float a | int | asint(a)
asint_u | uint a | int | asint(a)
asint_f2 | float2 a | int2 | asint(a)
asfloat_u | uint a | float | asfloat(a)
asfloat_i | int a | float | asfloat(a)
asfloat_f | float a | float | asfloat(a)
asfloat_u4 | uint4 a | float4 | asfloat(a)
asfloat_inf | | float | asfloat(0x7F800000u)
asfloat_ilit | | float | asfloat(0x3F800000)
asdouble_u | uint a, uint b | double | asdouble(a, b)
asdouble_lit | uint a | double | asdouble(a, 0x3FE00000u)
asdouble_lits | | double | asdouble(0u, 0x7FE00000u)
asdouble_u2 | uint2 a, uint2 b | double2 | asdouble(a, b)
asdouble_i | int a, int b | double | asdouble(a, b)
asuint_d | double a | void | asuint(a, o0, o1) | uint, uint
asuint_d2 | double2 a | void | asuint(a, o0, o1) | uint2, uint2
f16tof32_u | uint a | float | f16tof32(a)
f16tof32_i | int a | float | f16tof32(a)
f16tof32_u3 | uint3 a | float3 | f16tof32(a)
f16tof32_lit | | float | f16tof32(0x3C00u)
f16tof32_f | float a | float | f16tof32(a)
f16tof32_u64 | uint64_t a | float | f16tof32(a)
f32tof16_f | float a | uint | f32tof16(a)
f32tof16_d | double a | uint | f32tof16(a)
f32tof16_f2 | float2 a | uint2 | f32tof16(a)
f32tof16_u | uint a | uint | f32tof16(a)
# ---- select / and / or / any / all
select_s | bool c, float a, float b | float | select(c, a, b)
select_v_s | bool3 c, float a, float3 b | float3 | select(c, a, b)
select_v_v | bool3 c, uint3 a, uint3 b | uint3 | select(c, a, b)
select_s_v | bool c, uint3 a, uint3 b | uint3 | select(c, a, b)
select_v3_s_s | bool3 c, float a, float b | float3 | select(c, a, b)
select_s_v_s | bool c, float3 a, float b | float3 | select(c, a, b)
select_c2_v3 | bool2 c, float3 a, float3 b | float2 | select(c, a, b)
select_lit | bool c, uint a | uint | select(c, a, 0xFFFFFFFFu)
select_ilit | bool c, uint a | uint | select(c, a, 5)
select_i_u | bool c, int a, uint b | uint | select(c, a, b)
select_u_i | bool c, uint a, int b | uint | select(c, a, b)
select_f_i | bool c, float a, int b | float | select(c, a, b)
select_i_f | bool c, int a, float b | float | select(c, a, b)
select_flit | bool c, float a | float | select(c, 0.0f, a)
select_f_lit | bool c, float a | float | select(c, a, 1.5)
select_d_lit | bool c, double a | double | select(c, a, 1.1)
select_u_cond | uint c, float a, float b | float | select(c, a, b)
select_f_cond | float c, uint a, uint b | uint | select(c, a, b)
select_f_d | bool c, float a, double b | double | select(c, a, b)
select_b | bool c, bool a, bool b | bool | select(c, a, b)
select_u64 | bool c, uint64_t a, uint b | uint64_t | select(c, a, b)
select_same | bool c, float a | float | select(c, a, a)
select_true | uint a, uint b | uint | select(true, a, b)
and_s | bool a, bool b | bool | and(a, b)
and_v | bool3 a, bool3 b | bool3 | and(a, b)
and_v_s | bool3 a, bool b | bool3 | and(a, b)
and_u | uint a, bool b | bool | and(a, b)
and_v3_v2 | bool3 a, bool2 b | bool2 | and(a, b)
or_s | bool a, bool b | bool | or(a, b)
or_v_s | bool b, bool3 a | bool3 | or(b, a)
or_f | float a, float b | bool | or(a, b)
any_b | bool3 a | bool | any(a)
any_u | uint3 a | bool | any(a) | | O0
any_f | float4 a | bool | any(a)
any_s | float a | bool | any(a)
any_i2 | int2 a | bool | any(a) | | O0
all_b | bool4 a | bool | all(a)
all_f | float2 a | bool | all(a)
all_u | uint a | bool | all(a)
all_d | double2 a | bool | all(a)
all_u64 | uint64_t2 a | bool | all(a)
isnan_f | float a | bool | isnan(a)
isnan_f3 | float3 a | bool3 | isnan(a)
isnan_d | double a | bool | isnan(a)
isnan_i | int a | bool | isnan(a)
isinf_f | float a | bool | isinf(a)
isfinite_f | float a | bool | isfinite(a)
isnormal_f2 | float2 a | bool2 | isnormal(a)
# ---- numeric
abs_f | float a | float | abs(a)
abs_f3 | float3 a | float3 | abs(a)
abs_d | double a | double | abs(a)
abs_i | int a | int | abs(a)
abs_i3 | int3 a | int3 | abs(a)
abs_i64 | int64_t a | int64_t | abs(a)
abs_u | uint a | uint | abs(a)
abs_u64 | uint64_t a | uint64_t | abs(a)
abs_b | bool a | float | abs(a)
sign_f | float a | int | sign(a) | | O0
sign_f3 | float3 a | int3 | sign(a) | | O0
sign_d | double a | int | sign(a) | | O0
sign_i | int a | int | sign(a) | | O0
sign_i64 | int64_t a | int | sign(a) | | O0
sign_u | uint a | int | sign(a)
sign_u64 | uint64_t a | int | sign(a)
min_f | float a, float b | float | min(a, b)
min_i | int a, int b | int | min(a, b)
min_u | uint a, uint b | uint | min(a, b)
min_d | double a, double b | double | min(a, b)
min_u64 | uint64_t a, uint64_t b | uint64_t | min(a, b)
max_f | float a, float b | float | max(a, b)
max_i | int a, int b | int | max(a, b)
max_u | uint a, uint b | uint | max(a, b)
max_i64 | int64_t a, int64_t b | int64_t | max(a, b)
min_u_lit | uint a | uint | min(a, 5)
min_i_lit | int a | int | min(a, 5)
min_u_i | uint a, int b | uint | min(a, b)
min_i_u | int a, uint b | uint | min(a, b)
min_f_i | float a, int b | float | min(a, b)
min_i_f | int a, float b | float | min(a, b)
min_f_lit | float a | float | min(a, 1)
min_f_flit | float a | float | min(a, 1.5)
min_d_flit | double a | double | min(a, 1.1)
min_f_d | float a, double b | double | min(a, b)
min_f3_f | float3 a, float b | float3 | min(a, b)
min_u3_u | uint3 a, uint b | uint3 | min(a, b)
min_u3_u2 | uint3 a, uint2 b | uint2 | min(a, b)
min_i64_i | int64_t a, int b | int64_t | min(a, b)
min_u64_u | uint64_t a, uint b | uint64_t | min(a, b)
max_i_u64 | int a, uint64_t b | uint64_t | max(a, b)
min_b_b | bool a, bool b | float | min(a, b)
min_b_i | bool a, int b | int | min(a, b)
min_b_u | bool a, uint b | uint | min(a, b)
max_f_lit0 | float a | float | max(a, 0.0f)
clamp_f | float a, float b, float c | float | clamp(a, b, c)
clamp_u | uint a, uint b, uint c | uint | clamp(a, b, c)
clamp_i | int a | int | clamp(a, 0, 255)
clamp_f_lit | float a | float | clamp(a, 0, 1)
clamp_u_lit | uint a | uint | clamp(a, 1, 7)
clamp_i_u | int a, uint b | uint | clamp(a, 0, b) | | O0
clamp_d_i32 | double a | double | clamp(a, -2147483648.0, 2147483647.0)
clamp_d_u32 | double a | double | clamp(a, 0.0, 4294967295.0)
clamp_f3 | float3 a, float b, float3 c | float3 | clamp(a, b, c)
clamp_b | bool a, bool b, bool c | float | clamp(a, b, c)
saturate_f | float a | float | saturate(a)
saturate_f4 | float4 a | float4 | saturate(a)
saturate_d | double a | double | saturate(a)
saturate_i | int a | float | saturate(a)
mad_f | float a, float b, float c | float | mad(a, b, c)
mad_f3 | float3 a, float3 b, float3 c | float3 | mad(a, b, c)
mad_d | double a, double b, double c | double | mad(a, b, c)
mad_i | int a, int b, int c | int | mad(a, b, c)
mad_u | uint a, uint b, uint c | uint | mad(a, b, c)
mad_i64 | int64_t a, int64_t b, int64_t c | int64_t | mad(a, b, c)
mad_u64 | uint64_t a, uint64_t b, uint64_t c | uint64_t | mad(a, b, c)
mad_f_lit | float a, float b | float | mad(a, 2, b)
mad_i_u | int a, uint b, int c | uint | mad(a, b, c)
mad_u_lit | uint a, uint b | uint | mad(a, 3, b)
mad_b | bool a, bool b, bool c | float | mad(a, b, c)
fma_d | double a, double b, double c | double | fma(a, b, c)
fma_flit | double a | double | fma(a, 2.0, 1.0)
rcp_f | float a | float | rcp(a)
rcp_d | double a | double | rcp(a)
rcp_i | int a | float | rcp(a)
rcp_f3 | float3 a | float3 | rcp(a)
# ---- float math
sin_f | float a | float | sin(a)
sin_f3 | float3 a | float3 | sin(a)
sin_i | int a | float | sin(a)
sin_d | double a | float | sin(a)
cos_f | float a | float | cos(a)
tan_f | float a | float | tan(a)
asin_f | float a | float | asin(a)
acos_f | float a | float | acos(a)
atan_f | float a | float | atan(a)
sinh_f | float a | float | sinh(a)
cosh_f | float a | float | cosh(a)
tanh_f | float a | float | tanh(a)
sqrt_f | float a | float | sqrt(a)
sqrt_d | double a | float | sqrt(a)
rsqrt_f | float a | float | rsqrt(a)
exp_f | float a | float | exp(a)
exp_f2 | float2 a | float2 | exp(a)
exp2_f | float a | float | exp2(a)
log_f | float a | float | log(a)
log2_f | float a | float | log2(a)
log10_f | float a | float | log10(a)
degrees_f | float a | float | degrees(a)
radians_f | float a | float | radians(a)
floor_f | float a | float | floor(a)
floor_d | double a | float | floor(a)
ceil_f | float a | float | ceil(a)
round_f | float a | float | round(a)
trunc_f | float a | float | trunc(a)
frac_f | float a | float | frac(a)
pow_f | float a, float b | float | pow(a, b)
pow_f3_f | float3 a, float b | float3 | pow(a, b)
pow_f_2 | float a | float | pow(a, 2.0)
pow_f_2i | float a | float | pow(a, 2)
pow_f3_2 | float3 a | float3 | pow(a, 2.0)
pow_f_3 | float a | float | pow(a, 3.0)
pow_f_m2 | float a | float | pow(a, -2.0)
pow_f_half | float a | float | pow(a, 0.5)
atan2_f | float a, float b | float | atan2(a, b) | | O0
atan2_f2 | float2 a, float2 b | float2 | atan2(a, b) | | O0
sincos_f | float a | void | sincos(a, o0, o1) | float, float
fmod_f | float a, float b | float | fmod(a, b)
fmod_f3_f | float3 a, float b | float3 | fmod(a, b)
modf_f | float a | float | modf(a, o0) | float
modf_f3 | float3 a | float3 | modf(a, o0) | float3
modf_d | double a | float | modf(a, o0) | float
frexp_f | float a | float | frexp(a, o0) | float | O0
ldexp_f | float a, float b | float | ldexp(a, b)
ldexp_f_i | float a, int b | float | ldexp(a, b)
lerp_f | float a, float b, float c | float | lerp(a, b, c)
lerp_f3_f | float3 a, float3 b, float c | float3 | lerp(a, b, c)
lerp_i | int a, int b, int c | float | lerp(a, b, c)
lerp_d | double a, double b, double c | float | lerp(a, b, c)
step_f | float a, float b | float | step(a, b)
step_lit | float a | float | step(0.5, a)
smoothstep_f | float a, float b, float c | float | smoothstep(a, b, c)
ddx_f | float a | float | ddx(a) | | ps
ddx_f2 | float2 a | float2 | ddx(a) | | ps
ddy_f | float a | float | ddy(a) | | ps
ddx_coarse_f | float a | float | ddx_coarse(a) | | ps
ddy_coarse_f | float a | float | ddy_coarse(a) | | ps
ddx_fine_f | float a | float | ddx_fine(a) | | ps
ddy_fine_f | float a | float | ddy_fine(a) | | ps
ddx_i | int a | float | ddx(a) | | ps O0
fwidth_f | float a | float | fwidth(a) | | ps
# ---- vectors
dot_f2 | float2 a, float2 b | float | dot(a, b)
dot_f3 | float3 a, float3 b | float | dot(a, b)
dot_f4 | float4 a, float4 b | float | dot(a, b)
dot_f | float a, float b | float | dot(a, b)
dot_f3_f | float3 a, float b | float | dot(a, b)
dot_i3 | int3 a, int3 b | int | dot(a, b)
dot_u2 | uint2 a, uint2 b | uint | dot(a, b)
dot_b | bool2 a, bool2 b | float | dot(a, b)
cross_f | float3 a, float3 b | float3 | cross(a, b)
length_f3 | float3 a | float | length(a)
length_f2 | float2 a | float | length(a)
length_f4 | float4 a | float | length(a)
length_f | float a | float | length(a)
distance_f3 | float3 a, float3 b | float | distance(a, b)
distance_f | float a, float b | float | distance(a, b)
normalize_f3 | float3 a | float3 | normalize(a)
normalize_f2 | float2 a | float2 | normalize(a)
normalize_f | float a | float | normalize(a)
reflect_f3 | float3 a, float3 b | float3 | reflect(a, b)
refract_f3 | float3 a, float3 b, float c | float3 | refract(a, b, c)
faceforward_f3 | float3 a, float3 b, float3 c | float3 | faceforward(a, b, c)
mul_ff | float a, float b | float | mul(a, b)
mul_vv | float3 a, float3 b | float | mul(a, b)
mul_sv | float a, float3 b | float3 | mul(a, b)
mul_vs | float3 a, float b | float3 | mul(a, b)
mul_iu | int a, uint b | uint | mul(a, b)
mul_uu3 | uint3 a, uint3 b | uint | mul(a, b)
mul_mv | float2x2 a, float2 b | float2 | mul(a, b)
mul_vm | float2 a, float2x2 b | float2 | mul(a, b)
mul_mm | float2x2 a, float2x2 b | float2x2 | mul(a, b)
mul_sm | float a, float2x2 b | float2x2 | mul(a, b)
mul_ms | float2x2 a, float b | float2x2 | mul(a, b)
mul_umv | uint2x2 a, uint2 b | uint2 | mul(a, b)
dot4add_u | uint a, uint b, uint c | uint | dot4add_u8packed(a, b, c)
dot4add_i | uint a, uint b, int c | int | dot4add_i8packed(a, b, c)
# ---- bits
countbits_u | uint a | uint | countbits(a)
countbits_i | int a | uint | countbits(a)
countbits_u64 | uint64_t a | uint | countbits(a)
countbits_i64 | int64_t a | uint | countbits(a)
countbits_u2 | uint2 a | uint2 | countbits(a)
countbits_b | bool a | uint | countbits(a)
countbits_f | float a | uint | countbits(a)
reversebits_u | uint a | uint | reversebits(a)
reversebits_i | int a | int | reversebits(a)
reversebits_u64 | uint64_t a | uint64_t | reversebits(a)
firstbitlow_u | uint a | uint | firstbitlow(a)
firstbitlow_i | int a | uint | firstbitlow(a)
firstbitlow_u64 | uint64_t a | uint | firstbitlow(a)
firstbithigh_u | uint a | uint | firstbithigh(a)
firstbithigh_i | int a | uint | firstbithigh(a)
firstbithigh_u64 | uint64_t a | uint | firstbithigh(a)
firstbithigh_i64 | int64_t a | uint | firstbithigh(a)
firstbithigh_u3 | uint3 a | uint3 | firstbithigh(a)
firstbithigh_b | bool a | uint | firstbithigh(a)
# ---- wave and quad
wave_lane_index | | uint | WaveGetLaneIndex()
wave_lane_count | | uint | WaveGetLaneCount()
wave_is_first | | bool | WaveIsFirstLane()
wave_ballot | bool a | uint4 | WaveActiveBallot(a)
wave_ballot_true | | uint4 | WaveActiveBallot(true)
wave_ballot_u | uint a | uint4 | WaveActiveBallot(a)
wave_read_at_u | uint a, uint b | uint | WaveReadLaneAt(a, b)
wave_read_at_f3 | float3 a, uint b | float3 | WaveReadLaneAt(a, b)
wave_read_at_b | bool a, uint b | bool | WaveReadLaneAt(a, b)
wave_read_at_i | uint a, int b | uint | WaveReadLaneAt(a, b)
wave_read_at_d | double a, uint b | double | WaveReadLaneAt(a, b)
wave_read_at_u64 | uint64_t a, uint b | uint64_t | WaveReadLaneAt(a, b)
wave_read_at_lit | uint a | uint | WaveReadLaneAt(a, 3)
wave_read_at_flane | uint a, float b | uint | WaveReadLaneAt(a, b)
wave_read_first_u | uint a | uint | WaveReadLaneFirst(a)
wave_read_first_f2 | float2 a | float2 | WaveReadLaneFirst(a)
wave_read_first_b | bool a | bool | WaveReadLaneFirst(a)
wave_any | bool a | bool | WaveActiveAnyTrue(a)
wave_all | bool a | bool | WaveActiveAllTrue(a)
wave_alleq_u | uint a | bool | WaveActiveAllEqual(a)
wave_alleq_f3 | float3 a | bool3 | WaveActiveAllEqual(a)
wave_alleq_b | bool a | bool | WaveActiveAllEqual(a)
wave_alleq_d | double a | bool | WaveActiveAllEqual(a)
wave_countbits | bool a | uint | WaveActiveCountBits(a)
wave_countbits_true | | uint | WaveActiveCountBits(true)
wave_prefix_countbits | bool a | uint | WavePrefixCountBits(a)
wave_prefix_countbits_true | | uint | WavePrefixCountBits(true)
wave_sum_u | uint a | uint | WaveActiveSum(a)
wave_sum_i | int a | int | WaveActiveSum(a)
wave_sum_f | float a | float | WaveActiveSum(a)
wave_sum_u2 | uint2 a | uint2 | WaveActiveSum(a)
wave_sum_b | bool a | float | WaveActiveSum(a)
wave_sum_d | double a | double | WaveActiveSum(a)
wave_sum_u64 | uint64_t a | uint64_t | WaveActiveSum(a)
wave_sum_i64 | int64_t a | int64_t | WaveActiveSum(a)
wave_product_u | uint a | uint | WaveActiveProduct(a)
wave_product_f | float a | float | WaveActiveProduct(a)
wave_min_u | uint a | uint | WaveActiveMin(a)
wave_min_i | int a | int | WaveActiveMin(a)
wave_min_f | float a | float | WaveActiveMin(a)
wave_max_u | uint a | uint | WaveActiveMax(a)
wave_max_i | int a | int | WaveActiveMax(a)
wave_max_f | float a | float | WaveActiveMax(a)
wave_bitand | uint a | uint | WaveActiveBitAnd(a)
wave_bitor | uint a | uint | WaveActiveBitOr(a)
wave_bitxor | uint a | uint | WaveActiveBitXor(a)
wave_bitor_u64 | uint64_t a | uint64_t | WaveActiveBitOr(a)
wave_prefix_sum_u | uint a | uint | WavePrefixSum(a)
wave_prefix_sum_i | int a | int | WavePrefixSum(a)
wave_prefix_sum_f | float a | float | WavePrefixSum(a)
wave_prefix_product_u | uint a | uint | WavePrefixProduct(a)
wave_prefix_product_i | int a | int | WavePrefixProduct(a)
quad_read_at | uint a, uint b | uint | QuadReadLaneAt(a, b)
quad_read_at_f2 | float2 a, uint b | float2 | QuadReadLaneAt(a, b)
quad_read_at_lit | uint a | uint | QuadReadLaneAt(a, 2u)
quad_read_at_b | bool a, uint b | float | QuadReadLaneAt(a, b)
quad_x | float a | float | QuadReadAcrossX(a)
quad_y | uint a | uint | QuadReadAcrossY(a)
quad_diag | int a | int | QuadReadAcrossDiagonal(a)
quad_any | bool a | bool | QuadAny(a)
quad_all | bool a | bool | QuadAll(a)
# ---- effects and stage queries
barrier_g | | void | GroupMemoryBarrier()
barrier_gs | | void | GroupMemoryBarrierWithGroupSync()
barrier_d | | void | DeviceMemoryBarrier()
barrier_ds | | void | DeviceMemoryBarrierWithGroupSync()
barrier_a | | void | AllMemoryBarrier()
barrier_as | | void | AllMemoryBarrierWithGroupSync()
clip_f | float a | void | clip(a) | | ps
clip_f3 | float3 a | void | clip(a) | | ps
clip_i | int a | void | clip(a) | | ps O0
rt_sample_count | | uint | GetRenderTargetSampleCount() | | ps
rt_sample_pos | int a | float2 | GetRenderTargetSamplePosition(a) | | ps
is_helper | | bool | IsHelperLane() | | ps
set_mesh | uint a, uint b | void | SetMeshOutputCounts(a, b) | | ms
";

struct Fixture {
    name: String,
    /// (type, name)
    inputs: Vec<(String, String)>,
    ret: String,
    func: String,
    args: Vec<String>,
    outs: Vec<String>,
    stage: &'static str,
    o0: bool,
}

fn fixtures() -> Vec<Fixture> {
    let list = |s: &str| s.split(',').map(str::trim).filter(|x| !x.is_empty()).map(str::to_string).collect::<Vec<_>>();
    let mut out = Vec::new();
    for line in FIXTURES.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let p: Vec<&str> = line.split('|').map(str::trim).collect();
        assert!(p.len() >= 4, "fixture {line}");
        let (func, rest) = p[3].split_once('(').unwrap_or_else(|| panic!("fixture {line}"));
        let flags = p.get(5).copied().unwrap_or("");
        out.push(Fixture {
            name: p[0].to_string(),
            inputs: list(p[1]).iter().map(|i| i.split_once(' ').map(|(t, n)| (t.to_string(), n.to_string())).unwrap()).collect(),
            ret: p[2].to_string(),
            func: func.to_string(),
            args: list(rest.strip_suffix(')').unwrap()),
            outs: list(p.get(4).copied().unwrap_or("")),
            stage: if flags.contains("ps") { "ps" } else if flags.contains("ms") { "ms" } else { "cs" },
            o0: flags.contains("O0"),
        });
    }
    out
}

/// An HLSL numeric type name.
fn hty(s: &str) -> HTy {
    if s == "void" {
        return HTy::Void;
    }
    let kinds = [("uint64_t", Sk::Uint64), ("int64_t", Sk::Int64), ("double", Sk::Double), ("float", Sk::Float), ("uint", Sk::Uint), ("int", Sk::Int), ("bool", Sk::Bool)];
    let (k, dims) = kinds.iter().find_map(|&(n, k)| s.strip_prefix(n).map(|d| (k, d))).unwrap_or_else(|| panic!("type {s}"));
    let d: Vec<u8> = dims.split('x').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect();
    match d[..] {
        [] => HTy::scalar(k),
        [n] => HTy::vec(k, n),
        [r, c] => HTy::Num(k, r, c),
        _ => panic!("type {s}"),
    }
}

/// The type an input or a result goes through the buffer as: a bool input is `isnan` of a
/// float (an i1 the optimiser cannot see through), a matrix is built from the vector of its
/// components, which the constructor takes in row-major order like `Val`.
fn buffer_ty(s: &str) -> (String, String) {
    let HTy::Num(k, r, c) = hty(s) else { panic!("type {s}") };
    let base = match k {
        Sk::Bool | Sk::Float => "float",
        Sk::Uint => "uint",
        Sk::Int => "int",
        Sk::Double => "double",
        Sk::Uint64 => "uint64_t",
        _ => "int64_t",
    };
    let n = r as usize * c as usize;
    let vec = if n == 1 { base.to_string() } else { format!("{base}{n}") };
    let wrap = if k == Sk::Bool {
        "isnan"
    } else if r > 1 {
        s
    } else {
        ""
    };
    (vec, wrap.to_string())
}

fn hlsl(f: &Fixture) -> String {
    let mut s = String::from("RWByteAddressBuffer io : register(u0);\ntemplate<typename T> void put(uint off, T v) { io.Store<T>(off, v); }\n");
    s += match f.stage {
        "cs" => "[numthreads(1, 1, 1)]\nvoid main() {\n",
        "ms" => "struct V { float4 p : SV_Position; };\n[numthreads(1, 1, 1)]\n[outputtopology(\"triangle\")]\nvoid main(out vertices V v[3], out indices uint3 t[1]) {\n",
        _ => "void main() {\n",
    };
    for (i, (ty, name)) in f.inputs.iter().enumerate() {
        let (vec, wrap) = buffer_ty(ty);
        let _ = writeln!(s, "    {ty} {name} = {wrap}(io.Load<{vec}>({}));", 64 * i);
    }
    for (i, ty) in f.outs.iter().enumerate() {
        let _ = writeln!(s, "    {ty} o{i};");
    }
    let call = format!("{}({})", f.func, f.args.join(", "));
    if f.ret == "void" {
        let _ = writeln!(s, "    {call};");
    } else if matches!(hty(&f.ret), HTy::Num(_, r, _) if r > 1) {
        // a matrix is stored column-major: flatten it to the row-major vector `Val` holds
        let _ = writeln!(s, "    put(1024, ({})({call}));", buffer_ty(&f.ret).0);
    } else {
        let _ = writeln!(s, "    put(1024, {call});");
    }
    for i in 0..f.outs.len() {
        let _ = writeln!(s, "    put({}, o{i});", 1088 + 64 * i);
    }
    s + "}\n"
}

// ---- the Emit side ----

fn leaf_ir(k: Sk) -> Ty {
    if k == Sk::Bool {
        Ty::F32
    } else {
        k.ir()
    }
}

/// An argument that is not an input: an `out` placeholder or a literal. Unsuffixed float
/// literals are passed as double constants: `call` converts a literal constant from the
/// width it arrives in, and only a double keeps `4294967295.0`.
fn literal(e: &mut Emit, f: &Fixture, a: &str) -> Val {
    if let Some(i) = a.strip_prefix('o').and_then(|i| i.parse::<usize>().ok()) {
        let ty = hty(&f.outs[i]);
        let HTy::Num(k, r, c) = ty else { panic!() };
        let z = e.zero(k);
        return Val::new(ty, &vec![z; r as usize * c as usize]);
    }
    match a {
        "true" => return Val::scalar(Sk::Bool, e.c_bool(true)),
        "false" => return Val::scalar(Sk::Bool, e.c_bool(false)),
        _ => {}
    }
    let int = |t: &str| match t.strip_prefix("0x") {
        Some(h) => i64::from_str_radix(h, 16).unwrap(),
        None => t.parse::<i64>().unwrap(),
    };
    if a.contains('.') {
        return match a.strip_suffix('f') {
            Some(x) => Val::scalar(Sk::Float, e.c_f32(x.parse().unwrap())),
            None => Val::scalar(Sk::LitFloat, e.m.const_f64(a.parse().unwrap())),
        };
    }
    match a.strip_suffix('u') {
        Some(x) => Val::scalar(Sk::Uint, e.c_u32(int(x) as u32)),
        None => Val::scalar(Sk::LitInt, e.c_u32(int(a) as u32)),
    }
}

/// Store a result the way the template `put` does: a bool is widened to its i32 storage.
fn store(e: &mut Emit, sink: usize, v: &Val) {
    let HTy::Num(k, ..) = v.ty else { return };
    for (j, &c) in v.comps().iter().enumerate() {
        let c = if k == Sk::Bool { e.cast(CastOp::ZExt, Ty::I32, c) } else { c };
        let ty = e.ty_of(c);
        let g = e.m.global(&format!("out{sink}_{j}"), ty, 0, None, false);
        e.body.store(e.bb, g, c);
    }
}

/// Build the fixture's call through `Emit` and return the function text.
fn emit(f: &Fixture) -> Result<String, String> {
    let mut e = Emit::new();
    let mut vals: HashMap<&str, Val> = HashMap::new();
    for (i, (ty, name)) in f.inputs.iter().enumerate() {
        let t = hty(ty);
        let HTy::Num(k, r, c) = t else { panic!() };
        let mut comps = Vec::new();
        for j in 0..r as usize * c as usize {
            let g = e.m.global(&format!("in{i}_{j}"), leaf_ir(k), 0, None, false);
            let mut v = e.body.load(e.bb, leaf_ir(k), g);
            if k == Sk::Bool {
                v = e.dx(OpCode::IsNaN, Ty::F32, &[v]);
            }
            comps.push(v);
        }
        vals.insert(name, Val::new(t, &comps));
    }
    let args: Vec<Val> = f.args.iter().map(|a| vals.get(a.as_str()).cloned().unwrap_or_else(|| literal(&mut e, f, a))).collect();
    let called = match intrinsics::call(&mut e, &f.func, &args) {
        Ok(Some(c)) => c,
        Ok(None) => return Err("not an intrinsic".into()),
        Err(err) => return Err(format!("{err}")),
    };
    if called.ret.ty != hty(&f.ret) {
        return Err(format!("result type {:?}, expected {}", called.ret.ty, f.ret));
    }
    store(&mut e, 0, &called.ret);
    if called.outs.len() != f.outs.len() {
        return Err(format!("{} out values, expected {}", called.outs.len(), f.outs.len()));
    }
    for (n, (index, v)) in called.outs.iter().enumerate() {
        if f.args[*index] != format!("o{n}") || v.ty != hty(&f.outs[n]) {
            return Err(format!("out {n}: argument {index} of type {:?}", v.ty));
        }
        store(&mut e, 1 + n, v);
    }
    e.body.ret(e.bb);
    Ok(irprint::function(&e.m, &e.body))
}

// ---- the normaliser ----

#[derive(Clone)]
enum Opnd {
    Node(usize),
    /// a constant or an input (`in0_2`)
    Text(String),
}

/// An operation: text pieces around its operands (one more piece than operands).
struct Node {
    pieces: Vec<String>,
    ops: Vec<Opnd>,
}

#[derive(Default)]
struct Canon<'a> {
    /// `%N` → the instruction text that defines it
    defs: HashMap<&'a str, &'a str>,
    memo: HashMap<&'a str, Opnd>,
    nodes: Vec<Node>,
    /// the fully expanded text of each node: its identity
    exprs: Vec<String>,
    by_expr: HashMap<String, usize>,
}

fn is_ref(t: &str) -> bool {
    t.len() > 1 && t.starts_with('%') && t[1..].bytes().all(|b| b.is_ascii_digit())
}

const COMMUTATIVE: &[&str] = &["add", "mul", "and", "or", "xor", "fadd", "fmul"];
const BINARY: &[&str] = &["add", "sub", "mul", "udiv", "sdiv", "urem", "srem", "shl", "lshr", "ashr", "and", "or", "xor", "fadd", "fsub", "fmul", "fdiv", "frem"];

impl<'a> Canon<'a> {
    fn expr<'b>(&'b self, o: &'b Opnd) -> &'b str {
        match o {
            Opnd::Node(n) => &self.exprs[*n],
            Opnd::Text(t) => t,
        }
    }

    fn intern(&mut self, pieces: Vec<String>, ops: Vec<Opnd>) -> Opnd {
        let mut key = String::new();
        for (i, p) in pieces.iter().enumerate() {
            key += p;
            if let Some(o) = ops.get(i) {
                let _ = write!(key, "({})", self.expr(o));
            }
        }
        if let Some(&n) = self.by_expr.get(&key) {
            return Opnd::Node(n);
        }
        self.nodes.push(Node { pieces, ops });
        self.exprs.push(key.clone());
        self.by_expr.insert(key, self.nodes.len() - 1);
        Opnd::Node(self.nodes.len() - 1)
    }

    fn operand(&mut self, t: &'a str) -> Result<Opnd, String> {
        if !is_ref(t) {
            return Ok(Opnd::Text(t.to_string()));
        }
        if let Some(o) = self.memo.get(t) {
            return Ok(o.clone());
        }
        let text = *self.defs.get(t).ok_or_else(|| format!("{t} is not defined"))?;
        let o = self.node(text)?;
        self.memo.insert(t, o.clone());
        Ok(o)
    }

    /// The node of one instruction (the text after `%N = `).
    fn node(&mut self, text: &'a str) -> Result<Opnd, String> {
        let (body, precise) = match text.find(", !dx.precise") {
            Some(at) => (&text[..at], true),
            None => (text, false),
        };
        let (op, rest) = body.split_once(' ').unwrap_or((body, ""));
        if BINARY.contains(&op) {
            // flags the optimiser infers say nothing about the operation
            let mut rest = rest;
            while let Some(r) = ["nsw ", "nuw ", "exact "].iter().find_map(|f| rest.strip_prefix(f)) {
                rest = r;
            }
            let (ty, ab) = rest.split_once(' ').ok_or(text)?;
            let (a, b) = ab.split_once(", ").ok_or(text)?;
            let (mut a, mut b) = (self.operand(a)?, self.operand(b)?);
            if COMMUTATIVE.contains(&op) && self.expr(&a) > self.expr(&b) {
                std::mem::swap(&mut a, &mut b);
            }
            return Ok(self.intern(vec![format!("{op} {ty} "), ", ".into(), String::new()], vec![a, b]));
        }
        if op == "icmp" || op == "fcmp" {
            let (pred, rest) = rest.split_once(' ').ok_or(text)?;
            let (ty, ab) = rest.split_once(' ').ok_or(text)?;
            let (a, b) = ab.split_once(", ").ok_or(text)?;
            // the round trips of a bool through its i32 form that -O0 leaves in
            if pred == "ne" && ty == "i1" && b == "false" {
                return self.operand(a);
            }
            if pred == "ne" && ty == "i32" && b == "0" && is_ref(a) {
                if let Some(inner) = self.defs.get(a).and_then(|d| d.strip_prefix("zext i1 ")).and_then(|d| d.strip_suffix(" to i32")) {
                    return self.operand(inner);
                }
            }
            let (mut a, mut b) = (self.operand(a)?, self.operand(b)?);
            // a > b is b < a; an (in)equality has no order
            let pred = match pred {
                "sgt" => "slt<",
                "sge" => "sle<",
                "ugt" if op == "icmp" => "ult<",
                "uge" if op == "icmp" => "ule<",
                "ogt" => "olt<",
                "oge" => "ole<",
                "ugt" => "ult<",
                "uge" => "ule<",
                p => p,
            };
            let pred = match pred.strip_suffix('<') {
                Some(p) => {
                    std::mem::swap(&mut a, &mut b);
                    p
                }
                None => pred,
            };
            if ["eq", "ne", "oeq", "one", "ueq", "une", "ord", "uno"].contains(&pred) && self.expr(&a) > self.expr(&b) {
                std::mem::swap(&mut a, &mut b);
            }
            return Ok(self.intern(vec![format!("{op} {pred} {ty} "), ", ".into(), String::new()], vec![a, b]));
        }
        // anything else: the text with its value operands cut out
        let (mut pieces, mut ops, mut last) = (Vec::new(), Vec::new(), 0);
        let bytes = body.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() && (i == 0 || bytes[i - 1] == b' ' || bytes[i - 1] == b'(') {
                let mut j = i + 1;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                pieces.push(body[last..i].to_string());
                ops.push(self.operand(&body[i..j])?);
                last = j;
                i = j;
            } else {
                i += 1;
            }
        }
        let mut tail = body[last..].to_string();
        if precise {
            tail += " !precise";
        }
        pieces.push(tail);
        Ok(self.intern(pieces, ops))
    }
}

/// The argument list of a call instruction.
fn call_args(text: &str) -> Vec<&str> {
    let open = text.find('(').map_or(0, |i| i + 1);
    let close = text.rfind(')').unwrap_or(text.len());
    text[open..close].split(", ").collect()
}

/// Reduce the text of a one-block function (DXC's listing of a fixture shader, or
/// `irprint::function` of the `Emit` build) to the operations behind what it stores.
fn canonical(text: &str) -> Result<String, String> {
    let mut c = Canon::default();
    // name → (offset of the rawBufferLoad)
    let mut loads: HashMap<&str, u32> = HashMap::new();
    // what an alloca (or an element of one) holds
    let mut slots: HashMap<String, Opnd> = HashMap::new();
    let mut geps: HashMap<&str, String> = HashMap::new();
    let mut sinks: Vec<(u32, u32, String, Opnd)> = Vec::new();
    let mut effects: Vec<Opnd> = Vec::new();
    let slot_key = |geps: &HashMap<&str, String>, p: &str| geps.get(p).cloned().unwrap_or_else(|| p.to_string());
    for line in text.lines() {
        let line = line.split(';').next().unwrap().trim();
        // DXC's attribute group reference on a call
        let line = match line.rfind(" #") {
            Some(at) if line[at + 2..].bytes().all(|b| b.is_ascii_digit()) => line[..at].trim_end(),
            _ => line,
        };
        if line.is_empty() || line.ends_with(':') || line.starts_with("ret ") || line.starts_with("define ") || line == "}" {
            continue;
        }
        if let Some((name, inst)) = line.split_once(" = ").filter(|(n, _)| is_ref(n)) {
            if inst.contains("@dx.op.rawBufferLoad.") {
                let off = call_args(inst)[2].strip_prefix("i32 ").and_then(|o| o.parse().ok()).ok_or(line)?;
                loads.insert(name, off);
            } else if let Some(rest) = inst.strip_prefix("extractvalue %dx.types.ResRet.") {
                let (agg, index) = rest.split_once(' ').and_then(|(_, r)| r.split_once(", ")).ok_or(line)?;
                let off = *loads.get(agg).ok_or(line)?;
                c.memo.insert(name, Opnd::Text(format!("in{}_{index}", off / 64)));
            } else if let Some(rest) = inst.strip_prefix("load ") {
                let ptr = rest.split(", ").nth(1).and_then(|p| p.split_once("* ")).map(|p| p.1).ok_or(line)?;
                if let Some(g) = ptr.strip_prefix("@in") {
                    c.memo.insert(name, Opnd::Text(format!("in{g}")));
                } else if let Some(v) = slots.get(&slot_key(&geps, ptr)) {
                    c.memo.insert(name, v.clone());
                }
            } else if let Some(rest) = inst.strip_prefix("getelementptr ") {
                let mut p = rest.split(", ").skip(1);
                let base = p.next().and_then(|b| b.rsplit(' ').next()).ok_or(line)?;
                let index: Vec<&str> = p.collect();
                geps.insert(name, format!("{}[{}]", slot_key(&geps, base), index.join(",")));
            } else {
                c.defs.insert(name, inst);
            }
        } else if let Some(rest) = line.strip_prefix("store ") {
            let mut p = rest.split(", ");
            let (value, ptr) = (p.next().ok_or(line)?, p.next().and_then(|p| p.split_once("* ")).map(|p| p.1).ok_or(line)?);
            let (ty, v) = value.split_once(' ').ok_or(line)?;
            let v = c.operand(v)?;
            match ptr.strip_prefix("@out").and_then(|g| g.split_once('_')) {
                Some((sink, comp)) => sinks.push((sink.parse().map_err(|_| line)?, comp.parse().map_err(|_| line)?, ty.to_string(), v)),
                None => {
                    slots.insert(slot_key(&geps, ptr), v);
                }
            }
        } else if line.contains("@dx.op.rawBufferStore.") {
            let args = call_args(line);
            let off: u32 = args[2].strip_prefix("i32 ").and_then(|o| o.parse().ok()).ok_or(line)?;
            for (j, a) in args[4..8].iter().enumerate() {
                let (ty, v) = a.split_once(' ').ok_or(line)?;
                if v != "undef" {
                    let v = c.operand(v)?;
                    sinks.push(((off - 1024) / 64, j as u32, ty.to_string(), v));
                }
            }
        } else if line.starts_with("call void @dx.op.") {
            effects.push(c.node(line)?);
        } else if !line.contains("@llvm.lifetime.") {
            return Err(format!("unexpected line: {line}"));
        }
    }
    sinks.sort_by_key(|s| (s.0, s.1));
    // number the operations reachable from the effects and the stored values, operands first
    let (mut number, mut out) = (HashMap::new(), String::new());
    fn show(c: &Canon, n: usize, number: &mut HashMap<usize, usize>, out: &mut String) {
        if number.contains_key(&n) {
            return;
        }
        let node = &c.nodes[n];
        for o in &node.ops {
            if let Opnd::Node(m) = o {
                show(c, *m, number, out);
            }
        }
        let mut line = String::new();
        for (i, p) in node.pieces.iter().enumerate() {
            line += p;
            match node.ops.get(i) {
                Some(Opnd::Node(m)) => {
                    let _ = write!(line, "%{}", number[m]);
                }
                Some(Opnd::Text(t)) => line += t,
                None => {}
            }
        }
        let id = number.len() + 1;
        number.insert(n, id);
        if line.starts_with("call void ") {
            let _ = writeln!(out, "  {line}");
        } else {
            let _ = writeln!(out, "  %{id} = {line}");
        }
    }
    for o in effects.iter().chain(sinks.iter().map(|s| &s.3)) {
        if let Opnd::Node(n) = o {
            show(&c, *n, &mut number, &mut out);
        }
    }
    for (sink, comp, ty, v) in &sinks {
        let v = match v {
            Opnd::Node(n) => format!("%{}", number[n]),
            Opnd::Text(t) => t.clone(),
        };
        let _ = writeln!(out, "  out{sink}_{comp} = {ty} {v}");
    }
    Ok(out)
}

// ---- goldens ----

fn goldens() -> HashMap<&'static str, String> {
    let mut map = HashMap::new();
    for block in intrinsics_golden::GOLDEN.split("=== ").skip(1) {
        let (name, body) = block.split_once('\n').unwrap();
        map.insert(name.trim(), body.to_string());
    }
    map
}

#[test]
fn emit_matches_dxc_goldens() {
    let golden = goldens();
    let (mut failed, mut report) = (0, String::new());
    let all = fixtures();
    for f in &all {
        let got = emit(f).and_then(|ir| canonical(&ir).map_err(|e| format!("{e}\n{ir}")));
        let want = golden.get(f.name.as_str());
        if want.is_some() && got.as_ref().ok() == want {
            continue;
        }
        failed += 1;
        let _ = writeln!(report, "=== {}: {}({})\n--- DXC\n{}--- ours\n{}", f.name, f.func, f.args.join(", "), want.map_or("(no golden)\n", |w| w.as_str()), got.unwrap_or_else(|e| format!("error: {e}\n")));
    }
    assert!(failed == 0, "{failed} of {} fixtures differ from DXC:\n{report}", all.len());
}

/// Run DXC on a fixture and normalise its listing.
fn dxc(f: &Fixture, root: &str, dir: &std::path::Path) -> Result<String, String> {
    let (src, ll) = (dir.join(format!("{}.hlsl", f.name)), dir.join(format!("{}.ll", f.name)));
    std::fs::write(&src, hlsl(f)).map_err(|e| e.to_string())?;
    let mut cmd = std::process::Command::new(format!("{root}/bin/dxc"));
    cmd.env("LD_LIBRARY_PATH", format!("{root}/lib"));
    cmd.args(["-E", "main", "-T", &format!("{}_6_6", f.stage), "-HV", "2021", "-Gis", if f.o0 { "-O0" } else { "-O1" }]);
    if f.stage == "ms" {
        // the fixture writes no vertex: the validator would object to that, not to the call
        cmd.arg("-Vd");
    }
    let out = cmd.arg("-Fc").arg(&ll).arg(&src).output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    let text = std::fs::read_to_string(&ll).map_err(|e| e.to_string())?;
    let body = text.split_once("define void @main() {").and_then(|(_, b)| b.split_once("\n}")).ok_or("no main")?.0;
    canonical(body).map_err(|e| format!("{e}\n{body}"))
}

#[test]
fn goldens_are_what_dxc_emits() {
    let Ok(root) = std::env::var("XCL_DXIL_DXC") else {
        eprintln!("XCL_DXIL_DXC is not set: the goldens are not checked against DXC");
        return;
    };
    let dir = std::env::temp_dir().join(format!("xcl-dxil-intrinsics-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let all = fixtures();
    let mut results: Vec<Result<String, String>> = Vec::new();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    for chunk in all.chunks(all.len().div_ceil(threads)) {
        let (root, dir) = (&root, &dir);
        results.extend(std::thread::scope(|s| {
            let handles: Vec<_> = chunk.iter().map(|f| s.spawn(move || dxc(f, root, dir))).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
        }));
    }
    let _ = std::fs::remove_dir_all(&dir);
    if std::env::var("XCL_DXIL_BLESS").is_ok() {
        let mut text = String::from("//! DXC 1.9's normalised output for the fixtures of intrinsics_oracle.rs. Generated: run that\n//! test with XCL_DXIL_DXC set and XCL_DXIL_BLESS=1 to rewrite it.\n#![allow(dead_code)]\n\npub const GOLDEN: &str = r#\"\n");
        for (f, r) in all.iter().zip(&results) {
            let body = r.as_ref().unwrap_or_else(|e| panic!("{}: {e}", f.name));
            let _ = write!(text, "=== {}\n{body}", f.name);
        }
        text += "\"#;\n";
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/intrinsics_golden.rs"), text).unwrap();
        return;
    }
    let golden = goldens();
    let mut report = String::new();
    for (f, r) in all.iter().zip(&results) {
        match (r, golden.get(f.name.as_str())) {
            (Ok(got), Some(want)) if got == want => {}
            (r, want) => {
                let _ = writeln!(report, "=== {}\n--- golden\n{}--- DXC\n{}", f.name, want.map_or("(none)\n", |w| w.as_str()), r.clone().unwrap_or_else(|e| format!("error: {e}\n")));
            }
        }
    }
    assert!(report.is_empty(), "goldens differ from DXC:\n{report}");
}

/// Constant arguments over a sweep of values, against DXC's two evaluators: the front end
/// (all that folds at -O0) and constant propagation (-O1). Whatever folds here must be the
/// constant DXC computes; what DXC evaluates with its C library may stay a call here.
#[test]
fn constant_folds_agree_with_dxc() {
    let Ok(root) = std::env::var("XCL_DXIL_DXC") else {
        eprintln!("XCL_DXIL_DXC is not set: constant folds are not checked against DXC");
        return;
    };
    const FLOATS: &[&str] = &[
        "0.0f", "-0.0f", "1.0f", "2.0f", "3.0f", "4.0f", "8.0f", "0.1f", "0.25f", "0.5f", "1.5f", "2.5f", "3.5f", "-1.5f", "-2.5f", "-14.0f", "7.0f", "10.0f", "100.0f", "0.001f",
        "1.0e10f", "123456.789f", "16777217.0f", "3.4e38f", "1.0e-40f", "127.0f", "128.0f", "-126.0f", "-127.0f", "-149.0f", "-150.0f", "0.75f", "-0.75f", "1024.0f", "5.0f", "6.0f",
    ];
    const UINTS: &[&str] = &["0u", "1u", "5u", "40u", "0x100u", "0xF0F0u", "0x7FFFFFFFu", "0x80000000u", "0xFFFFFFFFu"];
    let mut cases: Vec<Fixture> = Vec::new();
    let mut case = |ret: &str, func: &str, args: &[&str]| {
        for o0 in [false, true] {
            cases.push(Fixture { name: format!("sweep{}", cases.len()), inputs: Vec::new(), ret: ret.into(), func: func.into(), args: args.iter().map(|a| a.to_string()).collect(), outs: Vec::new(), stage: "cs", o0 });
        }
    };
    for (i, x) in FLOATS.iter().enumerate() {
        for func in ["exp2", "log2", "sqrt", "rsqrt", "floor", "ceil", "round", "trunc", "frac", "abs", "saturate"] {
            case("float", func, &[x]);
        }
        case("bool", "isnan", &[x]);
        let (y, z) = (FLOATS[(i + 7) % FLOATS.len()], FLOATS[(i + 13) % FLOATS.len()]);
        case("float", "min", &[x, y]);
        case("float", "max", &[x, y]);
        case("float", "mad", &[x, y, z]);
        case("float", "clamp", &[x, y, z]);
    }
    for (i, x) in UINTS.iter().enumerate() {
        for func in ["reversebits", "countbits", "firstbitlow", "firstbithigh", "abs"] {
            case("uint", func, &[x]);
        }
        let (y, z) = (UINTS[(i + 2) % UINTS.len()], UINTS[(i + 5) % UINTS.len()]);
        case("uint", "min", &[x, y]);
        case("uint", "max", &[x, y]);
        case("uint", "mad", &[x, y, z]);
        case("uint", "clamp", &[x, y, z]);
    }
    let dir = std::env::temp_dir().join(format!("xcl-dxil-intrinsics-sweep-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut results: Vec<Result<String, String>> = Vec::new();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    for chunk in cases.chunks(4 * threads) {
        let (root, dir) = (&root, &dir);
        results.extend(std::thread::scope(|s| {
            let handles: Vec<_> = chunk.iter().map(|f| s.spawn(move || dxc(f, root, dir))).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
        }));
    }
    let _ = std::fs::remove_dir_all(&dir);
    let (mut folded, mut report) = (0, String::new());
    for (f, theirs) in cases.iter().zip(&results) {
        let call = format!("{}({}) at {}", f.func, f.args.join(", "), if f.o0 { "-O0" } else { "-O1" });
        let theirs = theirs.as_ref().unwrap_or_else(|e| panic!("{call}: {e}"));
        let ours = emit(f).and_then(|ir| canonical(&ir));
        let ours = match ours {
            Ok(text) => text,
            // DXC's evaluators disagree: the call is left to DXC
            Err(_) if f.func == "clamp" => continue,
            Err(e) => panic!("{call}: {e}"),
        };
        if ours.contains('%') {
            continue;
        }
        folded += 1;
        // at -O0 DXC folds what its front end evaluates and leaves the rest as calls
        if ours != *theirs && !(f.o0 && theirs.contains('%')) {
            let _ = writeln!(report, "{call}
--- DXC
{theirs}--- ours
{ours}");
        }
    }
    assert!(report.is_empty(), "constant folds differ from DXC:
{report}");
    assert!(folded * 2 > cases.len(), "only {folded} of {} constant calls folded", cases.len());
    eprintln!("{folded} of {} constant calls fold, all to DXC's constant", cases.len());
}

// ---- what the goldens cannot show ----

fn scalar(e: &mut Emit, k: Sk) -> Val {
    let g = e.m.global(&format!("x{}", e.m.globals.len()), k.ir(), 0, None, false);
    Val::scalar(k, e.body.load(e.bb, k.ir(), g))
}

fn unsupported(name: &str, kinds: &[Sk]) -> bool {
    let mut e = Emit::new();
    let args: Vec<Val> = kinds.iter().map(|&k| scalar(&mut e, k)).collect();
    matches!(intrinsics::call(&mut e, name, &args), Err(Error::Unsupported(_)))
}

/// Forms DXC rejects or evaluates in a way this backend cannot reproduce exactly fall back.
#[test]
fn forms_left_to_dxc() {
    // DXC: no matching function
    assert!(unsupported("asuint", &[Sk::Double]));
    assert!(unsupported("asuint", &[Sk::Uint64]));
    assert!(unsupported("asfloat", &[Sk::Bool]));
    assert!(unsupported("fma", &[Sk::Float, Sk::Float, Sk::Float]));
    assert!(unsupported("WaveActiveBitAnd", &[Sk::Int]));
    // DXC's own output fails validation (no double overload of Dot)
    let mut e = Emit::new();
    let (a, b) = (scalar(&mut e, Sk::Double), scalar(&mut e, Sk::Double));
    let v = |x: &Val| Val::new(HTy::vec(Sk::Double, 2), &[x.one(), x.one()]);
    assert!(matches!(intrinsics::call(&mut e, "dot", &[v(&a), v(&b)]), Err(Error::Unsupported(_))));
    // a vector lane index
    let lane = Val::new(HTy::vec(Sk::Uint, 2), &[a.one(), a.one()]);
    assert!(matches!(intrinsics::call(&mut e, "WaveReadLaneAt", &[a.clone(), lane]), Err(Error::Unsupported(_))));
    // evaluated at compile time in literal precision
    let lit = Val::scalar(Sk::LitFloat, e.m.const_f64(2.0));
    assert!(matches!(intrinsics::call(&mut e, "sqrt", &[lit.clone()]), Err(Error::Unsupported(_))));
    assert!(matches!(intrinsics::call(&mut e, "pow", &[lit.clone(), lit.clone()]), Err(Error::Unsupported(_))));
    let one = Val::scalar(Sk::LitInt, e.c_u32(1));
    assert!(matches!(intrinsics::call(&mut e, "WaveActiveSum", &[one]), Err(Error::Unsupported(_))));
    // a wrong argument count is an error in the source
    assert!(matches!(intrinsics::call(&mut e, "sqrt", &[a.clone(), b.clone()]), Err(Error::Source { .. })));
    // not intrinsics, or the lowering's own
    for name in ["frexp_mant", "NonUniformResourceIndex", "InterlockedAdd", "InterlockedCompareExchange"] {
        assert!(matches!(intrinsics::call(&mut e, name, &[]), Ok(None)), "{name}");
    }
    assert!(unsupported("EvaluateAttributeCentroid", &[Sk::Float]));
}

/// Constant arguments fold only to results that do not depend on a C library.
#[test]
fn inexact_constant_calls_stay_calls() {
    let mut e = Emit::new();
    let f = |e: &mut Emit, v: f32| Val::scalar(Sk::Float, e.c_f32(v));
    let call = |e: &mut Emit, name: &str, args: &[Val]| intrinsics::call(e, name, args).map(|c| c.unwrap().ret.one());
    for (name, x, folds) in [("exp2", -14.0, true), ("exp2", 0.5, false), ("log2", 8.0, true), ("log2", 3.0, false), ("sin", 1.0, false), ("cos", 1.0, false), ("sqrt", 2.0, true), ("floor", -1.5, true)] {
        let x = f(&mut e, x);
        let r = call(&mut e, name, &[x.clone()]).unwrap();
        assert_eq!(e.m.as_const(r).is_some(), folds, "{name}");
    }
    let x = f(&mut e, -14.0);
    let r = call(&mut e, "exp2", &[x]).unwrap();
    assert_eq!(e.m.as_float_bits(r), Some(0x3880_0000));
    // DXC never evaluates the half conversions or the classification of a NaN
    let h = Val::scalar(Sk::Uint, e.c_u32(0x3BFF));
    let r = call(&mut e, "f16tof32", &[h]).unwrap();
    assert!(e.m.as_const(r).is_none());
    let nan = Val::scalar(Sk::Float, e.m.const_f32_bits(0x7FC0_0000));
    let r = call(&mut e, "isnan", &[nan]).unwrap();
    assert!(e.m.as_const(r).is_none());
    // DXC's front end and its lowering give a constant clamp with lo > hi different values
    let (x, lo, hi) = (f(&mut e, 0.0), f(&mut e, 2.0), f(&mut e, 1.0));
    assert!(matches!(call(&mut e, "clamp", &[x.clone(), lo.clone(), hi.clone()]), Err(Error::Unsupported(_))));
    let r = call(&mut e, "clamp", &[x, hi, lo]).unwrap();
    assert_eq!(e.m.as_float_bits(r), Some(1f32.to_bits() as u64));
    let v = scalar(&mut e, Sk::Float);
    let (lo, hi) = (f(&mut e, 2.0), f(&mut e, 1.0));
    assert!(call(&mut e, "clamp", &[v, lo, hi]).is_ok());
}

/// Literal arguments DXC evaluates stay literals of the value DXC computes.
#[test]
fn literal_results() {
    let mut e = Emit::new();
    let int = |e: &mut Emit, v: i32| Val::scalar(Sk::LitInt, e.c_u32(v as u32));
    let flt = |e: &mut Emit, v: f64| Val::scalar(Sk::LitFloat, e.m.const_f64(v));
    let (three, five, neg) = (int(&mut e, 3), int(&mut e, 5), int(&mut e, -2));
    let r = intrinsics::call(&mut e, "min", &[three.clone(), five.clone()]).unwrap().unwrap().ret;
    assert_eq!((r.ty, e.m.as_int(r.one())), (HTy::scalar(Sk::LitInt), Some(3)));
    let r = intrinsics::call(&mut e, "max", &[neg.clone(), five.clone()]).unwrap().unwrap().ret;
    assert_eq!(e.m.as_int(r.one()), Some(5));
    let r = intrinsics::call(&mut e, "abs", &[neg.clone()]).unwrap().unwrap().ret;
    assert_eq!((r.ty, e.m.as_int(r.one())), (HTy::scalar(Sk::LitInt), Some(2)));
    let r = intrinsics::call(&mut e, "clamp", &[five.clone(), neg, three]).unwrap().unwrap().ret;
    assert_eq!(e.m.as_int(r.one()), Some(3));
    // 1.1 is not a float: the literal survives at the precision it came in
    let (a, b) = (flt(&mut e, 1.1), flt(&mut e, 2.2));
    let r = intrinsics::call(&mut e, "min", &[a.clone(), b.clone()]).unwrap().unwrap().ret;
    assert_eq!((r.ty, e.m.as_float_bits(r.one())), (HTy::scalar(Sk::LitFloat), Some(1.1f64.to_bits())));
    let m = flt(&mut e, -2.5);
    let r = intrinsics::call(&mut e, "abs", &[m]).unwrap().unwrap().ret;
    assert_eq!(e.m.as_float_bits(r.one()), Some(2.5f64.to_bits()));
    // a select between literals is still a literal
    let c = scalar(&mut e, Sk::Uint);
    let r = intrinsics::call(&mut e, "select", &[c, a, b]).unwrap().unwrap().ret;
    assert_eq!(r.ty, HTy::scalar(Sk::LitFloat));
}

/// The fixture shaders keep the shape the normaliser reads.
#[test]
fn fixture_shader_text() {
    let all = fixtures();
    let f = all.iter().find(|f| f.name == "modf_f3").unwrap();
    assert_eq!(
        hlsl(f),
        "RWByteAddressBuffer io : register(u0);\ntemplate<typename T> void put(uint off, T v) { io.Store<T>(off, v); }\n[numthreads(1, 1, 1)]\nvoid main() {\n    float3 a = (io.Load<float3>(0));\n    float3 o0;\n    put(1024, modf(a, o0));\n    put(1088, o0);\n}\n"
    );
    let f = all.iter().find(|f| f.name == "mul_mv").unwrap();
    assert!(hlsl(f).contains("float2x2 a = float2x2(io.Load<float4>(0));"));
    let f = all.iter().find(|f| f.name == "select_s").unwrap();
    assert!(hlsl(f).contains("bool c = isnan(io.Load<float>(0));"));
    let mut names: Vec<&str> = all.iter().map(|f| f.name.as_str()).collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), all.len(), "fixture names are unique");
}
