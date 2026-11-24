//! HLSL intrinsic functions → dx.op calls and LLVM instructions, as DXC 1.9 lowers them
//! (HLOperationLower.cpp and, for matrix `mul`, HLMatrixLowerPass.cpp), under -Gis rules.
//! Owner: the intrinsics subagent.
//!
//! [`call`] does what DXC's Sema and lowering do together for one call: it picks the overload
//! from the argument types (the template classes of gen_intrin_main.txt: `numeric`,
//! `float_like`, `any_int`, ...), converts the arguments, and emits the expansion per scalar
//! component, in the operand order of DXC's source. Expansions are the unoptimised ones.
//!
//! Constants: DXC evaluates a call whose arguments are constants (`exp2(-14.0f)`,
//! `reversebits(40u)`), so a dx.op with constant operands folds here too, but only where the
//! result is exact: see [`fold`]. `sin(1.0f)` stays a call, which DXC would have replaced by
//! what its C library computes. Bit casts of constants fold (`asfloat(0x7F800000u)`,
//! `asdouble(lo, hi)`), and so do unsuffixed literal arguments where the result is one of
//! the arguments (`min`, `max`, `clamp`, `abs`).
//!
//! Three things DXC does that look wrong and are matched on purpose:
//! * `float_like` intrinsics (`sqrt`, `isnan`, `floor`, `lerp`, ...) narrow a `double` argument to
//!   float and return float; only `numeric` / `any_float` ones (`abs`, `min`, `max`, `clamp`,
//!   `mad`, `saturate`, `rcp`, `sign`) and `fma` work in double,
//! * `numeric` intrinsics given only `bool` arguments work in float (`abs(bool)` is `FAbs` of a
//!   `uitofp`),
//! * mixed signed / unsigned integer arguments select the unsigned operation.
//!
//! A call whose templated arguments are all unsuffixed literals is evaluated by DXC at
//! compile time in literal precision (double, 64-bit int). Apart from the exact cases above
//! that is [`Error::Unsupported`] here: a transcendental folded on the host is not bit-exact.
//!
//! Not here, by contract with the lowering: `NonUniformResourceIndex` (a marker on the heap
//! index: the lowering sets `HTy::HeapElem::nonuniform` before any call), the free-function
//! `Interlocked*` forms (they need the lvalue of their first argument) and `discard` (a
//! statement). [`call`] returns `Ok(None)` for those names.

use super::emit::Emit;
use super::types::{common_sk, HTy, Sk, Val};
use crate::consts::OpCode;
use crate::ir::{BinOp, CastOp, Const, FPred, IPred, Ty, Value};
use crate::{Error, Result};

mod fold;
use fold::dx;

/// The result of an intrinsic call.
pub struct Called {
    pub ret: Val,
    /// values of `out` arguments: (argument index, value)
    pub outs: Vec<(usize, Val)>,
}

/// Lower a call of intrinsic `name` with already evaluated arguments (an `out` argument is
/// passed as its current value and ignored). Ok(None): `name` is not an intrinsic handled
/// here (the caller reports it). Arguments are converted here (overload resolution).
///
/// The value of an `out` argument has the type the intrinsic produces (`modf`'s integer part
/// is a float of the first argument's shape); converting it to the variable's type is the
/// caller's assignment.
pub fn call(e: &mut Emit, name: &str, args: &[Val]) -> Result<Option<Called>> {
    use OpCode as O;
    let mut c = Cx { e, name, args };
    let called = match name {
        // the hot ones first: the translator emits these tens of thousands of times a program
        "select" => c.select()?,
        "asfloat" => c.reinterpret(Sk::Float)?,
        "asuint" if args.len() == 3 => c.split_double()?,
        "asuint" => c.reinterpret(Sk::Uint)?,
        "and" => c.logical(BinOp::And)?,
        "or" => c.logical(BinOp::Or)?,
        "mad" => c.mad()?,
        "min" => c.min_max(false)?,
        "max" => c.min_max(true)?,
        "isnan" => c.special_float(O::IsNaN)?,
        "abs" => c.abs()?,
        "rcp" => c.rcp()?,
        "saturate" => c.saturate()?,
        "countbits" => c.bit_count(O::Countbits)?,
        "exp2" => c.float1(O::Exp)?,
        "f16tof32" => c.f16_to_f32()?,
        "f32tof16" => c.f32_to_f16()?,
        "rsqrt" => c.float1(O::Rsqrt)?,
        "sqrt" => c.float1(O::Sqrt)?,

        "asint" => c.reinterpret(Sk::Int)?,
        "asdouble" => c.make_double()?,
        "clamp" => c.clamp()?,
        "sign" => c.sign()?,
        "fma" => c.fma()?,
        "any" => c.any_all(BinOp::Or)?,
        "all" => c.any_all(BinOp::And)?,
        "isinf" => c.special_float(O::IsInf)?,
        "isfinite" => c.special_float(O::IsFinite)?,
        "isnormal" => c.special_float(O::IsNormal)?,

        "sin" => c.float1(O::Sin)?,
        "cos" => c.float1(O::Cos)?,
        "tan" => c.float1(O::Tan)?,
        "asin" => c.float1(O::Asin)?,
        "acos" => c.float1(O::Acos)?,
        "atan" => c.float1(O::Atan)?,
        "sinh" => c.float1(O::Hsin)?,
        "cosh" => c.float1(O::Hcos)?,
        "tanh" => c.float1(O::Htan)?,
        "log2" => c.float1(O::Log)?,
        "floor" => c.float1(O::Round_ni)?,
        "ceil" => c.float1(O::Round_pi)?,
        "round" => c.float1(O::Round_ne)?,
        "trunc" => c.float1(O::Round_z)?,
        "frac" => c.float1(O::Frc)?,
        "ddx" | "ddx_coarse" => c.float1(O::DerivCoarseX)?,
        "ddy" | "ddy_coarse" => c.float1(O::DerivCoarseY)?,
        "ddx_fine" => c.float1(O::DerivFineX)?,
        "ddy_fine" => c.float1(O::DerivFineY)?,
        "exp" => c.exp()?,
        "log" => c.log(std::f64::consts::LN_2)?,
        "log10" => c.log(std::f64::consts::LN_2 / std::f64::consts::LN_10)?,
        "degrees" => c.scale(180.0 / std::f64::consts::PI)?,
        "radians" => c.scale(std::f64::consts::PI / 180.0)?,
        "pow" => c.pow()?,
        "atan2" => c.atan2()?,
        "sincos" => c.sincos()?,
        "fmod" => c.fmod()?,
        "modf" => c.modf()?,
        "frexp" => c.frexp()?,
        "ldexp" => c.ldexp()?,
        "lerp" => c.lerp()?,
        "step" => c.step()?,
        "smoothstep" => c.smoothstep()?,
        "fwidth" => c.fwidth()?,

        "dot" => c.dot()?,
        "cross" => c.cross()?,
        "length" => c.length()?,
        "distance" => c.distance()?,
        "normalize" => c.normalize()?,
        "reflect" => c.reflect()?,
        "refract" => c.refract()?,
        "faceforward" => c.faceforward()?,
        "mul" => c.mul()?,
        "dot4add_u8packed" => c.dot4add(O::Dot4AddU8Packed, Sk::Uint)?,
        "dot4add_i8packed" => c.dot4add(O::Dot4AddI8Packed, Sk::Int)?,

        "reversebits" => c.reverse_bits()?,
        "firstbitlow" => c.bit_count(O::FirstbitLo)?,
        "firstbithigh" => c.first_bit_high()?,

        "WaveGetLaneIndex" => c.no_arg(O::WaveGetLaneIndex, Ty::VOID, Sk::Uint)?,
        "WaveGetLaneCount" => c.no_arg(O::WaveGetLaneCount, Ty::VOID, Sk::Uint)?,
        "WaveIsFirstLane" => c.no_arg(O::WaveIsFirstLane, Ty::VOID, Sk::Bool)?,
        "IsHelperLane" => c.no_arg(O::IsHelperLane, Ty::I1, Sk::Bool)?,
        "GetRenderTargetSampleCount" => c.no_arg(O::RenderTargetGetSampleCount, Ty::VOID, Sk::Uint)?,
        "WaveActiveAnyTrue" => c.wave_bool(O::WaveAnyTrue, Sk::Bool)?,
        "WaveActiveAllTrue" => c.wave_bool(O::WaveAllTrue, Sk::Bool)?,
        "WaveActiveCountBits" => c.wave_bool(O::WaveAllBitCount, Sk::Uint)?,
        "WavePrefixCountBits" => c.wave_bool(O::WavePrefixBitCount, Sk::Uint)?,
        "WaveActiveBallot" => c.wave_ballot()?,
        "WaveActiveAllEqual" => c.wave_all_equal()?,
        "WaveReadLaneAt" => c.read_lane_at(O::WaveReadLaneAt, Cls::Any)?,
        "WaveReadLaneFirst" => c.wave_read_first()?,
        "WaveActiveSum" => c.wave_op(O::WaveActiveOp, 0)?,
        "WaveActiveProduct" => c.wave_op(O::WaveActiveOp, 1)?,
        "WaveActiveMin" => c.wave_op(O::WaveActiveOp, 2)?,
        "WaveActiveMax" => c.wave_op(O::WaveActiveOp, 3)?,
        "WavePrefixSum" => c.wave_op(O::WavePrefixOp, 0)?,
        "WavePrefixProduct" => c.wave_op(O::WavePrefixOp, 1)?,
        "WaveActiveBitAnd" => c.wave_bit(0)?,
        "WaveActiveBitOr" => c.wave_bit(1)?,
        "WaveActiveBitXor" => c.wave_bit(2)?,
        "QuadReadLaneAt" => c.read_lane_at(O::QuadReadLaneAt, Cls::Numeric)?,
        "QuadReadAcrossX" => c.quad_across(0)?,
        "QuadReadAcrossY" => c.quad_across(1)?,
        "QuadReadAcrossDiagonal" => c.quad_across(2)?,
        "QuadAny" => c.quad_vote(BinOp::Or)?,
        "QuadAll" => c.quad_vote(BinOp::And)?,

        // DXIL::BarrierMode: SyncThreadGroup 1, UAVFenceGlobal 2, UAVFenceThreadGroup 4, TGSMFence 8
        "GroupMemoryBarrier" => c.barrier(8)?,
        "GroupMemoryBarrierWithGroupSync" => c.barrier(9)?,
        "DeviceMemoryBarrier" => c.barrier(2)?,
        "DeviceMemoryBarrierWithGroupSync" => c.barrier(3)?,
        "AllMemoryBarrier" => c.barrier(10)?,
        "AllMemoryBarrierWithGroupSync" => c.barrier(11)?,
        "clip" => c.clip()?,
        "GetRenderTargetSamplePosition" => c.rt_sample_position()?,
        "SetMeshOutputCounts" => c.set_mesh_output_counts()?,

        _ if OTHER_INTRINSICS.binary_search(&name).is_ok() => return c.unsupported("is not implemented"),
        _ => return Ok(None),
    };
    Ok(Some(called))
}

/// DXC's other free-function intrinsics (gen_intrin_main.txt, `namespace Intrinsics`), sorted:
/// known names this backend does not lower, so the caller falls back to DXC instead of
/// reporting an undeclared function. `EvaluateAttribute*` and `GetAttributeAtVertex` take an
/// input signature element, not a value, and `DispatchMesh` a payload variable: those belong to
/// the stage lowering if the dialect ever uses them.
const OTHER_INTRINSICS: &[&str] = &[
    "AcceptHitAndEndSearch",
    "AddUint64",
    "AllocateRayQuery",
    "Barrier",
    "CallShader",
    "CheckAccessFullyMapped",
    "ClusterID",
    "D3DCOLORtoUBYTE4",
    "DebugBreak",
    "DispatchMesh",
    "DispatchRaysDimensions",
    "DispatchRaysIndex",
    "EvaluateAttributeAtSample",
    "EvaluateAttributeCentroid",
    "EvaluateAttributeSnapped",
    "GeometryIndex",
    "GetAttributeAtVertex",
    "GetGroupWaveCount",
    "GetGroupWaveIndex",
    "GetRemainingRecursionLevels",
    "HitKind",
    "IgnoreHit",
    "InstanceID",
    "InstanceIndex",
    "ObjectRayDirection",
    "ObjectRayOrigin",
    "ObjectToWorld",
    "ObjectToWorld3x4",
    "ObjectToWorld4x3",
    "PrimitiveIndex",
    "Process2DQuadTessFactorsAvg",
    "Process2DQuadTessFactorsMax",
    "Process2DQuadTessFactorsMin",
    "ProcessIsolineTessFactors",
    "ProcessQuadTessFactorsAvg",
    "ProcessQuadTessFactorsMax",
    "ProcessQuadTessFactorsMin",
    "ProcessTriTessFactorsAvg",
    "ProcessTriTessFactorsMax",
    "ProcessTriTessFactorsMin",
    "RayFlags",
    "RayTCurrent",
    "RayTMin",
    "ReportHit",
    "TraceRay",
    "WaveMatch",
    "WaveMultiPrefixBitAnd",
    "WaveMultiPrefixBitOr",
    "WaveMultiPrefixBitXor",
    "WaveMultiPrefixCountBits",
    "WaveMultiPrefixProduct",
    "WaveMultiPrefixSum",
    "WorldRayDirection",
    "WorldRayOrigin",
    "WorldToObject",
    "WorldToObject3x4",
    "WorldToObject4x3",
    "abort",
    "asfloat16",
    "asint16",
    "asuint16",
    "determinant",
    "dot2add",
    "dst",
    "lit",
    "msad4",
    "pack_clamp_s8",
    "pack_clamp_u8",
    "pack_s8",
    "pack_u8",
    "printf",
    "source_mark",
    "tex1D",
    "tex1Dbias",
    "tex1Dgrad",
    "tex1Dlod",
    "tex1Dproj",
    "tex2D",
    "tex2Dbias",
    "tex2Dgrad",
    "tex2Dlod",
    "tex2Dproj",
    "tex3D",
    "tex3Dbias",
    "tex3Dgrad",
    "tex3Dlod",
    "tex3Dproj",
    "texCUBE",
    "texCUBEbias",
    "texCUBEgrad",
    "texCUBElod",
    "texCUBEproj",
    "transpose",
    "unpack_s8s16",
    "unpack_s8s32",
    "unpack_u8u16",
    "unpack_u8u32",
];

/// (rows, columns) of a numeric type; (1, 1) is a scalar.
type Shape = (u8, u8);

/// The scalar components of one converted argument or of a result, on the stack (a 4x4
/// matrix is the largest numeric value).
#[derive(Copy, Clone)]
struct Comps {
    n: usize,
    v: [Value; 16],
}

impl Comps {
    fn new(shape: Shape) -> Comps {
        Comps { n: (shape.0 as usize * shape.1 as usize).min(16), v: [Value::NONE; 16] }
    }
    fn get(&self) -> &[Value] {
        &self.v[..self.n]
    }
}

/// The component classes of DXC's intrinsic templates: which scalar kind the arguments
/// sharing a template parameter are converted to.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Cls {
    /// `any<>`: the common kind as it is, bool included
    Any,
    /// `numeric<>`: the common kind, float when every argument is bool
    Numeric,
    /// `float_like<>`: float, whatever comes in (double is narrowed)
    FloatLike,
    /// `any_float<>`: double stays double, everything else is float
    AnyFloat,
    /// `any_int<>`: the 32- and 64-bit integers; bool and float become int
    AnyInt,
}

const LITERALS: &str = "on literal arguments (DXC evaluates it at compile time)";

fn is_lit(k: Sk) -> bool {
    matches!(k, Sk::LitInt | Sk::LitFloat)
}
fn is_unsigned(k: Sk) -> bool {
    matches!(k, Sk::Uint | Sk::Uint64)
}

/// Sign-extend the low `bits` of `x`.
fn sext(x: u64, bits: u32) -> i64 {
    if bits == 0 || bits >= 64 {
        x as i64
    } else {
        ((x << (64 - bits)) as i64) >> (64 - bits)
    }
}

/// A float constant of kind `k` (in the IR type values of that kind have).
fn const_float(e: &mut Emit, k: Sk, x: f64) -> Value {
    if k.ir() == Ty::F64 {
        e.m.const_f64(x)
    } else {
        e.m.const_f32(x as f32)
    }
}

/// The value of a float constant of either width.
fn float_const(e: &Emit, v: Value) -> Option<f64> {
    let c = e.m.as_const(v)?;
    match (c.ty, &c.c) {
        (Ty::F32, Const::Float(b)) => Some(f32::from_bits(*b as u32) as f64),
        (Ty::F64, Const::Float(b)) => Some(f64::from_bits(*b)),
        _ => None,
    }
}

/// Convert a literal constant to kind `to` exactly, from the width the constant was kept in:
/// a literal against a double must not go through float (`clamp(d, 0.0, 4294967295.0)`).
/// None: not a literal constant, or a conversion whose result the target decides.
fn fold_literal(e: &mut Emit, v: Value, from: Sk, to: Sk) -> Option<Value> {
    let c = e.m.as_const(v)?;
    let (cty, cc) = (c.ty, c.c.clone());
    let ir = to.ir();
    match (from, cc) {
        (Sk::LitInt, Const::Int(bits)) => {
            let s = sext(bits, e.m.bits(cty));
            Some(match to.concrete() {
                Sk::Bool => e.m.const_bool(s != 0),
                Sk::Float | Sk::Double => const_float(e, to, if ir == Ty::F64 { s as f64 } else { s as f32 as f64 }),
                _ => e.m.const_int(ir, s as u64),
            })
        }
        (Sk::LitFloat, Const::Float(_)) => {
            let x = float_const(e, v)?;
            Some(match to.concrete() {
                Sk::Bool => e.m.const_bool(x != 0.0),
                Sk::Float | Sk::Double => const_float(e, to, x),
                k => {
                    // fptosi / fptoui of an out-of-range value is undefined: leave it to the target
                    let t = x.trunc();
                    let (lo, hi) = match k {
                        Sk::Int => (-2147483648.0, 2147483647.0),
                        Sk::Uint => (0.0, 4294967295.0),
                        Sk::Int64 => (-9223372036854775808.0, 9223372036854774784.0),
                        _ => (0.0, 18446744073709549568.0),
                    };
                    if !(t >= lo && t <= hi) {
                        return None;
                    }
                    let bits = if k.is_signed() { t as i64 as u64 } else { t as u64 };
                    e.m.const_int(ir, bits)
                }
            })
        }
        _ => None,
    }
}

/// Convert one scalar between kinds as DXC does for an argument.
fn conv_scalar(e: &mut Emit, v: Value, from: Sk, to: Sk) -> Value {
    if from == to {
        return v;
    }
    if is_lit(from) {
        if let Some(c) = fold_literal(e, v, from, to) {
            return c;
        }
    }
    e.convert_scalar(v, from, to)
}

/// Reinterpret the bits of a 32-bit value (`asfloat` / `asint` / `asuint`); constants fold, as
/// LLVM folds a bitcast of a constant.
fn bitcast(e: &mut Emit, v: Value, to: Ty) -> Value {
    let Some(c) = e.m.as_const(v) else {
        return e.cast(CastOp::BitCast, to, v);
    };
    if c.ty == to {
        return v;
    }
    match (&c.c, to) {
        (Const::Int(b), Ty::F32) => e.m.const_f32_bits(*b as u32),
        (Const::Float(b), Ty::I32) => e.m.const_i32(*b as u32),
        _ => e.cast(CastOp::BitCast, to, v),
    }
}

/// `v != 0` for a value of kind `k` (DXC's GenerateCmpNEZero; a bool is itself).
fn ne_zero(e: &mut Emit, v: Value, k: Sk) -> Value {
    let z = e.zero(k);
    match k.concrete() {
        Sk::Bool => v,
        Sk::Float | Sk::Double => e.fcmp(FPred::Une, v, z),
        _ => e.icmp(IPred::Ne, v, z),
    }
}

/// `-x` as IRBuilder::CreateFNeg writes it in LLVM 3.7: `fsub -0.0, x`.
fn fneg(e: &mut Emit, k: Sk, x: Value) -> Value {
    let z = const_float(e, k, -0.0);
    e.bin(BinOp::FSub, z, x)
}

/// DXC's TranslateFDot: `dot2` / `dot3` / `dot4`, or a multiply for one component.
fn fdot(e: &mut Emit, a: &[Value], b: &[Value]) -> Value {
    let op = match a.len() {
        2 => OpCode::Dot2,
        3 => OpCode::Dot3,
        4 => OpCode::Dot4,
        _ => return e.bin(BinOp::FMul, a[0], b[0]),
    };
    let mut args = [Value::NONE; 8];
    args[..a.len()].copy_from_slice(a);
    args[a.len()..a.len() * 2].copy_from_slice(b);
    dx(e, op, Ty::F32, &args[..a.len() * 2])
}

/// The float / signed / unsigned variant of an operation for kind `k`.
fn fiu(k: Sk, f: OpCode, i: OpCode, u: OpCode) -> OpCode {
    if k.is_float() {
        f
    } else if is_unsigned(k) {
        u
    } else {
        i
    }
}

/// `min` / `max` of two literal constants: the one DXC's evaluation picks, unchanged. None:
/// not both constants, or a NaN.
fn fold_min_max(e: &Emit, k: Sk, max: bool, a: Value, b: Value) -> Option<Value> {
    let a_wins = if k == Sk::LitFloat {
        let (x, y) = (float_const(e, a)?, float_const(e, b)?);
        if x.is_nan() || y.is_nan() {
            return None;
        }
        if max {
            x >= y
        } else {
            x <= y
        }
    } else {
        let bits = e.m.bits(e.m.as_const(a)?.ty);
        let (x, y) = (sext(e.m.as_int(a)?, bits), sext(e.m.as_int(b)?, bits));
        if max {
            x >= y
        } else {
            x <= y
        }
    };
    Some(if a_wins { a } else { b })
}

/// One call being lowered.
struct Cx<'a> {
    e: &'a mut Emit,
    name: &'a str,
    args: &'a [Val],
}

impl Cx<'_> {
    fn unsupported<T>(&self, why: &str) -> Result<T> {
        Err(Error::Unsupported(format!("intrinsic {} {why}", self.name)))
    }

    /// DXC rejects a call with the wrong number of arguments.
    fn argc(&self, n: usize) -> Result<()> {
        if self.args.len() == n {
            return Ok(());
        }
        self.e.err(format!("{} takes {n} argument(s), not {}", self.name, self.args.len()))
    }

    /// Kind and shape of argument `i`, which must be numeric.
    fn num(&self, i: usize) -> Result<(Sk, Shape)> {
        match self.args[i].ty {
            HTy::Num(k, r, c) if r as usize * c as usize <= 16 && self.args[i].len() == r as usize * c as usize => Ok((k, (r, c))),
            HTy::Num(..) => Err(Error::Internal(format!("intrinsic {}: argument {i} is malformed", self.name))),
            _ => self.unsupported("with a non-numeric argument"),
        }
    }

    /// The kind the arguments `idx`, which share one template parameter of class `cls`, are
    /// converted to. A literal kind comes back only for `Any` and `Numeric` (every argument a
    /// literal): the caller folds or gives up.
    fn kind(&self, cls: Cls, idx: &[usize]) -> Result<Sk> {
        let mut k = self.num(idx[0])?.0;
        for &i in &idx[1..] {
            k = common_sk(k, self.num(i)?.0);
        }
        match cls {
            Cls::Any => Ok(k),
            Cls::Numeric => Ok(if k == Sk::Bool { Sk::Float } else { k }),
            _ if is_lit(k) => self.unsupported(LITERALS),
            Cls::FloatLike => Ok(Sk::Float),
            Cls::AnyFloat => Ok(if k == Sk::Double { Sk::Double } else { Sk::Float }),
            Cls::AnyInt => match k {
                Sk::Int | Sk::Uint | Sk::Int64 | Sk::Uint64 => Ok(k),
                Sk::Bool | Sk::Float => Ok(Sk::Int),
                _ => self.unsupported("on a double"),
            },
        }
    }

    /// The shape the arguments `idx` share: scalars adapt to the others, vectors of different
    /// lengths truncate to the shortest (DXC warns and does the same).
    fn shape(&self, idx: &[usize]) -> Result<Shape> {
        let mut s = (1, 1);
        for &i in idx {
            let (_, a) = self.num(i)?;
            if a == (1, 1) || a == s {
                continue;
            }
            s = match s {
                (1, 1) => a,
                (1, n) if a.0 == 1 => (1, n.min(a.1)),
                _ => return self.unsupported("with arguments of different shapes"),
            };
        }
        Ok(s)
    }

    /// Argument `i` converted to kind `k` and `shape`.
    fn conv(&mut self, i: usize, k: Sk, shape: Shape) -> Result<Comps> {
        let (from, s) = self.num(i)?;
        let src = self.args[i].comps();
        let mut out = Comps::new(shape);
        if s == (1, 1) {
            let v = conv_scalar(self.e, src[0], from, k);
            out.v[..out.n].fill(v);
        } else if s == shape || (s.0 == 1 && shape.0 == 1 && shape.1 < s.1) {
            for j in 0..out.n {
                out.v[j] = conv_scalar(self.e, src[j], from, k);
            }
        } else {
            return self.unsupported("with arguments of different shapes");
        }
        Ok(out)
    }

    /// Argument `i` as a scalar of kind `k` (a parameter DXC declares scalar).
    fn scalar(&mut self, i: usize, k: Sk) -> Result<Value> {
        if self.num(i)?.1 != (1, 1) {
            return self.unsupported("with a vector where a scalar is expected");
        }
        Ok(self.conv(i, k, (1, 1))?.v[0])
    }

    /// The arguments `idx` converted to `k` / `shape` and `f` applied per component.
    fn zip<const N: usize>(&mut self, idx: [usize; N], k: Sk, shape: Shape, mut f: impl FnMut(&mut Emit, [Value; N]) -> Value) -> Result<Comps> {
        let mut a = [Comps::new(shape); N];
        for (slot, &i) in a.iter_mut().zip(&idx) {
            *slot = self.conv(i, k, shape)?;
        }
        let mut out = Comps::new(shape);
        for j in 0..out.n {
            let mut x = [Value::NONE; N];
            for (slot, c) in x.iter_mut().zip(&a) {
                *slot = c.v[j];
            }
            out.v[j] = f(self.e, x);
        }
        Ok(out)
    }

    fn ret(&self, k: Sk, shape: Shape, c: &Comps) -> Result<Called> {
        Ok(Called { ret: Val::new(HTy::Num(k, shape.0, shape.1), c.get()), outs: Vec::new() })
    }
    fn ret_scalar(&self, k: Sk, v: Value) -> Result<Called> {
        Ok(Called { ret: Val::scalar(k, v), outs: Vec::new() })
    }
    fn ret_void(&self) -> Result<Called> {
        Ok(Called { ret: Val::void(), outs: Vec::new() })
    }

    /// A `float_like` intrinsic of `N` arguments sharing one type: `f` per component on floats.
    fn float_n<const N: usize>(&mut self, f: impl FnMut(&mut Emit, [Value; N]) -> Value) -> Result<Called> {
        self.argc(N)?;
        let mut idx = [0; N];
        for (i, slot) in idx.iter_mut().enumerate() {
            *slot = i;
        }
        let (k, shape) = (self.kind(Cls::FloatLike, &idx)?, self.shape(&idx)?);
        let out = self.zip(idx, k, shape, f)?;
        self.ret(k, shape, &out)
    }

    /// A `float_like` intrinsic that is one unary dx.op.
    fn float1(&mut self, op: OpCode) -> Result<Called> {
        self.float_n(|e, [x]| dx(e, op, Ty::F32, &[x]))
    }

    // ---- casts ----

    /// `asfloat` / `asint` / `asuint` of `numeric32_only`: a bitcast, or nothing between the
    /// integer kinds. DXC has no overload for bool, double or the 64-bit integers.
    fn reinterpret(&mut self, to: Sk) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = self.num(0)?;
        let from = match k {
            Sk::Float | Sk::LitFloat => Sk::Float,
            Sk::Int | Sk::LitInt => Sk::Int,
            Sk::Uint => Sk::Uint,
            _ => return self.unsupported("of a bool or 64-bit value"),
        };
        let out = self.zip([0], from, shape, |e, [x]| bitcast(e, x, to.ir()))?;
        self.ret(to, shape, &out)
    }

    /// `asuint(double, out lo, out hi)`: SplitDouble per component.
    fn split_double(&mut self) -> Result<Called> {
        let shape = self.num(0)?.1;
        let d = self.conv(0, Sk::Double, shape)?;
        let (mut lo, mut hi) = (Comps::new(shape), Comps::new(shape));
        for j in 0..d.n {
            let r = dx(self.e, OpCode::SplitDouble, Ty::F64, &[d.v[j]]);
            lo.v[j] = self.e.extract(Ty::I32, r, 0);
            hi.v[j] = self.e.extract(Ty::I32, r, 1);
        }
        let ty = HTy::Num(Sk::Uint, shape.0, shape.1);
        Ok(Called { ret: Val::void(), outs: vec![(1, Val::new(ty, lo.get())), (2, Val::new(ty, hi.get()))] })
    }

    /// `asdouble(lo, hi)`: MakeDouble per component; two constants are the double they spell
    /// (`r *= asdouble(0u, 0x7FE00000u)` multiplies by a constant in DXC's output too).
    fn make_double(&mut self) -> Result<Called> {
        self.argc(2)?;
        let shape = self.shape(&[0, 1])?;
        let out = self.zip([0, 1], Sk::Uint, shape, |e, [lo, hi]| match (e.m.as_int(lo), e.m.as_int(hi)) {
            (Some(l), Some(h)) => e.m.const_f64(f64::from_bits(h << 32 | l & 0xFFFF_FFFF)),
            _ => dx(e, OpCode::MakeDouble, Ty::F64, &[lo, hi]),
        })?;
        self.ret(Sk::Double, shape, &out)
    }

    fn f16_to_f32(&mut self) -> Result<Called> {
        self.argc(1)?;
        let shape = self.num(0)?.1;
        let out = self.zip([0], Sk::Uint, shape, |e, [x]| dx(e, OpCode::LegacyF16ToF32, Ty::VOID, &[x]))?;
        self.ret(Sk::Float, shape, &out)
    }

    fn f32_to_f16(&mut self) -> Result<Called> {
        self.argc(1)?;
        let shape = self.num(0)?.1;
        let out = self.zip([0], Sk::Float, shape, |e, [x]| dx(e, OpCode::LegacyF32ToF16, Ty::VOID, &[x]))?;
        self.ret(Sk::Uint, shape, &out)
    }

    // ---- logic ----

    /// `select(cond, t, f)`: the component-wise ternary of HLSL 2021. `t` and `f` take their
    /// common type; a scalar condition selects whole vectors.
    fn select(&mut self) -> Result<Called> {
        self.argc(3)?;
        let (k, shape) = (self.kind(Cls::Any, &[1, 2])?, self.shape(&[0, 1, 2])?);
        let (c, t, f) = (self.conv(0, Sk::Bool, shape)?, self.conv(1, k, shape)?, self.conv(2, k, shape)?);
        let mut out = Comps::new(shape);
        for j in 0..out.n {
            out.v[j] = self.e.select(c.v[j], t.v[j], f.v[j]);
        }
        self.ret(k, shape, &out)
    }

    /// `and` / `or`: the component-wise logical operators, on bools.
    fn logical(&mut self, op: BinOp) -> Result<Called> {
        self.argc(2)?;
        let shape = self.shape(&[0, 1])?;
        let out = self.zip([0, 1], Sk::Bool, shape, |e, [a, b]| e.bin(op, a, b))?;
        self.ret(Sk::Bool, shape, &out)
    }

    /// `any` / `all` (TranslateBitwisePredicate): each component compared with zero, reduced
    /// left to right.
    fn any_all(&mut self, op: BinOp) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = self.num(0)?;
        let x = self.conv(0, k, shape)?;
        let mut r = ne_zero(self.e, x.v[0], k);
        for j in 1..x.n {
            let c = ne_zero(self.e, x.v[j], k);
            r = self.e.bin(op, r, c);
        }
        self.ret_scalar(Sk::Bool, r)
    }

    /// `isnan` / `isinf` / `isfinite` / `isnormal`: IsSpecialFloat on floats (a double is
    /// narrowed first: the translator's preludes test doubles from their bits for that reason).
    fn special_float(&mut self, op: OpCode) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = (self.kind(Cls::FloatLike, &[0])?, self.num(0)?.1);
        let out = self.zip([0], k, shape, |e, [x]| dx(e, op, Ty::F32, &[x]))?;
        self.ret(Sk::Bool, shape, &out)
    }

    // ---- numeric ----

    fn min_max_op(&mut self, k: Sk, max: bool, a: Value, b: Value) -> Result<Value> {
        if is_lit(k) {
            return match fold_min_max(self.e, k, max, a, b) {
                Some(v) => Ok(v),
                None => self.unsupported(LITERALS),
            };
        }
        let op = if max { fiu(k, OpCode::FMax, OpCode::IMax, OpCode::UMax) } else { fiu(k, OpCode::FMin, OpCode::IMin, OpCode::UMin) };
        Ok(dx(self.e, op, k.ir(), &[a, b]))
    }

    fn min_max(&mut self, max: bool) -> Result<Called> {
        self.argc(2)?;
        let (k, shape) = (self.kind(Cls::Numeric, &[0, 1])?, self.shape(&[0, 1])?);
        let (a, b) = (self.conv(0, k, shape)?, self.conv(1, k, shape)?);
        let mut out = Comps::new(shape);
        for j in 0..out.n {
            out.v[j] = self.min_max_op(k, max, a.v[j], b.v[j])?;
        }
        self.ret(k, shape, &out)
    }

    /// `clamp(x, lo, hi)`: `min(max(x, lo), hi)`. DXC's front end evaluates a constant float or
    /// signed clamp as `x < lo ? lo : x > hi ? hi : x` instead, which is another value when
    /// `lo > hi`; constant propagation folds the min and max. Nothing tells the two apart here.
    fn clamp(&mut self) -> Result<Called> {
        self.argc(3)?;
        let (k, shape) = (self.kind(Cls::Numeric, &[0, 1, 2])?, self.shape(&[0, 1, 2])?);
        let (x, lo, hi) = (self.conv(0, k, shape)?, self.conv(1, k, shape)?, self.conv(2, k, shape)?);
        let mut out = Comps::new(shape);
        for j in 0..out.n {
            if !is_unsigned(k) && self.e.m.as_const(x.v[j]).is_some() {
                let m = &self.e.m;
                let inverted = match (float_const(self.e, lo.v[j]), float_const(self.e, hi.v[j])) {
                    (Some(l), Some(h)) => l > h,
                    _ => match (m.as_int(lo.v[j]), m.as_int(hi.v[j])) {
                        (Some(l), Some(h)) => sext(l, m.bits(k.ir())) > sext(h, m.bits(k.ir())),
                        _ => false,
                    },
                };
                if inverted {
                    return self.unsupported("of constants with its bounds out of order has two values in DXC");
                }
            }
            let m = self.min_max_op(k, true, x.v[j], lo.v[j])?;
            out.v[j] = self.min_max_op(k, false, m, hi.v[j])?;
        }
        self.ret(k, shape, &out)
    }

    /// `abs`: FAbs, `IMax(x, 0 - x)` for signed integers, nothing for unsigned ones.
    fn abs(&mut self) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = (self.kind(Cls::Numeric, &[0])?, self.num(0)?.1);
        let x = self.conv(0, k, shape)?;
        let mut out = Comps::new(shape);
        for j in 0..out.n {
            let v = x.v[j];
            out.v[j] = match k {
                Sk::LitFloat => match float_const(self.e, v) {
                    Some(f) if f.is_sign_negative() => {
                        let wide = self.e.ty_of(v) == Ty::F64;
                        if wide {
                            self.e.m.const_f64(-f)
                        } else {
                            self.e.m.const_f32(-f as f32)
                        }
                    }
                    Some(_) => v,
                    None => return self.unsupported(LITERALS),
                },
                Sk::LitInt => match self.e.m.as_int(v) {
                    Some(bits) => {
                        let ty = self.e.ty_of(v);
                        let s = sext(bits, self.e.m.bits(ty));
                        self.e.m.const_int(ty, s.unsigned_abs())
                    }
                    None => return self.unsupported(LITERALS),
                },
                Sk::Float | Sk::Double => dx(self.e, OpCode::FAbs, k.ir(), &[v]),
                Sk::Uint | Sk::Uint64 => v,
                _ => {
                    let z = self.e.zero(k);
                    let neg = self.e.bin(BinOp::Sub, z, v);
                    dx(self.e, OpCode::IMax, k.ir(), &[v, neg])
                }
            };
        }
        self.ret(k, shape, &out)
    }

    /// `sign`: `(0 < x) - (x < 0)` as int; `x != 0` for unsigned kinds (TranslateUSign, which
    /// bool takes too).
    fn sign(&mut self) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = self.num(0)?;
        if is_lit(k) {
            return self.unsupported(LITERALS);
        }
        let out = self.zip([0], k, shape, |e, [x]| {
            let z = e.zero(k);
            match k {
                Sk::Bool => e.cast(CastOp::ZExt, Ty::I32, x),
                Sk::Uint | Sk::Uint64 => {
                    let nz = e.icmp(IPred::Ne, x, z);
                    e.cast(CastOp::ZExt, Ty::I32, nz)
                }
                _ => {
                    let (pos, neg) = if k.is_float() { (e.fcmp(FPred::Olt, z, x), e.fcmp(FPred::Olt, x, z)) } else { (e.icmp(IPred::Slt, z, x), e.icmp(IPred::Slt, x, z)) };
                    let (pos, neg) = (e.cast(CastOp::ZExt, Ty::I32, pos), e.cast(CastOp::ZExt, Ty::I32, neg));
                    e.bin(BinOp::Sub, pos, neg)
                }
            }
        })?;
        self.ret(Sk::Int, shape, &out)
    }

    /// `mad(a, b, c)`: FMad / IMad / UMad.
    fn mad(&mut self) -> Result<Called> {
        self.argc(3)?;
        let (k, shape) = (self.kind(Cls::Numeric, &[0, 1, 2])?, self.shape(&[0, 1, 2])?);
        if is_lit(k) {
            return self.unsupported(LITERALS);
        }
        let op = fiu(k, OpCode::FMad, OpCode::IMad, OpCode::UMad);
        let out = self.zip([0, 1, 2], k, shape, |e, [a, b, c]| dx(e, op, k.ir(), &[a, b, c]))?;
        self.ret(k, shape, &out)
    }

    /// `fma`: `double_only` (DXC has no float overload; literals adapt).
    fn fma(&mut self) -> Result<Called> {
        self.argc(3)?;
        for i in 0..3 {
            if !matches!(self.num(i)?.0, Sk::Double | Sk::LitFloat | Sk::LitInt) {
                return self.unsupported("on other than doubles");
            }
        }
        let (k, shape) = (self.kind(Cls::Any, &[0, 1, 2])?, self.shape(&[0, 1, 2])?);
        if k != Sk::Double {
            return self.unsupported(LITERALS);
        }
        let out = self.zip([0, 1, 2], k, shape, |e, [a, b, c]| dx(e, OpCode::Fma, Ty::F64, &[a, b, c]))?;
        self.ret(k, shape, &out)
    }

    fn saturate(&mut self) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = (self.kind(Cls::AnyFloat, &[0])?, self.num(0)?.1);
        let out = self.zip([0], k, shape, |e, [x]| dx(e, OpCode::Saturate, k.ir(), &[x]))?;
        self.ret(k, shape, &out)
    }

    /// `rcp`: `1.0 / x` (TranslateRCP; no dx.op).
    fn rcp(&mut self) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = (self.kind(Cls::AnyFloat, &[0])?, self.num(0)?.1);
        let out = self.zip([0], k, shape, |e, [x]| {
            let one = const_float(e, k, 1.0);
            e.bin(BinOp::FDiv, one, x)
        })?;
        self.ret(k, shape, &out)
    }

    // ---- float math built from the base-2 operations ----

    /// `exp(x)`: `Exp(log2(e) * x)` (DXIL's Exp is 2^x).
    fn exp(&mut self) -> Result<Called> {
        self.float_n(|e, [x]| {
            let c = e.c_f32(std::f64::consts::LOG2_E as f32);
            let x = e.bin(BinOp::FMul, c, x);
            dx(e, OpCode::Exp, Ty::F32, &[x])
        })
    }

    /// `log` / `log10`: `scale * Log(x)` (DXIL's Log is log2).
    fn log(&mut self, scale: f64) -> Result<Called> {
        self.float_n(|e, [x]| {
            let c = e.c_f32(scale as f32);
            let l = dx(e, OpCode::Log, Ty::F32, &[x]);
            e.bin(BinOp::FMul, c, l)
        })
    }

    /// `degrees` / `radians`.
    fn scale(&mut self, by: f64) -> Result<Called> {
        self.float_n(|e, [x]| {
            let c = e.c_f32(by as f32);
            e.bin(BinOp::FMul, c, x)
        })
    }

    /// `pow(x, y)`: `Exp(Log(x) * y)`, except that a constant exponent of exactly 2 (every
    /// component) is `x * x` (TranslatePowImpl outside FXC compatibility mode).
    fn pow(&mut self) -> Result<Called> {
        self.argc(2)?;
        let (k, shape) = (self.kind(Cls::FloatLike, &[0, 1])?, self.shape(&[0, 1])?);
        let (x, y) = (self.conv(0, k, shape)?, self.conv(1, k, shape)?);
        let square = y.get().iter().all(|&v| v == y.v[0]) && float_const(self.e, y.v[0]) == Some(2.0);
        let mut out = Comps::new(shape);
        for j in 0..out.n {
            out.v[j] = if square {
                self.e.bin(BinOp::FMul, x.v[j], x.v[j])
            } else {
                let l = dx(self.e, OpCode::Log, Ty::F32, &[x.v[j]]);
                let m = self.e.bin(BinOp::FMul, l, y.v[j]);
                dx(self.e, OpCode::Exp, Ty::F32, &[m])
            };
        }
        self.ret(k, shape, &out)
    }

    /// `atan2(y, x)`: `Atan(y / x)` moved to the right quadrant (TranslateAtan2).
    fn atan2(&mut self) -> Result<Called> {
        use std::f64::consts::PI;
        self.float_n(|e, [y, x]| {
            let (pi, half_pi, neg_half_pi, zero) = (e.c_f32(PI as f32), e.c_f32((PI / 2.0) as f32), e.c_f32((-PI / 2.0) as f32), e.c_f32(0.0));
            let tan = e.bin(BinOp::FDiv, y, x);
            let atan = dx(e, OpCode::Atan, Ty::F32, &[tan]);
            let (add_pi, sub_pi) = (e.bin(BinOp::FAdd, atan, pi), e.bin(BinOp::FSub, atan, pi));
            let (x_lt, x_eq) = (e.fcmp(FPred::Olt, x, zero), e.fcmp(FPred::Oeq, x, zero));
            let (y_ge, y_lt) = (e.fcmp(FPred::Oge, y, zero), e.fcmp(FPred::Olt, y, zero));
            let c = e.bin(BinOp::And, x_lt, y_ge);
            let r = e.select(c, add_pi, atan);
            let c = e.bin(BinOp::And, x_lt, y_lt);
            let r = e.select(c, sub_pi, r);
            let c = e.bin(BinOp::And, x_eq, y_lt);
            let r = e.select(c, neg_half_pi, r);
            let c = e.bin(BinOp::And, x_eq, y_ge);
            e.select(c, half_pi, r)
        })
    }

    /// `sincos(x, out s, out c)`.
    fn sincos(&mut self) -> Result<Called> {
        self.argc(3)?;
        let (k, shape) = (self.kind(Cls::FloatLike, &[0])?, self.num(0)?.1);
        let s = self.zip([0], k, shape, |e, [x]| dx(e, OpCode::Sin, Ty::F32, &[x]))?;
        let c = self.zip([0], k, shape, |e, [x]| dx(e, OpCode::Cos, Ty::F32, &[x]))?;
        let ty = HTy::Num(k, shape.0, shape.1);
        Ok(Called { ret: Val::void(), outs: vec![(1, Val::new(ty, s.get())), (2, Val::new(ty, c.get()))] })
    }

    /// `fmod(a, b)`: `frac(|a / b|)` with the sign of the quotient, times `b` (TranslateFMod).
    fn fmod(&mut self) -> Result<Called> {
        self.float_n(|e, [a, b]| {
            let div = e.bin(BinOp::FDiv, a, b);
            let neg_div = fneg(e, Sk::Float, div);
            let ge = e.fcmp(FPred::Oge, div, neg_div);
            let abs = dx(e, OpCode::FAbs, Ty::F32, &[div]);
            let frc = dx(e, OpCode::Frc, Ty::F32, &[abs]);
            let neg_frc = fneg(e, Sk::Float, frc);
            let real = e.select(ge, frc, neg_frc);
            e.bin(BinOp::FMul, real, b)
        })
    }

    /// `modf(x, out ip)`: `ip = Round_z(x)`, returns `x - ip`.
    fn modf(&mut self) -> Result<Called> {
        self.argc(2)?;
        let (k, shape) = (self.kind(Cls::FloatLike, &[0])?, self.num(0)?.1);
        let x = self.conv(0, k, shape)?;
        let (mut ip, mut frac) = (Comps::new(shape), Comps::new(shape));
        for j in 0..x.n {
            ip.v[j] = dx(self.e, OpCode::Round_z, Ty::F32, &[x.v[j]]);
            frac.v[j] = self.e.bin(BinOp::FSub, x.v[j], ip.v[j]);
        }
        let ty = HTy::Num(k, shape.0, shape.1);
        Ok(Called { ret: Val::new(ty, frac.get()), outs: vec![(1, Val::new(ty, ip.get()))] })
    }

    /// `frexp(x, out exp)`: TranslateFrexp's bit arithmetic on the float (no denormal or
    /// inf / NaN handling: the translator's preludes add their own).
    fn frexp(&mut self) -> Result<Called> {
        self.argc(2)?;
        let (k, shape) = (self.kind(Cls::FloatLike, &[0])?, self.num(0)?.1);
        let x = self.conv(0, k, shape)?;
        let (mut exp, mut mant) = (Comps::new(shape), Comps::new(shape));
        for j in 0..x.n {
            let e = &mut *self.e;
            let zero = e.c_f32(0.0);
            let (exp_mask, mant_mask, shift) = (e.c_u32(0x7f80_0000), e.c_u32(0x007f_ffff), e.c_u32(23));
            let (mant_or, bias) = (e.c_u32(0x3f00_0000), e.c_u32(0x3f00_0000u32.wrapping_neg()));
            let not_zero = e.fcmp(FPred::Une, x.v[j], zero);
            let not_zero = e.cast(CastOp::SExt, Ty::I32, not_zero);
            let bits = bitcast(e, x.v[j], Ty::I32);
            let t = e.bin(BinOp::And, bits, exp_mask);
            let t = e.bin(BinOp::Add, t, bias);
            let t = e.bin(BinOp::And, t, not_zero);
            let t = e.bin(BinOp::AShr, t, shift);
            exp.v[j] = e.cast(CastOp::SIToFP, Ty::F32, t);
            let t = e.bin(BinOp::And, bits, mant_mask);
            let t = e.bin(BinOp::Or, t, mant_or);
            let t = e.bin(BinOp::And, t, not_zero);
            mant.v[j] = bitcast(e, t, Ty::F32);
        }
        let ty = HTy::Num(k, shape.0, shape.1);
        Ok(Called { ret: Val::new(ty, mant.get()), outs: vec![(1, Val::new(ty, exp.get()))] })
    }

    /// `ldexp(x, exp)`: `Exp(exp) * x`.
    fn ldexp(&mut self) -> Result<Called> {
        self.float_n(|e, [x, exp]| {
            let p = dx(e, OpCode::Exp, Ty::F32, &[exp]);
            e.bin(BinOp::FMul, p, x)
        })
    }

    /// `lerp(x, y, s)`: `x + s * (y - x)`.
    fn lerp(&mut self) -> Result<Called> {
        self.float_n(|e, [x, y, s]| {
            let d = e.bin(BinOp::FSub, y, x);
            let m = e.bin(BinOp::FMul, s, d);
            e.bin(BinOp::FAdd, x, m)
        })
    }

    /// `step(edge, x)`: `x < edge ? 0.0 : 1.0`.
    fn step(&mut self) -> Result<Called> {
        self.float_n(|e, [edge, x]| {
            let (zero, one) = (e.c_f32(0.0), e.c_f32(1.0));
            let c = e.fcmp(FPred::Olt, x, edge);
            e.select(c, zero, one)
        })
    }

    /// `smoothstep(min, max, x)`: `s = saturate((x - min) / (max - min)); s * (s * (3 - s * 2))`.
    fn smoothstep(&mut self) -> Result<Called> {
        self.float_n(|e, [min, max, x]| {
            let (two, three) = (e.c_f32(2.0), e.c_f32(3.0));
            let range = e.bin(BinOp::FSub, max, min);
            let off = e.bin(BinOp::FSub, x, min);
            let d = e.bin(BinOp::FDiv, off, range);
            let s = dx(e, OpCode::Saturate, Ty::F32, &[d]);
            let r = e.bin(BinOp::FMul, s, two);
            let r = e.bin(BinOp::FSub, three, r);
            let r = e.bin(BinOp::FMul, s, r);
            e.bin(BinOp::FMul, s, r)
        })
    }

    /// `fwidth(x)`: `|ddx_coarse(x)| + |ddy_coarse(x)|`.
    fn fwidth(&mut self) -> Result<Called> {
        self.float_n(|e, [x]| {
            let ddx = dx(e, OpCode::DerivCoarseX, Ty::F32, &[x]);
            let ax = dx(e, OpCode::FAbs, Ty::F32, &[ddx]);
            let ddy = dx(e, OpCode::DerivCoarseY, Ty::F32, &[x]);
            let ay = dx(e, OpCode::FAbs, Ty::F32, &[ddy]);
            e.bin(BinOp::FAdd, ax, ay)
        })
    }

    // ---- vectors ----

    /// The arguments `idx` as float vectors of one length (a scalar is a vector of one).
    fn float_vectors<const N: usize>(&mut self, idx: [usize; N]) -> Result<(Shape, [Comps; N])> {
        let (k, shape) = (self.kind(Cls::FloatLike, &idx)?, self.shape(&idx)?);
        if shape.0 != 1 || shape.1 > 4 {
            return self.unsupported("of matrices");
        }
        let mut a = [Comps::new(shape); N];
        for (slot, &i) in a.iter_mut().zip(&idx) {
            *slot = self.conv(i, k, shape)?;
        }
        Ok((shape, a))
    }

    /// `dot`: Dot2 / Dot3 / Dot4 on floats, a multiply and IMad / UMad chain on integers
    /// (ExpandDot). DXIL has no double dot (DXC's own output fails validation).
    fn dot(&mut self) -> Result<Called> {
        self.argc(2)?;
        let (k, shape) = (self.kind(Cls::Numeric, &[0, 1])?, self.shape(&[0, 1])?);
        if shape.0 != 1 || shape.1 > 4 || k == Sk::Double || is_lit(k) {
            return self.unsupported("of matrices, doubles or literals");
        }
        let (a, b) = (self.conv(0, k, shape)?, self.conv(1, k, shape)?);
        let r = self.dot_of(k, a.get(), b.get());
        self.ret_scalar(k, r)
    }

    fn dot_of(&mut self, k: Sk, a: &[Value], b: &[Value]) -> Value {
        if k.is_float() {
            return fdot(self.e, a, b);
        }
        let mad = if is_unsigned(k) { OpCode::UMad } else { OpCode::IMad };
        let mut r = self.e.bin(BinOp::Mul, a[0], b[0]);
        for j in 1..a.len() {
            r = dx(self.e, mad, k.ir(), &[a[j], b[j], r]);
        }
        r
    }

    fn cross(&mut self) -> Result<Called> {
        self.argc(2)?;
        let k = self.kind(Cls::FloatLike, &[0, 1])?;
        let (a, b) = (self.conv(0, k, (1, 3))?, self.conv(1, k, (1, 3))?);
        let e = &mut *self.e;
        let mut mul_sub = |x0: usize, y0: usize| {
            let xy = e.bin(BinOp::FMul, a.v[x0], b.v[y0]);
            let yx = e.bin(BinOp::FMul, a.v[y0], b.v[x0]);
            e.bin(BinOp::FSub, xy, yx)
        };
        let mut out = Comps::new((1, 3));
        out.v[0] = mul_sub(1, 2);
        out.v[1] = mul_sub(2, 0);
        out.v[2] = mul_sub(0, 1);
        self.ret(k, (1, 3), &out)
    }

    /// TranslateLength: `Sqrt` of the summed squares; FAbs for one component.
    fn length_of(&mut self, x: &[Value]) -> Value {
        if x.len() == 1 {
            return dx(self.e, OpCode::FAbs, Ty::F32, &[x[0]]);
        }
        let mut sum = self.e.bin(BinOp::FMul, x[0], x[0]);
        for &c in &x[1..] {
            let sq = self.e.bin(BinOp::FMul, c, c);
            sum = self.e.bin(BinOp::FAdd, sum, sq);
        }
        dx(self.e, OpCode::Sqrt, Ty::F32, &[sum])
    }

    fn length(&mut self) -> Result<Called> {
        self.argc(1)?;
        let (_, [x]) = self.float_vectors([0])?;
        let r = self.length_of(x.get());
        self.ret_scalar(Sk::Float, r)
    }

    fn distance(&mut self) -> Result<Called> {
        self.argc(2)?;
        let (shape, [a, b]) = self.float_vectors([0, 1])?;
        let mut d = Comps::new(shape);
        for j in 0..d.n {
            d.v[j] = self.e.bin(BinOp::FSub, a.v[j], b.v[j]);
        }
        let r = self.length_of(d.get());
        self.ret_scalar(Sk::Float, r)
    }

    /// `normalize(x)`: `x * Rsqrt(dot(x, x))`.
    fn normalize(&mut self) -> Result<Called> {
        self.argc(1)?;
        let (shape, [x]) = self.float_vectors([0])?;
        let d = fdot(self.e, x.get(), x.get());
        let rsqrt = dx(self.e, OpCode::Rsqrt, Ty::F32, &[d]);
        let mut out = Comps::new(shape);
        for j in 0..out.n {
            out.v[j] = self.e.bin(BinOp::FMul, x.v[j], rsqrt);
        }
        self.ret(Sk::Float, shape, &out)
    }

    /// `reflect(i, n)`: `i - (2 * dot(i, n)) * n`.
    fn reflect(&mut self) -> Result<Called> {
        self.argc(2)?;
        let (shape, [i, n]) = self.float_vectors([0, 1])?;
        let two = self.e.c_f32(2.0);
        let d = fdot(self.e, i.get(), n.get());
        let d = self.e.bin(BinOp::FMul, two, d);
        let mut out = Comps::new(shape);
        for j in 0..out.n {
            let nd = self.e.bin(BinOp::FMul, d, n.v[j]);
            out.v[j] = self.e.bin(BinOp::FSub, i.v[j], nd);
        }
        self.ret(Sk::Float, shape, &out)
    }

    /// `refract(i, n, eta)` (TranslateRefract).
    fn refract(&mut self) -> Result<Called> {
        self.argc(3)?;
        let (shape, [i, n]) = self.float_vectors([0, 1])?;
        let eta = self.scalar(2, Sk::Float)?;
        let e = &mut *self.e;
        let (one, zero) = (e.c_f32(1.0), e.c_f32(0.0));
        let d = fdot(e, i.get(), n.get());
        let eta2 = e.bin(BinOp::FMul, eta, eta);
        let d2 = e.bin(BinOp::FMul, d, d);
        let d2 = e.bin(BinOp::FSub, one, d2);
        let eta2 = e.bin(BinOp::FMul, d2, eta2);
        let t = e.bin(BinOp::FSub, one, eta2);
        let cond = e.fcmp(FPred::Oge, t, zero);
        let mut eta_i = Comps::new(shape);
        for j in 0..eta_i.n {
            eta_i.v[j] = e.bin(BinOp::FMul, i.v[j], eta);
        }
        let sqrt = dx(e, OpCode::Sqrt, Ty::F32, &[t]);
        let eta_d = e.bin(BinOp::FMul, eta, d);
        let eta_sqrt = e.bin(BinOp::FAdd, eta_d, sqrt);
        let mut out = Comps::new(shape);
        for j in 0..out.n {
            let r = e.bin(BinOp::FMul, eta_sqrt, n.v[j]);
            let r = e.bin(BinOp::FSub, eta_i.v[j], r);
            out.v[j] = e.select(cond, r, zero);
        }
        self.ret(Sk::Float, shape, &out)
    }

    /// `faceforward(n, i, ng)`: `dot(i, ng) < 0 ? n : -n`.
    fn faceforward(&mut self) -> Result<Called> {
        self.argc(3)?;
        let (shape, [n, i, ng]) = self.float_vectors([0, 1, 2])?;
        let zero = self.e.c_f32(0.0);
        let d = fdot(self.e, i.get(), ng.get());
        let lt = self.e.fcmp(FPred::Olt, d, zero);
        let mut out = Comps::new(shape);
        for j in 0..out.n {
            let neg = fneg(self.e, Sk::Float, n.v[j]);
            out.v[j] = self.e.select(lt, n.v[j], neg);
        }
        self.ret(Sk::Float, shape, &out)
    }

    /// `mul`: a product with a scalar, a dot product of two vectors, or the matrix product
    /// (HLMatrixLowerPass::lowerHLMulIntrinsic: a multiply and a mad chain per element,
    /// values row-major).
    fn mul(&mut self) -> Result<Called> {
        self.argc(2)?;
        let k = self.kind(Cls::Numeric, &[0, 1])?;
        if is_lit(k) {
            return self.unsupported(LITERALS);
        }
        let (sa, sb) = (self.num(0)?.1, self.num(1)?.1);
        let op = if k.is_float() { BinOp::FMul } else { BinOp::Mul };
        if sa == (1, 1) || sb == (1, 1) {
            let shape = if sa == (1, 1) { sb } else { sa };
            let out = self.zip([0, 1], k, shape, |e, [a, b]| e.bin(op, a, b))?;
            return self.ret(k, shape, &out);
        }
        let (a, b) = (self.conv(0, k, sa)?, self.conv(1, k, sb)?);
        if sa.0 == 1 && sb.0 == 1 {
            if sa != sb || sa.1 > 4 || k == Sk::Double {
                return self.unsupported("of vectors of different lengths, or of doubles");
            }
            let r = self.dot_of(k, a.get(), b.get());
            return self.ret_scalar(k, r);
        }
        // a vector on the left is a row, on the right a column
        let (lr, lc, rr, rc) = match (sa.0, sb.0) {
            (1, _) => (1, sa.1, sb.0, sb.1),
            (_, 1) => (sa.0, sa.1, sb.1, 1),
            _ => (sa.0, sa.1, sb.0, sb.1),
        };
        if lc != rr {
            return self.unsupported("of mismatched dimensions");
        }
        let mad = fiu(k, OpCode::FMad, OpCode::IMad, OpCode::UMad);
        let shape = if lr == 1 || rc == 1 { (1, lr.max(rc)) } else { (lr, rc) };
        let (lc, rc) = (lc as usize, rc as usize);
        let mut out = Comps::new(shape);
        for row in 0..lr as usize {
            for col in 0..rc {
                let mut r = Value::NONE;
                for acc in 0..lc {
                    let (x, y) = (a.v[row * lc + acc], b.v[acc * rc + col]);
                    r = if acc == 0 { self.e.bin(op, x, y) } else { dx(self.e, mad, k.ir(), &[x, y, r]) };
                }
                out.v[row * rc + col] = r;
            }
        }
        self.ret(k, shape, &out)
    }

    /// `dot4add_u8packed(a, b, acc)` / `dot4add_i8packed`: Dot4AddPacked(acc, a, b).
    fn dot4add(&mut self, op: OpCode, acc_kind: Sk) -> Result<Called> {
        self.argc(3)?;
        let (a, b, acc) = (self.scalar(0, Sk::Uint)?, self.scalar(1, Sk::Uint)?, self.scalar(2, acc_kind)?);
        let r = dx(self.e, op, Ty::I32, &[acc, a, b]);
        self.ret_scalar(acc_kind, r)
    }

    // ---- bits ----

    /// `countbits` / `firstbitlow`: the operation on the argument's width, a uint per component.
    fn bit_count(&mut self, op: OpCode) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = (self.kind(Cls::AnyInt, &[0])?, self.num(0)?.1);
        let out = self.zip([0], k, shape, |e, [x]| dx(e, op, k.ir(), &[x]))?;
        self.ret(Sk::Uint, shape, &out)
    }

    fn reverse_bits(&mut self) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = (self.kind(Cls::AnyInt, &[0])?, self.num(0)?.1);
        let out = self.zip([0], k, shape, |e, [x]| dx(e, OpCode::Bfrev, k.ir(), &[x]))?;
        self.ret(k, shape, &out)
    }

    /// `firstbithigh`: DXIL counts from the most significant bit, HLSL from the least, so
    /// DXC emits `r == -1 ? -1 : (width - 1) - r` (TranslateFirstbitHi); FirstbitSHi for signed
    /// arguments.
    fn first_bit_high(&mut self) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = (self.kind(Cls::AnyInt, &[0])?, self.num(0)?.1);
        let op = if is_unsigned(k) { OpCode::FirstbitHi } else { OpCode::FirstbitSHi };
        let top = if k.is_64() { 63 } else { 31 };
        let out = self.zip([0], k, shape, |e, [x]| {
            let neg1 = e.c_u32(u32::MAX);
            let r = dx(e, op, k.ir(), &[x]);
            if let Some(c) = e.m.as_int(r) {
                return if c as u32 == u32::MAX { neg1 } else { e.c_u32(top - c as u32) };
            }
            let top = e.c_u32(top);
            let sub = e.bin(BinOp::Sub, top, r);
            let none = e.icmp(IPred::Eq, neg1, r);
            e.select(none, neg1, sub)
        })?;
        self.ret(Sk::Uint, shape, &out)
    }

    // ---- wave and quad ----

    fn no_arg(&mut self, op: OpCode, overload: Ty, k: Sk) -> Result<Called> {
        self.argc(0)?;
        let r = dx(self.e, op, overload, &[]);
        self.ret_scalar(k, r)
    }

    /// The wave operations of one bool: WaveActiveAnyTrue / AllTrue, WaveActiveCountBits,
    /// WavePrefixCountBits.
    fn wave_bool(&mut self, op: OpCode, ret: Sk) -> Result<Called> {
        self.argc(1)?;
        let c = self.scalar(0, Sk::Bool)?;
        let r = dx(self.e, op, Ty::VOID, &[c]);
        self.ret_scalar(ret, r)
    }

    /// `WaveActiveBallot`: the four fields of `dx.types.fouri32` as a uint4.
    fn wave_ballot(&mut self) -> Result<Called> {
        self.argc(1)?;
        let c = self.scalar(0, Sk::Bool)?;
        let r = dx(self.e, OpCode::WaveActiveBallot, Ty::VOID, &[c]);
        let mut out = Comps::new((1, 4));
        for j in 0..4 {
            out.v[j] = self.e.extract(Ty::I32, r, j as u32);
        }
        self.ret(Sk::Uint, (1, 4), &out)
    }

    /// The kind a value keeps through an `any<>` / `numeric<>` wave or quad operation.
    fn lane_kind(&self, cls: Cls) -> Result<Sk> {
        let k = self.kind(cls, &[0])?;
        if is_lit(k) {
            return self.unsupported("on a literal (DXC widens it to 64 bits)");
        }
        Ok(k)
    }

    fn wave_all_equal(&mut self) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = (self.lane_kind(Cls::Any)?, self.num(0)?.1);
        let out = self.zip([0], k, shape, |e, [x]| dx(e, OpCode::WaveActiveAllEqual, k.ir(), &[x]))?;
        self.ret(Sk::Bool, shape, &out)
    }

    /// `WaveReadLaneAt(value, lane)` (`any<>`) / `QuadReadLaneAt` (`numeric<>`).
    fn read_lane_at(&mut self, op: OpCode, cls: Cls) -> Result<Called> {
        self.argc(2)?;
        let (k, shape) = (self.lane_kind(cls)?, self.num(0)?.1);
        let lane = self.scalar(1, Sk::Uint)?;
        let out = self.zip([0], k, shape, |e, [x]| dx(e, op, k.ir(), &[x, lane]))?;
        self.ret(k, shape, &out)
    }

    fn wave_read_first(&mut self) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = (self.lane_kind(Cls::Any)?, self.num(0)?.1);
        let out = self.zip([0], k, shape, |e, [x]| dx(e, OpCode::WaveReadLaneFirst, k.ir(), &[x]))?;
        self.ret(k, shape, &out)
    }

    /// WaveActiveSum / Product / Min / Max and WavePrefixSum / Product: (value, i8 op, i8
    /// sign) with DXIL::WaveOpKind and DXIL::SignedOpKind (1: unsigned).
    fn wave_op(&mut self, op: OpCode, kind: u8) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = (self.lane_kind(Cls::Numeric)?, self.num(0)?.1);
        let (kind, sign) = (self.e.m.const_i8(kind), self.e.m.const_i8(is_unsigned(k) as u8));
        let out = self.zip([0], k, shape, |e, [x]| dx(e, op, k.ir(), &[x, kind, sign]))?;
        self.ret(k, shape, &out)
    }

    /// WaveActiveBitAnd / Or / Xor: `uint_only` (DXC rejects an int), DXIL::WaveBitOpKind.
    fn wave_bit(&mut self, kind: u8) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = self.num(0)?;
        if !is_unsigned(k) {
            return self.unsupported("on other than uint");
        }
        let kind = self.e.m.const_i8(kind);
        let out = self.zip([0], k, shape, |e, [x]| dx(e, OpCode::WaveActiveBit, k.ir(), &[x, kind]))?;
        self.ret(k, shape, &out)
    }

    /// QuadReadAcrossX / Y / Diagonal: QuadOp with DXIL::QuadOpKind.
    fn quad_across(&mut self, kind: u8) -> Result<Called> {
        self.argc(1)?;
        let (k, shape) = (self.lane_kind(Cls::Numeric)?, self.num(0)?.1);
        let kind = self.e.m.const_i8(kind);
        let out = self.zip([0], k, shape, |e, [x]| dx(e, OpCode::QuadOp, k.ir(), &[x, kind]))?;
        self.ret(k, shape, &out)
    }

    /// QuadAny / QuadAll below SM 6.7 (no QuadVote): the condition sign-extended, read across
    /// X, Y and the diagonal, and or-ed / and-ed with itself.
    fn quad_vote(&mut self, op: BinOp) -> Result<Called> {
        self.argc(1)?;
        let c = self.scalar(0, Sk::Bool)?;
        let e = &mut *self.e;
        let c = e.cast(CastOp::SExt, Ty::I32, c);
        let mut across = [Value::NONE; 3];
        for (kind, slot) in across.iter_mut().enumerate() {
            let kind = e.m.const_i8(kind as u8);
            *slot = dx(e, OpCode::QuadOp, Ty::I32, &[c, kind]);
        }
        let r = e.bin(op, across[0], across[1]);
        let r = e.bin(op, r, across[2]);
        let r = e.bin(op, r, c);
        let r = e.cast(CastOp::Trunc, Ty::I1, r);
        self.ret_scalar(Sk::Bool, r)
    }

    // ---- effects ----

    fn barrier(&mut self, mode: u32) -> Result<Called> {
        self.argc(0)?;
        let mode = self.e.c_u32(mode);
        dx(self.e, OpCode::Barrier, Ty::VOID, &[mode]);
        self.ret_void()
    }

    /// `clip(x)`: Discard when any component is below zero.
    fn clip(&mut self) -> Result<Called> {
        self.argc(1)?;
        let shape = self.num(0)?.1;
        let x = self.conv(0, Sk::Float, shape)?;
        let zero = self.e.c_f32(0.0);
        let mut cond = self.e.fcmp(FPred::Olt, x.v[0], zero);
        for j in 1..x.n {
            let c = self.e.fcmp(FPred::Olt, x.v[j], zero);
            cond = self.e.bin(BinOp::Or, cond, c);
        }
        dx(self.e, OpCode::Discard, Ty::VOID, &[cond]);
        self.ret_void()
    }

    fn rt_sample_position(&mut self) -> Result<Called> {
        self.argc(1)?;
        let s = self.scalar(0, Sk::Int)?;
        let r = dx(self.e, OpCode::RenderTargetGetSamplePosition, Ty::VOID, &[s]);
        let mut out = Comps::new((1, 2));
        out.v[0] = self.e.extract(Ty::F32, r, 0);
        out.v[1] = self.e.extract(Ty::F32, r, 1);
        self.ret(Sk::Float, (1, 2), &out)
    }

    fn set_mesh_output_counts(&mut self) -> Result<Called> {
        self.argc(2)?;
        let (v, p) = (self.scalar(0, Sk::Uint)?, self.scalar(1, Sk::Uint)?);
        dx(self.e, OpCode::SetMeshOutputCounts, Ty::VOID, &[v, p]);
        self.ret_void()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn other_intrinsics_are_sorted_for_the_binary_search() {
        assert!(OTHER_INTRINSICS.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn unknown_names_are_left_to_the_caller() {
        let mut e = Emit::new();
        assert!(call(&mut e, "xcl_x_meet", &[]).unwrap().is_none());
        assert!(call(&mut e, "NonUniformResourceIndex", &[]).unwrap().is_none());
        assert!(call(&mut e, "InterlockedAdd", &[]).unwrap().is_none());
        assert!(matches!(call(&mut e, "msad4", &[]), Err(Error::Unsupported(_))));
    }

    #[test]
    fn literals_convert_at_their_own_precision() {
        let mut e = Emit::new();
        // 4294967295.0 is not a float: against a double it must stay exact
        let lit = e.m.const_f64(4294967295.0);
        let d = conv_scalar(&mut e, lit, Sk::LitFloat, Sk::Double);
        assert_eq!(e.m.as_float_bits(d), Some(4294967295.0f64.to_bits()));
        let f = conv_scalar(&mut e, lit, Sk::LitFloat, Sk::Float);
        assert_eq!(e.m.as_float_bits(f), Some(4294967296.0f32.to_bits() as u64));
        let i = e.m.const_i32(0xFFFF_FFFF);
        let x = conv_scalar(&mut e, i, Sk::LitInt, Sk::Float);
        assert_eq!(e.m.as_float_bits(x), Some((-1.0f32).to_bits() as u64));
        let u = conv_scalar(&mut e, i, Sk::LitInt, Sk::Uint64);
        assert_eq!(e.m.as_int(u), Some(u64::MAX));
    }
}
