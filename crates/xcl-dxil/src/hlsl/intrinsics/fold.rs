//! Constant folding of dx.op calls whose operands are all constants, for the operations whose
//! result is fixed by IEEE 754 or by integer arithmetic.
//!
//! DXC folds such calls in two places: the front end evaluates an intrinsic whose arguments
//! are constant expressions at any optimisation level (TryEvalIntrinsic in
//! CGHLSLMSFinishCodeGen.cpp: `exp2(-14.0f)` in the translator's f16 prelude is the constant
//! 2^-14 even at -O0), and at -O1 constant propagation folds the dx.op itself
//! (DxilConstantFolding.cpp). A GPU evaluating the call at run time need not produce the
//! same bits (`Exp`, `Rsqrt` and a fused `FMad` are not exact there), so emitting the call is
//! not equivalent.
//!
//! Only what both of DXC's evaluators and the host agree on bit for bit folds here. DXC
//! evaluates the transcendental functions with the C library it was built with: those are
//! never folded, except `Exp` of an integer and `Log` of a power of two, where every
//! library is exact. Where the two evaluators disagree the call is emitted: `Frc` of a
//! negative value (`x - floor(x)` in one, `fabs(modf(x))` in the other), a `FMax` / `FMin`
//! tie between zeros of opposite sign, and `Rsqrt` when rounding `1 / sqrt(x)` once and
//! twice differ. Like DXC, nothing folds on a NaN or an infinity.

use super::Emit;
use crate::consts::OpCode;
use crate::ir::{Const, Ty, Value};

/// A dx.op call at the insertion point, or the constant it evaluates to.
pub(super) fn dx(e: &mut Emit, op: OpCode, overload: Ty, args: &[Value]) -> Value {
    match fold(e, op, args) {
        Some(v) => v,
        None => e.dx(op, overload, args),
    }
}

fn fold(e: &mut Emit, op: OpCode, args: &[Value]) -> Option<Value> {
    // the common case leaves here: a first operand that is not a constant
    let first = e.m.as_const(*args.first()?)?;
    let ty = first.ty;
    match first.c {
        Const::Int(_) if args.len() <= 3 => {
            let mut x = [0u64; 3];
            for (slot, &a) in x.iter_mut().zip(args) {
                *slot = e.m.as_int(a)?;
            }
            let (r, bit_index) = int(op, e.m.bits(ty), &x[..args.len()])?;
            Some(e.m.const_int(if bit_index { Ty::I32 } else { ty }, r))
        }
        Const::Float(_) if args.len() <= 8 => {
            let mut bits = [0u64; 8];
            for (slot, &a) in bits.iter_mut().zip(args) {
                *slot = e.m.as_float_bits(a)?;
            }
            let bits = &bits[..args.len()];
            if ty == Ty::F32 {
                let mut x = [0f32; 8];
                for (slot, &b) in x.iter_mut().zip(bits) {
                    *slot = f32::from_bits(b as u32);
                }
                if op == OpCode::IsNaN {
                    return (!x[0].is_nan()).then(|| e.m.const_bool(false));
                }
                let r = float32(op, &x[..args.len()])?;
                Some(e.m.const_f32_bits(r.to_bits()))
            } else if ty == Ty::F64 {
                let mut x = [0f64; 8];
                for (slot, &b) in x.iter_mut().zip(bits) {
                    *slot = f64::from_bits(b);
                }
                let r = float64(op, &x[..args.len()])?;
                Some(e.m.const_f64(r))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// An integer operation on `bits`-wide operands: the result, and whether it is a bit index
/// (an i32 whatever the operand width, -1 for "no bit").
fn int(op: OpCode, bits: u32, x: &[u64]) -> Option<(u64, bool)> {
    use OpCode as O;
    if bits != 32 && bits != 64 {
        return None;
    }
    let pad = 64 - bits;
    let s = |v: u64| ((v << pad) as i64) >> pad;
    // a count of leading or trailing bits; the whole width means there is no such bit
    let index = |n: u32| (if n >= bits { u32::MAX as u64 } else { n as u64 }, true);
    Some(match (op, x) {
        (O::Bfrev, &[a]) => (a.reverse_bits() >> pad, false),
        (O::Countbits, &[a]) => (a.count_ones() as u64, true),
        (O::FirstbitLo, &[a]) => index(a.trailing_zeros()),
        (O::FirstbitHi, &[a]) => index((a << pad).leading_zeros()),
        (O::FirstbitSHi, &[a]) if s(a) < 0 => index((a << pad).leading_ones()),
        (O::FirstbitSHi, &[a]) => index((a << pad).leading_zeros()),
        (O::IMax, &[a, b]) => (if s(a) > s(b) { a } else { b }, false),
        (O::IMin, &[a, b]) => (if s(a) < s(b) { a } else { b }, false),
        (O::UMax, &[a, b]) => (a.max(b), false),
        (O::UMin, &[a, b]) => (a.min(b), false),
        (O::IMad | O::UMad, &[a, b, c]) => (a.wrapping_mul(b).wrapping_add(c), false),
        _ => return None,
    })
}

macro_rules! float_fold {
    ($name:ident, $t:ty, $u:ty, $mant:expr, $bias:expr) => {
        fn $name(op: OpCode, x: &[$t]) -> Option<$t> {
            use OpCode as O;
            if !x.iter().all(|v| v.is_finite()) {
                return None;
            }
            // 2^n from its bits: a normal number, or one bit of a subnormal
            let pow2 = |n: i32| {
                if n > $bias || n < 1 - $bias - $mant {
                    None
                } else if n >= 1 - $bias {
                    Some(<$t>::from_bits(((n + $bias) as $u) << $mant))
                } else {
                    Some(<$t>::from_bits((1 as $u) << (n - (1 - $bias - $mant))))
                }
            };
            Some(match (op, x) {
                (O::FAbs, &[a]) => a.abs(),
                // std::max(std::min(x, 1.0), 0.0), which keeps -0.0
                (O::Saturate, &[a]) => {
                    let m = if 1.0 < a { 1.0 } else { a };
                    if m < 0.0 {
                        0.0
                    } else {
                        m
                    }
                }
                (O::Round_ne, &[a]) => a.round_ties_even(),
                (O::Round_ni, &[a]) => a.floor(),
                (O::Round_pi, &[a]) => a.ceil(),
                (O::Round_z, &[a]) => a.trunc(),
                (O::Frc, &[a]) if a >= 0.0 => a - a.floor(),
                (O::Sqrt, &[a]) if a >= 0.0 => a.sqrt(),
                (O::Rsqrt, &[a]) if a >= 0.0 => {
                    let front_end = (1.0 / a.sqrt() as f64) as $t;
                    let propagated = (1.0 / (a as f64).sqrt()) as $t;
                    if front_end.to_bits() != propagated.to_bits() {
                        return None;
                    }
                    front_end
                }
                (O::Exp, &[a]) if a == a.trunc() && a.abs() < 4096.0 => pow2(a as i32)?,
                (O::Log, &[a]) if a.is_normal() && a > 0.0 && a.to_bits() & (((1 as $u) << $mant) - 1) == 0 => ((a.to_bits() >> $mant) as i32 - $bias) as $t,
                (O::FMax, &[a, b]) if a != b || a.to_bits() == b.to_bits() => {
                    if a > b {
                        a
                    } else {
                        b
                    }
                }
                (O::FMin, &[a, b]) if a != b || a.to_bits() == b.to_bits() => {
                    if a < b {
                        a
                    } else {
                        b
                    }
                }
                // two roundings, as APFloat multiplies and then adds
                (O::FMad, &[a, b, c]) => a * b + c,
                (O::Fma, &[a, b, c]) => a.mul_add(b, c),
                (O::Dot2 | O::Dot3 | O::Dot4, _) if x.len() % 2 == 0 => {
                    let (a, b) = x.split_at(x.len() / 2);
                    let mut sum: $t = 0.0;
                    for (p, q) in a.iter().zip(b) {
                        sum += p * q;
                    }
                    sum
                }
                _ => return None,
            })
        }
    };
}
float_fold!(float32, f32, u32, 23, 127);
float_fold!(float64, f64, u64, 52, 1023);

#[cfg(test)]
mod tests {
    use super::*;
    use OpCode as O;

    fn f(op: OpCode, x: &[f32]) -> Option<u32> {
        float32(op, x).map(f32::to_bits)
    }

    #[test]
    fn exact_float_operations_fold() {
        assert_eq!(f(O::Exp, &[-14.0]), Some(0x3880_0000));
        assert_eq!(f(O::Exp, &[127.0]), Some(0x7F00_0000));
        assert_eq!(f(O::Exp, &[-126.0]), Some(0x0080_0000));
        assert_eq!(f(O::Exp, &[-127.0]), Some(0x0040_0000));
        assert_eq!(f(O::Exp, &[-149.0]), Some(1));
        assert_eq!(float64(O::Exp, &[-1074.0]).map(f64::to_bits), Some(1));
        assert_eq!(float64(O::Exp, &[1023.0]), Some(f64::MAX / (2.0 - f64::EPSILON)));
        assert_eq!(f(O::Log, &[8.0]), Some(3f32.to_bits()));
        assert_eq!(f(O::Log, &[0.25]), Some((-2f32).to_bits()));
        assert_eq!(f(O::Sqrt, &[2.0]), Some(0x3FB5_04F3));
        assert_eq!(f(O::Rsqrt, &[4.0]), Some(0.5f32.to_bits()));
        assert_eq!(f(O::Round_ne, &[2.5]), Some(2f32.to_bits()));
        assert_eq!(f(O::Round_ne, &[-3.5]), Some((-4f32).to_bits()));
        assert_eq!(f(O::Round_ni, &[-0.5]), Some((-1f32).to_bits()));
        assert_eq!(f(O::Round_pi, &[-0.5]), Some((-0f32).to_bits()));
        assert_eq!(f(O::Frc, &[1.25]), Some(0.25f32.to_bits()));
        assert_eq!(f(O::Saturate, &[-0.0]), Some((-0f32).to_bits()));
        assert_eq!(f(O::Saturate, &[1.5]), Some(1f32.to_bits()));
        assert_eq!(f(O::FMad, &[2.0, 3.0, 0.1]), Some(0x40C3_3333));
        assert_eq!(f(O::Dot3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), Some(32f32.to_bits()));
        assert_eq!(f(O::FMax, &[1.0, 2.0]), Some(2f32.to_bits()));
        assert_eq!(f(O::FMin, &[-0.0, -0.0]), Some((-0f32).to_bits()));
    }

    #[test]
    fn what_dxc_evaluates_with_its_c_library_does_not_fold() {
        for op in [O::Sin, O::Cos, O::Tan, O::Asin, O::Acos, O::Atan, O::Hsin, O::Hcos, O::Htan] {
            assert_eq!(f(op, &[0.5]), None, "{op:?}");
        }
        assert_eq!(f(O::Exp, &[0.5]), None);
        assert_eq!(f(O::Exp, &[128.0]), None);
        assert_eq!(f(O::Exp, &[-150.0]), None);
        assert_eq!(f(O::Log, &[3.0]), None);
        assert_eq!(f(O::Log, &[0.0]), None);
        assert_eq!(f(O::Log, &[-2.0]), None);
        // DXC's two evaluators disagree, or there is no single answer
        assert_eq!(f(O::Frc, &[-1.25]), None);
        assert_eq!(f(O::FMax, &[0.0, -0.0]), None);
        assert_eq!(f(O::FMin, &[-0.0, 0.0]), None);
        assert_eq!(f(O::Sqrt, &[-1.0]), None);
        assert_eq!(f(O::FAbs, &[f32::NAN]), None);
        assert_eq!(f(O::FMin, &[f32::NAN, 1.0]), None);
        assert_eq!(f(O::Sqrt, &[f32::INFINITY]), None);
        assert_eq!(f(O::LegacyF16ToF32, &[1.0]), None);
    }

    #[test]
    fn integer_operations_fold() {
        assert_eq!(int(O::Bfrev, 32, &[40]), Some((335544320, false)));
        assert_eq!(int(O::Bfrev, 64, &[1]), Some((1 << 63, false)));
        assert_eq!(int(O::Countbits, 32, &[0xF0F0]), Some((8, true)));
        assert_eq!(int(O::FirstbitLo, 32, &[0x100]), Some((8, true)));
        assert_eq!(int(O::FirstbitLo, 32, &[0]), Some((0xFFFF_FFFF, true)));
        assert_eq!(int(O::FirstbitLo, 64, &[0]), Some((0xFFFF_FFFF, true)));
        assert_eq!(int(O::FirstbitHi, 32, &[0x100]), Some((23, true)));
        assert_eq!(int(O::FirstbitHi, 64, &[0x100]), Some((55, true)));
        assert_eq!(int(O::FirstbitHi, 32, &[0]), Some((0xFFFF_FFFF, true)));
        assert_eq!(int(O::FirstbitSHi, 32, &[0xFFFF_FF00]), Some((24, true)));
        assert_eq!(int(O::FirstbitSHi, 32, &[0xFFFF_FFFF]), Some((0xFFFF_FFFF, true)));
        assert_eq!(int(O::FirstbitSHi, 32, &[0x100]), Some((23, true)));
        assert_eq!(int(O::IMax, 32, &[0xFFFF_FFFF, 5]), Some((5, false)));
        assert_eq!(int(O::UMax, 32, &[0xFFFF_FFFF, 5]), Some((0xFFFF_FFFF, false)));
        assert_eq!(int(O::IMin, 64, &[u64::MAX, 5]), Some((u64::MAX, false)));
        assert_eq!(int(O::UMin, 32, &[11, 77]), Some((11, false)));
        assert_eq!(int(O::UMad, 32, &[3, 4, 5]), Some((17, false)));
        assert_eq!(int(O::Dot4AddU8Packed, 32, &[3, 4, 5]), None);
    }
}
