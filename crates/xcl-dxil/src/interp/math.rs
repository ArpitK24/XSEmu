//! Float and bit arithmetic of the interpreter, with the rules that make two shapes of one
//! program agree bit for bit.
//!
//! * NaN: an operation without a NaN operand that produces one gives the positive quiet NaN
//!   (what LLVM's constant folder gives, so a folded and an executed `0.0 / 0.0` agree; the
//!   x86 host would give the negative one), and a NaN operand comes back quieted, the first
//!   one winning. This does not depend on the host. The IR does not say which NaN an
//!   operation returns, and LLVM orders the operands of add and multiply as it likes: where
//!   two compilers may so disagree (two different NaN operands, a signalling one that a
//!   folded x * 1.0 would leave signalling) the run is marked ([take_nan_open]). Stored
//!   words that differ only in the payload of a NaN are a class of their own in the
//!   comparison, and any other difference after the mark is not called a miscompile.
//! * Transcendental functions are evaluated in double and rounded once, as DXC's own
//!   constant folder does (`DxilConstantFolding.cpp`), so both programs call one function
//!   with the same bits, and a call DXC folded matches the one it left.
//! * `FMax` / `FMin` are LLVM's maxnum / minnum (the other operand when one is NaN, the
//!   first when they compare equal), `FMad` is a multiply and an add rounded separately:
//!   again the folder's definitions.

use crate::consts::OpCode;
use std::cell::Cell;

const QUIET32: u32 = 0x0040_0000;
const QUIET64: u64 = 0x0008_0000_0000_0000;
pub const NAN32: u32 = 0x7fc0_0000;
pub const NAN64: u64 = 0x7ff8_0000_0000_0000;

thread_local! {
    /// this thread ran an operation whose NaN result the IR leaves open
    static NAN_OPEN: Cell<bool> = const { Cell::new(false) };
}

/// Whether this thread has run an operation whose NaN result the IR leaves open since the
/// last call.
pub fn take_nan_open() -> bool {
    NAN_OPEN.with(|f| f.replace(false))
}

/// The NaN an operation returns for its NaN operands (bits, widened; 0 for not a NaN).
#[cold]
fn pick_nan(a: u64, b: u64, quiet: u64, default: u64) -> u64 {
    if a | b == 0 {
        return default;
    }
    if (a != 0 && a & quiet == 0) || (b != 0 && b & quiet == 0) || (a != 0 && b != 0 && a != b) {
        NAN_OPEN.with(|f| f.set(true));
    }
    if a != 0 {
        a | quiet
    } else {
        b | quiet
    }
}

#[inline]
fn nan_bits32(x: f32) -> u64 {
    if x.is_nan() {
        x.to_bits() as u64
    } else {
        0
    }
}

#[inline]
fn nan_bits64(x: f64) -> u64 {
    if x.is_nan() {
        x.to_bits()
    } else {
        0
    }
}

#[inline]
pub fn nan1_32(r: f32, a: f32) -> f32 {
    if r.is_nan() {
        return f32::from_bits(pick_nan(nan_bits32(a), 0, QUIET32 as u64, NAN32 as u64) as u32);
    }
    r
}

#[inline]
pub fn nan2_32(r: f32, a: f32, b: f32) -> f32 {
    if r.is_nan() {
        return f32::from_bits(pick_nan(nan_bits32(a), nan_bits32(b), QUIET32 as u64, NAN32 as u64) as u32);
    }
    r
}

#[inline]
pub fn nan1_64(r: f64, a: f64) -> f64 {
    if r.is_nan() {
        return f64::from_bits(pick_nan(nan_bits64(a), 0, QUIET64, NAN64));
    }
    r
}

#[inline]
pub fn nan2_64(r: f64, a: f64, b: f64) -> f64 {
    if r.is_nan() {
        return f64::from_bits(pick_nan(nan_bits64(a), nan_bits64(b), QUIET64, NAN64));
    }
    r
}

/// binary16 → binary32, exact.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = (h >> 10) as u32 & 0x1f;
    let man = h as u32 & 0x3ff;
    let bits = if exp == 0x1f {
        sign | 0x7f80_0000 | man << 13
    } else if exp != 0 {
        sign | (exp + 112) << 23 | man << 13
    } else if man == 0 {
        sign
    } else {
        let shift = man.leading_zeros() - 21;
        sign | (113 - shift) << 23 | (man << shift & 0x3ff) << 13
    };
    f32::from_bits(bits)
}

/// binary64 → binary16, round to nearest even (binary32 widens exactly first).
pub fn f64_to_f16(x: f64) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let man = bits & 0x000f_ffff_ffff_ffff;
    if exp == 0x7ff {
        if man == 0 {
            return sign | 0x7c00;
        }
        // keep the top payload bits, quiet
        return sign | 0x7e00 | (man >> 42) as u16;
    }
    let e = exp - 1023;
    if e >= 16 {
        return sign | 0x7c00;
    }
    if exp == 0 || e < -25 {
        // zero, a double denormal, or below half the smallest half denormal
        return sign;
    }
    // the significand with its leading one, and how far to shift it down to 10 fraction
    // bits (normal) or to the denormal position
    let sig = man | 1 << 52;
    let shift = if e >= -14 { 42 } else { 42 + (-14 - e) as u32 };
    let mut out = sig >> shift;
    let rem = sig & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    if rem > half || (rem == half && out & 1 != 0) {
        out += 1;
    }
    // for a normal, out has the leading one at bit 10: the exponent field is added so that
    // a carry out of the fraction lands in it (and reaches infinity at the top)
    let biased = if e >= -14 { (e + 15) as u64 - 1 } else { 0 };
    sign | ((biased << 10) + out) as u16
}

pub fn f32_to_f16(x: f32) -> u16 {
    f64_to_f16(x as f64)
}

/// A float of `w` bits as a double (exact).
#[inline]
pub fn widen(bits: u64, w: u8) -> f64 {
    match w {
        64 => f64::from_bits(bits),
        32 => f32::from_bits(bits as u32) as f64,
        _ => f16_to_f32(bits as u16) as f64,
    }
}

/// A double rounded to a float of `w` bits.
#[inline]
pub fn narrow(x: f64, w: u8) -> u64 {
    match w {
        64 => x.to_bits(),
        32 => (x as f32).to_bits() as u64,
        _ => f64_to_f16(x) as u64,
    }
}

fn native(op: u32, x: f64) -> Option<f64> {
    Some(match op {
        _ if op == OpCode::Cos as u32 => x.cos(),
        _ if op == OpCode::Sin as u32 => x.sin(),
        _ if op == OpCode::Tan as u32 => x.tan(),
        _ if op == OpCode::Acos as u32 => x.acos(),
        _ if op == OpCode::Asin as u32 => x.asin(),
        _ if op == OpCode::Atan as u32 => x.atan(),
        _ if op == OpCode::Hcos as u32 => x.cosh(),
        _ if op == OpCode::Hsin as u32 => x.sinh(),
        _ if op == OpCode::Htan as u32 => x.tanh(),
        _ if op == OpCode::Exp as u32 => x.exp2(),
        _ if op == OpCode::Log as u32 => x.log2(),
        _ if op == OpCode::Rsqrt as u32 => 1.0 / x.sqrt(),
        _ => return None,
    })
}

/// A unary float dx.op on binary32 (FAbs … Round_z).
pub fn unary32(op: u32, x: f32) -> f32 {
    let r = match op {
        _ if op == OpCode::FAbs as u32 => return f32::from_bits(x.to_bits() & 0x7fff_ffff),
        _ if op == OpCode::Saturate as u32 => {
            return if x.is_nan() {
                0.0
            } else if x < 0.0 {
                0.0
            } else if x > 1.0 {
                1.0
            } else {
                x
            }
        }
        _ if op == OpCode::Frc as u32 => x - x.floor(),
        _ if op == OpCode::Sqrt as u32 => x.sqrt(),
        _ if op == OpCode::Round_ne as u32 => x.round_ties_even(),
        _ if op == OpCode::Round_ni as u32 => x.floor(),
        _ if op == OpCode::Round_pi as u32 => x.ceil(),
        _ if op == OpCode::Round_z as u32 => x.trunc(),
        _ => native(op, x as f64).unwrap_or(f64::NAN) as f32,
    };
    nan1_32(r, x)
}

pub fn unary64(op: u32, x: f64) -> f64 {
    let r = match op {
        _ if op == OpCode::FAbs as u32 => return f64::from_bits(x.to_bits() & !(1 << 63)),
        _ if op == OpCode::Saturate as u32 => {
            return if x.is_nan() {
                0.0
            } else if x < 0.0 {
                0.0
            } else if x > 1.0 {
                1.0
            } else {
                x
            }
        }
        _ if op == OpCode::Frc as u32 => x - x.floor(),
        _ if op == OpCode::Sqrt as u32 => x.sqrt(),
        _ if op == OpCode::Round_ne as u32 => x.round_ties_even(),
        _ if op == OpCode::Round_ni as u32 => x.floor(),
        _ if op == OpCode::Round_pi as u32 => x.ceil(),
        _ if op == OpCode::Round_z as u32 => x.trunc(),
        _ => native(op, x).unwrap_or(f64::NAN),
    };
    nan1_64(r, x)
}

/// LLVM's maxnum.
#[inline]
pub fn fmax<T: PartialOrd + Copy>(a: T, b: T, a_nan: bool, b_nan: bool) -> T {
    if a_nan {
        b
    } else if b_nan {
        a
    } else if a < b {
        b
    } else {
        a
    }
}

/// LLVM's minnum.
#[inline]
pub fn fmin<T: PartialOrd + Copy>(a: T, b: T, a_nan: bool, b_nan: bool) -> T {
    if a_nan {
        b
    } else if b_nan {
        a
    } else if b < a {
        b
    } else {
        a
    }
}

/// An LLVM float predicate (ir::FPred numbers) on exact doubles.
#[inline]
pub fn fcmp(pred: u16, a: f64, b: f64) -> bool {
    let uno = a.is_nan() || b.is_nan();
    match pred {
        1 => a == b,
        2 => a > b,
        3 => a >= b,
        4 => a < b,
        5 => a <= b,
        6 => !uno && a != b,
        7 => !uno,
        8 => uno,
        9 => uno || a == b,
        10 => uno || a > b,
        11 => uno || a >= b,
        12 => uno || a < b,
        13 => uno || a <= b,
        14 => a != b,
        15 => true,
        _ => false,
    }
}

/// DXC's bit field extract (`ComputeBFE`): width and offset are taken modulo the bit width.
pub fn bfe(bits: u32, width: u64, offset: u64, val: u64, signed: bool) -> u64 {
    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let width = (width & (bits as u64 - 1)) as u32;
    let offset = (offset & (bits as u64 - 1)) as u32;
    let val = val & mask;
    let shr = |v: u64, by: u32| -> u64 {
        if signed {
            let s = 64 - bits;
            ((((v << s) as i64) >> s) >> by) as u64 & mask
        } else {
            v >> by
        }
    };
    if width == 0 {
        0
    } else if width + offset < bits {
        shr((val << (bits - (width + offset))) & mask, bits - width)
    } else {
        shr(val, offset)
    }
}

/// DXC's bit field insert: `width` bits of `src` at `offset` into `dst`.
pub fn bfi(bits: u32, width: u64, offset: u64, src: u64, dst: u64) -> u64 {
    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let width = (width & (bits as u64 - 1)) as u32;
    let offset = (offset & (bits as u64 - 1)) as u32;
    let field = (((1u64 << width) - 1) << offset) & mask;
    ((src << offset) & field | dst & !field) & mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversions_are_exact_and_round_to_even() {
        for h in 0..=0xffffu16 {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f16_to_f32(f32_to_f16(f)).is_nan());
            } else {
                assert_eq!(f32_to_f16(f), h, "{h:#x}");
            }
        }
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        // ties: 1 + 2^-11 is halfway between 0x3c00 and 0x3c01
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 / 2048.0), 0x3c02);
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        assert_eq!(f32_to_f16(65519.0), 0x7bff);
        assert_eq!(f32_to_f16(1e-10), 0);
        assert_eq!(f32_to_f16(-1e-10), 0x8000);
        // halfway to the smallest denormal rounds to even (zero), just above rounds up
        assert_eq!(f32_to_f16(f32::from_bits(0x3300_0000)), 0);
        assert_eq!(f32_to_f16(f32::from_bits(0x3300_0001)), 1);
    }

    #[test]
    fn nan_rules() {
        let inf = f32::INFINITY;
        assert_eq!(nan2_32(inf - inf, inf, inf).to_bits(), NAN32);
        let payload = f32::from_bits(0x7f80_1234);
        assert_eq!(nan2_32(payload + 1.0, payload, 1.0).to_bits(), 0x7fc0_1234);
        // signalling: a folded x * 1.0 would have kept it so
        assert!(take_nan_open());
        let (p, q) = (f32::from_bits(0x7fff_ffff), f32::from_bits(0xffff_feff));
        assert_eq!(nan2_32(p + q, p, q).to_bits(), 0x7fff_ffff);
        assert!(take_nan_open());
        assert_eq!(nan2_32(q + p, q, p).to_bits(), 0xffff_feff);
        assert!(take_nan_open());
        assert_eq!(nan2_32(p + 1.0, p, 1.0).to_bits(), 0x7fff_ffff);
        assert_eq!(nan2_32(p + p, p, p).to_bits(), 0x7fff_ffff);
        assert!(!take_nan_open());
        assert_eq!(unary32(OpCode::Sqrt as u32, -1.0).to_bits(), NAN32);
        assert_eq!(unary32(OpCode::Saturate as u32, f32::NAN), 0.0);
        assert_eq!(unary32(OpCode::Frc as u32, -0.25), 0.75);
        assert_eq!(unary32(OpCode::Round_ne as u32, 2.5), 2.0);
        assert_eq!(unary32(OpCode::Exp as u32, -14.0), 1.0 / 16384.0);
    }

    #[test]
    fn bit_fields() {
        assert_eq!(bfe(32, 8, 4, 0x0000_0ab0, false), 0xab);
        assert_eq!(bfe(32, 8, 4, 0x0000_0fb0, true), 0xffff_fffb);
        assert_eq!(bfe(32, 0, 4, 0xffff_ffff, false), 0);
        assert_eq!(bfe(32, 16, 24, 0x8000_0000, true), 0xffff_ff80);
        assert_eq!(bfi(32, 8, 8, 0xab, 0xffff_ffff), 0xffff_abff);
    }
}
