//! The `Interlocked*` family: `dx.op.atomicBinOp` and `dx.op.atomicCompareExchange` on a
//! byte address of a raw buffer (the method form, `mem.InterlockedAdd(addr, v, old)`) or on
//! an element of a typed resource (the function form, `InterlockedAdd(tex[coord], v, old)`).
//! DXC: TranslateAtomicBinaryOperation / TranslateAtomicCmpXChg.

use super::{conv, no_overload, scalar};
use crate::consts::{AtomicBinOpCode, OpCode};
use crate::hlsl::emit::Emit;
use crate::hlsl::intrinsics::Called;
use crate::hlsl::types::{HTy, ResTy, Sk, Val};
use crate::ir::Value;
use crate::Result;

/// Where an atomic works: the handle and the three coordinate operands (byte address, or
/// element index and offset, or texel coordinates; `undef` where the resource has none).
pub(super) struct Target {
    pub h: Value,
    pub coords: [Value; 3],
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Family {
    Add,
    And,
    Or,
    Xor,
    Min,
    Max,
    Exchange,
    /// with the `out original` argument
    CompareExchange,
    CompareStore,
}

/// `InterlockedMin64` → (Min, true).
fn family(name: &str) -> Option<(Family, bool)> {
    let rest = name.strip_prefix("Interlocked")?;
    let (rest, wide) = match rest.strip_suffix("64") {
        Some(r) => (r, true),
        None => (rest, false),
    };
    let f = match rest {
        "Add" => Family::Add,
        "And" => Family::And,
        "Or" => Family::Or,
        "Xor" => Family::Xor,
        "Min" => Family::Min,
        "Max" => Family::Max,
        "Exchange" => Family::Exchange,
        "CompareExchange" => Family::CompareExchange,
        "CompareStore" => Family::CompareStore,
        _ => return None,
    };
    Some((f, wide))
}

pub(super) fn is_atomic(name: &str) -> bool {
    name.starts_with("Interlocked")
}

/// The method form on a raw buffer: `args` are (address, value[, original]) or (address,
/// compare, value[, original]); the address is already in `target`.
///
/// Parameter types are DXC's (gen_intrin_main.txt): the 32-bit forms take `uint` and return
/// the original as `uint`; `Min` / `Max` take any 32-bit integer and are signed
/// (`IMin` / `IMax`) or unsigned (`UMin` / `UMax`) by the type of the value argument; the
/// `64` forms work on 64-bit integers the same way.
pub(super) fn on_address(e: &mut Emit, target: Target, name: &str, args: &[Val]) -> Result<Called> {
    let Some((f, wide)) = family(name) else {
        return e.unsupported(format!("method {name} of a raw buffer"));
    };
    let value_at = if matches!(f, Family::CompareExchange | Family::CompareStore) { 2 } else { 1 };
    let Some(value) = args.get(value_at) else { return no_overload(e, name, args) };
    let signed = matches!(f, Family::Min | Family::Max | Family::Exchange) && matches!(value.ty, HTy::Num(k, ..) if k.is_signed());
    let k = match (wide, signed) {
        (false, false) => Sk::Uint,
        (false, true) => Sk::Int,
        (true, false) => Sk::Uint64,
        (true, true) => Sk::Int64,
    };
    // the 32-bit forms declare `out uint original`
    let out = if wide { k } else { Sk::Uint };
    lower(e, target, f, k, out, name, args, 1)
}

/// The function form on an element of `ty`: `args` are (value[, original]) or (compare,
/// value[, original]). The operation works on the element type, which also decides between
/// signed and unsigned min / max (the intrinsic overloads on its destination).
pub(super) fn on_element(e: &mut Emit, ty: &ResTy, target: Target, name: &str, args: &[Val]) -> Result<Called> {
    let Some((f, wide)) = family(name) else {
        return e.unsupported(format!("intrinsic {name} on a resource element"));
    };
    if wide {
        // the function form has no 64 suffix: the element type selects the width
        return e.err(format!("{name} is not an intrinsic function"));
    }
    if ty.elem_comps != 1 || !matches!(ty.elem, Sk::Int | Sk::Uint | Sk::Int64 | Sk::Uint64) {
        return e.unsupported(format!("{name} on an element of {:?}{}", ty.elem, ty.elem_comps));
    }
    lower(e, target, f, ty.elem, ty.elem, name, args, 0)
}

/// Emit the call. `args[first..]` are the value arguments; `k` is the kind the operation works
/// on, `out` the kind of the original value handed back. `out` indices count from the start of
/// the intrinsic's argument list, where a destination precedes `args` when `first` is 0.
#[allow(clippy::too_many_arguments)]
fn lower(e: &mut Emit, target: Target, f: Family, k: Sk, out: Sk, name: &str, args: &[Val], first: usize) -> Result<Called> {
    let rest = &args[first.min(args.len())..];
    let (values, with_original) = match f {
        Family::CompareExchange => (2, true),
        Family::CompareStore => (2, false),
        Family::Exchange => (1, true),
        _ => (1, rest.len() == 2),
    };
    if rest.len() != values + with_original as usize {
        return no_overload(e, name, args);
    }
    let ovl = k.ir();
    let [c0, c1, c2] = target.coords;
    let original = if values == 2 {
        let compare = scalar(e, &rest[0], k)?;
        let value = scalar(e, &rest[1], k)?;
        e.dx(OpCode::AtomicCompareExchange, ovl, &[target.h, c0, c1, c2, compare, value])
    } else {
        let value = scalar(e, &rest[0], k)?;
        let op = match (f, k.is_signed()) {
            (Family::Add, _) => AtomicBinOpCode::Add,
            (Family::And, _) => AtomicBinOpCode::And,
            (Family::Or, _) => AtomicBinOpCode::Or,
            (Family::Xor, _) => AtomicBinOpCode::Xor,
            (Family::Min, true) => AtomicBinOpCode::IMin,
            (Family::Min, false) => AtomicBinOpCode::UMin,
            (Family::Max, true) => AtomicBinOpCode::IMax,
            (Family::Max, false) => AtomicBinOpCode::UMax,
            _ => AtomicBinOpCode::Exchange,
        };
        let op = e.c_u32(op as u32);
        e.dx(OpCode::AtomicBinOp, ovl, &[target.h, op, c0, c1, c2, value])
    };
    let mut called = super::called(Val::void());
    if with_original {
        // the destination of the function form is argument 0 of the intrinsic
        let index = values + if first == 0 { 1 } else { first };
        let v = conv(e, &Val::scalar(k, original), HTy::scalar(out))?;
        called.outs.push((index, v));
    }
    Ok(called)
}
