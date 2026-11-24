//! Raw (`ByteAddressBuffer`), structured and typed (`Buffer<T>`) buffers:
//! `dx.op.rawBufferLoad` / `rawBufferStore` / `bufferLoad` / `bufferStore`, `getDimensions`
//! and the atomics. DXC: TranslateBufLoad, TranslateStore, TranslateGetDimensions.

use super::atomic::{self, Target};
use super::{called, components, no_overload, overload, scalar, status, undef, vector};
use crate::consts::{OpCode, ResourceKind};
use crate::dxop;
use crate::hlsl::emit::Emit;
use crate::hlsl::intrinsics::Called;
use crate::hlsl::types::{HTy, ResTy, Sk, Val};
use crate::ir::{Ty, Value};
use crate::Result;

/// The i8 component mask of a raw buffer access of `n` components.
fn mask(e: &mut Emit, n: usize) -> Value {
    e.m.const_i8((1u8 << n) - 1)
}

/// Bytes of one scalar of kind `k` in buffer memory.
fn scalar_bytes(k: Sk) -> u32 {
    if k.is_64() {
        8
    } else {
        4
    }
}

pub(super) fn method(e: &mut Emit, ty: &ResTy, h: Value, name: &str, args: &[Val]) -> Result<Called> {
    match (ty.kind, name) {
        (ResourceKind::RawBuffer, "Load" | "Load2" | "Load3" | "Load4") => {
            let n = name[4..].parse().unwrap_or(1);
            if args.is_empty() || args.len() > 2 {
                return no_overload(e, name, args);
            }
            let addr = scalar(e, &args[0], Sk::Uint)?;
            let (v, res_ret) = raw_load(e, h, addr, n);
            let mut c = called(v);
            if args.len() == 2 {
                c.outs.push((1, status(e, res_ret)));
            }
            Ok(c)
        }
        (ResourceKind::RawBuffer, "Store" | "Store2" | "Store3" | "Store4") => {
            if !ty.uav {
                return e.err(format!("{name} on a read-only buffer"));
            }
            if args.len() != 2 {
                return no_overload(e, name, args);
            }
            let addr = scalar(e, &args[0], Sk::Uint)?;
            // Store is templated on its value; StoreN takes uintN
            let value = if name == "Store" {
                match args[1].ty {
                    HTy::Num(k, 1, 1) if matches!(k.concrete(), Sk::Int | Sk::Uint | Sk::Float) => vector(e, &args[1], k.concrete(), 1)?,
                    other => return e.unsupported(format!("ByteAddressBuffer Store of {other:?}")),
                }
            } else {
                vector(e, &args[1], Sk::Uint, name[5..].parse().unwrap_or(1))?
            };
            let u = undef(e, Ty::I32);
            raw_store(e, h, addr, u, &value, 4);
            Ok(called(Val::void()))
        }
        (ResourceKind::RawBuffer, _) if atomic::is_atomic(name) => {
            if !ty.uav {
                return e.err(format!("{name} on a read-only buffer"));
            }
            let Some(addr) = args.first() else { return no_overload(e, name, args) };
            let addr = scalar(e, addr, Sk::Uint)?;
            let u = undef(e, Ty::I32);
            atomic::on_address(e, Target { h, coords: [addr, u, u] }, name, args)
        }
        (ResourceKind::StructuredBuffer | ResourceKind::TypedBuffer, "Load") => {
            if args.is_empty() || args.len() > 2 {
                return no_overload(e, name, args);
            }
            let (v, res_ret) = element_load(e, ty, h, &args[0])?;
            let mut c = called(v);
            if args.len() == 2 {
                c.outs.push((1, status(e, res_ret)));
            }
            Ok(c)
        }
        (_, "GetDimensions") => {
            // (out width), or (out count, out stride) of a structured buffer
            let structured = ty.kind == ResourceKind::StructuredBuffer;
            if args.len() != 1 + structured as usize {
                return no_overload(e, name, args);
            }
            let dims_ty = dxop::dimensions_ty(&mut e.m);
            let _ = dims_ty;
            let u = undef(e, Ty::I32);
            let dims = e.dx(OpCode::GetDimensions, Ty::VOID, &[h, u]);
            let width = e.extract(Ty::I32, dims, 0);
            let mut c = called(Val::void());
            c.outs.push((0, Val::scalar(Sk::Uint, width)));
            if structured {
                c.outs.push((1, Val::scalar(Sk::Uint, e.c_u32(ty.stride))));
            }
            Ok(c)
        }
        // IncrementCounter / DecrementCounter mark the resource as having a counter, which
        // changes its record and every annotation of its handle: not expressible per call
        (k, _) => e.unsupported(format!("method {name} of {k:?}")),
    }
}

/// `rawBufferLoad.i32(handle, address, undef, mask, 4)`: `n` dwords as a uint vector, and the
/// `ResRet`.
fn raw_load(e: &mut Emit, h: Value, addr: Value, n: usize) -> (Val, Value) {
    let u = undef(e, Ty::I32);
    let m = mask(e, n);
    let align = e.c_u32(4);
    let r = e.dx(OpCode::RawBufferLoad, Ty::I32, &[h, addr, u, m, align]);
    (components(e, r, Sk::Uint, n), r)
}

/// `rawBufferStore(handle, index, offset, v0..v3, mask, alignment)`: values the access does
/// not have are undef.
fn raw_store(e: &mut Emit, h: Value, index: Value, offset: Value, value: &Val, align: u32) {
    let ir = match value.ty {
        HTy::Num(k, ..) => k.ir(),
        _ => Ty::I32,
    };
    let u = undef(e, ir);
    let mut v = [u; 4];
    v[..value.len()].copy_from_slice(value.comps());
    let m = mask(e, value.len());
    let align = e.c_u32(align);
    e.dx(OpCode::RawBufferStore, ir, &[h, index, offset, v[0], v[1], v[2], v[3], m, align]);
}

/// The element of a structured or typed buffer at `index`: a structured buffer is read with
/// `rawBufferLoad(handle, index, 0, mask, alignment)` (the alignment is the scalar size, at
/// most 8), a typed one with `bufferLoad(handle, index, undef)`.
fn element_load(e: &mut Emit, ty: &ResTy, h: Value, index: &Val) -> Result<(Val, Value)> {
    let idx = scalar(e, index, Sk::Int)?;
    let ovl = overload(e, ty.elem)?;
    let n = ty.elem_comps as usize;
    if n > 4 {
        return e.unsupported("buffer element of more than 4 components");
    }
    let r = if ty.kind == ResourceKind::StructuredBuffer {
        let zero = e.c_u32(0);
        let m = mask(e, n);
        let align = e.c_u32(scalar_bytes(ty.elem));
        e.dx(OpCode::RawBufferLoad, ovl, &[h, idx, zero, m, align])
    } else {
        let u = undef(e, Ty::I32);
        e.dx(OpCode::BufferLoad, ovl, &[h, idx, u])
    };
    Ok((components(e, r, ty.elem, n), r))
}

pub(super) fn index_load(e: &mut Emit, ty: &ResTy, h: Value, index: &Val) -> Result<Val> {
    Ok(element_load(e, ty, h, index)?.0)
}

/// `buffer[index] = value`: `rawBufferStore(handle, index, 0, ...)` for a structured buffer;
/// `bufferStore(handle, index, undef, v0..v3, 15)` for a typed one, where the values past the
/// element's components repeat the first (DXC fills a typed store that way) and the mask is
/// always all four.
pub(super) fn index_store(e: &mut Emit, ty: &ResTy, h: Value, index: &Val, value: &Val) -> Result<()> {
    let idx = scalar(e, index, Sk::Int)?;
    let ovl = overload(e, ty.elem)?;
    let n = ty.elem_comps as usize;
    if n > 4 {
        return e.unsupported("buffer element of more than 4 components");
    }
    let value = vector(e, value, ty.elem, n)?;
    if ty.kind == ResourceKind::StructuredBuffer {
        let zero = e.c_u32(0);
        raw_store(e, h, idx, zero, &value, scalar_bytes(ty.elem));
    } else {
        let u = undef(e, Ty::I32);
        let c = value.comps();
        let v = |i: usize| if i < n { c[i] } else { c[0] };
        let m = e.m.const_i8(15);
        e.dx(OpCode::BufferStore, ovl, &[h, idx, u, v(0), v(1), v(2), v(3), m]);
    }
    Ok(())
}

/// The atomic target of `buffer[index]`: (index, 0, undef) in a structured buffer (the second
/// coordinate is the byte offset in the element), (index, undef, undef) in a typed one.
pub(super) fn element(e: &mut Emit, ty: &ResTy, h: Value, index: &Val) -> Result<Target> {
    let idx = scalar(e, index, Sk::Int)?;
    let u = undef(e, Ty::I32);
    let c1 = if ty.kind == ResourceKind::StructuredBuffer { e.c_u32(0) } else { u };
    Ok(Target { h, coords: [idx, c1, u] })
}
