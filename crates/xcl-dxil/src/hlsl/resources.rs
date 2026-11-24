//! Buffers, textures, samplers and descriptor heaps: handle creation and every access
//! operation, as DXC 1.9 lowers them for SM 6.6 (`createHandleFromBinding` /
//! `createHandleFromHeap` + `annotateHandle`, then one dx.op per access). Owner: the resources
//! subagent.
//!
//! The reference is DXC's HLOperationLower.cpp (the `Translate*` functions for resource
//! methods and subscripts) and DxilResourceProperties.cpp; every access here has a fixture in
//! `tests/resources_fixtures.rs` that compares the calls emitted with what DXC emits for the
//! same HLSL. Layout:
//!
//! * [`props`]: resource types and records ([`res_ty`], [`declare`]) and the
//!   `ResourceProperties` words ([`properties`]),
//! * [`buffer`]: raw, structured and typed buffers,
//! * [`texture`]: sampling, gathers, loads, stores, dimensions, level of detail,
//! * [`atomic`]: the `Interlocked*` family on a buffer address or a texel.
//!
//! What is not handled returns [`crate::Error::Unsupported`] naming the member, and the caller
//! falls back to DXC; a call DXC itself would reject (wrong argument count, a sampler of the
//! wrong kind) is a source error.

use super::emit::Emit;
use super::intrinsics::Called;
use super::types::{HTy, ResTy, Sk, Val};
use crate::consts::{OpCode, ResourceClass, ResourceKind};
use crate::dxop;
use crate::ir::{CastOp, Ty, Value};
use crate::shader::Resource;
use crate::Result;

mod atomic;
mod buffer;
mod props;
mod texture;

pub use props::{declare, properties, record_properties, res_ty, sampler_properties};

/// `dx.op.createHandleFromBinding` for a register-bound resource, without the annotation:
/// the `%dx.types.ResBind` constant is { lower bound, upper bound, space, class }, the index
/// is the register itself (ranges of more than one register are not indexed here) and the
/// index is uniform. DXC creates this handle once, at the start of the entry function, and
/// annotates it where it is used ([`annotate`]); callers that do the same keep one call per
/// resource.
pub fn binding_handle(e: &mut Emit, res: &Resource) -> Value {
    let ty = dxop::res_bind_ty(&mut e.m);
    let upper = if res.range_size == u32::MAX { u32::MAX } else { res.lower_bound.wrapping_add(res.range_size).wrapping_sub(1) };
    let bind = if (res.lower_bound, upper, res.space, res.class) == (0, 0, 0, ResourceClass::SRV) {
        // an all-zero struct is a zeroinitializer in LLVM, and DXC's module has it that way
        e.m.zero(ty)
    } else {
        let fields = vec![e.c_u32(res.lower_bound), e.c_u32(upper), e.c_u32(res.space), e.m.const_i8(res.class as u8)];
        e.m.aggregate(ty, fields)
    };
    let index = e.c_u32(res.lower_bound);
    let uniform = e.c_bool(false);
    e.dx(OpCode::CreateHandleFromBinding, Ty::VOID, &[bind, index, uniform])
}

/// `dx.op.annotateHandle` with the two `%dx.types.ResourceProperties` words
/// ([`properties`], [`sampler_properties`], [`record_properties`]).
pub fn annotate(e: &mut Emit, handle: Value, words: [u32; 2]) -> Value {
    let props = match e.res_props.iter().find(|p| p.0 == words) {
        Some(&(_, v)) => v,
        None => {
            let ty = dxop::res_props_ty(&mut e.m);
            let v = if words == [0, 0] {
                e.m.zero(ty)
            } else {
                let fields = vec![e.c_u32(words[0]), e.c_u32(words[1])];
                e.m.aggregate(ty, fields)
            };
            e.res_props.push((words, v));
            v
        }
    };
    e.dx(OpCode::AnnotateHandle, Ty::VOID, &[handle, props])
}

/// The annotated handle of a register-bound resource (`RWByteAddressBuffer io : register(u0)`),
/// created at the insertion point. `res` is its record (already in the ShaderInfo's list);
/// `ty`: its HLSL type. The annotation is the record's ([`record_properties`]), as in DXC,
/// which is the type's for every record [`declare`] makes.
pub fn bound_handle(e: &mut Emit, res: &Resource, ty: &ResTy) -> Result<Value> {
    if res.class == ResourceClass::Sampler || res.kind != ty.kind {
        return Err(crate::Error::Internal(format!("resources::bound_handle: record {} is not a {:?}", res.name, ty.kind)));
    }
    let h = binding_handle(e, res);
    Ok(annotate(e, h, record_properties(res)))
}

/// The annotated handle of a register-bound sampler (`SamplerState s : register(s0, space1)`).
pub fn bound_sampler(e: &mut Emit, res: &Resource) -> Result<Value> {
    if res.class != ResourceClass::Sampler {
        return Err(crate::Error::Internal(format!("resources::bound_sampler: record {} is not a sampler", res.name)));
    }
    let h = binding_handle(e, res);
    Ok(annotate(e, h, record_properties(res)))
}

/// The annotated handle of `ResourceDescriptorHeap[index]` (or the sampler heap's) given the
/// type it is used as (`ty` None: a sampler, `comparison` says which kind):
/// `createHandleFromHeap(index, samplerHeap, nonUniformIndex)` + `annotateHandle`.
pub fn heap_handle(e: &mut Emit, index: Value, nonuniform: bool, ty: Option<&ResTy>, comparison: bool) -> Result<Value> {
    let sampler_heap = e.c_bool(ty.is_none());
    let nonuniform = e.c_bool(nonuniform);
    let h = e.dx(OpCode::CreateHandleFromHeap, Ty::VOID, &[index, sampler_heap, nonuniform]);
    let words = match ty {
        Some(t) => properties(t),
        None => sampler_properties(comparison),
    };
    Ok(annotate(e, h, words))
}

/// `obj.Method(args)` on a resource value (`obj.ty` is `HTy::Res`): Load*, Store*,
/// Interlocked*, Sample*, Gather*, GetDimensions, CalculateLevelOfDetail*, ...
/// Sampler arguments arrive as handle values (`HTy::Sampler`). Arguments are converted to the
/// parameter types of the overload the argument count selects; an `out` argument is passed
/// as any value of its type (only the type is read: `GetDimensions` converts to float outs)
/// and comes back in [`Called::outs`] under its index in `args`.
pub fn method(e: &mut Emit, obj: &Val, name: &str, args: &[Val]) -> Result<Called> {
    let (ty, h) = resource(e, obj)?;
    match ty.kind {
        ResourceKind::RawBuffer | ResourceKind::StructuredBuffer | ResourceKind::TypedBuffer => buffer::method(e, &ty, h, name, args),
        k if props::is_texture(k) => texture::method(e, &ty, h, name, args),
        k => e.unsupported(format!("method {name} of {k:?}")),
    }
}

/// `tex[coord]` as an rvalue (also `buffer[index]` of a typed or structured buffer): a load
/// of mip 0 (sample 0 of a multisampled texture).
pub fn index_load(e: &mut Emit, obj: &Val, coord: &Val) -> Result<Val> {
    let (ty, h) = resource(e, obj)?;
    match ty.kind {
        ResourceKind::StructuredBuffer | ResourceKind::TypedBuffer => buffer::index_load(e, &ty, h, coord),
        k if props::is_texture(k) => texture::index_load(e, &ty, h, coord),
        k => e.err(format!("{k:?} has no subscript operator")),
    }
}

/// `tex[coord] = value` (also `buffer[index] = value`): the whole element is written; a
/// store to one component (`tex[c].x = v`) is a load, an update and this.
pub fn index_store(e: &mut Emit, obj: &Val, coord: &Val, value: &Val) -> Result<()> {
    let (ty, h) = resource(e, obj)?;
    if !ty.uav {
        return e.err("assignment to an element of a read-only resource");
    }
    match ty.kind {
        ResourceKind::StructuredBuffer | ResourceKind::TypedBuffer => buffer::index_store(e, &ty, h, coord, value),
        k if props::is_texture(k) => texture::index_store(e, &ty, h, coord, value),
        k => e.err(format!("{k:?} has no subscript operator")),
    }
}

/// The intrinsic function form of an atomic on a resource element:
/// `InterlockedAdd(tex[coord], value, original)` and the rest of the family (`name` is the
/// function name; `args` are the arguments after the destination). Signed or unsigned min /
/// max follows the element type of the resource, as DXC's overload on the destination does.
/// `out` values come back under their index in the whole argument list (the destination is
/// argument 0, so `original` of `InterlockedAdd` is 2).
pub fn index_atomic(e: &mut Emit, obj: &Val, coord: &Val, name: &str, args: &[Val]) -> Result<Called> {
    let (ty, h) = resource(e, obj)?;
    if !ty.uav {
        return e.err(format!("{name} on an element of a read-only resource"));
    }
    let target = match ty.kind {
        ResourceKind::StructuredBuffer | ResourceKind::TypedBuffer => buffer::element(e, &ty, h, coord)?,
        k if props::is_texture(k) => texture::element(e, &ty, h, coord)?,
        k => return e.err(format!("{k:?} has no subscript operator")),
    };
    atomic::on_element(e, &ty, target, name, args)
}

fn resource(e: &Emit, obj: &Val) -> Result<(ResTy, Value)> {
    match obj.ty {
        HTy::Res(t) => Ok((t, obj.one())),
        other => e.err(format!("{other:?} is not a resource")),
    }
}

// ---------------------------------------------------------------------------------------
// shared by the submodules

/// The overload type of an access to elements of kind `k`.
fn overload(e: &Emit, k: Sk) -> Result<Ty> {
    match k {
        Sk::Float | Sk::Int | Sk::Uint | Sk::Int64 | Sk::Uint64 | Sk::Double => Ok(k.ir()),
        other => e.unsupported(format!("resource element of {other:?}")),
    }
}

/// Convert an argument to a parameter type. As [`Emit::convert`], but an integer constant
/// becomes a float constant (`SampleLevel(s, uv, 0)`), as DXC's constant folder leaves it.
fn conv(e: &mut Emit, v: &Val, to: HTy) -> Result<Val> {
    if v.ty == to {
        return Ok(v.clone());
    }
    if let (HTy::Num(from, ..), HTy::Num(Sk::Float, ..)) = (v.ty, to) {
        if !from.is_float() && v.comps().iter().any(|&c| e.m.as_int(c).is_some()) {
            let mut comps = [Value::NONE; 16];
            for (slot, &c) in comps.iter_mut().zip(v.comps()) {
                *slot = match e.m.as_int(c) {
                    Some(x) => {
                        let f = match from.concrete() {
                            Sk::Int => x as u32 as i32 as f32,
                            Sk::Int64 => x as i64 as f32,
                            Sk::Uint => x as u32 as f32,
                            _ => x as f32,
                        };
                        e.c_f32(f)
                    }
                    None => e.convert_scalar(c, from, Sk::Float),
                };
            }
            let folded = Val::new(v.ty.with_sk(Sk::Float), &comps[..v.len().min(16)]);
            return e.convert(&folded, to);
        }
    }
    e.convert(v, to)
}

/// One argument as a scalar of kind `k`.
fn scalar(e: &mut Emit, v: &Val, k: Sk) -> Result<Value> {
    Ok(conv(e, v, HTy::scalar(k))?.one())
}

/// An argument as `n` components of kind `k`.
fn vector(e: &mut Emit, v: &Val, k: Sk, n: usize) -> Result<Val> {
    conv(e, v, HTy::vec(k, n as u8))
}

fn undef(e: &mut Emit, ty: Ty) -> Value {
    e.undef(ty)
}

/// The value of an `out uint status` argument from a `ResRet`: DXC's UpdateStatus,
/// `zext(checkAccessFullyMapped(status field))`.
fn status(e: &mut Emit, res_ret: Value) -> Val {
    let raw = e.extract(Ty::I32, res_ret, 4);
    let mapped = e.dx(OpCode::CheckAccessFullyMapped, Ty::I32, &[raw]);
    Val::scalar(Sk::Uint, e.cast(CastOp::ZExt, Ty::I32, mapped))
}

/// The first `n` fields of a `ResRet` as a value of `n` components of kind `k`.
fn components(e: &mut Emit, res_ret: Value, k: Sk, n: usize) -> Val {
    let mut comps = [Value::NONE; 4];
    for (i, slot) in comps.iter_mut().enumerate().take(n) {
        *slot = e.extract(k.ir(), res_ret, i as u32);
    }
    Val::new(HTy::vec(k, n as u8), &comps[..n])
}

fn called(ret: Val) -> Called {
    Called { ret, outs: Vec::new() }
}

/// A call no overload of `name` takes: DXC rejects it too.
fn no_overload<T>(e: &Emit, name: &str, args: &[Val]) -> Result<T> {
    e.err(format!("no overload of {name} takes {} arguments", args.len()))
}
