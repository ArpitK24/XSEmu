//! Textures: `dx.op.sample*`, `textureGather*`, `textureLoad`, `textureStore`,
//! `getDimensions`, `calculateLOD`. DXC: SampleHelper / TranslateSample, GatherHelper /
//! TranslateGather, ResLoadHelper / TranslateBufLoad, TranslateStore, TranslateGetDimensions,
//! TranslateCalculateLOD.
//!
//! Every sample-like call has four coordinate operands and three offset operands whatever the
//! texture: the ones the texture kind has no use for are `undef`, an offset that is not
//! written is 0 in the dimensions the kind has ([`Shape`]). Optional trailing arguments
//! (offset, clamp, `out status`) are told apart by the argument count, as DXC's HL operand
//! indices do; cube textures take no offset, so their later arguments come one earlier.

use super::atomic::Target;
use super::{called, components, no_overload, overload, scalar, status, undef, vector};
use crate::consts::{OpCode, ResourceKind};
use crate::hlsl::emit::Emit;
use crate::hlsl::intrinsics::Called;
use crate::hlsl::types::{HTy, ResTy, Sk, Val};
use crate::ir::{CastOp, Ty, Value};
use crate::Result;

/// What a texture kind takes: DXC's DxilResource::GetNumCoords / GetNumOffsets /
/// GetNumDimensionsForCalcLOD tables.
#[derive(Copy, Clone)]
struct Shape {
    /// coordinates of a sample or load, with the array slice
    coords: usize,
    /// texel offset components (0 for cubes)
    offsets: usize,
    /// components of a gradient and of the coordinate of CalculateLevelOfDetail (no slice)
    lod: usize,
    /// values GetDimensions returns before the level or sample count
    sizes: usize,
    cube: bool,
    ms: bool,
}

fn shape(kind: ResourceKind) -> Shape {
    use ResourceKind::*;
    let (coords, offsets, lod, sizes) = match kind {
        Texture1D => (1, 1, 1, 1),
        Texture1DArray => (2, 1, 1, 2),
        Texture2D => (2, 2, 2, 2),
        Texture2DMS => (2, 2, 2, 2),
        Texture2DArray => (3, 2, 2, 3),
        Texture2DMSArray => (3, 2, 2, 3),
        Texture3D => (3, 3, 3, 3),
        TextureCube => (3, 0, 3, 2),
        _ => (4, 0, 3, 3),
    };
    Shape { coords, offsets, lod, sizes, cube: matches!(kind, TextureCube | TextureCubeArray), ms: matches!(kind, Texture2DMS | Texture2DMSArray) }
}

/// `n` operands from the components of `v`, the rest `fill`.
fn spread<const N: usize>(v: &Val, fill: Value) -> [Value; N] {
    let mut out = [fill; N];
    for (slot, &c) in out.iter_mut().zip(v.comps()) {
        *slot = c;
    }
    out
}

/// The three offset operands: the written offset (which must be made of constants: SM 6.6
/// takes immediate texel offsets only, DXC rejects anything else), else zeros, in the
/// dimensions the kind has; `undef` in the others.
fn offsets(e: &mut Emit, sh: Shape, written: Option<&Val>) -> Result<[Value; 3]> {
    let u = undef(e, Ty::I32);
    let mut out = [u; 3];
    match written {
        Some(o) => {
            let o = vector(e, o, Sk::Int, sh.offsets)?;
            if o.comps().iter().any(|&c| e.m.as_int(c).is_none()) {
                return e.unsupported("texture offset that is not a constant");
            }
            out[..sh.offsets].copy_from_slice(o.comps());
        }
        None => {
            let zero = e.c_u32(0);
            out[..sh.offsets].fill(zero);
        }
    }
    Ok(out)
}

/// The optional arguments after the fixed ones of a sample-like method, by position: an
/// offset unless the texture is a cube, then (where the method has one) a clamp, then the
/// `out status`.
struct Tail<'a> {
    offset: Option<&'a Val>,
    clamp: Option<&'a Val>,
    /// index in the method's argument list
    status: Option<usize>,
}

fn tail<'a>(e: &Emit, sh: Shape, name: &str, args: &'a [Val], fixed: usize, has_clamp: bool) -> Result<Tail<'a>> {
    let mut slots = Vec::with_capacity(3);
    if !sh.cube {
        slots.push(0);
    }
    if has_clamp {
        slots.push(1);
    }
    slots.push(2);
    let rest = &args[fixed.min(args.len())..];
    if args.len() < fixed || rest.len() > slots.len() {
        return no_overload(e, name, args);
    }
    let mut t = Tail { offset: None, clamp: None, status: None };
    for (i, a) in rest.iter().enumerate() {
        match slots[i] {
            0 => t.offset = Some(a),
            1 => t.clamp = Some(a),
            _ => t.status = Some(fixed + i),
        }
    }
    Ok(t)
}

fn sampler(e: &Emit, name: &str, args: &[Val], comparison: Option<bool>) -> Result<Value> {
    match args.first().map(|a| a.ty) {
        Some(HTy::Sampler { comparison: c }) if comparison.is_none_or(|want| want == c) => Ok(args[0].one()),
        Some(HTy::Sampler { .. }) => e.err(format!("{name} takes a {}", if comparison == Some(true) { "SamplerComparisonState" } else { "SamplerState" })),
        _ => e.err(format!("the first argument of {name} is a sampler")),
    }
}

/// The clamp operand: the written value (its first component), else undef.
fn clamp(e: &mut Emit, written: Option<&Val>) -> Result<Value> {
    match written {
        Some(c) => scalar(e, c, Sk::Float),
        None => Ok(undef(e, Ty::F32)),
    }
}

pub(super) fn method(e: &mut Emit, ty: &ResTy, h: Value, name: &str, args: &[Val]) -> Result<Called> {
    let sh = shape(ty.kind);
    match name {
        "Sample" | "SampleBias" | "SampleLevel" | "SampleGrad" | "SampleCmp" | "SampleCmpLevelZero" => sample(e, ty, sh, h, name, args),
        "CalculateLevelOfDetail" | "CalculateLevelOfDetailUnclamped" => {
            if ty.uav || sh.ms {
                return e.err(format!("{name} is not a member of this texture type"));
            }
            if args.len() != 2 {
                return no_overload(e, name, args);
            }
            // with a comparison sampler it is an SM 6.8 overload: DXC rejects it for 6.6
            let s = sampler(e, name, args, Some(false))?;
            let coord = vector(e, &args[1], Sk::Float, sh.lod)?;
            let u = undef(e, Ty::F32);
            let [c0, c1, c2] = spread(&coord, u);
            let clamped = e.c_bool(name == "CalculateLevelOfDetail");
            Ok(called(Val::scalar(Sk::Float, e.dx(OpCode::CalculateLOD, Ty::F32, &[h, s, c0, c1, c2, clamped]))))
        }
        "Load" => load(e, ty, sh, h, args),
        "GetDimensions" => dimensions(e, ty, sh, h, args),
        _ if name.starts_with("Gather") => gather(e, ty, sh, h, name, args),
        // SampleCmpLevel is SM 6.7, SampleCmpBias / SampleCmpGrad SM 6.8, GatherRaw SM 6.7;
        // GetSamplePosition is not in the dialect
        _ => e.unsupported(format!("texture method {name}")),
    }
}

/// The `Sample*` family. Operand order after the two handles, four coordinates and three
/// offsets: Sample (clamp), SampleBias (bias, clamp), SampleLevel (lod), SampleGrad (ddx ×3,
/// ddy ×3, clamp), SampleCmp (compare value, clamp), SampleCmpLevelZero (compare value).
fn sample(e: &mut Emit, ty: &ResTy, sh: Shape, h: Value, name: &str, args: &[Val]) -> Result<Called> {
    if ty.uav || sh.ms {
        return e.err(format!("{name} is not a member of this texture type"));
    }
    // DXC (IsValidObjectElement): only a float element is sampled; integers from SM 6.7 on,
    // and never with a comparison. The validator says the same of the instruction.
    if ty.elem != Sk::Float {
        let elem = match ty.elem {
            Sk::Bool => "bool",
            Sk::Int | Sk::LitInt => "int",
            Sk::Uint => "uint",
            Sk::Int64 => "int64_t",
            Sk::Uint64 => "uint64_t",
            _ => "double",
        };
        let n = if ty.elem_comps > 1 { ty.elem_comps.to_string() } else { String::new() };
        return e.err(format!("cannot {name} from resource containing {elem}{n}"));
    }
    let cmp = name.starts_with("SampleCmp");
    if cmp && ty.kind == ResourceKind::Texture3D {
        return e.err(format!("{name} is not a member of Texture3D"));
    }
    // arguments before the optional tail: sampler, coordinate, then per method
    let (fixed, has_clamp) = match name {
        "Sample" => (2, true),
        "SampleBias" | "SampleCmp" => (3, true),
        "SampleLevel" | "SampleCmpLevelZero" => (3, false),
        _ => (4, true),
    };
    let t = tail(e, sh, name, args, fixed, has_clamp)?;
    let s = sampler(e, name, args, Some(cmp))?;
    let uf = undef(e, Ty::F32);
    let coord = vector(e, &args[1], Sk::Float, sh.coords)?;
    let c: [Value; 4] = spread(&coord, uf);
    let o = offsets(e, sh, t.offset)?;
    let mut ops = vec![h, s, c[0], c[1], c[2], c[3], o[0], o[1], o[2]];
    let op = match name {
        "Sample" => OpCode::Sample,
        "SampleBias" => {
            let mut bias = scalar(e, &args[2], Sk::Float)?;
            // an immediate bias is clamped to the range the format allows (SampleHelper::SetBias)
            if let Some(bits) = e.m.as_float_bits(bias) {
                let v = f32::from_bits(bits as u32);
                if v > 15.99 {
                    bias = e.c_f32(15.99);
                } else if v < -16.0 {
                    bias = e.c_f32(-16.0);
                }
            }
            ops.push(bias);
            OpCode::SampleBias
        }
        "SampleLevel" => {
            ops.push(scalar(e, &args[2], Sk::Float)?);
            OpCode::SampleLevel
        }
        "SampleGrad" => {
            for a in &args[2..4] {
                let g = vector(e, a, Sk::Float, sh.lod)?;
                ops.extend_from_slice(&spread::<3>(&g, uf));
            }
            OpCode::SampleGrad
        }
        "SampleCmp" => {
            ops.push(scalar(e, &args[2], Sk::Float)?);
            OpCode::SampleCmp
        }
        _ => {
            ops.push(scalar(e, &args[2], Sk::Float)?);
            OpCode::SampleCmpLevelZero
        }
    };
    if has_clamp {
        ops.push(clamp(e, t.clamp)?);
    }
    // a comparison returns one float; the others the element
    let (k, n) = if cmp { (Sk::Float, 1) } else { (ty.elem, ty.elem_comps as usize) };
    let ovl = overload(e, k)?;
    let r = e.dx(op, ovl, &ops);
    let mut c = called(components(e, r, k, n));
    if let Some(i) = t.status {
        c.outs.push((i, status(e, r)));
    }
    Ok(c)
}

/// `Gather`, `GatherRed` .. `GatherAlpha` and the `GatherCmp` forms:
/// `textureGather(tex, sampler, c0..c3, o0, o1, channel)` (`textureGatherCmp` adds the compare
/// value), four values of the element's scalar type. With four offsets there are four calls,
/// one per offset, and the result takes component i of call i.
fn gather(e: &mut Emit, ty: &ResTy, sh: Shape, h: Value, name: &str, args: &[Val]) -> Result<Called> {
    let (cmp, channel) = match name.strip_prefix("GatherCmp").map(|c| (true, c)).or(name.strip_prefix("Gather").map(|c| (false, c))) {
        Some((cmp, "" | "Red")) => (cmp, 0),
        Some((cmp, "Green")) => (cmp, 1),
        Some((cmp, "Blue")) => (cmp, 2),
        Some((cmp, "Alpha")) => (cmp, 3),
        _ => return e.unsupported(format!("texture method {name}")),
    };
    let gatherable = matches!(ty.kind, ResourceKind::Texture2D | ResourceKind::Texture2DArray | ResourceKind::TextureCube | ResourceKind::TextureCubeArray);
    if ty.uav || !gatherable {
        return e.err(format!("{name} is not a member of this texture type"));
    }
    let fixed = 2 + cmp as usize;
    if args.len() < fixed {
        return no_overload(e, name, args);
    }
    let rest = &args[fixed..];
    // (), (offset), (offset, status), (o1, o2, o3, o4), (o1, o2, o3, o4, status); a cube: (), (status)
    let (first, more, status_at): (Option<&Val>, &[Val], Option<usize>) = match (sh.cube, rest.len()) {
        (_, 0) => (None, &[], None),
        (true, 1) => (None, &[], Some(fixed)),
        (false, 1) => (Some(&rest[0]), &[], None),
        (false, 2) => (Some(&rest[0]), &[], Some(fixed + 1)),
        (false, 4 | 5) if !name.ends_with("Gather") && !name.ends_with("GatherCmp") => (Some(&rest[0]), &rest[1..4], (rest.len() == 5).then_some(fixed + 4)),
        _ => return no_overload(e, name, args),
    };
    let s = sampler(e, name, args, Some(cmp))?;
    let uf = undef(e, Ty::F32);
    let coord = vector(e, &args[1], Sk::Float, sh.coords)?;
    let c: [Value; 4] = spread(&coord, uf);
    let o = offsets(e, sh, first)?;
    let ch = e.c_u32(channel);
    let mut ops = vec![h, s, c[0], c[1], c[2], c[3], o[0], o[1], ch];
    if cmp {
        ops.push(scalar(e, &args[2], Sk::Float)?);
    }
    let op = if cmp { OpCode::TextureGatherCmp } else { OpCode::TextureGather };
    let k = ty.elem;
    let ovl = overload(e, k)?;
    let r = e.dx(op, ovl, &ops);
    let v = if more.is_empty() {
        components(e, r, k, 4)
    } else {
        let mut comps = [e.extract(ovl, r, 0); 4];
        for (i, extra) in more.iter().enumerate() {
            let o = offsets(e, sh, Some(extra))?;
            ops[6] = o[0];
            ops[7] = o[1];
            let ri = e.dx(op, ovl, &ops);
            comps[i + 1] = e.extract(ovl, ri, i as u32 + 1);
        }
        Val::new(HTy::vec(k, 4), &comps)
    };
    let mut c = called(v);
    if let Some(i) = status_at {
        c.outs.push((i, status(e, r)));
    }
    Ok(c)
}

/// `textureLoad(tex, mip or sample, c0, c1, c2, o0, o1, o2)`: coordinates the kind lacks and
/// offsets that are not written are undef.
fn texture_load(e: &mut Emit, ty: &ResTy, sh: Shape, h: Value, mip: Value, coord: &Val, offset: Option<&Val>) -> Result<(Val, Value)> {
    let u = undef(e, Ty::I32);
    let c: [Value; 3] = spread(coord, u);
    let o = match offset {
        Some(_) => offsets(e, sh, offset)?,
        None => [u; 3],
    };
    let ovl = overload(e, ty.elem)?;
    let r = e.dx(OpCode::TextureLoad, ovl, &[h, mip, c[0], c[1], c[2], o[0], o[1], o[2]]);
    Ok((components(e, r, ty.elem, ty.elem_comps as usize), r))
}

/// `Load`: a read-only texture takes (location[, offset[, status]]) where the location's
/// last component is the mip level; a multisampled one (coordinate, sample[, offset[,
/// status]]); a read-write one (coordinate[, status]), and its mip operand is undef.
fn load(e: &mut Emit, ty: &ResTy, sh: Shape, h: Value, args: &[Val]) -> Result<Called> {
    if sh.cube {
        return e.err("Load is not a member of a cube texture");
    }
    let (fixed, max) = match (sh.ms, ty.uav) {
        (true, _) => (2, 4),
        (false, true) => (1, 2),
        (false, false) => (1, 3),
    };
    if args.len() < fixed || args.len() > max {
        return no_overload(e, "Load", args);
    }
    let (mip, coord) = if sh.ms {
        (scalar(e, &args[1], Sk::Int)?, vector(e, &args[0], Sk::Int, sh.coords)?)
    } else if ty.uav {
        (undef(e, Ty::I32), vector(e, &args[0], Sk::Int, sh.coords)?)
    } else {
        let loc = vector(e, &args[0], Sk::Int, sh.coords + 1)?;
        (loc.comps()[sh.coords], Val::new(HTy::vec(Sk::Int, sh.coords as u8), &loc.comps()[..sh.coords]))
    };
    let (offset, status_at) = if ty.uav && !sh.ms { (None, (args.len() > fixed).then_some(fixed)) } else { (args.get(fixed), (args.len() > fixed + 1).then_some(fixed + 1)) };
    let (v, r) = texture_load(e, ty, sh, h, mip, &coord, offset)?;
    let mut c = called(v);
    if let Some(i) = status_at {
        c.outs.push((i, status(e, r)));
    }
    Ok(c)
}

/// `tex[coord]`: mip 0 of a read-only texture, sample 0 of a multisampled one; a read-write
/// texture has no mip operand (undef).
pub(super) fn index_load(e: &mut Emit, ty: &ResTy, h: Value, coord: &Val) -> Result<Val> {
    let sh = shape(ty.kind);
    if sh.cube {
        return e.err("a cube texture has no subscript operator");
    }
    let coord = vector(e, coord, Sk::Int, sh.coords)?;
    let mip = if ty.uav && !sh.ms { undef(e, Ty::I32) } else { e.c_u32(0) };
    Ok(texture_load(e, ty, sh, h, mip, &coord, None)?.0)
}

/// `tex[coord] = value`: `textureStore(tex, c0, c1, c2, v0..v3, 15)`. All four values are
/// always written: those past the element's components repeat the first, as DXC fills them.
pub(super) fn index_store(e: &mut Emit, ty: &ResTy, h: Value, coord: &Val, value: &Val) -> Result<()> {
    let sh = shape(ty.kind);
    if sh.cube || sh.ms {
        return e.unsupported(format!("store to {:?}", ty.kind));
    }
    let coord = vector(e, coord, Sk::Int, sh.coords)?;
    let n = ty.elem_comps as usize;
    let value = vector(e, value, ty.elem, n)?;
    let u = undef(e, Ty::I32);
    let c: [Value; 3] = spread(&coord, u);
    let v: [Value; 4] = spread(&value, value.comps()[0]);
    let mask = e.m.const_i8(15);
    let ovl = overload(e, ty.elem)?;
    e.dx(OpCode::TextureStore, ovl, &[h, c[0], c[1], c[2], v[0], v[1], v[2], v[3], mask]);
    Ok(())
}

/// The atomic target of `tex[coord]`: the texel coordinates, undef past the kind's.
pub(super) fn element(e: &mut Emit, ty: &ResTy, h: Value, coord: &Val) -> Result<Target> {
    let sh = shape(ty.kind);
    if sh.cube || sh.ms {
        return e.unsupported(format!("atomic on {:?}", ty.kind));
    }
    let coord = vector(e, coord, Sk::Int, sh.coords)?;
    let u = undef(e, Ty::I32);
    Ok(Target { h, coords: spread(&coord, u) })
}

/// `GetDimensions`: `getDimensions(tex, mip)` returns { width, height, depth or elements,
/// levels or samples }. The overloads are (mip, out sizes.., out levels) and (out sizes..)
/// with mip 0, told apart by the argument count; a multisampled texture has only
/// (out sizes.., out samples) and an undef mip operand. An `out` of float type gets the value
/// converted as a signed integer.
fn dimensions(e: &mut Emit, ty: &ResTy, sh: Shape, h: Value, args: &[Val]) -> Result<Called> {
    // (first out argument, values in order, whether the last one is the count in field 3)
    let (mip, first, count) = if sh.ms {
        if args.len() != sh.sizes + 1 {
            return no_overload(e, "GetDimensions", args);
        }
        (undef(e, Ty::I32), 0, true)
    } else if args.len() == sh.sizes + 2 && !ty.uav {
        (scalar(e, &args[0], Sk::Uint)?, 1, true)
    } else if args.len() == sh.sizes {
        (e.c_u32(0), 0, false)
    } else {
        return no_overload(e, "GetDimensions", args);
    };
    let dims = e.dx(OpCode::GetDimensions, Ty::VOID, &[h, mip]);
    let mut c = called(Val::void());
    let outs = &args[first..];
    for (i, a) in outs.iter().enumerate() {
        let field = if count && i + 1 == outs.len() { 3 } else { i as u32 };
        let v = e.extract(Ty::I32, dims, field);
        let v = match a.ty {
            HTy::Num(k, 1, 1) if k.is_float() => Val::scalar(Sk::Float, e.cast(CastOp::SIToFP, Ty::F32, v)),
            HTy::Num(_, 1, 1) => Val::scalar(Sk::Uint, v),
            other => return e.err(format!("GetDimensions out argument of {other:?}")),
        };
        c.outs.push((first + i, v));
    }
    Ok(c)
}
