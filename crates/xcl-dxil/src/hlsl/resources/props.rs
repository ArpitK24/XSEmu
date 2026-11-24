//! What a resource type is to DXIL: the `%dx.types.ResourceProperties` words of
//! `annotateHandle`, the [`ResTy`] of a declared HLSL type and the [`Resource`] record of a
//! register-bound declaration. All pure: no IR is touched here.

use crate::consts::{ComponentType, ResourceClass, ResourceKind, SamplerKind};
use crate::hlsl::ast::{Register, ResKind};
use crate::hlsl::types::{HTy, ResTy, Sk, TypeTable};
use crate::shader::Resource;
use crate::{Error, Result};

/// Bits of the first properties word above the resource kind (DXC's
/// `DxilResourceProperties::BasicProps`; bits 8..11 are the base alignment of a structured
/// buffer, which DXC leaves 0 for everything HLSL declares).
const IS_UAV: u32 = 1 << 12;
const IS_ROV: u32 = 1 << 13;
const GLOBALLY_COHERENT: u32 = 1 << 14;
/// a comparison sampler, or a structured buffer with a counter
const CMP_OR_COUNTER: u32 = 1 << 15;

/// The second word of a typed resource: component type, component count and sample count
/// (`TypedProps`).
fn typed_word(comp: ComponentType, count: u32, samples: u32) -> u32 {
    comp as u32 | (count & 0xFF) << 8 | (samples & 0xFF) << 16
}

fn is_typed(kind: ResourceKind) -> bool {
    is_texture(kind) || kind == ResourceKind::TypedBuffer
}

pub(super) fn is_texture(kind: ResourceKind) -> bool {
    use ResourceKind::*;
    matches!(kind, Texture1D | Texture2D | Texture2DMS | Texture3D | TextureCube | Texture1DArray | Texture2DArray | Texture2DMSArray | TextureCubeArray)
}

/// The two `ResourceProperties` words of a resource type, as DXC packs them for
/// `annotateHandle` (DxilResourceProperties.cpp, `loadPropsFromResourceBase`): kind and
/// UAV / coherence flags in the first; component type, count and sample count of a typed
/// resource or the stride of a structured buffer in the second (0 for a raw buffer).
pub fn properties(ty: &ResTy) -> [u32; 2] {
    let mut w0 = ty.kind as u32;
    if ty.uav {
        w0 |= IS_UAV;
        if ty.coherent {
            w0 |= GLOBALLY_COHERENT;
        }
    }
    let w1 = if ty.kind == ResourceKind::StructuredBuffer {
        ty.stride
    } else if is_typed(ty.kind) {
        typed_word(ty.elem.component_type(), ty.elem_comps as u32, ty.samples as u32)
    } else {
        0
    };
    [w0, w1]
}

/// The properties of a sampler: the kind, and the comparison flag.
pub fn sampler_properties(comparison: bool) -> [u32; 2] {
    [ResourceKind::Sampler as u32 | if comparison { CMP_OR_COUNTER } else { 0 }, 0]
}

/// The properties of a register-bound resource from its record: the same words
/// [`properties`] gives for its type, plus what only the record knows (a counter, ROV).
pub fn record_properties(res: &Resource) -> [u32; 2] {
    match res.class {
        ResourceClass::Sampler => sampler_properties(res.sampler_kind == SamplerKind::Comparison),
        ResourceClass::CBuffer => [res.kind as u32, res.cbuffer_bytes],
        _ => {
            let mut w0 = res.kind as u32;
            if res.class == ResourceClass::UAV {
                w0 |= IS_UAV;
                if res.globally_coherent {
                    w0 |= GLOBALLY_COHERENT;
                }
                if res.has_counter {
                    w0 |= CMP_OR_COUNTER;
                }
                if res.rov {
                    w0 |= IS_ROV;
                }
            }
            let w1 = if res.kind == ResourceKind::StructuredBuffer {
                res.struct_stride
            } else if is_typed(res.kind) {
                typed_word(res.comp_type, res.comp_count, res.sample_count)
            } else {
                0
            };
            [w0, w1]
        }
    }
}

/// Bytes of one scalar in a structured buffer element (bool is stored as a 32-bit integer).
fn scalar_bytes(k: Sk) -> u32 {
    if k.is_64() {
        8
    } else {
        4
    }
}

/// The resource type of a declaration: `kind` is the template name, `elem` its element type
/// argument (None: not written, `float4` for textures and `Buffer`), `samples` the sample
/// count argument of a multisampled texture, `coherent` the `globallycoherent` qualifier.
///
/// Typed resources take 32-bit float / int / uint elements of 1 to 4 components (what DXIL's
/// `ResRet` carries without the 64-bit pair encoding); structured buffers take numeric
/// elements of up to 4 components. Everything else (struct elements, append / consume
/// buffers, constant buffers, acceleration structures) is [`Error::Unsupported`].
pub fn res_ty(kind: ResKind, elem: Option<HTy>, samples: Option<u32>, coherent: bool, _types: &TypeTable) -> Result<ResTy> {
    use ResKind as K;
    use ResourceKind as R;
    let (rk, uav) = match kind {
        K::Texture1D => (R::Texture1D, false),
        K::Texture1DArray => (R::Texture1DArray, false),
        K::Texture2D => (R::Texture2D, false),
        K::Texture2DArray => (R::Texture2DArray, false),
        K::Texture2DMS => (R::Texture2DMS, false),
        K::Texture2DMSArray => (R::Texture2DMSArray, false),
        K::Texture3D => (R::Texture3D, false),
        K::TextureCube => (R::TextureCube, false),
        K::TextureCubeArray => (R::TextureCubeArray, false),
        K::RWTexture1D => (R::Texture1D, true),
        K::RWTexture1DArray => (R::Texture1DArray, true),
        K::RWTexture2D => (R::Texture2D, true),
        K::RWTexture2DArray => (R::Texture2DArray, true),
        K::RWTexture3D => (R::Texture3D, true),
        K::Buffer => (R::TypedBuffer, false),
        K::RWBuffer => (R::TypedBuffer, true),
        K::ByteAddressBuffer => (R::RawBuffer, false),
        K::RWByteAddressBuffer => (R::RawBuffer, true),
        K::StructuredBuffer => (R::StructuredBuffer, false),
        K::RWStructuredBuffer => (R::StructuredBuffer, true),
        K::AppendStructuredBuffer | K::ConsumeStructuredBuffer | K::ConstantBuffer | K::RaytracingAccelerationStructure => {
            return crate::unsupported(format!("resource type {kind:?}"));
        }
    };
    if coherent && !uav {
        return Err(Error::Source { line: 0, msg: "globallycoherent can only be used with unordered access view buffers".into() });
    }
    let ms = matches!(rk, R::Texture2DMS | R::Texture2DMSArray);
    if samples.is_some() && !ms {
        return Err(Error::Source { line: 0, msg: format!("{kind:?} takes no sample count") });
    }
    let samples = samples.unwrap_or(0);
    if samples > 255 {
        return Err(Error::Source { line: 0, msg: "sample count out of range".into() });
    }
    let mut ty = ResTy { kind: rk, uav, elem: Sk::Uint, elem_comps: 1, samples: samples as u8, coherent, stride: 0 };
    match rk {
        R::RawBuffer => {
            if elem.is_some() {
                return Err(Error::Source { line: 0, msg: format!("{kind:?} takes no template arguments") });
            }
        }
        R::StructuredBuffer => {
            let Some(HTy::Num(k, rows, cols)) = elem else {
                return crate::unsupported("structured buffer of a struct type");
            };
            let k = k.concrete();
            if rows != 1 || k == Sk::Bool {
                return crate::unsupported(format!("structured buffer of {k:?}{rows}x{cols}"));
            }
            ty.elem = k;
            ty.elem_comps = cols;
            ty.stride = cols as u32 * scalar_bytes(k);
        }
        _ => {
            let (k, n) = match elem {
                None => (Sk::Float, 4),
                Some(HTy::Num(k, 1, n)) => (k.concrete(), n),
                Some(other) => return crate::unsupported(format!("{kind:?} of {other:?}")),
            };
            if !matches!(k, Sk::Float | Sk::Int | Sk::Uint) {
                return crate::unsupported(format!("{kind:?} of {k:?}"));
            }
            ty.elem = k;
            ty.elem_comps = n;
        }
    }
    Ok(ty)
}

/// Register a declared resource or sampler (`ty` is [`HTy::Res`] or [`HTy::Sampler`]) bound
/// with `: register(...)`: appends its record to `resources` (the ShaderInfo's list) and returns
/// its index there. The record follows DXC's DxilResource for the declaration: the class
/// from the type, the id counting within the class in declaration order, one register, the
/// component type and count of a typed resource (`Invalid` / 0 for raw and structured buffers
/// and samplers), the sample count, the stride of a structured buffer, coherence, and the
/// sampler kind.
///
/// A declaration without a register is [`Error::Unsupported`] (DXC would allocate one; the
/// dialect always writes it), a register of the wrong class is a source error, as in DXC.
pub fn declare(resources: &mut Vec<Resource>, name: &str, ty: HTy, reg: Option<&Register>) -> Result<usize> {
    let (class, letter) = match ty {
        HTy::Res(t) if t.uav => (ResourceClass::UAV, b'u'),
        HTy::Res(_) => (ResourceClass::SRV, b't'),
        HTy::Sampler { .. } => (ResourceClass::Sampler, b's'),
        other => return Err(Error::Internal(format!("resources::declare of {other:?}"))),
    };
    let Some(reg) = reg else {
        return crate::unsupported("resource declared without register()");
    };
    if reg.class != letter {
        return Err(Error::Source { line: 0, msg: format!("invalid register specification for {name}: expected '{}' binding", letter as char) });
    }
    let id = resources.iter().filter(|r| r.class == class).count() as u32;
    let mut r = match ty {
        HTy::Res(t) => {
            let mut r = Resource::new(class, id, name, t.kind, reg.index, reg.space);
            if t.kind == ResourceKind::StructuredBuffer {
                r.struct_stride = t.stride;
            } else if is_typed(t.kind) {
                r.comp_type = t.elem.component_type();
                r.comp_count = t.elem_comps as u32;
                r.sample_count = t.samples as u32;
            }
            r.globally_coherent = t.uav && t.coherent;
            r
        }
        _ => Resource::new(class, id, name, ResourceKind::Sampler, reg.index, reg.space),
    };
    if let HTy::Sampler { comparison: true } = ty {
        r.sampler_kind = SamplerKind::Comparison;
    }
    resources.push(r);
    Ok(resources.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ty(kind: ResKind, elem: Option<HTy>, samples: Option<u32>, coherent: bool) -> ResTy {
        res_ty(kind, elem, samples, coherent, &TypeTable::default()).unwrap()
    }

    /// The words DXC 1.9 prints in `annotateHandle` for these types (cs_6_6 disassembly).
    #[test]
    fn properties_match_dxc() {
        let f4 = Some(HTy::vec(Sk::Float, 4));
        assert_eq!(properties(&ty(ResKind::RWByteAddressBuffer, None, None, false)), [4107, 0]);
        assert_eq!(properties(&ty(ResKind::RWByteAddressBuffer, None, None, true)), [20491, 0]);
        assert_eq!(properties(&ty(ResKind::ByteAddressBuffer, None, None, false)), [11, 0]);
        assert_eq!(properties(&ty(ResKind::Texture2D, None, None, false)), [2, 1033]);
        assert_eq!(properties(&ty(ResKind::Texture2D, f4, None, false)), [2, 1033]);
        assert_eq!(properties(&ty(ResKind::Texture2D, Some(HTy::vec(Sk::Uint, 4)), None, false)), [2, 1029]);
        assert_eq!(properties(&ty(ResKind::Texture2D, Some(HTy::vec(Sk::Int, 4)), None, false)), [2, 1028]);
        assert_eq!(properties(&ty(ResKind::RWTexture2D, Some(HTy::UINT), None, false)), [4098, 261]);
        assert_eq!(properties(&ty(ResKind::Texture2DMS, f4, Some(4), false)), [3, 0x40409]);
        assert_eq!(properties(&ty(ResKind::RWStructuredBuffer, Some(HTy::UINT), None, false)), [4108, 4]);
        assert_eq!(sampler_properties(false), [14, 0]);
        assert_eq!(sampler_properties(true), [32782, 0]);
    }

    #[test]
    fn declared_records_follow_the_type() {
        let mut list = Vec::new();
        let bab = HTy::Res(ty(ResKind::RWByteAddressBuffer, None, None, false));
        let reg = |class: u8, index: u32, space: u32| Register { class, index, space };
        assert_eq!(declare(&mut list, "io", bab, Some(&reg(b'u', 0, 0))).unwrap(), 0);
        assert_eq!(declare(&mut list, "res", bab, Some(&reg(b'u', 1, 0))).unwrap(), 1);
        let tex = HTy::Res(ty(ResKind::Texture2DMS, Some(HTy::vec(Sk::Int, 3)), Some(8), false));
        declare(&mut list, "t", tex, Some(&reg(b't', 5, 2))).unwrap();
        declare(&mut list, "s", HTy::Sampler { comparison: true }, Some(&reg(b's', 3, 1))).unwrap();
        let sb = HTy::Res(ty(ResKind::RWStructuredBuffer, Some(HTy::vec(Sk::Float, 3)), None, true));
        declare(&mut list, "sb", sb, Some(&reg(b'u', 7, 0))).unwrap();

        assert_eq!((list[1].class, list[1].id, list[1].lower_bound, list[1].range_size), (ResourceClass::UAV, 1, 1, 1));
        assert_eq!((list[1].kind, list[1].comp_type, list[1].comp_count), (ResourceKind::RawBuffer, ComponentType::Invalid, 0));
        assert_eq!((list[2].class, list[2].id, list[2].space, list[2].lower_bound), (ResourceClass::SRV, 0, 2, 5));
        assert_eq!((list[2].kind, list[2].comp_type, list[2].comp_count, list[2].sample_count), (ResourceKind::Texture2DMS, ComponentType::I32, 3, 8));
        assert_eq!((list[3].class, list[3].id, list[3].kind, list[3].sampler_kind), (ResourceClass::Sampler, 0, ResourceKind::Sampler, SamplerKind::Comparison));
        assert_eq!((list[4].id, list[4].struct_stride, list[4].globally_coherent, list[4].comp_type), (2, 12, true, ComponentType::Invalid));
        // the record gives the same annotation as the type
        for (r, t) in [(&list[1], bab), (&list[2], tex), (&list[4], sb)] {
            let HTy::Res(t) = t else { unreachable!() };
            assert_eq!(record_properties(r), properties(&t));
        }
        assert_eq!(record_properties(&list[3]), sampler_properties(true));

        assert!(matches!(declare(&mut list, "x", bab, None), Err(Error::Unsupported(_))));
        assert!(matches!(declare(&mut list, "x", bab, Some(&reg(b't', 0, 0))), Err(Error::Source { .. })));
    }

    #[test]
    fn unsupported_types_say_so() {
        let t = TypeTable::default();
        assert!(matches!(res_ty(ResKind::AppendStructuredBuffer, Some(HTy::UINT), None, false, &t), Err(Error::Unsupported(_))));
        assert!(matches!(res_ty(ResKind::Texture2D, Some(HTy::UINT64), None, false, &t), Err(Error::Unsupported(_))));
        assert!(matches!(res_ty(ResKind::StructuredBuffer, Some(HTy::Struct(0)), None, false, &t), Err(Error::Unsupported(_))));
        assert!(matches!(res_ty(ResKind::Texture2D, None, None, true, &t), Err(Error::Source { .. })));
        assert!(matches!(res_ty(ResKind::Texture2D, None, Some(4), false, &t), Err(Error::Source { .. })));
    }
}
