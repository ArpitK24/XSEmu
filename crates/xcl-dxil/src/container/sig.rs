//! ISG1 / OSG1 / PSG1: the D3D-style signature parts, as DXC's `DxilProgramSignatureWriter`
//! (lib/DxilContainer/DxilContainerAssembler.cpp) serialises a `DxilSignature` for validator
//! 1.7 and later (names shared between elements, part padded to 4 bytes). The validator
//! rebuilds each part from the module's metadata and compares the bytes.

use super::stdsort::{self, Flavor};
use crate::consts::*;
use crate::shader::{ShaderInfo, SigElement, StageProps};
use crate::{Error, Result};

/// Which of a shader's three signatures (DXC's `DXIL::SignatureKind`).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SigKind {
    Input,
    Output,
    /// HS patch constant outputs, DS patch constant inputs, MS per-primitive outputs
    PatchConstOrPrim,
}

/// What a semantic means at a signature point (DXC's `SemanticInterpretationKind`, reduced to
/// what the container needs).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Interp {
    /// not valid there: DXC's element has the Invalid semantic
    NotAvailable,
    /// valid, read through a dx.op of its own; in the signature metadata and PSV0 if the
    /// front end lists it, never in ISG1 / OSG1 / PSG1
    NotInSig,
    /// an ordinary element there, whatever its name: DXC's element has the Arbitrary semantic
    Arbitrary,
    /// a system value (packed or not) with a row in the signature part
    System,
}

/// DXC's semantic interpretation table (DxilSigPoint.inl, INTERPRETATION-TABLE) for the newest
/// shader model, which is what `DxilSignatureElement::GetInterpretation` asks for: one row per
/// `SemanticKind`, one column per `SigPointKind` (VSIn, VSOut, PCIn, HSIn, HSCPIn, HSCPOut,
/// PCOut, DSIn, DSCPIn, DSOut, GSVIn, GSIn, GSOut, PSIn, PSOut, CSIn, MSIn, MSOut, MSPOut,
/// ASIn). `-`: not available, `n`: not in the signature, `a`: arbitrary, `s`: system value.
const INTERPRETATION: [&[u8; 20]; 33] = [
    b"aa--aaaaaaa-aa---aa-", // Arbitrary
    b"s-------------------", // VertexID
    b"sa--aa--aaa-aa------", // InstanceID
    b"as--ssaasss-ss---s--", // Position
    b"as--ssaasss-ss----s-", // RenderTargetArrayIndex
    b"as--ssaasss-ss----s-", // ViewPortArrayIndex
    b"as--ssaasss-ss---s--", // ClipDistance
    b"as--ssaasss-ss---s--", // CullDistance
    b"---n----------------", // OutputControlPointID
    b"-------n------------", // DomainLocation
    b"--nn---n---sss----s-", // PrimitiveID
    b"-----------n--------", // GSInstanceID
    b"-------------s------", // SampleIndex
    b"------------ss------", // IsFrontFace
    b"-------------ns-----", // Coverage
    b"-------------n------", // InnerCoverage
    b"--------------s-----", // Target
    b"--------------s-----", // Depth
    b"--------------s-----", // DepthLessEqual
    b"--------------s-----", // DepthGreaterEqual
    b"--------------s-----", // StencilRef
    b"---------------nn--n", // DispatchThreadID
    b"---------------nn--n", // GroupID
    b"---------------nn--n", // GroupIndex
    b"---------------nn--n", // GroupThreadID
    b"------ss------------", // TessFactor
    b"------ss------------", // InsideTessFactor
    b"n-nn---n---n-n--n---", // ViewID
    b"-------------s------", // Barycentrics
    b"-s--ss--sss-ss----s-", // ShadingRate
    b"-------------n----s-", // CullPrimitive
    b"n-------------------", // StartVertexLocation
    b"n-------------------", // StartInstanceLocation
];

/// The signature point of a whole signature (DXC's `SigPoint::GetKind` without the special
/// inputs, which is how `DxilSignature` is constructed when a module is loaded).
pub fn sig_point(stage: ShaderKind, sig: SigKind) -> SigPointKind {
    use ShaderKind as K;
    use SigKind::*;
    use SigPointKind as P;
    match (stage, sig) {
        (K::Vertex, Input) => P::VSIn,
        (K::Vertex, Output) => P::VSOut,
        (K::Hull, Input) => P::HSCPIn,
        (K::Hull, Output) => P::HSCPOut,
        (K::Hull, PatchConstOrPrim) => P::PCOut,
        (K::Domain, Input) => P::DSCPIn,
        (K::Domain, Output) => P::DSOut,
        (K::Domain, PatchConstOrPrim) => P::DSIn,
        (K::Geometry, Input) => P::GSVIn,
        (K::Geometry, Output) => P::GSOut,
        (K::Pixel, Input) => P::PSIn,
        (K::Pixel, Output) => P::PSOut,
        (K::Compute, Input) => P::CSIn,
        (K::Mesh, Input) => P::MSIn,
        (K::Mesh, Output) => P::MSOut,
        (K::Mesh, PatchConstOrPrim) => P::MSPOut,
        (K::Amplification, Input) => P::ASIn,
        _ => P::Invalid,
    }
}

/// What `kind` means in the signature at `point`. A GS primitive id is the one element of
/// the vertex input signature that belongs to another point (DXC's `SigPoint::RecoverKind`).
pub fn interpretation(kind: SemanticKind, point: SigPointKind) -> Interp {
    let point = if kind == SemanticKind::PrimitiveID && point == SigPointKind::GSVIn { SigPointKind::GSIn } else { point };
    let Some(row) = INTERPRETATION.get(kind as usize) else { return Interp::NotAvailable };
    match row.get(point as usize) {
        Some(b'n') => Interp::NotInSig,
        Some(b'a') => Interp::Arbitrary,
        Some(b's') => Interp::System,
        _ => Interp::NotAvailable,
    }
}

/// DXC's canonical spelling of a system value semantic (`Semantic::GetName`), which replaces
/// the source's in metadata and container; `None` for Arbitrary and Invalid.
pub fn semantic_name(kind: SemanticKind) -> Option<&'static str> {
    use SemanticKind::*;
    Some(match kind {
        VertexID => "SV_VertexID",
        InstanceID => "SV_InstanceID",
        Position => "SV_Position",
        RenderTargetArrayIndex => "SV_RenderTargetArrayIndex",
        ViewPortArrayIndex => "SV_ViewportArrayIndex",
        ClipDistance => "SV_ClipDistance",
        CullDistance => "SV_CullDistance",
        OutputControlPointID => "SV_OutputControlPointID",
        DomainLocation => "SV_DomainLocation",
        PrimitiveID => "SV_PrimitiveID",
        GSInstanceID => "SV_GSInstanceID",
        SampleIndex => "SV_SampleIndex",
        IsFrontFace => "SV_IsFrontFace",
        Coverage => "SV_Coverage",
        InnerCoverage => "SV_InnerCoverage",
        Target => "SV_Target",
        Depth => "SV_Depth",
        DepthLessEqual => "SV_DepthLessEqual",
        DepthGreaterEqual => "SV_DepthGreaterEqual",
        StencilRef => "SV_StencilRef",
        DispatchThreadID => "SV_DispatchThreadID",
        GroupID => "SV_GroupID",
        GroupIndex => "SV_GroupIndex",
        GroupThreadID => "SV_GroupThreadID",
        TessFactor => "SV_TessFactor",
        InsideTessFactor => "SV_InsideTessFactor",
        ViewID => "SV_ViewID",
        Barycentrics => "SV_Barycentrics",
        ShadingRate => "SV_ShadingRate",
        CullPrimitive => "SV_CullPrimitive",
        StartVertexLocation => "SV_StartVertexLocation",
        StartInstanceLocation => "SV_StartInstanceLocation",
        Arbitrary | Invalid => return None,
    })
}

/// A signature element as DXC sees it once the module is loaded: the semantic resolved against
/// the signature point (`DxilSignatureElement::SetKind`), so that a system value name where
/// it is only an ordinary element (SV_Position into a vertex shader) is Arbitrary under its
/// own name, and the canonical name everywhere else.
pub(crate) struct Resolved<'a> {
    pub e: &'a SigElement,
    pub kind: SemanticKind,
    pub name: &'a str,
    pub interp: Interp,
}

impl Resolved<'_> {
    /// DXC's `IsAllocated`: packed into a register (not SV_Depth, SV_Coverage, ...).
    pub fn allocated(&self) -> bool {
        self.e.start_row >= 0 && self.e.start_col >= 0
    }
}

pub(crate) fn resolve<'a>(stage: ShaderKind, sig: SigKind, elements: &'a [SigElement]) -> Result<Vec<Resolved<'a>>> {
    let point = sig_point(stage, sig);
    elements
        .iter()
        .map(|e| {
            let interp = interpretation(e.kind, point);
            let (kind, name) = match (interp, semantic_name(e.kind)) {
                (Interp::NotAvailable, _) => return Err(Error::Internal(format!("signature element {} ({:?}) is not valid in the {stage:?} {sig:?} signature", e.name, e.kind))),
                (Interp::Arbitrary, _) | (_, None) => (SemanticKind::Arbitrary, e.name.as_str()),
                (_, Some(name)) => (e.kind, name),
            };
            if e.indices.is_empty() || e.indices.len() != e.rows as usize || !(1..=4).contains(&e.cols) || e.rows > 32 {
                return Err(Error::Internal(format!("signature element {name}: {} rows, {} semantic indices, {} columns", e.rows, e.indices.len(), e.cols)));
            }
            let allocated = e.start_row >= 0 && e.start_col >= 0;
            if (allocated && (e.start_col as u32 + e.cols > 4 || e.start_row as u32 + e.rows > 32)) || e.stream > 3 {
                return Err(Error::Internal(format!("signature element {name}: row {} column {} stream {} is outside the signature", e.start_row, e.start_col, e.stream)));
            }
            Ok(Resolved { e, kind, name, interp })
        })
        .collect()
}

/// DXC's `CompTypeToSigCompType` for validator 1.5 and later (bool is a uint).
pub(crate) fn sig_comp_type(t: ComponentType) -> u32 {
    use ComponentType::*;
    match t {
        I1 | U32 => 1,
        I32 => 2,
        F32 => 3,
        U16 => 4,
        I16 => 5,
        F16 => 6,
        U64 => 7,
        I64 => 8,
        F64 => 9,
        _ => 0,
    }
}

/// DXC's `SemanticKindToSystemValue`: the D3D name value (`D3D_NAME`) of a system value. The
/// tessellation factors are named after the domain, and everything DXC's switch does not list
/// falls into its default, the stencil reference.
fn system_value(kind: SemanticKind, domain: TessellatorDomain) -> u32 {
    use SemanticKind::*;
    match kind {
        Arbitrary | Invalid => 0,
        Position => 1,
        ClipDistance => 2,
        CullDistance => 3,
        RenderTargetArrayIndex => 4,
        ViewPortArrayIndex => 5,
        VertexID => 6,
        PrimitiveID => 7,
        InstanceID => 8,
        IsFrontFace => 9,
        SampleIndex => 10,
        TessFactor => match domain {
            TessellatorDomain::Quad => 11,
            TessellatorDomain::Tri => 13,
            // the density factor; the detail factor of the second row follows in `write`
            TessellatorDomain::IsoLine => 16,
            _ => 0,
        },
        InsideTessFactor => match domain {
            TessellatorDomain::Quad => 12,
            TessellatorDomain::Tri => 14,
            _ => 0,
        },
        Barycentrics => 23,
        ShadingRate => 24,
        CullPrimitive => 25,
        Target => 64,
        Depth => 65,
        Coverage => 66,
        DepthGreaterEqual => 67,
        DepthLessEqual => 68,
        InnerCoverage => 70,
        _ => 69,
    }
}

/// `DxilProgramSignatureElement`, 32 bytes in the part.
#[derive(Copy, Clone)]
struct Row {
    stream: u32,
    name: u32,
    index: u32,
    system_value: u32,
    comp_type: u32,
    register: u32,
    mask: u8,
    /// outputs: components never written; inputs: components always read
    rw_mask: u8,
    min_precision: u32,
}

/// One signature part's payload, as the DXC of this platform writes it.
pub fn write(info: &ShaderInfo, sig: SigKind) -> Result<Vec<u8>> {
    write_sorted(info, sig, Flavor::HOST)
}

/// [`write`] with the rows in the order of a given standard library's `std::sort` (they
/// differ in where they leave rows of one name packed into one register).
pub fn write_sorted(info: &ShaderInfo, sig: SigKind, flavor: Flavor) -> Result<Vec<u8>> {
    let (elements, is_input) = match sig {
        SigKind::Input => (&info.inputs, true),
        SigKind::Output => (&info.outputs, false),
        SigKind::PatchConstOrPrim => (&info.patch_or_prim, info.kind == ShaderKind::Domain),
    };
    let domain = match &info.props {
        StageProps::Hull { domain, .. } | StageProps::Domain { domain, .. } => *domain,
        _ => TessellatorDomain::Undefined,
    };
    // DXC's UseMinPrecision: min16float and friends are hints unless the module was built
    // with native 16-bit types (the UseNativeLowPrecision shader flag)
    let min_precision = info.flags & crate::flags::bit::USE_NATIVE_LOW_PRECISION == 0;
    let elements = resolve(info.kind, sig, elements)?;
    let in_part = |r: &Resolved| r.interp != Interp::NotInSig;

    let count: usize = elements.iter().filter(|r| in_part(r)).map(|r| r.e.indices.len()).sum();
    let fixed = 8 + 32 * count;
    // every element's name gets a string, in element order, also of the elements without a
    // row (DXC's calcSizes); equal names share one
    let mut names: Vec<(&str, u32)> = Vec::new();
    let mut end = fixed as u32;
    let offsets: Vec<u32> = elements
        .iter()
        .map(|r| match names.iter().find(|(n, _)| *n == r.name) {
            Some(&(_, at)) => at,
            None => {
                let at = end;
                names.push((r.name, at));
                end += r.name.len() as u32 + 1;
                at
            }
        })
        .collect();

    let mut rows = Vec::with_capacity(count);
    for (r, &name) in elements.iter().zip(&offsets) {
        if !in_part(r) {
            continue;
        }
        let e = r.e;
        let shift = if r.allocated() { e.start_col as u32 } else { 0 };
        let usage = (e.usage_mask as u32) << shift;
        let mut system_value = system_value(r.kind, domain);
        let comp = e.comp_type;
        for (i, &index) in e.indices.iter().enumerate() {
            rows.push(Row {
                stream: e.stream,
                name,
                index,
                system_value,
                comp_type: sig_comp_type(comp),
                register: if r.allocated() { e.start_row as u32 + i as u32 } else { e.start_row as u32 },
                mask: (((1u32 << e.cols) - 1) << shift) as u8,
                rw_mask: if is_input { usage as u8 } else { (usage as u8 ^ 0xF) & 0xF },
                min_precision: match comp {
                    ComponentType::F16 if min_precision => 1,
                    ComponentType::I16 if min_precision => 4,
                    ComponentType::U16 if min_precision => 5,
                    _ => 0,
                },
            });
            if system_value == 16 {
                system_value = 15;
            }
        }
    }
    // DXC's sort_sig: by stream, register, name, with std::sort. Rows that agree in all three
    // (one name packed twice into a register) end up where that algorithm leaves them, and the
    // validator, sorting the same way, expects them there.
    stdsort::sort(&mut rows, flavor, |a, b| (a.stream, a.register, a.name) < (b.stream, b.register, b.name));

    let mut out = Vec::with_capacity((end as usize + 3) & !3);
    out.extend_from_slice(&(count as u32).to_le_bytes());
    out.extend_from_slice(&8u32.to_le_bytes());
    for r in &rows {
        for w in [r.stream, r.name, r.index, r.system_value, r.comp_type, r.register] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&[r.mask, r.rw_mask, 0, 0]);
        out.extend_from_slice(&r.min_precision.to_le_bytes());
    }
    for (name, _) in &names {
        out.extend_from_slice(name.as_bytes());
        out.push(0);
    }
    out.resize((out.len() + 3) & !3, 0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpretation_matches_dxc_table() {
        use SemanticKind as S;
        use SigPointKind as P;
        assert_eq!(interpretation(S::Position, P::VSIn), Interp::Arbitrary);
        assert_eq!(interpretation(S::Position, P::VSOut), Interp::System);
        assert_eq!(interpretation(S::Position, P::PCOut), Interp::Arbitrary);
        assert_eq!(interpretation(S::PrimitiveID, P::GSVIn), Interp::System);
        assert_eq!(interpretation(S::PrimitiveID, P::HSCPIn), Interp::NotAvailable);
        assert_eq!(interpretation(S::Coverage, P::PSIn), Interp::NotInSig);
        assert_eq!(interpretation(S::Coverage, P::PSOut), Interp::System);
        assert_eq!(interpretation(S::Arbitrary, P::PSOut), Interp::NotAvailable);
        assert_eq!(interpretation(S::GroupIndex, P::CSIn), Interp::NotInSig);
        assert_eq!(interpretation(S::CullPrimitive, P::MSPOut), Interp::System);
        assert_eq!(interpretation(S::Invalid, P::VSIn), Interp::NotAvailable);
        assert_eq!(interpretation(S::Target, P::Invalid), Interp::NotAvailable);
    }

    #[test]
    fn sig_points() {
        assert_eq!(sig_point(ShaderKind::Domain, SigKind::PatchConstOrPrim), SigPointKind::DSIn);
        assert_eq!(sig_point(ShaderKind::Mesh, SigKind::PatchConstOrPrim), SigPointKind::MSPOut);
        assert_eq!(sig_point(ShaderKind::Compute, SigKind::Output), SigPointKind::Invalid);
    }
}
