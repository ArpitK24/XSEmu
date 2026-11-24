//! Semantics: names, what each means at a signature point, and interpolation modes, as DXC's
//! tables have them (lib/DXIL/DxilSemantic.cpp, include/dxc/DXIL/DxilSigPoint.inl,
//! lib/DXIL/DxilInterpolationMode.cpp).

use super::pack::{Interpretation, Packing};
use crate::consts::{InterpolationMode, SemanticKind, SigPointKind};
use crate::hlsl::ast::qual;

const KINDS: [SemanticKind; 33] = {
    use SemanticKind::*;
    [
        Arbitrary,
        VertexID,
        InstanceID,
        Position,
        RenderTargetArrayIndex,
        ViewPortArrayIndex,
        ClipDistance,
        CullDistance,
        OutputControlPointID,
        DomainLocation,
        PrimitiveID,
        GSInstanceID,
        SampleIndex,
        IsFrontFace,
        Coverage,
        InnerCoverage,
        Target,
        Depth,
        DepthLessEqual,
        DepthGreaterEqual,
        StencilRef,
        DispatchThreadID,
        GroupID,
        GroupIndex,
        GroupThreadID,
        TessFactor,
        InsideTessFactor,
        ViewID,
        Barycentrics,
        ShadingRate,
        CullPrimitive,
        StartVertexLocation,
        StartInstanceLocation,
    ]
};

/// DXC's interpretation table for shader model 6.6 (the two 6.8 semantics are not available):
/// one row per `SemanticKind`, one column per `SigPointKind` (VSIn, VSOut, PCIn, HSIn, HSCPIn,
/// HSCPOut, PCOut, DSIn, DSCPIn, DSOut, GSVIn, GSIn, GSOut, PSIn, PSOut, CSIn, MSIn, MSOut,
/// MSPOut, ASIn). `-` not available, `A` arbitrary, `S` system value, `G` system-generated
/// value, `N` not in the signature, `P` not packed, `T` target, `F` tessellation factor, `H`
/// shadow, `C` clip / cull distance.
const INTERPRETATION: [&[u8; 20]; 33] = [
    b"AA--AAAAAAA-AA---AA-", // Arbitrary
    b"S-------------------", // VertexID
    b"SA--AA--AAA-AA------", // InstanceID
    b"AS--SSAASSS-SS---S--", // Position
    b"AS--SSAASSS-SS----S-", // RenderTargetArrayIndex
    b"AS--SSAASSS-SS----S-", // ViewPortArrayIndex
    b"AC--CCAACCC-CC---C--", // ClipDistance
    b"AC--CCAACCC-CC---C--", // CullDistance
    b"---N----------------", // OutputControlPointID
    b"-------N------------", // DomainLocation
    b"--NN---N---HGG----S-", // PrimitiveID
    b"-----------N--------", // GSInstanceID
    b"-------------H------", // SampleIndex
    b"------------GG------", // IsFrontFace
    b"-------------NP-----", // Coverage
    b"-------------N------", // InnerCoverage
    b"--------------T-----", // Target
    b"--------------P-----", // Depth
    b"--------------P-----", // DepthLessEqual
    b"--------------P-----", // DepthGreaterEqual
    b"--------------P-----", // StencilRef
    b"---------------NN--N", // DispatchThreadID
    b"---------------NN--N", // GroupID
    b"---------------NN--N", // GroupIndex
    b"---------------NN--N", // GroupThreadID
    b"------FF------------", // TessFactor
    b"------FF------------", // InsideTessFactor
    b"N-NN---N---N-N--N---", // ViewID
    b"-------------P------", // Barycentrics
    b"-S--SS--SSS-SS----S-", // ShadingRate
    b"-------------N----P-", // CullPrimitive
    b"--------------------", // StartVertexLocation (6.8)
    b"--------------------", // StartInstanceLocation (6.8)
];

/// DXC's `SigPoint::GetInterpretation`.
pub fn interpretation(kind: SemanticKind, point: SigPointKind) -> Interpretation {
    match INTERPRETATION.get(kind as usize).and_then(|row| row.get(point as usize)) {
        Some(b'A') => Interpretation::Arb,
        Some(b'S') => Interpretation::SV,
        Some(b'G') => Interpretation::SGV,
        Some(b'N') => Interpretation::NotInSig,
        Some(b'P') => Interpretation::NotPacked,
        Some(b'T') => Interpretation::Target,
        Some(b'F') => Interpretation::TessFactor,
        Some(b'H') => Interpretation::Shadow,
        Some(b'C') => Interpretation::ClipCull,
        _ => Interpretation::NA,
    }
}

/// How the signature a point belongs to is packed; None: the point has no signature of its
/// own (its elements, if any, sit unallocated in a related one) and no interpolation modes.
pub fn packing(point: SigPointKind) -> Option<Packing> {
    use SigPointKind::*;
    match point {
        VSIn => Some(Packing::InputAssembler),
        VSOut | HSCPIn | HSCPOut | DSCPIn | DSOut | GSVIn | GSOut | PSIn | MSOut | MSPOut | PCOut | DSIn => Some(Packing::Vertex),
        PSOut => Some(Packing::Target),
        _ => None,
    }
}

/// DXC's `SigPoint::NeedsInterpMode`: the points packed as vertex data (patch constants are
/// not).
pub fn needs_interp_mode(point: SigPointKind) -> bool {
    packing(point) == Some(Packing::Vertex) && !matches!(point, SigPointKind::PCOut | SigPointKind::DSIn)
}

/// `Semantic::DecomposeNameAndIndex`: "TEXCOORD3" is ("TEXCOORD", 3), "A" is ("A", 0).
pub fn decompose(full: &str) -> (&str, u32) {
    let name = full.trim_end_matches(|c: char| c.is_ascii_digit());
    (name, full[name.len()..].parse().unwrap_or(0))
}

/// `Semantic::GetByName`: Arbitrary without the SV_ prefix, the system value of that name
/// (any case) with it, None for an unknown SV_ name.
pub fn by_name(name: &str) -> Option<SemanticKind> {
    let b = name.as_bytes();
    if !(b.len() >= 3 && b[0].eq_ignore_ascii_case(&b's') && b[1].eq_ignore_ascii_case(&b'v') && b[2] == b'_') {
        return Some(SemanticKind::Arbitrary);
    }
    KINDS[1..].iter().copied().find(|&k| crate::container::sig::semantic_name(k).is_some_and(|n| n.eq_ignore_ascii_case(name)))
}

/// The interpolation mode the qualifiers of a declaration spell (DXC's `interpModeTab`);
/// Undefined without any, None for a combination DXC rejects.
pub fn mode_of(quals: u32) -> Option<InterpolationMode> {
    use InterpolationMode::*;
    let bit = |q: u32, b: u32| if quals & q != 0 { b } else { 0 };
    let i = bit(qual::NOINTERPOLATION, 16) | bit(qual::LINEAR, 8) | bit(qual::NOPERSPECTIVE, 4) | bit(qual::CENTROID, 2) | bit(qual::SAMPLE, 1);
    Some(match i {
        0 => Undefined,
        16 => Constant,
        17.. => return None,
        _ => match (i & 4 != 0, i & 1 != 0, i & 2 != 0) {
            (false, true, _) => LinearSample,
            (false, false, true) => LinearCentroid,
            (false, false, false) => Linear,
            (true, true, _) => LinearNoperspectiveSample,
            (true, false, true) => LinearNoperspectiveCentroid,
            (true, false, false) => LinearNoperspective,
        },
    })
}

/// The mode an element gets (DXC's `ProcessArgument`): constant for per-primitive outputs,
/// none where the point has none, by type where nothing is written (linear for float, constant
/// for the rest), and SV_Position always without perspective. None: an invalid mode
/// (SV_Position that is not linear).
pub fn element_mode(written: InterpolationMode, point: SigPointKind, float: bool, position: bool) -> Option<InterpolationMode> {
    use InterpolationMode::*;
    let mode = if point == SigPointKind::MSPOut {
        Constant
    } else if !needs_interp_mode(point) {
        return Some(Undefined);
    } else if written == Undefined {
        if float {
            Linear
        } else {
            Constant
        }
    } else {
        written
    };
    if !position {
        return Some(mode);
    }
    Some(match mode {
        Linear => LinearNoperspective,
        LinearCentroid => LinearNoperspectiveCentroid,
        LinearSample => LinearNoperspectiveSample,
        LinearNoperspective | LinearNoperspectiveCentroid | LinearNoperspectiveSample => mode,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(decompose("TEXCOORD3"), ("TEXCOORD", 3));
        assert_eq!(decompose("Color01"), ("Color", 1));
        assert_eq!(decompose("A"), ("A", 0));
        assert_eq!(by_name("sv_position"), Some(SemanticKind::Position));
        assert_eq!(by_name("SV_VERTEXID"), Some(SemanticKind::VertexID));
        assert_eq!(by_name("POSITION"), Some(SemanticKind::Arbitrary));
        assert_eq!(by_name("SV_Nothing"), None);
        assert_eq!(by_name("SV_StartVertexLocation"), Some(SemanticKind::StartVertexLocation));
    }

    #[test]
    fn table() {
        use Interpretation::*;
        use SemanticKind as K;
        use SigPointKind as P;
        assert_eq!(interpretation(K::Position, P::VSIn), Arb);
        assert_eq!(interpretation(K::Position, P::VSOut), SV);
        assert_eq!(interpretation(K::InstanceID, P::PSIn), Arb);
        assert_eq!(interpretation(K::PrimitiveID, P::GSIn), Shadow);
        assert_eq!(interpretation(K::PrimitiveID, P::GSOut), SGV);
        assert_eq!(interpretation(K::PrimitiveID, P::MSPOut), SV);
        assert_eq!(interpretation(K::IsFrontFace, P::PSIn), SGV);
        assert_eq!(interpretation(K::Coverage, P::PSIn), NotInSig);
        assert_eq!(interpretation(K::Coverage, P::PSOut), NotPacked);
        assert_eq!(interpretation(K::Target, P::PSOut), Target);
        assert_eq!(interpretation(K::ClipDistance, P::MSOut), ClipCull);
        assert_eq!(interpretation(K::GroupID, P::MSIn), NotInSig);
        assert_eq!(interpretation(K::GroupID, P::VSIn), NA);
        assert_eq!(interpretation(K::StartVertexLocation, P::VSIn), NA);
        // the container's reduced table agrees wherever an element exists
        for (i, &k) in KINDS.iter().enumerate() {
            assert_eq!(k as usize, i);
            for p in 0..20u32 {
                let point = [P::VSIn, P::VSOut, P::PCIn, P::HSIn, P::HSCPIn, P::HSCPOut, P::PCOut, P::DSIn, P::DSCPIn, P::DSOut, P::GSVIn, P::GSIn, P::GSOut, P::PSIn, P::PSOut, P::CSIn, P::MSIn, P::MSOut, P::MSPOut, P::ASIn][p as usize];
                assert_eq!(point as u32, p);
                let ours = interpretation(k, point);
                let theirs = crate::container::sig::interpretation(k, point);
                use crate::container::sig::Interp;
                let want = match ours {
                    NA => Interp::NotAvailable,
                    NotInSig => Interp::NotInSig,
                    Arb => Interp::Arbitrary,
                    _ => Interp::System,
                };
                // the container reads a GS primitive id in the vertex input signature as GSIn
                if !(k == K::PrimitiveID && point == P::GSVIn) && !matches!(k, K::StartVertexLocation | K::StartInstanceLocation) {
                    assert_eq!(theirs, want, "{k:?} at {point:?}");
                }
            }
        }
    }

    #[test]
    fn modes() {
        use InterpolationMode::*;
        assert_eq!(mode_of(0), Some(Undefined));
        assert_eq!(mode_of(qual::NOINTERPOLATION), Some(Constant));
        assert_eq!(mode_of(qual::SAMPLE), Some(LinearSample));
        assert_eq!(mode_of(qual::CENTROID), Some(LinearCentroid));
        assert_eq!(mode_of(qual::NOPERSPECTIVE | qual::SAMPLE), Some(LinearNoperspectiveSample));
        assert_eq!(mode_of(qual::NOPERSPECTIVE | qual::CENTROID | qual::LINEAR), Some(LinearNoperspectiveCentroid));
        assert_eq!(mode_of(qual::NOPERSPECTIVE), Some(LinearNoperspective));
        assert_eq!(mode_of(qual::NOINTERPOLATION | qual::LINEAR), None);
        assert_eq!(element_mode(Undefined, SigPointKind::PSIn, true, false), Some(Linear));
        assert_eq!(element_mode(Undefined, SigPointKind::PSIn, false, false), Some(Constant));
        assert_eq!(element_mode(Undefined, SigPointKind::VSOut, true, true), Some(LinearNoperspective));
        assert_eq!(element_mode(LinearCentroid, SigPointKind::PSIn, true, true), Some(LinearNoperspectiveCentroid));
        assert_eq!(element_mode(Linear, SigPointKind::VSIn, true, false), Some(Undefined));
        assert_eq!(element_mode(Linear, SigPointKind::MSPOut, true, false), Some(Constant));
        assert_eq!(element_mode(Constant, SigPointKind::PSIn, true, true), None);
    }
}
