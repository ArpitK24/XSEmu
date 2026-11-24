//! The DXBC container around the bitcode, as DXC's `SerializeDxilContainerForModule`
//! (lib/DxilContainer/DxilContainerAssembler.cpp) assembles it for a non-library shader and
//! validator 1.8: the parts SFI0, ISG1, OSG1, PSG1 (only with patch constant or primitive
//! elements), PSV0, HASH and DXIL in that order, then the digest in the header that D3D12
//! demands of every DXIL shader.
//!
//! The DXIL validator rebuilds SFI0, the signatures and PSV0 from the module's dx.* metadata
//! and compares them with the container byte for byte, so each serialiser here follows DXC's
//! writer of the same part, including what it does with input DXC itself never produces.
//! What DXC adds for tools only is left out: STAT (reflection), ILDB / ILDN (debug info and
//! its name), RTS0 (a root signature from the source; XCL sets its own on the pipeline).
//! `dxc -Qstrip_reflect` writes exactly the parts written here.

pub mod hash;
pub mod psv;
pub mod sig;
pub mod stdsort;

pub use sig::{semantic_name, SigKind};

use crate::consts::*;
use crate::shader::{ShaderInfo, DXIL_VERSION};
use crate::{Error, Result};

/// (shader flag bit, feature bit): DXC's `ShaderFlags::GetFeatureInfo`, the flags that say what
/// the shader needs of the device (`D3D_SHADER_REQUIRES_*`). The two low-precision features
/// are derived from a pair of flags in [`feature_info`].
const FEATURES: [(u8, u8); 33] = [
    (2, 0),   // EnableDoublePrecision → Doubles
    (17, 1),  // CSRawAndStructuredViaShader4X → ComputeShadersPlusRawAndStructuredBuffersViaShader4X
    (16, 2),  // UAVsAtEveryStage
    (15, 3),  // 64UAVs
    (6, 5),   // EnableDoubleExtensions → 11_1_DoubleExtensions
    (7, 6),   // EnableMSAD → 11_1_ShaderExtensions
    (14, 7),  // Level9ComparisonFiltering
    (12, 8),  // TiledResources
    (11, 9),  // StencilRef
    (10, 10), // InnerCoverage
    (13, 11), // UAVLoadAdditionalFormats → TypedUAVLoadAdditionalFormats
    (18, 12), // ROVs
    (9, 13),  // ViewportAndRTArrayIndex → ViewportAndRTArrayIndexFromAnyShaderFeedingRasterizer
    (19, 14), // WaveOps
    (20, 15), // Int64Ops
    (21, 16), // ViewID
    (22, 17), // Barycentrics
    (24, 19), // ShadingRate
    (25, 20), // RaytracingTier1_1
    (26, 21), // SamplerFeedback
    (27, 22), // AtomicInt64OnTypedResource
    (28, 23), // AtomicInt64OnGroupShared
    (29, 24), // DerivativesInMeshAndAmpShaders
    (30, 25), // ResourceDescriptorHeapIndexing
    (31, 26), // SamplerDescriptorHeapIndexing
    (32, 28), // AtomicInt64OnHeapResource
    (34, 29), // AdvancedTextureOps
    (35, 30), // WriteableMSAATextures
    (37, 31), // SampleCmpGradientOrBias
    (38, 32), // ExtendedCommandInfo
    (41, 33), // LinearAlgebra
    (39, 40), // UsesDerivatives → OptFeatureInfo_UsesDerivatives
    (40, 41), // RequiresGroup → OptFeatureInfo_RequiresGroup
];

/// SFI0's feature bits for a module with these DXIL shader flags.
pub fn feature_info(flags: u64) -> u64 {
    use crate::flags::bit::{LOW_PRECISION_PRESENT, USE_NATIVE_LOW_PRECISION};
    let mut features = FEATURES.iter().filter(|&&(flag, _)| flags >> flag & 1 != 0).fold(0u64, |f, &(_, feature)| f | 1 << feature);
    if flags & LOW_PRECISION_PRESENT != 0 {
        // MinimumPrecision, or NativeLowPrecision with 16-bit types enabled
        features |= if flags & USE_NATIVE_LOW_PRECISION != 0 { 1 << 18 } else { 1 << 4 };
    }
    features
}

/// The bytes before the first part: "DXBC", digest, version 1.0, size, part count, offsets.
fn header_size(parts: usize) -> usize {
    32 + 4 * parts
}

/// Assemble and sign the container for `bitcode`, the module [`crate::bcwriter`] wrote from the
/// same `info` that [`crate::meta`] turned into its metadata.
pub fn build(info: &ShaderInfo, bitcode: &[u8]) -> Result<Vec<u8>> {
    let mut out = open(info)?;
    out.reserve_exact(bitcode.len() + 3);
    out.extend_from_slice(bitcode);
    close(out)
}

/// The container up to where the bitcode starts: the header, the parts before DXIL, and the
/// DXIL part's own headers, with the sizes that depend on the bitcode and the hashes still
/// open. The bitcode is appended to it ([`crate::bcwriter::write_sealed_into`] writes it
/// there, so the 13 MB of a large program are written once and never copied) and [`close`]
/// finishes it.
pub fn open(info: &ShaderInfo) -> Result<Vec<u8>> {
    if !matches!(
        info.kind,
        ShaderKind::Pixel | ShaderKind::Vertex | ShaderKind::Geometry | ShaderKind::Hull | ShaderKind::Domain | ShaderKind::Compute | ShaderKind::Mesh | ShaderKind::Amplification
    ) {
        return crate::unsupported(format!("container for a {:?} shader", info.kind));
    }
    let mut parts: Vec<(&[u8; 4], Vec<u8>)> = vec![
        (b"SFI0", feature_info(info.flags).to_le_bytes().to_vec()),
        (b"ISG1", sig::write(info, SigKind::Input)?),
        (b"OSG1", sig::write(info, SigKind::Output)?),
    ];
    if !info.patch_or_prim.is_empty() {
        parts.push((b"PSG1", sig::write(info, SigKind::PatchConstOrPrim)?));
    }
    parts.push((b"PSV0", psv::write(info)?));
    // DxilShaderHash: flags (0: the hash is of the bitcode alone, no source), MD5
    parts.push((b"HASH", vec![0u8; 20]));

    // the sizes that hold the bitcode's length are written by `close`
    let (size, dxil_size) = (0, 0);
    let mut out = Vec::with_capacity(header_size(parts.len() + 1) + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>() + 32);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0u8; 16]);
    out.extend_from_slice(&[1, 0, 0, 0]);
    let put = |out: &mut Vec<u8>, v: usize| out.extend_from_slice(&(v as u32).to_le_bytes());
    put(&mut out, size);
    put(&mut out, parts.len() + 1);
    let mut offset = header_size(parts.len() + 1);
    for (_, data) in &parts {
        put(&mut out, offset);
        offset += 8 + data.len();
    }
    put(&mut out, offset);
    for (name, data) in &parts {
        out.extend_from_slice(*name);
        put(&mut out, data.len());
        out.extend_from_slice(data);
    }
    out.extend_from_slice(b"DXIL");
    put(&mut out, dxil_size);
    put(&mut out, (info.kind as usize) << 16 | (info.sm.0 as usize) << 4 | info.sm.1 as usize);
    put(&mut out, dxil_size / 4);
    out.extend_from_slice(b"DXIL");
    put(&mut out, (DXIL_VERSION.0 as usize) << 8 | DXIL_VERSION.1 as usize);
    put(&mut out, 16);
    put(&mut out, 0);
    Ok(out)
}

/// Finish a container that [`open`] started and the bitcode was appended to: pad the bitcode
/// to whole words, write the sizes, the bitcode's MD5 into HASH, and the digest.
pub fn close(mut out: Vec<u8>) -> Result<Vec<u8>> {
    let word = |out: &[u8], at: usize| out.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
    let open = || Error::Internal("not a container that container::open started".to_string());
    // the last two parts are HASH and DXIL
    let count = word(&out, 28).filter(|&n| n >= 2).ok_or_else(open)?;
    let (hash, dxil) = (word(&out, 32 + 4 * (count - 2)).ok_or_else(open)?, word(&out, 32 + 4 * (count - 1)).ok_or_else(open)?);
    let start = dxil + 32;
    if out.get(hash..hash + 4) != Some(&b"HASH"[..]) || hash + 28 != dxil || out.get(dxil..dxil + 4) != Some(&b"DXIL"[..]) || out.len() < start {
        return Err(open());
    }
    let bitcode = out.len() - start;
    // DxilProgramHeader in front of the bitcode, which is padded to whole words
    let dxil_size = 24 + ((bitcode + 3) & !3);
    let size = dxil + 8 + dxil_size;
    if size > u32::MAX as usize {
        return Err(Error::Internal(format!("container of {size} bytes")));
    }
    let md5 = hash::md5(&out[start..]);
    out[hash + 12..hash + 28].copy_from_slice(&md5);
    out.resize(size, 0);
    let mut put = |at: usize, v: usize| out[at..at + 4].copy_from_slice(&(v as u32).to_le_bytes());
    put(24, size);
    put(dxil + 4, dxil_size);
    put(dxil + 12, dxil_size / 4);
    put(dxil + 28, bitcode);
    sign(&mut out);
    Ok(out)
}

/// Write the digest of a complete container into its header (what `dxv -o` and dxil.dll's
/// validator do after a successful validation).
pub fn sign(container: &mut [u8]) {
    if container.len() >= 20 {
        let digest = hash::digest(&container[20..]);
        container[4..20].copy_from_slice(&digest);
    }
}

/// Whether the digest in the header is the container's.
pub fn is_signed(container: &[u8]) -> bool {
    container.len() >= 20 && container[4..20] == hash::digest(&container[20..])
}

/// The parts of a container, in order: (four-character code, payload). `None` if the header,
/// the offset table or a part does not fit the blob.
pub fn parts(container: &[u8]) -> Option<Vec<([u8; 4], &[u8])>> {
    let word = |at: usize| Some(u32::from_le_bytes(container.get(at..at.checked_add(4)?)?.try_into().ok()?) as usize);
    if container.get(..4)? != b"DXBC" || word(24)? != container.len() {
        return None;
    }
    let count = word(28)?;
    let mut out = Vec::with_capacity(count.min(64));
    for i in 0..count {
        let at = word(32usize.checked_add(i.checked_mul(4)?)?)?;
        let name = container.get(at..at.checked_add(4)?)?.try_into().ok()?;
        let size = word(at.checked_add(4)?)?;
        out.push((name, container.get(at + 8..(at + 8).checked_add(size)?)?));
    }
    Some(out)
}

/// The payload of the first part called `name`.
pub fn part<'a>(container: &'a [u8], name: &[u8; 4]) -> Option<&'a [u8]> {
    parts(container)?.into_iter().find(|(n, _)| n == name).map(|(_, d)| d)
}

/// The LLVM bitcode inside a container's DXIL part.
pub fn bitcode(container: &[u8]) -> Option<&[u8]> {
    let p = part(container, b"DXIL")?;
    let word = |at: usize| Some(u32::from_le_bytes(p.get(at..at + 4)?.try_into().ok()?) as usize);
    if p.get(8..12)? != b"DXIL" {
        return None;
    }
    let start = 8usize.checked_add(word(16)?)?;
    p.get(start..start.checked_add(word(20)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::{Resource, SigElement, StageProps};

    fn compute() -> ShaderInfo {
        let mut info = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [64, 1, 1], wave_size: None });
        info.resources.push(Resource::new(ResourceClass::UAV, 0, "io", ResourceKind::RawBuffer, 0, 0));
        info
    }

    /// The header and part table of a container, and what xcl's cache asks of a blob before
    /// it hands it to D3D12 (`valid_dxil_container` in gpu::recomp::dxc): the header's size,
    /// 4-aligned parts inside the blob that do not overlap, a DXIL part with its magic.
    #[test]
    fn layout() {
        for len in [0usize, 1, 2, 3, 4, 61, 4096] {
            let bitcode: Vec<u8> = (0..len).map(|i| i as u8).collect();
            let c = build(&compute(), &bitcode).unwrap();
            assert_eq!(&c[..4], b"DXBC");
            assert_eq!(c[20..24], [1, 0, 0, 0]);
            assert_eq!(u32::from_le_bytes(c[24..28].try_into().unwrap()) as usize, c.len());
            assert!(is_signed(&c));
            let p = parts(&c).unwrap();
            let names: Vec<&[u8; 4]> = p.iter().map(|(n, _)| n).collect();
            assert_eq!(names, [b"SFI0", b"ISG1", b"OSG1", b"PSV0", b"HASH", b"DXIL"]);
            let mut end = header_size(p.len());
            for (i, (_, data)) in p.iter().enumerate() {
                let at = u32::from_le_bytes(c[32 + 4 * i..36 + 4 * i].try_into().unwrap()) as usize;
                assert_eq!(at, end, "parts follow each other without gaps");
                assert_eq!(at % 4, 0);
                end = at + 8 + data.len();
            }
            assert_eq!(end, c.len());
            let dxil = part(&c, b"DXIL").unwrap();
            // program version: compute (5) 6.6, then the size in words, the magic and DXIL 1.6
            assert_eq!(dxil[..4], [0x66, 0, 5, 0]);
            assert_eq!(u32::from_le_bytes(dxil[4..8].try_into().unwrap()) as usize * 4, dxil.len());
            assert_eq!(&dxil[8..12], b"DXIL");
            assert_eq!(dxil[12..16], [6, 1, 0, 0]);
            assert_eq!(self::bitcode(&c).unwrap(), &bitcode[..]);
            assert_eq!(part(&c, b"HASH").unwrap()[4..], hash::md5(&bitcode));
        }
    }

    #[test]
    fn signature_parts() {
        let mut info = ShaderInfo::new(ShaderKind::Hull, "main", StageProps::Vertex);
        assert!(build(&info, &[]).is_err(), "stage properties of another stage");
        info = ShaderInfo::new(ShaderKind::Library, "main", StageProps::Vertex);
        assert!(matches!(build(&info, &[]), Err(Error::Unsupported(_))));

        let mut info = ShaderInfo::new(ShaderKind::Mesh, "main", StageProps::Mesh { numthreads: [1, 1, 1], max_vertices: 3, max_primitives: 1, topology: MeshOutputTopology::Triangle, payload_bytes: 0 });
        let element = |name: &str, kind, start_row| SigElement {
            name: name.to_string(),
            indices: vec![0],
            kind,
            comp_type: ComponentType::F32,
            interp: InterpolationMode::Linear,
            rows: 1,
            cols: 4,
            start_row,
            start_col: 0,
            stream: 0,
            usage_mask: 0xF,
            dynamic_index_mask: 0,
        };
        info.outputs.push(element("SV_Position", SemanticKind::Position, 0));
        info.patch_or_prim.push(element("COLOR", SemanticKind::Arbitrary, 0));
        let c = build(&info, &[]).unwrap();
        let names: Vec<[u8; 4]> = parts(&c).unwrap().iter().map(|(n, _)| *n).collect();
        assert_eq!(names, [*b"SFI0", *b"ISG1", *b"OSG1", *b"PSG1", *b"PSV0", *b"HASH", *b"DXIL"]);
        // an element where its semantic means nothing is the front end's mistake
        info.inputs.push(element("COLOR", SemanticKind::Arbitrary, 0));
        assert!(matches!(build(&info, &[]), Err(Error::Internal(_))));
    }

    #[test]
    fn parts_of_garbage() {
        assert!(parts(b"").is_none());
        assert!(parts(b"DXBC").is_none());
        let mut c = build(&compute(), &[1, 2, 3]).unwrap();
        assert!(parts(&c[..c.len() - 1]).is_none());
        let n = c.len();
        c[n - 1] ^= 1;
        assert!(!is_signed(&c));
        // a part offset past the end
        c[32..36].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parts(&c).is_none());
    }
}
