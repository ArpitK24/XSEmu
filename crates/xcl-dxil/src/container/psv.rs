//! PSV0, "pipeline state validation": what the D3D12 runtime reads instead of the bitcode when
//! it checks a shader against a pipeline and a root signature (stage properties, resource
//! bindings, the signatures with their packing, and which outputs depend on which inputs).
//! Laid out as DXC's `DxilPipelineStateValidation::ReadOrWrite` and filled as its
//! `DxilPSVWriter` for PSV version 3, the one validators 1.8 and 1.9 expect: runtime info of
//! 52 bytes ending in the entry name, resource records of 24 bytes.

use super::sig::{resolve, sig_comp_type, Resolved, SigKind};
use crate::consts::*;
use crate::shader::{Resource, ShaderInfo, StageProps};
use crate::{Error, Result};

/// `sizeof(PSVRuntimeInfo3)`
const RUNTIME_INFO_SIZE: u32 = 52;
/// `sizeof(PSVResourceBindInfo1)`
const BIND_INFO_SIZE: u32 = 24;
/// `sizeof(PSVSignatureElement0)`
const ELEMENT_SIZE: u32 = 16;

fn put(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// DXC's `DxilSignature::NumVectorsUsed`: registers of one stream up to the last allocated row.
fn vectors(elements: &[Resolved], stream: u32) -> Result<u8> {
    let n = elements.iter().filter(|r| r.allocated() && r.e.stream == stream).map(|r| r.e.start_row as u32 + r.e.rows).max().unwrap_or(0);
    u8::try_from(n).map_err(|_| Error::Internal(format!("signature uses {n} registers")))
}

/// Words of a mask with a bit per component of `vectors` registers
/// (`PSVComputeMaskDwordsFromVectors`).
fn mask_words(vectors: u8) -> usize {
    (vectors as usize + 7) >> 3
}

/// DXC's `InitPSVResourceBinding`: PSVResourceType.
fn resource_type(r: &Resource) -> u32 {
    let uav = r.class == ResourceClass::UAV;
    match r.kind {
        ResourceKind::Sampler => 1,
        ResourceKind::CBuffer => 2,
        ResourceKind::StructuredBuffer if uav && r.has_counter => 9,
        ResourceKind::StructuredBuffer if uav => 8,
        ResourceKind::StructuredBuffer => 5,
        ResourceKind::RTAccelerationStructure => 4,
        ResourceKind::RawBuffer if uav => 7,
        ResourceKind::RawBuffer => 4,
        _ if uav => 6,
        _ => 3,
    }
}

/// The dependency tables of [`ShaderInfo::view_id_state`], taken apart as DXC's
/// `StoreViewIDStateToPSV` does: PSV0 keeps the ViewID masks and the input → output tables in
/// separate runs, padded to whole registers.
#[derive(Default)]
struct Tables {
    /// per stream: outputs that depend on SV_ViewID
    view_id: [Vec<u32>; 4],
    /// HS patch constant / MS primitive outputs that depend on SV_ViewID
    view_id_pc: Vec<u32>,
    /// per stream: for each input component, the outputs it contributes to
    in_out: [Vec<u32>; 4],
    /// HS: inputs → patch constant outputs
    in_pc: Vec<u32>,
    /// DS: patch constant inputs → outputs
    pc_out: Vec<u32>,
}

struct Counts {
    uses_view_id: bool,
    input: u8,
    output: [u8; 4],
    pc: u8,
}

impl Tables {
    fn new(kind: ShaderKind, c: &Counts) -> Tables {
        let mut t = Tables::default();
        let streams = if kind == ShaderKind::Geometry { 4 } else { 1 };
        for s in 0..streams {
            if c.uses_view_id {
                t.view_id[s] = vec![0; mask_words(c.output[s])];
            }
            if kind != ShaderKind::Mesh {
                t.in_out[s] = vec![0; mask_words(c.output[s]) * c.input as usize * 4];
            }
        }
        if c.uses_view_id && matches!(kind, ShaderKind::Hull | ShaderKind::Mesh) {
            t.view_id_pc = vec![0; mask_words(c.pc)];
        }
        if kind == ShaderKind::Hull {
            t.in_pc = vec![0; mask_words(c.pc) * c.input as usize * 4];
        }
        if kind == ShaderKind::Domain {
            t.pc_out = vec![0; mask_words(c.output[0]) * c.pc as usize * 4];
        }
        t
    }

    /// Fill from the serialised state (the layout of dx.viewIdState, see
    /// [`ShaderInfo::view_id_state`]).
    fn load(&mut self, kind: ShaderKind, c: &Counts, state: &[u32]) -> Result<()> {
        fn bad(what: &str) -> Error {
            Error::Internal(format!("view_id_state does not describe this shader's signatures: {what}"))
        }
        struct Words<'a>(&'a [u32]);
        impl<'a> Words<'a> {
            fn take(&mut self, n: usize) -> Result<&'a [u32]> {
                if n > self.0.len() {
                    return Err(bad("too short"));
                }
                let (head, rest) = self.0.split_at(n);
                self.0 = rest;
                Ok(head)
            }
            /// A component count: up to the last used component, where PSV0 counts whole
            /// registers.
            fn scalars(&mut self, vectors: u8, what: &str) -> Result<usize> {
                let n = self.take(1)?[0];
                if n > vectors as u32 * 4 || vectors as u32 * 4 - n >= 4 {
                    return Err(bad(what));
                }
                Ok(n as usize)
            }
        }
        let mut words = Words(state);
        let inputs = words.scalars(c.input, "input components")?;
        // one output set: its ViewID mask, then a row of output bits per input component
        let mut set = |outputs: u8, mask: Option<&mut Vec<u32>>, table: Option<&mut Vec<u32>>| -> Result<()> {
            words.scalars(outputs, "output components")?;
            let n = mask_words(outputs);
            if let Some(mask) = mask {
                mask.copy_from_slice(words.take(n)?);
            }
            if let Some(table) = table {
                table[..n * inputs].copy_from_slice(words.take(n * inputs)?);
            }
            Ok(())
        };
        let streams = if kind == ShaderKind::Geometry { 4 } else { 1 };
        for s in 0..streams {
            let mask = c.uses_view_id.then_some(&mut self.view_id[s]);
            let table = (kind != ShaderKind::Mesh).then_some(&mut self.in_out[s]);
            set(c.output[s], mask, table)?;
        }
        match kind {
            ShaderKind::Hull | ShaderKind::Mesh => {
                let mask = c.uses_view_id.then_some(&mut self.view_id_pc);
                let table = (kind == ShaderKind::Hull).then_some(&mut self.in_pc);
                set(c.pc, mask, table)?;
            }
            ShaderKind::Domain => {
                let pcs = words.scalars(c.pc, "patch constant components")?;
                let n = mask_words(c.output[0]) * pcs;
                self.pc_out[..n].copy_from_slice(words.take(n)?);
            }
            _ => {}
        }
        if !words.0.is_empty() {
            return Err(bad("too long"));
        }
        Ok(())
    }
}

/// The PSV0 part's payload.
pub fn write(info: &ShaderInfo) -> Result<Vec<u8>> {
    let inputs = resolve(info.kind, SigKind::Input, &info.inputs)?;
    let outputs = resolve(info.kind, SigKind::Output, &info.outputs)?;
    let pcs = resolve(info.kind, SigKind::PatchConstOrPrim, &info.patch_or_prim)?;
    let count = |v: &[Resolved]| u8::try_from(v.len()).map_err(|_| Error::Internal(format!("{} signature elements", v.len())));
    let has_pc = matches!(info.kind, ShaderKind::Hull | ShaderKind::Domain | ShaderKind::Mesh);
    let c = Counts {
        uses_view_id: info.flags & crate::flags::bit::VIEW_ID != 0,
        input: vectors(&inputs, 0)?,
        output: [vectors(&outputs, 0)?, vectors(&outputs, 1)?, vectors(&outputs, 2)?, vectors(&outputs, 3)?],
        pc: if has_pc { vectors(&pcs, 0)? } else { 0 },
    };

    // PSVRuntimeInfo3: the stage's own 16 bytes, the wave lane range, then the shared fields
    let mut ri = [0u8; RUNTIME_INFO_SIZE as usize];
    let mut set = |at: usize, v: u32| ri[at..at + 4].copy_from_slice(&v.to_le_bytes());
    let has_position = outputs.iter().any(|r| r.kind == SemanticKind::Position) as u32;
    let mut lanes = (0u32, u32::MAX);
    let mut threads = [0u32; 3];
    let mut at26 = c.pc as u32;
    let mismatch = || Error::Internal(format!("stage properties {:?} for a {:?} shader", info.props, info.kind));
    match (&info.props, info.kind) {
        (StageProps::Vertex, ShaderKind::Vertex) => set(0, has_position),
        (StageProps::Pixel { .. }, ShaderKind::Pixel) => {
            let depth = outputs.iter().any(|r| matches!(r.kind, SemanticKind::Depth | SemanticKind::DepthLessEqual | SemanticKind::DepthGreaterEqual));
            let sample = inputs.iter().any(|r| {
                matches!(r.e.interp, InterpolationMode::LinearSample | InterpolationMode::LinearNoperspectiveSample) || r.kind == SemanticKind::SampleIndex
            });
            set(0, depth as u32 | (sample as u32) << 8);
        }
        (StageProps::Geometry { input_primitive, max_vertex_count, active_streams, stream_topology, .. }, ShaderKind::Geometry) => {
            set(0, *input_primitive as u32);
            set(4, *stream_topology as u32);
            // "what the runtime expects" of a shader that writes no stream
            set(8, if *active_streams == 0 { 1 } else { *active_streams });
            set(12, has_position);
            at26 = *max_vertex_count & 0xFFFF;
        }
        (StageProps::Hull { input_control_points, output_control_points, domain, output_primitive, .. }, ShaderKind::Hull) => {
            set(0, *input_control_points);
            set(4, *output_control_points);
            set(8, *domain as u32);
            set(12, *output_primitive as u32);
        }
        (StageProps::Domain { domain, input_control_points }, ShaderKind::Domain) => {
            set(0, *input_control_points);
            set(4, has_position);
            set(8, *domain as u32);
        }
        (StageProps::Compute { numthreads, wave_size }, ShaderKind::Compute) => {
            if let Some(n) = wave_size {
                lanes = (*n, *n);
            }
            threads = *numthreads;
        }
        (StageProps::Mesh { numthreads, max_vertices, max_primitives, topology, payload_bytes }, ShaderKind::Mesh) => {
            set(0, info.groupshared_bytes);
            set(8, *payload_bytes);
            set(12, (*max_vertices & 0xFFFF) | (*max_primitives & 0xFFFF) << 16);
            at26 |= (*topology as u32 & 0xFF) << 8;
            threads = *numthreads;
        }
        (StageProps::Amplification { numthreads, payload_bytes }, ShaderKind::Amplification) => {
            set(0, *payload_bytes);
            threads = *numthreads;
        }
        _ => return Err(mismatch()),
    }
    set(16, lanes.0);
    set(20, lanes.1);
    set(24, info.kind as u32 | (c.uses_view_id as u32) << 8 | at26 << 16);
    set(28, u32::from_le_bytes([count(&inputs)?, count(&outputs)?, count(&pcs)?, c.input]));
    set(32, u32::from_le_bytes(c.output));
    set(36, threads[0]);
    set(40, threads[1]);
    set(44, threads[2]);

    // string table: an empty name for system values first, a string per arbitrary element
    // (not shared), then the entry point's name; the index table shares runs
    let mut strings = vec![0u8];
    let mut indices: Vec<u32> = Vec::new();
    let mut elements = Vec::with_capacity((inputs.len() + outputs.len() + pcs.len()) * ELEMENT_SIZE as usize);
    for r in inputs.iter().chain(&outputs).chain(&pcs) {
        let e = r.e;
        let name = if r.kind == SemanticKind::Arbitrary && !r.name.is_empty() {
            strings.extend_from_slice(r.name.as_bytes());
            strings.push(0);
            strings.len() - r.name.len() - 1
        } else {
            0
        };
        let index = match indices.windows(e.indices.len()).position(|w| w == e.indices.as_slice()) {
            Some(at) => at,
            None => {
                indices.extend_from_slice(&e.indices);
                indices.len() - e.indices.len()
            }
        };
        put(&mut elements, name as u32);
        put(&mut elements, index as u32);
        let (start_row, cols_and_start) = if r.allocated() { (e.start_row as u8, e.cols as u8 | (e.start_col as u8) << 4 | 0x40) } else { (0, e.cols as u8) };
        elements.extend_from_slice(&[
            e.rows as u8,
            start_row,
            cols_and_start,
            r.kind as u8,
            sig_comp_type(e.comp_type) as u8,
            e.interp as u8,
            (e.stream as u8) << 4 | (e.dynamic_index_mask & 0xF),
            0,
        ]);
    }
    let entry = strings.len() as u32;
    strings.extend_from_slice(info.entry.as_bytes());
    strings.push(0);
    strings.resize((strings.len() + 3) & !3, 0);
    ri[48..52].copy_from_slice(&entry.to_le_bytes());

    let mut tables = Tables::new(info.kind, &c);
    if !info.view_id_state.is_empty() {
        tables.load(info.kind, &c, &info.view_id_state)?;
    }

    let mut out = Vec::with_capacity(256 + strings.len() + elements.len());
    put(&mut out, RUNTIME_INFO_SIZE);
    out.extend_from_slice(&ri);
    // resources by class in DXC's order (cbuffers, samplers, SRVs, UAVs), each class by id
    put(&mut out, info.resources.len() as u32);
    if !info.resources.is_empty() {
        put(&mut out, BIND_INFO_SIZE);
        let mut sorted: Vec<&Resource> = info.resources.iter().collect();
        let class_order = |r: &Resource| match r.class {
            ResourceClass::CBuffer => 0,
            ResourceClass::Sampler => 1,
            ResourceClass::SRV => 2,
            _ => 3,
        };
        sorted.sort_by_key(|r| (class_order(r), r.id));
        for r in sorted {
            let upper = if r.range_size == u32::MAX { u32::MAX } else { r.lower_bound.wrapping_add(r.range_size).wrapping_sub(1) };
            let flags = (r.class == ResourceClass::UAV && r.atomic64_use) as u32;
            for w in [resource_type(r), r.space, r.lower_bound, upper, r.kind as u32, flags] {
                put(&mut out, w);
            }
        }
    }
    put(&mut out, strings.len() as u32);
    out.extend_from_slice(&strings);
    put(&mut out, indices.len() as u32);
    for &i in &indices {
        put(&mut out, i);
    }
    if !elements.is_empty() {
        put(&mut out, ELEMENT_SIZE);
        out.extend_from_slice(&elements);
    }
    let Tables { view_id, view_id_pc, in_out, in_pc, pc_out } = &tables;
    for run in view_id.iter().chain([view_id_pc]).chain(in_out).chain([in_pc, pc_out]) {
        for &w in run {
            put(&mut out, w);
        }
    }
    Ok(out)
}
