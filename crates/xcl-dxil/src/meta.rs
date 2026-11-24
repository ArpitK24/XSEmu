//! dx.* module metadata from a [`ShaderInfo`], as DXC 1.9 writes it for shader model 6.6 and
//! validator version [`VALIDATOR_VERSION`] (lib/DXIL/DxilMetadataHelper.cpp, DxilModule.cpp,
//! with reflection stripped as `dxc` does by default: no resource names, no
//! dx.typeAnnotations): llvm.ident, dx.version, dx.valver, dx.shaderModel, dx.resources,
//! dx.viewIdState and dx.entryPoints (signatures, resources, shader flags and stage state).
//!
//! [`emit`] only serialises: the flags, the 64-bit-atomic marks of the resources and the
//! ViewID state it writes are those in the `ShaderInfo`, because the container
//! ([`crate::container`]) serialises the same values into SFI0 and PSV0 and the validator
//! compares the two. [`prepare`] computes all three from the module in the order they depend
//! on each other.
//!
//! [`view_id_state`] is DXC's ComputeViewIdState pass (lib/HLSL/ComputeViewIdStateBuilder.cpp):
//! which input scalars and whether SV_ViewID contribute to each output scalar, through data,
//! memory and control dependence.
//!
//! Owner: the metadata subagent.

use crate::bcwriter::{liveness, successors, Live};
use crate::consts::*;
use crate::flags;
use crate::ir::*;
use crate::shader::*;
use crate::{Error, Result};
use std::collections::{BTreeSet, HashMap};

/// Compute what [`emit`] and the container need from the finished module: the shader flags,
/// [`Resource::atomic64_use`] and the ViewID state.
pub fn prepare(m: &Module, info: &mut ShaderInfo) {
    let scan = flags::scan(m, info);
    info.flags = scan.flags;
    for (r, used) in info.resources.iter_mut().zip(scan.atomic64) {
        r.atomic64_use |= used;
    }
    info.view_id_state = view_id_state(m, info);
}

/// [`prepare`] for a sealed module: one scan for the flags and the atomic marks, and the
/// liveness the module holds for all three.
pub fn prepare_sealed(s: &Sealed, info: &mut ShaderInfo) {
    let scan = flags::scan_sealed(s, info);
    info.flags = scan.flags;
    for (r, used) in info.resources.iter_mut().zip(scan.atomic64) {
        r.atomic64_use |= used;
    }
    info.view_id_state = view_id_state_live(s.module(), info, s.lives());
}

/// Extended-property tags (DxilMDHelper): resources, signature elements, entry properties.
mod tag {
    pub const TYPED_BUFFER_ELEMENT_TYPE: u32 = 0;
    pub const STRUCTURED_BUFFER_ELEMENT_STRIDE: u32 = 1;
    pub const ATOMIC64_USE: u32 = 3;

    pub const SIG_OUTPUT_STREAM: u32 = 0;
    pub const SIG_DYN_IDX_COMP_MASK: u32 = 2;
    pub const SIG_USAGE_COMP_MASK: u32 = 3;

    pub const SHADER_FLAGS: u32 = 0;
    pub const GS_STATE: u32 = 1;
    pub const DS_STATE: u32 = 2;
    pub const HS_STATE: u32 = 3;
    pub const NUM_THREADS: u32 = 4;
    pub const MS_STATE: u32 = 9;
    pub const AS_STATE: u32 = 10;
    pub const WAVE_SIZE: u32 = 11;
}

/// Add the module metadata. The entry function is the defined function named `info.entry`
/// (and, for a hull shader, the patch constant function named in its properties).
pub fn emit(m: &mut Module, info: &ShaderInfo) -> Result<()> {
    let ident = m.md_str(&format!("xcl-dxil {}.{}", env!("CARGO_PKG_VERSION"), crate::VERSION));
    let ident = m.md_tuple(vec![Some(ident)]);
    m.named("llvm.ident", vec![ident]);

    let version = i32_tuple(m, &[DXIL_VERSION.0, DXIL_VERSION.1]);
    m.named("dx.version", vec![version]);
    let valver = i32_tuple(m, &[VALIDATOR_VERSION.0, VALIDATOR_VERSION.1]);
    m.named("dx.valver", vec![valver]);

    let ops = vec![Some(m.md_str(info.kind_name())), Some(m.md_i32(info.sm.0)), Some(m.md_i32(info.sm.1))];
    let model = m.md_tuple(ops);
    m.named("dx.shaderModel", vec![model]);

    let resources = resources(m, info)?;
    if let Some(r) = resources {
        m.named("dx.resources", vec![r]);
    }

    // DXC leaves a state of zeros out (and computes none for compute shaders)
    if info.kind != ShaderKind::Compute && info.view_id_state.iter().any(|&w| w != 0) {
        let words: Vec<Value> = info.view_id_state.iter().map(|&w| m.const_i32(w)).collect();
        let ty = m.array(words.len() as u64, Ty::I32);
        let array = m.aggregate(ty, words);
        let ops = vec![Some(m.md_val(array))];
        let state = m.md_tuple(ops);
        m.named("dx.viewIdState", vec![state]);
    }

    let entry = defined(m, &info.entry)?;
    let signatures = signatures(m, info);
    let properties = properties(m, info)?;
    let ops = vec![Some(m.md_val(entry)), Some(m.md_str(&info.entry)), signatures, resources, properties];
    let entry_point = m.md_tuple(ops);
    m.named("dx.entryPoints", vec![entry_point]);
    Ok(())
}

fn defined(m: &Module, name: &str) -> Result<Value> {
    m.funcs
        .iter()
        .position(|f| f.body.is_some() && f.name == name)
        .map(|i| Value::func(i as u32))
        .ok_or_else(|| Error::Internal(format!("the module defines no function {name}")))
}

fn i32_tuple(m: &mut Module, v: &[u32]) -> Md {
    let ops = v.iter().map(|&x| Some(m.md_i32(x))).collect();
    m.md_tuple(ops)
}

/// A name-value list, or null when it is empty.
fn extra(m: &mut Module, pairs: &[(u32, u32)]) -> Option<Md> {
    if pairs.is_empty() {
        return None;
    }
    let flat: Vec<u32> = pairs.iter().flat_map(|&(k, v)| [k, v]).collect();
    Some(i32_tuple(m, &flat))
}

/// The HLSL spelling of a component type inside a resource type name
/// (`Texture2D<vector<unsigned int, 2> >`).
fn comp_name(t: ComponentType) -> &'static str {
    match t {
        ComponentType::I1 => "bool",
        ComponentType::I16 => "min16int",
        ComponentType::U16 => "min16uint",
        ComponentType::I32 => "int",
        ComponentType::U32 => "unsigned int",
        ComponentType::I64 => "long long",
        ComponentType::U64 => "unsigned long long",
        ComponentType::F16 => "min16float",
        ComponentType::F64 | ComponentType::SNormF64 | ComponentType::UNormF64 => "double",
        _ => "float",
    }
}

fn comp_ty(t: ComponentType) -> Ty {
    match t {
        ComponentType::I1 => Ty::I1,
        ComponentType::I16 | ComponentType::U16 => Ty::I16,
        ComponentType::I32 | ComponentType::U32 => Ty::I32,
        ComponentType::I64 | ComponentType::U64 => Ty::I64,
        ComponentType::F16 | ComponentType::SNormF16 | ComponentType::UNormF16 => Ty::HALF,
        ComponentType::F64 | ComponentType::SNormF64 | ComponentType::UNormF64 => Ty::F64,
        _ => Ty::F32,
    }
}

/// The type of the resource variable DXC keeps in the record as `%T* undef`: the HLSL object
/// type as a named struct, in an array for a range. The validator reads from it whether the
/// resource is an array, that a constant buffer is a struct, and the size of a typed
/// resource's element (the first field).
///
/// The raw buffer and sampler types, the only bound resources XCL's translator declares, are
/// DXC's. For the others the names are DXC's as far as the record determines them, but the
/// module has no vector types, so an element of several components is an array here, and a
/// structured buffer's or constant buffer's layout is a run of 32-bit words of the right size.
fn symbol_ty(m: &mut Module, r: &Resource) -> Result<Ty> {
    let uav = r.class == ResourceClass::UAV;
    let rw = if uav { "RW" } else { "" };
    let word = |m: &mut Module, name: &str| m.named_struct(name, vec![Ty::I32]);
    let object = match r.kind {
        ResourceKind::Sampler => word(m, if r.sampler_kind == SamplerKind::Comparison { "struct.SamplerComparisonState" } else { "struct.SamplerState" }),
        ResourceKind::RawBuffer => word(m, &format!("struct.{rw}ByteAddressBuffer")),
        ResourceKind::CBuffer => {
            let words = m.array(r.cbuffer_bytes.div_ceil(4) as u64, Ty::I32);
            m.named_struct(&r.name, vec![words])
        }
        ResourceKind::StructuredBuffer => {
            let words = m.array(r.struct_stride.div_ceil(4) as u64, Ty::I32);
            let element = m.named_struct(&format!("struct.{}.element", r.name), vec![words]);
            m.named_struct(&format!("class.{rw}StructuredBuffer<{}.element>", r.name), vec![element])
        }
        ResourceKind::Texture1D
        | ResourceKind::Texture2D
        | ResourceKind::Texture2DMS
        | ResourceKind::Texture3D
        | ResourceKind::TextureCube
        | ResourceKind::Texture1DArray
        | ResourceKind::Texture2DArray
        | ResourceKind::Texture2DMSArray
        | ResourceKind::TextureCubeArray
        | ResourceKind::TypedBuffer => {
            let object = match r.kind {
                ResourceKind::Texture1D => "Texture1D",
                ResourceKind::Texture2D => "Texture2D",
                ResourceKind::Texture2DMS => "Texture2DMS",
                ResourceKind::Texture3D => "Texture3D",
                ResourceKind::TextureCube => "TextureCube",
                ResourceKind::Texture1DArray => "Texture1DArray",
                ResourceKind::Texture2DArray => "Texture2DArray",
                ResourceKind::Texture2DMSArray => "Texture2DMSArray",
                ResourceKind::TextureCubeArray => "TextureCubeArray",
                _ => "Buffer",
            };
            let n = r.comp_count.clamp(1, 4);
            let scalar = comp_ty(r.comp_type);
            let (element, mut args) = if n == 1 { (scalar, comp_name(r.comp_type).to_string()) } else { (m.array(n as u64, scalar), format!("vector<{}, {n}>", comp_name(r.comp_type))) };
            let ms = matches!(r.kind, ResourceKind::Texture2DMS | ResourceKind::Texture2DMSArray);
            if ms && r.sample_count != 0 {
                args.push_str(&format!(", {}", r.sample_count));
            }
            // clang's printer separates nested template brackets
            let name = format!("class.{rw}{object}<{args}{}>", if args.ends_with('>') { " " } else { "" });
            let mut fields = vec![element];
            // the mips / sample accessor member of the read-only texture objects
            let accessor = match r.kind {
                _ if uav => None,
                ResourceKind::Texture2DMS | ResourceKind::Texture2DMSArray => Some("sample_type"),
                ResourceKind::Texture1D | ResourceKind::Texture2D | ResourceKind::Texture3D | ResourceKind::Texture1DArray | ResourceKind::Texture2DArray => Some("mips_type"),
                _ => None,
            };
            if let Some(a) = accessor {
                fields.push(word(m, &format!("{name}::{a}")));
            }
            m.named_struct(&name, fields)
        }
        other => return crate::unsupported(format!("resource kind {other:?}")),
    };
    let symbol = match r.range_size {
        1 => object,
        u32::MAX => m.array(0, object),
        n => m.array(n as u64, object),
    };
    Ok(m.ptr(symbol))
}

/// dx.resources: `!{SRVs, UAVs, CBuffers, Samplers}`, each a list in id order or null.
fn resources(m: &mut Module, info: &ShaderInfo) -> Result<Option<Md>> {
    if info.resources.is_empty() {
        return Ok(None);
    }
    let mut lists: [Vec<Option<Md>>; 4] = Default::default();
    for class in [ResourceClass::SRV, ResourceClass::UAV, ResourceClass::CBuffer, ResourceClass::Sampler] {
        let mut of_class: Vec<&Resource> = info.resources.iter().filter(|r| r.class == class).collect();
        of_class.sort_by_key(|r| r.id);
        for (i, r) in of_class.into_iter().enumerate() {
            // the validator wants the ids of a class dense, and handles name them
            if r.id as usize != i {
                return Err(Error::Internal(format!("resource ids of {class:?} are not 0..n")));
            }
            let symbol = symbol_ty(m, r)?;
            let symbol = m.undef(symbol);
            // the name is reflection data, which dxc strips
            let mut ops = vec![Some(m.md_i32(r.id)), Some(m.md_val(symbol)), Some(m.md_str("")), Some(m.md_i32(r.space)), Some(m.md_i32(r.lower_bound)), Some(m.md_i32(r.range_size))];
            let typed = !matches!(r.kind, ResourceKind::StructuredBuffer | ResourceKind::RawBuffer);
            let mut pairs = Vec::new();
            match class {
                ResourceClass::SRV => {
                    ops.push(Some(m.md_i32(r.kind as u32)));
                    ops.push(Some(m.md_i32(r.sample_count)));
                    if typed {
                        pairs.push((tag::TYPED_BUFFER_ELEMENT_TYPE, r.comp_type as u32));
                    }
                    if r.kind == ResourceKind::StructuredBuffer {
                        pairs.push((tag::STRUCTURED_BUFFER_ELEMENT_STRIDE, r.struct_stride));
                    }
                }
                ResourceClass::UAV => {
                    ops.push(Some(m.md_i32(r.kind as u32)));
                    ops.push(Some(m.md_bool(r.globally_coherent)));
                    ops.push(Some(m.md_bool(r.has_counter)));
                    ops.push(Some(m.md_bool(r.rov)));
                    if typed && r.comp_type != ComponentType::Invalid {
                        pairs.push((tag::TYPED_BUFFER_ELEMENT_TYPE, r.comp_type as u32));
                    }
                    if r.kind == ResourceKind::StructuredBuffer {
                        pairs.push((tag::STRUCTURED_BUFFER_ELEMENT_STRIDE, r.struct_stride));
                    }
                    if r.atomic64_use {
                        pairs.push((tag::ATOMIC64_USE, 1));
                    }
                }
                ResourceClass::CBuffer => ops.push(Some(m.md_i32(r.cbuffer_bytes))),
                _ => ops.push(Some(m.md_i32(r.sampler_kind as u32))),
            }
            ops.push(extra(m, &pairs));
            let record = m.md_tuple(ops);
            lists[class as usize].push(Some(record));
        }
    }
    let ops = lists.into_iter().map(|l| if l.is_empty() { None } else { Some(m.md_tuple(l)) }).collect();
    Ok(Some(m.md_tuple(ops)))
}

/// `!{inputs, outputs, patch constant or primitive}`, null without any element.
fn signatures(m: &mut Module, info: &ShaderInfo) -> Option<Md> {
    if info.inputs.is_empty() && info.outputs.is_empty() && info.patch_or_prim.is_empty() {
        return None;
    }
    let mut ops = Vec::new();
    for sig in [&info.inputs, &info.outputs, &info.patch_or_prim] {
        let elements: Vec<Option<Md>> = sig.iter().enumerate().map(|(id, e)| Some(element(m, id as u32, e))).collect();
        ops.push(if elements.is_empty() { None } else { Some(m.md_tuple(elements)) });
    }
    Some(m.md_tuple(ops))
}

/// One signature element; its id is its position, which is what LoadInput / StoreOutput name.
fn element(m: &mut Module, id: u32, e: &SigElement) -> Md {
    let indices = i32_tuple(m, &e.indices);
    let mut pairs = Vec::new();
    if e.stream != 0 {
        pairs.push((tag::SIG_OUTPUT_STREAM, e.stream));
    }
    if e.dynamic_index_mask != 0 {
        pairs.push((tag::SIG_DYN_IDX_COMP_MASK, e.dynamic_index_mask as u32));
    }
    if e.usage_mask != 0 {
        pairs.push((tag::SIG_USAGE_COMP_MASK, e.usage_mask as u32));
    }
    let ops = vec![
        Some(m.md_i32(id)),
        Some(m.md_str(&e.name)),
        Some(m.md_i8(e.comp_type as u8)),
        Some(m.md_i8(e.kind as u8)),
        Some(indices),
        Some(m.md_i8(e.interp as u8)),
        Some(m.md_i32(e.rows)),
        Some(m.md_i8(e.cols as u8)),
        Some(m.md_i32(e.start_row as u32)),
        Some(m.md_i8(e.start_col as u8)),
        extra(m, &pairs),
    ];
    m.md_tuple(ops)
}

/// The entry point's tag-value list: the shader flags when not zero, then the stage's state.
fn properties(m: &mut Module, info: &ShaderInfo) -> Result<Option<Md>> {
    let mut ops: Vec<Option<Md>> = Vec::new();
    let mut flags = info.flags;
    if let StageProps::Pixel { early_depth_stencil: true } = info.props {
        flags |= flags::bit::FORCE_EARLY_DEPTH_STENCIL;
    }
    if flags != 0 {
        ops.push(Some(m.md_i32(tag::SHADER_FLAGS)));
        ops.push(Some(m.md_i64(flags)));
    }
    let mut state = |m: &mut Module, t: u32, node: Md| {
        ops.push(Some(m.md_i32(t)));
        ops.push(Some(node));
    };
    match &info.props {
        StageProps::Vertex | StageProps::Pixel { .. } => {}
        StageProps::Compute { numthreads, wave_size } => {
            let threads = i32_tuple(m, numthreads);
            state(m, tag::NUM_THREADS, threads);
            if let Some(w) = wave_size {
                let wave = i32_tuple(m, &[*w]);
                state(m, tag::WAVE_SIZE, wave);
            }
        }
        StageProps::Geometry { input_primitive, max_vertex_count, active_streams, stream_topology, instance_count } => {
            let gs = i32_tuple(m, &[*input_primitive as u32, *max_vertex_count, *active_streams, *stream_topology as u32, *instance_count]);
            state(m, tag::GS_STATE, gs);
        }
        StageProps::Domain { domain, input_control_points } => {
            let ds = i32_tuple(m, &[*domain as u32, *input_control_points]);
            state(m, tag::DS_STATE, ds);
        }
        StageProps::Hull { patch_constant_fn, input_control_points, output_control_points, domain, partitioning, output_primitive, max_tess_factor } => {
            let function = defined(m, patch_constant_fn)?;
            let mut hs = vec![Some(m.md_val(function))];
            for v in [*input_control_points, *output_control_points, *domain as u32, *partitioning as u32, *output_primitive as u32] {
                hs.push(Some(m.md_i32(v)));
            }
            hs.push(Some(m.md_f32(*max_tess_factor)));
            let hs = m.md_tuple(hs);
            state(m, tag::HS_STATE, hs);
        }
        StageProps::Mesh { numthreads, max_vertices, max_primitives, topology, payload_bytes } => {
            let mut ms = vec![Some(i32_tuple(m, numthreads))];
            for v in [*max_vertices, *max_primitives, *topology as u32, *payload_bytes] {
                ms.push(Some(m.md_i32(v)));
            }
            let ms = m.md_tuple(ms);
            state(m, tag::MS_STATE, ms);
        }
        StageProps::Amplification { numthreads, payload_bytes } => {
            let ops = vec![Some(i32_tuple(m, numthreads)), Some(m.md_i32(*payload_bytes))];
            let amp = m.md_tuple(ops);
            state(m, tag::AS_STATE, amp);
        }
    }
    Ok(if ops.is_empty() { None } else { Some(m.md_tuple(ops)) })
}

// ------------------------------------------------------------------------------------------
// ViewID state
// ------------------------------------------------------------------------------------------

const NONE: u32 = u32::MAX;

/// What a contributing instruction reads.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
enum SourceOp {
    ViewId,
    LoadInput,
    LoadOutputControlPoint,
    LoadPatchConstant,
}

/// A signature access: the element id, the row when it is a constant (None: any row of the
/// element) and the column.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct Access {
    id: u32,
    row: Option<u32>,
    col: u32,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct Source {
    op: SourceOp,
    at: Access,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum StoreOp {
    Output,
    VertexOutput,
    PrimitiveOutput,
    PatchConstant,
}

#[derive(Debug)]
struct Stored {
    op: StoreOp,
    at: Access,
    /// indices into [`Dependencies::sources`]
    sources: Vec<u32>,
}

/// The output stores of one function with the sources each depends on.
#[derive(Debug, Default)]
struct Dependencies {
    sources: Vec<Source>,
    stores: Vec<Stored>,
}

/// Immediate dominators of the nodes reachable from `root` (Cooper, Harvey and Kennedy's
/// iteration); `idom[root]` is the root itself and unreachable nodes have [`NONE`].
fn dominators(succs: &[Vec<u32>], preds: &[Vec<u32>], root: u32) -> Vec<u32> {
    let n = succs.len();
    let mut number = vec![NONE; n];
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    let mut stack: Vec<(u32, u32)> = vec![(root, 0)];
    seen[root as usize] = true;
    while let Some(top) = stack.last_mut() {
        let (b, k) = *top;
        if let Some(&s) = succs[b as usize].get(k as usize) {
            top.1 += 1;
            if !seen[s as usize] {
                seen[s as usize] = true;
                stack.push((s, 0));
            }
        } else {
            number[b as usize] = order.len() as u32;
            order.push(b);
            stack.pop();
        }
    }
    let mut idom = vec![NONE; n];
    idom[root as usize] = root;
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().rev() {
            if b == root {
                continue;
            }
            let mut new = NONE;
            for &p in &preds[b as usize] {
                if idom[p as usize] == NONE {
                    continue;
                }
                if new == NONE {
                    new = p;
                    continue;
                }
                let (mut x, mut y) = (p, new);
                while x != y {
                    while number[x as usize] < number[y as usize] {
                        x = idom[x as usize];
                    }
                    while number[y as usize] < number[x as usize] {
                        y = idom[y as usize];
                    }
                }
                new = x;
            }
            if new != NONE && idom[b as usize] != new {
                idom[b as usize] = new;
                changed = true;
            }
        }
    }
    idom
}

/// Per block, the blocks whose branch decides whether it runs (DXC's ControlDependence: the
/// post-dominance frontier, Allen and Kennedy figure 7.9), sorted.
fn control_dependence(succs: &[Vec<u32>], preds: &[Vec<u32>], reachable: &[bool]) -> Vec<Vec<u32>> {
    let n = succs.len();
    // the reverse graph, with a virtual exit that every block without successors leads to
    let exit = n as u32;
    let mut rsuccs: Vec<Vec<u32>> = preds.to_vec();
    let mut rpreds: Vec<Vec<u32>> = succs.to_vec();
    rsuccs.push(Vec::new());
    rpreds.push(Vec::new());
    for b in 0..n {
        if reachable[b] && succs[b].is_empty() {
            rsuccs[n].push(b as u32);
            rpreds[b].push(exit);
        }
    }
    let ipdom = dominators(&rsuccs, &rpreds, exit);
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); n + 1];
    for b in 0..n {
        if ipdom[b] != NONE {
            children[ipdom[b] as usize].push(b as u32);
        }
    }
    let mut cd: Vec<Vec<u32>> = vec![Vec::new(); n];
    // children before parents
    let mut stack: Vec<(u32, u32)> = vec![(exit, 0)];
    while let Some(top) = stack.last_mut() {
        let (x, k) = *top;
        if let Some(&z) = children[x as usize].get(k as usize) {
            top.1 += 1;
            stack.push((z, 0));
            continue;
        }
        stack.pop();
        if x == exit {
            continue;
        }
        let mut set: Vec<u32> = preds[x as usize].iter().copied().filter(|&y| ipdom[y as usize] != x).collect();
        for &z in &children[x as usize] {
            set.extend(cd[z as usize].iter().copied().filter(|&y| ipdom[y as usize] != x));
        }
        set.sort_unstable();
        set.dedup();
        cd[x as usize] = set;
    }
    cd
}

/// The incoming (value, predecessor) pairs of a phi instruction.
fn phi_in<'a>(body: &'a Body, phi: &Inst) -> &'a [(Value, Block)] {
    body.phis.get(phi.a.0 as usize).map_or(&[], |v| v.as_slice())
}

/// The instruction operands of an instruction, resolved (constants, globals and blocks are no
/// dependencies), as the writer emits them.
fn inst_operands(body: &Body, inst: &Inst, live_blocks: &[bool], mut f: impl FnMut(u32)) {
    let mut go = |v: Value| {
        if let ValueKind::Inst(i) = body.resolve(v).kind() {
            if !v.is_none() {
                f(i)
            }
        }
    };
    match inst.op {
        Op::Bin(_) | Op::ICmp(_) | Op::FCmp(_) | Op::Store | Op::AtomicRmw(_) => {
            go(inst.a);
            go(inst.b);
        }
        Op::Cast(_) | Op::Load | Op::ExtractValue | Op::CondBr | Op::Switch => go(inst.a),
        Op::Select | Op::CmpXchg => {
            go(inst.a);
            go(inst.b);
            go(inst.c);
        }
        Op::Call | Op::Gep => {
            go(inst.a);
            for &v in body.operands_of(inst) {
                go(v);
            }
        }
        Op::Phi => {
            for &(v, from) in phi_in(body, inst) {
                if live_blocks.get(from.0 as usize).copied().unwrap_or(false) {
                    go(v);
                }
            }
        }
        Op::Alloca | Op::Br | Op::Ret | Op::Unreachable => {}
    }
}

/// The variables (allocas, globals, and the payload a call returns) a pointer may address:
/// DXC's CollectReachingDecls, through GEPs, casts, phis and selects.
fn reaching_decls(body: &Body, live_blocks: &[bool], ptr: Value, out: &mut Vec<Value>, stack: &mut Vec<Value>, seen: &mut Vec<Value>) {
    out.clear();
    stack.clear();
    seen.clear();
    stack.push(body.resolve(ptr));
    while let Some(v) = stack.pop() {
        if seen.contains(&v) {
            continue;
        }
        seen.push(v);
        match v.kind() {
            ValueKind::Global(_) => out.push(v),
            ValueKind::Inst(i) => {
                let Some(inst) = body.insts.get(i as usize) else { continue };
                match inst.op {
                    Op::Gep | Op::Cast(_) => stack.push(body.resolve(inst.a)),
                    Op::Alloca | Op::Call => out.push(v),
                    Op::Phi => {
                        for &(x, from) in phi_in(body, inst) {
                            if live_blocks.get(from.0 as usize).copied().unwrap_or(false) {
                                stack.push(body.resolve(x));
                            }
                        }
                    }
                    Op::Select => {
                        stack.push(body.resolve(inst.b));
                        stack.push(body.resolve(inst.c));
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

/// DXC's DxilViewIdStateBuilder for one function: for every signature store, the signature
/// loads and SV_ViewID reads that reach its value through operands, through memory (a load
/// depends on every store to a variable its pointer may address) and through control (an
/// instruction depends on the branches its block is control dependent on; a constant that
/// flows into a phi depends on the branches that select it).
fn dependencies(m: &Module, body: &Body, live: &Live) -> Dependencies {
    let (ni, nb) = (body.insts.len(), body.blocks.len());
    let arg = |args: &[Value], k: usize| -> Option<u32> { m.as_int(body.resolve(*args.get(k)?)).and_then(|v| u32::try_from(v).ok()) };

    let mut out = Dependencies::default();
    let mut source_bit: HashMap<Source, u32> = HashMap::new();
    // (instruction, bit) and (instruction, index in out.stores)
    let mut source_insts: Vec<(u32, u32)> = Vec::new();
    let mut store_insts: Vec<u32> = Vec::new();
    let mut block_of = vec![NONE; ni];
    for (b, block) in body.blocks.iter().enumerate() {
        if !live.blocks[b] {
            continue;
        }
        for list in [&block.phis, &block.insts] {
            for &i in list {
                if !live.insts[i as usize] {
                    continue;
                }
                block_of[i as usize] = b as u32;
                let inst = &body.insts[i as usize];
                if inst.op != Op::Call || !crate::flags::calls_dx_op(m, inst) {
                    continue;
                }
                let args = body.operands_of(inst);
                let Some(op) = arg(args, 0) else { continue };
                // id and column are constants (DXC gives up on the module otherwise); the row may
                // be dynamic
                let access = || Some(Access { id: arg(args, 1)?, row: arg(args, 2), col: arg(args, 3)? });
                let source = match op {
                    x if x == OpCode::ViewID as u32 => Some(Source { op: SourceOp::ViewId, at: Access { id: 0, row: None, col: 0 } }),
                    x if x == OpCode::LoadInput as u32 => access().map(|at| Source { op: SourceOp::LoadInput, at }),
                    x if x == OpCode::LoadOutputControlPoint as u32 => access().map(|at| Source { op: SourceOp::LoadOutputControlPoint, at }),
                    x if x == OpCode::LoadPatchConstant as u32 => access().map(|at| Source { op: SourceOp::LoadPatchConstant, at }),
                    _ => None,
                };
                if let Some(s) = source {
                    let next = out.sources.len() as u32;
                    let bit = *source_bit.entry(s).or_insert(next);
                    if bit == next {
                        out.sources.push(s);
                    }
                    source_insts.push((i, bit));
                    continue;
                }
                let store = match op {
                    x if x == OpCode::StoreOutput as u32 => StoreOp::Output,
                    x if x == OpCode::StoreVertexOutput as u32 => StoreOp::VertexOutput,
                    x if x == OpCode::StorePrimitiveOutput as u32 => StoreOp::PrimitiveOutput,
                    x if x == OpCode::StorePatchConstant as u32 => StoreOp::PatchConstant,
                    _ => continue,
                };
                if let Some(at) = access() {
                    out.stores.push(Stored { op: store, at, sources: Vec::new() });
                    store_insts.push(i);
                }
            }
        }
    }
    if out.sources.is_empty() || out.stores.is_empty() {
        return out;
    }

    // the flow graph of the reachable blocks
    let mut succs: Vec<Vec<u32>> = vec![Vec::new(); nb];
    let mut preds: Vec<Vec<u32>> = vec![Vec::new(); nb];
    for (b, to) in succs.iter_mut().enumerate() {
        if !live.blocks[b] {
            continue;
        }
        let Some(t) = body.blocks[b].insts.last().and_then(|&i| body.insts.get(i as usize)) else { continue };
        successors(body, t, |s| {
            if !to.contains(&s) {
                to.push(s);
                preds[s as usize].push(b as u32);
            }
        });
    }
    let cd = control_dependence(&succs, &preds, &live.blocks);
    let tree = DomTree::new(dominators(&succs, &preds, 0), 0);

    // The dependence graph as (what is depended on, what depends on it) edges. Nodes: the
    // instructions, then per block "the control this block runs under", then per variable "the
    // stores to it".
    let ctrl = |b: u32| ni as u32 + b;
    let mut decl_node: HashMap<Value, u32> = HashMap::new();
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(ni * 2);
    let (mut decls, mut stack, mut seen) = (Vec::new(), Vec::new(), Vec::new());
    let mut phi = PhiScratch::default();
    for (b, block) in body.blocks.iter().enumerate() {
        if !live.blocks[b] {
            continue;
        }
        for &y in &cd[b] {
            if let Some(&t) = body.blocks[y as usize].insts.last() {
                edges.push((t, ctrl(b as u32)));
            }
        }
        for list in [&block.phis, &block.insts] {
            for &i in list {
                if !live.insts[i as usize] {
                    continue;
                }
                let inst = &body.insts[i as usize];
                inst_operands(body, inst, &live.blocks, |v| edges.push((v, i)));
                if !cd[b].is_empty() {
                    edges.push((ctrl(b as u32), i));
                }
                let reads = matches!(inst.op, Op::Load | Op::AtomicRmw(_) | Op::CmpXchg);
                let writes = matches!(inst.op, Op::Store | Op::AtomicRmw(_) | Op::CmpXchg);
                if reads || writes {
                    reaching_decls(body, &live.blocks, inst.a, &mut decls, &mut stack, &mut seen);
                    for &d in &decls {
                        let next = (ni + nb + decl_node.len()) as u32;
                        let node = *decl_node.entry(d).or_insert(next);
                        if reads {
                            edges.push((node, i));
                        }
                        if writes {
                            edges.push((i, node));
                        }
                    }
                }
                if inst.op == Op::Phi {
                    phi.incoming.clear();
                    for &(v, from) in phi_in(body, inst) {
                        if live.blocks.get(from.0 as usize).copied().unwrap_or(false) && tree.reachable(from.0) {
                            let v = body.resolve(v);
                            let def = match v.kind() {
                                ValueKind::Inst(d) => block_of.get(d as usize).copied().unwrap_or(0),
                                _ => 0,
                            };
                            phi.incoming.push((v, from.0, def));
                        }
                    }
                    if phi.incoming.iter().any(|&(v, ..)| !v.is_inst()) {
                        phi_definition_points(&tree, &mut phi, |at| {
                            if !cd[at as usize].is_empty() {
                                edges.push((ctrl(at), i));
                            }
                        });
                    }
                }
            }
        }
    }

    // users of each node (CSR)
    let nodes = ni + nb + decl_node.len();
    let mut first = vec![0u32; nodes + 1];
    for &(dep, _) in &edges {
        first[dep as usize + 1] += 1;
    }
    for k in 0..nodes {
        first[k + 1] += first[k];
    }
    let mut users = vec![0u32; edges.len()];
    let mut fill = first.clone();
    for &(dep, user) in &edges {
        users[fill[dep as usize] as usize] = user;
        fill[dep as usize] += 1;
    }
    drop(edges);

    // which sources reach each node: propagate until nothing changes
    let w = out.sources.len().div_ceil(64);
    let mut bits = vec![0u64; nodes * w];
    let mut queued = vec![false; nodes];
    let mut work: Vec<u32> = Vec::new();
    for &(i, bit) in &source_insts {
        bits[i as usize * w + bit as usize / 64] |= 1 << (bit % 64);
        if !queued[i as usize] {
            queued[i as usize] = true;
            work.push(i);
        }
    }
    let mut from = vec![0u64; w];
    while let Some(n) = work.pop() {
        queued[n as usize] = false;
        from.copy_from_slice(&bits[n as usize * w..(n as usize + 1) * w]);
        for &u in &users[first[n as usize] as usize..first[n as usize + 1] as usize] {
            let to = &mut bits[u as usize * w..(u as usize + 1) * w];
            let mut grew = false;
            for (t, f) in to.iter_mut().zip(&from) {
                grew |= *f & !*t != 0;
                *t |= *f;
            }
            if grew && !queued[u as usize] {
                queued[u as usize] = true;
                work.push(u);
            }
        }
    }

    // a store depends on its value and on the control of its block (not on its other operands)
    for (stored, &i) in out.stores.iter_mut().zip(&store_insts) {
        let inst = &body.insts[i as usize];
        from.iter_mut().for_each(|x| *x = 0);
        let mut add = |node: usize| from.iter_mut().zip(&bits[node * w..(node + 1) * w]).for_each(|(a, b)| *a |= *b);
        if let Some(ValueKind::Inst(v)) = body.operands_of(inst).get(4).map(|&v| body.resolve(v).kind()) {
            add(v as usize);
        }
        add(ctrl(block_of[i as usize]) as usize);
        stored.sources = (0..out.sources.len() as u32).filter(|&s| from[s as usize / 64] >> (s % 64) & 1 != 0).collect();
    }
    out
}

/// The dominator tree with the queries [`phi_definition_points`] makes of it, each in
/// O(log depth): every node keeps one jump pointer (Myers' skew-binary scheme), so the tree
/// costs a few words per block however deep it is.
struct DomTree {
    idom: Vec<u32>,
    depth: Vec<u32>,
    jump: Vec<u32>,
    /// position in a preorder walk, and one past the last position in the node's subtree
    tin: Vec<u32>,
    end: Vec<u32>,
}

impl DomTree {
    /// From [`dominators`]' result: `idom[root] == root`, [`NONE`] for unreachable nodes.
    fn new(idom: Vec<u32>, root: u32) -> DomTree {
        let n = idom.len();
        let mut first = vec![0u32; n + 1];
        let child = |b: usize, p: u32| p != NONE && p as usize != b;
        for (b, &p) in idom.iter().enumerate() {
            if child(b, p) {
                first[p as usize + 1] += 1;
            }
        }
        for k in 0..n {
            first[k + 1] += first[k];
        }
        let mut kids = vec![0u32; first[n] as usize];
        let mut fill = first.clone();
        for (b, &p) in idom.iter().enumerate() {
            if child(b, p) {
                kids[fill[p as usize] as usize] = b as u32;
                fill[p as usize] += 1;
            }
        }
        let (mut depth, mut jump, mut tin, mut end) = (vec![0u32; n], vec![root; n], vec![NONE; n], vec![0u32; n]);
        let mut order: Vec<u32> = Vec::with_capacity(n);
        let mut stack = vec![root];
        while let Some(b) = stack.pop() {
            tin[b as usize] = order.len() as u32;
            order.push(b);
            if b != root {
                let p = idom[b as usize];
                let j = jump[p as usize];
                depth[b as usize] = depth[p as usize] + 1;
                let even = depth[p as usize] - depth[j as usize] == depth[j as usize] - depth[jump[j as usize] as usize];
                jump[b as usize] = if even { jump[j as usize] } else { p };
            }
            stack.extend_from_slice(&kids[first[b as usize] as usize..first[b as usize + 1] as usize]);
        }
        for &b in order.iter().rev() {
            end[b as usize] += tin[b as usize] + 1;
            if b != root {
                end[idom[b as usize] as usize] += end[b as usize] - tin[b as usize];
            }
        }
        DomTree { idom, depth, jump, tin, end }
    }

    fn reachable(&self, b: u32) -> bool {
        self.idom.get(b as usize).is_some_and(|&p| p != NONE)
    }

    /// Whether `a` dominates `b` (a node dominates itself).
    fn dominates(&self, a: u32, b: u32) -> bool {
        self.tin[a as usize] <= self.tin[b as usize] && self.tin[b as usize] < self.end[a as usize]
    }

    /// The dominator of `b` at `depth` (at most `b`'s own).
    fn ancestor_at(&self, mut b: u32, depth: u32) -> u32 {
        while self.depth[b as usize] > depth {
            let j = self.jump[b as usize];
            b = if self.depth[j as usize] >= depth { j } else { self.idom[b as usize] };
        }
        b
    }

    /// The lowest node that dominates both.
    fn common(&self, mut a: u32, mut b: u32) -> u32 {
        if self.depth[a as usize] < self.depth[b as usize] {
            std::mem::swap(&mut a, &mut b);
        }
        a = self.ancestor_at(a, self.depth[b as usize]);
        while a != b {
            let same = self.jump[a as usize] == self.jump[b as usize];
            a = if same { self.idom[a as usize] } else { self.jump[a as usize] };
            b = if same { self.idom[b as usize] } else { self.jump[b as usize] };
        }
        a
    }
}

/// What DXC's phi walk leaves on a dominator tree node: nothing, the one value that may be
/// "defined" there, or the note that no single value can be.
#[derive(Clone, Copy, PartialEq)]
enum Mark {
    Unset,
    Is(Value),
    Illegal,
}

/// Buffers of [`phi_definition_points`], kept between phis.
#[derive(Default)]
struct PhiScratch {
    /// the phi's incoming edges on reachable blocks: (value, source block, the block that
    /// defines the value; the entry for a constant)
    incoming: Vec<(Value, u32, u32)>,
    /// the blocks that matter to the phi, in dominator tree preorder, with the nearest such
    /// block above each
    keys: Vec<u32>,
    parent: Vec<u32>,
    /// the mark on each of those blocks, and on the run of blocks between it and its parent
    node: Vec<Mark>,
    above: Vec<Mark>,
    top: Vec<u32>,
    stack: Vec<u32>,
}

/// DXC's CollectPhiCFValuesContributingToOutputRec: a constant flowing into a phi along an
/// edge has no instruction whose block says under which control it is chosen, so DXC looks for
/// the highest dominator of the edge's source from which only that value can reach the phi,
/// and takes that block's control dependence. Calls `f` with that block for every constant of
/// `s.incoming`.
///
/// DXC marks every incoming block with its value, then walks each value from its block up the
/// dominator tree to its definition, marking what is unmarked and stopping at the first mark
/// (which becomes illegal when the values differ and are not both constants), and last climbs
/// from each constant's block while the marks are that constant. Those walks are as long as
/// the tree is deep, for every phi. The same marks are made here on the tree reduced to the
/// blocks where anything can change: the incoming blocks, the definitions, and the points
/// where two of their paths to the root meet. A walk that passes one of these marks the whole
/// run of blocks up to the next, so one mark stands for the run.
fn phi_definition_points(tree: &DomTree, s: &mut PhiScratch, mut f: impl FnMut(u32)) {
    let by_tin = |keys: &mut Vec<u32>| {
        keys.sort_unstable_by_key(|&b| tree.tin[b as usize]);
        keys.dedup();
    };
    s.keys.clear();
    s.keys.push(0);
    for &(_, from, def) in &s.incoming {
        s.keys.push(from);
        if tree.reachable(def) {
            s.keys.push(def);
        }
    }
    by_tin(&mut s.keys);
    for k in 1..s.keys.len() {
        s.keys.push(tree.common(s.keys[k - 1], s.keys[k]));
    }
    by_tin(&mut s.keys);
    s.parent.clear();
    s.stack.clear();
    for (k, &b) in s.keys.iter().enumerate() {
        while s.stack.last().is_some_and(|&t| !tree.dominates(s.keys[t as usize], b)) {
            s.stack.pop();
        }
        s.parent.push(s.stack.last().copied().unwrap_or(NONE));
        s.stack.push(k as u32);
    }
    let n = s.keys.len();
    for marks in [&mut s.node, &mut s.above] {
        marks.clear();
        marks.resize(n, Mark::Unset);
    }
    s.top.clear();
    s.top.resize(n, NONE);
    let key = |keys: &[u32], b: u32| keys.partition_point(|&k| tree.tin[k as usize] < tree.tin[b as usize]);
    // whether blocks lie between a key and its parent
    let run = |keys: &[u32], k: usize, p: usize| tree.depth[keys[k] as usize] - tree.depth[keys[p] as usize] > 1;

    for &(v, from, _) in &s.incoming {
        let k = key(&s.keys, from);
        if s.node[k] == Mark::Unset {
            s.node[k] = Mark::Is(v);
        }
    }
    for &(v, from, def) in &s.incoming {
        if from == def {
            continue;
        }
        let mut k = key(&s.keys, from);
        while s.parent[k] != NONE {
            let p = s.parent[k] as usize;
            if run(&s.keys, k, p) {
                // marked already only by an earlier edge from the same block
                if s.above[k] != Mark::Unset {
                    break;
                }
                s.above[k] = Mark::Is(v);
            }
            match s.node[p] {
                Mark::Unset => s.node[p] = Mark::Is(v),
                Mark::Is(old) => {
                    if old != v && (old.is_inst() || v.is_inst()) {
                        s.node[p] = Mark::Illegal;
                    }
                    break;
                }
                Mark::Illegal => break,
            }
            if s.keys[p] == def {
                break;
            }
            k = p;
        }
    }
    for &(v, from, _) in &s.incoming {
        if v.is_inst() {
            continue;
        }
        let (mut k, mut point) = (key(&s.keys, from), from);
        s.stack.clear();
        while s.node[k] == Mark::Is(v) {
            if s.top[k] != NONE {
                point = s.top[k];
                break;
            }
            s.stack.push(k as u32);
            point = s.keys[k];
            if s.parent[k] == NONE {
                break;
            }
            let p = s.parent[k] as usize;
            if run(&s.keys, k, p) {
                if s.above[k] != Mark::Is(v) {
                    break;
                }
                point = tree.ancestor_at(s.keys[k], tree.depth[s.keys[p] as usize] + 1);
            }
            k = p;
        }
        for &k in &s.stack {
            s.top[k as usize] = point;
        }
        f(point);
    }
}

/// Scalars a signature takes in a stream: one past the highest packed location
/// (DetermineMaxPackedLocation).
fn scalars(sig: &[SigElement], stream: u32) -> u32 {
    sig.iter().filter(|e| e.start_row >= 0 && e.stream == stream && e.rows > 0 && e.cols > 0).map(|e| linear(e, e.rows - 1, e.cols - 1) + 1).max().unwrap_or(0)
}

/// The packed location of a component of an allocated element.
fn linear(e: &SigElement, row: u32, col: u32) -> u32 {
    (row + e.start_row as u32) * 4 + col + e.start_col.max(0) as u32
}

/// The locations an access covers: its row, or every row of the element when the row is
/// dynamic; nothing for an element without a register.
fn locations(sig: &[SigElement], at: Access) -> impl Iterator<Item = u32> + '_ {
    let e = sig.get(at.id as usize).filter(|e| e.start_row >= 0);
    let rows = match (e, at.row) {
        (None, _) => 0..0,
        (Some(_), Some(r)) => r..r + 1,
        (Some(e), None) => 0..e.rows,
    };
    rows.filter_map(move |row| e.map(|e| linear(e, row, at.col)))
}

/// Per output scalar: whether it depends on SV_ViewID, and the input scalars it depends on.
#[derive(Clone, Default)]
struct Sets {
    view_id: Vec<bool>,
    inputs: Vec<BTreeSet<u32>>,
}

impl Sets {
    fn new(n: u32) -> Sets {
        Sets { view_id: vec![false; n as usize], inputs: vec![BTreeSet::new(); n as usize] }
    }
    /// `words` of 32 output bits each: the outputs that depend on SV_ViewID.
    fn view_id_words(&self, out: &mut Vec<u32>) {
        let at = out.len();
        out.resize(at + self.view_id.len().div_ceil(32), 0);
        for (i, _) in self.view_id.iter().enumerate().filter(|(_, &v)| v) {
            out[at + i / 32] |= 1 << (i % 32);
        }
    }
    /// Per input scalar, a mask of the outputs it contributes to.
    fn input_table(&self, inputs: u32, out: &mut Vec<u32>) {
        let words = self.inputs.len().div_ceil(32);
        let at = out.len();
        out.resize(at + inputs as usize * words, 0);
        for (o, set) in self.inputs.iter().enumerate() {
            for &i in set.iter().filter(|&&i| i < inputs) {
                out[at + i as usize * words + o / 32] |= 1 << (o % 32);
            }
        }
    }
}

/// DXC's serialized DxilViewIdState for the module ([`ShaderInfo::view_id_state`]); empty for
/// a compute shader. Needs the signatures of `info` and the finished code.
pub fn view_id_state(m: &Module, info: &ShaderInfo) -> Vec<u32> {
    view_id_state_live(m, info, &[])
}

/// `lives`: the liveness of the bodies by function index; computed here for a body without one.
fn view_id_state_live(m: &Module, info: &ShaderInfo, lives: &[Option<Live>]) -> Vec<u32> {
    let kind = info.kind;
    if kind == ShaderKind::Compute {
        return Vec::new();
    }
    let of = |name: &str| -> Dependencies {
        let Some(i) = m.funcs.iter().position(|f| f.name == name) else { return Dependencies::default() };
        let Some(body) = &m.funcs[i].body else { return Dependencies::default() };
        match lives.get(i).and_then(|l| l.as_ref()) {
            Some(live) => dependencies(m, body, live),
            None => dependencies(m, body, &liveness(body)),
        }
    };
    let entry = of(&info.entry);
    let patch = match &info.props {
        StageProps::Hull { patch_constant_fn, .. } => of(patch_constant_fn),
        _ => Dependencies::default(),
    };
    let uses_view_id = entry.sources.iter().chain(&patch.sources).any(|s| s.op == SourceOp::ViewId);

    let streams = if kind == ShaderKind::Geometry { 4 } else { 1 };
    let inputs = scalars(&info.inputs, 0);
    let patch_scalars = scalars(&info.patch_or_prim, 0);
    let mut main: Vec<Sets> = (0..streams).map(|s| Sets::new(scalars(&info.outputs, s))).collect();
    // domain shader: patch constant inputs contributing to each output
    let mut from_patch = Sets::new(scalars(&info.outputs, 0));
    for stored in entry.stores.iter().filter(|s| matches!(s.op, StoreOp::Output | StoreOp::VertexOutput)) {
        let Some(e) = info.outputs.get(stored.at.id as usize) else { continue };
        let Some(sets) = main.get_mut(e.stream as usize) else { continue };
        for o in locations(&info.outputs, stored.at) {
            let o = o as usize;
            if o >= sets.view_id.len() {
                continue;
            }
            for &s in &stored.sources {
                let source = entry.sources[s as usize];
                match source.op {
                    SourceOp::ViewId => sets.view_id[o] = true,
                    // a domain shader reads its control points with either operation
                    SourceOp::LoadInput => sets.inputs[o].extend(locations(&info.inputs, source.at)),
                    SourceOp::LoadOutputControlPoint if kind == ShaderKind::Domain => sets.inputs[o].extend(locations(&info.inputs, source.at)),
                    SourceOp::LoadPatchConstant if kind == ShaderKind::Domain && e.stream == 0 => from_patch.inputs[o].extend(locations(&info.patch_or_prim, source.at)),
                    _ => {}
                }
            }
        }
    }
    // hull shader: the patch constant function; mesh shader: the per-primitive outputs
    let mut second = Sets::new(patch_scalars);
    let (deps, store_op) = if kind == ShaderKind::Hull { (&patch, StoreOp::PatchConstant) } else { (&entry, StoreOp::PrimitiveOutput) };
    if matches!(kind, ShaderKind::Hull | ShaderKind::Mesh) {
        for stored in deps.stores.iter().filter(|s| s.op == store_op) {
            for o in locations(&info.patch_or_prim, stored.at) {
                let o = o as usize;
                if o >= second.view_id.len() {
                    continue;
                }
                for &s in &stored.sources {
                    let source = deps.sources[s as usize];
                    match source.op {
                        SourceOp::ViewId => second.view_id[o] = true,
                        SourceOp::LoadInput => second.inputs[o].extend(locations(&info.inputs, source.at)),
                        // what the control point function wrote there depends on
                        SourceOp::LoadOutputControlPoint if kind == ShaderKind::Hull => {
                            for cp in locations(&info.outputs, source.at) {
                                if let (Some(&v), Some(set)) = (main[0].view_id.get(cp as usize), main[0].inputs.get(cp as usize)) {
                                    second.view_id[o] |= v;
                                    second.inputs[o].extend(set.iter().copied());
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let mut out = vec![inputs];
    for sets in &main {
        out.push(sets.view_id.len() as u32);
        if uses_view_id {
            sets.view_id_words(&mut out);
        }
        sets.input_table(inputs, &mut out);
    }
    match kind {
        ShaderKind::Hull | ShaderKind::Mesh => {
            out.push(patch_scalars);
            if uses_view_id {
                second.view_id_words(&mut out);
            }
            second.input_table(inputs, &mut out);
        }
        ShaderKind::Domain => {
            out.push(patch_scalars);
            from_patch.input_table(patch_scalars, &mut out);
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flags::testing::*;
    use crate::irprint;

    fn element(name: &str, kind: SemanticKind, cols: u32, start_row: i32) -> SigElement {
        SigElement {
            name: name.into(),
            indices: vec![0],
            kind,
            comp_type: ComponentType::F32,
            interp: InterpolationMode::Linear,
            rows: 1,
            cols,
            start_row,
            start_col: 0,
            stream: 0,
            usage_mask: (1 << cols) - 1,
            dynamic_index_mask: 0,
        }
    }

    fn metadata_text(m: &Module) -> String {
        let text = irprint::module(m);
        text[text.find("!llvm.ident").expect("metadata")..].trim_end().to_string()
    }

    /// The translator's compute shape (cs_basic of tests/meta_oracle.rs); the text is DXC's but
    /// for the producer.
    #[test]
    fn compute_shader_metadata() {
        let mut b = Builder::new();
        let bb = Body::ENTRY;
        let io = b.bound(bb, ResourceClass::UAV, 0, 0, 0, RAW_UAV);
        let zero = b.i32(0);
        let lane = b.op(bb, OpCode::WaveGetLaneIndex, Ty::VOID, &[]);
        let wide = b.body.cast(bb, CastOp::ZExt, Ty::I64, lane);
        let low = b.body.cast(bb, CastOp::Trunc, Ty::I32, wide);
        b.raw_store(bb, io, zero, low);
        let mut m = b.finish(bb, "main");
        let mut info = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [32, 1, 1], wave_size: Some(32) });
        for u in 0..2 {
            info.resources.push(Resource::new(ResourceClass::UAV, u, "io", ResourceKind::RawBuffer, u, 0));
        }
        prepare(&m, &mut info);
        assert!(info.view_id_state.is_empty());
        emit(&mut m, &info).unwrap();
        let expected = format!(
            r#"!llvm.ident = !{{!0}}
!dx.version = !{{!1}}
!dx.valver = !{{!2}}
!dx.shaderModel = !{{!3}}
!dx.resources = !{{!4}}
!dx.entryPoints = !{{!8}}

!0 = !{{!"xcl-dxil {}.{}"}}
!1 = !{{i32 1, i32 6}}
!2 = !{{i32 1, i32 8}}
!3 = !{{!"cs", i32 6, i32 6}}
!4 = !{{null, !5, null, null}}
!5 = !{{!6, !7}}
!6 = !{{i32 0, %struct.RWByteAddressBuffer* undef, !"", i32 0, i32 0, i32 1, i32 11, i1 false, i1 false, i1 false, null}}
!7 = !{{i32 1, %struct.RWByteAddressBuffer* undef, !"", i32 0, i32 1, i32 1, i32 11, i1 false, i1 false, i1 false, null}}
!8 = !{{void ()* @main, !"main", null, !4, !9}}
!9 = !{{i32 0, i64 1572880, i32 4, !10, i32 11, !11}}
!10 = !{{i32 32, i32 1, i32 1}}
!11 = !{{i32 32}}"#,
            env!("CARGO_PKG_VERSION"),
            crate::VERSION
        );
        let text = metadata_text(&m);
        // the writer's !dx.precise node may follow
        assert!(text.starts_with(&expected), "{text}");
    }

    #[test]
    fn what_cannot_be_emitted_is_an_error() {
        let info = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [1, 1, 1], wave_size: None });
        assert!(matches!(emit(&mut Module::new(), &info), Err(Error::Internal(_))));

        let module = || Builder::new().finish(Body::ENTRY, "main");
        let mut sparse = info.clone();
        sparse.resources.push(Resource::new(ResourceClass::UAV, 1, "", ResourceKind::RawBuffer, 0, 0));
        assert!(matches!(emit(&mut module(), &sparse), Err(Error::Internal(_))));

        let mut feedback = info.clone();
        feedback.resources.push(Resource::new(ResourceClass::UAV, 0, "", ResourceKind::FeedbackTexture2D, 0, 0));
        assert!(matches!(emit(&mut module(), &feedback), Err(Error::Unsupported(_))));

        let hull = ShaderInfo::new(
            ShaderKind::Hull,
            "main",
            StageProps::Hull {
                patch_constant_fn: "pcf".into(),
                input_control_points: 3,
                output_control_points: 3,
                domain: TessellatorDomain::Tri,
                partitioning: TessellatorPartitioning::Integer,
                output_primitive: TessellatorOutputPrimitive::TriangleCW,
                max_tess_factor: 64.0,
            },
        );
        assert!(matches!(emit(&mut module(), &hull), Err(Error::Internal(_))));
    }

    /// Inputs A (row 0, 4 columns), B (row 1, 2 columns), C (row 2, 1 column); outputs P (row
    /// 0, 4 columns), Q (row 1, 1 column).
    fn vertex() -> ShaderInfo {
        let mut info = ShaderInfo::new(ShaderKind::Vertex, "main", StageProps::Vertex);
        info.inputs = vec![element("A", SemanticKind::Arbitrary, 4, 0), element("B", SemanticKind::Arbitrary, 2, 1), element("C", SemanticKind::Arbitrary, 1, 2)];
        info.outputs = vec![element("SV_Position", SemanticKind::Position, 4, 0), element("Q", SemanticKind::Arbitrary, 1, 1)];
        info
    }

    /// Data, control and phi dependence:
    ///
    /// ```text
    /// P.x = A.x;  if (C.x > 0) { P.y = B.y; }  Q.x = C.x > 0 ? B.y : 0;
    /// ```
    ///
    /// P.y depends on the branch (C.x). Q.x is a phi of B.y, read before the branch, and a
    /// constant: no instruction of the phi's operands sits under the branch, and the constant
    /// can only be "defined" on its own edge, whose block the branch controls, so Q.x depends
    /// on C.x through the constant.
    #[test]
    fn view_id_state_of_data_and_control_dependence() {
        let mut b = Builder::new();
        let (entry, then, otherwise, merge) = (Body::ENTRY, b.body.new_block(), b.body.new_block(), b.body.new_block());
        let zero = b.i32(0);
        let a = b.load_input(entry, 0, zero, 0);
        let c = b.load_input(entry, 2, zero, 0);
        b.store_output(entry, 0, zero, 0, a);
        let fzero = b.m.const_f32(0.0);
        let positive = b.body.fcmp(entry, FPred::Ogt, c, fzero);
        let by = b.load_input(entry, 1, zero, 1);
        b.body.cond_br(entry, positive, then, otherwise);
        b.store_output(then, 0, zero, 1, by);
        b.body.br(then, merge);
        b.body.br(otherwise, merge);
        let phi = b.body.phi(merge, Ty::F32);
        b.body.phi_add(phi, by, then);
        b.body.phi_add(phi, fzero, otherwise);
        b.store_output(merge, 1, zero, 0, phi);
        let m = b.finish(merge, "main");
        // 9 input scalars, 5 output scalars, then per input the outputs it reaches
        assert_eq!(view_id_state(&m, &vertex()), [9, 5, 1 << 0, 0, 0, 0, 0, 1 << 1 | 1 << 4, 0, 0, 1 << 1 | 1 << 4]);
    }

    /// Memory dependence, a dynamic input row and SV_ViewID:
    ///
    /// ```text
    /// float t = 0;  if (C.x > 0) t = A.x;  P.x = t;  Q.x = B[ViewID].x-ish (a dynamic row of A)
    /// ```
    #[test]
    fn view_id_state_of_memory_and_view_id() {
        let mut b = Builder::new();
        let (entry, then, merge) = (Body::ENTRY, b.body.new_block(), b.body.new_block());
        let zero = b.i32(0);
        let slot_ty = b.m.ptr(Ty::F32);
        let slot = b.body.alloca(slot_ty);
        let fzero = b.m.const_f32(0.0);
        b.body.store(entry, slot, fzero);
        let c = b.load_input(entry, 2, zero, 0);
        let positive = b.body.fcmp(entry, FPred::Ogt, c, fzero);
        b.body.cond_br(entry, positive, then, merge);
        let a = b.load_input(then, 0, zero, 0);
        b.body.store(then, slot, a);
        b.body.br(then, merge);
        let t = b.body.load(merge, Ty::F32, slot);
        b.store_output(merge, 0, zero, 0, t);
        let view = b.op(merge, OpCode::ViewID, Ty::I32, &[]);
        let dynamic = b.load_input(merge, 1, view, 0);
        b.store_output(merge, 1, zero, 0, dynamic);
        let m = b.finish(merge, "main");
        let mut info = vertex();
        info.flags = crate::flags::collect(&m, &info);
        assert_eq!(info.flags, crate::flags::bit::VIEW_ID);
        // P.x: A.x through the store, C.x through the store's branch. Q.x: B.x and SV_ViewID.
        assert_eq!(view_id_state(&m, &info), [9, 5, 1 << 4, 1 << 0, 0, 0, 0, 1 << 4, 0, 0, 0, 1 << 0]);
    }

    #[test]
    fn dominators_and_control_dependence() {
        // 0 -> 1, 2; 1 -> 3; 2 -> 3; 3 -> 4, 1 (a loop back to 1); 4 exits
        let succs: Vec<Vec<u32>> = vec![vec![1, 2], vec![3], vec![3], vec![4, 1], vec![]];
        let mut preds: Vec<Vec<u32>> = vec![Vec::new(); 5];
        for (b, s) in succs.iter().enumerate() {
            for &t in s {
                preds[t as usize].push(b as u32);
            }
        }
        assert_eq!(dominators(&succs, &preds, 0), [0, 0, 0, 0, 3]);
        let cd = control_dependence(&succs, &preds, &[true; 5]);
        // 1 runs when 0 branches to it or 3 loops; 2 when 0 branches to it; 3 again when 3 loops
        assert_eq!(cd, [vec![], vec![0, 3], vec![0], vec![3], vec![]]);
    }

    /// DXC's phi walk as it is written, block by block: what [`phi_definition_points`] has to
    /// reproduce on its reduced tree.
    fn definition_points_by_walking(incoming: &[(Value, u32, u32)], idom: &[u32]) -> Vec<u32> {
        let mut marks: HashMap<u32, Option<Value>> = HashMap::new();
        for &(v, from, _) in incoming {
            marks.entry(from).or_insert(Some(v));
        }
        for &(v, from, def) in incoming {
            let mut b = from;
            while b != def && idom[b as usize] != b {
                b = idom[b as usize];
                if let Some(old) = marks.get_mut(&b) {
                    if old.is_some_and(|o| o != v && (o.is_inst() || v.is_inst())) {
                        *old = None;
                    }
                    break;
                }
                marks.insert(b, Some(v));
            }
        }
        let mut points = Vec::new();
        for &(v, from, _) in incoming {
            if v.is_inst() {
                continue;
            }
            let (mut b, mut point) = (from, from);
            while marks.get(&b) == Some(&Some(v)) {
                point = b;
                if idom[b as usize] == b {
                    break;
                }
                b = idom[b as usize];
            }
            points.push(point);
        }
        points
    }

    #[test]
    fn phi_definition_points_match_dxc_walk() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut rng = |below: u32| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as u32 % below
        };
        let mut scratch = PhiScratch::default();
        for trial in 0..40_000 {
            // trees with long runs (most blocks hang under the one before) and some branching
            let n = 1 + rng(48);
            let chain = 1 + rng(6);
            let idom: Vec<u32> = (0..n).map(|b| if b == 0 || rng(chain) != 0 { b.saturating_sub(1) } else { rng(b) }).collect();
            let tree = DomTree::new(idom.clone(), 0);
            for b in 1..n {
                assert_eq!(tree.ancestor_at(b, tree.depth[b as usize] - 1), idom[b as usize]);
            }
            let defs: Vec<u32> = (0..4).map(|_| rng(n)).collect();
            scratch.incoming.clear();
            for _ in 0..1 + rng(7) {
                let from = rng(n);
                if let Some(&twin) = scratch.incoming.iter().find(|e| e.1 == from) {
                    scratch.incoming.push(twin);
                    continue;
                }
                let k = rng(4);
                // now and then a definition that does not dominate the edge: no valid module
                // has one, and the walk then runs to the root
                let edge = if rng(2) == 0 && (tree.dominates(defs[k as usize], from) || rng(8) == 0) {
                    (Value::inst(k), from, defs[k as usize])
                } else {
                    (Value::constant(rng(3)), from, 0)
                };
                scratch.incoming.push(edge);
            }
            let expected = definition_points_by_walking(&scratch.incoming, &idom);
            let mut points = Vec::new();
            phi_definition_points(&tree, &mut scratch, |b| points.push(b));
            assert_eq!(points, expected, "trial {trial}: idom {idom:?}, incoming {:?}", scratch.incoming);
        }
    }
}
