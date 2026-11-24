//! Entry-point inputs and outputs of the graphics stages: the signatures of an entry function
//! as DXC builds them (lib/HLSL/HLSignatureLower.cpp, packed by DxilSignatureAllocator), the
//! operations that read and write them, and the stage properties. Owner: the stage subagent.
//!
//! [`super::lower`] stays generic: it asks [`begin`] for a [`StageIo`], then drives it. Values
//! cross the seam as **flattened components**: the scalar components of a value of the
//! parameter's (or its element's) type in the order of [`super::types::TypeTable::comps`]
//! (struct fields in declaration order, array elements in order, vector components in order),
//! each of the IR type its scalar kind has in SSA form (`bool` is i1: the conversions to and
//! from the signature's i32 happen here).
//!
//! What the lowering calls, and when (also in `scratch/stage/SEAM.md`):
//!
//! * [`begin`] before lowering the entry function's body;
//! * for every parameter, by its [`ParamKind`]: [`StageIo::input`] in the entry block for the
//!   initial value of an `Input` / `InOut` parameter; nothing for `Output` (its initial value is
//!   undefined); the object kinds are not values, their uses go through the hooks:
//!   [`StageIo::input_at`] for `v[i]` of an input array, [`StageIo::append`] /
//!   [`StageIo::restart_strip`] for the methods of a stream, [`StageIo::mesh_store`] for an
//!   assignment to (part of) an element of a mesh output array;
//! * [`StageIo::set_mesh_output_counts`] and [`StageIo::eval_attribute`] for the intrinsics of
//!   those names, [`discard`] for the statement;
//! * [`StageIo::output`] once per `Output` / `InOut` parameter and for the return value, in the
//!   function's single exit block before its `ret` (the lowering merges every `return` there,
//!   which is also where DXC stores the outputs);
//! * [`StageIo::finish`] with the finished body, before it is defined in the module.
//!
//! Stages: vertex, pixel, geometry and mesh, which are what the translator emits, and compute
//! (thread ids only). Hull, domain and amplification shaders, a mesh shader's payload, matrix,
//! 16-bit and 64-bit signature elements, arrays of structs and SV_Target arrays are
//! `Error::Unsupported`.
//!
//! What DXC rejects is `Error::Source` here too: its front end's semantic checks in [`begin`],
//! and the rules its validator has for signatures (it runs on DXC's own output and fails the
//! compile): the packed layout, the type, index and count of system values and the stage's
//! output limits in [`begin`], a system value output that the code does not write in
//! [`StageIo::finish`]. A module built through this file passes them.

mod pack;
mod sem;

use super::ast::{self, qual, AttrArg, Expr, ExprId, ScalarKind, StreamKind, Sym, TypeId, TypeRef, Unit};
use super::emit::Emit;
use super::types::Sk;
use crate::consts::{ComponentType, InputPrimitive, InterpolationMode, MeshOutputTopology, OpCode, PrimitiveTopology, SemanticKind, ShaderKind, SigPointKind};
use crate::ir::{Body, CastOp, Const, IPred, Inst, Module, Op, Ty, Value};
use crate::shader::{ShaderInfo, SigElement, StageProps};
use crate::{Error, Result};
use pack::{Interpretation, Packing};
use std::collections::HashMap;

/// What an entry parameter is to the lowering.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ParamKind {
    /// A value read at the start ([`StageIo::input`]): a scalar, vector or struct of inputs or
    /// system values (`XclIn xi`, `uint id : SV_VertexID`, `uint3 gid : SV_GroupID`). An
    /// ordinary local afterwards.
    Input,
    /// `out T x`: an ordinary local with an undefined initial value, written to the signature
    /// at the exit ([`StageIo::output`]).
    Output,
    /// `inout T x` (not a stream): both of the above.
    InOut,
    /// `triangle T v[3]` (GS): `len` elements read with [`StageIo::input_at`]; not assignable.
    InputArray { len: u32 },
    /// `inout TriangleStream<T> s` (GS): only `s.Append(x)` and `s.RestartStrip()`.
    Stream,
    /// `out vertices T v[n]` (MS): write-only, through [`StageIo::mesh_store`].
    MeshVertices { len: u32 },
    /// `out primitives T p[n]` (MS): as above.
    MeshPrimitives { len: u32 },
    /// `out indices uint3 t[n]` (`uint2` for lines): as above, a whole element at a time.
    MeshIndices { len: u32 },
}

/// One entry parameter.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub kind: ParamKind,
    /// The type whose flattened components cross the seam: the parameter's type for the value
    /// kinds, the element type for arrays and streams.
    pub elem: TypeId,
    /// Number of flattened components of that type.
    pub comps: u32,
}

/// Which output [`StageIo::output`] writes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// the entry function's return value
    Return,
    /// the `Output` / `InOut` parameter of that index
    Param(usize),
}

/// `EvaluateAttributeCentroid` / `AtSample` / `Snapped` (pixel shader).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Eval {
    Centroid,
    /// the argument is the sample index (i32)
    AtSample,
    /// the argument is the offset (two i32 components)
    Snapped,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Sig {
    In = 0,
    Out = 1,
    /// per-primitive outputs of a mesh shader
    Patch = 2,
}

/// How the components of a leaf are read or written.
#[derive(Copy, Clone, Debug)]
enum Access {
    /// element `id` of a signature
    Elem { id: u32 },
    /// a system value with an operation of its own
    Op(OpCode),
    /// the same, one call per component (the thread ids)
    OpComp(OpCode),
    /// SV_CullPrimitive read in a pixel shader: DXC replaces it with false
    Zero,
}

/// A run of flattened components with one semantic: a field (or the whole value) of numeric
/// type, `rows` array elements of `cols` components each.
#[derive(Copy, Clone, Debug)]
struct Leaf {
    /// first flattened component within the parameter's (element) type
    first: u32,
    sk: Sk,
    rows: u32,
    cols: u32,
    access: Access,
}

#[derive(Default)]
struct ParamIo {
    /// the input side, or the only side
    leaves: Vec<Leaf>,
    /// the output side of an `InOut` parameter
    out_leaves: Vec<Leaf>,
    /// output stream of a `Stream` parameter
    stream: u32,
}

/// How a component is stored: the three store operations share their operands, the mesh ones
/// add the vertex or primitive index.
#[derive(Copy, Clone)]
enum Store {
    Output,
    Vertex(Value),
    Primitive(Value),
}

/// The inputs and outputs of one entry function. Created by [`begin`].
pub struct StageIo {
    kind: ShaderKind,
    params: Vec<Param>,
    io: Vec<ParamIo>,
    ret: Vec<Leaf>,
    /// input, output, per-primitive
    sigs: [Vec<SigElement>; 3],
    props: StageProps,
}

fn kind_name(kind: ShaderKind) -> &'static str {
    match kind {
        ShaderKind::Pixel => "ps",
        ShaderKind::Vertex => "vs",
        ShaderKind::Geometry => "gs",
        ShaderKind::Hull => "hs",
        ShaderKind::Domain => "ds",
        ShaderKind::Compute => "cs",
        ShaderKind::Mesh => "ms",
        ShaderKind::Amplification => "as",
        _ => "lib",
    }
}

fn source<T>(line: u32, msg: impl Into<String>) -> Result<T> {
    Err(Error::Source { line, msg: msg.into() })
}

/// The entry attributes DXC reads for the stage properties.
#[derive(Default)]
struct Attrs {
    numthreads: Option<[u32; 3]>,
    wave_size: Option<u32>,
    max_vertex_count: Option<u32>,
    instance: Option<u32>,
    topology: Option<String>,
    early_depth_stencil: bool,
}

fn attrs(unit: &Unit, entry: &ast::Func, kind: ShaderKind) -> Result<Attrs> {
    let mut a = Attrs::default();
    for attr in &entry.attrs {
        let ints: Vec<u32> = attr.args.iter().filter_map(|x| if let AttrArg::Int(v) = x { u32::try_from(*v).ok() } else { None }).collect();
        let name = unit.name(attr.name).to_ascii_lowercase();
        let all_ints = ints.len() == attr.args.len();
        match (name.as_str(), kind) {
            ("numthreads", ShaderKind::Compute | ShaderKind::Mesh | ShaderKind::Amplification) if all_ints && ints.len() == 3 => a.numthreads = Some([ints[0], ints[1], ints[2]]),
            ("wavesize", ShaderKind::Compute) if all_ints && ints.len() == 1 => a.wave_size = Some(ints[0]),
            ("wavesize", _) => return source(entry.line, "attribute WaveSize only allowed on compute or node shaders"),
            ("maxvertexcount", ShaderKind::Geometry) if all_ints && ints.len() == 1 => a.max_vertex_count = Some(ints[0]),
            ("instance", ShaderKind::Geometry) if all_ints && ints.len() == 1 => a.instance = Some(ints[0]),
            ("outputtopology", ShaderKind::Mesh) => match attr.args.as_slice() {
                [AttrArg::Str(s)] => a.topology = Some(s.to_ascii_lowercase()),
                _ => return source(entry.line, "outputtopology takes one string"),
            },
            ("earlydepthstencil", ShaderKind::Pixel) if attr.args.is_empty() => a.early_depth_stencil = true,
            _ => return crate::unsupported(format!("entry attribute {name} on a {} shader", kind_name(kind))),
        }
    }
    Ok(a)
}

/// Builds the signatures while the entry's parameters are walked.
struct Builder<'a> {
    unit: &'a Unit,
    kind: ShaderKind,
    structs: HashMap<Sym, u32>,
    sigs: [Vec<SigElement>; 3],
    /// per element of `sigs`: what its semantic is at its signature point
    interpretations: [Vec<Interpretation>; 3],
}

impl Builder<'_> {
    fn dim(&self, x: ExprId, line: u32) -> Result<u32> {
        match *self.unit.expr(x) {
            Expr::Int(v, _) if (1..=1 << 16).contains(&v) => Ok(v as u32),
            Expr::Int(..) => source(line, "array size"),
            _ => crate::unsupported("an array size that is not a literal in an entry signature"),
        }
    }

    /// Walk a parameter's (or field's, or the return value's) type: one [`Leaf`] per numeric
    /// field in declaration order, which is DXC's order of flattened arguments and so of
    /// element ids. `first` counts flattened components.
    #[allow(clippy::too_many_arguments)]
    fn walk(&mut self, ty: TypeId, dims: &[ExprId], semantic: Option<Sym>, quals: u32, point: SigPointKind, stream: u32, line: u32, first: &mut u32, out: &mut Vec<Leaf>) -> Result<()> {
        match *self.unit.ty(ty) {
            TypeRef::Num { kind, rows, cols } => {
                if rows != 1 {
                    return crate::unsupported("a matrix in an entry signature");
                }
                let sk = match kind {
                    ScalarKind::Bool => Sk::Bool,
                    ScalarKind::Int => Sk::Int,
                    ScalarKind::Uint => Sk::Uint,
                    ScalarKind::Half | ScalarKind::Float => Sk::Float,
                    _ => return crate::unsupported("a 16-bit, 64-bit or minimum-precision type in an entry signature"),
                };
                let rows = match dims {
                    [] => 1,
                    [n] => self.dim(*n, line)?,
                    _ => return crate::unsupported("a multi-dimensional array in an entry signature"),
                };
                let Some(semantic) = semantic else {
                    return source(line, "Semantic must be defined for all parameters of an entry function or patch constant function");
                };
                let access = self.element(semantic, sk, quals, rows, cols as u32, point, stream, line)?;
                out.push(Leaf { first: *first, sk, rows, cols: cols as u32, access });
                *first += rows * cols as u32;
                Ok(())
            }
            TypeRef::Named(name) => {
                let Some(&s) = self.structs.get(&name) else {
                    return source(line, format!("unknown type {}", self.unit.name(name)));
                };
                if !dims.is_empty() {
                    return crate::unsupported("an array of structs in an entry signature");
                }
                if semantic.is_some() {
                    return crate::unsupported("a semantic on a struct in an entry signature");
                }
                let interp = quals & INTERP_QUALS;
                for f in &self.unit.structs[s as usize].fields {
                    // a field without interpolation qualifiers takes its struct's
                    let q = if f.quals & INTERP_QUALS != 0 { f.quals } else { f.quals | interp };
                    self.walk(f.ty, &f.dims, f.semantic, q, point, stream, f.line, first, out)?;
                }
                Ok(())
            }
            TypeRef::Void => Ok(()),
            _ => source(line, "a resource, stream or patch in an entry signature"),
        }
    }

    /// One flattened argument (DXC's `HLSignatureLower::ProcessArgument`): the element it
    /// adds to a signature, if any, and how it is accessed.
    #[allow(clippy::too_many_arguments)]
    fn element(&mut self, semantic: Sym, sk: Sk, quals: u32, rows: u32, cols: u32, point: SigPointKind, stream: u32, line: u32) -> Result<Access> {
        let full = self.unit.name(semantic);
        let (name, index) = sem::decompose(full);
        let Some(kind) = sem::by_name(name) else {
            return source(line, format!("invalid semantic '{full}'"));
        };
        let interpretation = sem::interpretation(kind, point);
        match interpretation {
            Interpretation::NA => return source(line, format!("Semantic {name} is invalid for shader model: {}", kind_name(self.kind))),
            Interpretation::NotInSig => return own_operation(kind, sk, rows, cols, name, line),
            _ => {}
        }
        use SigPointKind as P;
        let sig = match point {
            P::VSIn | P::PSIn | P::GSVIn | P::GSIn => Sig::In,
            P::VSOut | P::PSOut | P::GSOut | P::MSOut => Sig::Out,
            P::MSPOut => Sig::Patch,
            _ => return Err(Error::Internal(format!("no signature at {point:?}"))),
        };
        let Some(written) = sem::mode_of(quals) else {
            return source(line, format!("invalid interpolation mode for '{full}'"));
        };
        let Some(interp) = sem::element_mode(written, point, sk == Sk::Float, kind == SemanticKind::Position && interpretation == Interpretation::SV) else {
            return source(line, "invalid interpolation mode for SV_Position");
        };
        // DXC's patchIsFrontface: the bool is a uint in the signature since validator 1.2
        let comp_type = if kind == SemanticKind::IsFrontFace && matches!(point, P::PSIn | P::GSOut) { ComponentType::U32 } else { sk.component_type() };
        // a system value name where it is only an ordinary element keeps its spelling
        let (kind, name) = match (interpretation, crate::container::sig::semantic_name(kind)) {
            (Interpretation::Arb, _) | (_, None) => (SemanticKind::Arbitrary, name),
            (_, Some(canonical)) => (kind, canonical),
        };
        if interpretation == Interpretation::Target && rows != 1 {
            return crate::unsupported("an SV_Target array");
        }
        let elements = &mut self.sigs[sig as usize];
        let id = elements.len() as u32;
        if id >= 256 {
            return source(line, "too many signature elements");
        }
        elements.push(SigElement {
            name: name.to_string(),
            indices: (index..index + rows).collect(),
            kind,
            comp_type,
            interp,
            rows,
            cols,
            start_row: -1,
            start_col: -1,
            stream,
            usage_mask: 0,
            dynamic_index_mask: 0,
        });
        self.interpretations[sig as usize].push(interpretation);
        if interpretation == Interpretation::Shadow {
            return own_operation(kind, sk, rows, cols, name, line);
        }
        Ok(Access::Elem { id })
    }

    /// DXC rejects two elements of one signature (and stream) that share a semantic index:
    /// system values in `ProcessArgument`, the others in the validator.
    fn check_overlap(&self, line: u32) -> Result<()> {
        for sig in &self.sigs {
            for (i, a) in sig.iter().enumerate() {
                for b in &sig[..i] {
                    let same = a.stream == b.stream && a.kind == b.kind && a.name.eq_ignore_ascii_case(&b.name);
                    if same && a.indices[0] < b.indices[0] + b.rows && b.indices[0] < a.indices[0] + a.rows {
                        return source(line, format!("Parameter with semantic {} has overlapping semantic index at {}.", a.name, a.indices[0].max(b.indices[0])));
                    }
                }
            }
        }
        Ok(())
    }

    /// The validator's rules for the packed signatures that the declaration alone can break
    /// (`ValidateSignatureElement`, `ValidateSignature` and `ValidateEntrySignatures` in
    /// lib/DxilValidation/DxilValidation.cpp). DXC fails the compile on them ("error:
    /// validation errors"), so the same source is an error here.
    fn check_validation(&self, props: &StageProps, line: u32) -> Result<()> {
        use ComponentType as C;
        use InterpolationMode as M;
        use SemanticKind as K;
        let scalars = |sig: &[SigElement]| sig.iter().map(|e| e.rows * e.cols).sum::<u32>();
        let rows = |sig: &[SigElement]| sig.iter().map(|e| (e.start_row + e.rows as i32).max(0) as u32).max().unwrap_or(0);
        let [inputs, outputs, patch] = &self.sigs;
        match *props {
            StageProps::Pixel { .. } => {
                // a conservative depth output needs a position that is not perspective-divided
                // per sample, unless the shader runs per sample anyway
                let per_sample = inputs.iter().any(|e| e.kind == K::SampleIndex || matches!(e.interp, M::LinearSample | M::LinearNoperspectiveSample));
                let conservative = matches!(outputs.iter().find(|e| matches!(e.kind, K::Depth | K::DepthGreaterEqual | K::DepthLessEqual)), Some(e) if e.kind != K::Depth);
                let position = inputs.iter().find(|e| e.kind == K::Position);
                if !per_sample && conservative && position.is_some_and(|e| !matches!(e.interp, M::Undefined | M::LinearNoperspectiveCentroid | M::LinearNoperspectiveSample)) {
                    return source(line, "Interpolation mode for PS input position must be linear_noperspective_centroid or linear_noperspective_sample when outputting oDepthGE or oDepthLE and not running at sample frequency (which is forced by inputting SV_SampleIndex or declaring an input linear_sample or linear_noperspective_sample).");
                }
            }
            StageProps::Geometry { max_vertex_count, .. } => {
                let total = max_vertex_count * scalars(outputs);
                if total > 1024 {
                    return source(line, format!("Declared output vertex count ({max_vertex_count}) multiplied by the total number of declared scalar components of output data ({}) equals {total}. This value cannot be greater than 1024.", scalars(outputs)));
                }
            }
            StageProps::Mesh { max_vertices, max_primitives, payload_bytes, .. } => {
                if rows(outputs) > 32 || rows(patch) > 32 || rows(outputs) + rows(patch) > 32 {
                    return source(line, "For a mesh shader, the vertex and primitive output signatures together take up at most 32 rows.");
                }
                let bytes = 4 * (scalars(outputs) * max_vertices.next_multiple_of(32) + scalars(patch) * max_primitives.next_multiple_of(32));
                if bytes > 32 * 1024 {
                    return source(line, "For a mesh shader, the maximum size of the outputs is 32768 bytes.");
                }
                if bytes + payload_bytes > 47 * 1024 {
                    return source(line, "For a mesh shader, the maximum size of the payload and the outputs together is 48128 bytes.");
                }
            }
            _ => {}
        }
        for e in self.sigs.iter().flatten() {
            let float = matches!(e.comp_type, C::F16 | C::F32 | C::F64);
            let uint = matches!(e.comp_type, C::U16 | C::U32);
            if !float && matches!(e.interp, M::Linear | M::LinearCentroid | M::LinearNoperspective | M::LinearNoperspectiveCentroid | M::LinearSample | M::LinearNoperspectiveSample) {
                return source(line, format!("signature {} specifies invalid interpolation mode for integer component type.", e.name));
            }
            let wants = match e.kind {
                K::Depth | K::DepthGreaterEqual | K::DepthLessEqual if !float || e.cols != 1 => "float",
                K::Coverage | K::InnerCoverage | K::OutputControlPointID if e.comp_type != C::U32 || e.cols != 1 => "uint",
                K::Position if !float || e.cols != 4 => "float4",
                K::ClipDistance | K::CullDistance | K::TessFactor | K::InsideTessFactor | K::Barycentrics if !float => "float",
                K::Barycentrics if e.cols != 3 => "float3",
                K::IsFrontFace if !matches!(e.comp_type, C::I32 | C::U32) || e.cols != 1 => "uint",
                K::RenderTargetArrayIndex | K::ViewPortArrayIndex | K::VertexID | K::PrimitiveID | K::InstanceID | K::GSInstanceID | K::SampleIndex | K::StencilRef | K::ShadingRate
                    if !uint || e.cols != 1 =>
                {
                    "uint"
                }
                K::CullPrimitive if e.comp_type != C::I1 || e.cols != 1 => "bool",
                _ => "",
            };
            if !wants.is_empty() {
                return source(line, format!("{} must be {wants}", e.name));
            }
            let max_index = match e.kind {
                K::Arbitrary | K::ClipDistance | K::CullDistance => continue,
                K::Target if e.start_row + e.rows as i32 > 8 => 7,
                K::Target => continue,
                K::Barycentrics => 1,
                _ => 0,
            };
            if e.indices[0] > max_index {
                return source(line, format!("{} semantic index exceeds maximum ({max_index})", e.name));
            }
            if e.rows > 1 && !matches!(e.kind, K::TessFactor | K::InsideTessFactor) {
                return source(line, format!("signature element {} is a system value: rows must be 1", e.name));
            }
        }
        for sig in &self.sigs {
            let mut clip_rows: [Vec<i32>; 4] = Default::default();
            let mut clip_comps = [0u32; 4];
            let mut seen: [Vec<K>; 4] = Default::default();
            let mut depth = false;
            let mut barycentrics: Vec<bool> = Vec::new();
            for e in sig {
                let stream = e.stream as usize % 4;
                match e.kind {
                    K::Arbitrary | K::Target | K::Invalid => {}
                    K::ClipDistance | K::CullDistance => {
                        for row in e.start_row..e.start_row + e.rows as i32 {
                            if !clip_rows[stream].contains(&row) {
                                clip_rows[stream].push(row);
                            }
                        }
                        if clip_rows[stream].len() > 2 {
                            return source(line, "ClipDistance and CullDistance occupy more than the maximum of 2 rows combined.");
                        }
                        clip_comps[stream] += e.cols;
                        if clip_comps[stream] > 8 {
                            return source(line, "ClipDistance and CullDistance use more than the maximum of 8 components combined.");
                        }
                    }
                    K::Depth | K::DepthGreaterEqual | K::DepthLessEqual => {
                        if depth {
                            return source(line, "Pixel Shader only allows one type of depth semantic to be declared.");
                        }
                        depth = true;
                    }
                    K::Barycentrics => {
                        let noperspective = matches!(e.interp, InterpolationMode::LinearNoperspective | InterpolationMode::LinearNoperspectiveCentroid | InterpolationMode::LinearNoperspectiveSample);
                        if barycentrics.len() > 1 || barycentrics.last() == Some(&noperspective) {
                            return source(line, "There can only be up to two input attributes of SV_Barycentrics with different perspective interpolation mode.");
                        }
                        barycentrics.push(noperspective);
                    }
                    kind => {
                        if seen[stream].contains(&kind) {
                            return source(line, format!("System value {} appears more than once in the same signature.", e.name));
                        }
                        seen[stream].push(kind);
                    }
                }
            }
        }
        Ok(())
    }

    /// Give every element its register and column (DXC's `AllocateDxilInputOutputs`).
    fn pack(&mut self, line: u32) -> Result<()> {
        use ShaderKind as K;
        let schemes: [Option<Packing>; 3] = match self.kind {
            K::Vertex => [Some(Packing::InputAssembler), Some(Packing::Vertex), None],
            K::Pixel => [Some(Packing::Vertex), Some(Packing::Target), None],
            K::Geometry => [Some(Packing::Vertex), Some(Packing::Vertex), None],
            K::Mesh => [None, Some(Packing::Vertex), Some(Packing::Vertex)],
            _ => [None; 3],
        };
        for (s, scheme) in schemes.into_iter().enumerate() {
            let Some(scheme) = scheme else { continue };
            let mut elems: Vec<pack::Elem> = self.sigs[s]
                .iter()
                .zip(&self.interpretations[s])
                .map(|(e, &interpretation)| pack::Elem { interpretation, mode: e.interp, rows: e.rows, cols: e.cols, index: e.indices[0], stream: e.stream, at: None })
                .collect();
            let streams = self.kind == K::Geometry && s == Sig::Out as usize;
            if !pack::pack(&mut elems, scheme, streams) {
                return source(line, "Failed to allocate all signature elements in available space.");
            }
            if let Some((i, c)) = pack::conflict(&elems, streams) {
                let (e, at) = (&self.sigs[s][i], elems[i].at.unwrap_or_default());
                let what = match c {
                    pack::Conflict::Indexed => "has an indexing conflict with another signature element packed into the same row.",
                    pack::Conflict::Mode => "has interpolation mode that differs from another element packed into the same row.",
                    pack::Conflict::Overlap | pack::Conflict::Full => "overlaps another signature element.",
                    pack::Conflict::Order => "violates component ordering rule (arb < sv < sgv).",
                    pack::Conflict::Fit => "is out of range.",
                };
                return source(line, format!("signature element {} at location ({},{}) size ({},{}) {what}", e.name, at.0, at.1, e.rows, e.cols));
            }
            for (e, p) in self.sigs[s].iter_mut().zip(&elems) {
                if let Some((row, col)) = p.at {
                    (e.start_row, e.start_col) = (row as i32, col as i32);
                }
            }
        }
        Ok(())
    }
}

const INTERP_QUALS: u32 = qual::NOINTERPOLATION | qual::LINEAR | qual::CENTROID | qual::NOPERSPECTIVE | qual::SAMPLE;
const GS_PRIMITIVE_QUALS: u32 = qual::GS_POINT | qual::GS_LINE | qual::GS_TRIANGLE | qual::GS_LINEADJ | qual::GS_TRIANGLEADJ;

/// The operation that reads a system value which has one (DXC's
/// `replaceInputOutputWithIntrinsic`).
fn own_operation(kind: SemanticKind, sk: Sk, rows: u32, cols: u32, name: &str, line: u32) -> Result<Access> {
    use SemanticKind as K;
    let access = match kind {
        K::GSInstanceID => Access::Op(OpCode::GSInstanceID),
        K::PrimitiveID => Access::Op(OpCode::PrimitiveID),
        K::SampleIndex => Access::Op(OpCode::SampleIndex),
        K::Coverage => Access::Op(OpCode::Coverage),
        K::InnerCoverage => Access::Op(OpCode::InnerCoverage),
        K::ViewID => Access::Op(OpCode::ViewID),
        K::GroupIndex => Access::Op(OpCode::FlattenedThreadIdInGroup),
        K::GroupThreadID => Access::OpComp(OpCode::ThreadIdInGroup),
        K::GroupID => Access::OpComp(OpCode::GroupId),
        K::DispatchThreadID => Access::OpComp(OpCode::ThreadId),
        K::CullPrimitive => Access::Zero,
        // SV_OutputControlPointID and SV_DomainLocation belong to the tessellation stages
        _ => return crate::unsupported(format!("semantic {name}")),
    };
    let fits = match access {
        Access::Zero => rows == 1 && cols == 1,
        Access::OpComp(_) => rows == 1 && cols <= 3 && matches!(sk, Sk::Int | Sk::Uint),
        _ => rows == 1 && cols == 1 && matches!(sk, Sk::Int | Sk::Uint),
    };
    if !fits {
        return source(line, format!("invalid type used for '{name}' semantic"));
    }
    Ok(access)
}

/// Build the signatures and stage properties of `entry` for a shader of `kind` (compute
/// included: its thread-id parameters are `Input`s). The structs the entry's parameters use
/// are read from the AST (semantics and interpolation qualifiers are on their fields); nothing
/// is emitted. `Error::Unsupported` for a stage, semantic or parameter shape outside what is
/// implemented, `Error::Source` for what DXC rejects (a missing or overlapping semantic, a
/// semantic that is not valid at that point). `_e` is not used yet (a payload will need its
/// type table).
pub fn begin(_e: &mut Emit, unit: &Unit, entry: &ast::Func, kind: ShaderKind) -> Result<StageIo> {
    use ShaderKind as K;
    use SigPointKind as P;
    let (in_point, out_point) = match kind {
        K::Vertex => (P::VSIn, Some(P::VSOut)),
        K::Pixel => (P::PSIn, Some(P::PSOut)),
        K::Geometry => (P::GSIn, None),
        K::Mesh => (P::MSIn, None),
        K::Compute => (P::CSIn, None),
        _ => return crate::unsupported(format!("stage {}", kind_name(kind))),
    };
    let a = attrs(unit, entry, kind)?;
    let line = entry.line;
    let mut b = Builder { unit, kind, structs: HashMap::new(), sigs: Default::default(), interpretations: Default::default() };
    for (i, s) in unit.structs.iter().enumerate() {
        b.structs.insert(s.name, i as u32);
    }

    let mut params = Vec::with_capacity(entry.params.len());
    let mut io = Vec::with_capacity(entry.params.len());
    // the return value is DXC's first output argument
    let mut ret = Vec::new();
    if *unit.ty(entry.ret) != TypeRef::Void {
        let Some(point) = out_point else {
            return source(line, format!("a {} entry function returns void", kind_name(kind)));
        };
        b.walk(entry.ret, &[], entry.ret_semantic, 0, point, 0, line, &mut 0, &mut ret)?;
    }
    let (mut gs_input, mut gs_topology, mut streams) = (None, None, 0u32);
    let (mut max_vertices, mut max_primitives, mut index_len, mut index_cols) = (None, None, None, 0u8);
    for p in &entry.params {
        let mut pio = ParamIo::default();
        let mut comps = 0u32;
        let len = |b: &Builder| match p.dims.as_slice() {
            [n] => b.dim(*n, p.line),
            _ => source(p.line, "a one-dimensional array is expected"),
        };
        let (pkind, elem) = match *unit.ty(p.ty) {
            TypeRef::Stream { kind: sk, elem } => {
                if kind != K::Geometry || p.quals & qual::INOUT != qual::INOUT || !p.dims.is_empty() {
                    return source(p.line, "a stream is an inout parameter of a geometry shader");
                }
                let topology = match sk {
                    StreamKind::Point => PrimitiveTopology::PointList,
                    StreamKind::Line => PrimitiveTopology::LineStrip,
                    StreamKind::Triangle => PrimitiveTopology::TriangleStrip,
                };
                if streams >= 4 || gs_topology.is_some_and(|t| t != PrimitiveTopology::PointList || topology != t) {
                    return source(p.line, "when multiple GS output streams are used they must be pointlists");
                }
                gs_topology = Some(topology);
                pio.stream = streams;
                streams += 1;
                b.walk(elem, &[], None, 0, P::GSOut, pio.stream, p.line, &mut comps, &mut pio.leaves)?;
                (ParamKind::Stream, elem)
            }
            TypeRef::Patch { .. } => return crate::unsupported("InputPatch / OutputPatch"),
            _ if p.quals & qual::PAYLOAD != 0 => return crate::unsupported("a mesh shader payload"),
            _ if p.quals & (qual::VERTICES | qual::PRIMITIVES | qual::INDICES) != 0 => {
                if kind != K::Mesh || p.quals & qual::INOUT != qual::OUT {
                    return source(p.line, "vertices, primitives and indices are out parameters of a mesh shader");
                }
                let len = len(&b)?;
                if p.quals & qual::INDICES != 0 {
                    let TypeRef::Num { kind: ScalarKind::Uint, rows: 1, cols: cols @ 2..=3 } = *unit.ty(p.ty) else {
                        return source(p.line, "the indices of a mesh shader are uint2 or uint3");
                    };
                    (index_len, index_cols, comps) = (Some(len), cols, cols as u32);
                    (ParamKind::MeshIndices { len }, p.ty)
                } else if p.quals & qual::VERTICES != 0 {
                    max_vertices = Some(len);
                    b.walk(p.ty, &[], p.semantic, p.quals, P::MSOut, 0, p.line, &mut comps, &mut pio.leaves)?;
                    (ParamKind::MeshVertices { len }, p.ty)
                } else {
                    max_primitives = Some(len);
                    b.walk(p.ty, &[], p.semantic, p.quals, P::MSPOut, 0, p.line, &mut comps, &mut pio.leaves)?;
                    (ParamKind::MeshPrimitives { len }, p.ty)
                }
            }
            _ if p.quals & GS_PRIMITIVE_QUALS != 0 => {
                let (primitive, vertices) = match p.quals & GS_PRIMITIVE_QUALS {
                    qual::GS_POINT => (InputPrimitive::Point, 1),
                    qual::GS_LINE => (InputPrimitive::Line, 2),
                    qual::GS_TRIANGLE => (InputPrimitive::Triangle, 3),
                    qual::GS_LINEADJ => (InputPrimitive::LineWithAdjacency, 4),
                    qual::GS_TRIANGLEADJ => (InputPrimitive::TriangleWithAdjacency, 6),
                    _ => return source(p.line, "more than one primitive type"),
                };
                let len = len(&b)?;
                if kind != K::Geometry || p.quals & qual::OUT != 0 || gs_input.is_some() || len != vertices {
                    return source(p.line, "the input primitive of a geometry shader is one array of its vertices");
                }
                gs_input = Some(primitive);
                b.walk(p.ty, &[], p.semantic, p.quals, P::GSVIn, 0, p.line, &mut comps, &mut pio.leaves)?;
                (ParamKind::InputArray { len }, p.ty)
            }
            _ => {
                let pkind = match p.quals & qual::INOUT {
                    qual::OUT => ParamKind::Output,
                    qual::INOUT => ParamKind::InOut,
                    _ => ParamKind::Input,
                };
                if !p.dims.is_empty() {
                    return crate::unsupported("an array as an entry parameter");
                }
                if pkind != ParamKind::Input {
                    let Some(point) = out_point else {
                        return source(p.line, format!("an out parameter of a {} entry function", kind_name(kind)));
                    };
                    // DXC does the output side of an inout parameter first
                    let leaves = if pkind == ParamKind::InOut { &mut pio.out_leaves } else { &mut pio.leaves };
                    b.walk(p.ty, &p.dims, p.semantic, p.quals, point, 0, p.line, &mut comps, leaves)?;
                }
                if pkind != ParamKind::Output {
                    comps = 0;
                    b.walk(p.ty, &p.dims, p.semantic, p.quals, in_point, 0, p.line, &mut comps, &mut pio.leaves)?;
                }
                (pkind, p.ty)
            }
        };
        params.push(Param { kind: pkind, elem, comps });
        io.push(pio);
    }
    b.check_overlap(line)?;
    b.pack(line)?;

    let props = match kind {
        K::Vertex => StageProps::Vertex,
        K::Pixel => StageProps::Pixel { early_depth_stencil: a.early_depth_stencil },
        K::Compute => match a.numthreads {
            Some(numthreads) => StageProps::Compute { numthreads, wave_size: a.wave_size },
            None => return source(line, "a compute shader needs [numthreads]"),
        },
        K::Geometry => match (gs_input, gs_topology, a.max_vertex_count) {
            (Some(input_primitive), Some(stream_topology), Some(max_vertex_count)) => {
                StageProps::Geometry { input_primitive, max_vertex_count, active_streams: (1 << streams) - 1, stream_topology, instance_count: a.instance.unwrap_or(1) }
            }
            _ => return source(line, "a geometry shader needs [maxvertexcount], an input primitive and an output stream"),
        },
        _ => {
            let topology = match (a.topology.as_deref(), index_cols) {
                (Some("triangle"), 3) => MeshOutputTopology::Triangle,
                (Some("line"), 2) => MeshOutputTopology::Line,
                _ => return source(line, "a mesh shader needs [outputtopology] of \"line\" or \"triangle\" and indices of that size"),
            };
            let (Some(numthreads), Some(max_vertices), Some(indices)) = (a.numthreads, max_vertices, index_len) else {
                return source(line, "a mesh shader needs [numthreads], out vertices and out indices");
            };
            if max_primitives.is_some_and(|n| n != indices) {
                return source(line, "the primitives and indices of a mesh shader have the same size");
            }
            // DXC drops an attribute whose thread count is outside 1..=1024 and then has none;
            // the validator allows a mesh shader 128 per axis and 128 threads in all
            let threads = numthreads.iter().fold(1u64, |n, &x| n.saturating_mul(u64::from(x)));
            if threads == 0 || threads > 1024 {
                return source(line, "mesh entry point must have a valid numthreads attribute");
            }
            if let Some((axis, n)) = ["X", "Y", "Z"].iter().zip(numthreads).find(|&(_, n)| n > 128) {
                return source(line, format!("Declared Thread Group {axis} size {n} outside valid range [1..128]."));
            }
            if threads > 128 {
                return source(line, format!("Declared Thread Group Count {threads} (X*Y*Z) is beyond the valid maximum of 128."));
            }
            StageProps::Mesh { numthreads, max_vertices, max_primitives: indices, topology, payload_bytes: 0 }
        }
    };
    b.check_validation(&props, line)?;
    Ok(StageIo { kind, params, io, ret, sigs: b.sigs, props })
}

/// The `discard` statement: `dx.op.discard(true)` at the insertion point (DXC emits the call
/// where the statement is, under the statement's own control flow, and execution continues).
pub fn discard(e: &mut Emit) {
    let t = e.c_bool(true);
    e.dx(OpCode::Discard, Ty::VOID, &[t]);
}

/// Read one component of a leaf. `vertex`: the vertex of a geometry shader input, undef for
/// the stages whose inputs have none (as DXC writes it).
fn read(e: &mut Emit, leaf: &Leaf, row: u32, col: u32, vertex: Option<Value>) -> Value {
    match leaf.access {
        Access::Elem { id, .. } => {
            let args = [e.c_u32(id), e.c_u32(row), e.m.const_i8(col as u8), vertex.unwrap_or_else(|| e.m.undef(Ty::I32))];
            let v = e.dx(OpCode::LoadInput, if leaf.sk == Sk::Float { Ty::F32 } else { Ty::I32 }, &args);
            if leaf.sk == Sk::Bool {
                let zero = e.c_u32(0);
                e.icmp(IPred::Ne, v, zero)
            } else {
                v
            }
        }
        Access::Op(op) => e.dx(op, Ty::I32, &[]),
        Access::OpComp(op) => {
            let c = e.c_u32(col);
            e.dx(op, Ty::I32, &[c])
        }
        Access::Zero => e.zero(leaf.sk),
    }
}

fn is_undef(e: &Emit, v: Value) -> bool {
    e.m.as_const(v).is_some_and(|c| c.c == Const::Undef)
}

/// Store the components of `leaves` that lie in `first .. first + comps.len()` (flattened
/// component numbers of the whole value). An undefined component is not stored: DXC removes
/// stores of undef (RemoveStoreUndefOutput), and the element's usage mask then lacks it.
fn store(e: &mut Emit, leaves: &[Leaf], first: usize, comps: &[Value], how: Store) -> Result<()> {
    for leaf in leaves {
        let Access::Elem { id, .. } = leaf.access else {
            return Err(Error::Internal("an output without a signature element".into()));
        };
        for c in 0..leaf.rows * leaf.cols {
            let Some(&v) = (leaf.first + c).checked_sub(first as u32).and_then(|i| comps.get(i as usize)) else { continue };
            let v = e.body.resolve(v);
            if is_undef(e, v) {
                continue;
            }
            let (v, overload) = match leaf.sk {
                Sk::Float => (v, Ty::F32),
                Sk::Bool => (e.cast(CastOp::ZExt, Ty::I32, v), Ty::I32),
                _ => (v, Ty::I32),
            };
            let args = [e.c_u32(id), e.c_u32(c / leaf.cols), e.m.const_i8((c % leaf.cols) as u8), v];
            match how {
                Store::Output => e.dx(OpCode::StoreOutput, overload, &args),
                Store::Vertex(i) => e.dx(OpCode::StoreVertexOutput, overload, &[args[0], args[1], args[2], args[3], i]),
                Store::Primitive(i) => e.dx(OpCode::StorePrimitiveOutput, overload, &[args[0], args[1], args[2], args[3], i]),
            };
        }
    }
    Ok(())
}

impl StageIo {
    /// The entry parameters, in declaration order.
    pub fn params(&self) -> &[Param] {
        &self.params
    }

    fn param(&self, param: usize, what: &str, ok: impl Fn(ParamKind) -> bool) -> Result<&ParamIo> {
        match self.params.get(param) {
            Some(p) if ok(p.kind) => Ok(&self.io[param]),
            _ => Err(Error::Internal(format!("stage: parameter {param} is not {what}"))),
        }
    }

    /// The initial value of parameter `param` (kind `Input` or `InOut`): its flattened
    /// components, read at the insertion point (call it in the entry block, in parameter
    /// order: DXC reads every input at the start of the function). `loadInput` for signature
    /// elements, the operation of its own for a system value that has one (SV_PrimitiveID,
    /// SV_SampleIndex, SV_Coverage, SV_GSInstanceID, the thread ids, ...). Reads the lowered
    /// code never uses cost nothing: the writer drops them, and [`StageIo::finish`] counts
    /// only live ones in the usage masks.
    pub fn input(&mut self, e: &mut Emit, param: usize) -> Result<Vec<Value>> {
        let io = self.param(param, "an input", |k| matches!(k, ParamKind::Input | ParamKind::InOut))?;
        Ok(read_all(e, &io.leaves, None))
    }

    /// Element `vertex` (an i32 value, constant or not) of an `InputArray` parameter: its
    /// flattened components, read at the insertion point (`loadInput` with that vertex index).
    /// Call it where `v[i]` is evaluated; components the expression does not use are dropped
    /// as dead.
    pub fn input_at(&mut self, e: &mut Emit, param: usize, vertex: Value) -> Result<Vec<Value>> {
        let io = self.param(param, "an input array", |k| matches!(k, ParamKind::InputArray { .. }))?;
        Ok(read_all(e, &io.leaves, Some(vertex)))
    }

    /// Write an output at the insertion point: `comps` are the flattened components of the
    /// return value or of the `Output` / `InOut` parameter (`storeOutput` per component, in
    /// element order). Call it once per output, return value first and then the parameters
    /// in order, in the exit block every `return` of the entry function reaches. A component
    /// that is undefined there is not stored (and counts as never written), as in DXC.
    pub fn output(&mut self, e: &mut Emit, target: Target, comps: &[Value]) -> Result<()> {
        let (leaves, n) = match target {
            Target::Return => (&self.ret, self.ret.last().map_or(0, |l| l.first + l.rows * l.cols)),
            Target::Param(i) => {
                let io = self.param(i, "an output", |k| matches!(k, ParamKind::Output | ParamKind::InOut))?;
                (if self.params[i].kind == ParamKind::InOut { &io.out_leaves } else { &io.leaves }, self.params[i].comps)
            }
        };
        if comps.len() != n as usize {
            return Err(Error::Internal(format!("stage: {} components for an output of {n}", comps.len())));
        }
        store(e, leaves, 0, comps, Store::Output)
    }

    /// `s.Append(x)` on the `Stream` parameter `param`: `storeOutput` of every component of
    /// `x` (flattened, of the stream's element type), then `emitStream`.
    pub fn append(&mut self, e: &mut Emit, param: usize, comps: &[Value]) -> Result<()> {
        let io = self.param(param, "a stream", |k| k == ParamKind::Stream)?;
        if comps.len() != self.params[param].comps as usize {
            return Err(Error::Internal(format!("stage: {} components appended to a stream of {}", comps.len(), self.params[param].comps)));
        }
        store(e, &io.leaves, 0, comps, Store::Output)?;
        let stream = e.m.const_i8(io.stream as u8);
        e.dx(OpCode::EmitStream, Ty::VOID, &[stream]);
        Ok(())
    }

    /// `s.RestartStrip()`: `cutStream`.
    pub fn restart_strip(&mut self, e: &mut Emit, param: usize) -> Result<()> {
        let io = self.param(param, "a stream", |k| k == ParamKind::Stream)?;
        let stream = e.m.const_i8(io.stream as u8);
        e.dx(OpCode::CutStream, Ty::VOID, &[stream]);
        Ok(())
    }

    /// `SetMeshOutputCounts(vertices, primitives)` (two i32 values).
    pub fn set_mesh_output_counts(&mut self, e: &mut Emit, vertices: Value, primitives: Value) -> Result<()> {
        if self.kind != ShaderKind::Mesh {
            return source(e.line, "SetMeshOutputCounts outside a mesh shader");
        }
        e.dx(OpCode::SetMeshOutputCounts, Ty::VOID, &[vertices, primitives]);
        Ok(())
    }

    /// An assignment to element `index` (an i32 value) of a mesh output array parameter, or to
    /// part of that element: `comps` replace the flattened components
    /// `first_comp .. first_comp + comps.len()` of the element (`verts[i] = o` is the whole
    /// range, `verts[i].pos.y = v` one component). `storeVertexOutput` /
    /// `storePrimitiveOutput` per component; for `MeshIndices` the whole element at once
    /// (`emitIndices`, the third index of a line being 0; a partial write is `Unsupported`).
    /// The arrays cannot be read.
    pub fn mesh_store(&mut self, e: &mut Emit, param: usize, index: Value, first_comp: usize, comps: &[Value]) -> Result<()> {
        let io = self.param(param, "a mesh output", |k| matches!(k, ParamKind::MeshVertices { .. } | ParamKind::MeshPrimitives { .. } | ParamKind::MeshIndices { .. }))?;
        if first_comp + comps.len() > self.params[param].comps as usize {
            return Err(Error::Internal("stage: mesh output component range".into()));
        }
        match self.params[param].kind {
            ParamKind::MeshVertices { .. } => store(e, &io.leaves, first_comp, comps, Store::Vertex(index)),
            ParamKind::MeshPrimitives { .. } => store(e, &io.leaves, first_comp, comps, Store::Primitive(index)),
            _ => {
                if first_comp != 0 || comps.len() != self.params[param].comps as usize {
                    return crate::unsupported("a write to one component of a mesh shader's indices");
                }
                let third = comps.get(2).copied().unwrap_or_else(|| e.c_u32(0));
                e.dx(OpCode::EmitIndices, Ty::VOID, &[index, comps[0], comps[1], third]);
                Ok(())
            }
        }
    }

    /// `DispatchMesh(x, y, z, payload)` (amplification shader): `payload` are the flattened
    /// components of the payload struct `payload_ty`. Not implemented: amplification shaders
    /// are `Unsupported` in [`begin`] already.
    pub fn dispatch_mesh(&mut self, _e: &mut Emit, _xyz: [Value; 3], _payload_ty: TypeId, _payload: &[Value]) -> Result<()> {
        crate::unsupported(format!("DispatchMesh in a {} shader", kind_name(self.kind)))
    }

    /// `EvaluateAttributeCentroid(x)` / `EvaluateAttributeAtSample(x, i)` /
    /// `EvaluateAttributeSnapped(x, o)` where `x` is the flattened components
    /// `first_comp .. first_comp + n` of the `Input` parameter `param` of a pixel shader (the
    /// argument must be an input, or a field or swizzle of one, as DXC requires). `arg`: the
    /// second argument's components, converted to int. Returns the `n` results. These reads
    /// do not count in the element's usage mask (DXC's do not either).
    pub fn eval_attribute(&mut self, e: &mut Emit, how: Eval, param: usize, first_comp: usize, n: usize, arg: &[Value]) -> Result<Vec<Value>> {
        let io = self.param(param, "an input", |k| k == ParamKind::Input)?;
        if self.kind != ShaderKind::Pixel || arg.len() != [0, 1, 2][how as usize] {
            return source(e.line, "EvaluateAttribute* takes a pixel shader input");
        }
        let mut out = Vec::with_capacity(n);
        for c in first_comp as u32..(first_comp + n) as u32 {
            let leaf = io.leaves.iter().find(|l| (l.first..l.first + l.rows * l.cols).contains(&c));
            let Some(&Leaf { first, sk: Sk::Float, cols, access: Access::Elem { id, .. }, .. }) = leaf else {
                return source(e.line, "EvaluateAttribute* takes a float input of the signature");
            };
            let at = [e.c_u32(id), e.c_u32((c - first) / cols), e.m.const_i8(((c - first) % cols) as u8)];
            out.push(match how {
                Eval::Centroid => e.dx(OpCode::EvalCentroid, Ty::F32, &at),
                Eval::AtSample => e.dx(OpCode::EvalSampleIndex, Ty::F32, &[at[0], at[1], at[2], arg[0]]),
                Eval::Snapped => e.dx(OpCode::EvalSnapped, Ty::F32, &[at[0], at[1], at[2], arg[0], arg[1]]),
            });
        }
        Ok(out)
    }

    /// What the declaration alone decides: `info`'s signatures (`inputs`, `outputs`,
    /// `patch_or_prim`) with their registers but empty usage masks, and `props`.
    pub fn declared(&self, info: &mut ShaderInfo) {
        let [inputs, outputs, patch] = &self.sigs;
        info.inputs = inputs.clone();
        info.outputs = outputs.clone();
        info.patch_or_prim = patch.clone();
        info.props = self.props.clone();
    }

    /// Fill `info`'s signatures (`inputs`, `outputs`, `patch_or_prim`) and `props`. `body` is
    /// the finished entry function (before `Module::define`): the usage masks count the
    /// signature reads and writes that are live in it, as DXC's MarkUsedSignatureElements does
    /// after its optimiser (a component is used when some live load or store names its
    /// column, in any row; a row that is not a constant sets the dynamic-index mask too).
    ///
    /// `Error::Source` when a system value output is not written in every component by some
    /// store: the validator rejects that module ("Not all elements of output SV_Position
    /// were written", ValidateUninitializedOutput; ordinary elements and SV_Target may stay
    /// unwritten), so DXC fails to compile the same source.
    pub fn finish(mut self, m: &Module, body: &Body, info: &mut ShaderInfo) -> Result<()> {
        if self.sigs.iter().any(|s| !s.is_empty()) {
            let live = crate::bcwriter::liveness(body);
            if self.kind == ShaderKind::Mesh {
                mesh_counts(m, body, &live)?;
            }
            for (i, inst) in body.insts.iter().enumerate() {
                if inst.op != Op::Call || !live.insts[i] {
                    continue;
                }
                let args = body.operands_of(inst);
                let sig = match args.first().and_then(|&v| m.as_int(v)) {
                    Some(x) if x == OpCode::LoadInput as u64 => Sig::In,
                    Some(x) if x == OpCode::StoreOutput as u64 || x == OpCode::StoreVertexOutput as u64 => Sig::Out,
                    Some(x) if x == OpCode::StorePrimitiveOutput as u64 => Sig::Patch,
                    _ => continue,
                };
                if args.len() < 4 || !m.function(inst.a).name.starts_with("dx.op.") {
                    continue;
                }
                let (Some(id), Some(col)) = (m.as_int(args[1]), m.as_int(args[3])) else { continue };
                let Some(element) = self.sigs[sig as usize].get_mut(id as usize) else { continue };
                let bit = 1u8 << (col & 3);
                element.usage_mask |= bit;
                if m.as_int(body.resolve(args[2])).is_none() {
                    element.dynamic_index_mask |= bit;
                }
            }
        }
        if let Some(e) = self.sigs[Sig::Out as usize].iter().find(|e| !matches!(e.kind, SemanticKind::Arbitrary | SemanticKind::Target) && e.usage_mask != (1 << e.cols) - 1) {
            return source(0, format!("Not all elements of output {} were written.", e.name));
        }
        self.declared(info);
        Ok(())
    }
}

/// The validator's rules for `SetMeshOutputCounts` (ValidateMsIntrinsics), on the blocks
/// that get written: at most one call, and every store to a vertex or primitive output and
/// every `emitIndices` is in a block the call's block dominates (the same block counts,
/// whichever comes first in it). DXC fails to compile a source whose module breaks them.
fn mesh_counts(m: &Module, body: &Body, live: &crate::bcwriter::Live) -> Result<()> {
    let (mut call, mut stores) = (None, Vec::new());
    for (b, block) in body.blocks.iter().enumerate() {
        if !live.blocks[b] {
            continue;
        }
        for &i in &block.insts {
            let inst = &body.insts[i as usize];
            if inst.op != Op::Call || !live.insts[i as usize] || !m.function(inst.a).name.starts_with("dx.op.") {
                continue;
            }
            match body.operands_of(inst).first().and_then(|&v| m.as_int(v)) {
                Some(x) if x == OpCode::SetMeshOutputCounts as u64 => {
                    if call.replace(b).is_some() {
                        return source(0, "SetMeshOutputCounts cannot be called multiple times.");
                    }
                }
                Some(x) if x == OpCode::StoreVertexOutput as u64 || x == OpCode::StorePrimitiveOutput as u64 || x == OpCode::EmitIndices as u64 => {
                    if stores.last() != Some(&b) {
                        stores.push(b);
                    }
                }
                _ => {}
            }
        }
    }
    let Some(call) = call else {
        return if stores.is_empty() { Ok(()) } else { source(0, "Missing SetMeshOutputCounts call.") };
    };
    // a block the call's block does not dominate is one the entry reaches without it
    let mut seen = vec![false; body.blocks.len()];
    let mut work = Vec::new();
    if call != 0 {
        seen[0] = true;
        work.push(0u32);
    }
    while let Some(b) = work.pop() {
        let Some(t) = body.blocks[b as usize].insts.last() else { continue };
        crate::bcwriter::successors(body, &body.insts[*t as usize], |s| {
            if s as usize != call && !seen[s as usize] {
                seen[s as usize] = true;
                work.push(s);
            }
        });
    }
    if stores.iter().any(|&b| seen[b]) {
        return source(0, "Non-Dominating SetMeshOutputCounts call.");
    }
    Ok(())
}

/// The validator's rule for derivatives in a compute or mesh shader (CompatibilityChecker):
/// the thread group is one-dimensional with X a multiple of 4, or X and Y are both even, so
/// that the threads make quads. Counts the operations that get written, since DXC removes an
/// unused one as well. Call it when the function emitted some derivative operation.
pub fn derivative_group(m: &Module, body: &Body, props: &StageProps) -> Result<()> {
    let (StageProps::Compute { numthreads: [x, y, z], .. } | StageProps::Mesh { numthreads: [x, y, z], .. }) = *props else { return Ok(()) };
    if if y == 1 && z == 1 { x & 3 == 0 } else { (x | y) & 1 == 0 } {
        return Ok(());
    }
    let live = crate::bcwriter::liveness(body);
    let derivative = |(i, inst): (usize, &Inst)| {
        const OPS: [OpCode; 9] =
            [OpCode::DerivCoarseX, OpCode::DerivCoarseY, OpCode::DerivFineX, OpCode::DerivFineY, OpCode::CalculateLOD, OpCode::Sample, OpCode::SampleBias, OpCode::SampleCmp, OpCode::SampleCmpBias];
        inst.op == Op::Call
            && live.insts[i]
            && m.function(inst.a).name.starts_with("dx.op.")
            && body.operands_of(inst).first().and_then(|&v| m.as_int(v)).is_some_and(|x| OPS.iter().any(|&o| o as u64 == x))
    };
    if body.insts.iter().enumerate().any(derivative) {
        return source(0, format!("Function uses derivatives in compute-model shader with NumThreads ({x}, {y}, {z}); derivatives require NumThreads to be 1D and a multiple of 4, or 2D/3D with X and Y both being a multiple of 2."));
    }
    Ok(())
}

fn read_all(e: &mut Emit, leaves: &[Leaf], vertex: Option<Value>) -> Vec<Value> {
    let mut out = Vec::with_capacity(leaves.iter().map(|l| (l.rows * l.cols) as usize).sum());
    for leaf in leaves {
        for c in 0..leaf.rows * leaf.cols {
            out.push(read(e, leaf, c / leaf.cols, c % leaf.cols, vertex));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hlsl::parse;
    use crate::irprint;

    /// The entry of `src` ready to be driven as the lowering drives it.
    fn open(src: &str, kind: ShaderKind) -> (StageIo, Emit) {
        let unit = parse::parse(src).unwrap();
        let entry = unit.funcs.iter().rfind(|f| f.body.is_some() && unit.name(f.name) == "main").unwrap();
        let mut e = Emit::new();
        let io = begin(&mut e, &unit, entry, kind).unwrap();
        (io, e)
    }

    fn close(io: StageIo, mut e: Emit, kind: ShaderKind) -> (String, ShaderInfo) {
        e.body.ret(e.bb);
        let mut info = ShaderInfo::new(kind, "main", StageProps::Vertex);
        io.finish(&e.m, &e.body, &mut info).unwrap();
        (irprint::function(&e.m, &e.body), info)
    }

    fn masks(sig: &[SigElement]) -> Vec<u8> {
        sig.iter().map(|e| e.usage_mask).collect()
    }

    /// DXC's IR for scratch snippet ps_sv: a bool input is `icmp ne` of the i32 it is loaded
    /// as; SV_Coverage and SV_SampleIndex are operations, the primitive id of a pixel shader
    /// an element; an undefined component is not stored.
    #[test]
    fn pixel_inputs_and_outputs() {
        let src = "struct I { float4 pos : SV_Position; nointerpolation uint b : B; bool ff : SV_IsFrontFace; uint cov : SV_Coverage; uint pid : SV_PrimitiveID; uint si : SV_SampleIndex; };
struct O { float4 t0 : SV_Target0; float d : SV_Depth; uint cov : SV_Coverage; int2 t1 : SV_Target1; };
O main(I i) { O o; return o; }";
        let (mut io, mut e) = open(src, ShaderKind::Pixel);
        assert_eq!(io.params(), [Param { kind: ParamKind::Input, elem: io.params()[0].elem, comps: 9 }]);
        let v = io.input(&mut e, 0).unwrap();
        assert_eq!(v.len(), 9);
        let ff = e.convert_scalar(v[5], Sk::Bool, Sk::Float);
        let undef = e.m.undef(Ty::F32);
        discard(&mut e);
        // t0.xyz (w is never assigned), depth, coverage, t1
        io.output(&mut e, Target::Return, &[v[0], v[2], ff, undef, v[1], v[6], v[7], v[8]]).unwrap();
        let (text, info) = close(io, e, ShaderKind::Pixel);
        assert_eq!(
            text,
            "  %1 = call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 0, i32 undef), !dx.precise !{i32 1}
  %2 = call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 1, i32 undef), !dx.precise !{i32 1}
  %3 = call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 2, i32 undef), !dx.precise !{i32 1}
  %4 = call i32 @dx.op.loadInput.i32(i32 4, i32 2, i32 0, i8 0, i32 undef)
  %5 = icmp ne i32 %4, 0
  %6 = call i32 @dx.op.coverage.i32(i32 91)
  %7 = call i32 @dx.op.loadInput.i32(i32 4, i32 3, i32 0, i8 0, i32 undef)
  %8 = call i32 @dx.op.sampleIndex.i32(i32 90)
  %9 = uitofp i1 %5 to float
  call void @dx.op.discard(i32 82, i1 true)
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 0, float %1)
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 1, float %3)
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 2, float %9)
  call void @dx.op.storeOutput.f32(i32 5, i32 1, i32 0, i8 0, float %2)
  call void @dx.op.storeOutput.i32(i32 5, i32 2, i32 0, i8 0, i32 %6)
  call void @dx.op.storeOutput.i32(i32 5, i32 3, i32 0, i8 0, i32 %7)
  call void @dx.op.storeOutput.i32(i32 5, i32 3, i32 0, i8 1, i32 %8)
  ret void
"
        );
        // position xyz, B unread, front face, primitive id; SV_SampleIndex is a shadow element
        assert_eq!(masks(&info.inputs), [7, 0, 1, 1, 0]);
        assert_eq!(info.inputs[4].start_row, -1);
        // the undefined w is not stored and not in the mask
        assert_eq!(masks(&info.outputs), [7, 1, 1, 3]);
        assert_eq!(info.props, StageProps::Pixel { early_depth_stencil: false });
    }

    /// Scratch snippet ps_eval: the three evaluation operations name the element, row and
    /// column like a load, and do not count as reads of it.
    #[test]
    fn evaluate_attribute() {
        let src = "struct I { float4 pos : SV_Position; float4 a : A; noperspective float2 b : B; sample float3 c : C; };
float4 main(I i) : SV_Target { return i.a; }";
        let (mut io, mut e) = open(src, ShaderKind::Pixel);
        let c = io.eval_attribute(&mut e, Eval::Centroid, 0, 6, 2, &[]).unwrap();
        let two = e.c_u32(2);
        let s = io.eval_attribute(&mut e, Eval::AtSample, 0, 8, 2, &[two]).unwrap();
        let (x, y) = (e.c_u32(1), e.c_u32(-3i32 as u32));
        let n = io.eval_attribute(&mut e, Eval::Snapped, 0, 12, 1, &[x, y]).unwrap();
        io.output(&mut e, Target::Return, &[c[0], c[1], s[1], n[0]]).unwrap();
        // a second argument where the intrinsic takes none
        assert!(io.eval_attribute(&mut e, Eval::Centroid, 0, 0, 1, &[two]).is_err());
        let (text, info) = close(io, e, ShaderKind::Pixel);
        assert_eq!(
            text,
            "  %1 = call float @dx.op.evalCentroid.f32(i32 89, i32 1, i32 0, i8 2), !dx.precise !{i32 1}
  %2 = call float @dx.op.evalCentroid.f32(i32 89, i32 1, i32 0, i8 3), !dx.precise !{i32 1}
  %3 = call float @dx.op.evalSampleIndex.f32(i32 88, i32 2, i32 0, i8 1, i32 2), !dx.precise !{i32 1}
  %4 = call float @dx.op.evalSnapped.f32(i32 87, i32 3, i32 0, i8 2, i32 1, i32 -3), !dx.precise !{i32 1}
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 0, float %1)
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 1, float %2)
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 2, float %3)
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 3, float %4)
  ret void
"
        );
        assert_eq!(masks(&info.inputs), [0, 0, 0, 0]);
    }

    /// Scratch snippets gs_pt and gs_streams: a vertex input is loaded with its vertex index
    /// (a value or a constant), the primitive and instance ids are operations, `Append` stores
    /// the element then emits on the parameter's stream, `RestartStrip` cuts it.
    #[test]
    fn geometry_streams() {
        let src = "struct A { float4 pos : SV_Position; float2 a : A; };
struct B { float3 b : B; uint c : C; bool d : D; };
[maxvertexcount(2)] [instance(3)]
void main(point A v[1], uint pid : SV_PrimitiveID, uint gi : SV_GSInstanceID, inout PointStream<A> s0, inout PointStream<B> s1) { }";
        let (mut io, mut e) = open(src, ShaderKind::Geometry);
        let kinds: Vec<ParamKind> = io.params().iter().map(|p| p.kind).collect();
        assert_eq!(kinds, [ParamKind::InputArray { len: 1 }, ParamKind::Input, ParamKind::Input, ParamKind::Stream, ParamKind::Stream]);
        let pid = io.input(&mut e, 1).unwrap();
        let gi = io.input(&mut e, 2).unwrap();
        let v = io.input_at(&mut e, 0, gi[0]).unwrap();
        assert_eq!(v.len(), 6);
        io.append(&mut e, 3, &[v[0], v[1], v[2], v[3], v[4], v[5]]).unwrap();
        let t = e.c_bool(true);
        io.append(&mut e, 4, &[v[0], v[1], v[2], pid[0], t]).unwrap();
        io.restart_strip(&mut e, 4).unwrap();
        let (text, info) = close(io, e, ShaderKind::Geometry);
        assert_eq!(
            text,
            "  %1 = call i32 @dx.op.primitiveID.i32(i32 108)
  %2 = call i32 @dx.op.gsInstanceID.i32(i32 100)
  %3 = call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 0, i32 %2), !dx.precise !{i32 1}
  %4 = call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 1, i32 %2), !dx.precise !{i32 1}
  %5 = call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 2, i32 %2), !dx.precise !{i32 1}
  %6 = call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 3, i32 %2), !dx.precise !{i32 1}
  %7 = call float @dx.op.loadInput.f32(i32 4, i32 1, i32 0, i8 0, i32 %2), !dx.precise !{i32 1}
  %8 = call float @dx.op.loadInput.f32(i32 4, i32 1, i32 0, i8 1, i32 %2), !dx.precise !{i32 1}
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 0, float %3)
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 1, float %4)
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 2, float %5)
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 3, float %6)
  call void @dx.op.storeOutput.f32(i32 5, i32 1, i32 0, i8 0, float %7)
  call void @dx.op.storeOutput.f32(i32 5, i32 1, i32 0, i8 1, float %8)
  call void @dx.op.emitStream(i32 97, i8 0)
  call void @dx.op.storeOutput.f32(i32 5, i32 2, i32 0, i8 0, float %3)
  call void @dx.op.storeOutput.f32(i32 5, i32 2, i32 0, i8 1, float %4)
  call void @dx.op.storeOutput.f32(i32 5, i32 2, i32 0, i8 2, float %5)
  call void @dx.op.storeOutput.i32(i32 5, i32 3, i32 0, i8 0, i32 %1)
  call void @dx.op.storeOutput.i32(i32 5, i32 4, i32 0, i8 0, i32 1)
  call void @dx.op.emitStream(i32 97, i8 1)
  call void @dx.op.cutStream(i32 98, i8 1)
  ret void
"
        );
        // the primitive id is a shadow element of the input signature: no register, not read
        assert_eq!(info.inputs.iter().map(|e| (e.start_row, e.usage_mask)).collect::<Vec<_>>(), [(0, 15), (1, 3), (-1, 0)]);
        assert_eq!(info.outputs.iter().map(|e| (e.stream, e.start_row, e.start_col)).collect::<Vec<_>>(), [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]);
        assert_eq!(
            info.props,
            StageProps::Geometry { input_primitive: InputPrimitive::Point, max_vertex_count: 2, active_streams: 3, stream_topology: PrimitiveTopology::PointList, instance_count: 3 }
        );
    }

    /// Scratch snippets ms_prim and ms_line: the thread ids are operations (one call per
    /// component), vertex and primitive outputs are stored with the array index last, whole or
    /// in part, and an index element is one `emitIndices` (0 for the third index of a line).
    #[test]
    fn mesh_outputs() {
        let src = "struct V { float4 pos : SV_Position; float3 a : A; };
struct P { uint pid : SV_PrimitiveID; bool cull : SV_CullPrimitive; };
[outputtopology(\"triangle\")] [numthreads(4, 2, 1)]
void main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID, uint2 dt : SV_DispatchThreadID, out vertices V v[6], out primitives P p[2], out indices uint3 t[2]) { }";
        let (mut io, mut e) = open(src, ShaderKind::Mesh);
        let gi = io.input(&mut e, 0).unwrap()[0];
        let gid = io.input(&mut e, 1).unwrap();
        let dt = io.input(&mut e, 2).unwrap();
        assert_eq!((gid.len(), dt.len()), (3, 2));
        io.set_mesh_output_counts(&mut e, gi, dt[1]).unwrap();
        let f = e.c_f32(2.0);
        io.mesh_store(&mut e, 3, gi, 0, &[f, f, f, f, f, f, f]).unwrap();
        let one = e.c_u32(1);
        // v[1].a.y = 2
        io.mesh_store(&mut e, 3, one, 5, &[f]).unwrap();
        let t = e.c_bool(true);
        io.mesh_store(&mut e, 4, gid[2], 0, &[gid[0], t]).unwrap();
        io.mesh_store(&mut e, 5, one, 0, &[gi, one, gid[1]]).unwrap();
        assert!(matches!(io.mesh_store(&mut e, 5, one, 1, &[gi]), Err(Error::Unsupported(_))));
        let (text, info) = close(io, e, ShaderKind::Mesh);
        assert_eq!(
            text,
            "  %1 = call i32 @dx.op.flattenedThreadIdInGroup.i32(i32 96)
  %2 = call i32 @dx.op.groupId.i32(i32 94, i32 0)
  %3 = call i32 @dx.op.groupId.i32(i32 94, i32 1)
  %4 = call i32 @dx.op.groupId.i32(i32 94, i32 2)
  %5 = call i32 @dx.op.threadId.i32(i32 93, i32 1)
  call void @dx.op.setMeshOutputCounts(i32 168, i32 %1, i32 %5)
  call void @dx.op.storeVertexOutput.f32(i32 171, i32 0, i32 0, i8 0, float 2.000000e+00, i32 %1)
  call void @dx.op.storeVertexOutput.f32(i32 171, i32 0, i32 0, i8 1, float 2.000000e+00, i32 %1)
  call void @dx.op.storeVertexOutput.f32(i32 171, i32 0, i32 0, i8 2, float 2.000000e+00, i32 %1)
  call void @dx.op.storeVertexOutput.f32(i32 171, i32 0, i32 0, i8 3, float 2.000000e+00, i32 %1)
  call void @dx.op.storeVertexOutput.f32(i32 171, i32 1, i32 0, i8 0, float 2.000000e+00, i32 %1)
  call void @dx.op.storeVertexOutput.f32(i32 171, i32 1, i32 0, i8 1, float 2.000000e+00, i32 %1)
  call void @dx.op.storeVertexOutput.f32(i32 171, i32 1, i32 0, i8 2, float 2.000000e+00, i32 %1)
  call void @dx.op.storeVertexOutput.f32(i32 171, i32 1, i32 0, i8 1, float 2.000000e+00, i32 1)
  call void @dx.op.storePrimitiveOutput.i32(i32 172, i32 0, i32 0, i8 0, i32 %2, i32 %4)
  call void @dx.op.storePrimitiveOutput.i32(i32 172, i32 1, i32 0, i8 0, i32 1, i32 %4)
  call void @dx.op.emitIndices(i32 169, i32 1, i32 %1, i32 1, i32 %3)
  ret void
"
        );
        assert_eq!(masks(&info.outputs), [15, 7]);
        assert_eq!(masks(&info.patch_or_prim), [1, 1]);
        assert_eq!(info.props, StageProps::Mesh { numthreads: [4, 2, 1], max_vertices: 6, max_primitives: 2, topology: MeshOutputTopology::Triangle, payload_bytes: 0 });

        let line = "struct V { float4 pos : SV_Position; };
[outputtopology(\"line\")] [numthreads(1, 1, 1)]
void main(out vertices V v[2], out indices uint2 t[1]) { }";
        let (mut io, mut e) = open(line, ShaderKind::Mesh);
        let (zero, one) = (e.c_u32(0), e.c_u32(1));
        io.set_mesh_output_counts(&mut e, one, one).unwrap();
        io.mesh_store(&mut e, 1, zero, 0, &[zero, one]).unwrap();
        let f = e.c_f32(0.0);
        io.mesh_store(&mut e, 0, one, 0, &[f, f, f, f]).unwrap();
        let (text, info) = close(io, e, ShaderKind::Mesh);
        assert_eq!(
            text,
            "  call void @dx.op.setMeshOutputCounts(i32 168, i32 1, i32 1)
  call void @dx.op.emitIndices(i32 169, i32 0, i32 0, i32 1, i32 0)
  call void @dx.op.storeVertexOutput.f32(i32 171, i32 0, i32 0, i8 0, float 0.000000e+00, i32 1)
  call void @dx.op.storeVertexOutput.f32(i32 171, i32 0, i32 0, i8 1, float 0.000000e+00, i32 1)
  call void @dx.op.storeVertexOutput.f32(i32 171, i32 0, i32 0, i8 2, float 0.000000e+00, i32 1)
  call void @dx.op.storeVertexOutput.f32(i32 171, i32 0, i32 0, i8 3, float 0.000000e+00, i32 1)
  ret void
"
        );
        assert_eq!(info.props, StageProps::Mesh { numthreads: [1, 1, 1], max_vertices: 2, max_primitives: 1, topology: MeshOutputTopology::Line, payload_bytes: 0 });

        // a system value output that no store writes in full does not validate
        let (mut io, mut e) = open(line, ShaderKind::Mesh);
        let (one, f) = (e.c_u32(1), e.c_f32(0.0));
        io.set_mesh_output_counts(&mut e, one, one).unwrap();
        io.mesh_store(&mut e, 0, one, 0, &[f, f, f]).unwrap();
        e.body.ret(e.bb);
        let mut info = ShaderInfo::new(ShaderKind::Mesh, "main", StageProps::Vertex);
        assert!(matches!(io.finish(&e.m, &e.body, &mut info), Err(Error::Source { msg, .. }) if msg.starts_with("Not all elements of output SV_Position")));

        // nor does a store without the SetMeshOutputCounts call
        let (mut io, mut e) = open(line, ShaderKind::Mesh);
        let (one, f) = (e.c_u32(1), e.c_f32(0.0));
        io.mesh_store(&mut e, 0, one, 0, &[f, f, f, f]).unwrap();
        e.body.ret(e.bb);
        let mut info = ShaderInfo::new(ShaderKind::Mesh, "main", StageProps::Vertex);
        assert!(matches!(io.finish(&e.m, &e.body, &mut info), Err(Error::Source { msg, .. }) if msg == "Missing SetMeshOutputCounts call."));
    }

    /// The output side of an `inout` parameter is declared before its input side, the return
    /// value before every parameter (DXC's argument order, which numbers the elements).
    #[test]
    fn vertex_inout_and_return() {
        let src = "float4 main(float3 p : POSITION, inout float2 w : W, out uint k : K) : SV_Position { return 0; }";
        let (mut io, mut e) = open(src, ShaderKind::Vertex);
        let p = io.input(&mut e, 0).unwrap();
        let w = io.input(&mut e, 1).unwrap();
        let k = e.c_u32(7);
        io.output(&mut e, Target::Return, &[p[0], p[1], p[2], w[0]]).unwrap();
        io.output(&mut e, Target::Param(1), &[w[1], w[1]]).unwrap();
        io.output(&mut e, Target::Param(2), &[k]).unwrap();
        assert!(io.output(&mut e, Target::Param(0), &[k]).is_err());
        let (text, info) = close(io, e, ShaderKind::Vertex);
        assert!(text.ends_with(
            "  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 3, float %4)
  call void @dx.op.storeOutput.f32(i32 5, i32 1, i32 0, i8 0, float %5)
  call void @dx.op.storeOutput.f32(i32 5, i32 1, i32 0, i8 1, float %5)
  call void @dx.op.storeOutput.i32(i32 5, i32 2, i32 0, i8 0, i32 7)
  ret void
"
        ));
        assert_eq!(info.outputs.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), ["SV_Position", "W", "K"]);
        assert_eq!(info.inputs.iter().map(|e| (e.name.as_str(), e.start_row, e.usage_mask)).collect::<Vec<_>>(), [("POSITION", 0, 7), ("W", 1, 3)]);
    }

    #[test]
    fn what_is_refused() {
        let unsupported = |src: &str, kind: ShaderKind| {
            let unit = parse::parse(src).unwrap();
            let entry = unit.funcs.iter().rfind(|f| unit.name(f.name) == "main").unwrap();
            matches!(begin(&mut Emit::new(), &unit, entry, kind), Err(Error::Unsupported(_)))
        };
        assert!(unsupported("struct P { float4 a; }; [numthreads(1, 1, 1)] void main(uint gi : SV_GroupIndex) { }", ShaderKind::Amplification));
        assert!(unsupported("float4 main(float4x4 m : M) : SV_Position { return 0; }", ShaderKind::Vertex));
        assert!(unsupported("float4 main(uint64_t m : M) : SV_Position { return 0; }", ShaderKind::Vertex));
        assert!(unsupported("float4 main(float4 m[2] : M) : SV_Position { return 0; }", ShaderKind::Vertex));
        assert!(unsupported("float4 main(float4 m : M) : SV_Position { return 0; }", ShaderKind::Hull));
        assert!(unsupported("float4 main(float4 m : M) : SV_Position { return 0; }", ShaderKind::Domain));
    }
}
