//! What a shader is besides its code: the contract between the front end (which fills it
//! in), [`crate::flags`], [`crate::meta`] (dx.* metadata) and [`crate::container`] (SFI0,
//! ISG1 / OSG1 / PSG1, PSV0). Field meanings follow DXC's DxilModule / DxilSignatureElement /
//! DxilResource, since the validator compares the container parts with what DXC would
//! serialise from the metadata.

use crate::consts::*;

/// DXIL version of the modules we write (SM 6.6 is DXIL 1.6).
pub const DXIL_VERSION: (u32, u32) = (1, 6);
/// The validator version the module claims (dx.valver) and the container parts are laid out
/// for (PSV0's runtime-info version follows it). 1.8: the oldest validator on the machines we
/// run on (Windows SDK 10.0.26100's dxil.dll); newer validators accept it.
pub const VALIDATOR_VERSION: (u32, u32) = (1, 8);

#[derive(Debug, Clone)]
pub struct ShaderInfo {
    pub kind: ShaderKind,
    pub entry: String,
    /// shader model, (6, 6)
    pub sm: (u32, u32),
    pub props: StageProps,
    /// register-bound resources (heap-indexed ones have no record), in declaration order;
    /// [`Resource::id`] counts within the class
    pub resources: Vec<Resource>,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    /// patch constant (HS out / DS in) or per-primitive (MS out) elements
    pub patch_or_prim: Vec<SigElement>,
    /// DXIL shader flags (`DxilShaderFlags` bits): set from [`crate::flags::collect`]
    pub flags: u64,
    /// DXC's serialized `DxilViewIdState` (what dx.viewIdState holds: the number of input
    /// scalars, then per output stream the number of output scalars, the outputs that depend
    /// on SV_ViewID when the shader reads it, and per input scalar a bit mask of the outputs it
    /// contributes to; then the same for patch constant / per-primitive outputs): set from
    /// [`crate::meta::view_id_state`] after the flags. Empty for compute shaders. PSV0's ViewID
    /// and input-to-output tables are these same words, and the validator compares the two.
    pub view_id_state: Vec<u32>,
    /// Bytes of groupshared memory: the sum of the allocation sizes (LLVM's
    /// `DataLayout::getTypeAllocSize`, so with array padding) of every global variable in
    /// address space 3, used or not. PSV0 of a mesh shader records it
    /// (`MSInfo::GroupSharedBytesUsed`) and the validator recomputes it from the module's
    /// globals; other stages do not need it. 0 without groupshared variables.
    pub groupshared_bytes: u32,
}

impl ShaderInfo {
    pub fn new(kind: ShaderKind, entry: &str, props: StageProps) -> ShaderInfo {
        ShaderInfo { kind, entry: entry.to_string(), sm: (6, 6), props, resources: Vec::new(), inputs: Vec::new(), outputs: Vec::new(), patch_or_prim: Vec::new(), flags: 0, view_id_state: Vec::new(), groupshared_bytes: 0 }
    }
    /// "cs", "vs", ...
    pub fn kind_name(&self) -> &'static str {
        match self.kind {
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
}

/// "cs_6_6" → (Compute, 6, 6).
pub fn parse_profile(profile: &str) -> Option<(ShaderKind, u32, u32)> {
    let mut p = profile.split('_');
    let kind = match p.next()? {
        "ps" => ShaderKind::Pixel,
        "vs" => ShaderKind::Vertex,
        "gs" => ShaderKind::Geometry,
        "hs" => ShaderKind::Hull,
        "ds" => ShaderKind::Domain,
        "cs" => ShaderKind::Compute,
        "ms" => ShaderKind::Mesh,
        "as" => ShaderKind::Amplification,
        _ => return None,
    };
    let major = p.next()?.parse().ok()?;
    let minor = p.next()?.parse().ok()?;
    p.next().is_none().then_some((kind, major, minor))
}

#[derive(Debug, Clone, PartialEq)]
pub enum StageProps {
    Vertex,
    Pixel {
        /// `[earlydepthstencil]`
        early_depth_stencil: bool,
    },
    Geometry {
        input_primitive: InputPrimitive,
        /// `[maxvertexcount(n)]`
        max_vertex_count: u32,
        /// bit per output stream in use
        active_streams: u32,
        stream_topology: PrimitiveTopology,
        /// `[instance(n)]`, 1 without
        instance_count: u32,
    },
    Hull {
        /// name of the patch constant function (a second defined function of the module)
        patch_constant_fn: String,
        input_control_points: u32,
        output_control_points: u32,
        domain: TessellatorDomain,
        partitioning: TessellatorPartitioning,
        output_primitive: TessellatorOutputPrimitive,
        max_tess_factor: f32,
    },
    Domain {
        domain: TessellatorDomain,
        input_control_points: u32,
    },
    Compute {
        numthreads: [u32; 3],
        /// `[WaveSize(n)]`
        wave_size: Option<u32>,
    },
    Mesh {
        numthreads: [u32; 3],
        max_vertices: u32,
        max_primitives: u32,
        topology: MeshOutputTopology,
        payload_bytes: u32,
    },
    Amplification {
        numthreads: [u32; 3],
        payload_bytes: u32,
    },
}

/// One signature element (`DxilSignatureElement`).
#[derive(Debug, Clone, PartialEq)]
pub struct SigElement {
    /// the semantic name without its index: "TEXCOORD", "SV_Position"
    pub name: String,
    /// the semantic index of each row
    pub indices: Vec<u32>,
    pub kind: SemanticKind,
    pub comp_type: ComponentType,
    pub interp: InterpolationMode,
    pub rows: u32,
    pub cols: u32,
    /// the register (first row); -1: not allocated (SV_Depth, SV_Coverage, ...)
    pub start_row: i32,
    pub start_col: i32,
    /// GS output stream
    pub stream: u32,
    /// components the shader reads (inputs) or writes (outputs), bit per column from
    /// start_col 0: DXC's "usage mask"
    pub usage_mask: u8,
    /// components accessed with a dynamic row index
    pub dynamic_index_mask: u8,
}

/// One register-bound resource (`DxilResourceBase` / `DxilResource` / `DxilSampler` /
/// `DxilCBuffer`).
#[derive(Debug, Clone, PartialEq)]
pub struct Resource {
    pub class: ResourceClass,
    /// index within its class, from 0 in declaration order
    pub id: u32,
    /// the HLSL name
    pub name: String,
    pub space: u32,
    pub lower_bound: u32,
    /// registers in the range; u32::MAX: unbounded
    pub range_size: u32,
    /// `ResourceKind::Sampler` for samplers, `CBuffer` for constant buffers
    pub kind: ResourceKind,
    /// element component type of a typed resource (`Invalid` for raw / structured buffers,
    /// samplers and constant buffers)
    pub comp_type: ComponentType,
    /// components of the element type (1..4; 0 where there is no element type)
    pub comp_count: u32,
    /// multisampled textures (0: unknown)
    pub sample_count: u32,
    /// bytes per element of a structured buffer
    pub struct_stride: u32,
    pub globally_coherent: bool,
    pub has_counter: bool,
    pub rov: bool,
    pub sampler_kind: SamplerKind,
    pub cbuffer_bytes: u32,
    /// a 64-bit atomic operation (`dx.op.atomicBinOp.i64` / `atomicCompareExchange.i64`) uses
    /// a handle of this UAV (DXC's `HasAtomic64Use`: dx.resources tag 3 and PSV0's
    /// `UsedByAtomic64` resource flag): set from [`crate::flags::mark_atomic64`]
    pub atomic64_use: bool,
}

impl Resource {
    pub fn new(class: ResourceClass, id: u32, name: &str, kind: ResourceKind, lower_bound: u32, space: u32) -> Resource {
        Resource {
            class,
            id,
            name: name.to_string(),
            space,
            lower_bound,
            range_size: 1,
            kind,
            comp_type: ComponentType::Invalid,
            comp_count: 0,
            sample_count: 0,
            struct_stride: 0,
            globally_coherent: false,
            has_counter: false,
            rov: false,
            sampler_kind: SamplerKind::Default,
            cbuffer_bytes: 0,
            atomic64_use: false,
        }
    }
}
