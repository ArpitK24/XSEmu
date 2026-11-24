//! DXIL shader flags, as DXC's `ShaderFlags::CollectShaderFlags` (lib/DXIL/DxilShaderFlags.cpp) and
//! `DxilModule::CollectShaderFlagsForModule` compute them for validator version 1.8. The
//! validator recomputes the flags from the module it is given and compares them with the ones
//! in dx.entryPoints, and SFI0 / PSV0 are derived from them, so this has to see exactly what
//! the validator sees: the instructions [`crate::bcwriter`] writes (its liveness), with the
//! quirks of DXC's scan kept (they are named where they are reproduced).
//!
//! Owner: the metadata subagent.

use crate::bcwriter::{liveness, Live};
use crate::consts::*;
use crate::ir::*;
use crate::shader::{Resource, ShaderInfo, DXIL_VERSION, VALIDATOR_VERSION};

/// Bits of the raw flags word (`ShaderFlags`'s bit-field order, DxilShaderFlags.h): what
/// [`collect`] returns, [`ShaderInfo::flags`] holds and dx.entryPoints carries.
pub mod bit {
    pub const DISABLE_OPTIMIZATIONS: u64 = 1 << 0;
    pub const DISABLE_MATH_REFACTORING: u64 = 1 << 1;
    pub const ENABLE_DOUBLE_PRECISION: u64 = 1 << 2;
    /// Not part of the collected flags: [`crate::meta`] sets it in the metadata of a pixel
    /// shader with `[earlydepthstencil]`, as DXC's EmitDxilEntryProperties does.
    pub const FORCE_EARLY_DEPTH_STENCIL: u64 = 1 << 3;
    pub const ENABLE_RAW_AND_STRUCTURED_BUFFERS: u64 = 1 << 4;
    pub const LOW_PRECISION_PRESENT: u64 = 1 << 5;
    pub const ENABLE_DOUBLE_EXTENSIONS: u64 = 1 << 6;
    pub const ENABLE_MSAD: u64 = 1 << 7;
    pub const ALL_RESOURCES_BOUND: u64 = 1 << 8;
    pub const VIEWPORT_AND_RT_ARRAY_INDEX: u64 = 1 << 9;
    pub const INNER_COVERAGE: u64 = 1 << 10;
    pub const STENCIL_REF: u64 = 1 << 11;
    pub const TILED_RESOURCES: u64 = 1 << 12;
    pub const UAV_LOAD_ADDITIONAL_FORMATS: u64 = 1 << 13;
    pub const LEVEL9_COMPARISON_FILTERING: u64 = 1 << 14;
    pub const UAVS_64: u64 = 1 << 15;
    pub const UAVS_AT_EVERY_STAGE: u64 = 1 << 16;
    pub const CS_RAW_AND_STRUCTURED_VIA_SHADER_4X: u64 = 1 << 17;
    pub const ROVS: u64 = 1 << 18;
    pub const WAVE_OPS: u64 = 1 << 19;
    pub const INT64_OPS: u64 = 1 << 20;
    pub const VIEW_ID: u64 = 1 << 21;
    pub const BARYCENTRICS: u64 = 1 << 22;
    pub const USE_NATIVE_LOW_PRECISION: u64 = 1 << 23;
    pub const SHADING_RATE: u64 = 1 << 24;
    pub const RAYTRACING_TIER_1_1: u64 = 1 << 25;
    pub const SAMPLER_FEEDBACK: u64 = 1 << 26;
    pub const ATOMIC_INT64_ON_TYPED_RESOURCE: u64 = 1 << 27;
    pub const ATOMIC_INT64_ON_GROUP_SHARED: u64 = 1 << 28;
    pub const DERIVATIVES_IN_MESH_AND_AMP_SHADERS: u64 = 1 << 29;
    pub const RESOURCE_DESCRIPTOR_HEAP_INDEXING: u64 = 1 << 30;
    pub const SAMPLER_DESCRIPTOR_HEAP_INDEXING: u64 = 1 << 31;
    pub const ATOMIC_INT64_ON_HEAP_RESOURCE: u64 = 1 << 32;
    pub const RES_MAY_NOT_ALIAS: u64 = 1 << 33;
    pub const ADVANCED_TEXTURE_OPS: u64 = 1 << 34;
    pub const WRITEABLE_MSAA_TEXTURES: u64 = 1 << 35;
    pub const SAMPLE_CMP_GRADIENT_OR_BIAS: u64 = 1 << 37;
    pub const EXTENDED_COMMAND_INFO: u64 = 1 << 38;
    pub const LINEAR_ALGEBRA: u64 = 1 << 41;
}

/// What one pass over the module finds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Scan {
    /// the raw flags word ([`bit`])
    pub flags: u64,
    /// per [`ShaderInfo::resources`] entry: a 64-bit AtomicBinOp / AtomicCompareExchange uses
    /// a handle created from this binding (DXC's `DxilResource::HasAtomic64Use`, which it
    /// sets while collecting the flags)
    pub atomic64: Vec<bool>,
}

/// The flags of the module (see [`scan`]).
pub fn collect(m: &Module, info: &ShaderInfo) -> u64 {
    scan(m, info).flags
}

/// Set [`Resource::atomic64_use`] of every bound UAV a 64-bit atomic operation uses: dx.resources
/// (extended tag 3) and PSV0's resource flags carry it.
pub fn mark_atomic64(m: &Module, info: &mut ShaderInfo) {
    let s = scan(m, info);
    for (r, used) in info.resources.iter_mut().zip(s.atomic64) {
        r.atomic64_use |= used;
    }
}

fn at_least(v: (u32, u32), major: u32, minor: u32) -> bool {
    v.0 > major || (v.0 == major && v.1 >= minor)
}

/// `DxilResourceProperties` as its two dwords (what `dx.op.annotateHandle` carries).
#[derive(Copy, Clone, Default)]
struct Props(u32, u32);

impl Props {
    fn kind(self) -> u32 {
        self.0 & 0xFF
    }
    fn is_uav(self) -> bool {
        self.0 & 1 << 12 != 0
    }
    /// `DXIL::IsTyped`: a typed buffer or any texture
    fn is_typed(self) -> bool {
        (ResourceKind::Texture1D as u32..=ResourceKind::TypedBuffer as u32).contains(&self.kind())
    }
    fn comp_count(self) -> u32 {
        self.1 >> 8 & 0xFF
    }
    fn is_ms(self) -> bool {
        self.kind() == ResourceKind::Texture2DMS as u32 || self.kind() == ResourceKind::Texture2DMSArray as u32
    }
    /// `loadPropsFromResourceBase` for an SRV or UAV record (the path of the pre-6.6
    /// `dx.op.createHandle`)
    fn of_resource(r: &Resource) -> Props {
        let mut d0 = r.kind as u32;
        if r.class == ResourceClass::UAV {
            d0 |= 1 << 12 | (r.rov as u32) << 13 | (r.globally_coherent as u32) << 14 | (r.has_counter as u32) << 15;
        }
        let p = Props(d0, 0);
        let d1 = if r.kind == ResourceKind::StructuredBuffer {
            r.struct_stride
        } else if p.is_typed() {
            r.comp_type as u32 | r.comp_count << 8 | r.sample_count << 16
        } else {
            0
        };
        Props(d0, d1)
    }
}

fn upper_bound(r: &Resource) -> u32 {
    if r.range_size == u32::MAX {
        u32::MAX
    } else {
        r.lower_bound.wrapping_add(r.range_size).wrapping_sub(1)
    }
}

const OP_CREATE_HANDLE: u64 = OpCode::CreateHandle as u64;
const OP_ANNOTATE_HANDLE: u64 = OpCode::AnnotateHandle as u64;
const OP_CREATE_HANDLE_FROM_BINDING: u64 = OpCode::CreateHandleFromBinding as u64;
const OP_CREATE_HANDLE_FROM_HEAP: u64 = OpCode::CreateHandleFromHeap as u64;

/// What DXC's per-function scan accumulates (`CollectShaderFlags`'s locals).
#[derive(Default)]
struct Found {
    double: bool,
    double_ext: bool,
    int64: bool,
    low_precision: bool,
    wave: bool,
    lod_clamp: bool,
    check_access: bool,
    msad: bool,
    inner_coverage: bool,
    view_id: bool,
    multi_component_uav_load: bool,
    barycentrics: bool,
    feedback: bool,
    raytracing_1_1: bool,
    atomic64_typed: bool,
    atomic64_group_shared: bool,
    resource_heap: bool,
    sampler_heap: bool,
    atomic64_heap: bool,
    uavs: bool,
    advanced_texture_ops: bool,
    sample_cmp_grad_or_bias: bool,
    extended_command_info: bool,
    writeable_msaa: bool,
    derivatives: bool,
}

/// The scalar types the scan looks for, as bits.
const DOUBLE: u8 = 1;
const HALF: u8 = 2;
const I16: u8 = 4;
const I64: u8 = 8;
/// beside the type bits of a function's parameters: the function is a dx.op
const DX_OP: u8 = 0x80;

/// The bits of a type: the four types have fixed handles ([`Ty`]), so this is a lookup in a
/// word that holds four bits per handle.
#[inline(always)]
fn ty_class(t: Ty) -> u8 {
    const TABLE: u64 = (DOUBLE as u64) << (4 * Ty::F64.0) | (HALF as u64) << (4 * Ty::HALF.0) | (I16 as u64) << (4 * Ty::I16.0) | (I64 as u64) << (4 * Ty::I64.0);
    if t.0 < 16 {
        (TABLE >> (4 * t.0)) as u8 & 15
    } else {
        0
    }
}

/// What the scan does with a dx.op call, by opcode ([`OP_TABLE`]): most opcodes say nothing
/// about the flags, and a call of one is done with after one lookup.
const OP_LOOK: u8 = 1;
const OP_WAVE: u8 = 2;
const OP_FEEDBACK: u8 = 4;

const fn op_table() -> [u8; 320] {
    let mut t = [0u8; 320];
    // the opcodes [`Scanner::call`] has an arm for
    let look = [
        OpCode::CheckAccessFullyMapped,
        OpCode::Msad,
        OpCode::TextureLoad,
        OpCode::BufferLoad,
        OpCode::Fma,
        OpCode::InnerCoverage,
        OpCode::ViewID,
        OpCode::AllocateRayQuery,
        OpCode::AllocateRayQuery2,
        OpCode::GeometryIndex,
        OpCode::AttributeAtVertex,
        OpCode::AtomicBinOp,
        OpCode::AtomicCompareExchange,
        OpCode::SampleLevel,
        OpCode::SampleCmpLevelZero,
        OpCode::SampleGrad,
        OpCode::SampleCmpGrad,
        OpCode::Sample,
        OpCode::SampleBias,
        OpCode::SampleCmp,
        OpCode::SampleCmpBias,
        OpCode::DerivFineX,
        OpCode::DerivFineY,
        OpCode::DerivCoarseX,
        OpCode::DerivCoarseY,
        OpCode::CalculateLOD,
        OpCode::CreateHandleFromHeap,
        OpCode::CreateHandle,
        OpCode::AnnotateHandle,
        OpCode::TextureStoreSample,
        OpCode::SampleCmpLevel,
        OpCode::TextureGatherRaw,
        OpCode::StartVertexLocation,
        OpCode::StartInstanceLocation,
    ];
    let mut k = 0;
    while k < look.len() {
        t[look[k] as usize] |= OP_LOOK;
        k += 1;
    }
    // OP::IsDxilOpWave and OP::IsDxilOpFeedback (the experimental wave opcodes above 2^31 are
    // tested where the table is read)
    let mut op = 0;
    while op < t.len() {
        if matches!(op, 110..=123 | 135..=136 | 165..=167 | 222) {
            t[op] |= OP_WAVE;
        }
        if matches!(op, 174..=177) {
            t[op] |= OP_FEEDBACK;
        }
        op += 1;
    }
    t
}

static OP_TABLE: [u8; 320] = op_table();

struct Scanner<'a> {
    m: &'a Module,
    info: &'a ShaderInfo,
    body: &'a Body,
    live_blocks: &'a [bool],
    atomic64: &'a mut [bool],
    /// per function of the module: [`DX_OP`] for a dx.op declaration (DXC's OP::IsDxilOpFunc),
    /// and the type bits of its parameters, which are those of the arguments of a call
    dx_ops: &'a [u8],
    /// the body has forwarding, and operands are read through it
    forwarded: bool,
}

impl Scanner<'_> {
    #[inline(always)]
    fn resolved(&self, v: Value) -> Value {
        if self.forwarded {
            self.body.resolve(v)
        } else {
            v
        }
    }

    /// The type bits of an operand: of an instruction or a constant (a global or a function
    /// is a pointer, never one of the four scalar types looked for).
    #[inline(always)]
    fn class(&self, v: Value) -> u8 {
        let v = self.resolved(v);
        let i = (v.0 & 0x3fff_ffff) as usize;
        match v.0 >> 30 {
            0 => self.body.insts.get(i).map_or(0, |x| ty_class(x.ty)),
            1 => self.m.consts.get(i).map_or(0, |c| ty_class(c.ty)),
            _ => 0,
        }
    }

    fn int_arg(&self, args: &[Value], k: usize) -> Option<u64> {
        self.m.as_int(self.resolved(*args.get(k)?))
    }

    /// `isa<Constant>` of a call argument; an argument the call does not have reads the
    /// callee, as `getArgOperand` past the end does in DXC, and that is a constant.
    fn const_arg(&self, args: &[Value], k: usize) -> bool {
        args.get(k).is_none_or(|&v| !self.resolved(v).is_inst())
    }

    /// DXC's `FindCallToCreateHandle`: through phis (first incoming value) and selects (true
    /// value) to the call that made a handle.
    fn handle_call(&self, v: Value) -> Option<&Inst> {
        let mut v = self.resolved(v);
        // a phi cycle hangs DXC; give up instead
        for _ in 0..64 {
            let ValueKind::Inst(i) = v.kind() else { return None };
            let inst = self.body.insts.get(i as usize)?;
            v = match inst.op {
                Op::Call => return Some(inst),
                Op::Phi => {
                    let incoming = self.body.phis.get(inst.a.0 as usize)?;
                    // the writer drops incomings from unreachable blocks
                    let first = incoming.iter().find(|(_, from)| self.live_blocks.get(from.0 as usize).copied().unwrap_or(false))?;
                    self.resolved(first.0)
                }
                Op::Select => self.resolved(inst.b),
                _ => return None,
            };
        }
        None
    }

    /// `GetResourcePropertyFromHandleCall`.
    fn props_of(&self, call: Option<&Inst>) -> Props {
        let Some(call) = call else { return Props::default() };
        let args = self.body.operands_of(call);
        match self.int_arg(args, 0) {
            Some(OP_ANNOTATE_HANDLE) => {
                let Some(c) = args.get(2).and_then(|&v| self.m.as_const(self.resolved(v))) else { return Props::default() };
                match &c.c {
                    Const::Aggregate(f) if f.len() == 2 => Props(self.m.as_int(f[0]).unwrap_or(0) as u32, self.m.as_int(f[1]).unwrap_or(0) as u32),
                    _ => Props::default(),
                }
            }
            Some(OP_CREATE_HANDLE) => {
                let (Some(class), Some(id)) = (self.int_arg(args, 1), self.int_arg(args, 2)) else { return Props::default() };
                if class != ResourceClass::UAV as u64 && class != ResourceClass::SRV as u64 {
                    return Props::default();
                }
                self.info.resources.iter().find(|r| r.class as u64 == class && r.id as u64 == id).map_or(Props::default(), Props::of_resource)
            }
            _ => Props::default(),
        }
    }

    /// `GetResourceFromAnnotateHandle`: the bound UAV record behind an annotated handle that
    /// was created from a binding; None for a heap handle.
    fn bound_uav(&self, call: Option<&Inst>) -> Option<usize> {
        let call = call?;
        let args = self.body.operands_of(call);
        if self.int_arg(args, 0) != Some(OP_ANNOTATE_HANDLE) {
            return None;
        }
        let ValueKind::Inst(i) = self.resolved(*args.get(1)?).kind() else { return None };
        let create = self.body.insts.get(i as usize).filter(|x| x.op == Op::Call)?;
        let cargs = self.body.operands_of(create);
        if self.int_arg(cargs, 0) != Some(OP_CREATE_HANDLE_FROM_BINDING) {
            return None;
        }
        // %dx.types.ResBind { lower bound, upper bound, space, class }
        let bind = match &self.m.as_const(self.resolved(*cargs.get(1)?))?.c {
            Const::Aggregate(f) if f.len() == 4 => [self.m.as_int(f[0])?, self.m.as_int(f[1])?, self.m.as_int(f[2])?, self.m.as_int(f[3])?],
            Const::Zero => [0; 4],
            _ => return None,
        };
        self.info.resources.iter().position(|r| {
            r.class == ResourceClass::UAV && bind[3] == ResourceClass::UAV as u64 && r.space as u64 == bind[2] && r.lower_bound as u64 == bind[0] && upper_bound(r) as u64 == bind[1]
        })
    }

    /// `hasSampleClamp`: the last argument is a float constant other than zero, or not a
    /// constant float and not undef.
    fn sample_clamp(&self, args: &[Value]) -> bool {
        let Some(&last) = args.last() else { return false };
        match self.m.as_const(self.resolved(last)) {
            Some(c) => match c.c {
                Const::Float(bits) => bits << (64 - self.m.bits(c.ty).clamp(1, 64)) << 1 != 0,
                Const::Undef => false,
                _ => true,
            },
            None => true,
        }
    }

    fn sample_offsets_dynamic(&self, args: &[Value]) -> bool {
        !(self.const_arg(args, 7) && self.const_arg(args, 8) && self.const_arg(args, 9))
    }

    /// One pass over the live instructions in block order (the order matters to
    /// `sample_cmp_grad_or_bias`): the type bits of the instruction and of its operands, then
    /// for a dx.op call what its opcode asks for.
    fn run(&mut self, live_insts: &[bool]) -> Found {
        let mut f = Found::default();
        let body = self.body;
        for (b, block) in body.blocks.iter().enumerate() {
            if !self.live_blocks[b] {
                continue;
            }
            for list in [&block.phis, &block.insts] {
                for &i in list {
                    if !live_insts[i as usize] {
                        continue;
                    }
                    let inst = &body.insts[i as usize];
                    let mut seen = ty_class(inst.ty);
                    match inst.op {
                        Op::Bin(_) | Op::ICmp(_) | Op::FCmp(_) | Op::Store | Op::AtomicRmw(_) => seen |= self.class(inst.a) | self.class(inst.b),
                        Op::Cast(_) | Op::Load | Op::ExtractValue | Op::CondBr | Op::Switch => seen |= self.class(inst.a),
                        Op::Select | Op::CmpXchg => seen |= self.class(inst.a) | self.class(inst.b) | self.class(inst.c),
                        Op::Call => {
                            // DXC looks at no call but those of dx.op functions (llvm.lifetime.start
                            // takes an i64 and is no 64-bit arithmetic)
                            let callee = if inst.a.0 >> 30 == 3 { self.dx_ops.get((inst.a.0 & 0x3fff_ffff) as usize).copied().unwrap_or(0) } else { 0 };
                            if callee & DX_OP == 0 {
                                continue;
                            }
                            // an argument has the type of its parameter
                            seen |= callee & !DX_OP;
                            self.call(inst, body.operands_of(inst), seen & DOUBLE != 0, seen & I64 != 0, &mut f);
                        }
                        Op::Gep => {
                            for &v in body.operands_of(inst) {
                                seen |= self.class(v);
                            }
                        }
                        // the incoming values have the phi's type
                        Op::Phi | Op::Alloca | Op::Br | Op::Ret | Op::Unreachable => {}
                    }
                    if seen == 0 {
                        continue;
                    }
                    if seen & DOUBLE != 0 {
                        f.double = true;
                        // ddiv, d2i, d2u, i2d, u2d
                        f.double_ext |= matches!(inst.op, Op::Bin(BinOp::FDiv) | Op::Cast(CastOp::UIToFP | CastOp::SIToFP | CastOp::FPToUI | CastOp::FPToSI));
                    }
                    if seen & I64 != 0 {
                        f.int64 = true;
                        // the LLVM atomics only address groupshared memory in DXIL
                        f.atomic64_group_shared |= matches!(inst.op, Op::CmpXchg | Op::AtomicRmw(_));
                    }
                    f.low_precision |= seen & (HALF | I16) != 0;
                }
            }
        }
        f
    }

    #[inline(always)]
    fn call(&mut self, inst: &Inst, args: &[Value], is_double: bool, is_int64: bool, f: &mut Found) {
        let Some(op) = self.int_arg(args, 0) else { return };
        let Some(&what) = usize::try_from(op).ok().and_then(|op| OP_TABLE.get(op)) else {
            f.wave |= matches!(op, 2147483649..=2147483650);
            return;
        };
        if what != 0 {
            f.wave |= what & OP_WAVE != 0;
            f.feedback |= what & OP_FEEDBACK != 0;
            if what & OP_LOOK != 0 {
                self.look(inst, args, op, is_double, is_int64, f);
            }
        }
    }

    /// A call of one of the opcodes the flags depend on.
    #[inline(never)]
    fn look(&mut self, inst: &Inst, args: &[Value], op: u64, is_double: bool, is_int64: bool, f: &mut Found) {
        const TEXTURE_LOAD: u64 = OpCode::TextureLoad as u64;
        const BUFFER_LOAD: u64 = OpCode::BufferLoad as u64;
        match op {
            x if x == OpCode::CheckAccessFullyMapped as u64 => f.check_access = true,
            x if x == OpCode::Msad as u64 => f.msad = true,
            TEXTURE_LOAD | BUFFER_LOAD => {
                // kTextureLoadOffset{0,1,2}OpIdx are 6, 8 and 9 in DXC (not 6, 7, 8): the middle
                // offset is never looked at, and 9 is past the arguments
                if op == TEXTURE_LOAD && !(self.const_arg(args, 6) && self.const_arg(args, 8) && self.const_arg(args, 9)) {
                    f.advanced_texture_ops = true;
                }
                if !f.multi_component_uav_load {
                    if let Some(call) = args.get(1).and_then(|&h| self.handle_call(h)) {
                        let p = self.props_of(Some(call));
                        f.multi_component_uav_load = p.is_uav() && p.is_typed() && p.comp_count() > 1;
                    }
                }
            }
            x if x == OpCode::Fma as u64 => f.double_ext |= is_double,
            x if x == OpCode::InnerCoverage as u64 => f.inner_coverage = true,
            x if x == OpCode::ViewID as u64 => f.view_id = true,
            x if x == OpCode::AllocateRayQuery as u64 || x == OpCode::AllocateRayQuery2 as u64 || x == OpCode::GeometryIndex as u64 => f.raytracing_1_1 = true,
            x if x == OpCode::AttributeAtVertex as u64 => f.barycentrics = true,
            x if (x == OpCode::AtomicBinOp as u64 || x == OpCode::AtomicCompareExchange as u64) && is_int64 => {
                let call = args.get(1).and_then(|&h| self.handle_call(h));
                f.atomic64_typed |= self.props_of(call).is_typed();
                match self.bound_uav(call) {
                    Some(r) => self.atomic64[r] = true,
                    // not created from a binding: a descriptor heap handle
                    None => f.atomic64_heap = true,
                }
            }
            x if x == OpCode::SampleLevel as u64 || x == OpCode::SampleCmpLevelZero as u64 => f.advanced_texture_ops |= self.sample_offsets_dynamic(args),
            x if x == OpCode::SampleGrad as u64 || x == OpCode::SampleCmpGrad as u64 => {
                f.advanced_texture_ops |= self.sample_offsets_dynamic(args);
                f.lod_clamp |= self.sample_clamp(args);
                // an assignment in DXC: the last sampling call of the function decides
                f.sample_cmp_grad_or_bias = x == OpCode::SampleCmpGrad as u64;
            }
            x if x == OpCode::Sample as u64 || x == OpCode::SampleBias as u64 || x == OpCode::SampleCmp as u64 || x == OpCode::SampleCmpBias as u64 => {
                f.advanced_texture_ops |= self.sample_offsets_dynamic(args);
                f.lod_clamp |= self.sample_clamp(args);
                f.sample_cmp_grad_or_bias = x == OpCode::SampleCmpBias as u64;
                f.derivatives = true;
            }
            x if x == OpCode::DerivFineX as u64 || x == OpCode::DerivFineY as u64 || x == OpCode::DerivCoarseX as u64 || x == OpCode::DerivCoarseY as u64 || x == OpCode::CalculateLOD as u64 => {
                f.derivatives = true
            }
            OP_CREATE_HANDLE_FROM_HEAP => {
                if self.int_arg(args, 2).unwrap_or(0) != 0 {
                    f.sampler_heap = true;
                } else {
                    // DXC also asks whether the handle is a UAV here, of a call that carries no
                    // properties: never
                    f.resource_heap = true;
                }
            }
            OP_CREATE_HANDLE | OP_ANNOTATE_HANDLE => {
                let p = self.props_of(Some(inst));
                if p.is_uav() {
                    f.uavs = true;
                    f.writeable_msaa |= p.is_ms();
                }
            }
            x if x == OpCode::TextureStoreSample as u64 => {
                f.writeable_msaa = true;
                f.advanced_texture_ops = true;
            }
            x if x == OpCode::SampleCmpLevel as u64 || x == OpCode::TextureGatherRaw as u64 => f.advanced_texture_ops = true,
            x if x == OpCode::StartVertexLocation as u64 || x == OpCode::StartInstanceLocation as u64 => f.extended_command_info = true,
            _ => {}
        }
    }
}

/// DXC's OP::IsDxilOpFunc of the callee of a call: by its name, as DXC tells.
pub(crate) fn calls_dx_op(m: &Module, call: &Inst) -> bool {
    matches!(call.a.kind(), ValueKind::Func(f) if m.funcs.get(f as usize).is_some_and(|f| f.name.starts_with("dx.op.")))
}

/// One pass over the live instructions of every defined function, and the resource and
/// signature rules of `CollectShaderFlagsForModule`.
pub fn scan(m: &Module, info: &ShaderInfo) -> Scan {
    scan_live(m, info, &[])
}

/// [`scan`] of a sealed module, with the liveness it holds.
pub fn scan_sealed(s: &Sealed, info: &ShaderInfo) -> Scan {
    scan_live(s.module(), info, s.lives())
}

/// `lives`: the liveness of the bodies by function index; computed here for a body without one.
fn scan_live(m: &Module, info: &ShaderInfo, lives: &[Option<Live>]) -> Scan {
    use bit::*;
    let val_1_8 = at_least(VALIDATOR_VERSION, 1, 8);
    let mut atomic64 = vec![false; info.resources.len()];
    let mut flags = 0u64;
    let mut derivatives = false;
    let dx_ops: Vec<u8> = m
        .funcs
        .iter()
        .map(|f| {
            if !f.name.starts_with("dx.op.") {
                return 0;
            }
            match m.types.get(f.ty.0 as usize) {
                Some(TypeDef::Func { params, .. }) => params.iter().fold(DX_OP, |bits, &p| bits | ty_class(p)),
                _ => DX_OP,
            }
        })
        .collect();
    for (i, func) in m.funcs.iter().enumerate() {
        let Some(body) = &func.body else { continue };
        let own;
        let live = match lives.get(i).and_then(|l| l.as_ref()) {
            Some(live) => live,
            None => {
                own = liveness(body);
                &own
            }
        };
        let f = Scanner { m, info, body, live_blocks: &live.blocks, atomic64: &mut atomic64, dx_ops: &dx_ops, forwarded: body.has_replacements() }.run(&live.insts);
        derivatives |= f.derivatives;
        let set: [(bool, u64); 22] = [
            (f.double, ENABLE_DOUBLE_PRECISION),
            (f.double_ext, ENABLE_DOUBLE_EXTENSIONS),
            (f.int64, INT64_OPS),
            (f.low_precision, LOW_PRECISION_PRESENT),
            (f.wave, WAVE_OPS),
            // a validator before 1.8 did not know that a LOD clamp needs tiled resources
            (f.check_access || (val_1_8 && f.lod_clamp), TILED_RESOURCES),
            (f.msad, ENABLE_MSAD),
            (f.inner_coverage, INNER_COVERAGE),
            (f.view_id, VIEW_ID),
            (f.multi_component_uav_load, UAV_LOAD_ADDITIONAL_FORMATS),
            (f.barycentrics, BARYCENTRICS),
            // not set before validator 1.9
            (f.feedback && at_least(VALIDATOR_VERSION, 1, 9), SAMPLER_FEEDBACK),
            (f.raytracing_1_1, RAYTRACING_TIER_1_1),
            (f.atomic64_typed, ATOMIC_INT64_ON_TYPED_RESOURCE),
            (f.atomic64_group_shared, ATOMIC_INT64_ON_GROUP_SHARED),
            (f.resource_heap, RESOURCE_DESCRIPTOR_HEAP_INDEXING),
            (f.sampler_heap, SAMPLER_DESCRIPTOR_HEAP_INDEXING),
            (f.atomic64_heap, ATOMIC_INT64_ON_HEAP_RESOURCE),
            (f.advanced_texture_ops, ADVANCED_TEXTURE_OPS),
            (f.writeable_msaa, WRITEABLE_MSAA_TEXTURES),
            (f.sample_cmp_grad_or_bias, SAMPLE_CMP_GRADIENT_OR_BIAS),
            (f.extended_command_info, EXTENDED_COMMAND_INFO),
        ];
        flags |= set.iter().filter(|(on, _)| *on).fold(0, |a, (_, b)| a | b);
        // DXIL before 1.7 has no such flag (drivers reject unknown ones); resources may not
        // alias unless -res-may-alias, which XCL does not pass
        if f.uavs && at_least(DXIL_VERSION, 1, 7) {
            flags |= RES_MAY_NOT_ALIAS;
        }
    }

    // the signatures of the entry point
    let kind = info.kind;
    let in_rt_index = matches!(kind, ShaderKind::Geometry | ShaderKind::Domain | ShaderKind::Hull);
    let out_rt_index = matches!(kind, ShaderKind::Vertex | ShaderKind::Domain | ShaderKind::Hull);
    for e in &info.inputs {
        match e.kind {
            SemanticKind::ViewPortArrayIndex | SemanticKind::RenderTargetArrayIndex if in_rt_index => flags |= VIEWPORT_AND_RT_ARRAY_INDEX,
            SemanticKind::ShadingRate => flags |= SHADING_RATE,
            SemanticKind::Barycentrics => flags |= BARYCENTRICS,
            _ => {}
        }
    }
    for e in &info.outputs {
        match e.kind {
            SemanticKind::ViewPortArrayIndex | SemanticKind::RenderTargetArrayIndex if out_rt_index => flags |= VIEWPORT_AND_RT_ARRAY_INDEX,
            SemanticKind::StencilRef if kind == ShaderKind::Pixel => flags |= STENCIL_REF,
            SemanticKind::InnerCoverage if kind == ShaderKind::Pixel => flags |= INNER_COVERAGE,
            SemanticKind::ShadingRate => flags |= SHADING_RATE,
            _ => {}
        }
    }
    if derivatives && matches!(kind, ShaderKind::Mesh | ShaderKind::Amplification) {
        flags |= DERIVATIVES_IN_MESH_AND_AMP_SHADERS;
    }

    // the resource records
    let mut uavs = 0u32;
    for r in &info.resources {
        let raw = matches!(r.kind, ResourceKind::RawBuffer | ResourceKind::StructuredBuffer);
        match r.class {
            ResourceClass::UAV => {
                uavs += r.range_size.min(9);
                if r.rov {
                    flags |= ROVS;
                }
                if raw {
                    flags |= ENABLE_RAW_AND_STRUCTURED_BUFFERS;
                }
            }
            ResourceClass::SRV if raw => flags |= ENABLE_RAW_AND_STRUCTURED_BUFFERS,
            _ => {}
        }
    }
    if uavs > 8 {
        flags |= UAVS_64;
    }
    if uavs != 0 && matches!(kind, ShaderKind::Vertex | ShaderKind::Hull | ShaderKind::Domain | ShaderKind::Geometry) {
        flags |= UAVS_AT_EVERY_STAGE;
    }
    Scan { flags, atomic64 }
}

/// Small modules for the tests of this file and of [`crate::meta`].
#[cfg(test)]
pub(crate) mod testing {
    use crate::consts::*;
    use crate::dxop;
    use crate::ir::*;

    pub struct Builder {
        pub m: Module,
        pub body: Body,
    }

    impl Builder {
        pub fn new() -> Builder {
            Builder { m: Module::new(), body: Body::new() }
        }
        pub fn i32(&mut self, v: u32) -> Value {
            self.m.const_i32(v)
        }
        pub fn op(&mut self, bb: Block, opcode: OpCode, overload: Ty, args: &[Value]) -> Value {
            dxop::try_call(&mut self.m, &mut self.body, bb, opcode, overload, args, true).unwrap_or_else(|e| panic!("{opcode:?}: {e}"))
        }
        /// `dx.op.annotateHandle` with `DxilResourceProperties` dwords.
        pub fn annotate(&mut self, bb: Block, handle: Value, props: (u32, u32)) -> Value {
            let ty = dxop::res_props_ty(&mut self.m);
            let fields = vec![self.i32(props.0), self.i32(props.1)];
            let props = self.m.aggregate(ty, fields);
            self.op(bb, OpCode::AnnotateHandle, Ty::VOID, &[handle, props])
        }
        /// An annotated handle of the binding `lower..=upper` in `space`.
        pub fn bound(&mut self, bb: Block, class: ResourceClass, lower: u32, upper: u32, space: u32, props: (u32, u32)) -> Value {
            let ty = dxop::res_bind_ty(&mut self.m);
            let fields = vec![self.i32(lower), self.i32(upper), self.i32(space), self.m.const_i8(class as u8)];
            let bind = self.m.aggregate(ty, fields);
            let (index, uniform) = (self.i32(lower), self.m.const_bool(false));
            let h = self.op(bb, OpCode::CreateHandleFromBinding, Ty::VOID, &[bind, index, uniform]);
            self.annotate(bb, h, props)
        }
        /// An annotated handle from the resource or sampler descriptor heap.
        pub fn heap(&mut self, bb: Block, index: Value, sampler: bool, props: (u32, u32)) -> Value {
            let (sampler, uniform) = (self.m.const_bool(sampler), self.m.const_bool(false));
            let h = self.op(bb, OpCode::CreateHandleFromHeap, Ty::VOID, &[index, sampler, uniform]);
            self.annotate(bb, h, props)
        }
        /// `io.Load(4 * k)` of a raw buffer handle.
        pub fn raw_load(&mut self, bb: Block, handle: Value, at: Value) -> Value {
            let (undef, mask, align) = (self.m.undef(Ty::I32), self.m.const_i8(1), self.i32(4));
            let r = self.op(bb, OpCode::RawBufferLoad, Ty::I32, &[handle, at, undef, mask, align]);
            self.body.extract(bb, Ty::I32, r, 0)
        }
        pub fn raw_store(&mut self, bb: Block, handle: Value, at: Value, v: Value) {
            let (undef, mask, align) = (self.m.undef(Ty::I32), self.m.const_i8(1), self.i32(4));
            self.op(bb, OpCode::RawBufferStore, Ty::I32, &[handle, at, undef, v, undef, undef, undef, mask, align]);
        }
        pub fn load_input(&mut self, bb: Block, id: u32, row: Value, col: u8) -> Value {
            let (id, col, undef) = (self.i32(id), self.m.const_i8(col), self.m.undef(Ty::I32));
            self.op(bb, OpCode::LoadInput, Ty::F32, &[id, row, col, undef])
        }
        pub fn store_output(&mut self, bb: Block, id: u32, row: Value, col: u8, v: Value) {
            let (id, col) = (self.i32(id), self.m.const_i8(col));
            self.op(bb, OpCode::StoreOutput, Ty::F32, &[id, row, col, v]);
        }
        /// Close `bb` with `ret void` and define the function.
        pub fn finish(mut self, bb: Block, name: &str) -> Module {
            self.body.ret(bb);
            self.m.define(name, self.body);
            self.m
        }
    }

    /// `DxilResourceProperties` of a raw UAV, and of a typed UAV texture.
    pub const RAW_UAV: (u32, u32) = (ResourceKind::RawBuffer as u32 | 1 << 12, 0);
    pub fn texture_uav(kind: ResourceKind, comp: ComponentType, count: u32) -> (u32, u32) {
        (kind as u32 | 1 << 12, comp as u32 | count << 8)
    }
}

#[cfg(test)]
mod tests {
    use super::bit::*;
    use super::testing::*;
    use super::*;
    use crate::shader::{SigElement, StageProps};

    fn compute() -> ShaderInfo {
        let mut info = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [1, 1, 1], wave_size: None });
        for u in 0..2 {
            info.resources.push(Resource::new(ResourceClass::UAV, u, "", ResourceKind::RawBuffer, u, 0));
        }
        info
    }

    /// The translator's compute shape: two raw UAVs, wave operations, 64-bit integers. Dead
    /// code is not in the module the validator sees.
    #[test]
    fn raw_buffers_wave_ops_and_int64() {
        let mut b = Builder::new();
        let bb = Body::ENTRY;
        let io = b.bound(bb, ResourceClass::UAV, 0, 0, 0, RAW_UAV);
        let zero = b.i32(0);
        let a = b.raw_load(bb, io, zero);
        let lane = b.op(bb, OpCode::WaveGetLaneIndex, Ty::VOID, &[]);
        let wide = b.body.cast(bb, CastOp::ZExt, Ty::I64, a);
        let lane64 = b.body.cast(bb, CastOp::ZExt, Ty::I64, lane);
        let product = b.body.bin(bb, BinOp::Mul, Ty::I64, wide, lane64);
        let low = b.body.cast(bb, CastOp::Trunc, Ty::I32, product);
        // never used: the writer drops it
        b.body.cast(bb, CastOp::UIToFP, Ty::F64, a);
        b.raw_store(bb, io, zero, low);
        let m = b.finish(bb, "main");
        // DXC gives the same shader (cs_basic of tests/meta_oracle.rs) 0x180010
        assert_eq!(collect(&m, &compute()), ENABLE_RAW_AND_STRUCTURED_BUFFERS | WAVE_OPS | INT64_OPS);
    }

    #[test]
    fn calls_of_other_functions_are_not_looked_at() {
        let mut b = Builder::new();
        let bb = Body::ENTRY;
        let io = b.bound(bb, ResourceClass::UAV, 0, 0, 0, RAW_UAV);
        let zero = b.i32(0);
        let a = b.raw_load(bb, io, zero);
        // as llvm.lifetime.start: an i64 operand, and here one that reads as WaveIsFirstLane
        let fty = b.m.func_ty(Ty::VOID, vec![Ty::I64]);
        let other = b.m.declare("llvm.something", fty, FnAttrs::Nounwind);
        let size = b.m.const_i64(OpCode::WaveIsFirstLane as u64);
        b.body.call(bb, Ty::VOID, other, &[size], true, 0);
        b.raw_store(bb, io, zero, a);
        assert_eq!(collect(&b.finish(bb, "main"), &compute()), ENABLE_RAW_AND_STRUCTURED_BUFFERS);
    }

    #[test]
    fn doubles_and_their_extensions() {
        let double = |convert: bool| {
            let mut b = Builder::new();
            let bb = Body::ENTRY;
            let io = b.bound(bb, ResourceClass::UAV, 0, 0, 0, RAW_UAV);
            let zero = b.i32(0);
            let a = b.raw_load(bb, io, zero);
            let low = if convert {
                // i2d, d2u: double extensions
                let d = b.body.cast(bb, CastOp::UIToFP, Ty::F64, a);
                b.body.cast(bb, CastOp::FPToUI, Ty::I32, d)
            } else {
                let d = b.op(bb, OpCode::MakeDouble, Ty::F64, &[a, a]);
                let sum = b.body.bin(bb, BinOp::FAdd, Ty::F64, d, d);
                let parts = b.op(bb, OpCode::SplitDouble, Ty::F64, &[sum]);
                b.body.extract(bb, Ty::I32, parts, 0)
            };
            b.raw_store(bb, io, zero, low);
            collect(&b.finish(bb, "main"), &compute())
        };
        assert_eq!(double(false), ENABLE_RAW_AND_STRUCTURED_BUFFERS | ENABLE_DOUBLE_PRECISION);
        assert_eq!(double(true), ENABLE_RAW_AND_STRUCTURED_BUFFERS | ENABLE_DOUBLE_PRECISION | ENABLE_DOUBLE_EXTENSIONS);
    }

    /// 64-bit atomics on a bound raw buffer (marks the record), a heap buffer, a typed
    /// texture and groupshared memory: DXC's 0x158100010 for cs_atomic64.
    #[test]
    fn atomics_on_64_bit_integers() {
        let mut b = Builder::new();
        let bb = Body::ENTRY;
        let mut info = compute();
        let mut tex = Resource::new(ResourceClass::UAV, 2, "", ResourceKind::Texture2D, 2, 0);
        (tex.comp_type, tex.comp_count) = (ComponentType::U32, 1);
        info.resources.push(tex);
        let (zero, undef, add) = (b.i32(0), b.m.undef(Ty::I32), b.i32(0));
        let one = b.m.const_i64(1);
        let res = b.bound(bb, ResourceClass::UAV, 1, 1, 0, RAW_UAV);
        b.op(bb, OpCode::AtomicBinOp, Ty::I64, &[res, add, zero, undef, undef, one]);
        let mb = b.heap(bb, zero, false, RAW_UAV);
        b.op(bb, OpCode::AtomicBinOp, Ty::I64, &[mb, add, zero, undef, undef, one]);
        let tex = b.bound(bb, ResourceClass::UAV, 2, 2, 0, texture_uav(ResourceKind::Texture2D, ComponentType::U32, 1));
        b.op(bb, OpCode::AtomicCompareExchange, Ty::I64, &[tex, zero, zero, undef, one, one]);
        let shared = b.m.global("gs64", Ty::I64, 3, None, false);
        b.body.atomic_rmw(bb, RmwOp::Add, Ty::I64, shared, one);
        let m = b.finish(bb, "main");
        let s = scan(&m, &info);
        assert_eq!(s.flags, 0x1_5810_0010);
        assert_eq!(
            s.flags,
            ENABLE_RAW_AND_STRUCTURED_BUFFERS | INT64_OPS | ATOMIC_INT64_ON_TYPED_RESOURCE | ATOMIC_INT64_ON_GROUP_SHARED | RESOURCE_DESCRIPTOR_HEAP_INDEXING | ATOMIC_INT64_ON_HEAP_RESOURCE
        );
        assert_eq!(s.atomic64, [false, true, true]);
        mark_atomic64(&m, &mut info);
        assert!(info.resources[1].atomic64_use && !info.resources[0].atomic64_use);
    }

    /// A typed UAV load of more than one component needs the additional-formats feature; a
    /// dynamic texel offset is an advanced texture op, except the middle one, which DXC's scan
    /// never reads.
    #[test]
    fn typed_uav_loads_and_load_offsets() {
        let load = |offset: usize, count: u32| {
            let mut b = Builder::new();
            let bb = Body::ENTRY;
            let io = b.bound(bb, ResourceClass::UAV, 0, 0, 0, RAW_UAV);
            let zero = b.i32(0);
            let dynamic = b.raw_load(bb, io, zero);
            let tex = b.heap(bb, zero, false, texture_uav(ResourceKind::Texture2D, ComponentType::F32, count));
            let undef = b.m.undef(Ty::I32);
            let mut args = [tex, undef, zero, zero, undef, undef, undef, undef];
            if offset != 0 {
                args[offset] = dynamic;
            }
            let texel = b.op(bb, OpCode::TextureLoad, Ty::F32, &args);
            let x = b.body.extract(bb, Ty::F32, texel, 0);
            let bits = b.body.cast(bb, CastOp::BitCast, Ty::I32, x);
            b.raw_store(bb, io, zero, bits);
            collect(&b.finish(bb, "main"), &compute())
        };
        let base = ENABLE_RAW_AND_STRUCTURED_BUFFERS | RESOURCE_DESCRIPTOR_HEAP_INDEXING;
        assert_eq!(load(0, 1), base);
        assert_eq!(load(0, 4), base | UAV_LOAD_ADDITIONAL_FORMATS);
        // operands after the opcode: 5, 6, 7 are the offsets
        assert_eq!(load(5, 1), base | ADVANCED_TEXTURE_OPS);
        assert_eq!(load(6, 1), base);
        assert_eq!(load(7, 1), base | ADVANCED_TEXTURE_OPS);
    }

    /// Implicit derivatives need a feature in mesh and amplification shaders only; a LOD
    /// clamp needs tiled resources.
    #[test]
    fn samples_derivatives_and_clamps() {
        let sample = |kind: ShaderKind, clamp: Option<f32>| {
            let mut b = Builder::new();
            let bb = Body::ENTRY;
            let zero = b.i32(0);
            let tex = b.heap(bb, zero, false, (ResourceKind::Texture2D as u32, ComponentType::F32 as u32 | 4 << 8));
            let samp = b.heap(bb, zero, true, (ResourceKind::Sampler as u32, 0));
            let (uv, nof, noi) = (b.m.const_f32(0.5), b.m.undef(Ty::F32), b.m.undef(Ty::I32));
            let clamp = clamp.map_or(nof, |c| b.m.const_f32(c));
            b.op(bb, OpCode::Sample, Ty::F32, &[tex, samp, uv, uv, nof, nof, noi, noi, noi, clamp]);
            // sampling has no effect: keep it alive through a derivative that is stored
            let io = b.bound(bb, ResourceClass::UAV, 0, 0, 0, RAW_UAV);
            let texel = b.op(bb, OpCode::Sample, Ty::F32, &[tex, samp, uv, uv, nof, nof, noi, noi, noi, clamp]);
            let x = b.body.extract(bb, Ty::F32, texel, 0);
            let bits = b.body.cast(bb, CastOp::BitCast, Ty::I32, x);
            b.raw_store(bb, io, zero, bits);
            let mut info = compute();
            info.kind = kind;
            collect(&b.finish(bb, "main"), &info)
        };
        let base = ENABLE_RAW_AND_STRUCTURED_BUFFERS | RESOURCE_DESCRIPTOR_HEAP_INDEXING | SAMPLER_DESCRIPTOR_HEAP_INDEXING;
        assert_eq!(sample(ShaderKind::Compute, None), base);
        assert_eq!(sample(ShaderKind::Compute, Some(0.0)), base);
        assert_eq!(sample(ShaderKind::Compute, Some(2.0)), base | TILED_RESOURCES);
        assert_eq!(sample(ShaderKind::Mesh, None), base | DERIVATIVES_IN_MESH_AND_AMP_SHADERS);
    }

    /// The flags that come from the signatures and the resource records alone.
    #[test]
    fn signatures_and_resource_records() {
        let element = |kind: SemanticKind| SigElement {
            name: String::new(),
            indices: vec![0],
            kind,
            comp_type: ComponentType::U32,
            interp: InterpolationMode::Constant,
            rows: 1,
            cols: 1,
            start_row: 0,
            start_col: 0,
            stream: 0,
            usage_mask: 1,
            dynamic_index_mask: 0,
        };
        let m = Builder::new().finish(Body::ENTRY, "main");
        let mut vs = ShaderInfo::new(ShaderKind::Vertex, "main", StageProps::Vertex);
        vs.outputs.push(element(SemanticKind::RenderTargetArrayIndex));
        assert_eq!(collect(&m, &vs), VIEWPORT_AND_RT_ARRAY_INDEX);
        vs.resources.push(Resource::new(ResourceClass::UAV, 0, "", ResourceKind::RawBuffer, 0, 0));
        assert_eq!(collect(&m, &vs), VIEWPORT_AND_RT_ARRAY_INDEX | UAVS_AT_EVERY_STAGE | ENABLE_RAW_AND_STRUCTURED_BUFFERS);

        // a geometry shader's output index and a pixel shader's input index are no feature
        let mut ps = ShaderInfo::new(ShaderKind::Pixel, "main", StageProps::Pixel { early_depth_stencil: true });
        ps.inputs.push(element(SemanticKind::ViewPortArrayIndex));
        ps.outputs.push(element(SemanticKind::StencilRef));
        assert_eq!(collect(&m, &ps), STENCIL_REF);

        // nine registers of UAVs, an array counting at most nine
        let mut cs = compute();
        cs.resources[1].range_size = u32::MAX;
        cs.resources[1].kind = ResourceKind::Texture2D;
        assert_eq!(collect(&m, &cs), UAVS_64 | ENABLE_RAW_AND_STRUCTURED_BUFFERS);
        cs.resources[1].range_size = 7;
        cs.resources[0].rov = true;
        assert_eq!(collect(&m, &cs), ROVS | ENABLE_RAW_AND_STRUCTURED_BUFFERS);
    }
}
