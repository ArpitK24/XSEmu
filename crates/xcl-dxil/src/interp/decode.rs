//! [`ir::Module`] → [`Program`]: the entry function pre-decoded for the wave interpreter.
//!
//! What the decoder settles once so that execution never looks anything up:
//! * every value gets a *row* of the wave's register file (one `u64` per lane). Values used
//!   only inside the block piece that defines them share a small scratch area that is
//!   reused by every block; the others, the phis' inputs and the constants get rows of
//!   their own. `extractvalue`, `zext` and the int ↔ float bit casts are not instructions
//!   here: they name (part of) their operand's rows.
//! * blocks are renumbered into the schedule order of [`super::cfg`], and cut after every
//!   group-sync barrier and (in a group of several waves) after every atomic, which are the
//!   points where a wave can be suspended;
//! * memory is laid out in cells (one per scalar): a frame per thread for the allocas and
//!   the mutable globals, a read-only image of the constant globals, and one groupshared
//!   area per thread group. Pointers are a region tag and a cell index.

use super::cfg;
use crate::consts::OpCode;
use crate::ir::*;
use crate::{unsupported, Error, Result};
use std::collections::HashMap;

pub const NONE: u32 = u32::MAX;

/// Pointer encoding: the region in the top two bits, the cell index below.
pub const REGION_SHIFT: u32 = 62;
pub const REGION_FRAME: u64 = 0;
pub const REGION_CONST: u64 = 1;
pub const REGION_SHARED: u64 = 2;
pub const CELL_MASK: u64 = (1 << REGION_SHIFT) - 1;

/// Overload codes ([`DInst::w`] of a dx.op): the bit width, plus this flag for floats.
pub const OVL_FLOAT: u8 = 0x80;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Opc {
    Add,
    Sub,
    Mul,
    UDiv,
    SDiv,
    URem,
    SRem,
    Shl,
    LShr,
    AShr,
    And,
    Or,
    Xor,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
    /// aux = LLVM predicate
    ICmp,
    FCmp,
    Select,
    /// w = destination bits
    Trunc,
    /// w = source bits, aux = destination bits (for all the casts below)
    SExt,
    FPToUI,
    FPToSI,
    UIToFP,
    SIToFP,
    FPTrunc,
    FPExt,
    /// d = a + b cells (b is a signed immediate)
    GepC,
    /// d = a + index(b) · extra[c]; extra[c + 1] = the array bound; w = index bits
    GepI,
    /// a = pointer row
    Load,
    /// a = pointer row, b = value row
    Store,
    /// a = cell of the thread's frame (immediate)
    LoadFrame,
    /// a = cell (immediate), b = value row
    StoreFrame,
    /// aux = LLVM rmw operation; a = pointer, b = value
    AtomicRmw,
    /// a = pointer, b = expected, c = new; d, d + 1 = old value, success
    CmpXchg,
    /// aux = DXIL opcode, w = overload, a = first operand in extra (rows, or immediates
    /// where noted in exec), b = operand count, c = 1 when the call may write memory
    Dx,
    /// a = target block
    Br,
    /// a = condition row, b / c = target blocks
    CondBr,
    /// a = value row, b = default block, c = index in switches
    Switch,
    Ret,
    Unreachable,
    /// after a group-sync barrier: a = the block that continues
    Barrier,
    /// after an atomic, when the group has several waves: a = the block that continues
    Yield,
}

#[derive(Clone, Copy, Debug)]
pub struct DInst {
    pub op: Opc,
    pub w: u8,
    pub aux: u16,
    pub d: u32,
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DBlock {
    /// instructions [first, end); the last one is the terminator
    pub first: u32,
    pub end: u32,
    /// phis [phi_first, phi_first + nphis) in phi_dst; their sources are in phi_src, npreds
    /// per phi, in the order of preds [pred_first, pred_first + npreds)
    pub phi_first: u32,
    pub nphis: u32,
    pub pred_first: u32,
    pub npreds: u32,
    pub src_first: u32,
    /// a phi of this block reads another phi of this block: copy through a temporary
    pub parallel: bool,
    /// the last block of the loop this block heads (itself if it heads none)
    pub loop_end: u32,
}

/// What a createHandleFromBinding names: a register range of one class and space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Bind {
    pub lower: u32,
    pub upper: u32,
    pub space: u32,
    pub class: u32,
}

#[derive(Debug, Default, Clone)]
pub struct ProgramStats {
    pub ir_insts: usize,
    pub insts: usize,
    pub blocks: usize,
    pub loops: usize,
    pub persistent_rows: usize,
    pub scratch_rows: usize,
    pub const_rows: usize,
}

/// The stages the interpreter runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    Compute,
    Vertex,
    Pixel,
}

/// An element of the input signature, as far as generating its values needs it.
#[derive(Clone, Copy, Debug)]
pub struct SigElement {
    pub id: u32,
    /// DXIL SemanticKind
    pub kind: u8,
    /// DXIL InterpolationMode
    pub interp: u8,
}

pub struct Program {
    pub entry: String,
    pub stage: Stage,
    /// the input signature (not of a compute program)
    pub inputs: Vec<SigElement>,
    /// of a compute program; a vertex or pixel program runs as one wave of `wave_size`
    /// vertices or pixels, and this is [wave_size, 1, 1]
    pub numthreads: [u32; 3],
    pub wave_size: u32,
    /// rows of a wave's register file
    pub rows: u32,
    /// (row, value) of the constant rows
    pub row_init: Vec<(u32, u64)>,
    pub blocks: Vec<DBlock>,
    pub insts: Vec<DInst>,
    pub extra: Vec<u32>,
    pub phi_dst: Vec<u32>,
    pub phi_src: Vec<u32>,
    pub preds: Vec<u32>,
    pub max_phis: u32,
    /// sorted (case value, target block)
    pub switches: Vec<Vec<(u64, u32)>>,
    /// initial contents of a thread's frame
    pub frame_init: Vec<u64>,
    pub const_mem: Vec<u64>,
    pub shared_cells: u32,
    /// groupshared cell → an address that does not depend on the order of the globals: a
    /// hash of the global's name above the cell's offset in it
    pub shared_addr: Vec<u64>,
    pub binds: Vec<Bind>,
    /// dx.types.ResourceProperties of the annotateHandle calls
    pub props: Vec<[u32; 2]>,
    pub stats: ProgramStats,
}

impl Program {
    pub fn threads(&self) -> u32 {
        self.numthreads[0] * self.numthreads[1] * self.numthreads[2]
    }
    pub fn waves(&self) -> u32 {
        self.threads().div_ceil(self.wave_size)
    }
    /// (ran, all) for a [`super::Run::covered`] map, of the blocks, of the instructions, and
    /// of the stores to resources and outputs and the atomics. A block here is a basic
    /// block of the function, or a piece of one that a barrier or an atomic cuts.
    pub fn coverage_detail(&self, covered: &[bool]) -> [(usize, usize); 3] {
        let writes = |i: &DInst| match i.op {
            Opc::AtomicRmw | Opc::CmpXchg => true,
            Opc::Dx => [OpCode::StoreOutput, OpCode::TextureStore, OpCode::BufferStore, OpCode::BufferUpdateCounter, OpCode::AtomicBinOp, OpCode::AtomicCompareExchange, OpCode::RawBufferStore].iter().any(|op| *op as u16 == i.aux),
            _ => false,
        };
        let mut out = [(0, 0); 3];
        for (k, b) in self.blocks.iter().enumerate() {
            let insts = &self.insts[b.first as usize..b.end as usize];
            let counts = [1, insts.len(), insts.iter().filter(|i| writes(i)).count()];
            for (o, n) in out.iter_mut().zip(counts) {
                o.1 += n;
                if covered.get(k).copied().unwrap_or(false) {
                    o.0 += n;
                }
            }
        }
        out
    }
    /// (instructions in the blocks marked, instructions) for a [`super::Run::covered`] map.
    pub fn coverage(&self, covered: &[bool]) -> (usize, usize) {
        let ran = self.blocks.iter().zip(covered).filter(|(_, c)| **c).map(|(b, _)| (b.end - b.first) as usize).sum();
        (ran, self.insts.len())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FnKind {
    /// a dx.op function and the overload its name ends in
    Dx(u8),
    /// llvm.lifetime.start / end: nothing to do
    Lifetime,
    Other,
}

fn overload_of(name: &str) -> u8 {
    match name.rsplit('.').next() {
        Some("i1") => 1,
        Some("i8") => 8,
        Some("i16") => 16,
        Some("i32") => 32,
        Some("i64") => 64,
        Some("f16") => OVL_FLOAT | 16,
        Some("f32") => OVL_FLOAT | 32,
        Some("f64") => OVL_FLOAT | 64,
        _ => 0,
    }
}

struct Dec<'a> {
    m: &'a Module,
    body: &'a Body,
    fn_kind: Vec<FnKind>,
    /// several waves in a group: atomics are scheduling points
    yields: bool,
    /// cells of the global each index names, and its pointer
    global_ptr: Vec<u64>,
    /// instruction → rows it owns (NONE: an alias, a constant or nothing)
    row: Vec<u32>,
    /// instruction → its value when the decoder knows it (an alloca's address, a constant
    /// table entry)
    known: HashMap<u32, u64>,
    /// instruction → block piece that defines it
    piece: Vec<u32>,
    persistent: Vec<bool>,
    const_rows: HashMap<u64, u32>,
    next_const_row: u32,
    p: Program,
}

fn bad(msg: impl Into<String>) -> Error {
    Error::Internal(format!("interp: {}", msg.into()))
}

impl<'a> Dec<'a> {
    fn ty_of(&self, v: Value) -> Ty {
        match v.kind() {
            ValueKind::Inst(i) => self.body.insts[i as usize].ty,
            ValueKind::Const(i) => self.m.consts[i as usize].ty,
            ValueKind::Global(i) => self.m.globals[i as usize].ptr_ty,
            ValueKind::Func(_) => Ty::VOID,
        }
    }

    /// Bits of a scalar type, with [`OVL_FLOAT`] for floats; pointers count as 64-bit integers.
    fn scalar(&self, t: Ty) -> Result<u8> {
        Ok(match self.m.def(t) {
            TypeDef::Int(b) if *b <= 64 => *b as u8,
            TypeDef::Half => OVL_FLOAT | 16,
            TypeDef::Float => OVL_FLOAT | 32,
            TypeDef::Double => OVL_FLOAT | 64,
            TypeDef::Ptr { .. } => 64,
            other => return unsupported(format!("value of type {other:?}")),
        })
    }

    /// Cells a type takes in memory, which is also the rows a value of it takes.
    fn cells(&self, t: Ty) -> Result<u64> {
        Ok(match self.m.def(t) {
            TypeDef::Void | TypeDef::Label | TypeDef::Metadata | TypeDef::Func { .. } => 0,
            TypeDef::Int(_) | TypeDef::Half | TypeDef::Float | TypeDef::Double | TypeDef::Ptr { .. } => 1,
            TypeDef::Array { n, of } => n.checked_mul(self.cells(*of)?).filter(|c| *c < 1 << 40).ok_or_else(|| bad("array too large"))?,
            TypeDef::Vector { n, of } => *n as u64 * self.cells(*of)?,
            TypeDef::Struct { fields, .. } => {
                let mut c = 0;
                for f in fields {
                    c += self.cells(*f)?;
                }
                c
            }
        })
    }

    /// The cells of a constant, flattened.
    fn flatten(&self, v: Value, out: &mut Vec<u64>) -> Result<()> {
        let Some(c) = self.m.as_const(v) else { return unsupported("global initialiser that is not a constant") };
        match &c.c {
            Const::Int(x) | Const::Float(x) => out.push(*x),
            Const::Undef | Const::Zero => out.resize(out.len() + self.cells(c.ty)? as usize, 0),
            Const::Aggregate(fields) => {
                if matches!(self.m.def(c.ty), TypeDef::Ptr { .. }) {
                    out.push(self.const_gep(fields)?);
                } else {
                    for f in fields {
                        self.flatten(*f, out)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The pointer a constant getelementptr expression is: its base global, then the indices.
    fn const_gep(&self, ops: &[Value]) -> Result<u64> {
        let Some(ValueKind::Global(g)) = ops.first().map(|v| v.kind()) else { return unsupported("constant expression that is not a getelementptr of a global") };
        let mut ty = self.m.globals[g as usize].value_ty;
        let mut ptr = self.global_ptr[g as usize];
        for (k, ix) in ops[1..].iter().enumerate() {
            let Some(ix) = self.m.as_int(*ix) else { return unsupported("constant getelementptr with an index that is not an integer") };
            if k == 0 {
                if ix != 0 {
                    return unsupported("getelementptr that steps over its base object");
                }
                continue;
            }
            let (off, inner, _) = self.step(ty, ix)?;
            ptr += off;
            ty = inner;
        }
        Ok(ptr)
    }

    /// One constant getelementptr index into a value of type `ty`: (cell offset, the type
    /// reached, the bound of the index).
    fn step(&self, ty: Ty, ix: u64) -> Result<(u64, Ty, u64)> {
        match self.m.def(ty) {
            TypeDef::Array { n, of } => Ok((ix.min(n.saturating_sub(1)) * self.cells(*of)?, *of, *n)),
            TypeDef::Vector { n, of } => Ok((ix.min(*n as u64 - 1) * self.cells(*of)?, *of, *n as u64)),
            TypeDef::Struct { fields, .. } => {
                let k = ix as usize;
                if k >= fields.len() {
                    return Err(bad("struct index out of range"));
                }
                let mut off = 0;
                for f in &fields[..k] {
                    off += self.cells(*f)?;
                }
                Ok((off, fields[k], fields.len() as u64))
            }
            other => unsupported(format!("getelementptr into {other:?}")),
        }
    }

    fn dx_opcode(&self, inst: &Inst) -> Option<(u32, u8)> {
        if inst.op != Op::Call {
            return None;
        }
        let ValueKind::Func(f) = inst.a.kind() else { return None };
        let FnKind::Dx(ovl) = self.fn_kind[f as usize] else { return None };
        let first = *self.body.operands_of(inst).first()?;
        Some((self.m.as_int(first)? as u32, ovl))
    }

    /// The value an instruction merely renames, and the row offset inside it.
    fn alias(&self, inst: &Inst) -> Option<(Value, u32)> {
        match inst.op {
            Op::ExtractValue => Some((inst.a, inst.b.0)),
            Op::Cast(CastOp::ZExt) | Op::Cast(CastOp::BitCast) => Some((inst.a, 0)),
            Op::Call => {
                let (op, _) = self.dx_opcode(inst)?;
                (OpCode::BitcastI16toF16 as u32..=OpCode::BitcastF64toI64 as u32).contains(&op).then(|| (self.body.operands_of(inst)[1], 0))
            }
            _ => None,
        }
    }

    fn root(&self, mut v: Value) -> (Value, u32) {
        let mut off = 0;
        loop {
            v = self.body.resolve(v);
            let ValueKind::Inst(i) = v.kind() else { return (v, off) };
            match self.alias(&self.body.insts[i as usize]) {
                Some((to, o)) => {
                    v = to;
                    off += o;
                }
                None => return (v, off),
            }
        }
    }

    /// Whether the block is cut after this instruction.
    fn cuts(&self, inst: &Inst) -> bool {
        match inst.op {
            Op::AtomicRmw(_) | Op::CmpXchg => self.yields,
            Op::Call => match self.dx_opcode(inst) {
                Some((op, _)) if op == OpCode::Barrier as u32 => self.body.operands_of(inst).get(1).and_then(|v| self.m.as_int(*v)).is_some_and(|mode| mode & 1 != 0),
                Some((op, _)) => self.yields && (op == OpCode::AtomicBinOp as u32 || op == OpCode::AtomicCompareExchange as u32 || op == OpCode::BufferUpdateCounter as u32),
                None => false,
            },
            _ => false,
        }
    }

    fn for_operands(&self, inst: &Inst, f: &mut dyn FnMut(Value)) {
        match inst.op {
            Op::Bin(_) | Op::ICmp(_) | Op::FCmp(_) | Op::Store | Op::AtomicRmw(_) => {
                f(inst.a);
                f(inst.b);
            }
            Op::Cast(_) | Op::ExtractValue | Op::Load | Op::CondBr | Op::Switch => f(inst.a),
            Op::Select | Op::CmpXchg => {
                f(inst.a);
                f(inst.b);
                f(inst.c);
            }
            Op::Call => self.body.operands_of(inst).iter().for_each(|v| f(*v)),
            Op::Gep => {
                f(inst.a);
                self.body.operands_of(inst).iter().for_each(|v| f(*v));
            }
            Op::Phi => self.body.phis[inst.a.0 as usize].iter().for_each(|(v, _)| f(*v)),
            Op::Alloca | Op::Br | Op::Ret | Op::Unreachable => {}
        }
    }

    fn const_row(&mut self, bits: u64) -> u32 {
        if let Some(&r) = self.const_rows.get(&bits) {
            return r;
        }
        let r = self.next_const_row;
        self.next_const_row += 1;
        self.const_rows.insert(bits, r);
        self.p.row_init.push((r, bits));
        r
    }

    /// The row an operand is read from.
    fn opnd(&mut self, v: Value) -> Result<u32> {
        let (r, off) = self.root(v);
        match r.kind() {
            ValueKind::Inst(i) => {
                if let Some(&k) = self.known.get(&i) {
                    return Ok(self.const_row(k));
                }
                match self.row[i as usize] {
                    NONE => Err(bad("use of a value that has no rows")),
                    row => Ok(row + off),
                }
            }
            ValueKind::Const(c) => {
                let c = &self.m.consts[c as usize];
                let bits = match &c.c {
                    Const::Int(x) | Const::Float(x) => *x,
                    Const::Undef | Const::Zero => 0,
                    Const::Aggregate(fields) => {
                        if matches!(self.m.def(c.ty), TypeDef::Ptr { .. }) {
                            self.const_gep(fields)?
                        } else {
                            let f = *fields.get(off as usize).ok_or_else(|| bad("field of a constant out of range"))?;
                            return self.opnd(f);
                        }
                    }
                };
                Ok(self.const_row(bits))
            }
            ValueKind::Global(g) => Ok(self.const_row(self.global_ptr[g as usize])),
            ValueKind::Func(_) => unsupported("function used as a value"),
        }
    }

    /// The value of an operand the decoder must know (an opcode, a struct of properties).
    fn const_field(&self, v: Value, field: usize) -> Option<u64> {
        let c = self.m.as_const(v)?;
        match &c.c {
            Const::Int(x) | Const::Float(x) => (field == 0).then_some(*x),
            Const::Undef | Const::Zero => Some(0),
            Const::Aggregate(fields) => self.const_field(*fields.get(field)?, 0),
        }
    }

    /// The pointer an operand is when the decoder knows it.
    fn known_ptr(&self, v: Value) -> Option<u64> {
        let (r, off) = self.root(v);
        if off != 0 {
            return None;
        }
        match r.kind() {
            ValueKind::Inst(i) => self.known.get(&i).copied(),
            ValueKind::Global(g) => Some(self.global_ptr[g as usize]),
            ValueKind::Const(c) => match &self.m.consts[c as usize] {
                ConstData { ty, c: Const::Aggregate(fields) } if matches!(self.m.def(*ty), TypeDef::Ptr { .. }) => self.const_gep(fields).ok(),
                _ => None,
            },
            ValueKind::Func(_) => None,
        }
    }

    fn push(&mut self, op: Opc, w: u8, aux: u16, d: u32, a: u32, b: u32, c: u32) {
        self.p.insts.push(DInst { op, w, aux, d, a, b, c });
    }

    fn dst(&self, i: u32) -> u32 {
        self.row[i as usize]
    }
}

fn entry_props(m: &Module) -> Result<(u32, [u32; 3], u32, Stage, Vec<SigElement>)> {
    let int = |md: Option<Md>| -> Option<u64> {
        match &m.md[md?.0 as usize] {
            MdNode::Val(v) => m.as_int(*v),
            _ => None,
        }
    };
    let tuple = |md: Option<Md>| -> Option<&Vec<Option<Md>>> {
        match &m.md[md?.0 as usize] {
            MdNode::Tuple(t) => Some(t),
            _ => None,
        }
    };
    let mut kind = Stage::Compute;
    if let Some((_, model)) = m.named_md.iter().find(|(n, _)| n == "dx.shaderModel") {
        let stage = tuple(model.first().copied()).and_then(|t| t.first().copied().flatten()).map(|s| match &m.md[s.0 as usize] {
            MdNode::Str(s) => s.clone(),
            _ => String::new(),
        });
        kind = match stage.as_deref() {
            Some("cs") => Stage::Compute,
            Some("vs") => Stage::Vertex,
            Some("ps") => Stage::Pixel,
            _ => return unsupported(format!("stage {}", stage.unwrap_or_default())),
        };
    }
    let (_, entries) = m.named_md.iter().find(|(n, _)| n == "dx.entryPoints").ok_or_else(|| bad("no dx.entryPoints"))?;
    for e in entries {
        let Some(t) = tuple(Some(*e)) else { continue };
        let Some(MdNode::Val(f)) = t.first().copied().flatten().map(|x| &m.md[x.0 as usize]) else { continue };
        let ValueKind::Func(fi) = f.kind() else { continue };
        if m.funcs[fi as usize].body.is_none() {
            continue;
        }
        let mut numthreads = [1, 1, 1];
        let mut wave = 0;
        if let Some(props) = tuple(t.get(4).copied().flatten()) {
            for pair in props.chunks(2) {
                match (int(pair[0]), pair.get(1).copied().flatten()) {
                    (Some(4), v) => {
                        if let Some(n) = tuple(v) {
                            for k in 0..3 {
                                numthreads[k] = n.get(k).and_then(|x| int(*x)).unwrap_or(1) as u32;
                            }
                        }
                    }
                    // kDxilWaveSizeTag; kDxilRangedWaveSizeTag (min, max, preferred)
                    (Some(11), v) => wave = tuple(v).and_then(|n| int(*n.first()?)).unwrap_or(0) as u32,
                    (Some(23), v) => {
                        if let Some(n) = tuple(v) {
                            let pick = n.get(2).and_then(|x| int(*x)).filter(|x| *x != 0).or_else(|| n.first().and_then(|x| int(*x)));
                            wave = pick.unwrap_or(0) as u32;
                        }
                    }
                    _ => {}
                }
            }
        }
        // signatures: { inputs, outputs, patch constants }; an element is { id, name,
        // component type, semantic kind, semantic indices, interpolation mode, ... }
        let mut inputs = Vec::new();
        if let Some(elements) = tuple(t.get(2).copied().flatten()).and_then(|sigs| tuple(sigs.first().copied().flatten())) {
            for e in elements {
                let Some(e) = tuple(*e) else { continue };
                let field = |k: usize| e.get(k).and_then(|x| int(*x)).unwrap_or(0);
                inputs.push(SigElement { id: field(0) as u32, kind: field(3) as u8, interp: field(5) as u8 });
            }
        }
        return Ok((fi, numthreads, wave, kind, inputs));
    }
    Err(bad("no entry point with a body"))
}

/// Decode the entry point of `m` (a compute, vertex or pixel program). `default_wave`: the wave size of a program that
/// declares none.
pub fn decode(m: &Module, default_wave: u32) -> Result<Program> {
    let (fi, numthreads, wave, stage, inputs) = entry_props(m)?;
    let wave_size = if wave == 0 { default_wave } else { wave };
    if !(1..=128).contains(&wave_size) {
        return unsupported(format!("wave size {wave_size}"));
    }
    if stage == Stage::Pixel && wave_size % 4 != 0 {
        return unsupported(format!("a wave of {wave_size} pixels (not whole quads)"));
    }
    let numthreads = if stage == Stage::Compute { numthreads } else { [wave_size, 1, 1] };
    let threads = numthreads.iter().map(|x| *x as u64).product::<u64>();
    if threads == 0 || threads > 1 << 16 {
        return unsupported(format!("thread group of {threads} threads"));
    }
    let body = m.funcs[fi as usize].body.as_ref().ok_or_else(|| bad("entry without a body"))?;
    let fn_kind = m
        .funcs
        .iter()
        .map(|f| if f.name.starts_with("dx.op.") { FnKind::Dx(overload_of(&f.name)) } else if f.name.starts_with("llvm.lifetime.") { FnKind::Lifetime } else { FnKind::Other })
        .collect();
    let mut d = Dec {
        m,
        body,
        fn_kind,
        yields: threads > wave_size as u64,
        global_ptr: Vec::new(),
        row: vec![NONE; body.insts.len()],
        known: HashMap::new(),
        piece: vec![NONE; body.insts.len()],
        persistent: vec![false; body.insts.len()],
        const_rows: HashMap::new(),
        next_const_row: 0,
        p: Program {
            entry: m.funcs[fi as usize].name.clone(),
            stage,
            inputs,
            numthreads,
            wave_size,
            rows: 0,
            row_init: Vec::new(),
            blocks: Vec::new(),
            insts: Vec::new(),
            extra: Vec::new(),
            phi_dst: Vec::new(),
            phi_src: Vec::new(),
            preds: Vec::new(),
            max_phis: 0,
            switches: Vec::new(),
            frame_init: Vec::new(),
            const_mem: Vec::new(),
            shared_cells: 0,
            shared_addr: Vec::new(),
            binds: Vec::new(),
            props: Vec::new(),
            stats: ProgramStats::default(),
        },
    };
    d.layout_globals()?;
    let ord = cfg::order(body.blocks.len(), &|b, out| {
        let Some(&t) = body.blocks[b as usize].insts.last() else { return };
        let t = &body.insts[t as usize];
        match t.op {
            Op::Br => out.push(t.a.0),
            Op::CondBr => out.extend([t.b.0, t.c.0]),
            Op::Switch => {
                out.push(t.b.0);
                out.extend(body.switches[t.c.0 as usize].iter().map(|c| c.1 .0));
            }
            _ => {}
        }
    })
    .map_err(Error::Unsupported)?;
    d.p.stats.loops = ord.loops;
    d.p.stats.ir_insts = body.insts.len();
    d.assign(&ord)?;
    d.emit(&ord)?;
    d.p.rows = d.next_const_row;
    d.p.stats.insts = d.p.insts.len();
    d.p.stats.blocks = d.p.blocks.len();
    d.p.stats.const_rows = d.p.row_init.len();
    Ok(d.p)
}

impl<'a> Dec<'a> {
    fn layout_globals(&mut self) -> Result<()> {
        // two rounds: a constant initialiser may hold the address of another global
        let mut frame = 0u64;
        let mut konst = 0u64;
        let mut shared = 0u64;
        for g in &self.m.globals {
            let n = self.cells(g.value_ty)?;
            let (region, at) = if g.space == 3 {
                (REGION_SHARED, &mut shared)
            } else if g.constant && g.init.is_some() {
                (REGION_CONST, &mut konst)
            } else {
                (REGION_FRAME, &mut frame)
            };
            self.global_ptr.push(region << REGION_SHIFT | *at);
            *at += n;
            if g.space == 3 {
                let name = g.name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
                self.p.shared_addr.extend((0..n).map(|k| (name >> 40) << 40 | k));
            }
        }
        if shared > 1 << 24 || frame > 1 << 24 || konst > 1 << 28 {
            return unsupported("globals too large to model");
        }
        let mut frame_init = Vec::with_capacity(frame as usize);
        let mut const_mem = Vec::with_capacity(konst as usize);
        for g in &self.m.globals {
            let out = if g.space == 3 {
                continue;
            } else if g.constant && g.init.is_some() {
                &mut const_mem
            } else {
                &mut frame_init
            };
            match g.init {
                Some(v) => self.flatten(v, out)?,
                None => out.resize(out.len() + self.cells(g.value_ty)? as usize, 0),
            }
        }
        if frame_init.len() as u64 != frame || const_mem.len() as u64 != konst {
            return Err(bad("global initialiser of another size than its type"));
        }
        self.p.frame_init = frame_init;
        self.p.const_mem = const_mem;
        self.p.shared_cells = shared as u32;
        Ok(())
    }

    /// Rows: which values outlive their block piece, then a row for each.
    fn assign(&mut self, ord: &cfg::Order) -> Result<()> {
        let body = self.body;
        let mut piece = 0u32;
        let mut used: Vec<Value> = Vec::new();
        for &b in &ord.order {
            let block = &body.blocks[b as usize];
            for &i in &block.phis {
                self.piece[i as usize] = piece;
                used.clear();
                self.for_operands(&body.insts[i as usize], &mut |v| used.push(v));
                for &v in &used {
                    if let (ValueKind::Inst(r), _) = (self.root(v).0.kind(), ()) {
                        self.persistent[r as usize] = true;
                    }
                }
            }
            for &i in &block.insts {
                let inst = &body.insts[i as usize];
                self.piece[i as usize] = piece;
                used.clear();
                self.for_operands(inst, &mut |v| used.push(v));
                for &v in &used {
                    if let ValueKind::Inst(r) = self.root(v).0.kind() {
                        if self.piece[r as usize] != piece {
                            self.persistent[r as usize] = true;
                        }
                    }
                }
                if self.cuts(inst) {
                    piece += 1;
                }
            }
            piece += 1;
        }
        // values the decoder knows, then rows for the rest
        let mut persistent_rows = 0u32;
        let mut scratch_max = 0u32;
        let mut scratch = 0u32;
        let mut current = NONE;
        let mut local: Vec<(u32, u32)> = Vec::new();
        let mut frame = self.p.frame_init.len() as u64;
        for &b in &ord.order {
            let block = &body.blocks[b as usize];
            for &i in block.phis.iter().chain(&block.insts) {
                let inst = &body.insts[i as usize];
                if self.piece[i as usize] != current {
                    current = self.piece[i as usize];
                    scratch = 0;
                }
                match inst.op {
                    Op::Alloca => {
                        let n = self.cells(self.m.pointee(inst.ty))?;
                        self.known.insert(i, REGION_FRAME << REGION_SHIFT | frame);
                        frame += n;
                        if frame > 1 << 24 {
                            return unsupported("local arrays too large to model");
                        }
                        continue;
                    }
                    Op::Gep => {
                        if let Some(p) = self.static_gep(inst)? {
                            self.known.insert(i, p);
                            continue;
                        }
                    }
                    Op::Load => {
                        if let Some(p) = self.known_ptr(inst.a) {
                            if p >> REGION_SHIFT == REGION_CONST {
                                let v = self.p.const_mem.get((p & CELL_MASK) as usize).copied().unwrap_or(0);
                                self.known.insert(i, v);
                                continue;
                            }
                        }
                    }
                    _ => {}
                }
                if self.alias(inst).is_some() {
                    continue;
                }
                let n = self.cells(inst.ty)? as u32;
                if n == 0 {
                    continue;
                }
                if self.persistent[i as usize] {
                    self.row[i as usize] = persistent_rows;
                    persistent_rows += n;
                } else {
                    local.push((i, scratch));
                    scratch += n;
                    scratch_max = scratch_max.max(scratch);
                }
            }
        }
        for (i, at) in local {
            self.row[i as usize] = persistent_rows + at;
        }
        self.p.frame_init.resize(frame as usize, 0);
        self.next_const_row = persistent_rows + scratch_max;
        self.p.stats.persistent_rows = persistent_rows as usize;
        self.p.stats.scratch_rows = scratch_max as usize;
        Ok(())
    }

    /// The pointer of a getelementptr whose base and indices the decoder knows.
    fn static_gep(&self, inst: &Inst) -> Result<Option<u64>> {
        let Some(mut ptr) = self.known_ptr(inst.a) else { return Ok(None) };
        let mut ty = self.m.pointee(self.ty_of(inst.a));
        for (k, ix) in self.body.operands_of(inst).iter().enumerate() {
            let Some(ix) = self.m.as_int(self.body.resolve(*ix)) else { return Ok(None) };
            if k == 0 {
                if ix != 0 {
                    return unsupported("getelementptr that steps over its base object");
                }
                continue;
            }
            let (off, inner, bound) = self.step(ty, ix)?;
            if ix >= bound {
                // out of bounds whenever it runs: leave it to the instruction, which reports it
                return Ok(None);
            }
            ptr += off;
            ty = inner;
        }
        Ok(Some(ptr))
    }

    fn emit(&mut self, ord: &cfg::Order) -> Result<()> {
        let body = self.body;
        // the pieces of each block, to number the targets
        let mut first_piece = vec![NONE; body.blocks.len()];
        let mut last_piece = vec![NONE; body.blocks.len()];
        let mut n = 0u32;
        for &b in &ord.order {
            first_piece[b as usize] = n;
            n += 1 + body.blocks[b as usize].insts.iter().filter(|&&i| self.cuts(&body.insts[i as usize])).count() as u32;
            last_piece[b as usize] = n - 1;
        }
        let target = |b: u32| -> Result<u32> {
            match first_piece.get(b as usize) {
                Some(&p) if p != NONE => Ok(p),
                _ => Err(bad("branch to an unreachable block")),
            }
        };
        let mut switch_cases: Vec<(u64, u32)> = Vec::new();
        for (at, &b) in ord.order.iter().enumerate() {
            let block = &body.blocks[b as usize];
            let mut db = DBlock { first: self.p.insts.len() as u32, loop_end: last_piece[ord.order[ord.loop_end[at] as usize] as usize], ..DBlock::default() };
            // phis
            db.phi_first = self.p.phi_dst.len() as u32;
            db.pred_first = self.p.preds.len() as u32;
            db.src_first = self.p.phi_src.len() as u32;
            let mut pred_blocks: Vec<u32> = Vec::new();
            for (k, &i) in block.phis.iter().enumerate() {
                let inst = &body.insts[i as usize];
                let incoming = &body.phis[inst.a.0 as usize];
                if k == 0 {
                    for (_, from) in incoming {
                        if ord.pos[from.0 as usize] != NONE && !pred_blocks.contains(&from.0) {
                            pred_blocks.push(from.0);
                        }
                    }
                    if pred_blocks.len() > 64 {
                        // a switch's merge block: find the incoming values by map below
                    }
                    self.p.preds.extend(pred_blocks.iter().map(|b| last_piece[*b as usize]));
                }
                let dst = self.dst(i);
                if dst == NONE {
                    return Err(bad("phi without rows"));
                }
                self.p.phi_dst.push(dst);
                let base = self.p.phi_src.len();
                self.p.phi_src.resize(base + pred_blocks.len(), NONE);
                if pred_blocks.len() <= 16 {
                    for (v, from) in incoming {
                        if let Some(j) = pred_blocks.iter().position(|b| *b == from.0) {
                            self.p.phi_src[base + j] = self.opnd(*v)?;
                        }
                    }
                } else {
                    let index: HashMap<u32, usize> = pred_blocks.iter().enumerate().map(|(j, b)| (*b, j)).collect();
                    for (v, from) in incoming {
                        if let Some(&j) = index.get(&from.0) {
                            self.p.phi_src[base + j] = self.opnd(*v)?;
                        }
                    }
                }
                if self.p.phi_src[base..].contains(&NONE) {
                    return Err(bad("phi without a value for a predecessor"));
                }
                for (v, _) in incoming {
                    if let ValueKind::Inst(r) = self.root(*v).0.kind() {
                        if self.body.insts[r as usize].op == Op::Phi && self.piece[r as usize] == first_piece[b as usize] {
                            db.parallel = true;
                        }
                    }
                }
            }
            db.nphis = block.phis.len() as u32;
            db.npreds = pred_blocks.len() as u32;
            self.p.max_phis = self.p.max_phis.max(db.nphis);

            let mut piece = first_piece[b as usize];
            for &i in &block.insts {
                let inst = &body.insts[i as usize];
                self.inst(i, inst, &target, &mut switch_cases)?;
                if self.cuts(inst) {
                    let is_barrier = inst.op == Op::Call && self.dx_opcode(inst).is_some_and(|(op, _)| op == OpCode::Barrier as u32);
                    self.push(if is_barrier { Opc::Barrier } else { Opc::Yield }, 0, 0, NONE, piece + 1, 0, 0);
                    db.end = self.p.insts.len() as u32;
                    self.p.blocks.push(db);
                    piece += 1;
                    db = DBlock { first: self.p.insts.len() as u32, loop_end: piece, phi_first: self.p.phi_dst.len() as u32, pred_first: self.p.preds.len() as u32, src_first: self.p.phi_src.len() as u32, ..DBlock::default() };
                }
            }
            match block.insts.last().map(|&i| body.insts[i as usize].op) {
                Some(op) if op.is_terminator() => {}
                _ => return Err(bad("block without a terminator")),
            }
            db.end = self.p.insts.len() as u32;
            self.p.blocks.push(db);
            debug_assert_eq!(piece, last_piece[b as usize]);
        }
        Ok(())
    }

    fn inst(&mut self, i: u32, inst: &Inst, target: &dyn Fn(u32) -> Result<u32>, cases: &mut Vec<(u64, u32)>) -> Result<()> {
        if self.known.contains_key(&i) || self.alias(inst).is_some() {
            return Ok(());
        }
        let d = self.dst(i);
        match inst.op {
            Op::Bin(op) => {
                let w = self.scalar(inst.ty)?;
                let (a, b) = (self.opnd(inst.a)?, self.opnd(inst.b)?);
                let opc = match op {
                    BinOp::Add => Opc::Add,
                    BinOp::Sub => Opc::Sub,
                    BinOp::Mul => Opc::Mul,
                    BinOp::UDiv => Opc::UDiv,
                    BinOp::SDiv => Opc::SDiv,
                    BinOp::URem => Opc::URem,
                    BinOp::SRem => Opc::SRem,
                    BinOp::Shl => Opc::Shl,
                    BinOp::LShr => Opc::LShr,
                    BinOp::AShr => Opc::AShr,
                    BinOp::And => Opc::And,
                    BinOp::Or => Opc::Or,
                    BinOp::Xor => Opc::Xor,
                    BinOp::FAdd => Opc::FAdd,
                    BinOp::FSub => Opc::FSub,
                    BinOp::FMul => Opc::FMul,
                    BinOp::FDiv => Opc::FDiv,
                    BinOp::FRem => Opc::FRem,
                };
                self.push(opc, w & !OVL_FLOAT, 0, d, a, b, 0);
            }
            Op::Cast(op) => {
                let from = self.scalar(self.ty_of(inst.a))? & !OVL_FLOAT;
                let to = self.scalar(inst.ty)? & !OVL_FLOAT;
                let a = self.opnd(inst.a)?;
                let opc = match op {
                    CastOp::Trunc => {
                        self.push(Opc::Trunc, to, 0, d, a, 0, 0);
                        return Ok(());
                    }
                    CastOp::SExt => Opc::SExt,
                    CastOp::FPToUI => Opc::FPToUI,
                    CastOp::FPToSI => Opc::FPToSI,
                    CastOp::UIToFP => Opc::UIToFP,
                    CastOp::SIToFP => Opc::SIToFP,
                    CastOp::FPTrunc => Opc::FPTrunc,
                    CastOp::FPExt => Opc::FPExt,
                    CastOp::ZExt | CastOp::BitCast => return Err(bad("cast that should be an alias")),
                };
                self.push(opc, from, to as u16, d, a, 0, 0);
            }
            Op::ICmp(p) => {
                let w = self.scalar(self.ty_of(inst.a))?;
                let (a, b) = (self.opnd(inst.a)?, self.opnd(inst.b)?);
                self.push(Opc::ICmp, w, p as u16, d, a, b, 0);
            }
            Op::FCmp(p) => {
                let w = self.scalar(self.ty_of(inst.a))? & !OVL_FLOAT;
                let (a, b) = (self.opnd(inst.a)?, self.opnd(inst.b)?);
                self.push(Opc::FCmp, w, p as u16, d, a, b, 0);
            }
            Op::Select => {
                if self.cells(inst.ty)? != 1 {
                    return unsupported("select of an aggregate");
                }
                let (a, b, c) = (self.opnd(inst.a)?, self.opnd(inst.b)?, self.opnd(inst.c)?);
                self.push(Opc::Select, 0, 0, d, a, b, c);
            }
            Op::ExtractValue | Op::Alloca | Op::Phi => {}
            Op::Load => {
                if self.cells(inst.ty)? != 1 {
                    return unsupported("load of an aggregate");
                }
                match self.known_ptr(inst.a) {
                    Some(p) if p >> REGION_SHIFT == REGION_FRAME => self.push(Opc::LoadFrame, 0, 0, d, (p & CELL_MASK) as u32, 0, 0),
                    _ => {
                        let a = self.opnd(inst.a)?;
                        self.push(Opc::Load, 0, 0, d, a, 0, 0);
                    }
                }
            }
            Op::Store => {
                if self.cells(self.ty_of(inst.b))? != 1 {
                    return unsupported("store of an aggregate");
                }
                let b = self.opnd(inst.b)?;
                match self.known_ptr(inst.a) {
                    Some(p) if p >> REGION_SHIFT == REGION_FRAME => self.push(Opc::StoreFrame, 0, 0, NONE, (p & CELL_MASK) as u32, b, 0),
                    _ => {
                        let a = self.opnd(inst.a)?;
                        self.push(Opc::Store, 0, 0, NONE, a, b, 0);
                    }
                }
            }
            Op::Gep => {
                let mut ty = self.m.pointee(self.ty_of(inst.a));
                let mut cur = self.opnd(inst.a)?;
                let mut fixed = 0u64;
                let ixs = self.body.operands_of(inst).to_vec();
                let mut wrote = false;
                for (k, ix) in ixs.iter().enumerate() {
                    let ix = self.body.resolve(*ix);
                    let konst = self.m.as_int(ix);
                    if k == 0 {
                        if konst != Some(0) {
                            return unsupported("getelementptr that steps over its base object");
                        }
                        continue;
                    }
                    let (bound, of) = match self.m.def(ty) {
                        TypeDef::Struct { .. } => {
                            let Some(c) = konst else { return unsupported("getelementptr with a variable struct index") };
                            let (off, inner, _) = self.step(ty, c)?;
                            fixed += off;
                            ty = inner;
                            continue;
                        }
                        TypeDef::Array { n, of } => (*n, *of),
                        TypeDef::Vector { n, of } => (*n as u64, *of),
                        other => return unsupported(format!("getelementptr into {other:?}")),
                    };
                    let scale = self.cells(of)?;
                    ty = of;
                    if let Some(c) = konst.filter(|c| *c < bound) {
                        fixed += c * scale;
                        continue;
                    }
                    // a variable index, or a constant one out of bounds (reported when it runs)
                    let w = self.scalar(self.ty_of(ix))?;
                    let row = self.opnd(ix)?;
                    let at = self.p.extra.len() as u32;
                    self.p.extra.extend([scale as u32, bound.min(u32::MAX as u64) as u32]);
                    self.push(Opc::GepI, w, 0, d, cur, row, at);
                    cur = d;
                    wrote = true;
                }
                if fixed != 0 || !wrote {
                    self.push(Opc::GepC, 0, 0, d, cur, fixed as u32, 0);
                }
            }
            Op::AtomicRmw(op) => {
                let w = self.scalar(inst.ty)?;
                let (a, b) = (self.opnd(inst.a)?, self.opnd(inst.b)?);
                self.push(Opc::AtomicRmw, w, op as u16, d, a, b, 0);
            }
            Op::CmpXchg => {
                let w = self.scalar(self.ty_of(inst.b))?;
                let (a, b, c) = (self.opnd(inst.a)?, self.opnd(inst.b)?, self.opnd(inst.c)?);
                self.push(Opc::CmpXchg, w, 0, d, a, b, c);
            }
            Op::Call => self.call(i, inst)?,
            Op::Br => {
                let t = target(inst.a.0)?;
                self.push(Opc::Br, 0, 0, NONE, t, 0, 0);
            }
            Op::CondBr => {
                let a = self.opnd(inst.a)?;
                let (t, f) = (target(inst.b.0)?, target(inst.c.0)?);
                if t == f {
                    self.push(Opc::Br, 0, 0, NONE, t, 0, 0);
                } else {
                    self.push(Opc::CondBr, 0, 0, NONE, a, t, f);
                }
            }
            Op::Switch => {
                let a = self.opnd(inst.a)?;
                let w = self.scalar(self.ty_of(inst.a))?;
                let mask = if w >= 64 { u64::MAX } else { (1u64 << w) - 1 };
                cases.clear();
                for (v, b) in &self.body.switches[inst.c.0 as usize] {
                    cases.push((v & mask, target(b.0)?));
                }
                cases.sort_unstable();
                cases.dedup_by_key(|c| c.0);
                let default = target(inst.b.0)?;
                self.p.switches.push(cases.clone());
                self.push(Opc::Switch, 0, 0, NONE, a, default, self.p.switches.len() as u32 - 1);
            }
            Op::Ret => self.push(Opc::Ret, 0, 0, NONE, 0, 0, 0),
            Op::Unreachable => self.push(Opc::Unreachable, 0, 0, NONE, 0, 0, 0),
        }
        Ok(())
    }

    fn call(&mut self, i: u32, inst: &Inst) -> Result<()> {
        let ValueKind::Func(f) = inst.a.kind() else { return unsupported("indirect call") };
        let func = &self.m.funcs[f as usize];
        let ovl = match self.fn_kind[f as usize] {
            FnKind::Lifetime => return Ok(()),
            FnKind::Other => return unsupported(format!("call of {}", func.name)),
            FnKind::Dx(ovl) => ovl,
        };
        let args = self.body.operands_of(inst).to_vec();
        let Some(op) = args.first().and_then(|v| self.m.as_int(*v)) else { return unsupported("dx.op call without a constant opcode") };
        let op = op as u32;
        if !super::exec::supports(op) {
            return unsupported(format!("{} (opcode {op})", func.name));
        }
        let d = self.dst(i);
        let at = self.p.extra.len() as u32;
        let need = |n: usize| if args.len() < n { Err(bad(format!("{} with {} operands", func.name, args.len()))) } else { Ok(()) };
        if op == OpCode::CreateHandleFromBinding as u32 {
            need(4)?;
            let field = |k| self.const_field(args[1], k).map(|x| x as u32);
            let (Some(lower), Some(upper), Some(space), Some(class)) = (field(0), field(1), field(2), field(3)) else { return unsupported("createHandleFromBinding of a binding that is not a constant") };
            let bind = Bind { lower, upper, space, class };
            let id = match self.p.binds.iter().position(|b| *b == bind) {
                Some(id) => id,
                None => {
                    self.p.binds.push(bind);
                    self.p.binds.len() - 1
                }
            };
            if space >= 1 << 8 || class > 3 {
                return unsupported("binding in a register space above 255");
            }
            let index = self.opnd(args[2])?;
            self.p.extra.extend([id as u32, index]);
        } else if op == OpCode::AnnotateHandle as u32 {
            need(3)?;
            let (Some(p0), Some(p1)) = (self.const_field(args[2], 0), self.const_field(args[2], 1)) else { return unsupported("annotateHandle with properties that are not a constant") };
            let props = [p0 as u32, p1 as u32];
            let id = match self.p.props.iter().position(|p| *p == props) {
                Some(id) => id,
                None => {
                    self.p.props.push(props);
                    self.p.props.len() - 1
                }
            };
            let h = self.opnd(args[1])?;
            self.p.extra.extend([h, id as u32]);
        } else {
            for v in &args[1..] {
                let row = self.opnd(*v)?;
                self.p.extra.push(row);
            }
        }
        let n = self.p.extra.len() as u32 - at;
        if (n as usize) < super::exec::min_operands(op) {
            return Err(bad(format!("{} with {} operands", func.name, args.len())));
        }
        // what LLVM takes for a call that may write memory: loads are not moved across it
        let clobbers = !matches!(func.attrs, FnAttrs::NounwindReadnone | FnAttrs::NounwindReadonly);
        self.push(Opc::Dx, ovl, op as u16, d, at, n, clobbers as u32);
        Ok(())
    }
}
