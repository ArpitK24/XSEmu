//! The LLVM 3.7 subset DXIL uses, as an append-only in-memory module.
//!
//! Built for speed on programs of millions of instructions: values are 32-bit handles, types
//! and constants are interned, instructions are 20-byte records, and nothing is ever erased.
//! Instead of use lists there is forwarding ([`Body::replace`]): the SSA builder replaces a
//! trivial phi by pointing it at its value, and the writer ([`crate::bcwriter`]) resolves
//! every operand, drops instructions nothing live uses, and numbers what is left.
//!
//! DXIL facts the model relies on: a shader module defines one function (two for hull
//! shaders), `void()`, so there are no arguments; there are no vectors (DXC scalarises); and
//! every other function is a `dx.op.*` declaration.

use std::collections::HashMap;

/// An interned type. The common ones have fixed handles.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Ty(pub u32);

impl Ty {
    pub const VOID: Ty = Ty(0);
    pub const I1: Ty = Ty(1);
    pub const I8: Ty = Ty(2);
    pub const I16: Ty = Ty(3);
    pub const I32: Ty = Ty(4);
    pub const I64: Ty = Ty(5);
    pub const HALF: Ty = Ty(6);
    pub const F32: Ty = Ty(7);
    pub const F64: Ty = Ty(8);
    pub const LABEL: Ty = Ty(9);
    pub const METADATA: Ty = Ty(10);
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TypeDef {
    Void,
    Int(u32),
    Half,
    Float,
    Double,
    Label,
    Metadata,
    Ptr { to: Ty, space: u32 },
    Array { n: u64, of: Ty },
    /// only inside the resource class structs DXC names in dx.resources (`{ <4 x float> }`);
    /// DXIL code has no vector values
    Vector { n: u32, of: Ty },
    /// `name`: None is a literal struct; named structs are distinct by name
    Struct { name: Option<String>, fields: Vec<Ty> },
    Func { ret: Ty, params: Vec<Ty> },
}

/// A value: an instruction of the body being built, a constant, a global or a function.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Value(pub u32);

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ValueKind {
    Inst(u32),
    Const(u32),
    Global(u32),
    Func(u32),
}

impl Value {
    pub const NONE: Value = Value(u32::MAX);
    const SHIFT: u32 = 30;
    const MASK: u32 = (1 << Self::SHIFT) - 1;

    pub fn inst(i: u32) -> Value {
        debug_assert!(i < Self::MASK);
        Value(i)
    }
    pub fn constant(i: u32) -> Value {
        Value(1 << Self::SHIFT | i)
    }
    pub fn global(i: u32) -> Value {
        Value(2 << Self::SHIFT | i)
    }
    pub fn func(i: u32) -> Value {
        // u32::MAX is NONE: function indices stay below MASK
        Value(3 << Self::SHIFT | i)
    }
    pub fn kind(self) -> ValueKind {
        let i = self.0 & Self::MASK;
        match self.0 >> Self::SHIFT {
            0 => ValueKind::Inst(i),
            1 => ValueKind::Const(i),
            2 => ValueKind::Global(i),
            _ => ValueKind::Func(i),
        }
    }
    pub fn is_inst(self) -> bool {
        self.0 >> Self::SHIFT == 0
    }
    pub fn is_const(self) -> bool {
        self.0 >> Self::SHIFT == 1
    }
    pub fn is_none(self) -> bool {
        self == Self::NONE
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Block(pub u32);

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Const {
    /// the value's bits, zero-extended to 64 (i1 true is 1, i32 -1 is 0xFFFF_FFFF)
    Int(u64),
    /// IEEE bits in the type's own width: f32 bits for float, f64 bits for double, 16 for half
    Float(u64),
    Undef,
    /// null / zeroinitializer
    Zero,
    /// struct or array constant. With a pointer type: a `getelementptr inbounds` constant
    /// expression, the base (a global) first, then the indices: see [`Module::const_gep`]
    Aggregate(Vec<Value>),
}

#[derive(Clone, Debug)]
pub struct ConstData {
    pub ty: Ty,
    pub c: Const,
}

#[derive(Clone, Debug)]
pub struct Global {
    pub name: String,
    /// the pointer type of the global as a value
    pub ptr_ty: Ty,
    /// the type of what it points at
    pub value_ty: Ty,
    /// 0, or 3 for groupshared (DXIL's TGSM address space)
    pub space: u32,
    /// None: external (resource globals are not emitted in SM 6.6 handles-from-binding form)
    pub init: Option<Value>,
    pub constant: bool,
    /// internal linkage (everything XCL defines), else external
    pub internal: bool,
    pub align: u32,
    /// `unnamed_addr` (DXC's optimiser marks the constant tables it keeps)
    pub unnamed_addr: bool,
}

/// Function attribute sets DXIL declarations carry.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FnAttrs {
    None,
    Nounwind,
    NounwindReadnone,
    NounwindReadonly,
    NounwindNoreturn,
    /// noduplicate nounwind (dx.op.barrier)
    NounwindNoduplicate,
    /// nounwind, and the second parameter nocapture: `llvm.lifetime.start` / `llvm.lifetime.end`
    /// as DXC declares them
    Lifetime,
}

#[derive(Debug)]
pub struct Function {
    pub name: String,
    /// a `TypeDef::Func`
    pub ty: Ty,
    pub attrs: FnAttrs,
    /// None: a declaration
    pub body: Option<Body>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum BinOp {
    Add,
    FAdd,
    Sub,
    FSub,
    Mul,
    FMul,
    UDiv,
    SDiv,
    FDiv,
    URem,
    SRem,
    FRem,
    Shl,
    LShr,
    AShr,
    And,
    Or,
    Xor,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum CastOp {
    Trunc,
    ZExt,
    SExt,
    FPToUI,
    FPToSI,
    UIToFP,
    SIToFP,
    FPTrunc,
    FPExt,
    BitCast,
}

/// Integer predicates, with LLVM's numbers.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum IPred {
    Eq = 32,
    Ne = 33,
    Ugt = 34,
    Uge = 35,
    Ult = 36,
    Ule = 37,
    Sgt = 38,
    Sge = 39,
    Slt = 40,
    Sle = 41,
}

/// Float predicates, with LLVM's numbers.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum FPred {
    Oeq = 1,
    Ogt = 2,
    Oge = 3,
    Olt = 4,
    Ole = 5,
    One = 6,
    Ord = 7,
    Uno = 8,
    Ueq = 9,
    Ugt = 10,
    Uge = 11,
    Ult = 12,
    Ule = 13,
    Une = 14,
}

/// atomicrmw operations, with LLVM's numbers.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum RmwOp {
    Xchg = 0,
    Add = 1,
    Sub = 2,
    And = 3,
    Nand = 4,
    Or = 5,
    Xor = 6,
    Max = 7,
    Min = 8,
    UMax = 9,
    UMin = 10,
}

/// What an instruction is; its operands are in [`Inst::a`], `b`, `c` as noted.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Op {
    /// a, b
    Bin(BinOp),
    /// a; the result type is the instruction's
    Cast(CastOp),
    /// a, b; result i1
    ICmp(IPred),
    /// a, b; result i1
    FCmp(FPred),
    /// a = condition (i1), b = if true, c = if false
    Select,
    /// a = aggregate, b.0 = field index (a raw number, not a value)
    ExtractValue,
    /// a = callee, b.0 = first operand in [`Body::operands`], c.0 = operand count
    Call,
    /// a = pointer
    Load,
    /// a = pointer, b = value; type void
    Store,
    /// inbounds getelementptr: a = base pointer, b.0 = first index in [`Body::operands`],
    /// c.0 = index count; the instruction's type is the resulting pointer type
    Gep,
    /// the instruction's type is the pointer type; always in the entry block
    Alloca,
    /// a.0 = index in [`Body::phis`]
    Phi,
    /// a = pointer, b = value; seq_cst
    AtomicRmw(RmwOp),
    /// a = pointer, b = expected, c = new; seq_cst; the type is the struct { T, i1 }
    CmpXchg,
    /// a.0 = target block
    Br,
    /// a = condition, b.0 = block if true, c.0 = block if false
    CondBr,
    /// a = value, b.0 = default block, c.0 = index in [`Body::switches`]
    Switch,
    /// `ret void`
    Ret,
    Unreachable,
}

impl Op {
    pub fn is_terminator(self) -> bool {
        matches!(self, Op::Br | Op::CondBr | Op::Switch | Op::Ret | Op::Unreachable)
    }
}

/// [`Inst::flags`]
pub mod flag {
    /// `!dx.precise !{i32 1}` on a call (DXC's -Gis marks every float dx.op call)
    pub const PRECISE: u8 = 1;
    /// the `fast` fast-math flags on a float binary operation (never set under -Gis rules)
    pub const FAST: u8 = 2;
    pub const NUW: u8 = 4;
    pub const NSW: u8 = 8;
    pub const EXACT: u8 = 16;
    /// a getelementptr without `inbounds` (DXC leaves it off where it indexes a groupshared
    /// array); [`super::Body::gep`] builds inbounds ones
    pub const NO_INBOUNDS: u8 = 32;
    /// On a conditional br or a switch: the `!dx.controlflow.hints` DXC attaches for
    /// `[branch]`, `[flatten]`, `[fastopt]`, `[allow_uav_condition]`, `[forcecase]` and `[call]`,
    /// bit (hint - 1) of DXIL's ControlFlowHint. They share their bits with the flags above,
    /// which mean nothing on a terminator.
    pub const HINT_BRANCH: u8 = 1;
    pub const HINT_FLATTEN: u8 = 2;
    pub const HINT_FAST_OPT: u8 = 4;
    pub const HINT_ALLOW_UAV_CONDITION: u8 = 8;
    pub const HINT_FORCE_CASE: u8 = 16;
    pub const HINT_CALL: u8 = 32;
    /// every hint bit
    pub const HINTS: u8 = 63;
    /// On the br or conditional br that closes a loop: DXC's mark of a `[loop]` loop,
    /// `!llvm.loop !{distinct, !{!"llvm.loop.unroll.disable"}}`, a node of its own per loop.
    pub const LOOP_UNROLL_DISABLE: u8 = 64;
    /// the instruction has an effect beyond its value (the writer never drops it); set by
    /// [`Body::push`] for stores, atomics, terminators and calls of functions that are not
    /// `readnone` / `readonly`
    pub const EFFECT: u8 = 128;
}

/// Room for instructions a block's list gets with its first one ([`Body::push`]).
const FIRST_BLOCK_ROOM: usize = 16;

#[derive(Copy, Clone, Debug)]
pub struct Inst {
    pub op: Op,
    pub flags: u8,
    pub ty: Ty,
    pub a: Value,
    pub b: Value,
    pub c: Value,
}

#[derive(Default, Debug)]
pub struct BlockData {
    /// phi instructions, written first
    pub phis: Vec<u32>,
    /// the other instructions in order; the last one is the terminator once the block is closed
    pub insts: Vec<u32>,
}

/// The incoming (value, predecessor) pairs of one phi: a vector that holds up to three in
/// place. Nine phis in ten have three or fewer, and a large program has hundreds of
/// thousands of them: in place, they take no allocation each, and the passes that read them
/// (the liveness, the flags, the writer) find them next to each other in [`Body::phis`].
#[derive(Clone)]
pub struct PhiList(Repr);

#[derive(Clone)]
enum Repr {
    Inline(u8, [(Value, Block); INLINE_PHI]),
    Heap(Vec<(Value, Block)>),
}

const INLINE_PHI: usize = 3;
const NO_INCOMING: (Value, Block) = (Value::NONE, Block(u32::MAX));

impl Default for PhiList {
    fn default() -> PhiList {
        PhiList(Repr::Inline(0, [NO_INCOMING; INLINE_PHI]))
    }
}

impl PhiList {
    pub fn as_slice(&self) -> &[(Value, Block)] {
        match &self.0 {
            Repr::Inline(n, a) => &a[..*n as usize],
            Repr::Heap(v) => v,
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [(Value, Block)] {
        match &mut self.0 {
            Repr::Inline(n, a) => &mut a[..*n as usize],
            Repr::Heap(v) => v,
        }
    }

    pub fn push(&mut self, x: (Value, Block)) {
        match &mut self.0 {
            Repr::Inline(n, a) if (*n as usize) < INLINE_PHI => {
                a[*n as usize] = x;
                *n += 1;
            }
            Repr::Inline(_, a) => {
                let mut v = Vec::with_capacity(2 * INLINE_PHI);
                v.extend_from_slice(a);
                v.push(x);
                self.0 = Repr::Heap(v);
            }
            Repr::Heap(v) => v.push(x),
        }
    }

    pub fn pop(&mut self) -> Option<(Value, Block)> {
        match &mut self.0 {
            Repr::Inline(0, _) => None,
            Repr::Inline(n, a) => {
                *n -= 1;
                Some(std::mem::replace(&mut a[*n as usize], NO_INCOMING))
            }
            Repr::Heap(v) => v.pop(),
        }
    }

    pub fn truncate(&mut self, len: usize) {
        while self.len() > len {
            self.pop();
        }
    }

    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// As [`Vec::swap_remove`].
    pub fn swap_remove(&mut self, i: usize) -> (Value, Block) {
        let last = self.len() - 1;
        self.as_mut_slice().swap(i, last);
        self.pop().unwrap_or(NO_INCOMING)
    }

    /// As [`Vec::retain`].
    pub fn retain(&mut self, mut keep: impl FnMut(&(Value, Block)) -> bool) {
        let s = self.as_mut_slice();
        let mut kept = 0;
        for i in 0..s.len() {
            if keep(&s[i]) {
                s.swap(kept, i);
                kept += 1;
            }
        }
        self.truncate(kept);
    }

    pub fn extend_from_slice(&mut self, xs: &[(Value, Block)]) {
        for &x in xs {
            self.push(x);
        }
    }
}

impl std::ops::Deref for PhiList {
    type Target = [(Value, Block)];
    fn deref(&self) -> &[(Value, Block)] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for PhiList {
    fn deref_mut(&mut self) -> &mut [(Value, Block)] {
        self.as_mut_slice()
    }
}

impl std::fmt::Debug for PhiList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl PartialEq for PhiList {
    fn eq(&self, other: &PhiList) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Extend<(Value, Block)> for PhiList {
    fn extend<I: IntoIterator<Item = (Value, Block)>>(&mut self, xs: I) {
        for x in xs {
            self.push(x);
        }
    }
}

impl FromIterator<(Value, Block)> for PhiList {
    fn from_iter<I: IntoIterator<Item = (Value, Block)>>(xs: I) -> PhiList {
        let mut l = PhiList::default();
        l.extend(xs);
        l
    }
}

impl From<Vec<(Value, Block)>> for PhiList {
    fn from(v: Vec<(Value, Block)>) -> PhiList {
        if v.len() > INLINE_PHI {
            return PhiList(Repr::Heap(v));
        }
        v.into_iter().collect()
    }
}

impl<'a> IntoIterator for &'a PhiList {
    type Item = &'a (Value, Block);
    type IntoIter = std::slice::Iter<'a, (Value, Block)>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a> IntoIterator for &'a mut PhiList {
    type Item = &'a mut (Value, Block);
    type IntoIter = std::slice::IterMut<'a, (Value, Block)>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

/// The body of a defined function.
#[derive(Default, Debug)]
pub struct Body {
    pub insts: Vec<Inst>,
    /// block 0 is the entry block
    pub blocks: Vec<BlockData>,
    /// call arguments and GEP indices
    pub operands: Vec<Value>,
    /// incoming (value, predecessor) pairs of each phi
    pub phis: Vec<PhiList>,
    /// (case value, target) lists of each switch
    pub switches: Vec<Vec<(u64, Block)>>,
    /// The alignment in bytes (0: none stated) of each alloca, load and store whose alignment
    /// is not that of its type, by instruction index. DXC's output has a few (8 on the
    /// payload of an amplification shader, none on what -O0 leaves behind); nothing here
    /// needs to build them.
    pub aligns: HashMap<u32, u32>,
    /// forwarding: instruction index → the value that replaces it (sparse)
    replaced: HashMap<u32, Value>,
}

impl Body {
    pub fn new() -> Body {
        let mut b = Body::default();
        b.new_block();
        b
    }

    pub const ENTRY: Block = Block(0);

    pub fn new_block(&mut self) -> Block {
        self.blocks.push(BlockData::default());
        Block(self.blocks.len() as u32 - 1)
    }

    /// Whether the block already ends in a terminator.
    pub fn terminated(&self, b: Block) -> bool {
        self.blocks[b.0 as usize].insts.last().is_some_and(|&i| self.insts[i as usize].op.is_terminator())
    }

    /// Append an instruction to a block (a phi goes to the block's phi list) and return its
    /// value. Appending to a terminated block is a bug in the caller.
    pub fn push(&mut self, bb: Block, inst: Inst) -> Value {
        debug_assert!(!self.terminated(bb) || inst.op == Op::Phi, "push after terminator");
        let i = self.insts.len() as u32;
        let block = &mut self.blocks[bb.0 as usize];
        if inst.op == Op::Phi {
            block.phis.push(i);
        } else {
            // a block of the translator's code holds some tens of instructions: room for a
            // first run at once, in place of the steps 4, 8, 16 a list grows by on its own
            if block.insts.capacity() == 0 {
                block.insts.reserve_exact(FIRST_BLOCK_ROOM);
            }
            block.insts.push(i);
        }
        self.insts.push(inst);
        Value::inst(i)
    }

    fn simple(&mut self, bb: Block, op: Op, ty: Ty, a: Value, b: Value, c: Value, flags: u8) -> Value {
        self.push(bb, Inst { op, flags, ty, a, b, c })
    }

    pub fn bin(&mut self, bb: Block, op: BinOp, ty: Ty, a: Value, b: Value) -> Value {
        self.simple(bb, Op::Bin(op), ty, a, b, Value::NONE, 0)
    }
    pub fn cast(&mut self, bb: Block, op: CastOp, to: Ty, a: Value) -> Value {
        self.simple(bb, Op::Cast(op), to, a, Value::NONE, Value::NONE, 0)
    }
    pub fn icmp(&mut self, bb: Block, p: IPred, a: Value, b: Value) -> Value {
        self.simple(bb, Op::ICmp(p), Ty::I1, a, b, Value::NONE, 0)
    }
    pub fn fcmp(&mut self, bb: Block, p: FPred, a: Value, b: Value) -> Value {
        self.simple(bb, Op::FCmp(p), Ty::I1, a, b, Value::NONE, 0)
    }
    pub fn select(&mut self, bb: Block, ty: Ty, c: Value, t: Value, f: Value) -> Value {
        self.simple(bb, Op::Select, ty, c, t, f, 0)
    }
    /// `field_ty`: the type of the extracted field.
    pub fn extract(&mut self, bb: Block, field_ty: Ty, agg: Value, index: u32) -> Value {
        self.simple(bb, Op::ExtractValue, field_ty, agg, Value(index), Value::NONE, 0)
    }
    /// `ret`: the callee's return type; `effect`: the callee is neither readnone nor
    /// readonly (see [`flag::EFFECT`]); `flags`: e.g. [`flag::PRECISE`].
    pub fn call(&mut self, bb: Block, ret: Ty, callee: Value, args: &[Value], effect: bool, flags: u8) -> Value {
        let first = self.operands.len() as u32;
        self.operands.extend_from_slice(args);
        self.simple(bb, Op::Call, ret, callee, Value(first), Value(args.len() as u32), flags | if effect { flag::EFFECT } else { 0 })
    }
    /// The arguments of a call or the indices of a GEP.
    pub fn operands_of(&self, inst: &Inst) -> &[Value] {
        &self.operands[inst.b.0 as usize..(inst.b.0 + inst.c.0) as usize]
    }
    pub fn load(&mut self, bb: Block, ty: Ty, ptr: Value) -> Value {
        self.simple(bb, Op::Load, ty, ptr, Value::NONE, Value::NONE, 0)
    }
    pub fn store(&mut self, bb: Block, ptr: Value, v: Value) {
        self.simple(bb, Op::Store, Ty::VOID, ptr, v, Value::NONE, flag::EFFECT);
    }
    /// `ptr_ty`: the type of the result (pointer to the addressed element).
    pub fn gep(&mut self, bb: Block, ptr_ty: Ty, base: Value, indices: &[Value]) -> Value {
        let first = self.operands.len() as u32;
        self.operands.extend_from_slice(indices);
        self.simple(bb, Op::Gep, ptr_ty, base, Value(first), Value(indices.len() as u32), 0)
    }
    /// An alloca in the entry block (before its terminator, if it has one already).
    /// `ptr_ty`: pointer to the allocated type.
    pub fn alloca(&mut self, ptr_ty: Ty) -> Value {
        let i = self.insts.len() as u32;
        self.insts.push(Inst { op: Op::Alloca, flags: 0, ty: ptr_ty, a: Value::NONE, b: Value::NONE, c: Value::NONE });
        let entry = &mut self.blocks[0].insts;
        // allocas lead the entry block, in creation order
        let at = entry.iter().position(|&k| self.insts[k as usize].op != Op::Alloca).unwrap_or(entry.len());
        entry.insert(at, i);
        Value::inst(i)
    }
    pub fn atomic_rmw(&mut self, bb: Block, op: RmwOp, ty: Ty, ptr: Value, v: Value) -> Value {
        self.simple(bb, Op::AtomicRmw(op), ty, ptr, v, Value::NONE, flag::EFFECT)
    }
    /// `pair_ty`: the literal struct { T, i1 }.
    pub fn cmpxchg(&mut self, bb: Block, pair_ty: Ty, ptr: Value, expected: Value, new: Value) -> Value {
        self.simple(bb, Op::CmpXchg, pair_ty, ptr, expected, new, flag::EFFECT)
    }

    /// A phi without incoming values yet.
    pub fn phi(&mut self, bb: Block, ty: Ty) -> Value {
        self.phis.push(PhiList::default());
        self.push(bb, Inst { op: Op::Phi, flags: 0, ty, a: Value(self.phis.len() as u32 - 1), b: Value::NONE, c: Value::NONE })
    }
    pub fn phi_add(&mut self, phi: Value, v: Value, from: Block) {
        let k = self.insts[phi.0 as usize].a.0 as usize;
        self.phis[k].push((v, from));
    }
    pub fn phi_incoming(&self, phi: Value) -> &[(Value, Block)] {
        &self.phis[self.insts[phi.0 as usize].a.0 as usize]
    }

    pub fn br(&mut self, bb: Block, to: Block) {
        self.simple(bb, Op::Br, Ty::VOID, Value(to.0), Value::NONE, Value::NONE, flag::EFFECT);
    }
    pub fn cond_br(&mut self, bb: Block, cond: Value, t: Block, f: Block) {
        self.simple(bb, Op::CondBr, Ty::VOID, cond, Value(t.0), Value(f.0), flag::EFFECT);
    }
    pub fn switch(&mut self, bb: Block, v: Value, default: Block, cases: Vec<(u64, Block)>) {
        self.switches.push(cases);
        self.simple(bb, Op::Switch, Ty::VOID, v, Value(default.0), Value(self.switches.len() as u32 - 1), flag::EFFECT);
    }
    pub fn ret(&mut self, bb: Block) {
        self.simple(bb, Op::Ret, Ty::VOID, Value::NONE, Value::NONE, Value::NONE, flag::EFFECT);
    }
    pub fn unreachable(&mut self, bb: Block) {
        self.simple(bb, Op::Unreachable, Ty::VOID, Value::NONE, Value::NONE, Value::NONE, flag::EFFECT);
    }

    /// Forward every use of instruction `old` to `new` (the writer applies it; `old` is then
    /// dead). `new` must not resolve back to `old`.
    pub fn replace(&mut self, old: Value, new: Value) {
        debug_assert!(old.is_inst() && old != new);
        self.replaced.insert(old.0, new);
    }
    /// The value `v` stands for after forwarding.
    pub fn resolve(&self, mut v: Value) -> Value {
        if self.replaced.is_empty() {
            return v;
        }
        while v.is_inst() {
            match self.replaced.get(&v.0) {
                Some(&n) => v = n,
                None => break,
            }
        }
        v
    }
    /// Every forwarding entry: (replaced instruction index, the value it points at), in no
    /// particular order.
    pub fn replacements(&self) -> impl Iterator<Item = (u32, Value)> + '_ {
        self.replaced.iter().map(|(&k, &v)| (k, v))
    }
    pub fn has_replacements(&self) -> bool {
        !self.replaced.is_empty()
    }

    /// The type of an instruction value of this body.
    pub fn inst_ty(&self, v: Value) -> Ty {
        self.insts[v.0 as usize].ty
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Md(pub u32);

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum MdNode {
    Str(String),
    /// a constant, global or function (written with its type)
    Val(Value),
    /// None: a null operand
    Tuple(Vec<Option<Md>>),
}

#[derive(Debug)]
pub struct Module {
    pub types: Vec<TypeDef>,
    type_map: HashMap<TypeDef, Ty>,
    pub consts: Vec<ConstData>,
    const_map: HashMap<(Ty, Const), Value>,
    pub globals: Vec<Global>,
    pub funcs: Vec<Function>,
    func_map: HashMap<String, u32>,
    pub md: Vec<MdNode>,
    md_map: HashMap<MdNode, Md>,
    pub named_md: Vec<(String, Vec<Md>)>,
    /// what [`crate::dxop::call`] has declared so far
    pub dxop: crate::dxop::Cache,
}

impl Default for Module {
    fn default() -> Self {
        Self::new()
    }
}

impl Module {
    pub fn new() -> Module {
        let mut m = Module {
            types: Vec::new(),
            type_map: HashMap::new(),
            consts: Vec::new(),
            const_map: HashMap::new(),
            globals: Vec::new(),
            funcs: Vec::new(),
            func_map: HashMap::new(),
            md: Vec::new(),
            md_map: HashMap::new(),
            named_md: Vec::new(),
            dxop: Default::default(),
        };
        // the fixed handles of `Ty`
        for (i, t) in [TypeDef::Void, TypeDef::Int(1), TypeDef::Int(8), TypeDef::Int(16), TypeDef::Int(32), TypeDef::Int(64), TypeDef::Half, TypeDef::Float, TypeDef::Double, TypeDef::Label, TypeDef::Metadata]
            .into_iter()
            .enumerate()
        {
            let ty = m.ty(t);
            debug_assert_eq!(ty.0 as usize, i);
        }
        m
    }

    pub fn ty(&mut self, def: TypeDef) -> Ty {
        if let Some(&t) = self.type_map.get(&def) {
            return t;
        }
        let t = Ty(self.types.len() as u32);
        self.types.push(def.clone());
        self.type_map.insert(def, t);
        t
    }
    pub fn def(&self, t: Ty) -> &TypeDef {
        &self.types[t.0 as usize]
    }
    pub fn int(&mut self, bits: u32) -> Ty {
        self.ty(TypeDef::Int(bits))
    }
    pub fn ptr(&mut self, to: Ty) -> Ty {
        self.ty(TypeDef::Ptr { to, space: 0 })
    }
    pub fn ptr_in(&mut self, to: Ty, space: u32) -> Ty {
        self.ty(TypeDef::Ptr { to, space })
    }
    pub fn array(&mut self, n: u64, of: Ty) -> Ty {
        self.ty(TypeDef::Array { n, of })
    }
    pub fn vector(&mut self, n: u32, of: Ty) -> Ty {
        self.ty(TypeDef::Vector { n, of })
    }
    pub fn named_struct(&mut self, name: &str, fields: Vec<Ty>) -> Ty {
        self.ty(TypeDef::Struct { name: Some(name.to_string()), fields })
    }
    pub fn literal_struct(&mut self, fields: Vec<Ty>) -> Ty {
        self.ty(TypeDef::Struct { name: None, fields })
    }
    pub fn func_ty(&mut self, ret: Ty, params: Vec<Ty>) -> Ty {
        self.ty(TypeDef::Func { ret, params })
    }
    /// The type a pointer type points at.
    pub fn pointee(&self, ptr: Ty) -> Ty {
        match self.def(ptr) {
            TypeDef::Ptr { to, .. } => *to,
            other => panic!("pointee of {other:?}"),
        }
    }
    pub fn is_float(&self, t: Ty) -> bool {
        matches!(t, Ty::HALF | Ty::F32 | Ty::F64)
    }
    /// Bits of an integer or float type (0 for others).
    pub fn bits(&self, t: Ty) -> u32 {
        match self.def(t) {
            TypeDef::Int(b) => *b,
            TypeDef::Half => 16,
            TypeDef::Float => 32,
            TypeDef::Double => 64,
            _ => 0,
        }
    }

    pub fn constant(&mut self, ty: Ty, c: Const) -> Value {
        if let Some(&v) = self.const_map.get(&(ty, c.clone())) {
            return v;
        }
        let v = Value::constant(self.consts.len() as u32);
        self.consts.push(ConstData { ty, c: c.clone() });
        self.const_map.insert((ty, c), v);
        v
    }
    /// The constant of that type and value, if it exists already.
    pub fn lookup_constant(&self, ty: Ty, c: Const) -> Option<Value> {
        self.const_map.get(&(ty, c)).copied()
    }
    /// `v` is truncated to the type's width.
    pub fn const_int(&mut self, ty: Ty, v: u64) -> Value {
        let bits = self.bits(ty);
        let v = if bits >= 64 { v } else { v & ((1u64 << bits) - 1) };
        self.constant(ty, Const::Int(v))
    }
    pub fn const_i32(&mut self, v: u32) -> Value {
        self.constant(Ty::I32, Const::Int(v as u64))
    }
    pub fn const_i64(&mut self, v: u64) -> Value {
        self.constant(Ty::I64, Const::Int(v))
    }
    pub fn const_i8(&mut self, v: u8) -> Value {
        self.constant(Ty::I8, Const::Int(v as u64))
    }
    pub fn const_bool(&mut self, v: bool) -> Value {
        self.constant(Ty::I1, Const::Int(v as u64))
    }
    pub fn const_f32(&mut self, v: f32) -> Value {
        self.constant(Ty::F32, Const::Float(v.to_bits() as u64))
    }
    pub fn const_f32_bits(&mut self, bits: u32) -> Value {
        self.constant(Ty::F32, Const::Float(bits as u64))
    }
    pub fn const_f64(&mut self, v: f64) -> Value {
        self.constant(Ty::F64, Const::Float(v.to_bits()))
    }
    pub fn undef(&mut self, ty: Ty) -> Value {
        self.constant(ty, Const::Undef)
    }
    pub fn zero(&mut self, ty: Ty) -> Value {
        match ty {
            Ty::I1 | Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => self.constant(ty, Const::Int(0)),
            Ty::HALF | Ty::F32 | Ty::F64 => self.constant(ty, Const::Float(0)),
            _ => self.constant(ty, Const::Zero),
        }
    }
    pub fn aggregate(&mut self, ty: Ty, fields: Vec<Value>) -> Value {
        self.constant(ty, Const::Aggregate(fields))
    }
    /// `getelementptr inbounds (base, indices...)` as a constant of type `ptr_ty`: the address of
    /// an element of a global. DXC's optimiser folds a getelementptr whose operands are all
    /// constants into one; the lowering has no need to build them.
    pub fn const_gep(&mut self, ptr_ty: Ty, base: Value, indices: &[Value]) -> Value {
        let mut fields = Vec::with_capacity(indices.len() + 1);
        fields.push(base);
        fields.extend_from_slice(indices);
        self.constant(ptr_ty, Const::Aggregate(fields))
    }
    /// `bitcast (v to ptr_ty)` as a constant: a global's address as a pointer to another type
    /// (DXC's `i8*` argument of llvm.lifetime). Held as a pointer-typed `Const::Aggregate` of
    /// one value.
    pub fn const_bitcast(&mut self, ptr_ty: Ty, v: Value) -> Value {
        self.constant(ptr_ty, Const::Aggregate(vec![v]))
    }
    /// The operand of a [`Module::const_bitcast`] constant.
    pub fn as_const_bitcast(&self, v: Value) -> Option<Value> {
        let c = self.as_const(v)?;
        match (&c.c, self.types.get(c.ty.0 as usize)) {
            (Const::Aggregate(fields), Some(TypeDef::Ptr { .. })) if fields.len() == 1 => Some(fields[0]),
            _ => None,
        }
    }
    /// The base and the indices of a [`Module::const_gep`] constant.
    pub fn as_const_gep(&self, v: Value) -> Option<(Value, &[Value])> {
        let c = self.as_const(v)?;
        match (&c.c, self.types.get(c.ty.0 as usize)) {
            (Const::Aggregate(fields), Some(TypeDef::Ptr { .. })) => fields.split_first().map(|(&base, indices)| (base, indices)).filter(|(_, indices)| !indices.is_empty()),
            _ => None,
        }
    }
    /// The constant behind a value, if it is one.
    pub fn as_const(&self, v: Value) -> Option<&ConstData> {
        match v.kind() {
            ValueKind::Const(i) => Some(&self.consts[i as usize]),
            _ => None,
        }
    }
    /// The bits of an integer constant.
    pub fn as_int(&self, v: Value) -> Option<u64> {
        match self.as_const(v)?.c {
            Const::Int(x) => Some(x),
            _ => None,
        }
    }
    /// The bits of a float constant.
    pub fn as_float_bits(&self, v: Value) -> Option<u64> {
        match self.as_const(v)?.c {
            Const::Float(x) => Some(x),
            _ => None,
        }
    }

    /// A global variable; `value_ty` is what it holds. Names must be unique.
    pub fn global(&mut self, name: &str, value_ty: Ty, space: u32, init: Option<Value>, constant: bool) -> Value {
        let ptr_ty = self.ptr_in(value_ty, space);
        self.globals.push(Global { name: name.to_string(), ptr_ty, value_ty, space, init, constant, internal: true, align: 4, unnamed_addr: constant });
        Value::global(self.globals.len() as u32 - 1)
    }

    /// Declare a function once per name (a second declaration returns the first).
    pub fn declare(&mut self, name: &str, ty: Ty, attrs: FnAttrs) -> Value {
        if let Some(&i) = self.func_map.get(name) {
            return Value::func(i);
        }
        let i = self.funcs.len() as u32;
        self.funcs.push(Function { name: name.to_string(), ty, attrs, body: None });
        self.func_map.insert(name.to_string(), i);
        Value::func(i)
    }
    /// Define `void name()` with a finished body.
    pub fn define(&mut self, name: &str, body: Body) -> Value {
        let ty = self.func_ty(Ty::VOID, Vec::new());
        let i = self.funcs.len() as u32;
        self.funcs.push(Function { name: name.to_string(), ty, attrs: FnAttrs::None, body: Some(body) });
        self.func_map.insert(name.to_string(), i);
        Value::func(i)
    }
    pub fn function(&self, v: Value) -> &Function {
        match v.kind() {
            ValueKind::Func(i) => &self.funcs[i as usize],
            other => panic!("not a function: {other:?}"),
        }
    }

    /// The type of any value; `body` is the body instruction values belong to.
    pub fn type_of(&mut self, body: &Body, v: Value) -> Ty {
        match v.kind() {
            ValueKind::Inst(i) => body.insts[i as usize].ty,
            ValueKind::Const(i) => self.consts[i as usize].ty,
            ValueKind::Global(i) => self.globals[i as usize].ptr_ty,
            ValueKind::Func(i) => {
                let t = self.funcs[i as usize].ty;
                self.ptr(t)
            }
        }
    }

    fn md_node(&mut self, n: MdNode) -> Md {
        if let Some(&m) = self.md_map.get(&n) {
            return m;
        }
        let m = Md(self.md.len() as u32);
        self.md.push(n.clone());
        self.md_map.insert(n, m);
        m
    }
    pub fn md_str(&mut self, s: &str) -> Md {
        self.md_node(MdNode::Str(s.to_string()))
    }
    pub fn md_val(&mut self, v: Value) -> Md {
        self.md_node(MdNode::Val(v))
    }
    pub fn md_i32(&mut self, v: u32) -> Md {
        let c = self.const_i32(v);
        self.md_val(c)
    }
    pub fn md_i64(&mut self, v: u64) -> Md {
        let c = self.const_i64(v);
        self.md_val(c)
    }
    pub fn md_i8(&mut self, v: u8) -> Md {
        let c = self.const_i8(v);
        self.md_val(c)
    }
    pub fn md_bool(&mut self, v: bool) -> Md {
        let c = self.const_bool(v);
        self.md_val(c)
    }
    pub fn md_f32(&mut self, v: f32) -> Md {
        let c = self.const_f32(v);
        self.md_val(c)
    }
    /// A tuple node; identical tuples are one node, as in LLVM's uniqued metadata.
    pub fn md_tuple(&mut self, ops: Vec<Option<Md>>) -> Md {
        self.md_node(MdNode::Tuple(ops))
    }
    pub fn named(&mut self, name: &str, nodes: Vec<Md>) {
        self.named_md.push((name.to_string(), nodes));
    }
}

/// A module whose function bodies are finished, together with their liveness
/// ([`crate::bcwriter::liveness`]): what the flags, the ViewID state and the writer all need
/// of a body, computed once.
///
/// The liveness cannot go stale: the module is owned here and handed out by shared
/// reference only, and [`Sealed::extend`], the one way to add to it, takes the bodies out of
/// reach while it runs.
#[derive(Debug)]
pub struct Sealed {
    module: Module,
    /// by function index; None for a declaration
    lives: Vec<Option<crate::bcwriter::Live>>,
}

impl Sealed {
    /// Fails on forwarding that the writer would refuse (a cycle).
    pub fn new(module: Module) -> crate::Result<Sealed> {
        let lives = crate::bcwriter::module_liveness(&module)?;
        Ok(Sealed { module, lives })
    }

    pub fn module(&self) -> &Module {
        &self.module
    }

    /// The liveness of each function's body, by function index.
    pub fn lives(&self) -> &[Option<crate::bcwriter::Live>] {
        &self.lives
    }

    /// Add to the module what is not code: metadata, constants, types, declarations. `f`
    /// sees every defined function with an empty body in place of its own, so nothing it does
    /// can change what the liveness describes; a body it fills or adds is an error.
    pub fn extend<T>(&mut self, f: impl FnOnce(&mut Module) -> crate::Result<T>) -> crate::Result<T> {
        let held: Vec<Option<Body>> = self.module.funcs.iter_mut().map(|func| func.body.as_mut().map(std::mem::take)).collect();
        let out = f(&mut self.module);
        let funcs = &mut self.module.funcs;
        let pristine = |b: &Body| b.insts.is_empty() && b.blocks.is_empty() && b.operands.is_empty() && b.phis.is_empty() && b.switches.is_empty() && b.aligns.is_empty() && b.replaced.is_empty();
        let mut intact = funcs.len() >= held.len() && funcs[held.len()..].iter().all(|func| func.body.is_none());
        for (func, body) in funcs.iter_mut().zip(held) {
            intact &= match (&func.body, &body) {
                (Some(now), Some(_)) => pristine(now),
                (None, None) => true,
                _ => false,
            };
            func.body = body;
        }
        if !intact {
            return Err(crate::Error::Internal("a function body was changed or added in a sealed module".to_string()));
        }
        self.lives.resize_with(funcs.len(), || None);
        out
    }

    pub fn into_module(self) -> Module {
        self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sealed module takes metadata and constants, and refuses a change of a body.
    #[test]
    fn sealed_module_keeps_its_bodies() {
        let mut m = Module::new();
        let mut b = Body::new();
        let one = m.const_i32(1);
        let x = b.bin(Body::ENTRY, BinOp::Add, Ty::I32, one, one);
        let slot_ty = m.ptr(Ty::I32);
        let slot = b.alloca(slot_ty);
        b.store(Body::ENTRY, slot, x);
        b.ret(Body::ENTRY);
        m.define("main", b);
        let mut s = Sealed::new(m).unwrap();
        assert_eq!(s.lives()[0].as_ref().unwrap().insts, [true, true, true, true]);
        let node = s
            .extend(|m| {
                assert!(m.funcs[0].body.as_ref().is_some_and(|b| b.insts.is_empty()));
                let two = m.md_i32(2);
                Ok(m.md_tuple(vec![Some(two)]))
            })
            .unwrap();
        assert_eq!(s.module().md.len(), node.0 as usize + 1);
        assert_eq!(s.module().funcs[0].body.as_ref().unwrap().insts.len(), 4);
        let changed = s.extend(|m| {
            m.funcs[0].body.as_mut().unwrap().new_block();
            Ok(())
        });
        assert!(matches!(changed, Err(crate::Error::Internal(_))));
        // the body itself is as it was
        assert_eq!(s.module().funcs[0].body.as_ref().unwrap().blocks.len(), 1);
        let added = s.extend(|m| {
            m.define("other", Body::new());
            Ok(())
        });
        assert!(added.is_err());
    }

    #[test]
    fn fixed_type_handles_and_interning() {
        let mut m = Module::new();
        assert_eq!(m.int(32), Ty::I32);
        assert_eq!(m.ty(TypeDef::Float), Ty::F32);
        let p = m.ptr(Ty::I32);
        assert_eq!(m.ptr(Ty::I32), p);
        assert_eq!(m.pointee(p), Ty::I32);
        assert_eq!(m.const_i32(7), m.const_int(Ty::I32, 7 | 1 << 40));
        assert_ne!(m.const_i32(0), m.const_f32(0.0));
    }

    #[test]
    fn blocks_phis_and_forwarding() {
        let mut m = Module::new();
        let mut b = Body::new();
        let one = m.const_i32(1);
        let next = b.new_block();
        let x = b.bin(Body::ENTRY, BinOp::Add, Ty::I32, one, one);
        b.br(Body::ENTRY, next);
        assert!(b.terminated(Body::ENTRY));
        let phi = b.phi(next, Ty::I32);
        b.phi_add(phi, x, Body::ENTRY);
        b.replace(phi, x);
        assert_eq!(b.resolve(phi), x);
        let slot_ty = m.ptr(Ty::I32);
        let slot = b.alloca(slot_ty);
        assert_eq!(b.blocks[0].insts[0], slot.0);
        b.ret(next);
        assert_eq!(m.type_of(&b, x), Ty::I32);
    }
}
