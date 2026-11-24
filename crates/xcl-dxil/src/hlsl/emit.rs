//! Instruction emission at the insertion point, with the numeric conversions of HLSL. Owner:
//! the lowering subagent ([`super::intrinsics`] and [`super::resources`] call what is public
//! here). Constant folding and value numbering belong in these methods, so every caller
//! gets them.
//!
//! What the methods do beyond appending an instruction (all of it linear time, and all of it
//! exact: -Gis forbids anything that could change a float result):
//! * integer and boolean operations on constants fold, as do casts, comparisons and selects
//!   of constants; float arithmetic on two constants folds only when the IEEE result is not
//!   a NaN (the host computes the same bits DXC's APFloat does),
//! * algebraic identities on integers (`x + 0`, `x & x`, `x >> 0`, `(x & a) & b`), compare /
//!   negate pairs (`!(a < b)` is the inverse compare), cast pairs that cancel
//!   (`asuint(asfloat(x))`, `(uint)(uint64_t)x`), `(zext b) != 0`,
//! * scoped value numbering: a pure instruction equal to one that dominates the insertion
//!   point is that instruction. The scope is the caller's: [`Emit::vn_mark`] before lowering
//!   code that does not dominate what follows it and [`Emit::vn_release`] after.

use super::types::{HTy, Sk, TypeTable, Val};
use crate::consts::OpCode;
use crate::ir::{flag, BinOp, Block, Body, CastOp, FPred, FnAttrs, IPred, Inst, Module, Op, Ty, Value, ValueKind};
use crate::{Error, Result};

pub struct Emit {
    pub m: Module,
    pub body: Body,
    /// the block instructions are appended to
    pub bb: Block,
    pub types: TypeTable,
    /// source line being lowered, for error messages
    pub line: u32,
    vn: Vn,
    /// the integer constants asked for last, by a hash of type and value: most constants
    /// are asked for again and again, and interning one costs a hash of the whole constant
    ints: Vec<(Ty, u64, Value)>,
    by_ty: Vec<[Value; 2]>,
    /// the `%dx.types.ResourceProperties` constants made so far, by their two words
    pub res_props: Vec<([u32; 2], Value)>,
    // ---- what `lower/dxbreak.rs` reads (see there); idle unless `track` is set ----
    /// record in `touched` every instruction value an operation is asked to read
    pub track: bool,
    pub touched: Vec<Value>,
    /// the first instruction of the block DXC would be in ([`Emit::boundary`])
    pub range_start: u32,
    /// the loop being lowered (0: none), and (first instruction, loop) for every change
    pub loop_id: u32,
    pub loop_runs: Vec<(u32, u32)>,
    /// the wave-sensitive operations: (instruction, loop)
    pub wave_roots: Vec<(u32, u32)>,
    /// how many operations were emitted whose result depends on the lanes that run the
    /// block with it (wave, quad, derivatives, samples that take them): see `lower/thread.rs`
    pub lane_ops: u32,
    /// how many of them take derivatives: [`super::stage::derivative_group`]
    pub deriv_ops: u32,
    /// the selects that stand for a phi of DXC's: [`Emit::select_merge`]
    pub merge_sels: Vec<u32>,
}

/// The value-numbering table: open addressing over instruction indices, with the entries in
/// insertion order so a scope is released by removing the newest ones (removing in reverse
/// insertion order never breaks a probe chain).
#[derive(Default)]
struct Vn {
    /// instruction index + 1; 0: empty
    table: Vec<u32>,
    /// (instruction index, hash) in insertion order
    order: Vec<(u32, u32)>,
}

impl Vn {
    fn find(&self, hash: u32, mut same: impl FnMut(u32) -> bool) -> Option<u32> {
        if self.table.is_empty() {
            return None;
        }
        let mask = self.table.len() - 1;
        let mut i = hash as usize & mask;
        loop {
            match self.table[i] {
                0 => return None,
                k if same(k - 1) => return Some(k - 1),
                _ => i = (i + 1) & mask,
            }
        }
    }
    fn place(table: &mut [u32], hash: u32, inst: u32) {
        let mask = table.len() - 1;
        let mut i = hash as usize & mask;
        while table[i] != 0 {
            i = (i + 1) & mask;
        }
        table[i] = inst + 1;
    }
    fn insert(&mut self, hash: u32, inst: u32) {
        if (self.order.len() + 1) * 2 > self.table.len() {
            let n = (self.table.len() * 2).max(1024);
            self.table.clear();
            self.table.resize(n, 0);
            // in insertion order, so that release stays last in, first out
            for &(i, h) in &self.order {
                Self::place(&mut self.table, h, i);
            }
        }
        Self::place(&mut self.table, hash, inst);
        self.order.push((inst, hash));
    }
    fn release(&mut self, mark: usize) {
        if self.table.is_empty() {
            return;
        }
        let mask = self.table.len() - 1;
        while self.order.len() > mark {
            let (inst, hash) = self.order.pop().unwrap_or((0, 0));
            let mut i = hash as usize & mask;
            while self.table[i] != inst + 1 {
                i = (i + 1) & mask;
            }
            self.table[i] = 0;
        }
    }
}

fn mix(h: u32, v: u32) -> u32 {
    (h.rotate_left(5) ^ v).wrapping_mul(0x9E37_79B9)
}

fn op_code(op: Op) -> u32 {
    match op {
        Op::Bin(b) => 0x100 + b as u32,
        Op::Cast(c) => 0x200 + c as u32,
        Op::ICmp(p) => 0x300 + p as u32,
        Op::FCmp(p) => 0x400 + p as u32,
        Op::Select => 1,
        Op::ExtractValue => 2,
        Op::Call => 3,
        _ => 0,
    }
}

fn inverse_ipred(p: IPred) -> IPred {
    match p {
        IPred::Eq => IPred::Ne,
        IPred::Ne => IPred::Eq,
        IPred::Ugt => IPred::Ule,
        IPred::Uge => IPred::Ult,
        IPred::Ult => IPred::Uge,
        IPred::Ule => IPred::Ugt,
        IPred::Sgt => IPred::Sle,
        IPred::Sge => IPred::Slt,
        IPred::Slt => IPred::Sge,
        IPred::Sle => IPred::Sgt,
    }
}

fn swapped_ipred(p: IPred) -> IPred {
    match p {
        IPred::Eq => IPred::Eq,
        IPred::Ne => IPred::Ne,
        IPred::Ugt => IPred::Ult,
        IPred::Uge => IPred::Ule,
        IPred::Ult => IPred::Ugt,
        IPred::Ule => IPred::Uge,
        IPred::Sgt => IPred::Slt,
        IPred::Sge => IPred::Sle,
        IPred::Slt => IPred::Sgt,
        IPred::Sle => IPred::Sge,
    }
}

/// The predicate that is true exactly when `p` is false (ordered ↔ unordered).
fn inverse_fpred(p: FPred) -> FPred {
    match p {
        FPred::Oeq => FPred::Une,
        FPred::Ogt => FPred::Ule,
        FPred::Oge => FPred::Ult,
        FPred::Olt => FPred::Uge,
        FPred::Ole => FPred::Ugt,
        FPred::One => FPred::Ueq,
        FPred::Ord => FPred::Uno,
        FPred::Uno => FPred::Ord,
        FPred::Ueq => FPred::One,
        FPred::Ugt => FPred::Ole,
        FPred::Uge => FPred::Olt,
        FPred::Ult => FPred::Oge,
        FPred::Ule => FPred::Ogt,
        FPred::Une => FPred::Oeq,
    }
}

impl Emit {
    pub fn new() -> Emit {
        Emit {
            m: Module::new(),
            body: Body::new(),
            bb: Body::ENTRY,
            types: TypeTable::default(),
            line: 0,
            vn: Vn::default(),
            ints: vec![(Ty::VOID, 0, Value::NONE); 4096],
            by_ty: Vec::new(),
            res_props: Vec::new(),
            track: false,
            touched: Vec::new(),
            range_start: 0,
            loop_id: 0,
            loop_runs: Vec::new(),
            wave_roots: Vec::new(),
            lane_ops: 0,
            deriv_ops: 0,
            merge_sels: Vec::new(),
        }
    }

    /// Note that an operation reads `v` here (see `lower/dxbreak.rs`).
    #[inline]
    pub fn touch(&mut self, v: Value) {
        if self.track && v.is_inst() {
            self.touched.push(v);
        }
    }
    /// What follows is in a new basic block of DXC's (also where it is not one here).
    #[inline]
    pub fn boundary(&mut self) {
        if self.track {
            self.touched.clear();
            self.range_start = self.body.insts.len() as u32;
        }
    }
    pub fn set_loop(&mut self, id: u32) {
        self.loop_id = id;
        self.loop_runs.push((self.body.insts.len() as u32, id));
    }
    /// The select that merges the two values a variable has after a branch that was not
    /// emitted as one: DXC has a phi there, which reads the two values and not the condition.
    pub fn select_merge(&mut self, c: Value, t: Value, f: Value) -> Value {
        if !self.track {
            return self.select(c, t, f);
        }
        self.track = false;
        let n = self.body.insts.len() as u32;
        let v = self.select(c, t, f);
        self.track = true;
        if v.is_inst() && v.0 >= n {
            self.merge_sels.push(v.0);
        } else {
            self.touch(t);
            self.touch(f);
        }
        v
    }

    pub fn err<T>(&self, msg: impl Into<String>) -> Result<T> {
        Err(Error::Source { line: self.line, msg: msg.into() })
    }
    pub fn unsupported<T>(&self, what: impl Into<String>) -> Result<T> {
        Err(Error::Unsupported(what.into()))
    }

    pub fn ty_of(&mut self, v: Value) -> Ty {
        self.m.type_of(&self.body, v)
    }
    /// A scalar constant by its bits, through a small direct-mapped cache in front of the
    /// module's table (whose lookup hashes the constant: the lowering asks for the same few
    /// constants hundreds of thousands of times).
    #[inline]
    fn scalar_const(&mut self, ty: Ty, v: u64) -> Value {
        let i = ((v ^ ty.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 52) as usize;
        let e = self.ints[i];
        if e.0 == ty && e.1 == v && !e.2.is_none() {
            return e.2;
        }
        let c = match ty {
            Ty::F32 => self.m.const_f32_bits(v as u32),
            Ty::F64 => self.m.const_f64(f64::from_bits(v)),
            _ => self.m.const_int(ty, v),
        };
        self.ints[i] = (ty, v, c);
        c
    }
    #[inline]
    fn int_const(&mut self, ty: Ty, v: u64) -> Value {
        debug_assert!(ty != Ty::F32 && ty != Ty::F64);
        self.scalar_const(ty, v)
    }
    /// The undefined (0) or zero (1) constant of a type, kept by type.
    #[inline]
    fn by_type(&mut self, ty: Ty, k: usize) -> Value {
        let i = ty.0 as usize;
        if i >= self.by_ty.len() {
            self.by_ty.resize(i + 1, [Value::NONE; 2]);
        }
        let v = self.by_ty[i][k];
        if !v.is_none() {
            return v;
        }
        let v = if k == 0 { self.m.undef(ty) } else { self.m.zero(ty) };
        self.by_ty[i][k] = v;
        v
    }
    pub fn c_u32(&mut self, v: u32) -> Value {
        self.int_const(Ty::I32, v as u64)
    }
    pub fn c_bool(&mut self, v: bool) -> Value {
        self.int_const(Ty::I1, v as u64)
    }
    pub fn c_f32(&mut self, v: f32) -> Value {
        self.scalar_const(Ty::F32, v.to_bits() as u64)
    }
    pub fn c_f64(&mut self, v: f64) -> Value {
        self.scalar_const(Ty::F64, v.to_bits())
    }
    /// An integer constant of a scalar kind (truncated to its width).
    pub fn c_int(&mut self, k: Sk, v: u64) -> Value {
        self.int_const(k.ir(), v)
    }
    /// An integer constant of an IR type (truncated to its width).
    pub fn c_of(&mut self, ty: Ty, v: u64) -> Value {
        self.int_const(ty, v)
    }
    /// The zero of a scalar kind.
    pub fn zero(&mut self, k: Sk) -> Value {
        self.by_type(k.ir(), 1)
    }
    /// The zero of an IR type.
    pub fn zero_of(&mut self, ty: Ty) -> Value {
        self.by_type(ty, 1)
    }
    pub fn undef(&mut self, ty: Ty) -> Value {
        self.by_type(ty, 0)
    }
    /// Whether `v` is the undefined constant: what a variable holds before it is assigned.
    #[inline]
    pub fn is_undef(&self, v: Value) -> bool {
        v.is_const() && matches!(self.m.as_const(v), Some(crate::ir::ConstData { c: crate::ir::Const::Undef, .. }))
    }
    /// Zero in place of every undefined component of a value that leaves the lowering as
    /// the argument of a dx.op (a coordinate, a stored value, an output): the validator
    /// rejects most of those, and DXC rejects the program. Handles are left alone.
    pub fn define(&mut self, comps: &mut [Value]) {
        for c in comps {
            if self.is_undef(*c) {
                let ty = self.ty_of(*c);
                if matches!(ty, Ty::I1 | Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::HALF | Ty::F32 | Ty::F64) {
                    *c = self.zero_of(ty);
                }
            }
        }
    }
    /// A binary operation with an undefined operand, folded as LLVM's simplifier folds it
    /// for DXC: the sum, difference and exclusive or with anything are undefined, a product
    /// and a conjunction are 0, a disjunction is all ones, an undefined value shifted or
    /// divided is 0. A float operation is not folded there (DXC's output then fails
    /// validation): the undefined operand is read as 0.
    fn bin_undef(&mut self, op: BinOp, ty: Ty, a: Value, b: Value) -> Value {
        let first = self.is_undef(a);
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Xor => self.undef(ty),
            BinOp::Mul | BinOp::And => self.zero_of(ty),
            BinOp::Or => self.int_const(ty, u64::MAX),
            BinOp::Shl | BinOp::LShr | BinOp::AShr | BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem => {
                if first {
                    self.zero_of(ty)
                } else {
                    self.undef(ty)
                }
            }
            BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => {
                let z = self.zero_of(ty);
                let second = self.is_undef(b);
                self.bin(op, if first { z } else { a }, if second { z } else { b })
            }
        }
    }

    /// The scope position of the value-numbering table: see [`Emit::vn_release`].
    pub fn vn_mark(&self) -> usize {
        self.vn.order.len()
    }
    /// Forget the pure instructions emitted since `mark`: call it when the insertion point
    /// moves to a place they do not dominate (the end of a branch, a join).
    pub fn vn_release(&mut self, mark: usize) {
        self.vn.release(mark);
    }

    /// The instruction that defines `v`, if it is one of this body.
    pub fn def(&self, v: Value) -> Option<&Inst> {
        match v.kind() {
            ValueKind::Inst(i) => self.body.insts.get(i as usize),
            _ => None,
        }
    }

    /// Emit a pure instruction, or return the equal one that dominates the insertion point.
    fn numbered(&mut self, op: Op, ty: Ty, a: Value, b: Value, c: Value) -> Value {
        // a commutative integer operation equals its mirror image (the operands keep the
        // order they were written in: only the lookup is symmetric)
        let commutes = matches!(op, Op::Bin(BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor) | Op::ICmp(IPred::Eq | IPred::Ne));
        let (lo, hi) = if commutes && a.0 > b.0 { (b, a) } else { (a, b) };
        let hash = mix(mix(mix(mix(op_code(op), ty.0), lo.0), hi.0), c.0);
        let insts = &self.body.insts;
        if let Some(i) = self.vn.find(hash, |i| {
            let x = &insts[i as usize];
            x.op == op && x.ty == ty && x.c == c && ((x.a == a && x.b == b) || (commutes && x.a == b && x.b == a))
        }) {
            return Value::inst(i);
        }
        let v = self.body.push(self.bb, Inst { op, flags: 0, ty, a, b, c });
        self.vn.insert(hash, v.0);
        v
    }

    fn float_bits(&self, v: Value) -> Option<(Ty, u64)> {
        let c = self.m.as_const(v)?;
        match c.c {
            crate::ir::Const::Float(b) => Some((c.ty, b)),
            _ => None,
        }
    }
    fn c_float(&mut self, ty: Ty, v: f64) -> Value {
        if ty == Ty::F32 {
            self.c_f32(v as f32)
        } else {
            self.c_f64(v)
        }
    }

    /// `!v` of an i1.
    pub fn not(&mut self, v: Value) -> Value {
        let t = self.c_bool(true);
        self.bin(BinOp::Xor, v, t)
    }

    /// The constant a binary operation on two constants is, when it folds: an integer
    /// operation the GPU would not decide (see [`fold_int`]), a float one whose result is
    /// not a NaN.
    fn fold_const(&mut self, op: BinOp, ty: Ty, a: Value, b: Value) -> Option<Value> {
        if let (Some(x), Some(y)) = (self.m.as_int(a), self.m.as_int(b)) {
            let bits = self.m.bits(ty);
            return fold_int(op, bits, x, y).map(|r| self.int_const(ty, r));
        }
        let (Some((_, x)), Some((_, y))) = (self.float_bits(a), self.float_bits(b)) else { return None };
        if ty == Ty::F32 {
            let (x, y) = (f32::from_bits(x as u32), f32::from_bits(y as u32));
            let r = match op {
                BinOp::FAdd => x + y,
                BinOp::FSub => x - y,
                BinOp::FMul => x * y,
                BinOp::FDiv => x / y,
                _ => f32::NAN,
            };
            (!r.is_nan()).then(|| self.c_f32(r))
        } else if ty == Ty::F64 {
            let (x, y) = (f64::from_bits(x), f64::from_bits(y));
            let r = match op {
                BinOp::FAdd => x + y,
                BinOp::FSub => x - y,
                BinOp::FMul => x * y,
                BinOp::FDiv => x / y,
                _ => f64::NAN,
            };
            (!r.is_nan()).then(|| self.c_f64(r))
        } else {
            None
        }
    }

    /// A binary operation on two values of the same IR type. Integer operations fold and
    /// simplify (see the module text); float operations fold only for two constants with a
    /// non-NaN result and are otherwise emitted as written (no fast-math flags: -Gis).
    pub fn bin(&mut self, op: BinOp, mut a: Value, mut b: Value) -> Value {
        if self.track {
            self.touch(a);
            self.touch(b);
        }
        let ty = self.ty_of(a);
        if (a.is_const() || b.is_const()) && (self.is_undef(a) || self.is_undef(b)) {
            return self.bin_undef(op, ty, a, b);
        }
        let bits = self.m.bits(ty);
        let is_int = !self.m.is_float(ty);
        if a.is_const() != b.is_const() {
            // instcombine's FoldOpIntoSelect, which DXC's output shows: an operation on a
            // select of two constants and a constant is the select of the two results
            // (`(c ? 1.5 : 2.5) * 3.0` is `c ? 4.5 : 7.5`), when both fold
            let s = if a.is_const() { b } else { a };
            if let Some(&x) = self.def(s) {
                if x.op == Op::Select && x.b.is_const() && x.c.is_const() {
                    let (t, f) = if a.is_const() { (self.fold_const(op, ty, a, x.b), self.fold_const(op, ty, a, x.c)) } else { (self.fold_const(op, ty, x.b, b), self.fold_const(op, ty, x.c, b)) };
                    if let (Some(t), Some(f)) = (t, f) {
                        return self.select(x.a, t, f);
                    }
                }
            }
        }
        if !is_int {
            if let Some(r) = self.fold_const(op, ty, a, b) {
                return r;
            }
            return self.numbered(Op::Bin(op), ty, a, b, Value::NONE);
        }
        let commutative = matches!(op, BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor);
        let (mut ca, mut cb) = (self.m.as_int(a), self.m.as_int(b));
        if let (Some(x), Some(y)) = (ca, cb) {
            if let Some(r) = fold_int(op, bits, x, y) {
                return self.int_const(ty, r);
            }
        }
        if commutative && ca.is_some() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut ca, &mut cb);
        }
        let ones = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        if let Some(y) = cb {
            match op {
                BinOp::Add | BinOp::Sub | BinOp::Or | BinOp::Xor | BinOp::Shl | BinOp::LShr | BinOp::AShr if y == 0 => return a,
                BinOp::Mul | BinOp::And if y == 0 => return b,
                // LLVM's simplifier (so DXC): undefined; the validator rejects the instruction
                BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem if y == 0 => return self.undef(ty),
                BinOp::Mul | BinOp::UDiv | BinOp::SDiv if y == 1 => return a,
                BinOp::URem if y == 1 => return self.int_const(ty, 0),
                BinOp::And if y == ones => return a,
                BinOp::Or if y == ones => return b,
                // the power-of-two forms instcombine gives DXC's output
                BinOp::UDiv if y.is_power_of_two() => {
                    let s = self.int_const(ty, y.trailing_zeros() as u64);
                    return self.bin(BinOp::LShr, a, s);
                }
                BinOp::URem if y.is_power_of_two() => {
                    let s = self.int_const(ty, y - 1);
                    return self.bin(BinOp::And, a, s);
                }
                BinOp::Mul if y.is_power_of_two() => {
                    let s = self.int_const(ty, y.trailing_zeros() as u64);
                    return self.bin(BinOp::Shl, a, s);
                }
                _ => {}
            }
            if op == BinOp::Shl && y < bits as u64 {
                // (x & c1) << c2 is (x << c2) & (c1 << c2), when the mask was made for this shift
                // (instcombine does it for a mask with one use): a mask after it then folds in
                if let Some(&x) = self.def(a) {
                    let last = self.body.blocks[self.bb.0 as usize].insts.last() == Some(&a.0);
                    if let (Op::Bin(BinOp::And), Some(c1), true) = (x.op, self.m.as_int(x.b), last) {
                        let s = self.bin(BinOp::Shl, x.a, b);
                        let c = self.int_const(ty, (c1 << y) & ones);
                        return self.bin(BinOp::And, s, c);
                    }
                }
            }
            if op == BinOp::Xor && ty == Ty::I1 {
                // !(compare) is the inverse compare; !!x is x
                if let Some(&x) = self.def(a) {
                    match x.op {
                        Op::ICmp(p) => return self.icmp(inverse_ipred(p), x.a, x.b),
                        Op::FCmp(p) => return self.fcmp(inverse_fpred(p), x.a, x.b),
                        Op::Bin(BinOp::Xor) if self.m.as_int(x.b) == Some(1) => return x.a,
                        _ => {}
                    }
                }
            }
            if matches!(op, BinOp::Add | BinOp::Sub) {
                // (x ± c1) ± c2 is x + (±c1 ± c2), as instcombine gives it DXC
                if let Some(&x) = self.def(a) {
                    if let (Op::Bin(inner @ (BinOp::Add | BinOp::Sub)), Some(z)) = (x.op, self.m.as_int(x.b)) {
                        if inner == BinOp::Sub || op == BinOp::Sub {
                            let z = if inner == BinOp::Add { z } else { z.wrapping_neg() };
                            let y = if op == BinOp::Add { y } else { y.wrapping_neg() };
                            let c = self.int_const(ty, z.wrapping_add(y) & ones);
                            return self.bin(BinOp::Add, x.a, c);
                        }
                    }
                }
            }
            if commutative {
                // (x op c1) op c2 is x op (c1 op c2)
                if let Some(&x) = self.def(a) {
                    if x.op == Op::Bin(op) {
                        if let Some(z) = self.m.as_int(x.b) {
                            if let Some(r) = fold_int(op, bits, z, y) {
                                let c = self.int_const(ty, r);
                                return self.bin(op, x.a, c);
                            }
                        }
                    }
                }
            }
        } else if let Some(x) = ca {
            match op {
                BinOp::Shl | BinOp::LShr | BinOp::AShr | BinOp::UDiv | BinOp::URem if x == 0 => return a,
                _ => {}
            }
        }
        if a == b {
            match op {
                BinOp::And | BinOp::Or => return a,
                BinOp::Xor | BinOp::Sub => return self.int_const(ty, 0),
                _ => {}
            }
        }
        self.numbered(Op::Bin(op), ty, a, b, Value::NONE)
    }

    /// Whether [`Emit::cast`] of `a` takes no cast instruction: `a` is a constant whose cast
    /// folds, or a select of such values (`depth` levels of them).
    fn cast_folds(&mut self, op: CastOp, to: Ty, a: Value, depth: u32) -> bool {
        if !a.is_const() {
            return match self.def(a) {
                Some(&x) if x.op == Op::Select && depth > 0 => self.cast_folds(op, to, x.b, depth - 1) && self.cast_folds(op, to, x.c, depth - 1),
                _ => false,
            };
        }
        let n = self.body.insts.len();
        let mark = self.vn_mark();
        let v = self.cast(op, to, a);
        if self.body.insts.len() == n {
            return v.is_const();
        }
        // it took an instruction: take it back out
        self.vn_release(mark);
        self.body.insts.truncate(n);
        self.body.blocks[self.bb.0 as usize].insts.pop();
        false
    }

    pub fn cast(&mut self, op: CastOp, to: Ty, a: Value) -> Value {
        let from = self.ty_of(a);
        if from == to {
            return a;
        }
        self.touch(a);
        if self.is_undef(a) {
            // as LLVM folds it: an extension or an integer made a float is 0, the rest undefined
            return match op {
                CastOp::ZExt | CastOp::SExt | CastOp::UIToFP | CastOp::SIToFP => self.zero_of(to),
                _ => self.undef(to),
            };
        }
        if let Some(x) = self.m.as_int(a) {
            let from_bits = self.m.bits(from);
            let sx = if from_bits == 0 || from_bits >= 64 { x as i64 } else { ((x << (64 - from_bits)) as i64) >> (64 - from_bits) };
            match op {
                CastOp::Trunc | CastOp::ZExt => return self.int_const(to, x),
                CastOp::SExt => return self.int_const(to, sx as u64),
                CastOp::UIToFP => return self.c_float(to, if to == Ty::F32 { (x as f32) as f64 } else { x as f64 }),
                CastOp::SIToFP => return self.c_float(to, if to == Ty::F32 { (sx as f32) as f64 } else { sx as f64 }),
                CastOp::BitCast if to == Ty::F32 => return self.m.const_f32_bits(x as u32),
                CastOp::BitCast if to == Ty::F64 => return self.c_f64(f64::from_bits(x)),
                _ => {}
            }
        } else if let Some((fty, fb)) = self.float_bits(a) {
            let x = if fty == Ty::F32 { f32::from_bits(fb as u32) as f64 } else { f64::from_bits(fb) };
            let to_bits = self.m.bits(to);
            match op {
                CastOp::BitCast if fty != Ty::HALF => return self.int_const(to, fb),
                CastOp::FPExt if fty == Ty::F32 && to == Ty::F64 && !x.is_nan() => return self.c_f64(x),
                CastOp::FPTrunc if fty == Ty::F64 && to == Ty::F32 && !x.is_nan() => return self.c_f32(x as f32),
                // out of range is undefined in LLVM: left to the GPU
                CastOp::FPToUI if fty != Ty::HALF && to_bits <= 64 && x > -1.0 && x < 2f64.powi(to_bits as i32) => return self.int_const(to, x.trunc() as u64),
                CastOp::FPToSI if fty != Ty::HALF && to_bits <= 64 && x > -(2f64.powi(to_bits as i32 - 1)) - 1.0 && x < 2f64.powi(to_bits as i32 - 1) => {
                    return self.int_const(to, x.trunc() as i64 as u64)
                }
                _ => {}
            }
        } else if let Some(&x) = self.def(a) {
            if let Op::Cast(inner) = x.op {
                let src = self.ty_of(x.a);
                match (op, inner) {
                    (CastOp::BitCast, CastOp::BitCast) | (CastOp::Trunc, CastOp::ZExt | CastOp::SExt) if src == to => return x.a,
                    (CastOp::ZExt, CastOp::ZExt) | (CastOp::SExt, CastOp::SExt) | (CastOp::Trunc, CastOp::Trunc) => return self.cast(op, to, x.a),
                    _ => {}
                }
            }
            // instcombine's FoldOpIntoSelect again: the cast of a select of two constants is
            // the select of the casts (a literal `c ? 0.1 : 0.2`, a double in DXC as here,
            // meets a float as `select i1 c, float 0.1, float 0.2`), when both fold
            // (an arm may be such a select itself: `c ? (d ? 0.1 : 0.7) : 0.2`)
            if x.op == Op::Select && self.cast_folds(op, to, x.b, 3) && self.cast_folds(op, to, x.c, 3) {
                let (t, f) = (self.cast(op, to, x.b), self.cast(op, to, x.c));
                return self.select(x.a, t, f);
            }
        }
        self.numbered(Op::Cast(op), to, a, Value::NONE, Value::NONE)
    }

    pub fn icmp(&mut self, mut p: IPred, mut a: Value, mut b: Value) -> Value {
        if self.track {
            self.touch(a);
            self.touch(b);
        }
        if (a.is_const() || b.is_const()) && (self.is_undef(a) || self.is_undef(b)) {
            // as LLVM (so DXC): the undefined side is taken to equal the other
            return self.c_bool(matches!(p, IPred::Eq | IPred::Uge | IPred::Ule | IPred::Sge | IPred::Sle));
        }
        let (mut ca, mut cb) = (self.m.as_int(a), self.m.as_int(b));
        if let (Some(x), Some(y)) = (ca, cb) {
            let ty = self.ty_of(a);
            let bits = self.m.bits(ty).clamp(1, 64);
            let s = |v: u64| ((v << (64 - bits)) as i64) >> (64 - bits);
            let r = match p {
                IPred::Eq => x == y,
                IPred::Ne => x != y,
                IPred::Ugt => x > y,
                IPred::Uge => x >= y,
                IPred::Ult => x < y,
                IPred::Ule => x <= y,
                IPred::Sgt => s(x) > s(y),
                IPred::Sge => s(x) >= s(y),
                IPred::Slt => s(x) < s(y),
                IPred::Sle => s(x) <= s(y),
            };
            return self.c_bool(r);
        }
        if a == b {
            return self.c_bool(matches!(p, IPred::Eq | IPred::Uge | IPred::Ule | IPred::Sge | IPred::Sle));
        }
        if ca.is_some() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut ca, &mut cb);
            p = swapped_ipred(p);
        }
        if let Some(y) = cb {
            match p {
                IPred::Ult if y == 0 => return self.c_bool(false),
                IPred::Uge if y == 0 => return self.c_bool(true),
                IPred::Eq | IPred::Ne if y <= 1 => {
                    // (zext b) != 0 is b
                    if let Some(&x) = self.def(a) {
                        if x.op == Op::Cast(CastOp::ZExt) && self.ty_of(x.a) == Ty::I1 {
                            return if (p == IPred::Ne) == (y == 0) { x.a } else { self.not(x.a) };
                        }
                        // ((v >> n) & c) == 0 is (v & (c << n)) == 0, as instcombine has it: the
                        // mask is one value for every test of bit n (the lane's bit of each
                        // execution mask), where the shift is one per value tested
                        if y == 0 && x.op == Op::Bin(BinOp::And) && x.b.is_const() {
                            if let Some(&s) = self.def(x.a) {
                                if s.op == Op::Bin(BinOp::LShr) && !s.a.is_const() && !s.b.is_const() {
                                    let mask = self.bin(BinOp::Shl, x.b, s.b);
                                    a = self.bin(BinOp::And, s.a, mask);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        self.numbered(Op::ICmp(p), Ty::I1, a, b, Value::NONE)
    }

    pub fn fcmp(&mut self, p: FPred, a: Value, b: Value) -> Value {
        if self.track {
            self.touch(a);
            self.touch(b);
        }
        if (a.is_const() || b.is_const()) && (self.is_undef(a) || self.is_undef(b)) {
            return self.c_bool(matches!(p, FPred::Oeq | FPred::Oge | FPred::Ole | FPred::Ueq | FPred::Uge | FPred::Ule));
        }
        if let (Some((ta, x)), Some((_, y))) = (self.float_bits(a), self.float_bits(b)) {
            if ta != Ty::HALF {
                let (x, y) = if ta == Ty::F32 { (f32::from_bits(x as u32) as f64, f32::from_bits(y as u32) as f64) } else { (f64::from_bits(x), f64::from_bits(y)) };
                let un = x.is_nan() || y.is_nan();
                let r = match p {
                    FPred::Oeq => x == y,
                    FPred::Ogt => x > y,
                    FPred::Oge => x >= y,
                    FPred::Olt => x < y,
                    FPred::Ole => x <= y,
                    FPred::One => !un && x != y,
                    FPred::Ord => !un,
                    FPred::Uno => un,
                    FPred::Ueq => un || x == y,
                    FPred::Ugt => un || x > y,
                    FPred::Uge => un || x >= y,
                    FPred::Ult => un || x < y,
                    FPred::Ule => un || x <= y,
                    FPred::Une => un || x != y,
                };
                return self.c_bool(r);
            }
        }
        self.numbered(Op::FCmp(p), Ty::I1, a, b, Value::NONE)
    }

    pub fn select(&mut self, c: Value, mut t: Value, mut f: Value) -> Value {
        if self.track {
            self.touch(c);
            self.touch(t);
            self.touch(f);
        }
        if t == f {
            return t;
        }
        if let Some(x) = self.m.as_int(c) {
            return if x != 0 { t } else { f };
        }
        // an arm that is itself a select on the same condition is that select's arm on this
        // side: `c ? (c ? a : b) : d` is `c ? a : d`. A variable assigned twice under one
        // condition (the lane mask of the block) is one select, not a chain of them.
        if let Some(&x) = self.def(t) {
            if x.op == Op::Select && x.a == c {
                t = x.b;
            }
        }
        if let Some(&x) = self.def(f) {
            if x.op == Op::Select && x.a == c {
                f = x.c;
            }
        }
        if t == f {
            return t;
        }
        if self.is_undef(c) {
            // LLVM's choice for an undefined condition
            return if f.is_const() { f } else { t };
        }
        // an undefined arm is whatever the other one is (and the validator rejects an
        // instruction that reads undef, which a phi may)
        match (self.m.as_const(t).map(|k| &k.c), self.m.as_const(f).map(|k| &k.c)) {
            (Some(crate::ir::Const::Undef), _) => return f,
            (_, Some(crate::ir::Const::Undef)) => return t,
            _ => {}
        }
        let ty = self.ty_of(t);
        if ty == Ty::I1 {
            match (self.m.as_int(t), self.m.as_int(f)) {
                (Some(1), Some(0)) => return c,
                (Some(0), Some(1)) => return self.not(c),
                _ => {}
            }
        }
        self.numbered(Op::Select, ty, c, t, f)
    }

    pub fn extract(&mut self, field_ty: Ty, agg: Value, index: u32) -> Value {
        self.touch(agg);
        self.numbered(Op::ExtractValue, field_ty, agg, Value(index), Value::NONE)
    }

    /// A dx.op call at the insertion point; float operations are marked precise (-Gis).
    /// A `readnone` operation equal to one that dominates the insertion point is that
    /// one, except operations whose result depends on the lanes that are active where they
    /// run (wave and quad operations, derivatives): those stay where the source put them.
    pub fn dx(&mut self, op: OpCode, overload: Ty, args: &[Value]) -> Value {
        let v = crate::dxop::call(&mut self.m, &mut self.body, self.bb, op, overload, args, true);
        if self.track {
            for &a in args {
                self.touch(a);
            }
        }
        let Some(&x) = self.def(v) else { return v };
        if matches!(
            op,
            OpCode::WaveIsFirstLane
                | OpCode::WaveAnyTrue
                | OpCode::WaveAllTrue
                | OpCode::WaveActiveAllEqual
                | OpCode::WaveActiveBallot
                | OpCode::WaveReadLaneFirst
                | OpCode::WaveActiveOp
                | OpCode::WaveActiveBit
                | OpCode::WavePrefixOp
                | OpCode::WaveAllBitCount
                | OpCode::WavePrefixBitCount
                | OpCode::WaveMatch
                | OpCode::WaveMultiPrefixOp
                | OpCode::WaveMultiPrefixBitCount
        ) {
            self.wave_roots.push((v.0, self.loop_id));
        }
        self.deriv_ops += matches!(
            op,
            OpCode::DerivCoarseX | OpCode::DerivCoarseY | OpCode::DerivFineX | OpCode::DerivFineY | OpCode::CalculateLOD | OpCode::Sample | OpCode::SampleBias | OpCode::SampleCmp | OpCode::SampleCmpBias
        ) as u32;
        self.lane_ops += matches!(
            op,
            OpCode::WaveIsFirstLane
                | OpCode::WaveAnyTrue
                | OpCode::WaveAllTrue
                | OpCode::WaveActiveAllEqual
                | OpCode::WaveActiveBallot
                | OpCode::WaveReadLaneAt
                | OpCode::WaveReadLaneFirst
                | OpCode::WaveActiveOp
                | OpCode::WaveActiveBit
                | OpCode::WavePrefixOp
                | OpCode::WaveAllBitCount
                | OpCode::WavePrefixBitCount
                | OpCode::WaveMatch
                | OpCode::WaveMultiPrefixOp
                | OpCode::WaveMultiPrefixBitCount
                | OpCode::QuadReadLaneAt
                | OpCode::QuadOp
                | OpCode::QuadVote
                | OpCode::DerivCoarseX
                | OpCode::DerivCoarseY
                | OpCode::DerivFineX
                | OpCode::DerivFineY
                | OpCode::CalculateLOD
                | OpCode::Sample
                | OpCode::SampleBias
                | OpCode::SampleCmp
                | OpCode::SampleCmpBias
                | OpCode::WriteSamplerFeedback
                | OpCode::WriteSamplerFeedbackBias
        ) as u32;
        if x.op != Op::Call || v.0 as usize + 1 != self.body.insts.len() || x.flags & flag::EFFECT != 0 {
            return v;
        }
        match x.a.kind() {
            ValueKind::Func(f) if self.m.funcs[f as usize].attrs == FnAttrs::NounwindReadnone => {}
            _ => return v,
        }
        let name = crate::dxop::name(op);
        // the lane count and a lane's own index are the same wherever they are asked for
        let lane_constant = matches!(op, OpCode::WaveGetLaneCount | OpCode::WaveGetLaneIndex);
        if !lane_constant && (name.starts_with("Wave") || name.starts_with("Quad") || name.starts_with("Deriv") || name.starts_with("CalculateLOD")) {
            return v;
        }
        let (first, n) = (x.b.0 as usize, x.c.0 as usize);
        let mut hash = mix(mix(op_code(Op::Call), x.ty.0), x.a.0);
        for k in first..first + n {
            hash = mix(hash, self.body.operands[k].0);
        }
        let body = &self.body;
        let found = self.vn.find(hash, |i| {
            let y = &body.insts[i as usize];
            y.op == Op::Call && y.a == x.a && y.ty == x.ty && y.flags == x.flags && y.c == x.c && body.operands[y.b.0 as usize..y.b.0 as usize + n] == body.operands[first..first + n]
        });
        match found {
            Some(i) => {
                // take the duplicate back out: it is the last instruction of everything
                self.body.insts.pop();
                self.body.blocks[self.bb.0 as usize].insts.pop();
                self.body.operands.truncate(first);
                Value::inst(i)
            }
            None => {
                self.vn.insert(hash, v.0);
                v
            }
        }
    }

    /// `v`, the last instruction: a load at `index` of `table`, an internal constant global
    /// (through the GEP just before it, or of `table` itself when `index` is NONE). The table
    /// never changes, so the same element read where an earlier read dominates is that read.
    pub fn number_table_load(&mut self, v: Value, table: Value, index: Value) -> Value {
        let x = self.body.insts[v.0 as usize];
        if x.op != Op::Load || v.0 as usize + 1 != self.body.insts.len() {
            return v;
        }
        let hash = mix(mix(mix(0x500, x.ty.0), table.0), index.0);
        let body = &self.body;
        let at = |y: &Inst| -> Option<Value> {
            if y.op != Op::Load {
                return None;
            }
            if y.a == table {
                return Some(Value::NONE);
            }
            let ValueKind::Inst(g) = y.a.kind() else { return None };
            let g = &body.insts[g as usize];
            (g.op == Op::Gep && g.a == table && g.c.0 == 2).then(|| body.operands[g.b.0 as usize + 1])
        };
        if at(&x) != Some(index) {
            return v;
        }
        let found = self.vn.find(hash, |i| {
            let y = &body.insts[i as usize];
            y.ty == x.ty && at(y) == Some(index)
        });
        let Some(i) = found else {
            self.vn.insert(hash, v.0);
            return v;
        };
        // take the duplicate back out, and its GEP when that is the instruction before it
        self.body.insts.pop();
        self.body.blocks[self.bb.0 as usize].insts.pop();
        if x.a.kind() == ValueKind::Inst(v.0 - 1) {
            let g = self.body.insts.pop().expect("the GEP");
            self.body.blocks[self.bb.0 as usize].insts.pop();
            self.body.operands.truncate(g.b.0 as usize);
        }
        Value::inst(i)
    }

    /// Convert one scalar between kinds, as DXC does for an implicit or explicit cast. A
    /// literal float is a double constant until it meets its type (as in DXC, where the
    /// literal type is 64 bits wide), so `from` == LitFloat reads the value's own type.
    pub fn convert_scalar(&mut self, v: Value, from: Sk, to: Sk) -> Value {
        use Sk::*;
        let from = if from == LitFloat && self.ty_of(v) == Ty::F64 { Double } else { from.concrete() };
        // (and what becomes a literal float, a literal int next to one, becomes a double)
        let to = if to == LitFloat { Double } else { to.concrete() };
        if from == to {
            return v;
        }
        let ir = to.ir();
        match (from, to) {
            (Bool, Int | Uint | Int64 | Uint64) => self.cast(CastOp::ZExt, ir, v),
            (Bool, Float | Double) => self.cast(CastOp::UIToFP, ir, v),
            (Int | Uint | Int64 | Uint64, Bool) => {
                let z = self.zero(from);
                self.icmp(IPred::Ne, v, z)
            }
            (Float | Double, Bool) => {
                let z = self.zero(from);
                self.fcmp(FPred::Une, v, z)
            }
            (Int, Uint) | (Uint, Int) | (Int64, Uint64) | (Uint64, Int64) => v,
            (Int, Int64 | Uint64) => self.cast(CastOp::SExt, ir, v),
            (Uint, Int64 | Uint64) => self.cast(CastOp::ZExt, ir, v),
            (Int64 | Uint64, Int | Uint) => self.cast(CastOp::Trunc, ir, v),
            (Int | Int64, Float | Double) => self.cast(CastOp::SIToFP, ir, v),
            (Uint | Uint64, Float | Double) => self.cast(CastOp::UIToFP, ir, v),
            (Float | Double, Int | Int64) => self.cast(CastOp::FPToSI, ir, v),
            (Float | Double, Uint | Uint64) => self.cast(CastOp::FPToUI, ir, v),
            (Float, Double) => self.cast(CastOp::FPExt, ir, v),
            (Double, Float) => self.cast(CastOp::FPTrunc, ir, v),
            _ => v,
        }
    }

    /// Convert a numeric value to another numeric type: per-component kind conversion, a
    /// scalar splats to a vector, a vector truncates to a shorter one (or to a scalar).
    pub fn convert(&mut self, v: &Val, to: HTy) -> Result<Val> {
        if v.ty == to {
            return Ok(v.clone());
        }
        let (HTy::Num(fk, fr, fc), HTy::Num(tk, tr, tc)) = (v.ty, to) else {
            return self.err(format!("cannot convert {:?} to {:?}", v.ty, to));
        };
        let (fn_, tn) = (fr as usize * fc as usize, tr as usize * tc as usize);
        let mut out = [Value::NONE; 16];
        if tn > 16 || (fn_ != 1 && fn_ < tn) || (fr != tr && fn_ != 1 && tn != 1 && fr != 1) {
            return self.err(format!("cannot convert {:?} to {:?}", v.ty, to));
        }
        for (i, slot) in out.iter_mut().enumerate().take(tn) {
            // matrix → smaller matrix keeps (row, column) positions; vectors and scalars index
            let src = if fn_ == 1 {
                0
            } else if fr == tr {
                (i / tc as usize) * fc as usize + i % tc as usize
            } else {
                i
            };
            *slot = self.convert_scalar(v.comps()[src], fk, tk);
        }
        Ok(Val::new(to, &out[..tn]))
    }

    /// A literal value as a concrete one: LitInt is int, LitFloat is float (double when
    /// `double` is set: the other operand's kind). What [`super::intrinsics`] and
    /// [`super::resources`] receive never holds a double-typed literal.
    pub fn concrete(&mut self, v: &Val, double: bool) -> Result<Val> {
        match v.ty {
            HTy::Num(Sk::LitFloat, r, c) => self.convert(v, HTy::Num(if double { Sk::Double } else { Sk::Float }, r, c)),
            _ => Ok(v.clone()),
        }
    }
}

impl Default for Emit {
    fn default() -> Self {
        Self::new()
    }
}

/// Fold an integer binary operation on `bits`-wide operands (None: not folded, e.g. a division
/// by zero or an over-wide shift, whose result the GPU decides).
pub fn fold_int(op: BinOp, bits: u32, x: u64, y: u64) -> Option<u64> {
    if bits == 0 || bits > 64 {
        return None;
    }
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let sx = |v: u64| ((v << (64 - bits)) as i64) >> (64 - bits);
    let r = match op {
        BinOp::Add => x.wrapping_add(y),
        BinOp::Sub => x.wrapping_sub(y),
        BinOp::Mul => x.wrapping_mul(y),
        BinOp::And => x & y,
        BinOp::Or => x | y,
        BinOp::Xor => x ^ y,
        BinOp::Shl if y < bits as u64 => x << y,
        BinOp::LShr if y < bits as u64 => x >> y,
        BinOp::AShr if y < bits as u64 => (sx(x) >> y) as u64,
        BinOp::UDiv if y != 0 => x / y,
        BinOp::URem if y != 0 => x % y,
        BinOp::SDiv if y != 0 && !(sx(x) == i64::MIN >> (64 - bits) && sx(y) == -1) => sx(x).wrapping_div(sx(y)) as u64,
        BinOp::SRem if y != 0 && !(sx(x) == i64::MIN >> (64 - bits) && sx(y) == -1) => sx(x).wrapping_rem(sx(y)) as u64,
        _ => return None,
    };
    Some(r & mask)
}
