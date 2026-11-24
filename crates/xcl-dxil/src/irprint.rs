//! LLVM-style text of the IR, for tests and debugging (owner: the bitcode subagent).
//!
//! The layout follows LLVM 3.7's assembly writer closely enough to diff against
//! `dxc -dumpbin`: unnamed blocks and values share one `%N` counter in block order (the entry
//! block is `%0`), constants print as LLVM prints them (`float 1.000000e+00`, hex for floats
//! that have no exact short decimal), metadata tuples are numbered from the named metadata
//! in pre-order. It is not meant to round-trip through `llvm-as`.
//!
//! [`function`] and [`module`] print what [`crate::bcwriter`] writes: forwarding resolved,
//! unreachable blocks and dead instructions left out. [`function_all`] prints every block and
//! instruction of a body as it was built (forwarding still resolved in operands).

use crate::bcwriter::{align_of, liveness, successors, Live};
use crate::ir::*;
use std::collections::HashMap;
use std::fmt::Write;

pub const DATALAYOUT: &str = "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64";
pub const TRIPLE: &str = "dxil-ms-dx";

/// The whole module as the writer emits it: only the declarations and globals that live code
/// or metadata references.
pub fn module(m: &Module) -> String {
    let mut p = Printer::new(m);
    p.module();
    p.out
}

/// One body as the writer emits it, without the `define` line: live instructions of reachable
/// blocks, numbered as LLVM numbers them.
pub fn function(m: &Module, body: &Body) -> String {
    let mut p = Printer::new(m);
    let live = liveness(body);
    p.body(body, Some(&live));
    p.out
}

/// Like [`function`], but with every block and instruction, dead or not.
pub fn function_all(m: &Module, body: &Body) -> String {
    let mut p = Printer::new(m);
    p.body(body, None);
    p.out
}

/// The text of a type (`%dx.types.Handle`, `[4 x float]*`, ...).
pub fn ty(m: &Module, t: Ty) -> String {
    let mut p = Printer::new(m);
    p.ty(t);
    p.out
}

/// The text of a constant, global or function operand with its type (`i32 7`).
pub fn value(m: &Module, v: Value) -> String {
    let mut p = Printer::new(m);
    p.typed(None, &[], v);
    p.out
}

struct Printer<'a> {
    m: &'a Module,
    out: String,
    /// slot of each metadata node that has one (tuples reachable from named metadata)
    md_slots: HashMap<u32, u32>,
    /// the slot of the `!{i32 1}` node `!dx.precise` points at, when printing a module
    precise_slot: Option<u32>,
    /// the slot of the `!dx.controlflow.hints` node of each set of hints, when printing a
    /// module
    hint_slots: HashMap<u8, u32>,
    /// the slot of the node all `!llvm.loop` nodes hold, when printing a module, and the number
    /// of loops printed so far (each has the next slot)
    loop_slot: Option<u32>,
    loops: u32,
}

/// LLVM's `%e`: six digits, exponent of at least two digits with a sign.
fn c_exp(v: f64) -> String {
    let s = format!("{v:.6e}");
    match s.split_once('e') {
        Some((mant, exp)) => {
            let e: i32 = exp.parse().unwrap_or(0);
            format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
        }
        None => s,
    }
}

fn plain_name(s: &str) -> bool {
    !s.is_empty() && !s.as_bytes()[0].is_ascii_digit() && s.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'$' | b'.' | b'_'))
}

fn escaped(out: &mut String, s: &str) {
    for c in s.bytes() {
        if (0x20..0x7f).contains(&c) && c != b'"' && c != b'\\' {
            out.push(c as char);
        } else {
            let _ = write!(out, "\\{c:02X}");
        }
    }
}

impl<'a> Printer<'a> {
    fn new(m: &'a Module) -> Self {
        Printer { m, out: String::new(), md_slots: HashMap::new(), precise_slot: None, hint_slots: HashMap::new(), loop_slot: None, loops: 0 }
    }

    fn name(&mut self, sigil: char, s: &str) {
        self.out.push(sigil);
        if plain_name(s) {
            self.out.push_str(s);
        } else {
            self.out.push('"');
            escaped(&mut self.out, s);
            self.out.push('"');
        }
    }

    fn ty(&mut self, t: Ty) {
        let m = self.m;
        match m.types.get(t.0 as usize) {
            None => {
                let _ = write!(self.out, "<badtype {}>", t.0);
            }
            Some(TypeDef::Void) => self.out.push_str("void"),
            Some(TypeDef::Int(b)) => {
                let _ = write!(self.out, "i{b}");
            }
            Some(TypeDef::Half) => self.out.push_str("half"),
            Some(TypeDef::Float) => self.out.push_str("float"),
            Some(TypeDef::Double) => self.out.push_str("double"),
            Some(TypeDef::Label) => self.out.push_str("label"),
            Some(TypeDef::Metadata) => self.out.push_str("metadata"),
            Some(TypeDef::Ptr { to, space }) => {
                self.ty(*to);
                if *space != 0 {
                    let _ = write!(self.out, " addrspace({space})");
                }
                self.out.push('*');
            }
            Some(TypeDef::Array { n, of }) => {
                let _ = write!(self.out, "[{n} x ");
                self.ty(*of);
                self.out.push(']');
            }
            Some(TypeDef::Vector { n, of }) => {
                let _ = write!(self.out, "<{n} x ");
                self.ty(*of);
                self.out.push('>');
            }
            Some(TypeDef::Struct { name: Some(n), .. }) => self.name('%', n),
            Some(TypeDef::Struct { name: None, fields }) => self.struct_body(fields),
            Some(TypeDef::Func { ret, params }) => {
                self.ty(*ret);
                self.out.push_str(" (");
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        self.out.push_str(", ");
                    }
                    self.ty(*p);
                }
                self.out.push(')');
            }
        }
    }

    fn struct_body(&mut self, fields: &[Ty]) {
        if fields.is_empty() {
            self.out.push_str("{}");
            return;
        }
        self.out.push_str("{ ");
        for (i, f) in fields.iter().enumerate() {
            if i > 0 {
                self.out.push_str(", ");
            }
            self.ty(*f);
        }
        self.out.push_str(" }");
    }

    /// The type of a value; None for what has no interned type here (functions).
    fn value_ty(&self, body: Option<&Body>, v: Value) -> Option<Ty> {
        match v.kind() {
            ValueKind::Inst(i) => body.and_then(|b| b.insts.get(i as usize)).map(|x| x.ty),
            ValueKind::Const(i) => self.m.consts.get(i as usize).map(|c| c.ty),
            ValueKind::Global(i) => self.m.globals.get(i as usize).map(|g| g.ptr_ty),
            ValueKind::Func(_) => None,
        }
    }

    fn pointee(&self, t: Option<Ty>) -> Option<Ty> {
        match self.m.types.get(t?.0 as usize) {
            Some(TypeDef::Ptr { to, .. }) => Some(*to),
            _ => None,
        }
    }

    fn opt_ty(&mut self, t: Option<Ty>) {
        match t {
            Some(t) => self.ty(t),
            None => self.out.push_str("<notype>"),
        }
    }

    fn type_of_value(&mut self, body: Option<&Body>, v: Value) {
        match v.kind() {
            ValueKind::Func(i) if !v.is_none() => match self.m.funcs.get(i as usize) {
                Some(f) => {
                    self.ty(f.ty);
                    self.out.push('*');
                }
                None => self.out.push_str("<badfunc>"),
            },
            _ => {
                let t = self.value_ty(body, v);
                self.opt_ty(t)
            }
        }
    }

    fn constant(&mut self, i: u32) {
        let m = self.m;
        let Some(c) = m.consts.get(i as usize) else {
            let _ = write!(self.out, "<badconst {i}>");
            return;
        };
        match &c.c {
            Const::Int(v) => {
                let bits = m.bits(c.ty);
                if bits == 1 {
                    self.out.push_str(if *v & 1 != 0 { "true" } else { "false" });
                } else {
                    let s = if bits == 0 || bits >= 64 { *v as i64 } else { ((*v << (64 - bits)) as i64) >> (64 - bits) };
                    let _ = write!(self.out, "{s}");
                }
            }
            Const::Float(bits) => match c.ty {
                Ty::HALF => {
                    let _ = write!(self.out, "0xH{:04X}", *bits as u16);
                }
                _ => {
                    let v = if c.ty == Ty::F32 { f32::from_bits(*bits as u32) as f64 } else { f64::from_bits(*bits) };
                    let s = c_exp(v);
                    if v.is_finite() && s.parse::<f64>().map(|r| r.to_bits() == v.to_bits()).unwrap_or(false) {
                        self.out.push_str(&s);
                    } else {
                        let _ = write!(self.out, "0x{:016X}", v.to_bits());
                    }
                }
            },
            Const::Undef => self.out.push_str("undef"),
            Const::Zero => self.out.push_str(if matches!(m.types.get(c.ty.0 as usize), Some(TypeDef::Ptr { .. })) { "null" } else { "zeroinitializer" }),
            // constant expressions (see Module::const_bitcast, Module::const_gep)
            Const::Aggregate(operands) if operands.len() == 1 && matches!(m.types.get(c.ty.0 as usize), Some(TypeDef::Ptr { .. })) => {
                self.out.push_str("bitcast (");
                self.typed(None, &[], operands[0]);
                self.out.push_str(" to ");
                self.ty(c.ty);
                self.out.push(')');
            }
            Const::Aggregate(operands) if matches!(m.types.get(c.ty.0 as usize), Some(TypeDef::Ptr { .. })) => {
                self.out.push_str("getelementptr inbounds (");
                let source = self.pointee(operands.first().and_then(|&base| self.value_ty(None, base)));
                self.opt_ty(source);
                for v in operands {
                    self.out.push_str(", ");
                    self.typed(None, &[], *v);
                }
                self.out.push(')');
            }
            Const::Aggregate(fields) => {
                let (open, close) = match m.types.get(c.ty.0 as usize) {
                    Some(TypeDef::Array { .. }) => ("[", "]"),
                    Some(TypeDef::Vector { .. }) => ("<", ">"),
                    _ => ("{ ", " }"),
                };
                self.out.push_str(open);
                for (k, f) in fields.iter().enumerate() {
                    if k > 0 {
                        self.out.push_str(", ");
                    }
                    self.typed(None, &[], *f);
                }
                self.out.push_str(close);
            }
        }
    }

    /// A value without its type: `%7`, `42`, `@g`.
    fn bare(&mut self, body: Option<&Body>, slots: &[u32], v: Value) {
        let v = body.map_or(v, |b| b.resolve(v));
        if v.is_none() {
            self.out.push_str("<none>");
            return;
        }
        match v.kind() {
            ValueKind::Inst(i) => match slots.get(i as usize) {
                Some(&s) if s != u32::MAX => {
                    let _ = write!(self.out, "%{s}");
                }
                _ => {
                    let _ = write!(self.out, "%<dead {i}>");
                }
            },
            ValueKind::Const(i) => self.constant(i),
            ValueKind::Global(i) => match self.m.globals.get(i as usize) {
                Some(g) => self.name('@', &g.name),
                None => self.out.push_str("@<badglobal>"),
            },
            ValueKind::Func(i) => match self.m.funcs.get(i as usize) {
                Some(f) => self.name('@', &f.name),
                None => self.out.push_str("@<badfunc>"),
            },
        }
    }

    /// A value with its type: `i32 %7`.
    fn typed(&mut self, body: Option<&Body>, slots: &[u32], v: Value) {
        let v = body.map_or(v, |b| b.resolve(v));
        self.type_of_value(body, v);
        self.out.push(' ');
        self.bare(body, slots, v);
    }

    fn label(&mut self, block_slots: &[u32], b: u32) {
        match block_slots.get(b as usize) {
            Some(&s) if s != u32::MAX => {
                let _ = write!(self.out, "%{s}");
            }
            _ => {
                let _ = write!(self.out, "%<deadblock {b}>");
            }
        }
    }

    fn body(&mut self, body: &Body, live: Option<&Live>) {
        let block_live = |b: usize| live.map_or(true, |l| l.blocks.get(b).copied().unwrap_or(false));
        let inst_live = |i: u32| live.map_or(true, |l| l.insts.get(i as usize).copied().unwrap_or(false));
        let mut slots = vec![u32::MAX; body.insts.len()];
        let mut block_slots = vec![u32::MAX; body.blocks.len()];
        let mut preds: Vec<Vec<u32>> = vec![Vec::new(); body.blocks.len()];
        let mut next = 0u32;
        for (b, data) in body.blocks.iter().enumerate() {
            if !block_live(b) {
                continue;
            }
            block_slots[b] = next;
            next += 1;
            for &i in data.phis.iter().chain(&data.insts) {
                let Some(inst) = body.insts.get(i as usize) else { continue };
                if inst_live(i) && inst.ty != Ty::VOID {
                    slots[i as usize] = next;
                    next += 1;
                }
            }
            if let Some(t) = data.insts.last().and_then(|&i| body.insts.get(i as usize)) {
                successors(body, t, |s| {
                    if !preds[s as usize].contains(&(b as u32)) {
                        preds[s as usize].push(b as u32)
                    }
                });
            }
        }
        let mut first = true;
        for (b, data) in body.blocks.iter().enumerate() {
            if !block_live(b) {
                continue;
            }
            if !first {
                let _ = write!(self.out, "\n; <label>:{}", block_slots[b]);
                if !preds[b].is_empty() {
                    self.out.push_str("  ; preds = ");
                    for (k, &p) in preds[b].iter().enumerate() {
                        if k > 0 {
                            self.out.push_str(", ");
                        }
                        self.label(&block_slots, p);
                    }
                }
                self.out.push('\n');
            }
            first = false;
            for &i in data.phis.iter().chain(&data.insts) {
                if !inst_live(i) {
                    continue;
                }
                let Some(inst) = body.insts.get(i as usize) else {
                    let _ = writeln!(self.out, "  <badinst {i}>");
                    continue;
                };
                self.out.push_str("  ");
                if inst.ty != Ty::VOID {
                    let _ = write!(self.out, "%{} = ", slots[i as usize]);
                }
                self.inst(body, inst, &slots, &block_slots, live);
                self.out.push('\n');
            }
        }
    }

    fn inst(&mut self, body: &Body, inst: &Inst, slots: &[u32], block_slots: &[u32], live: Option<&Live>) {
        let b = Some(body);
        let f = inst.flags;
        match inst.op {
            Op::Bin(op) => {
                let (name, float) = match op {
                    BinOp::Add => ("add", false),
                    BinOp::FAdd => ("fadd", true),
                    BinOp::Sub => ("sub", false),
                    BinOp::FSub => ("fsub", true),
                    BinOp::Mul => ("mul", false),
                    BinOp::FMul => ("fmul", true),
                    BinOp::UDiv => ("udiv", false),
                    BinOp::SDiv => ("sdiv", false),
                    BinOp::FDiv => ("fdiv", true),
                    BinOp::URem => ("urem", false),
                    BinOp::SRem => ("srem", false),
                    BinOp::FRem => ("frem", true),
                    BinOp::Shl => ("shl", false),
                    BinOp::LShr => ("lshr", false),
                    BinOp::AShr => ("ashr", false),
                    BinOp::And => ("and", false),
                    BinOp::Or => ("or", false),
                    BinOp::Xor => ("xor", false),
                };
                self.out.push_str(name);
                if float && f & flag::FAST != 0 {
                    self.out.push_str(" fast");
                }
                if !float {
                    for (bit, text) in [(flag::NUW, " nuw"), (flag::NSW, " nsw"), (flag::EXACT, " exact")] {
                        if f & bit != 0 {
                            self.out.push_str(text);
                        }
                    }
                }
                self.out.push(' ');
                self.typed(b, slots, inst.a);
                self.out.push_str(", ");
                self.bare(b, slots, inst.b);
            }
            Op::Cast(op) => {
                self.out.push_str(match op {
                    CastOp::Trunc => "trunc ",
                    CastOp::ZExt => "zext ",
                    CastOp::SExt => "sext ",
                    CastOp::FPToUI => "fptoui ",
                    CastOp::FPToSI => "fptosi ",
                    CastOp::UIToFP => "uitofp ",
                    CastOp::SIToFP => "sitofp ",
                    CastOp::FPTrunc => "fptrunc ",
                    CastOp::FPExt => "fpext ",
                    CastOp::BitCast => "bitcast ",
                });
                self.typed(b, slots, inst.a);
                self.out.push_str(" to ");
                self.ty(inst.ty);
            }
            Op::ICmp(p) => {
                self.out.push_str("icmp ");
                self.out.push_str(match p {
                    IPred::Eq => "eq",
                    IPred::Ne => "ne",
                    IPred::Ugt => "ugt",
                    IPred::Uge => "uge",
                    IPred::Ult => "ult",
                    IPred::Ule => "ule",
                    IPred::Sgt => "sgt",
                    IPred::Sge => "sge",
                    IPred::Slt => "slt",
                    IPred::Sle => "sle",
                });
                self.out.push(' ');
                self.typed(b, slots, inst.a);
                self.out.push_str(", ");
                self.bare(b, slots, inst.b);
            }
            Op::FCmp(p) => {
                self.out.push_str("fcmp ");
                if f & flag::FAST != 0 {
                    self.out.push_str("fast ");
                }
                self.out.push_str(match p {
                    FPred::Oeq => "oeq",
                    FPred::Ogt => "ogt",
                    FPred::Oge => "oge",
                    FPred::Olt => "olt",
                    FPred::Ole => "ole",
                    FPred::One => "one",
                    FPred::Ord => "ord",
                    FPred::Uno => "uno",
                    FPred::Ueq => "ueq",
                    FPred::Ugt => "ugt",
                    FPred::Uge => "uge",
                    FPred::Ult => "ult",
                    FPred::Ule => "ule",
                    FPred::Une => "une",
                });
                self.out.push(' ');
                self.typed(b, slots, inst.a);
                self.out.push_str(", ");
                self.bare(b, slots, inst.b);
            }
            Op::Select => {
                self.out.push_str("select ");
                self.typed(b, slots, inst.a);
                self.out.push_str(", ");
                self.typed(b, slots, inst.b);
                self.out.push_str(", ");
                self.typed(b, slots, inst.c);
            }
            Op::ExtractValue => {
                self.out.push_str("extractvalue ");
                self.typed(b, slots, inst.a);
                let _ = write!(self.out, ", {}", inst.b.0);
            }
            Op::Call => {
                self.out.push_str("call ");
                self.ty(inst.ty);
                self.out.push(' ');
                self.bare(b, slots, inst.a);
                self.out.push('(');
                let (first, n) = (inst.b.0 as usize, inst.c.0 as usize);
                match body.operands.get(first..first.saturating_add(n)) {
                    Some(args) => {
                        for (k, &a) in args.iter().enumerate() {
                            if k > 0 {
                                self.out.push_str(", ");
                            }
                            self.typed(b, slots, a);
                        }
                    }
                    None => self.out.push_str("<badoperands>"),
                }
                self.out.push(')');
                self.precise(f);
            }
            Op::Load => {
                self.out.push_str("load ");
                self.ty(inst.ty);
                self.out.push_str(", ");
                self.typed(b, slots, inst.a);
                self.align(body, inst, Some(inst.ty));
            }
            Op::Store => {
                self.out.push_str("store ");
                self.typed(b, slots, inst.b);
                self.out.push_str(", ");
                self.typed(b, slots, inst.a);
                let vt = self.value_ty(b, body.resolve(inst.b));
                self.align(body, inst, vt);
            }
            Op::Gep => {
                self.out.push_str(if f & flag::NO_INBOUNDS != 0 { "getelementptr " } else { "getelementptr inbounds " });
                let base = body.resolve(inst.a);
                let src = self.pointee(self.value_ty(b, base));
                self.opt_ty(src);
                self.out.push_str(", ");
                self.typed(b, slots, base);
                let (first, n) = (inst.b.0 as usize, inst.c.0 as usize);
                match body.operands.get(first..first.saturating_add(n)) {
                    Some(idx) => {
                        for &a in idx {
                            self.out.push_str(", ");
                            self.typed(b, slots, a);
                        }
                    }
                    None => self.out.push_str(", <badoperands>"),
                }
            }
            Op::Alloca => {
                self.out.push_str("alloca ");
                let t = self.pointee(Some(inst.ty));
                self.opt_ty(t);
                self.align(body, inst, t);
                self.precise(f);
            }
            Op::Phi => {
                self.out.push_str("phi ");
                self.ty(inst.ty);
                let incoming = body.phis.get(inst.a.0 as usize).map(|v| v.as_slice()).unwrap_or(&[]);
                let mut k = 0;
                for &(v, from) in incoming {
                    if live.is_some_and(|l| !l.blocks.get(from.0 as usize).copied().unwrap_or(false)) {
                        continue;
                    }
                    self.out.push_str(if k > 0 { ", [ " } else { " [ " });
                    k += 1;
                    self.bare(b, slots, v);
                    self.out.push_str(", ");
                    self.label(block_slots, from.0);
                    self.out.push_str(" ]");
                }
            }
            Op::AtomicRmw(op) => {
                self.out.push_str("atomicrmw ");
                self.out.push_str(match op {
                    RmwOp::Xchg => "xchg",
                    RmwOp::Add => "add",
                    RmwOp::Sub => "sub",
                    RmwOp::And => "and",
                    RmwOp::Nand => "nand",
                    RmwOp::Or => "or",
                    RmwOp::Xor => "xor",
                    RmwOp::Max => "max",
                    RmwOp::Min => "min",
                    RmwOp::UMax => "umax",
                    RmwOp::UMin => "umin",
                });
                self.out.push(' ');
                self.typed(b, slots, inst.a);
                self.out.push_str(", ");
                self.typed(b, slots, inst.b);
                self.out.push_str(" seq_cst");
            }
            Op::CmpXchg => {
                self.out.push_str("cmpxchg ");
                self.typed(b, slots, inst.a);
                self.out.push_str(", ");
                self.typed(b, slots, inst.b);
                self.out.push_str(", ");
                self.typed(b, slots, inst.c);
                self.out.push_str(" seq_cst seq_cst");
            }
            Op::Br => {
                self.out.push_str("br label ");
                self.label(block_slots, inst.a.0);
                self.loop_mark(f);
            }
            Op::CondBr => {
                self.out.push_str("br ");
                self.typed(b, slots, inst.a);
                self.out.push_str(", label ");
                self.label(block_slots, inst.b.0);
                self.out.push_str(", label ");
                self.label(block_slots, inst.c.0);
                self.hints(f);
                self.loop_mark(f);
            }
            Op::Switch => {
                self.out.push_str("switch ");
                self.typed(b, slots, inst.a);
                self.out.push_str(", label ");
                self.label(block_slots, inst.b.0);
                self.out.push_str(" [\n");
                let vt = self.value_ty(b, body.resolve(inst.a));
                let bits = vt.map_or(32, |t| self.m.bits(t));
                for &(v, to) in body.switches.get(inst.c.0 as usize).map(|v| v.as_slice()).unwrap_or(&[]) {
                    self.out.push_str("    ");
                    self.opt_ty(vt);
                    let s = if bits == 0 || bits >= 64 { v as i64 } else { ((v << (64 - bits)) as i64) >> (64 - bits) };
                    let _ = write!(self.out, " {s}, label ");
                    self.label(block_slots, to.0);
                    self.out.push('\n');
                }
                self.out.push_str("  ]");
                self.hints(f);
            }
            Op::Ret => self.out.push_str("ret void"),
            Op::Unreachable => self.out.push_str("unreachable"),
        }
    }

    /// The alignment of an alloca, load or store of a value of type `ty`.
    fn align(&mut self, body: &Body, inst: &Inst, ty: Option<Ty>) {
        // the instruction's index: it is an element of the body's list
        let at = (inst as *const Inst as usize).wrapping_sub(body.insts.as_ptr() as usize) / std::mem::size_of::<Inst>();
        let align = match body.aligns.get(&(at as u32)) {
            Some(&a) if !body.aligns.is_empty() => a,
            _ => ty.map_or(4, |t| align_of(self.m, t)),
        };
        if align != 0 {
            let _ = write!(self.out, ", align {align}");
        }
    }

    /// The `!dx.precise` of a call (or, in DXC's -O0 output, of an alloca).
    fn precise(&mut self, flags: u8) {
        if flags & flag::PRECISE == 0 {
            return;
        }
        match self.precise_slot {
            Some(s) => {
                let _ = write!(self.out, ", !dx.precise !{s}");
            }
            None => self.out.push_str(", !dx.precise !{i32 1}"),
        }
    }

    /// The `!llvm.loop` of a branch.
    fn loop_mark(&mut self, flags: u8) {
        if flags & flag::LOOP_UNROLL_DISABLE == 0 {
            return;
        }
        match self.loop_slot {
            Some(s) => {
                self.loops += 1;
                let _ = write!(self.out, ", !llvm.loop !{}", s + self.loops);
            }
            None => self.out.push_str(", !llvm.loop !{!{!\"llvm.loop.unroll.disable\"}}"),
        }
    }

    /// The `!dx.controlflow.hints` of a terminator.
    fn hints(&mut self, flags: u8) {
        let set = flags & flag::HINTS;
        if set == 0 {
            return;
        }
        self.out.push_str(", !dx.controlflow.hints ");
        match self.hint_slots.get(&set) {
            Some(s) => {
                let _ = write!(self.out, "!{s}");
            }
            None => self.hint_node(set, None),
        }
    }

    fn hint_node(&mut self, set: u8, slot: Option<u32>) {
        match slot {
            Some(s) => {
                let _ = write!(self.out, "distinct !{{!{s}, ");
            }
            None => self.out.push_str("!{"),
        }
        self.out.push_str("!\"dx.controlflow.hints\"");
        for h in 0..6 {
            if set >> h & 1 != 0 {
                let _ = write!(self.out, ", i32 {}", h + 1);
            }
        }
        self.out.push('}');
    }

    fn md_slot(&mut self, md: Md, next: &mut u32, order: &mut Vec<Md>) {
        let Some(MdNode::Tuple(ops)) = self.m.md.get(md.0 as usize) else { return };
        if self.md_slots.contains_key(&md.0) {
            return;
        }
        self.md_slots.insert(md.0, *next);
        *next += 1;
        order.push(md);
        for op in ops.iter().flatten() {
            self.md_slot(*op, next, order);
        }
    }

    fn md_operand(&mut self, md: Option<Md>) {
        let Some(md) = md else {
            self.out.push_str("null");
            return;
        };
        match self.m.md.get(md.0 as usize) {
            None => self.out.push_str("<badmd>"),
            Some(MdNode::Str(s)) => {
                self.out.push_str("!\"");
                escaped(&mut self.out, s);
                self.out.push('"');
            }
            Some(MdNode::Val(v)) => self.typed(None, &[], *v),
            Some(MdNode::Tuple(_)) => match self.md_slots.get(&md.0) {
                Some(s) => {
                    let _ = write!(self.out, "!{s}");
                }
                None => self.out.push_str("!<unnumbered>"),
            },
        }
    }

    fn module(&mut self) {
        let m = self.m;
        let _ = writeln!(self.out, "target datalayout = \"{DATALAYOUT}\"\ntarget triple = \"{TRIPLE}\"\n");

        // what live code and metadata reference
        let lives: Vec<Option<Live>> = m.funcs.iter().map(|f| f.body.as_ref().map(liveness)).collect();
        let mut used_funcs = vec![false; m.funcs.len()];
        let mut used_globals = vec![false; m.globals.len()];
        let mut any_precise = false;
        let mut hints = 0u64;
        fn mark_value(m: &Module, v: Value, used_funcs: &mut [bool], used_globals: &mut [bool]) {
            match v.kind() {
                ValueKind::Func(i) => {
                    if let Some(u) = used_funcs.get_mut(i as usize) {
                        *u = true;
                    }
                }
                ValueKind::Global(i) => {
                    if let Some(u) = used_globals.get_mut(i as usize) {
                        *u = true;
                    }
                }
                ValueKind::Const(i) => {
                    if let Some(ConstData { c: Const::Aggregate(fields), .. }) = m.consts.get(i as usize) {
                        for &f in fields {
                            mark_value(m, f, used_funcs, used_globals);
                        }
                    }
                }
                ValueKind::Inst(_) => {}
            }
        }
        for (k, f) in m.funcs.iter().enumerate() {
            let (Some(body), Some(live)) = (&f.body, &lives[k]) else { continue };
            used_funcs[k] = true;
            for (i, inst) in body.insts.iter().enumerate() {
                if !live.insts[i] {
                    continue;
                }
                any_precise |= matches!(inst.op, Op::Call | Op::Alloca) && inst.flags & flag::PRECISE != 0;
                if matches!(inst.op, Op::CondBr | Op::Switch) {
                    hints |= 1 << (inst.flags & flag::HINTS);
                }
                let mut vals = [Value::NONE; 3];
                let mut extra: &[Value] = &[];
                match inst.op {
                    Op::Bin(_) | Op::ICmp(_) | Op::FCmp(_) | Op::Store | Op::AtomicRmw(_) => vals = [inst.a, inst.b, Value::NONE],
                    Op::Cast(_) | Op::Load | Op::ExtractValue | Op::CondBr | Op::Switch => vals[0] = inst.a,
                    Op::Select | Op::CmpXchg => vals = [inst.a, inst.b, inst.c],
                    Op::Call | Op::Gep => {
                        vals[0] = inst.a;
                        extra = body.operands.get(inst.b.0 as usize..(inst.b.0 as usize).saturating_add(inst.c.0 as usize)).unwrap_or(&[]);
                    }
                    Op::Phi => {
                        for &(v, _) in body.phis.get(inst.a.0 as usize).map(|v| v.as_slice()).unwrap_or(&[]) {
                            mark_value(m, body.resolve(v), &mut used_funcs, &mut used_globals);
                        }
                    }
                    _ => {}
                }
                for &v in vals.iter().chain(extra) {
                    if !v.is_none() {
                        mark_value(m, body.resolve(v), &mut used_funcs, &mut used_globals);
                    }
                }
            }
        }
        let mut next = 0u32;
        let mut order = Vec::new();
        for (_, nodes) in &m.named_md {
            for &n in nodes {
                self.md_slot(n, &mut next, &mut order);
            }
        }
        for &md in &order {
            if let Some(MdNode::Tuple(ops)) = m.md.get(md.0 as usize) {
                for op in ops.iter().flatten() {
                    if let Some(MdNode::Val(v)) = m.md.get(op.0 as usize) {
                        mark_value(m, *v, &mut used_funcs, &mut used_globals);
                    }
                }
            }
        }
        for (k, g) in m.globals.iter().enumerate() {
            if used_globals[k] {
                if let Some(init) = g.init {
                    // an initialiser can only name constants here; marking keeps the rule uniform
                    let mut none = vec![false; 0];
                    mark_value(m, init, &mut used_funcs, &mut none);
                }
            }
        }
        if any_precise {
            let one = order.iter().copied().find(|md| match m.md.get(md.0 as usize) {
                Some(MdNode::Tuple(ops)) if ops.len() == 1 => {
                    matches!(ops[0].and_then(|o| m.md.get(o.0 as usize)), Some(MdNode::Val(v)) if m.as_const(*v).is_some_and(|c| c.ty == Ty::I32 && c.c == Const::Int(1)))
                }
                _ => false,
            });
            self.precise_slot = Some(match one {
                Some(md) => self.md_slots[&md.0],
                None => next,
            });
        }
        let mut hint_slot = next + (self.precise_slot == Some(next)) as u32;
        for set in 1..64u8 {
            if hints >> set & 1 != 0 {
                self.hint_slots.insert(set, hint_slot);
                hint_slot += 1;
            }
        }
        self.loop_slot = Some(hint_slot);

        let mut any = false;
        for t in &m.types {
            if let TypeDef::Struct { name: Some(n), fields } = t {
                self.name('%', n);
                self.out.push_str(" = type ");
                self.struct_body(fields);
                self.out.push('\n');
                any = true;
            }
        }
        if any {
            self.out.push('\n');
        }

        any = false;
        for (k, g) in m.globals.iter().enumerate() {
            if !used_globals[k] {
                continue;
            }
            any = true;
            self.name('@', &g.name);
            self.out.push_str(" = ");
            self.out.push_str(match (g.internal, g.init.is_some()) {
                (true, _) => "internal ",
                (false, false) => "external ",
                (false, true) => "",
            });
            if g.unnamed_addr {
                self.out.push_str("unnamed_addr ");
            }
            if g.space != 0 {
                let _ = write!(self.out, "addrspace({}) ", g.space);
            }
            self.out.push_str(if g.constant { "constant " } else { "global " });
            self.ty(g.value_ty);
            if let Some(init) = g.init {
                self.out.push(' ');
                self.bare(None, &[], init);
            }
            if g.align != 0 {
                let _ = write!(self.out, ", align {}", g.align);
            }
            self.out.push('\n');
        }
        if any {
            self.out.push('\n');
        }

        let mut groups: Vec<FnAttrs> = Vec::new();
        for (k, f) in m.funcs.iter().enumerate() {
            if !used_funcs[k] {
                continue;
            }
            let group = match f.attrs {
                FnAttrs::None => None,
                a => Some(groups.iter().position(|&g| attrs_text(g) == attrs_text(a)).unwrap_or_else(|| {
                    groups.push(a);
                    groups.len() - 1
                })),
            };
            let (ret, params): (Ty, &[Ty]) = match m.types.get(f.ty.0 as usize) {
                Some(TypeDef::Func { ret, params }) => (*ret, params),
                _ => (Ty::VOID, &[]),
            };
            if let Some(g) = group {
                let _ = writeln!(self.out, "; Function Attrs: {}", attrs_text(groups[g]));
            }
            self.out.push_str(if f.body.is_some() { "define " } else { "declare " });
            self.ty(ret);
            self.out.push(' ');
            self.name('@', &f.name);
            self.out.push('(');
            for (i, p) in params.iter().enumerate() {
                if i > 0 {
                    self.out.push_str(", ");
                }
                self.ty(*p);
                if i == 1 && f.attrs == FnAttrs::Lifetime {
                    self.out.push_str(" nocapture");
                }
            }
            self.out.push(')');
            if let Some(g) = group {
                let _ = write!(self.out, " #{g}");
            }
            match (&f.body, &lives[k]) {
                (Some(body), Some(live)) => {
                    self.out.push_str(" {\n");
                    self.body(body, Some(live));
                    self.out.push_str("}\n\n");
                }
                _ => self.out.push_str("\n\n"),
            }
        }
        for (g, a) in groups.iter().enumerate() {
            let _ = writeln!(self.out, "attributes #{g} = {{ {} }}", attrs_text(*a));
        }
        if !groups.is_empty() {
            self.out.push('\n');
        }

        for (name, nodes) in &m.named_md {
            self.out.push('!');
            if plain_name(name) {
                self.out.push_str(name);
            } else {
                escaped(&mut self.out, name);
            }
            self.out.push_str(" = !{");
            for (i, &n) in nodes.iter().enumerate() {
                if i > 0 {
                    self.out.push_str(", ");
                }
                self.md_operand(Some(n));
            }
            self.out.push_str("}\n");
        }
        if !m.named_md.is_empty() {
            self.out.push('\n');
        }
        for &md in &order {
            let Some(MdNode::Tuple(ops)) = m.md.get(md.0 as usize) else { continue };
            let _ = write!(self.out, "!{} = !{{", self.md_slots[&md.0]);
            for (i, op) in ops.iter().enumerate() {
                if i > 0 {
                    self.out.push_str(", ");
                }
                self.md_operand(*op);
            }
            self.out.push_str("}\n");
        }
        if self.precise_slot == Some(next) {
            let _ = writeln!(self.out, "!{next} = !{{i32 1}}");
        }
        for set in 1..64u8 {
            if let Some(&slot) = self.hint_slots.get(&set) {
                let _ = write!(self.out, "!{slot} = ");
                self.hint_node(set, Some(slot));
                self.out.push('\n');
            }
        }
        if let (Some(slot), true) = (self.loop_slot, self.loops != 0) {
            let _ = writeln!(self.out, "!{slot} = !{{!\"llvm.loop.unroll.disable\"}}");
            for k in 1..=self.loops {
                let _ = writeln!(self.out, "!{0} = distinct !{{!{0}, !{slot}}}", slot + k);
            }
        }
    }
}

fn attrs_text(a: FnAttrs) -> &'static str {
    match a {
        FnAttrs::None => "",
        FnAttrs::Nounwind => "nounwind",
        FnAttrs::NounwindReadnone => "nounwind readnone",
        FnAttrs::NounwindReadonly => "nounwind readonly",
        FnAttrs::NounwindNoreturn => "noreturn nounwind",
        FnAttrs::NounwindNoduplicate => "noduplicate nounwind",
        FnAttrs::Lifetime => "nounwind",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_llvm_does() {
        let mut m = Module::new();
        let cases = [(1.0f32, "float 1.000000e+00"), (0.5, "float 5.000000e-01"), (-2.0, "float -2.000000e+00"), (0.1, "float 0x3FB99999A0000000"), (f32::INFINITY, "float 0x7FF0000000000000")];
        for (v, text) in cases {
            let c = m.const_f32(v);
            assert_eq!(value(&m, c), text);
        }
        let d = m.const_f64(1.0e10);
        assert_eq!(value(&m, d), "double 1.000000e+10");
        let h = m.constant(Ty::HALF, Const::Float(0x3C00));
        assert_eq!(value(&m, h), "half 0xH3C00");
        let neg = m.const_i32(u32::MAX);
        assert_eq!(value(&m, neg), "i32 -1");
        let t = m.const_bool(true);
        assert_eq!(value(&m, t), "i1 true");
    }

    #[test]
    fn body_numbering_liveness_and_forwarding() {
        let mut m = Module::new();
        let mut b = Body::new();
        let fty = m.func_ty(Ty::VOID, vec![Ty::I32, Ty::I32]);
        let sink = m.declare("sink", fty, FnAttrs::Nounwind);
        let (one, two) = (m.const_i32(1), m.const_i32(2));
        let (then, join, dead_block) = (b.new_block(), b.new_block(), b.new_block());
        let x = b.bin(Body::ENTRY, BinOp::Add, Ty::I32, one, two);
        let _dead = b.bin(Body::ENTRY, BinOp::Mul, Ty::I32, x, x);
        let c = b.icmp(Body::ENTRY, IPred::Ult, x, two);
        b.cond_br(Body::ENTRY, c, then, join);
        let y = b.bin(then, BinOp::Shl, Ty::I32, x, one);
        b.br(then, join);
        let phi = b.phi(join, Ty::I32);
        b.phi_add(phi, y, then);
        b.phi_add(phi, x, Body::ENTRY);
        b.phi_add(phi, one, dead_block);
        let trivial = b.phi(join, Ty::I32);
        b.replace(trivial, x);
        b.call(join, Ty::VOID, sink, &[phi, trivial], true, 0);
        b.ret(join);
        b.br(dead_block, join);
        let text = function(&m, &b);
        let expect = "  %1 = add i32 1, 2\n  %2 = icmp ult i32 %1, 2\n  br i1 %2, label %3, label %5\n\n; <label>:3  ; preds = %0\n  %4 = shl i32 %1, 1\n  br label %5\n\n; <label>:5  ; preds = %0, %3\n  %6 = phi i32 [ %4, %3 ], [ %1, %0 ]\n  call void @sink(i32 %6, i32 %1)\n  ret void\n";
        assert_eq!(text, expect);
        let all = function_all(&m, &b);
        assert!(all.contains("mul i32 %1, %1"), "{all}");
        assert!(all.contains("[ 1, %"), "{all}");
    }

    #[test]
    fn module_text() {
        let mut m = Module::new();
        let handle = {
            let p = m.ptr(Ty::I8);
            m.named_struct("dx.types.Handle", vec![p])
        };
        let arr = m.array(2, Ty::F32);
        let (a, c) = (m.const_f32(1.0), m.const_f32(2.0));
        let init = m.aggregate(arr, vec![a, c]);
        let table = m.global("TABLE", arr, 0, Some(init), true);
        let _unused = m.global("UNUSED", Ty::I32, 3, None, false);
        let fty = m.func_ty(Ty::F32, vec![Ty::I32, Ty::F32]);
        let unary = m.declare("dx.op.unary.f32", fty, FnAttrs::NounwindReadnone);
        let fty = m.func_ty(Ty::VOID, vec![Ty::I32, handle, Ty::F32]);
        let store = m.declare("dx.op.storeOutput.f32", fty, FnAttrs::Nounwind);
        let mut b = Body::new();
        let zero = m.const_i32(0);
        let pf = m.ptr(Ty::F32);
        let p = b.gep(Body::ENTRY, pf, table, &[zero, zero]);
        let v = b.load(Body::ENTRY, Ty::F32, p);
        let six = m.const_i32(6);
        let r = b.call(Body::ENTRY, Ty::F32, unary, &[six, v], false, flag::PRECISE);
        let h = m.undef(handle);
        b.call(Body::ENTRY, Ty::VOID, store, &[zero, h, r], true, 0);
        b.ret(Body::ENTRY);
        let main = m.define("main", b);
        let ver = {
            let (x, y) = (m.md_i32(1), m.md_i32(6));
            m.md_tuple(vec![Some(x), Some(y)])
        };
        m.named("dx.version", vec![ver]);
        let entry = {
            let (f, n) = (m.md_val(main), m.md_str("main"));
            m.md_tuple(vec![Some(f), Some(n), None])
        };
        m.named("dx.entryPoints", vec![entry]);
        let text = module(&m);
        for line in [
            "target triple = \"dxil-ms-dx\"",
            "%dx.types.Handle = type { i8* }",
            "@TABLE = internal unnamed_addr constant [2 x float] [float 1.000000e+00, float 2.000000e+00], align 4",
            "define void @main() {",
            "  %1 = getelementptr inbounds [2 x float], [2 x float]* @TABLE, i32 0, i32 0",
            "  %2 = load float, float* %1, align 4",
            "  %3 = call float @dx.op.unary.f32(i32 6, float %2), !dx.precise !2",
            "  call void @dx.op.storeOutput.f32(i32 0, %dx.types.Handle undef, float %3)",
            "declare float @dx.op.unary.f32(i32, float) #0",
            "declare void @dx.op.storeOutput.f32(i32, %dx.types.Handle, float) #1",
            "attributes #0 = { nounwind readnone }",
            "attributes #1 = { nounwind }",
            "!dx.version = !{!0}",
            "!dx.entryPoints = !{!1}",
            "!0 = !{i32 1, i32 6}",
            "!1 = !{void ()* @main, !\"main\", null}",
            "!2 = !{i32 1}",
        ] {
            assert!(text.lines().any(|l| l == line), "missing line {line:?} in:\n{text}");
        }
        assert!(!text.contains("UNUSED"), "{text}");
    }
}
