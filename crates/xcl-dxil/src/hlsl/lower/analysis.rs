//! What the lowering has to know about code before it lowers it, from one walk over every
//! function:
//!
//! * the variables each loop may assign (its header gets phis for exactly those, see
//!   `super::ssa`); the lowering checks every write against it, so a miss is an error and
//!   never wrong code,
//! * whether an expression may be evaluated where the source would have skipped it: the
//!   operands of `&&`, `||`, `?:` become `and`, `or`, `select` only when they have no side
//!   effect and do not depend on control flow (wave and quad operations, implicit
//!   derivatives, loops).
//!
//! It also gathers, behind `XCL_DXIL_LANE_CHECK`, which values can differ between lanes
//! ([`super::lanes`]).
//!
//! Names are resolved conservatively: an assignment to a name counts for every variable of
//! that name, and a call counts for every overload with that many parameters.

use super::super::ast::{qual, BinOp, Expr, ExprId, Stmt, StmtId, Sym, UnOp, Unit};
use super::lanes::{Lanes, COLLECT, NONE, VONLY};
use std::collections::HashMap;

/// the function (or what it calls) writes a global, memory or a resource, or discards
const IMPURE: u8 = 1;
/// it runs an operation whose result depends on control flow
const CF: u8 = 2;
const LOOP: u8 = 4;
/// it has `out` / `inout` parameters
const OUT: u8 = 8;
/// it fetches from a descriptor heap, or uses a resource that is not a register-bound
/// global (a local or a parameter, which may hold a heap index): where the source guards the
/// access, the index need not be valid without the guard
const HEAP: u8 = 16;
/// it runs (or what it calls runs) an intrinsic DXC marks wave-sensitive: [`WAVE_SENSITIVE`]
const WAVE: u8 = 32;
/// it runs (or what it calls runs) an operation whose result depends on the other lanes of
/// its wave or quad: wave and quad operations, derivatives, samples that take them
const LANE: u8 = 64;
/// it calls (or what it calls calls) `xcl_x_meet`, the meeting of the two halves of a split
/// console wave
const MEET: u8 = 128;
/// The split-wave functions the half check does not look into: the meeting, whose internals
/// are by design, and the exchanges, whose results combine both halves.
const EXCHANGES: &[&str] = &["xcl_x_meet", "xcl_x_ballot", "xcl_x_rla_uniform", "xcl_x_rla_any"];
const NOT_SPECULATABLE: u8 = IMPURE | CF | LOOP | OUT | HEAP;

/// The intrinsics DXC marks wave-sensitive (`[[wv]]` in its intrinsic table: the wave
/// operations whose result depends on which lanes are active). The lane count and index,
/// `WaveReadLaneAt`, the quad operations and the derivatives are not among them.
const WAVE_SENSITIVE: &[&str] = &[
    "WaveIsFirstLane", "WaveActiveAnyTrue", "WaveActiveAllTrue", "WaveActiveAllEqual", "WaveActiveBallot", "WaveReadLaneFirst", "WaveActiveCountBits", "WaveActiveSum", "WaveActiveProduct",
    "WaveActiveBitAnd", "WaveActiveBitOr", "WaveActiveBitXor", "WaveActiveMin", "WaveActiveMax", "WavePrefixCountBits", "WavePrefixSum", "WavePrefixProduct", "WaveMatch", "WaveMultiPrefixBitAnd",
    "WaveMultiPrefixBitOr", "WaveMultiPrefixBitXor", "WaveMultiPrefixCountBits", "WaveMultiPrefixProduct", "WaveMultiPrefixSum",
];

/// Globals of translated programs that the lane check ([`Lanes`]) takes as uniform whatever
/// their assignments read. A wave32 console program on 64-lane hardware runs as two console
/// waves in one hardware wave: `xcl_half = xcl_w32 ? WaveGetLaneIndex() >> 5 : 0u` tells
/// them apart, so it differs between the halves, but each half is a wave of its own: its
/// control flow is uniform within the half, and its wave operations read only its own half
/// (`ballot` picks the half's word, `rla` reads lane `l + 32u * xcl_half`). What the
/// lowering needs is control flow that is uniform within a console wave, so these count as
/// uniform.
pub const UNIFORM_GLOBALS: &[&str] = &["xcl_half", "xcl_w32"];

/// Local arrays up to this many elements are SSA values; larger ones are memory.
pub const SSA_ARRAY_MAX: u64 = 16;

#[derive(Copy, Clone, Default)]
struct FnInfo {
    /// 0: not scanned, 1: being scanned, 2: done
    state: u8,
    flags: u8,
    /// its global writes: a range of `gmods`
    mods: (u32, u32),
}

pub struct Analysis {
    /// assigned names in walk order; each loop owns a range
    events: Vec<Sym>,
    loops: HashMap<u32, (u32, u32)>,
    funcs: Vec<FnInfo>,
    gmods: Vec<Sym>,
    is_global: Vec<bool>,
    /// functions by name: (first, count) in `by_name_list`, indexed by symbol
    by_name: Vec<(u32, u32)>,
    by_name_list: Vec<u32>,
    /// per symbol: 0 unknown, 1 a pure intrinsic, 2 anything else
    intrinsic: Vec<u8>,
    seen: Vec<u32>,
    seen_id: u32,
    cur_flags: u8,
    /// the loop statements that run a wave-sensitive intrinsic (in what they call too)
    wave_loops: std::collections::HashSet<u32>,
    /// by function: its own body calls a wave-sensitive intrinsic
    wave_lex: Vec<bool>,
    cur_lex: bool,
    /// the function being scanned
    cur_fn: u32,
    pub lanes: Lanes,
}

/// Intrinsics that only compute a value from their arguments, the same value wherever they
/// run.
const PURE_INTRINSICS: &[&str] = &[
    "abs", "acos", "all", "and", "any", "asdouble", "asfloat", "asin", "asint", "asuint", "atan", "atan2", "ceil", "clamp", "cos", "cosh", "countbits", "cross", "degrees", "determinant",
    "distance", "dot", "dst", "exp", "exp2", "f16tof32", "f32tof16", "faceforward", "firstbithigh", "firstbitlow", "floor", "fma", "fmod", "frac", "isfinite", "isinf", "isnan", "ldexp",
    "length", "lerp", "lit", "log", "log10", "log2", "mad", "max", "min", "mul", "normalize", "or", "pow", "radians", "rcp", "reflect", "refract", "reversebits", "round", "rsqrt", "saturate",
    "select", "sign", "sin", "sinh", "smoothstep", "sqrt", "step", "tan", "tanh", "transpose", "trunc", "NonUniformResourceIndex", "WaveGetLaneIndex", "WaveGetLaneCount",
];

/// The variable an lvalue expression writes.
pub fn root(u: &Unit, mut x: ExprId) -> Option<Sym> {
    loop {
        match *u.expr(x) {
            Expr::Ident(s) => return Some(s),
            Expr::Member(e, _) | Expr::Index(e, _) => x = e,
            _ => return None,
        }
    }
}

impl Analysis {
    pub fn new(u: &Unit, by_name: Vec<(u32, u32)>, by_name_list: Vec<u32>, is_global: Vec<bool>) -> Analysis {
        // 1: refuse where lanes may leave a loop apart (see `super::thread`); 2: the same,
        // with the chain of what makes the loop's exit vary in the message; 3: the half check
        // instead (`Lanes::half_report`)
        let check = std::env::var("XCL_DXIL_LANE_CHECK").unwrap_or_default();
        let lanes = Lanes::new(matches!(check.as_str(), "1" | "2" | "3"), check != "1", check == "3", u.syms.len(), u.funcs.len(), &is_global);
        let mut a = Analysis {
            events: Vec::new(),
            loops: HashMap::new(),
            funcs: vec![FnInfo::default(); u.funcs.len()],
            gmods: Vec::new(),
            is_global,
            by_name,
            by_name_list,
            intrinsic: vec![0; u.syms.len()],
            seen: vec![0; u.syms.len()],
            seen_id: 0,
            cur_flags: 0,
            wave_loops: Default::default(),
            wave_lex: vec![false; u.funcs.len()],
            cur_lex: false,
            cur_fn: 0,
            lanes,
        };
        if a.lanes.on && !a.lanes.half {
            for name in UNIFORM_GLOBALS {
                if let Some(s) = u.syms.get(name) {
                    a.lanes.pin(s);
                }
            }
        }
        for f in 0..u.funcs.len() {
            a.scan_fn(u, f as u32);
        }
        a.lanes.finish();
        a
    }

    pub(super) fn overloads(&self, name: Sym) -> &[u32] {
        let (start, n) = self.by_name.get(name.0 as usize).copied().unwrap_or((0, 0));
        &self.by_name_list[start as usize..(start + n) as usize]
    }

    fn pure_intrinsic(&mut self, u: &Unit, name: Sym) -> bool {
        let c = &mut self.intrinsic[name.0 as usize];
        if *c == 0 {
            *c = if PURE_INTRINSICS.contains(&u.name(name)) { 1 } else { 2 };
        }
        *c == 1
    }

    fn scan_fn(&mut self, u: &Unit, f: u32) {
        if self.funcs[f as usize].state != 0 {
            return;
        }
        self.funcs[f as usize].state = 1;
        let func = &u.funcs[f as usize];
        let saved = std::mem::replace(&mut self.cur_flags, 0);
        let saved_lex = std::mem::replace(&mut self.cur_lex, false);
        let saved_fn = std::mem::replace(&mut self.cur_fn, f);
        let start = self.events.len();
        if func.params.iter().any(|p| p.quals & qual::OUT != 0) {
            self.cur_flags |= OUT;
        }
        // a parameter of the entry is a stage input: the group id is the same for the wave,
        // the other system values are not; a parameter of anything else is a parameter
        let starts = func.params.iter().map(|p| {
            let start = match p.semantic {
                Some(s) if u.name(s).eq_ignore_ascii_case("SV_GroupID") => 0,
                Some(_) => 1,
                None => 2,
            };
            (p.name, start)
        });
        let lanes = self.lanes.enter_fn(f, starts);
        if let Some(body) = func.body {
            for &s in u.block(body) {
                self.stmt(u, s);
            }
        }
        // its writes to globals, once each
        self.seen_id += 1;
        let first = self.gmods.len() as u32;
        for i in start..self.events.len() {
            let s = self.events[i];
            if self.is_global.get(s.0 as usize).copied().unwrap_or(false) && self.seen[s.0 as usize] != self.seen_id {
                self.seen[s.0 as usize] = self.seen_id;
                self.gmods.push(s);
            }
        }
        let n = self.gmods.len() as u32 - first;
        if n != 0 {
            self.cur_flags |= IMPURE;
        }
        let outs = func.params.iter().enumerate().filter(|(_, p)| p.quals & qual::OUT != 0).map(|(i, _)| i);
        self.lanes.leave_fn(f, lanes, outs, &self.gmods[first as usize..]);
        self.funcs[f as usize] = FnInfo { state: 2, flags: self.cur_flags, mods: (first, n) };
        self.wave_lex[f as usize] = self.cur_lex;
        self.cur_flags = saved;
        self.cur_lex = saved_lex;
        self.cur_fn = saved_fn;
    }

    fn assigned(&mut self, u: &Unit, x: ExprId) {
        if let Some(s) = root(u, x) {
            self.events.push(s);
        }
    }

    /// Walk `x` with its reads going to the lane node `to`.
    fn expr_to(&mut self, u: &Unit, x: ExprId, to: u32) {
        let saved = std::mem::replace(&mut self.lanes.tgt, to);
        self.expr(u, x);
        self.lanes.tgt = saved;
    }

    fn stmt(&mut self, u: &Unit, s: StmtId) {
        let on = self.lanes.on;
        if self.lanes.why {
            self.lanes.line = u.stmt_lines.get(s.0 as usize).copied().unwrap_or(0);
        }
        match *u.stmt(s) {
            Stmt::Var(r) => {
                for v in r.of(&u.vars) {
                    let mut n = 1u64;
                    for &d in &v.dims {
                        n = match *u.expr(d) {
                            Expr::Int(k, _) => n.saturating_mul(k),
                            _ => u64::MAX,
                        };
                    }
                    if n > SSA_ARRAY_MAX {
                        // an array in memory: its writes are stores
                        self.cur_flags |= IMPURE;
                    }
                    let n = if on { self.lanes.var(v.name) } else { NONE };
                    if let Some(i) = v.init {
                        self.expr_to(u, i, n);
                    }
                    if on {
                        self.lanes.assign(n);
                        self.lanes.bind(v.name, n);
                    }
                }
            }
            Stmt::Expr(x) => self.expr(u, x),
            Stmt::Block(r) => {
                let mark = self.lanes.scope();
                for &t in u.block(r) {
                    self.stmt(u, t);
                }
                self.lanes.unwind(mark);
            }
            Stmt::If { cond, then, els } => {
                let c = self.lanes.cond_begin();
                self.expr(u, cond);
                let outer = self.lanes.cond_end(c);
                let inner = self.lanes.ctx;
                self.lanes.branch_begin();
                self.stmt(u, then);
                let a = self.lanes.arm_end();
                let mut b = Vec::new();
                if let Some(e) = els {
                    self.lanes.ctx = inner;
                    self.stmt(u, e);
                    b = self.lanes.arm_end();
                }
                self.lanes.branch_end(&[a, b], els.is_none());
                if on {
                    self.lanes.ctx = outer;
                }
            }
            Stmt::For { init, cond, step, body } => {
                let mark = self.lanes.scope();
                if let Some(i) = init {
                    self.stmt(u, i);
                }
                let start = self.events.len() as u32;
                let outer = self.cur_flags & (WAVE | LANE);
                self.cur_flags &= !(WAVE | LANE);
                let ls = self.lanes.loop_begin();
                if let Some(c) = cond {
                    self.expr_to(u, c, ls.x);
                }
                if let Some(x) = step {
                    self.expr(u, x);
                }
                self.stmt(u, body);
                self.lanes.loop_end(ls, s.0, self.cur_flags & LANE != 0);
                self.lanes.unwind(mark);
                self.loops.insert(s.0, (start, self.events.len() as u32));
                if self.cur_flags & WAVE != 0 {
                    self.wave_loops.insert(s.0);
                }
                self.cur_flags |= LOOP | outer;
            }
            Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                let start = self.events.len() as u32;
                let outer = self.cur_flags & (WAVE | LANE);
                self.cur_flags &= !(WAVE | LANE);
                let ls = self.lanes.loop_begin();
                self.expr_to(u, cond, ls.x);
                self.stmt(u, body);
                self.lanes.loop_end(ls, s.0, self.cur_flags & LANE != 0);
                self.loops.insert(s.0, (start, self.events.len() as u32));
                if self.cur_flags & WAVE != 0 {
                    self.wave_loops.insert(s.0);
                }
                self.cur_flags |= LOOP | outer;
            }
            Stmt::Switch { value, cases } => {
                let c = self.lanes.cond_begin();
                self.expr(u, value);
                let outer = self.lanes.cond_end(c);
                let inner = self.lanes.ctx;
                let brk = self.lanes.switch_begin();
                // (the half check: one arm for all the cases, which may fall through)
                self.lanes.branch_begin();
                for c in cases.of(&u.cases) {
                    self.lanes.ctx = inner;
                    for &t in u.block(c.body) {
                        self.stmt(u, t);
                    }
                }
                let arm = self.lanes.arm_end();
                self.lanes.branch_end(&[arm], true);
                self.lanes.switch_end(brk);
                if on {
                    self.lanes.ctx = outer;
                }
            }
            Stmt::Discard => {
                self.cur_flags |= IMPURE | CF;
                if on {
                    self.lanes.on_return();
                }
            }
            Stmt::Return(Some(x)) => {
                let to = if on { self.lanes.on_return() } else { NONE };
                self.expr_to(u, x, to);
            }
            Stmt::Return(None) if on => {
                self.lanes.on_return();
            }
            Stmt::Break if on => self.lanes.on_break(),
            Stmt::Continue if on => self.lanes.on_continue(),
            Stmt::Break | Stmt::Continue | Stmt::Return(None) | Stmt::Empty => {}
        }
    }

    fn expr(&mut self, u: &Unit, x: ExprId) {
        let on = self.lanes.on;
        match *u.expr(x) {
            Expr::Int(..) | Expr::Float(..) | Expr::Bool(_) => {}
            Expr::Ident(s) => {
                if on && self.lanes.tgt != NONE {
                    let n = self.lanes.node(s);
                    self.lanes.read(n);
                }
            }
            Expr::Unary(op, a) => {
                if matches!(op, UnOp::PreInc | UnOp::PreDec | UnOp::PostInc | UnOp::PostDec) {
                    self.assigned(u, a);
                    if let (true, Some(r)) = (on, root(u, a)) {
                        match self.lanes.version(r) {
                            Some(d) => {
                                let old = self.lanes.node(r);
                                self.lanes.edge(old, d);
                                self.lanes.assign(d);
                                self.lanes.set(r, d);
                            }
                            None => {
                                let d = self.lanes.node(r);
                                self.lanes.assign(d);
                            }
                        }
                    }
                }
                self.expr(u, a);
            }
            Expr::Binary(op, a, b) if self.lanes.half && matches!(op, BinOp::LogAnd | BinOp::LogOr) => {
                // (the half check: the right operand runs where the left one says)
                let c = self.lanes.cond_begin();
                self.expr(u, a);
                let outer = self.lanes.cond_end(c);
                let guard = self.lanes.ctx;
                self.lanes.read(guard);
                self.lanes.branch_begin();
                self.expr(u, b);
                let arm = self.lanes.arm_end();
                self.lanes.branch_end(&[arm], true);
                self.lanes.ctx = outer;
            }
            Expr::Binary(_, a, b) => {
                self.expr(u, a);
                self.expr(u, b);
            }
            Expr::Assign(op, l, r) => {
                self.assigned(u, l);
                match root(u, l).filter(|_| on) {
                    // the half check: a new version of a local (a whole one replaces what it held)
                    Some(s) if self.lanes.versioned(s) => {
                        let d = self.lanes.version(s).expect("a local");
                        if op.is_some() || !matches!(*u.expr(l), Expr::Ident(_)) {
                            self.expr_to(u, l, d);
                        }
                        self.expr_to(u, r, d);
                        self.lanes.assign(d);
                        self.lanes.set(s, d);
                        self.lanes.read(d);
                    }
                    Some(s) => {
                        let d = self.lanes.node(s);
                        self.expr_to(u, l, d);
                        self.expr_to(u, r, d);
                        self.lanes.assign(d);
                        self.lanes.read(d);
                    }
                    None => {
                        self.expr(u, l);
                        self.expr(u, r);
                    }
                }
            }
            Expr::Ternary(c, t, f) if self.lanes.half => {
                let cb = self.lanes.cond_begin();
                self.expr(u, c);
                let outer = self.lanes.cond_end(cb);
                let guard = self.lanes.ctx;
                self.lanes.read(guard);
                self.lanes.branch_begin();
                self.expr(u, t);
                let a = self.lanes.arm_end();
                self.expr(u, f);
                let b = self.lanes.arm_end();
                self.lanes.branch_end(&[a, b], false);
                self.lanes.ctx = outer;
            }
            Expr::Ternary(c, t, f) => {
                self.expr(u, c);
                self.expr(u, t);
                self.expr(u, f);
            }
            Expr::Call(name, args) => {
                let args = u.args(args);
                let st = self.lanes.reads.len();
                let saved = if on { std::mem::replace(&mut self.lanes.tgt, COLLECT) } else { NONE };
                let mut bounds = Vec::new();
                for &a in args {
                    if on {
                        bounds.push(self.lanes.reads.len());
                    }
                    self.expr(u, a);
                }
                let mut user = false;
                let mut keep = 0;
                let mut called = Vec::new();
                for k in 0..self.overloads(name).len() {
                    let f = self.overloads(name)[k];
                    let func = &u.funcs[f as usize];
                    if func.params.len() != args.len() {
                        continue;
                    }
                    user = true;
                    if self.funcs[f as usize].state == 0 {
                        self.scan_fn(u, f);
                    }
                    for (p, &a) in func.params.iter().zip(args) {
                        if p.quals & qual::OUT != 0 {
                            self.assigned(u, a);
                        }
                    }
                    if on {
                        // (the reads of an argument reach the results if they depend on
                        // that parameter)
                        let p = self.lanes.fn_reads_params(f);
                        keep |= p;
                        for (i, (prm, &a)) in func.params.iter().zip(args).enumerate() {
                            if let (true, Some(r)) = (prm.quals & qual::OUT != 0, root(u, a)) {
                                let d = self.lanes.node(r);
                                let from = self.lanes.fn_param(f, i) | VONLY;
                                self.lanes.edge(from, d);
                                self.lanes.args_into(&bounds, p, d);
                            }
                        }
                        called.push(f);
                    }
                    let info = self.funcs[f as usize];
                    self.cur_flags |= info.flags & (IMPURE | CF | LOOP | HEAP | WAVE | LANE | MEET);
                    if on {
                        self.lanes.mark_called(f);
                    }
                    for i in info.mods.0..info.mods.0 + info.mods.1 {
                        let s = self.gmods[i as usize];
                        self.events.push(s);
                    }
                }
                if !user && !(self.pure_intrinsic(u, name) && !(u.name(name) == "asuint" && args.len() != 1)) {
                    // an intrinsic that may write its arguments (asuint(d, lo, hi), frexp,
                    // Interlocked*) or depends on where it runs
                    self.cur_flags |= IMPURE | CF;
                    for &a in args {
                        self.assigned(u, a);
                    }
                    if WAVE_SENSITIVE.contains(&u.name(name)) {
                        self.cur_flags |= WAVE;
                        self.cur_lex = true;
                    }
                }
                if !user && lane_sensitive(u.name(name)) {
                    self.cur_flags |= LANE;
                }
                let callee = u.name(name);
                if user && callee == "xcl_x_meet" {
                    self.cur_flags |= MEET;
                }
                let meets = user && (callee == "xcl_x_meet" || called.iter().any(|&f| self.funcs[f as usize].flags & MEET != 0));
                if meets && self.lanes.half && !EXCHANGES.contains(&u.name(u.funcs[self.cur_fn as usize].name)) {
                    self.lanes.meet_site(self.cur_fn);
                }
                if on {
                    if user && self.lanes.half && EXCHANGES[1..].contains(&callee) {
                        // the same in both halves, but for a lane argument that is not
                        let lane = if callee == "xcl_x_ballot" { 0 } else { super::lanes::param_bit(1) };
                        self.lanes.pass_args(st, &bounds, lane, saved);
                    } else if user {
                        self.lanes.pass_args(st, &bounds, keep, saved);
                        for f in called {
                            let ret = self.lanes.fn_ret(f) | VONLY;
                            self.lanes.read(ret);
                        }
                    } else {
                        let first = bounds.get(1).copied().unwrap_or(self.lanes.reads.len());
                        self.intrinsic_lanes(u, u.name(name), args, st, first, saved);
                    }
                }
            }
            Expr::Construct(_, args) | Expr::InitList(args) => {
                for &a in u.args(args) {
                    self.expr(u, a);
                }
            }
            Expr::Cast(_, a) | Expr::Member(a, _) => self.expr(u, a),
            Expr::Method(name, all) => {
                let (obj, args) = u.method(all);
                let st = self.lanes.reads.len();
                let saved = if on { std::mem::replace(&mut self.lanes.tgt, COLLECT) } else { NONE };
                self.expr(u, obj);
                let args = u.args(args);
                for &a in args {
                    self.expr(u, a);
                    self.assigned(u, a);
                }
                let m = u.name(name);
                if on {
                    // GetDimensions writes what the resource says; an atomic operation's
                    // original value is the lane's own (other outputs, such as a status, are
                    // not followed)
                    let atomic_out = m.starts_with("Interlocked") && m != "InterlockedCompareStore" && args.len() >= 3;
                    for (i, &a) in args.iter().enumerate() {
                        let Some(r) = root(u, a) else { continue };
                        if atomic_out && i + 1 == args.len() {
                            let d = self.lanes.node(r);
                            self.lanes.assign_varying(d);
                        } else if m == "GetDimensions" {
                            let d = self.lanes.node(r);
                            self.lanes.reads_into(st, d);
                        }
                    }
                    self.lanes.pass_on(st, saved);
                }
                if matches!(m, "Sample" | "SampleBias" | "SampleCmp" | "CalculateLevelOfDetail" | "CalculateLevelOfDetailUnclamped") {
                    self.cur_flags |= LANE;
                }
                if !pure_method(u.name(name), args.len()) {
                    self.cur_flags |= IMPURE | CF;
                }
                if !self.bound_resource(u, obj) {
                    self.cur_flags |= HEAP;
                }
            }
            Expr::Index(a, i) => {
                if heap(u, a) {
                    self.cur_flags |= HEAP;
                }
                self.expr(u, a);
                self.expr(u, i);
            }
        }
    }

    /// The reads of a call's arguments since `st` do not reach its value: drop them.
    fn drop_arg_reads(&mut self, st: usize, saved: u32) {
        self.lanes.tgt = saved;
        self.lanes.reads.truncate(st);
    }

    /// What the value of a call of the intrinsic `name` and its outputs vary by. `st`: the
    /// first read of its arguments, `first` the first after its first argument.
    fn intrinsic_lanes(&mut self, u: &Unit, name: &str, args: &[ExprId], st: usize, first: usize, saved: u32) {
        let interlocked = name.starts_with("Interlocked");
        // the outputs: an atomic operation's original value is the lane's own; the others
        // are computed from the inputs
        let outs: &[usize] = match (name, args.len()) {
            ("asuint", 3) | ("sincos", 3) => &[1, 2],
            ("frexp" | "modf", 2) => &[1],
            ("InterlockedCompareExchange", 4) => &[3],
            ("InterlockedCompareStore", _) => &[],
            (_, 3) if interlocked => &[2],
            _ => &[],
        };
        for &i in outs {
            if let Some(r) = root(u, args[i]) {
                let d = self.lanes.node(r);
                if interlocked {
                    self.lanes.assign_varying(d);
                } else {
                    self.lanes.reads_into(st, d);
                }
            }
        }
        if interlocked {
            if let Some(r) = args.first().and_then(|&a| root(u, a)) {
                let d = self.lanes.node(r);
                self.lanes.reads_into(st, d);
            }
        }
        match lane_value(name, self.lanes.half) {
            Value::Args => self.lanes.pass_on(st, saved),
            Value::Uniform => self.drop_arg_reads(st, saved),
            Value::Varying => {
                self.drop_arg_reads(st, saved);
                self.lanes.read_varying();
            }
            Value::LaneArg => {
                self.lanes.reads.drain(st..first);
                self.lanes.pass_on(st, saved);
            }
        }
    }

    /// Whether lanes can leave the loop statement `s` at different times while it holds a
    /// lane-sensitive operation (always false unless `XCL_DXIL_LANE_CHECK` is set).
    pub fn loop_varies(&self, s: StmtId) -> bool {
        self.lanes.on && self.lanes.loop_varies(s.0)
    }

    /// Whether any function runs a wave-sensitive intrinsic.
    pub fn any_wave(&self) -> bool {
        self.funcs.iter().any(|f| f.flags & WAVE != 0)
    }
    /// Whether the body of function `f` itself calls a wave-sensitive intrinsic.
    pub fn fn_wave_lex(&self, f: u32) -> bool {
        self.wave_lex[f as usize]
    }
    /// Whether the loop statement `s` runs a wave-sensitive intrinsic, in what it calls too.
    pub fn loop_wave(&self, s: StmtId) -> bool {
        self.wave_loops.contains(&s.0)
    }

    /// The names the loop statement `s` may assign (with repeats).
    pub fn loop_mods(&self, s: StmtId) -> &[Sym] {
        match self.loops.get(&s.0) {
            Some(&(a, b)) => &self.events[a as usize..b as usize],
            None => &[],
        }
    }

    /// Whether evaluating `x` where the source would not is unobservable.
    pub fn speculatable(&mut self, u: &Unit, x: ExprId) -> bool {
        match *u.expr(x) {
            Expr::Int(..) | Expr::Float(..) | Expr::Bool(_) | Expr::Ident(_) => true,
            Expr::Unary(op, a) => !matches!(op, UnOp::PreInc | UnOp::PreDec | UnOp::PostInc | UnOp::PostDec) && self.speculatable(u, a),
            Expr::Binary(_, a, b) => self.speculatable(u, a) && self.speculatable(u, b),
            Expr::Assign(..) => false,
            Expr::Ternary(c, t, f) => self.speculatable(u, c) && self.speculatable(u, t) && self.speculatable(u, f),
            Expr::Call(name, args) => {
                let args = u.args(args);
                let mut user = false;
                for k in 0..self.overloads(name).len() {
                    let f = self.overloads(name)[k] as usize;
                    if u.funcs[f].params.len() == args.len() {
                        user = true;
                        if self.funcs[f].state != 2 || self.funcs[f].flags & NOT_SPECULATABLE != 0 {
                            return false;
                        }
                    }
                }
                if !user && !(self.pure_intrinsic(u, name) && !(u.name(name) == "asuint" && args.len() != 1)) {
                    return false;
                }
                args.iter().all(|&a| self.speculatable(u, a))
            }
            Expr::Construct(_, args) | Expr::InitList(args) => u.args(args).iter().all(|&a| self.speculatable(u, a)),
            Expr::Cast(_, a) | Expr::Member(a, _) => self.speculatable(u, a),
            Expr::Method(name, all) => {
                let (obj, args) = u.method(all);
                pure_method(u.name(name), args.len as usize) && self.bound_resource(u, obj) && u.args(args).iter().all(|&a| self.speculatable(u, a))
            }
            Expr::Index(a, i) => !heap(u, a) && self.speculatable(u, a) && self.speculatable(u, i),
        }
    }

    /// Whether `x` names a global (for the object of a method: a register-bound resource).
    fn bound_resource(&self, u: &Unit, x: ExprId) -> bool {
        matches!(*u.expr(x), Expr::Ident(s) if self.is_global[s.0 as usize])
    }
}

/// Whether `x` is one of the descriptor heaps.
fn heap(u: &Unit, x: ExprId) -> bool {
    matches!(*u.expr(x), Expr::Ident(s) if matches!(u.name(s), "ResourceDescriptorHeap" | "SamplerDescriptorHeap"))
}

/// Resource methods that only read, and read the same wherever they run (no implicit
/// derivatives, no `out` status argument).
fn pure_method(name: &str, args: usize) -> bool {
    match name {
        "Load" | "Load2" | "Load3" | "Load4" => args == 1,
        "SampleLevel" => args == 3,
        "Gather" | "GatherRed" | "GatherGreen" | "GatherBlue" | "GatherAlpha" => args == 2,
        _ => false,
    }
}

/// What a call of an intrinsic varies by, for [`Lanes`].
enum Value {
    /// its arguments
    Args,
    /// nothing: it is the same for every lane of the wave
    Uniform,
    /// it is the lane's own
    Varying,
    /// the lane it reads (`WaveReadLaneAt(value, lane)`)
    LaneArg,
}

/// `half`: for the half check, where a wave operation's result is its own half's.
fn lane_value(name: &str, half: bool) -> Value {
    match name {
        _ if half && name.starts_with("Wave") && name != "WaveGetLaneCount" => Value::Varying,
        "WaveGetLaneIndex" | "WaveIsFirstLane" | "WaveMatch" => Value::Varying,
        "WaveReadLaneAt" => Value::LaneArg,
        _ if name.starts_with("WavePrefix") || name.starts_with("WaveMultiPrefix") || name.starts_with("Quad") => Value::Varying,
        _ if name.starts_with("ddx") || name.starts_with("ddy") || name == "fwidth" || name.starts_with("EvaluateAttribute") => Value::Varying,
        _ if name.starts_with("WaveActive") || name == "WaveReadLaneFirst" || name == "WaveGetLaneCount" => Value::Uniform,
        _ => Value::Args,
    }
}

/// The intrinsics whose result depends on the other lanes of the wave or quad.
fn lane_sensitive(name: &str) -> bool {
    (name.starts_with("Wave") && !matches!(name, "WaveGetLaneIndex" | "WaveGetLaneCount")) || name.starts_with("Quad") || name.starts_with("ddx") || name.starts_with("ddy") || name == "fwidth"
}
