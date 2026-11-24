//! Which values can differ between the lanes of a wave, for [`super::analysis`]: a taint
//! over names that does not follow the order of statements, gathered during the analysis
//! walk (no walk of its own) and closed over at the end.
//!
//! A variable is *varying* when some assignment to it reads a varying source (a lane's own
//! index, a per-lane stage input, the old value an atomic operation returns, a prefix
//! operation, another varying variable) or sits under a branch or in a loop whose condition
//! is varying. Constants, loads at uniform addresses and the wave operations that reduce
//! over the active lanes (`WaveActive*`, `WaveReadLaneFirst`) are uniform, whatever their
//! arguments.
//!
//! The graph: one node per global (by name), per local declaration (names are scoped: a
//! block's declarations end with it), per function result and per condition; an edge
//! `a → b` says "b varies if a does". Conditions are nodes too: every assignment gets an
//! edge from the condition it sits under, which has one from the condition around it. A
//! loop has two, its exit (the condition, and the conditions of every `break` and `return`
//! in it) and its body (the exit, plus what is around the loop); a `continue` makes the
//! body conditional on its condition. A `return` under a condition makes every global the
//! function writes (and its `out` parameters) depend on that condition: the lanes that
//! returned did not write them. Locals do not: after it, only the other lanes run.
//! Equal conditions under the same condition share a node and equal edges are dropped
//! early, which is what keeps this small on translated programs: their registers are few
//! names, and their thousands of `if (pc == N)` are one condition.
//!
//! Calls are not bound to parameters (that would make `load(a)` varying everywhere as soon
//! as one caller passes a per-lane address). A function is summarised instead: which of
//! its parameters its result (or an `out` parameter) depends on (a bit per parameter,
//! [`P`], closed over the function's own edges when its scan ends, which is before any
//! call is looked at). A call reads those arguments, and in any case what the function's
//! result varies by on its own. Inside the function, a value that depends on a parameter
//! counts as possibly varying.
//!
//! The one question asked of all this: can lanes leave this loop at different times
//! ([`Lanes::loop_varies`])? `super::thread` asks it where DXC would thread a way out of a
//! loop that holds a lane-sensitive operation and this lowering does not.

use super::super::ast::{Sym, Unit};
use std::collections::{HashSet, VecDeque};

pub(super) const NONE: u32 = u32::MAX;
/// As the target: gather the reads in [`Lanes::reads`].
pub(super) const COLLECT: u32 = u32::MAX - 1;
/// On a read: only "varying" passes, not "depends on a parameter" (a function's result,
/// whose parameters are not the caller's).
pub(super) const VONLY: u32 = 1 << 31;
/// varying
const V: u32 = 1;
/// depends on a parameter of the function it is in: bit 1 + i for parameter i (the last
/// bit for every parameter from the thirty-first on)
const P: u32 = !1;
pub(super) fn param_bit(i: usize) -> u32 {
    1 << (1 + i.min(30))
}

const FILTER_BITS: u32 = 15;
const CONDS: usize = 1 << 12;

#[derive(Copy, Clone)]
struct CondSlot {
    hash: u64,
    node: u32,
    /// its key in `cond_keys`
    at: u32,
    len: u32,
}

/// What [`Lanes::enter_fn`] puts aside.
pub(super) struct FnSave {
    mark: usize,
    ret: u32,
    root: u32,
    ctx: u32,
    tgt: u32,
    loops: Vec<(u32, u32)>,
    brk: bool,
    edges: usize,
    after: Vec<u32>,
    frames: Vec<Frame>,
    loop_id: u32,
    params: (u32, u32),
}

/// The half check follows the assignments to locals in program order (each one is a node of
/// its own, a version): a branch, a loop or a switch is a frame, and what its arms leave in a
/// local meets in a fresh node after it ([`Lanes::branch_end`], [`Lanes::loop_end`]).
struct Frame {
    /// a loop: its locals get a header node at their first use in it, which the versions at
    /// its end and at its `continue`s feed
    is_loop: bool,
    id: u32,
    /// the declarations from here on are inside it
    undo_mark: usize,
    /// a branch: its changes to outer locals from here on in `vlog`
    log_mark: usize,
    /// a loop: (symbol, header node), and (symbol, version) at each `break`
    headers: Vec<(u32, u32)>,
    breaks: Vec<(u32, u32)>,
}

/// What [`Lanes::loop_begin`] puts aside.
pub(super) struct LoopSave {
    /// the loop's exit node: the target of its condition
    pub x: u32,
    ctx: u32,
    brk: bool,
}

/// What a node stands for, for [`Lanes::why`].
#[derive(Copy, Clone, Default)]
enum Label {
    #[default]
    None,
    Var(Sym),
    Cond(u32),
    LoopExit(u32),
    LoopBody(u32),
    Result(u32),
    Source(u32),
}

pub(super) struct Lanes {
    /// off: nothing is gathered and every question is answered "no"
    pub on: bool,
    /// the half check (`XCL_DXIL_LANE_CHECK=3`): uniform means the same in both halves of a
    /// console wave that runs as two hardware waves ([`Lanes::half_report`])
    pub half: bool,
    /// the calls that reach `xcl_x_meet`: (function, line, the node of their guard)
    sites: Vec<(u32, u32, u32)>,
    /// by function: some call names it
    called: Vec<bool>,
    /// keep what [`Lanes::why`] needs
    pub why: bool,
    /// the line of the statement being walked (kept with `why` only)
    pub line: u32,
    labels: Vec<Label>,
    /// the conditions under which lanes have returned (or discarded) in the function being
    /// scanned: what it writes that outlives it differs for them ([`Lanes::leave_fn`])
    after: Vec<u32>,
    bits: Vec<u32>,
    /// (from, to); `from` may carry [`VONLY`]
    edges: Vec<(u32, u32)>,
    /// the edges seen last, by hash: most repeats end here
    filter: Vec<u64>,
    /// by symbol: the node of the local of that name in scope
    loc: Vec<u32>,
    /// (symbol, what `loc` and `decl_at` held): the declarations to take back, innermost last
    undo: Vec<(u32, u32, u32)>,
    /// by symbol: the index in `undo` of the declaration in scope (the half check)
    decl_at: Vec<u32>,
    frames: Vec<Frame>,
    /// (symbol, its node before): the changes made in the branch frames that are open
    vlog: Vec<(u32, u32)>,
    /// the innermost loop frame's id (0: none), and by symbol the loop whose headers it has
    loop_id: u32,
    next_frame: u32,
    hdr_ok: Vec<u32>,
    seen: Vec<u32>,
    stamp: u32,
    /// the declarations (indices in `undo`) of the parameters of the function being scanned,
    /// which keep one node: what a function leaves in its `out` parameters is read from it
    params: (u32, u32),
    is_global: Vec<bool>,
    /// the node that is always varying
    vary: u32,
    /// where the reads of the expression being walked go: a node, [`NONE`], [`COLLECT`]
    pub tgt: u32,
    /// the condition the statement being walked sits under
    pub ctx: u32,
    pub reads: Vec<u32>,
    conds: Vec<CondSlot>,
    cond_keys: Vec<u32>,
    /// of the function being scanned: its result, and the condition its body starts under
    ret: u32,
    root: u32,
    /// its loops around the statement being walked: (exit, body)
    loops: Vec<(u32, u32)>,
    /// whether a `break` here leaves a loop (not a `switch`)
    brk: bool,
    /// by function: its result node, the first of its parameter nodes in `param_nodes`,
    /// the parameters its result or an `out` parameter depends on ([`param_bit`])
    fn_ret: Vec<u32>,
    fn_params: Vec<u32>,
    fn_p: Vec<u32>,
    param_nodes: Vec<u32>,
    /// the loops that hold a lane-sensitive operation: (statement, exit node)
    lane_loops: Vec<(u32, u32)>,
    varying: HashSet<u32>,
    /// the globals taken as uniform whatever their assignments read ([`Lanes::pin`])
    pinned: Vec<u32>,
}

impl Lanes {
    pub fn new(on: bool, why: bool, half: bool, syms: usize, funcs: usize, is_global: &[bool]) -> Lanes {
        let mut l = Lanes {
            on,
            half: on && half,
            sites: Vec::new(),
            called: vec![false; if on { funcs } else { 0 }],
            why: on && why,
            line: 0,
            labels: Vec::new(),
            after: Vec::new(),
            bits: Vec::new(),
            edges: Vec::new(),
            filter: Vec::new(),
            loc: Vec::new(),
            undo: Vec::new(),
            decl_at: Vec::new(),
            frames: Vec::new(),
            vlog: Vec::new(),
            loop_id: 0,
            next_frame: 1,
            hdr_ok: Vec::new(),
            seen: Vec::new(),
            stamp: 0,
            params: (0, 0),
            is_global: Vec::new(),
            vary: syms as u32,
            tgt: NONE,
            ctx: NONE,
            reads: Vec::new(),
            conds: Vec::new(),
            cond_keys: Vec::new(),
            ret: NONE,
            root: NONE,
            loops: Vec::new(),
            brk: false,
            fn_ret: Vec::new(),
            fn_params: Vec::new(),
            fn_p: Vec::new(),
            param_nodes: Vec::new(),
            lane_loops: Vec::new(),
            varying: HashSet::new(),
            pinned: Vec::new(),
        };
        if on {
            // nodes 0..syms are the globals, by symbol; then the varying one
            l.bits = vec![0; syms + 1];
            l.bits[syms] = V;
            l.filter = vec![u64::MAX; 1 << FILTER_BITS];
            l.loc = vec![NONE; syms];
            l.is_global = is_global.to_vec();
            if l.half {
                l.decl_at = vec![u32::MAX; syms];
                l.hdr_ok = vec![0; syms];
                l.seen = vec![0; syms];
            }
            l.conds = vec![CondSlot { hash: 0, node: NONE, at: 0, len: 0 }; CONDS];
            l.fn_ret = vec![NONE; funcs];
            l.fn_params = vec![0; funcs];
            // a function whose scan has not ended (recursion): taken to depend on them all
            l.fn_p = vec![P; funcs];
            if l.why {
                l.labels = vec![Label::None; syms + 1];
            }
        }
        l
    }

    /// Take the global `s` as uniform, whatever its assignments read: see
    /// [`super::analysis::UNIFORM_GLOBALS`].
    pub fn pin(&mut self, s: Sym) {
        if self.on && self.is_global[s.0 as usize] {
            self.pinned.push(s.0);
        }
    }

    pub fn fresh(&mut self) -> u32 {
        self.bits.push(0);
        if self.why {
            self.labels.push(Label::None);
        }
        self.bits.len() as u32 - 1
    }
    fn label(&mut self, n: u32, l: Label) {
        if self.why {
            self.labels[n as usize] = l;
        }
    }
    /// A fresh node for the declaration of `s`.
    pub fn var(&mut self, s: Sym) -> u32 {
        let n = self.fresh();
        self.label(n, Label::Var(s));
        n
    }

    /// The node of the variable `s` names here.
    #[inline]
    pub fn node(&mut self, s: Sym) -> u32 {
        let n = self.loc[s.0 as usize];
        if n != NONE {
            if self.half {
                self.touch(s.0);
                return self.loc[s.0 as usize];
            }
            return n;
        }
        if self.is_global[s.0 as usize] {
            return s.0;
        }
        // not declared in sight (a parameter of an enclosing construct the walk skips):
        // one node for the rest of the function
        let n = self.var(s);
        self.bind(s, n);
        n
    }

    /// Declare `s` as node `n` until [`Lanes::unwind`] passes this point.
    pub fn bind(&mut self, s: Sym, n: u32) {
        let decl = self.decl_at.get(s.0 as usize).copied().unwrap_or(u32::MAX);
        self.undo.push((s.0, self.loc[s.0 as usize], decl));
        if self.half {
            self.decl_at[s.0 as usize] = self.undo.len() as u32 - 1;
        }
        self.loc[s.0 as usize] = n;
    }
    pub fn scope(&self) -> usize {
        self.undo.len()
    }
    pub fn unwind(&mut self, mark: usize) {
        while self.undo.len() > mark {
            let (s, old, decl) = self.undo.pop().unwrap();
            self.loc[s as usize] = old;
            if self.half {
                self.decl_at[s as usize] = decl;
            }
        }
    }

    /// The half check: a fresh node for an assignment to the local `s`, which takes its place
    /// once [`Lanes::set`] says so (after the right side is walked: it reads the old one).
    /// None for a global, or outside the half check.
    pub fn version(&mut self, s: Sym) -> Option<u32> {
        if !self.half || self.loc[s.0 as usize] == NONE || self.is_param(s.0) {
            return None;
        }
        self.touch(s.0);
        Some(self.var(s))
    }
    /// The local `s` holds node `n` from here on.
    pub fn set(&mut self, s: Sym, n: u32) {
        self.set_in(self.frames.len(), s.0, n);
    }
    /// `set`, logged in the branch frame below `top` that it changes an outer local of.
    fn set_in(&mut self, top: usize, s: u32, n: u32) {
        if let Some(f) = top.checked_sub(1).map(|k| &self.frames[k]) {
            if !f.is_loop && (self.decl_at[s as usize] as usize) < f.undo_mark {
                self.vlog.push((s, self.loc[s as usize]));
            }
        }
        self.loc[s as usize] = n;
    }
    /// The first use of the local `s` in each loop around: its header there, fed by what it
    /// held before the loop.
    /// Whether an assignment to `s` is a version of its own (the half check, a local).
    pub fn versioned(&self, s: Sym) -> bool {
        self.half && self.loc[s.0 as usize] != NONE && !self.is_param(s.0)
    }
    fn is_param(&self, s: u32) -> bool {
        (self.params.0..self.params.1).contains(&self.decl_at[s as usize])
    }
    fn touch(&mut self, s: u32) {
        if self.loop_id == 0 || self.hdr_ok[s as usize] == self.loop_id || self.is_param(s) {
            return;
        }
        let decl = self.decl_at[s as usize] as usize;
        for k in 0..self.frames.len() {
            let f = &self.frames[k];
            if !f.is_loop || decl >= f.undo_mark || f.headers.iter().any(|h| h.0 == s) {
                continue;
            }
            let h = self.fresh();
            self.edge(self.loc[s as usize], h);
            self.frames[k].headers.push((s, h));
            self.set_in(k, s, h);
        }
        self.hdr_ok[s as usize] = self.loop_id;
    }
    pub fn branch_begin(&mut self) {
        if self.half {
            let id = self.next_frame;
            self.next_frame += 1;
            self.frames.push(Frame { is_loop: false, id, undo_mark: self.undo.len(), log_mark: self.vlog.len(), headers: Vec::new(), breaks: Vec::new() });
        }
    }
    /// The end of an arm of the open branch frame: what it changed (symbol, its node at the
    /// end), with every local back as it was at the branch.
    pub fn arm_end(&mut self) -> Vec<(u32, u32)> {
        let Some(f) = self.frames.last().filter(|_| self.half) else { return Vec::new() };
        let mark = f.log_mark;
        self.stamp += 1;
        let mut out = Vec::new();
        while self.vlog.len() > mark {
            let (s, old) = self.vlog.pop().unwrap();
            if self.seen[s as usize] != self.stamp {
                self.seen[s as usize] = self.stamp;
                out.push((s, self.loc[s as usize]));
            }
            self.loc[s as usize] = old;
        }
        out
    }
    /// The branch frame closes: each local an arm changed meets in a fresh node, fed by what
    /// each arm left in it (what it held before, for an arm that did not change it, and for
    /// the way past every arm when `past`).
    pub fn branch_end(&mut self, arms: &[Vec<(u32, u32)>], past: bool) {
        if !self.half {
            return;
        }
        self.frames.pop();
        self.stamp += 1;
        let stamp = self.stamp;
        for (a, arm) in arms.iter().enumerate() {
            for &(s, _) in arm {
                if self.seen[s as usize] == stamp {
                    continue;
                }
                self.seen[s as usize] = stamp;
                let entry = self.loc[s as usize];
                let m = self.fresh();
                let mut missing = past;
                for other in &arms[a..] {
                    match other.iter().find(|c| c.0 == s) {
                        Some(&(_, v)) => self.edge(v, m),
                        None => missing = true,
                    }
                }
                missing |= a != 0;
                if missing {
                    self.edge(entry, m);
                }
                self.set(Sym(s), m);
            }
        }
    }

    #[inline]
    pub fn edge(&mut self, from: u32, to: u32) {
        if from & !VONLY == to || (to < self.vary && self.pinned.contains(&to)) {
            return;
        }
        let k = (from as u64) << 32 | to as u64;
        let slot = (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - FILTER_BITS)) as usize;
        if self.filter[slot] != k {
            self.filter[slot] = k;
            self.edges.push((from, to));
        }
    }

    /// A read of node `n` (which may carry [`VONLY`]) by the expression being walked.
    #[inline]
    pub fn read(&mut self, n: u32) {
        match self.tgt {
            NONE => {}
            COLLECT => self.reads.push(n),
            t => self.edge(n, t),
        }
    }
    /// A read of a per-lane value (with `why`, through a node that says where).
    pub fn read_varying(&mut self) {
        let v = self.source();
        self.read(v);
    }
    fn source(&mut self) -> u32 {
        if !self.why {
            return self.vary;
        }
        let n = self.fresh();
        self.label(n, Label::Source(self.line));
        self.edge(self.vary, n);
        n
    }
    /// An assignment to node `d` here: it sits under the current condition.
    pub fn assign(&mut self, d: u32) {
        self.edge(self.ctx, d);
    }
    pub fn assign_varying(&mut self, d: u32) {
        let v = self.source();
        self.edge(v, d);
        self.edge(self.ctx, d);
    }

    /// Hand the reads gathered since `start` to the target `saved`, which becomes the
    /// target again.
    pub fn pass_on(&mut self, start: usize, saved: u32) {
        self.tgt = saved;
        match saved {
            COLLECT => {}
            NONE => self.reads.truncate(start),
            t => {
                for i in start..self.reads.len() {
                    self.edge(self.reads[i], t);
                }
                self.reads.truncate(start);
            }
        }
    }
    /// The reads gathered since `start` also reach node `d`, which is assigned here.
    pub fn reads_into(&mut self, start: usize, d: u32) {
        for i in start..self.reads.len() {
            self.edge(self.reads[i], d);
        }
        self.edge(self.ctx, d);
    }

    /// Before the condition of an `if` or the value of a `switch` is walked.
    pub fn cond_begin(&mut self) -> (usize, u32) {
        if !self.on {
            return (0, NONE);
        }
        (self.reads.len(), std::mem::replace(&mut self.tgt, COLLECT))
    }
    /// After it: what it read is the condition the statements under it sit under from here.
    /// Returns the condition to go back to ([`Lanes::ctx`]) after them.
    pub fn cond_end(&mut self, (start, tgt): (usize, u32)) -> u32 {
        if !self.on {
            return NONE;
        }
        self.tgt = tgt;
        let reads = &mut self.reads[start..];
        reads.sort_unstable();
        let mut n = 0;
        for i in 0..reads.len() {
            if i == 0 || reads[i] != reads[i - 1] {
                reads[n] = reads[i];
                n += 1;
            }
        }
        self.reads.truncate(start + n);
        let outer = self.ctx;
        if n == 0 {
            return outer;
        }
        let mut hash = outer as u64 ^ 0x51_7C_C1_B7_27_22_0A_95;
        for &r in &self.reads[start..] {
            hash = (hash.rotate_left(5) ^ r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
        let slot = (hash >> 40) as usize & (CONDS - 1);
        let c = self.conds[slot];
        let same = c.node != NONE && c.hash == hash && c.len as usize == n + 1 && {
            let key = &self.cond_keys[c.at as usize..(c.at + c.len) as usize];
            key[0] == outer && key[1..] == self.reads[start..]
        };
        let node = if same {
            c.node
        } else {
            let node = self.fresh();
            self.label(node, Label::Cond(self.line));
            let at = self.cond_keys.len() as u32;
            self.cond_keys.push(outer);
            self.cond_keys.extend_from_slice(&self.reads[start..]);
            self.conds[slot] = CondSlot { hash, node, at, len: n as u32 + 1 };
            self.edge(outer, node);
            for i in start..start + n {
                self.edge(self.reads[i], node);
            }
            node
        };
        self.reads.truncate(start);
        std::mem::replace(&mut self.ctx, node)
    }

    pub fn loop_begin(&mut self) -> LoopSave {
        if !self.on {
            return LoopSave { x: NONE, ctx: NONE, brk: false };
        }
        let (x, body) = (self.fresh(), self.fresh());
        self.label(x, Label::LoopExit(self.line));
        self.label(body, Label::LoopBody(self.line));
        self.edge(self.ctx, body);
        self.edge(x, body);
        self.loops.push((x, body));
        if self.half {
            let id = self.next_frame;
            self.next_frame += 1;
            self.frames.push(Frame { is_loop: true, id, undo_mark: self.undo.len(), log_mark: self.vlog.len(), headers: Vec::new(), breaks: Vec::new() });
            self.loop_id = id;
        }
        LoopSave { x, ctx: std::mem::replace(&mut self.ctx, body), brk: std::mem::replace(&mut self.brk, true) }
    }
    /// `lane`: the loop statement `s` holds a lane-sensitive operation.
    pub fn loop_end(&mut self, save: LoopSave, s: u32, lane: bool) {
        if !self.on {
            return;
        }
        self.loops.pop();
        self.ctx = save.ctx;
        self.brk = save.brk;
        if self.half {
            // what a local holds after the loop: its header (the lanes that leave at the
            // condition) or what it held at a `break`; the header also takes what it holds
            // at the end of the body
            let f = self.frames.pop().expect("the loop's frame");
            self.loop_id = self.frames.iter().rev().find(|f| f.is_loop).map_or(0, |f| f.id);
            for &(s, h) in &f.headers {
                self.edge(self.loc[s as usize], h);
                let m = self.fresh();
                self.edge(h, m);
                for &(_, v) in f.breaks.iter().filter(|b| b.0 == s) {
                    self.edge(v, m);
                }
                self.set(Sym(s), m);
            }
        }
        if lane {
            self.lane_loops.push((s, save.x));
        }
    }
    pub fn switch_begin(&mut self) -> bool {
        std::mem::replace(&mut self.brk, false)
    }
    pub fn switch_end(&mut self, brk: bool) {
        self.brk = brk;
    }
    pub fn on_break(&mut self) {
        if let (true, true, Some(&(x, _))) = (self.on, self.brk, self.loops.last()) {
            self.edge(self.ctx, x);
            if let Some(k) = self.frames.iter().rposition(|f| f.is_loop) {
                for i in 0..self.frames[k].headers.len() {
                    let s = self.frames[k].headers[i].0;
                    let v = self.loc[s as usize];
                    self.frames[k].breaks.push((s, v));
                }
            }
        }
    }
    pub fn on_continue(&mut self) {
        if let (true, Some(&(_, body))) = (self.on, self.loops.last()) {
            self.edge(self.ctx, body);
            if let Some(k) = self.frames.iter().rposition(|f| f.is_loop) {
                for i in 0..self.frames[k].headers.len() {
                    let (s, h) = self.frames[k].headers[i];
                    self.edge(self.loc[s as usize], h);
                }
            }
        }
    }
    /// A `return` (or `discard`): returns the function's result node, the target of the
    /// returned value. The lanes that take it leave every loop around it.
    pub fn on_return(&mut self) -> u32 {
        self.edge(self.ctx, self.ret);
        if self.ctx != self.root {
            self.after.push(self.ctx);
            for i in 0..self.loops.len() {
                self.edge(self.ctx, self.loops[i].0);
            }
        }
        self.ret
    }
    /// The scan of function `f` begins. `params`: per parameter its name and how it starts:
    /// 0 uniform, 1 varying, 2 a parameter like any other.
    pub fn enter_fn(&mut self, f: u32, params: impl Iterator<Item = (Sym, u8)>) -> Option<FnSave> {
        if !self.on {
            return None;
        }
        let (ret, root) = (self.fresh(), self.fresh());
        let save = FnSave {
            mark: self.undo.len(),
            ret: std::mem::replace(&mut self.ret, ret),
            root: std::mem::replace(&mut self.root, root),
            ctx: std::mem::replace(&mut self.ctx, root),
            tgt: std::mem::replace(&mut self.tgt, NONE),
            loops: std::mem::take(&mut self.loops),
            brk: std::mem::replace(&mut self.brk, false),
            edges: self.edges.len(),
            after: std::mem::take(&mut self.after),
            frames: std::mem::take(&mut self.frames),
            loop_id: std::mem::replace(&mut self.loop_id, 0),
            params: self.params,
        };
        self.label(ret, Label::Result(f));
        self.fn_ret[f as usize] = ret;
        self.fn_params[f as usize] = self.param_nodes.len() as u32;
        let mut mine = Vec::new();
        for (i, (name, start)) in params.enumerate() {
            let n = self.var(name);
            self.bits[n as usize] = [0, V, param_bit(i)][start as usize];
            mine.push((name, n));
        }
        // (a callee scanned in between has added its own by now: keep these together)
        let first = self.undo.len() as u32;
        for (name, n) in mine {
            self.param_nodes.push(n);
            self.bind(name, n);
        }
        self.params = (first, self.undo.len() as u32);
        Some(save)
    }
    /// It ends. `outs`: the indices of its `out` and `inout` parameters; `globals`: the
    /// globals it (or what it calls) may write. Where lanes returned early, those differ
    /// for them; a local does not matter, nobody reads it after its function.
    pub fn leave_fn(&mut self, f: u32, save: Option<FnSave>, outs: impl Iterator<Item = usize>, globals: &[Sym]) {
        let Some(save) = save else { return };
        let first = self.fn_params[f as usize] as usize;
        let outs: Vec<usize> = outs.collect();
        for i in 0..self.after.len() {
            let c = self.after[i];
            for g in globals {
                self.edge(c, g.0);
            }
            for &o in &outs {
                self.edge(c, self.param_nodes[first + o]);
            }
        }
        // what depends on the parameters, over the function's own edges
        if self.param_nodes[first..].iter().any(|&n| self.bits[n as usize] & P != 0) {
            self.close(save.edges, V | P);
        }
        let mut p = self.bits[self.ret as usize] & P;
        for i in outs {
            p |= self.bits[self.param_nodes[first + i] as usize] & P;
        }
        self.fn_p[f as usize] = p;
        self.unwind(save.mark);
        self.ret = save.ret;
        self.root = save.root;
        self.ctx = save.ctx;
        self.tgt = save.tgt;
        self.loops = save.loops;
        self.brk = save.brk;
        self.after = save.after;
        self.frames = save.frames;
        self.loop_id = save.loop_id;
        self.params = save.params;
    }
    pub fn mark_called(&mut self, f: u32) {
        self.called[f as usize] = true;
    }
    /// A call in function `f` that reaches `xcl_x_meet`: it sits under the current
    /// condition and under every condition lanes returned on before it.
    pub fn meet_site(&mut self, f: u32) {
        let mut g = self.ctx;
        if !self.after.is_empty() {
            g = self.fresh();
            self.label(g, Label::Cond(self.line));
            self.edge(self.ctx, g);
            for i in 0..self.after.len() {
                self.edge(self.after[i], g);
            }
        }
        self.sites.push((f, self.line, g));
    }
    /// The half check's verdict: the calls (in the entry `entry` and in the functions some
    /// call names) that reach `xcl_x_meet` under a condition that can differ between the
    /// halves, one per line, with why. None when there are none.
    pub fn half_report(&self, u: &Unit, entry: &str) -> Option<String> {
        if !self.half {
            return None;
        }
        let mut lines: Vec<(u32, u32)> = self
            .sites
            .iter()
            .filter(|&&(f, _, g)| self.bits[g as usize] != 0 && (self.called[f as usize] || u.name(u.funcs[f as usize].name) == entry))
            .map(|&(_, l, g)| (l, g))
            .collect();
        lines.sort_unstable();
        lines.dedup_by_key(|s| s.0);
        if lines.is_empty() {
            return None;
        }
        let list: Vec<String> = lines.iter().map(|&(l, g)| format!("line {l}: {}", self.chain(g, u))).collect();
        Some(format!("{} calls that reach xcl_x_meet under a condition that can differ between the halves of a wave: {}", lines.len(), list.join("; ")))
    }
    pub fn fn_ret(&self, f: u32) -> u32 {
        self.fn_ret[f as usize]
    }
    pub fn fn_param(&self, f: u32, i: usize) -> u32 {
        self.param_nodes[self.fn_params[f as usize] as usize + i]
    }
    /// The parameters (as [`param_bit`]s) the results of `f` depend on.
    pub fn fn_reads_params(&self, f: u32) -> u32 {
        self.fn_p[f as usize]
    }

    /// Keep, of the reads since `st` (the arguments of a call, the one at `i` from
    /// `bounds[i]` on), those of the arguments `mask` names, and hand them to `saved`.
    pub fn pass_args(&mut self, st: usize, bounds: &[usize], mask: u32, saved: u32) {
        let mut n = st;
        for (i, &b) in bounds.iter().enumerate() {
            let end = bounds.get(i + 1).copied().unwrap_or(self.reads.len());
            if mask & param_bit(i) != 0 {
                self.reads.copy_within(b..end, n);
                n += end - b;
            }
        }
        self.reads.truncate(n);
        self.pass_on(st, saved);
    }
    /// The reads of the arguments `mask` names also reach node `d`, which is assigned here.
    pub fn args_into(&mut self, bounds: &[usize], mask: u32, d: u32) {
        for (i, &b) in bounds.iter().enumerate() {
            let end = bounds.get(i + 1).copied().unwrap_or(self.reads.len());
            if mask & param_bit(i) != 0 {
                for k in b..end {
                    self.edge(self.reads[k], d);
                }
            }
        }
        self.edge(self.ctx, d);
    }

    /// Carry `mask` along the edges from `first` on, to the end. A global never depends on
    /// a parameter: what a function stores in one from its parameters counts as varying.
    fn close(&mut self, first: usize, mask: u32) {
        let edges = &self.edges[first..];
        // the edges by source: counted over all nodes for the whole graph, sorted for the
        // edges of one function
        let (starts, adj): (Option<Vec<u32>>, Vec<(u32, u32)>) = if first == 0 {
            let mut starts = vec![0u32; self.bits.len() + 1];
            for &(f, _) in edges {
                starts[(f & !VONLY) as usize + 1] += 1;
            }
            for i in 1..starts.len() {
                starts[i] += starts[i - 1];
            }
            let mut fill = starts.clone();
            let mut adj = vec![(0, 0); edges.len()];
            for &(f, t) in edges {
                let at = &mut fill[(f & !VONLY) as usize];
                adj[*at as usize] = (f, t);
                *at += 1;
            }
            (Some(starts), adj)
        } else {
            let mut adj = edges.to_vec();
            adj.sort_unstable_by_key(|e| e.0 & !VONLY);
            (None, adj)
        };
        let mut work: Vec<u32> = adj.iter().map(|e| e.0 & !VONLY).filter(|&f| self.bits[f as usize] & mask != 0).collect();
        work.dedup();
        let globals = self.vary;
        while let Some(m) = work.pop() {
            let range = match &starts {
                Some(s) => s[m as usize] as usize..s[m as usize + 1] as usize,
                None => adj.partition_point(|e| e.0 & !VONLY < m)..adj.partition_point(|e| e.0 & !VONLY <= m),
            };
            let bm = self.bits[m as usize] & mask;
            for &(f, t) in &adj[range] {
                let mut b = if f & VONLY != 0 { bm & V } else { bm };
                if t < globals && b & P != 0 {
                    b = V;
                }
                let tb = &mut self.bits[t as usize];
                if *tb | b != *tb {
                    *tb |= b;
                    work.push(t);
                }
            }
        }
    }

    /// Every function is scanned: close over the whole graph and answer for the loops.
    pub fn finish(&mut self) {
        if !self.on {
            return;
        }
        self.close(0, V);
        for &(s, x) in &self.lane_loops {
            if self.bits[x as usize] != 0 {
                self.varying.insert(s);
            }
        }
    }

    /// Whether the loop statement `s` holds a lane-sensitive operation and lanes may leave
    /// it at different times.
    pub fn loop_varies(&self, s: u32) -> bool {
        self.varying.contains(&s)
    }
}

impl Lanes {
    /// Why the loop statement `s` counts as varying: a chain of what feeds its exit, from a
    /// per-lane source (or a parameter) to the exit. Empty without `why`.
    pub fn why(&self, s: u32, u: &Unit) -> String {
        let Some(&(_, x)) = self.lane_loops.iter().find(|l| l.0 == s).filter(|_| self.why) else { return String::new() };
        self.chain(x, u)
    }

    /// What makes node `x` vary: a chain from a per-lane source (or a parameter) to it.
    fn chain(&self, x: u32, u: &Unit) -> String {
        let n = self.bits.len();
        let mut rev: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &(f, t) in &self.edges {
            if f & VONLY == 0 || self.bits[(f & !VONLY) as usize] & V != 0 {
                rev[t as usize].push(f & !VONLY);
            }
        }
        let params: HashSet<u32> = self.param_nodes.iter().copied().filter(|&p| self.bits[p as usize] != 0).collect();
        let mut parent = vec![NONE; n];
        parent[x as usize] = x;
        let mut queue = VecDeque::from([x]);
        let mut end = NONE;
        // a varying exit is followed through varying nodes to a per-lane source (or to a
        // parameter whose value a function stores in a global); one that depends on a
        // parameter, to the parameter
        let varying = self.bits[x as usize] & V != 0;
        while let Some(m) = queue.pop_front() {
            if m == self.vary || (params.contains(&m) && (!varying || self.bits[m as usize] & V != 0)) {
                end = m;
                break;
            }
            for &p in &rev[m as usize] {
                let b = self.bits[p as usize];
                let fits = if varying { b & V != 0 || (m < self.vary && b & P != 0) } else { b & P != 0 };
                if parent[p as usize] == NONE && fits {
                    parent[p as usize] = m;
                    queue.push_back(p);
                }
            }
        }
        let mut out = Vec::new();
        let mut m = end;
        while m != NONE {
            let text = match self.labels[m as usize] {
                _ if m == self.vary => None,
                _ if m < self.vary => Some(format!("global {}", u.name(Sym(m)))),
                Label::Var(s) => Some(format!("{}{}", u.name(s), if params.contains(&m) { " (a parameter or stage input)" } else { "" })),
                Label::Cond(l) => Some(format!("condition at line {l}")),
                Label::LoopExit(l) => Some(format!("exit of the loop at line {l}")),
                Label::LoopBody(l) => Some(format!("body of the loop at line {l}")),
                Label::Result(f) => Some(format!("result of {}", u.name(u.funcs[f as usize].name))),
                Label::Source(l) => Some(format!("per-lane value at line {l}")),
                Label::None => None,
            };
            out.extend(text);
            if m == x {
                break;
            }
            m = parent[m as usize];
        }
        out.join(" -> ")
    }
}
