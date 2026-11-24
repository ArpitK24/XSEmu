//! DXC's `dx.break`: the `break` that stays inside its loop.
//!
//! A wave operation counts the lanes that are active where it runs, and for the driver's
//! compiler a block is inside a loop when it can reach the loop's back edge. A block that
//! ends in `break` cannot, so a wave operation in it would see every lane that has left the
//! loop so far instead of the lanes that break in this iteration. DXC therefore ends such a
//! block in a conditional branch the driver cannot fold:
//!
//! ```text
//! @dx.break.cond = internal constant [1 x i32] zeroinitializer
//!   %1 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.break.cond, i32 0, i32 0)
//!   %2 = icmp eq i32 %1, 0            ; both first in the entry block, once per function
//!   br i1 %2, label %exit, label %continue_block
//! ```
//!
//! and takes the condition out again wherever no wave operation depends on it. What is here
//! follows DXC's source rule for rule, because a container is compared with DXC's by what its
//! wave operations return:
//!
//! * `EmitBreakStmt` / `EmitHLSLCondBreak`: only in pixel and compute shaders, only a `break`
//!   that leaves a loop (not one that leaves a `switch`);
//! * `AddDxBreak`: only in a source function whose own body calls a wave-sensitive intrinsic
//!   (`analysis::WAVE_SENSITIVE`; a call in a function it calls does not count);
//! * `CleanupDxBreak`, which runs after inlining and mem2reg: the condition stays when the
//!   basic block that ends in the `break` holds a wave-sensitive call or an instruction that
//!   uses the result of one, directly or through other instructions, where the call is in the
//!   loop the `break` leaves or in a loop inside it, and every instruction on the way is in
//!   some loop. A value that is only copied is no instruction: `n = w; break;` does not
//!   count, `n = w + 1; break;` does, and so does a phi of `w` at the top of the block.
//!
//! "The basic block" is clang's, before any clean-up, which is not always a block here: an
//! `if` lowered to selects, `&&`, `||` and `?:` lowered to `and`, `or` and `select` all
//! start blocks in DXC. So the lowering tells [`Emit`] where DXC's blocks begin
//! ([`Emit::boundary`]), [`Emit`] records what the operations of the current one read
//! ([`Emit::touch`]), and a `break` that qualifies by the first two rules is emitted in the
//! conditional form with that record ([`Lower::dx_break`]). When the function is finished,
//! [`finish`] follows the results of the wave operations through the instructions, as
//! `CleanupDxBreak` does, and turns the breaks that do not depend on one back into plain
//! branches. No break is a candidate unless its loop runs a wave-sensitive intrinsic, so
//! programs without that shape pay nothing and their output does not change.
//!
//! Known differences, none of which a corpus program has: DXC counts dead computations in
//! the block (they are instructions when its pass runs) and the bool conversions clang puts
//! around a copy of a `bool`; here an expression that folds to a constant is no instruction,
//! and a phi counts when it has any use.

use super::super::emit::Emit;
use super::ssa::{JoinId, Term};
use super::Lower;
use crate::ir::{Block, Body, IPred, Inst, Op, Ty, Value};

struct LoopCtx {
    exit: JoinId,
    /// what to restore when the loop is left
    track: bool,
    outer: u32,
}

struct Candidate {
    /// the block that ends in the conditional branch, and its false target
    block: Block,
    other: Block,
    /// the loop the break leaves
    loop_id: u32,
    /// the instructions created in DXC's block, and what its operations read
    range: (u32, u32),
    touched: Vec<Value>,
}

pub(super) struct State {
    /// a pixel or compute shader that runs a wave-sensitive intrinsic somewhere
    pub on: bool,
    /// the body of the source function being lowered calls a wave-sensitive intrinsic
    pub fn_lex: bool,
    loops: Vec<LoopCtx>,
    /// by loop: the loop around it (0: none; entry 0 is unused)
    parent: Vec<u32>,
    candidates: Vec<Candidate>,
    /// the `load` and the `icmp` in the entry block, once a candidate needed them
    load: Value,
    cmp: Value,
}

impl Default for State {
    fn default() -> State {
        State { on: false, fn_lex: false, loops: Vec::new(), parent: vec![0], candidates: Vec::new(), load: Value::NONE, cmp: Value::NONE }
    }
}

impl Lower<'_> {
    /// A loop begins: its header is the insertion point and `exit` is where its breaks go.
    pub(super) fn dx_loop_enter(&mut self, wave: bool, exit: JoinId) {
        if !self.dxb.on {
            return;
        }
        let id = self.dxb.parent.len() as u32;
        self.dxb.parent.push(self.e.loop_id);
        self.dxb.loops.push(LoopCtx { exit, track: self.e.track, outer: self.e.loop_id });
        self.e.set_loop(id);
        self.e.track = wave;
        self.e.boundary();
    }

    /// The loop's body is done; its exit join is resolved next.
    pub(super) fn dx_loop_leave(&mut self) {
        if let Some(l) = self.dxb.loops.pop() {
            self.e.track = l.track;
            self.e.set_loop(l.outer);
        }
    }

    /// A `break` to the join `j`: when DXC would make it conditional on `dx.break`, emit it
    /// that way (true branch to the exit, false branch to where `continue` goes) and return
    /// true. [`finish`] decides whether it stays.
    pub(super) fn dx_break(&mut self, j: JoinId) -> bool {
        if !self.e.track || !self.dxb.fn_lex || self.dxb.loops.last().is_none_or(|l| l.exit != j) {
            return false;
        }
        let Some(&latch) = self.continues.last() else { return false };
        if self.dxb.cmp.is_none() {
            // first in the entry block, as DXC places them (its alloca insertion point);
            // the address is filled in by `finish` when a break keeps the condition
            let ptr = self.e.m.ptr(Ty::I32);
            let nowhere = self.e.m.undef(ptr);
            let zero = self.e.c_u32(0);
            let body = &mut self.e.body;
            let load = body.insts.len() as u32;
            body.insts.push(Inst { op: Op::Load, flags: 0, ty: Ty::I32, a: nowhere, b: Value::NONE, c: Value::NONE });
            body.insts.push(Inst { op: Op::ICmp(IPred::Eq), flags: 0, ty: Ty::I1, a: Value::inst(load), b: zero, c: Value::NONE });
            body.blocks[0].insts.splice(0..0, [load, load + 1]);
            self.dxb.load = Value::inst(load);
            self.dxb.cmp = Value::inst(load + 1);
        }
        let next = self.e.body.new_block();
        self.dxb.candidates.push(Candidate {
            block: self.e.bb,
            other: next,
            loop_id: self.e.loop_id,
            range: (self.e.range_start, self.e.body.insts.len() as u32),
            touched: self.e.touched.clone(),
        });
        self.ssa.add_edge(j, self.e.bb, Term::CondTrue { cond: self.dxb.cmp, other: next });
        self.e.bb = next;
        self.ssa.add_edge(latch, next, Term::Br);
        true
    }
}

fn operands(body: &Body, x: &Inst, skip_condition: bool, mut f: impl FnMut(Value)) {
    match x.op {
        Op::Bin(_) | Op::ICmp(_) | Op::FCmp(_) | Op::Store | Op::AtomicRmw(_) => {
            f(x.a);
            f(x.b);
        }
        Op::Cast(_) | Op::Load | Op::ExtractValue | Op::CondBr | Op::Switch => f(x.a),
        Op::Select | Op::CmpXchg => {
            if !skip_condition {
                f(x.a);
            }
            f(x.b);
            f(x.c);
        }
        Op::Call | Op::Gep => {
            f(x.a);
            body.operands_of(x).iter().for_each(|&v| f(v));
        }
        Op::Phi => body.phis[x.a.0 as usize].iter().for_each(|inc| f(inc.0)),
        Op::Alloca | Op::Br | Op::Ret | Op::Unreachable => {}
    }
}

/// Decide every candidate as `CleanupDxBreak` does, turn the ones that do not depend on a
/// wave operation back into plain branches, and give the load its address if one is left.
/// Runs on the finished body, before the forwarding of trivial phis is applied.
pub(super) fn finish(e: &mut Emit, st: State) {
    if st.candidates.is_empty() {
        return;
    }
    let n = e.body.insts.len();
    let mut keep = vec![false; st.candidates.len()];
    // a wave operation outside every loop keeps nothing
    let roots: Vec<(u32, u32)> = e.wave_roots.iter().copied().filter(|r| r.1 != 0).collect();
    if !roots.is_empty() {
        // one bit per loop that holds a wave operation; past 63 loops they share the last
        let mut bit_loops: Vec<Vec<u32>> = Vec::new();
        let mut taint = vec![0u64; n];
        let mut root = vec![false; n];
        for &(i, l) in &roots {
            let b = match bit_loops.iter().position(|x| x.contains(&l)) {
                Some(b) => b,
                None if bit_loops.len() < 64 => {
                    bit_loops.push(vec![l]);
                    bit_loops.len() - 1
                }
                None => {
                    bit_loops[63].push(l);
                    63
                }
            };
            taint[i as usize] |= 1 << b;
            root[i as usize] = true;
        }
        let mut in_loop = vec![false; n];
        for (k, &(first, l)) in e.loop_runs.iter().enumerate() {
            let end = e.loop_runs.get(k + 1).map_or(n, |r| r.0 as usize);
            if l != 0 {
                in_loop[(first as usize).min(n)..end.min(n)].fill(true);
            }
        }
        // the entry block's load and compare are not in a loop, wherever they were created
        for v in [st.load, st.cmp] {
            in_loop[v.0 as usize] = false;
        }
        let mut merge = vec![false; n];
        for &i in &e.merge_sels {
            merge[i as usize] = true;
        }
        // the users of the wave operations, through every instruction that is in a loop:
        // forward passes until nothing changes (a value that comes round a back edge
        // takes one more)
        let body = &e.body;
        loop {
            let mut changed = false;
            for i in 0..n {
                if !in_loop[i] {
                    continue;
                }
                let mut t = taint[i];
                operands(body, &body.insts[i], merge[i], |v| {
                    if v.is_inst() {
                        t |= taint[v.0 as usize];
                    }
                });
                if t != taint[i] {
                    taint[i] = t;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut used = vec![false; n];
        for x in &body.insts {
            operands(body, x, false, |v| {
                if v.is_inst() {
                    used[v.0 as usize] = true;
                }
            });
        }
        for (c, keep) in st.candidates.iter().zip(keep.iter_mut()) {
            // the loops with a wave operation that are the break's loop or inside it
            let mut inside = 0u64;
            for (b, loops) in bit_loops.iter().enumerate() {
                let within = |mut l: u32| {
                    while l != 0 && l != c.loop_id {
                        l = st.parent[l as usize];
                    }
                    l != 0
                };
                if loops.iter().any(|&l| within(l)) {
                    inside |= 1 << b;
                }
            }
            let mut t = 0u64;
            for &v in &c.touched {
                t |= taint[v.0 as usize] | taint.get(body.resolve(v).0 as usize).copied().filter(|_| body.resolve(v).is_inst()).unwrap_or(0);
            }
            for i in c.range.0 as usize..c.range.1 as usize {
                if root[i] || (used[i] && (merge[i] || body.insts[i].op == Op::Phi)) {
                    t |= taint[i];
                }
            }
            *keep = t & inside != 0;
        }
    }
    let body = &mut e.body;
    let mut folded = false;
    for (c, &keep) in st.candidates.iter().zip(&keep) {
        if keep {
            continue;
        }
        folded = true;
        let Some(&t) = body.blocks[c.block.0 as usize].insts.last() else { continue };
        let term = body.insts[t as usize];
        if term.op != Op::CondBr || term.a != st.cmp || term.c.0 != c.other.0 {
            continue;
        }
        body.insts[t as usize] = Inst { op: Op::Br, a: term.b, b: Value::NONE, c: Value::NONE, ..term };
        // the block that only passed the false edge on is gone with it
        let other = &body.blocks[c.other.0 as usize].insts;
        if other.len() != 1 || body.insts[other[0] as usize].op != Op::Br {
            continue;
        }
        let join = body.insts[other[0] as usize].a.0 as usize;
        body.blocks[c.other.0 as usize].insts.clear();
        for k in 0..body.blocks[join].phis.len() {
            let phi = body.blocks[join].phis[k];
            let list = &mut body.phis[body.insts[phi as usize].a.0 as usize];
            list.retain(|inc| inc.1 != c.other);
            if let [(v, _)] = list[..] {
                if v != Value::inst(phi) {
                    body.replace(Value::inst(phi), v);
                }
            }
        }
    }
    // a phi that the edges taken out leave with one value besides itself is that value (the
    // phi of a loop header whose latch phi just went, and what reads it in turn)
    while folded {
        folded = false;
        for b in 0..body.blocks.len() {
            for k in 0..body.blocks[b].phis.len() {
                let phi = Value::inst(body.blocks[b].phis[k]);
                if body.resolve(phi) != phi {
                    continue;
                }
                let mut only = Value::NONE;
                let mut trivial = true;
                for inc in &body.phis[body.insts[phi.0 as usize].a.0 as usize] {
                    let v = body.resolve(inc.0);
                    if v != phi && only.is_none() {
                        only = v;
                    } else if v != phi && v != only {
                        trivial = false;
                        break;
                    }
                }
                if trivial && !only.is_none() {
                    body.replace(phi, only);
                    folded = true;
                }
            }
        }
    }
    if keep.contains(&true) {
        let array = e.m.array(1, Ty::I32);
        let init = e.m.zero(array);
        let global = e.m.global("dx.break.cond", array, 0, Some(init), true);
        // DXC states no alignment on the global or on the load
        if let Some(g) = e.m.globals.last_mut() {
            g.unnamed_addr = false;
            g.align = 0;
        }
        let zero = e.c_u32(0);
        let ptr = e.m.ptr(Ty::I32);
        let address = e.m.const_gep(ptr, global, &[zero, zero]);
        e.body.insts[st.load.0 as usize].a = address;
        e.body.aligns.insert(st.load.0, 0);
        // and has the two first in the entry block, before the handles created since
        let entry = &mut e.body.blocks[0].insts;
        entry.retain(|&i| i != st.load.0 && i != st.cmp.0);
        entry.splice(0..0, [st.load.0, st.cmp.0]);
    }
}
