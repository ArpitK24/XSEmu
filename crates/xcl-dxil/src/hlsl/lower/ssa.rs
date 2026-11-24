//! SSA construction for structured code.
//!
//! Every scalar component of every variable is a *slot* whose current value is a plain array
//! entry ([`Ssa::cur`]); nothing is looked up per block. Control flow is handled by three
//! mechanisms, each linear in what a construct assigns:
//!
//! * A *region* is a piece of code that may or may not run (a branch, a loop, an inlined
//!   function body up to its returns). The first write to an outer slot inside a region
//!   logs the slot's previous value, so leaving the region can restore the state before it
//!   ([`Ssa::undo_region`]) or keep the writes ([`Ssa::commit_region`]).
//! * An *edge* to a [`Join`] (the block after an `if`, a loop exit, a loop latch, the
//!   point a call returns to) records the values of the slots written since the join was
//!   created. [`Ssa::resolve`] turns the edges into one block with a phi for every slot
//!   whose values differ; a join with a single plain edge is no block at all.
//! * A loop header gets its phis when the loop is entered, for the slots the loop assigns
//!   (known beforehand: `super::analysis`), which every write inside a loop is checked
//!   against. A phi whose incoming values turn out equal is removed with
//!   [`crate::ir::Body::replace`] when the loop is closed.
//!
//! This is Braun et al.'s on-the-fly construction specialised to structured control flow:
//! their per-block definition map and recursive predecessor lookup, which would create and
//! remove a trivial phi at every join between a definition and a use, are replaced by the
//! write logs. The terminators of edges are emitted when the join is resolved, so blocks
//! are created in an order in which every block follows its dominators.

use super::super::emit::Emit;
use crate::ir::{Block, Op, Ty, Value};

#[derive(Copy, Clone, Debug)]
pub enum Term {
    /// `br join`
    Br,
    /// `br cond, other, join`
    CondFalse { cond: Value, other: Block },
    /// `br cond, join, other`
    CondTrue { cond: Value, other: Block },
    /// `switch v, default join [table]`, the table being [`Ssa::switch_tables`]\[table\]
    SwitchDefault { v: Value, table: u32 },
    /// the terminator is emitted by the caller and already names the join's block
    Done,
}

struct Edge {
    block: Block,
    term: Term,
    /// range in the join's `snaps`
    start: u32,
    len: u32,
}

struct Join {
    edges: Vec<Edge>,
    snaps: Vec<(u32, Value)>,
    log_mark: u32,
    slot_limit: u32,
    vn_mark: usize,
    block: Option<Block>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct JoinId(u32);

struct Region {
    uid: u32,
    slot_base: u32,
    log_mark: u32,
    vn_mark: usize,
}

struct LogEntry {
    slot: u32,
    old: Value,
    /// the region the slot was logged in before
    prev: u32,
}

/// What [`Ssa::loop_enter`] returns for [`Ssa::loop_leave`].
#[derive(Copy, Clone)]
pub struct LoopSave {
    uid: u32,
    slot_base: u32,
}

#[derive(Default)]
pub struct Ssa {
    /// the value of each slot at the insertion point
    pub cur: Vec<Value>,
    pub slot_ty: Vec<Ty>,
    /// uid of the innermost region the slot is logged in
    logged_in: Vec<u32>,
    /// uid of the innermost loop that has a header phi for the slot
    loop_mark: Vec<u32>,
    log: Vec<LogEntry>,
    regions: Vec<Region>,
    next_uid: u32,
    joins: Vec<Option<Join>>,
    free_joins: Vec<u32>,
    pool: Vec<(Vec<Edge>, Vec<(u32, Value)>)>,
    stamp: Vec<u32>,
    pos: Vec<u32>,
    stamp_id: u32,
    tmp_slots: Vec<u32>,
    tmp_vals: Vec<Value>,
    loop_uid: u32,
    loop_slot_base: u32,
    /// bit per entry of `Body::phis`: the phi was removed as trivial
    trivial: Vec<u64>,
    pub switch_tables: Vec<Vec<(u64, Block)>>,
    /// a slot was written in a loop whose header has no phi for it: the assignment analysis
    /// missed it, and the function must not be emitted
    pub violation: bool,
}

impl Ssa {
    pub fn new() -> Ssa {
        let mut s = Ssa::default();
        s.next_uid = 2;
        s.regions.push(Region { uid: 1, slot_base: 0, log_mark: 0, vn_mark: 0 });
        s
    }

    pub fn len(&self) -> u32 {
        self.cur.len() as u32
    }
    pub fn alloc(&mut self, ty: Ty, v: Value) -> u32 {
        self.cur.push(v);
        self.slot_ty.push(ty);
        self.logged_in.push(0);
        self.loop_mark.push(0);
        self.cur.len() as u32 - 1
    }
    /// Release the slots from `n` on (the end of their scope).
    pub fn truncate(&mut self, n: u32) {
        let n = n as usize;
        self.cur.truncate(n);
        self.slot_ty.truncate(n);
        self.logged_in.truncate(n);
        self.loop_mark.truncate(n);
    }

    #[inline]
    pub fn read(&self, slot: u32) -> Value {
        self.cur[slot as usize]
    }
    #[inline]
    pub fn write(&mut self, slot: u32, v: Value) {
        let s = slot as usize;
        let r = &self.regions[self.regions.len() - 1];
        if slot < r.slot_base {
            if self.logged_in[s] != r.uid {
                self.log.push(LogEntry { slot, old: self.cur[s], prev: self.logged_in[s] });
                self.logged_in[s] = r.uid;
            }
            if slot < self.loop_slot_base && self.loop_mark[s] != self.loop_uid {
                self.violation = true;
            }
        }
        self.cur[s] = v;
    }
    /// A write whose value only reaches a jump out of every loop of the function being
    /// lowered (the value of a `return`): no loop header needs a phi for it, whatever
    /// loop it is written in.
    pub fn write_leaving(&mut self, slot: u32, v: Value) {
        let violation = self.violation;
        self.write(slot, v);
        self.violation = violation;
    }

    pub fn open_region(&mut self, e: &Emit) {
        let uid = self.next_uid;
        self.next_uid += 1;
        self.regions.push(Region { uid, slot_base: self.cur.len() as u32, log_mark: self.log.len() as u32, vn_mark: e.vn_mark() });
    }
    pub fn region_depth(&self) -> usize {
        self.regions.len()
    }
    /// The slots the innermost region wrote, with their values now.
    pub fn region_writes(&self, out: &mut Vec<(u32, Value)>) {
        let r = &self.regions[self.regions.len() - 1];
        for x in &self.log[r.log_mark as usize..] {
            out.push((x.slot, self.cur[x.slot as usize]));
        }
    }
    /// Leave the innermost region, restoring every slot it wrote (and the value numbers).
    pub fn undo_region(&mut self, e: &mut Emit) {
        let Some(r) = self.regions.pop() else { return };
        while self.log.len() > r.log_mark as usize {
            let Some(x) = self.log.pop() else { break };
            self.cur[x.slot as usize] = x.old;
            self.logged_in[x.slot as usize] = x.prev;
        }
        e.vn_release(r.vn_mark);
    }
    /// Leave the innermost region keeping its writes: it ran unconditionally after all.
    pub fn commit_region(&mut self) {
        let Some(r) = self.regions.pop() else { return };
        let parent = &self.regions[self.regions.len() - 1];
        let (puid, pbase) = (parent.uid, parent.slot_base);
        let mut keep = r.log_mark as usize;
        for i in r.log_mark as usize..self.log.len() {
            let (slot, prev) = (self.log[i].slot, self.log[i].prev);
            if slot < pbase && prev != puid {
                self.logged_in[slot as usize] = puid;
                self.log.swap(keep, i);
                keep += 1;
            } else {
                self.logged_in[slot as usize] = prev;
            }
        }
        self.log.truncate(keep);
    }

    /// A join whose edges carry the slots written from here on.
    pub fn new_join(&mut self, e: &Emit) -> JoinId {
        let (edges, snaps) = self.pool.pop().unwrap_or_default();
        self.insert_join(Join { edges, snaps, log_mark: self.log.len() as u32, slot_limit: self.cur.len() as u32, vn_mark: e.vn_mark(), block: None })
    }
    fn insert_join(&mut self, join: Join) -> JoinId {
        match self.free_joins.pop() {
            Some(i) => {
                self.joins[i as usize] = Some(join);
                JoinId(i)
            }
            None => {
                self.joins.push(Some(join));
                JoinId(self.joins.len() as u32 - 1)
            }
        }
    }
    pub fn has_block(&self, j: JoinId) -> bool {
        self.joins[j.0 as usize].as_ref().is_some_and(|j| j.block.is_some())
    }
    pub fn edge_count(&self, j: JoinId) -> usize {
        self.joins[j.0 as usize].as_ref().map_or(0, |j| j.edges.len())
    }
    /// The value `slot` has at the end of edge `ei` of the join.
    pub fn edge_value(&self, e: &Emit, j: JoinId, ei: usize, slot: u32) -> Value {
        if let Some(join) = &self.joins[j.0 as usize] {
            let x = &join.edges[ei];
            for &(s, v) in &join.snaps[x.start as usize..(x.start + x.len) as usize] {
                if s == slot {
                    return self.forwarded(e, v);
                }
            }
        }
        self.cur[slot as usize]
    }
    /// Move the edges of `from` to `to`, a join created before it, instead of resolving it:
    /// for when nothing happens between the two (the block `from` would become only passes
    /// its phis on). The state must be the one `from` was created in. False, with nothing
    /// done, when `from` already has a block.
    pub fn forward_join(&mut self, from: JoinId, to: JoinId) -> bool {
        if from == to || self.joins[to.0 as usize].is_none() || self.joins[from.0 as usize].as_ref().is_none_or(|j| j.block.is_some()) {
            return false;
        }
        let Some(mut src) = self.joins[from.0 as usize].take() else { return false };
        self.free_joins.push(from.0);
        let Some(mut dst) = self.joins[to.0 as usize].take() else { return false };
        for x in &src.edges {
            let id = self.next_stamp();
            let start = dst.snaps.len() as u32;
            for &(slot, v) in &src.snaps[x.start as usize..(x.start + x.len) as usize] {
                if slot < dst.slot_limit {
                    self.stamp[slot as usize] = id;
                    dst.snaps.push((slot, v));
                }
            }
            // what was written between the two joins
            for l in &self.log[dst.log_mark as usize..] {
                let s = l.slot as usize;
                if l.slot < dst.slot_limit && self.stamp[s] != id {
                    self.stamp[s] = id;
                    dst.snaps.push((l.slot, self.cur[s]));
                }
            }
            dst.edges.push(Edge { block: x.block, term: x.term, start, len: dst.snaps.len() as u32 - start });
        }
        self.joins[to.0 as usize] = Some(dst);
        src.edges.clear();
        src.snaps.clear();
        self.pool.push((src.edges, src.snaps));
        true
    }
    /// Take the edges whose `class` is 1 and those whose class is 2 out of the join, into
    /// two joins that stand for the same point as `j` (the state must be the one `j` was
    /// created in); the other edges stay.
    pub fn split_join(&mut self, j: JoinId, class: &[u8]) -> (JoinId, JoinId) {
        let (log_mark, slot_limit, vn_mark) = self.joins[j.0 as usize].as_ref().map_or((0, 0, 0), |j| (j.log_mark, j.slot_limit, j.vn_mark));
        let mut parts = [JoinId(0); 2];
        for p in &mut parts {
            let (edges, snaps) = self.pool.pop().unwrap_or_default();
            *p = self.insert_join(Join { edges, snaps, log_mark, slot_limit, vn_mark, block: None });
        }
        let Some(mut join) = self.joins[j.0 as usize].take() else { return (parts[0], parts[1]) };
        let edges = std::mem::take(&mut join.edges);
        let snaps = std::mem::take(&mut join.snaps);
        for (x, &c) in edges.into_iter().zip(class) {
            let dst = match c {
                1 | 2 => match &mut self.joins[parts[c as usize - 1].0 as usize] {
                    Some(d) => d,
                    None => continue,
                },
                _ => &mut join,
            };
            let start = dst.snaps.len() as u32;
            dst.snaps.extend_from_slice(&snaps[x.start as usize..(x.start + x.len) as usize]);
            dst.edges.push(Edge { block: x.block, term: x.term, start, len: x.len });
        }
        self.joins[j.0 as usize] = Some(join);
        (parts[0], parts[1])
    }
    /// Give the join a block now (for a terminator emitted before the join is resolved).
    pub fn set_block(&mut self, j: JoinId, b: Block) {
        if let Some(join) = &mut self.joins[j.0 as usize] {
            join.block = Some(b);
        }
    }
    pub fn has_edges(&self, j: JoinId) -> bool {
        self.joins[j.0 as usize].as_ref().is_some_and(|j| !j.edges.is_empty())
    }
    fn next_stamp(&mut self) -> u32 {
        if self.stamp.len() < self.cur.len() {
            self.stamp.resize(self.cur.len(), 0);
            self.pos.resize(self.cur.len(), 0);
        }
        self.stamp_id += 1;
        self.stamp_id
    }
    /// An edge from the end of `block` (which gets no more instructions) to the join, with
    /// the current values of the slots written since the join was created.
    pub fn add_edge(&mut self, j: JoinId, block: Block, term: Term) {
        let id = self.next_stamp();
        let Some(join) = &mut self.joins[j.0 as usize] else { return };
        let start = join.snaps.len() as u32;
        for x in &self.log[join.log_mark as usize..] {
            let s = x.slot as usize;
            if x.slot < join.slot_limit && self.stamp[s] != id {
                self.stamp[s] = id;
                join.snaps.push((x.slot, self.cur[s]));
            }
        }
        let len = join.snaps.len() as u32 - start;
        join.edges.push(Edge { block, term, start, len });
    }

    pub fn mark_trivial(&mut self, e: &Emit, phi: Value) {
        let k = e.body.insts[phi.0 as usize].a.0 as usize;
        if self.trivial.len() <= k / 64 {
            self.trivial.resize(k / 64 + 1, 0);
        }
        self.trivial[k / 64] |= 1 << (k % 64);
    }
    /// `v`, or what replaced it if it is a phi that was removed.
    pub fn forwarded(&self, e: &Emit, v: Value) -> Value {
        if self.trivial.is_empty() || !v.is_inst() {
            return v;
        }
        let x = &e.body.insts[v.0 as usize];
        if x.op != Op::Phi {
            return v;
        }
        let k = x.a.0 as usize;
        if self.trivial.get(k / 64).is_some_and(|w| w & (1 << (k % 64)) != 0) {
            e.body.resolve(v)
        } else {
            v
        }
    }

    /// Drop a join that got no edge.
    pub fn drop_join(&mut self, j: JoinId) {
        self.free_joins.push(j.0);
        if let Some(mut join) = self.joins[j.0 as usize].take() {
            join.edges.clear();
            join.snaps.clear();
            self.pool.push((join.edges, join.snaps));
        }
    }

    /// Turn a join into the insertion point: the state must be the one the join was created
    /// in (every region opened since has been left). Returns false when no edge reaches it (what follows is
    /// unreachable and the insertion point is unchanged).
    pub fn resolve(&mut self, e: &mut Emit, j: JoinId) -> bool {
        let Some(mut join) = self.joins[j.0 as usize].take() else { return false };
        // a join is the start of a basic block in DXC too, also when it is none here
        e.boundary();
        self.free_joins.push(j.0);
        e.vn_release(join.vn_mark);
        let reachable = !join.edges.is_empty();
        if join.edges.len() == 1 && join.block.is_none() && matches!(join.edges[0].term, Term::Br) {
            e.bb = join.edges[0].block;
            for i in 0..join.snaps.len() {
                let (slot, v) = join.snaps[i];
                let v = self.forwarded(e, v);
                self.write(slot, v);
            }
        } else if reachable {
            let jb = match join.block {
                Some(b) => b,
                None => e.body.new_block(),
            };
            for x in &join.edges {
                match x.term {
                    Term::Br => e.body.br(x.block, jb),
                    Term::CondFalse { cond, other } => e.body.cond_br(x.block, cond, other, jb),
                    Term::CondTrue { cond, other } => e.body.cond_br(x.block, cond, jb, other),
                    Term::SwitchDefault { v, table } => {
                        let t = std::mem::take(&mut self.switch_tables[table as usize]);
                        e.body.switch(x.block, v, jb, t);
                    }
                    Term::Done => {}
                }
            }
            e.bb = jb;
            let k = join.edges.len();
            let id = self.next_stamp();
            self.tmp_slots.clear();
            self.tmp_vals.clear();
            for (ei, x) in join.edges.iter().enumerate() {
                for &(slot, v) in &join.snaps[x.start as usize..(x.start + x.len) as usize] {
                    let s = slot as usize;
                    if self.stamp[s] != id {
                        self.stamp[s] = id;
                        self.pos[s] = self.tmp_slots.len() as u32;
                        self.tmp_slots.push(slot);
                        self.tmp_vals.resize(self.tmp_vals.len() + k, Value::NONE);
                    }
                    self.tmp_vals[self.pos[s] as usize * k + ei] = v;
                }
            }
            for p in 0..self.tmp_slots.len() {
                let slot = self.tmp_slots[p];
                let base = self.cur[slot as usize];
                let mut same = true;
                for ei in 0..k {
                    let v = self.tmp_vals[p * k + ei];
                    let v = if v.is_none() { base } else { self.forwarded(e, v) };
                    self.tmp_vals[p * k + ei] = v;
                    same &= v == self.tmp_vals[p * k];
                }
                let v = if same {
                    self.tmp_vals[p * k]
                } else {
                    let phi = e.body.phi(jb, self.slot_ty[slot as usize]);
                    for ei in 0..k {
                        e.body.phi_add(phi, self.tmp_vals[p * k + ei], join.edges[ei].block);
                    }
                    phi
                };
                self.write(slot, v);
            }
        }
        join.edges.clear();
        join.snaps.clear();
        self.pool.push((join.edges, join.snaps));
        reachable
    }

    /// Enter a loop: the slots that exist now are the ones its header may need phis for.
    pub fn loop_enter(&mut self) -> LoopSave {
        let save = LoopSave { uid: self.loop_uid, slot_base: self.loop_slot_base };
        self.loop_uid = self.next_uid;
        self.next_uid += 1;
        self.loop_slot_base = self.cur.len() as u32;
        save
    }
    /// Declare that the innermost loop's header has a phi for `slot`; returns what
    /// [`Ssa::loop_unmark`] takes when the loop is left.
    pub fn loop_mark(&mut self, slot: u32) -> u32 {
        std::mem::replace(&mut self.loop_mark[slot as usize], self.loop_uid)
    }
    pub fn loop_unmark(&mut self, slot: u32, prev: u32) {
        self.loop_mark[slot as usize] = prev;
    }
    pub fn loop_leave(&mut self, save: LoopSave) {
        self.loop_uid = save.uid;
        self.loop_slot_base = save.slot_base;
    }
}
