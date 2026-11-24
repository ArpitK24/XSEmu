//! Statements and control flow. Every loop is emitted in canonical form: the block before
//! it branches to a header that holds the loop's phis, one latch branches back (every
//! `continue` and the end of the body meet there first), and every `break` and the loop
//! condition leave to one exit block whose predecessors are all inside the loop.

use super::super::ast::{BinOp as B, Expr, ExprId, Stmt, StmtId, Sym, UnOp};
use super::super::types::HTy;
use super::ssa::Term;
use super::thread::{LoopFrame, Tail};
use super::{Lower, VarKind, NONE};
use crate::ir::Value;
use crate::Result;

impl<'a> Lower<'a> {
    pub(super) fn stmt(&mut self, s: StmtId) -> Result<()> {
        let rest = std::mem::take(&mut self.rest);
        let tail = self.tail.take();
        if self.dead {
            return Ok(());
        }
        let u = self.u;
        self.e.line = u.stmt_lines.get(s.0 as usize).copied().unwrap_or(self.e.line);
        // for `spec_store`: what the statement before this one stored
        let prev = self.last_store.take();
        match *u.stmt(s) {
            Stmt::Var(r) => {
                for d in r.of(&u.vars) {
                    self.declare(d, false)?;
                    self.last_store = if self.e.track && d.init.is_some() { Some(d.name) } else { None };
                }
            }
            Stmt::Expr(x) => {
                self.expr(x)?;
                if let (true, Expr::Assign(_, l, _)) = (self.e.track, *u.expr(x)) {
                    if let Expr::Ident(v) = *u.expr(l) {
                        self.last_store = Some(v);
                    }
                }
            }
            Stmt::Block(r) => {
                let scope = self.scope();
                // the block's last statement is the branch's last only if the block is
                self.stmts(u.block(r), if rest.is_empty() { tail } else { None })?;
                self.end_scope(scope);
            }
            Stmt::If { cond, then, els } => self.if_stmt(cond, then, els, prev, rest, tail)?,
            Stmt::For { init, cond, step, body } => {
                let scope = self.scope();
                if let Some(i) = init {
                    self.stmt(i)?;
                }
                self.loop_stmt(s, cond, step, body, false)?;
                self.end_scope(scope);
            }
            Stmt::While { cond, body } => self.loop_stmt(s, Some(cond), None, body, false)?,
            Stmt::DoWhile { body, cond } => self.loop_stmt(s, Some(cond), None, body, true)?,
            Stmt::Switch { value, cases } => self.switch_stmt(value, cases)?,
            Stmt::Break => {
                let Some(&j) = self.breaks.last() else { return self.err("break outside a loop or switch") };
                if !self.dx_break(j) {
                    self.ssa.add_edge(j, self.e.bb, Term::Br);
                }
                self.dead = true;
            }
            Stmt::Continue => {
                let Some(&j) = self.continues.last() else { return self.err("continue outside a loop") };
                self.ssa.add_edge(j, self.e.bb, Term::Br);
                self.dead = true;
            }
            Stmt::Discard => {
                if self.io.is_none() {
                    return self.err("discard is only valid in a pixel shader");
                }
                super::super::stage::discard(&mut self.e);
            }
            Stmt::Return(x) => {
                let Some(ctx) = self.rets.last() else { return self.err("return outside a function") };
                let (join, slot, ty, depth) = (ctx.join, ctx.slot, ctx.ty, ctx.depth);
                match (x, ty) {
                    (None, HTy::Void) => {}
                    (Some(x), t) if t != HTy::Void => {
                        let v = self.expr(x)?;
                        let v = self.coerce(&v, t)?;
                        for (i, &c) in v.comps().iter().enumerate() {
                            self.ssa.write_leaving(slot + i as u32, c);
                        }
                    }
                    _ => return self.err("return value"),
                }
                if self.ssa.region_depth() == depth && !self.ssa.has_edges(join) {
                    // the last statement of a straight-line body: no join needed
                    if let Some(ctx) = self.rets.last_mut() {
                        ctx.tail = true;
                    }
                } else {
                    self.ssa.add_edge(join, self.e.bb, Term::Br);
                }
                self.dead = true;
            }
            Stmt::Empty => {}
        }
        Ok(())
    }

    /// The statements of a block, in order. `tail`: the join the last one ends at, if the
    /// block is a branch of an `if` (see [`Lower::join_then`]).
    pub(super) fn stmts(&mut self, list: &'a [StmtId], tail: Option<Tail>) -> Result<()> {
        let mut i = 0;
        while i < list.len() {
            self.rest = &list[i + 1..];
            self.tail = tail;
            self.stmt(list[i])?;
            // an `if` may have lowered the statements that test what it set
            i += 1 + std::mem::take(&mut self.took);
        }
        Ok(())
    }

    /// A statement as the body of a branch or loop: its own scope.
    pub(super) fn scoped(&mut self, s: StmtId) -> Result<()> {
        let scope = self.scope();
        self.stmt(s)?;
        self.end_scope(scope);
        Ok(())
    }

    /// `break` / `continue`, alone or in braces: the join it goes to.
    fn jump_target(&self, s: StmtId) -> Option<super::ssa::JoinId> {
        match *self.u.stmt(s) {
            Stmt::Break => self.breaks.last().copied(),
            Stmt::Continue => self.continues.last().copied(),
            Stmt::Block(r) if r.len == 1 => self.jump_target(self.u.block(r)[0]),
            _ => None,
        }
    }

    fn if_stmt(&mut self, cond: ExprId, then: StmtId, els: Option<StmtId>, prev: Option<Sym>, rest: &'a [StmtId], tail: Option<Tail>) -> Result<()> {
        let c = self.cond(cond)?;
        let mark = self.e.lane_ops;
        if let Some(k) = self.e.m.as_int(c) {
            // decided now: the other branch is not code
            if rest.is_empty() && (k != 0 || els.is_some()) {
                self.tail = tail;
            }
            return match (k != 0, els) {
                (true, _) => self.scoped(then),
                (false, Some(e)) => self.scoped(e),
                (false, None) => Ok(()),
            };
        }
        let head = self.e.bb;
        if let (Some(target), None) = (self.jump_target(then), els) {
            // if (c) break; branches straight to the loop's exit
            let next = self.e.body.new_block();
            self.ssa.add_edge(target, head, Term::CondTrue { cond: c, other: next });
            self.e.bb = next;
            self.e.boundary();
            return Ok(());
        }
        if self.flat(then) && els.is_none_or(|e| self.flat(e)) {
            let spec_store = self.e.track && els.is_none() && self.spec_store(prev, cond, then);
            return self.if_select(c, then, els, spec_store);
        }
        let join = self.ssa.new_join(&self.e);
        let tb = self.e.body.new_block();
        self.ssa.open_region(&self.e);
        self.e.bb = tb;
        self.e.boundary();
        self.tail = Some(Tail::Join(join));
        self.scoped(then)?;
        if !self.dead {
            self.ssa.add_edge(join, self.e.bb, Term::Br);
        }
        self.ssa.undo_region(&mut self.e);
        self.dead = false;
        match els {
            Some(els) => {
                let eb = self.e.body.new_block();
                self.e.body.cond_br(head, c, tb, eb);
                self.ssa.open_region(&self.e);
                self.e.bb = eb;
                self.e.boundary();
                self.tail = Some(Tail::Join(join));
                self.scoped(els)?;
                if !self.dead {
                    self.ssa.add_edge(join, self.e.bb, Term::Br);
                }
                self.ssa.undo_region(&mut self.e);
            }
            None => self.ssa.add_edge(join, head, Term::CondFalse { cond: c, other: tb }),
        }
        // set last: the blocks lowered on the way read it for their own statements
        self.took = self.join_then(join, rest, tail, mark)?;
        Ok(())
    }

    /// Whether a branch body is so small that DXC's output has selects for it (SimplifyCFG
    /// folds a two-way branch whose sides are a couple of cheap instructions): up to two
    /// assignments to SSA variables of expressions without calls, divisions, memory or
    /// anything that could not run unconditionally.
    fn flat(&self, s: StmtId) -> bool {
        let mut budget = 4;
        match *self.u.stmt(s) {
            Stmt::Block(r) => r.len <= 2 && self.u.block(r).iter().all(|&t| self.flat_assign(t, &mut budget)),
            _ => self.flat_assign(s, &mut budget),
        }
    }
    fn flat_assign(&self, s: StmtId, budget: &mut i32) -> bool {
        let Stmt::Expr(x) = *self.u.stmt(s) else { return false };
        let Expr::Assign(op, l, r) = *self.u.expr(x) else { return false };
        if matches!(op, Some(B::Div | B::Rem)) {
            return false;
        }
        *budget -= op.is_some() as i32;
        let target = match *self.u.expr(l) {
            Expr::Ident(_) => l,
            Expr::Member(a, _) => a,
            _ => return false,
        };
        self.cheap(target, budget) && self.cheap(r, budget) && *budget >= 0
    }
    fn cheap(&self, x: ExprId, budget: &mut i32) -> bool {
        match *self.u.expr(x) {
            Expr::Int(..) | Expr::Float(..) | Expr::Bool(_) => true,
            Expr::Ident(s) => self.lookup(s).is_some_and(|i| matches!((self.vars[i as usize].kind, self.vars[i as usize].ty), (VarKind::Ssa(_), HTy::Num(..)))),
            Expr::Unary(op, a) => {
                *budget -= 1;
                matches!(op, UnOp::Neg | UnOp::Plus | UnOp::Not | UnOp::BitNot) && self.cheap(a, budget)
            }
            Expr::Binary(op, a, b) => {
                *budget -= 1;
                !matches!(op, B::Div | B::Rem | B::LogAnd | B::LogOr | B::Comma) && self.cheap(a, budget) && self.cheap(b, budget)
            }
            Expr::Cast(_, a) => {
                *budget -= 1;
                self.cheap(a, budget)
            }
            Expr::Member(a, _) => self.cheap(a, budget),
            // the intrinsic is one select per component
            Expr::Call(name, args) => {
                *budget -= 1;
                self.u.name(name) == "select" && args.len == 3 && self.an.overloads(name).is_empty() && self.u.args(args).iter().all(|&a| self.cheap(a, budget))
            }
            _ => false,
        }
    }

    /// An `if` whose bodies are [`Lower::flat`]: both run, and every variable they assign
    /// becomes a select of its two values.
    fn if_select(&mut self, c: Value, then: StmtId, els: Option<StmtId>, spec_store: bool) -> Result<()> {
        let mut writes: Vec<(u32, Value)> = Vec::new();
        // DXC has blocks here, except for the one store its SimplifyCFG runs unconditionally
        let blocks = self.e.track && !spec_store;
        self.ssa.open_region(&self.e);
        if blocks {
            self.e.boundary();
        }
        self.scoped(then)?;
        self.ssa.region_writes(&mut writes);
        self.ssa.undo_region(&mut self.e);
        let n = writes.len();
        if let Some(els) = els {
            self.ssa.open_region(&self.e);
            if blocks {
                self.e.boundary();
            }
            self.scoped(els)?;
            self.ssa.region_writes(&mut writes);
            self.ssa.undo_region(&mut self.e);
        }
        if blocks {
            self.e.boundary();
        }
        for i in 0..writes.len() {
            let (slot, v) = writes[i];
            let old = self.ssa.read(slot);
            let new = if i < n {
                // the other side's value, if it assigns the slot too
                let f = writes[n..].iter().find(|w| w.0 == slot).map_or(old, |w| w.1);
                self.e.select_merge(c, v, f)
            } else if writes[..n].iter().any(|w| w.0 == slot) {
                continue;
            } else {
                self.e.select_merge(c, old, v)
            };
            self.ssa.write(slot, new);
        }
        Ok(())
    }

    /// `v = a; if (c) v = literal;`: DXC's SimplifyCFG, which has run once when DXC decides
    /// its conditional breaks, turns the second store into a select and no block is left
    /// (isSafeToSpeculateStore: the store before the branch is to the same variable and at
    /// most nine instructions back, with no call in between). `prev` is the variable the
    /// statement before the `if` assigned.
    fn spec_store(&self, prev: Option<Sym>, cond: ExprId, then: StmtId) -> bool {
        let then = match *self.u.stmt(then) {
            Stmt::Block(r) if r.len == 1 => self.u.block(r)[0],
            _ => then,
        };
        let Stmt::Expr(x) = *self.u.stmt(then) else { return false };
        let Expr::Assign(None, l, r) = *self.u.expr(x) else { return false };
        let Expr::Ident(v) = *self.u.expr(l) else { return false };
        let literal = |x: ExprId| matches!(*self.u.expr(x), Expr::Int(..) | Expr::Float(..) | Expr::Bool(_));
        let constant = match *self.u.expr(r) {
            Expr::Unary(UnOp::Neg | UnOp::Plus, a) | Expr::Cast(_, a) => literal(a),
            _ => literal(r),
        };
        prev == Some(v) && constant && self.spec_cost(cond).is_some_and(|n| n <= 5)
    }

    /// `while`, `for` (its init statement already lowered) and, with `post`, `do … while`.
    fn loop_stmt(&mut self, s: StmtId, cond: Option<ExprId>, step: Option<ExprId>, body: StmtId, post: bool) -> Result<()> {
        // the slots the loop assigns, each once
        let mut slots = std::mem::take(&mut self.tmp_slots);
        let first = slots.len();
        self.stamp_id += 1;
        for &sym in self.an.loop_mods(s) {
            if std::mem::replace(&mut self.sym_stamp[sym.0 as usize], self.stamp_id) == self.stamp_id {
                continue;
            }
            let mut i = self.sym_top[sym.0 as usize];
            while i != NONE {
                let v = &self.vars[i as usize];
                if let (true, VarKind::Ssa(base)) = (v.frame == self.frame || v.frame == 0, v.kind) {
                    slots.extend(base..base + self.e.types.comps(v.ty) as u32);
                }
                // a local may hide the global of the same name that a callee assigns
                i = if v.frame == 0 { NONE } else { v.prev };
            }
        }
        let exit = self.ssa.new_join(&self.e);
        let pre = self.e.bb;
        let header = self.e.body.new_block();
        self.e.body.br(pre, header);
        self.e.bb = header;
        self.dx_loop_enter(self.an.loop_wave(s), exit);
        self.ssa.open_region(&self.e);
        let save = self.ssa.loop_enter();
        // (slot, phi, previous loop mark)
        let mut phis: Vec<(u32, Value, u32)> = Vec::with_capacity(slots.len() - first);
        for &slot in &slots[first..] {
            let phi = self.e.body.phi(header, self.ssa.slot_ty[slot as usize]);
            self.e.body.phi_add(phi, self.ssa.read(slot), pre);
            let prev = self.ssa.loop_mark(slot);
            self.ssa.write(slot, phi);
            phis.push((slot, phi, prev));
        }
        slots.truncate(first);
        self.tmp_slots = slots;
        self.breaks.push(exit);
        self.loops.push(LoopFrame { exit, cond, step, stmt: s });
        if let (Some(c), false) = (cond, post) {
            let v = self.cond(c)?;
            match self.e.m.as_int(v) {
                Some(0) => {
                    self.ssa.add_edge(exit, self.e.bb, Term::Br);
                    self.dead = true;
                }
                Some(_) => {}
                None => {
                    let b = self.e.body.new_block();
                    self.ssa.add_edge(exit, self.e.bb, Term::CondFalse { cond: v, other: b });
                    self.e.bb = b;
                    self.e.boundary();
                }
            }
        }
        let latch = self.ssa.new_join(&self.e);
        self.ssa.open_region(&self.e);
        self.continues.push(latch);
        self.tail = Some(Tail::LoopEnd);
        self.scoped(body)?;
        self.continues.pop();
        if self.ssa.has_edges(latch) {
            if !self.dead {
                self.ssa.add_edge(latch, self.e.bb, Term::Br);
            }
            self.ssa.undo_region(&mut self.e);
            self.dead = !self.ssa.resolve(&mut self.e, latch);
        } else {
            self.ssa.commit_region();
            self.ssa.drop_join(latch);
        }
        self.breaks.pop();
        self.loops.pop();
        // DXC's for.inc / do.cond
        self.e.boundary();
        if let (Some(x), false) = (step, self.dead) {
            self.expr(x)?;
        }
        // the back edge
        let mut back = None;
        if !self.dead {
            let v = match (cond, post) {
                (Some(c), true) => Some(self.cond(c)?),
                _ => None,
            };
            let end = self.e.bb;
            match v.map(|v| (v, self.e.m.as_int(v))) {
                Some((_, Some(0))) => self.ssa.add_edge(exit, end, Term::Br),
                Some((v, None)) => {
                    self.ssa.add_edge(exit, end, Term::CondFalse { cond: v, other: header });
                    back = Some(end);
                }
                _ => {
                    self.e.body.br(end, header);
                    back = Some(end);
                }
            }
        }
        if let Some(end) = back {
            self.loop_blocks.extend([pre.0, end.0]);
        }
        for &(slot, phi, prev) in &phis {
            let entry = self.e.body.phi_incoming(phi)[0].0;
            let mut trivial = true;
            if let Some(end) = back {
                let v = self.ssa.forwarded(&self.e, self.ssa.read(slot));
                self.e.body.phi_add(phi, v, end);
                trivial = v == phi || v == entry;
            }
            if trivial {
                self.e.body.replace(phi, entry);
                self.ssa.mark_trivial(&self.e, phi);
            }
            self.ssa.loop_unmark(slot, prev);
        }
        self.ssa.loop_leave(save);
        self.ssa.undo_region(&mut self.e);
        self.dx_loop_leave();
        self.dead = !self.ssa.resolve(&mut self.e, exit);
        if let (Some(_), false) = (back, self.dead) {
            self.loop_blocks.push(self.e.bb.0);
            self.loop_exits.push(self.e.bb.0);
        }
        Ok(())
    }

    /// A `switch` whose value is the constant `k`: the statements from its label (or
    /// `default`) on, through the labels it falls into, up to the `break`. DXC folds
    /// these the same way once it has propagated the constant.
    fn switch_const(&mut self, k: u64, cases: &[super::super::ast::Case]) -> Result<()> {
        let u = self.u;
        let (mut start, mut default) = (None, None);
        for (i, c) in cases.iter().enumerate() {
            match c.value {
                Some(x) if start.is_none() && self.const_int(x)? & 0xFFFF_FFFF == k => start = Some(i),
                Some(_) => {}
                None => default = Some(i),
            }
        }
        let Some(start) = start.or(default) else { return Ok(()) };
        let exit = self.ssa.new_join(&self.e);
        self.breaks.push(exit);
        self.ssa.open_region(&self.e);
        let depth = self.ssa.region_depth();
        'cases: for c in &cases[start..] {
            let scope = self.scope();
            for &t in u.block(c.body) {
                // the break that ends a straight-line case needs no jump
                if matches!(*u.stmt(t), Stmt::Break) && !self.dead && self.ssa.region_depth() == depth {
                    self.end_scope(scope);
                    break 'cases;
                }
                self.stmt(t)?;
            }
            self.end_scope(scope);
            if self.dead {
                break;
            }
        }
        self.breaks.pop();
        if !self.ssa.has_edges(exit) && !self.dead {
            self.ssa.commit_region();
            self.ssa.drop_join(exit);
            return Ok(());
        }
        if !self.dead {
            self.ssa.add_edge(exit, self.e.bb, Term::Br);
        }
        self.ssa.undo_region(&mut self.e);
        self.dead = !self.ssa.resolve(&mut self.e, exit);
        Ok(())
    }

    fn switch_stmt(&mut self, value: ExprId, cases: super::super::ast::Range) -> Result<()> {
        let u = self.u;
        let v = self.expr(value)?;
        let v = match v.ty {
            HTy::Num(k, 1, 1) if !k.is_float() => self.e.convert_scalar(v.one(), k, super::super::types::Sk::Uint),
            _ => return self.err("a scalar integer switch value is expected"),
        };
        let cases = cases.of(&u.cases);
        if let Some(k) = self.e.m.as_int(v) {
            return self.switch_const(k & 0xFFFF_FFFF, cases);
        }
        let head = self.e.bb;
        let exit = self.ssa.new_join(&self.e);
        self.breaks.push(exit);
        let mut table: Vec<(u64, crate::ir::Block)> = Vec::with_capacity(cases.len());
        let mut labels = std::collections::HashSet::with_capacity(cases.len());
        let mut default = None;
        let mut entry = self.ssa.new_join(&self.e);
        let mut i = 0;
        while i < cases.len() {
            // the labels of one body: empty bodies fall into the next label
            let block = self.e.body.new_block();
            self.ssa.set_block(entry, block);
            let mut last = i;
            loop {
                match cases[last].value {
                    Some(x) => {
                        let k = self.const_int(x)? & 0xFFFF_FFFF;
                        if !labels.insert(k) {
                            return self.err("duplicate case");
                        }
                        table.push((k, block));
                    }
                    None => default = Some(block),
                }
                // one phi entry per switch edge
                self.ssa.add_edge(entry, head, Term::Done);
                if cases[last].body.len != 0 || last + 1 == cases.len() {
                    break;
                }
                last += 1;
            }
            // what falls into this body from the one before is merged inside its region, so
            // the next label starts from the state at the switch again
            let more = last + 1 < cases.len();
            let next = if more { self.ssa.new_join(&self.e) } else { exit };
            self.ssa.open_region(&self.e);
            self.dead = !self.ssa.resolve(&mut self.e, entry);
            let scope = self.scope();
            self.stmts(u.block(cases[last].body), None)?;
            self.end_scope(scope);
            if !self.dead {
                self.ssa.add_edge(next, self.e.bb, Term::Br);
            }
            self.ssa.undo_region(&mut self.e);
            self.dead = false;
            entry = next;
            i = last + 1;
        }
        if cases.is_empty() {
            self.ssa.drop_join(entry);
        }
        self.breaks.pop();
        match default {
            Some(d) => self.e.body.switch(head, v, d, table),
            None => {
                self.ssa.switch_tables.push(table);
                self.ssa.add_edge(exit, head, Term::SwitchDefault { v, table: self.ssa.switch_tables.len() as u32 - 1 });
            }
        }
        self.dead = !self.ssa.resolve(&mut self.e, exit);
        Ok(())
    }
}
