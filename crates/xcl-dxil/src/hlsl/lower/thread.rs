//! Jump threading at the end of an `if`, where a lane-sensitive operation can tell.
//!
//! DXC runs LLVM's jump threading once, right after its mem2reg and its dx.break clean-up
//! (PassManagerBuilder: SROA, SimplifyInst, JumpThreading, under the lifetime markers that
//! are on by default). A join whose phi feeds the next branch then lets the edges on which
//! the phi is a constant go around the test:
//!
//! ```text
//! if (c) { n = WaveActiveCountBits(true); skip = true; }
//! if (skip) break;
//! ```
//!
//! The block that set `skip` branches straight out of the loop, so for a driver it is no
//! longer inside the loop and its wave operation counts every lane that has left so far.
//! That is the only thing the transformation changes that a program can observe, and it
//! costs phis (each side of the split join merges the variables again), so it is done here
//! exactly where it can be observed:
//!
//! * the code since the first `if` began holds a wave, quad or derivative operation
//!   ([`Emit::lane_ops`]), and we are in a loop;
//! * following the statements after the join with the values one edge has
//!   ([`Lower::fate`]) leads out of the loop: through tests the edge decides, to a `break`
//!   or `return`.
//!
//! Then that edge is taken out of the join ([`Ssa::split_join`]) and goes to the side of the
//! next `if` it chooses ([`Lower::threaded_if`]), and so on from that statement's join. An
//! `if` that is the last statement of a branch of another passes its edges on to that one's
//! join first ([`Ssa::forward_join`]; the same pass removes a block that only has phis and a
//! branch), so a flag set two levels down is followed too.
//!
//! DXC threads through a block by copying up to six instructions of it, which here would
//! mean lowering the statements between the join and the test once per side. That is not
//! done, and a program whose way out needs it is refused ([`Error::Unsupported`]: the
//! caller falls back to DXC), because a program lowered differently from DXC must not
//! come out as if it were not: `if (c) { …; skip = true; } m += i; if (skip) break;`,
//! unless the statements in between are clearly more than DXC copies ([`DUP_MAX`]
//! operations in the source: an estimate, and the reason for the margin).
//!
//! KNOWN DIFFERENCE, refused only with `XCL_DXIL_LANE_CHECK=1` (or `2`, which adds why to
//! the message) and only where lanes may leave the loop at different times
//! ([`super::lanes`]): DXC also threads the way out through the loop's own condition (it
//! has rotated the loop, the test is at its end), and through a branch that cannot be
//! followed here because it holds a loop:
//!
//! ```text
//! while (!done) { if (c) { n = WaveActiveCountBits(true); done = true; } }
//! while (pc >= 16u && pc < 64u) { if (pc == 16u) { …; pc = 100u; } if (pc == 20u) { … } }
//! ```
//!
//! The second is the dispatcher of translated programs, whose control flow is the same for
//! every lane, so that nothing can tell; refusing the shape would refuse those, and
//! threading it costs them thousands of phis. Telling the two apart needs to know which
//! conditions can differ between lanes ([`Fate::Bottom`] and [`Fate::Unknown`] are where
//! that knowledge goes).

use super::super::ast::{BinOp as B, Expr, ExprId, Stmt, StmtId, Sym, UnOp};
use super::super::types::HTy;
use super::analysis::root;
use super::ssa::{JoinId, Term};
use super::{Lower, VarKind};
use crate::Result;

/// What the last statement of a list ends at.
#[derive(Copy, Clone)]
pub(super) enum Tail {
    /// the join of the `if` the list is a branch of
    Join(JoinId),
    /// the end of the body of the innermost loop
    LoopEnd,
}

/// A loop being lowered.
pub(super) struct LoopFrame {
    pub exit: JoinId,
    pub cond: Option<ExprId>,
    pub step: Option<ExprId>,
    pub stmt: StmtId,
}

/// Where an edge of a join gets to when what follows the join is followed with its values.
enum Fate {
    /// not out of the loop by a way DXC threads
    Stays,
    /// out of the loop; the statements on the way that are in no branch of their own
    /// take this many operations (0: every step is a test the edge decides)
    Leaves(u32),
    /// out of the loop through the loop's own condition
    Bottom,
    /// what follows cannot be followed
    Unknown,
}

/// In the list of names an edge's values are not known for: every global.
const GLOBALS: Sym = Sym(u32::MAX);
/// Operations in the statements on a way out above which DXC is taken not to copy them (it
/// copies six instructions; the count here is of the source, hence the margin).
const DUP_MAX: u32 = 12;
const REFUSED: &str = "a wave, quad or derivative operation in a branch whose way out of the loop DXC makes by copying code";
const LANES_APART: &str = "a loop around a wave operation whose exit can differ between lanes";

impl<'a> Lower<'a> {
    /// The end of an `if`: its join becomes the insertion point, unless edges of it leave the
    /// loop from here (see the module). `rest`: the statements after the `if` in its block;
    /// `tail`: what the block ends at; `mark`: [`Emit::lane_ops`] when the first `if` of the
    /// chain began. Returns how many of the statements in `rest` were lowered here.
    pub(super) fn join_then(&mut self, join: JoinId, rest: &'a [StmtId], tail: Option<Tail>, mark: u32) -> Result<usize> {
        let u = self.u;
        let k = self.ssa.edge_count(join);
        if k >= 2 && self.e.lane_ops != mark && !self.loops.is_empty() && !self.ssa.has_block(join) {
            let mut class = std::mem::take(&mut self.tmp_class);
            class.clear();
            let mut unknown = Vec::new();
            let mut refused = false;
            let mut apart = false;
            for ei in 0..k {
                unknown.clear();
                class.push(match self.fate(join, ei, rest, tail, &mut unknown) {
                    Fate::Stays => 0,
                    Fate::Leaves(0) => 1,
                    Fate::Leaves(n) => {
                        refused |= n <= DUP_MAX;
                        0
                    }
                    // refused only behind the switch, where lanes may leave apart: see the module
                    Fate::Bottom | Fate::Unknown => {
                        apart = true;
                        0
                    }
                });
            }
            let next = rest.first().map(|&s| (s, *u.stmt(s)));
            if refused {
                self.tmp_class = class;
                return self.e.unsupported(REFUSED);
            }
            if let (true, Some(f)) = (apart, self.loops.last().filter(|f| self.an.loop_varies(f.stmt))) {
                self.tmp_class = class;
                let why = self.an.lanes.why(f.stmt.0, u);
                return self.e.unsupported(if why.is_empty() { LANES_APART.to_string() } else { format!("{LANES_APART}: {why}") });
            }
            if let (true, Some((s, Stmt::If { cond, then, els }))) = (class.contains(&1), next) {
                // which side each of them takes
                for ei in 0..k {
                    if class[ei] == 1 && self.edge_int(cond, join, ei, &[]) == Some(0) {
                        class[ei] = 2;
                    }
                }
                let (jt, jf) = self.ssa.split_join(join, &class);
                self.tmp_class = class;
                self.e.line = u.stmt_lines.get(s.0 as usize).copied().unwrap_or(self.e.line);
                return Ok(1 + self.threaded_if(join, jt, jf, cond, then, els, &rest[1..], tail, mark)?);
            }
            self.tmp_class = class;
        }
        if let (true, Some(Tail::Join(outer))) = (rest.is_empty(), tail) {
            if self.ssa.forward_join(join, outer) {
                self.dead = true;
                return Ok(0);
            }
        }
        self.dead = !self.ssa.resolve(&mut self.e, join);
        Ok(0)
    }

    /// `if (cond) then else els` after a join that [`Lower::join_then`] split: `jt` has the
    /// edges on which the condition is true, `jf` the ones on which it is false, and
    /// `join` is left with the edges that have to test it.
    #[allow(clippy::too_many_arguments)]
    fn threaded_if(&mut self, join: JoinId, jt: JoinId, jf: JoinId, cond: ExprId, then: StmtId, els: Option<StmtId>, rest: &'a [StmtId], tail: Option<Tail>, mark: u32) -> Result<usize> {
        // where the statement ends: without an else, where the false edges go
        let end = match els {
            Some(_) => self.ssa.new_join(&self.e),
            None => jf,
        };
        if self.ssa.has_edges(join) {
            self.ssa.open_region(&self.e);
            self.ssa.resolve(&mut self.e, join);
            self.dead = false;
            let c = self.cond(cond)?;
            let at = self.e.bb;
            match self.e.m.as_int(c) {
                Some(0) => self.ssa.add_edge(jf, at, Term::Br),
                Some(_) => self.ssa.add_edge(jt, at, Term::Br),
                None => {
                    // (neither join gets a block of its own here, so that both can be
                    // split or passed on in turn; the empty block is removed at the end)
                    let fb = self.e.body.new_block();
                    self.ssa.add_edge(jt, at, Term::CondTrue { cond: c, other: fb });
                    self.ssa.add_edge(jf, fb, Term::Br);
                }
            }
            self.ssa.undo_region(&mut self.e);
        } else {
            self.ssa.drop_join(join);
        }
        self.ssa.open_region(&self.e);
        if self.ssa.resolve(&mut self.e, jt) {
            // DXC decides its conditional breaks before it threads: no phi is here then
            self.e.boundary();
            self.dead = false;
            self.tail = Some(Tail::Join(end));
            self.scoped(then)?;
            if !self.dead {
                self.ssa.add_edge(end, self.e.bb, Term::Br);
            }
        }
        self.ssa.undo_region(&mut self.e);
        if let Some(els) = els {
            self.ssa.open_region(&self.e);
            if self.ssa.resolve(&mut self.e, jf) {
                self.e.boundary();
                self.dead = false;
                self.tail = Some(Tail::Join(end));
                self.scoped(els)?;
                if !self.dead {
                    self.ssa.add_edge(end, self.e.bb, Term::Br);
                }
            }
            self.ssa.undo_region(&mut self.e);
        }
        self.dead = false;
        self.join_then(end, rest, tail, mark)
    }

    /// Follow the statements `rest` (and then `tail`) with the values edge `ei` of `j` has.
    /// `unknown` collects the variables assigned on the way.
    fn fate(&self, j: JoinId, ei: usize, rest: &[StmtId], tail: Option<Tail>, unknown: &mut Vec<Sym>) -> Fate {
        let u = self.u;
        let mut cost = 0;
        // an edge that decides nothing is merged with the others at the first test
        let mut decided = false;
        // a test that is open only because of what was assigned on the way: DXC may know
        let open = |l: &Self, cond: ExprId, unknown: &[Sym]| if !unknown.is_empty() && l.edge_int(cond, j, ei, &[]).is_some() { Fate::Unknown } else { Fate::Stays };
        for &s in rest {
            match *u.stmt(s) {
                Stmt::If { cond, then, els } => {
                    let side = match self.edge_int(cond, j, ei, unknown) {
                        Some(0) => els,
                        Some(_) => Some(then),
                        None => return open(self, cond, unknown),
                    };
                    decided = true;
                    let Some(side) = side else { continue };
                    if self.leaves_loop(side) {
                        return Fate::Leaves(cost);
                    }
                    // the branch runs, and what follows is tested at its own join
                    if !self.assigned(side, unknown, &mut 0) {
                        return Fate::Unknown;
                    }
                }
                Stmt::Expr(_) | Stmt::Var(_) | Stmt::Empty => {
                    let mut n = 0;
                    self.assigned(s, unknown, &mut n);
                    cost += n.max(1);
                }
                _ => return if decided { Fate::Unknown } else { Fate::Stays },
            }
        }
        match (tail, self.loops.last()) {
            (Some(Tail::LoopEnd), Some(&LoopFrame { cond: Some(cond), step, .. })) => {
                if let Some(step) = step {
                    self.assigned_expr(step, unknown, &mut 0);
                }
                match self.edge_int(cond, j, ei, unknown) {
                    Some(0) => Fate::Bottom,
                    Some(_) => Fate::Stays,
                    None => open(self, cond, unknown),
                }
            }
            // (at a join the edge is followed again from there, if it is passed on)
            _ => Fate::Stays,
        }
    }

    /// Whether a branch ends by leaving the innermost loop.
    fn leaves_loop(&self, s: StmtId) -> bool {
        match *self.u.stmt(s) {
            Stmt::Break => self.loops.last().is_some_and(|f| self.breaks.last() == Some(&f.exit)),
            Stmt::Return(_) => true,
            Stmt::Block(r) => self.u.block(r).last().is_some_and(|&t| self.leaves_loop(t)),
            _ => false,
        }
    }

    /// The variables a statement may assign (into `out`) and the operations it takes.
    /// False if it holds a loop, a `switch` or a jump.
    fn assigned(&self, s: StmtId, out: &mut Vec<Sym>, cost: &mut u32) -> bool {
        let u = self.u;
        match *u.stmt(s) {
            Stmt::Empty => true,
            Stmt::Expr(x) => {
                self.assigned_expr(x, out, cost);
                true
            }
            Stmt::Var(r) => {
                for d in r.of(&u.vars) {
                    out.push(d.name);
                    if let Some(x) = d.init {
                        self.assigned_expr(x, out, cost);
                    }
                }
                true
            }
            Stmt::Block(r) => u.block(r).iter().all(|&t| self.assigned(t, out, cost)),
            Stmt::If { cond, then, els } => {
                self.assigned_expr(cond, out, cost);
                self.assigned(then, out, cost) && els.is_none_or(|e| self.assigned(e, out, cost))
            }
            _ => false,
        }
    }
    fn assigned_expr(&self, x: ExprId, out: &mut Vec<Sym>, cost: &mut u32) {
        let u = self.u;
        let target = |l: ExprId, out: &mut Vec<Sym>| out.extend(root(u, l));
        match *u.expr(x) {
            Expr::Int(..) | Expr::Float(..) | Expr::Bool(_) | Expr::Ident(_) => {}
            Expr::Unary(op, a) => {
                if matches!(op, UnOp::PreInc | UnOp::PreDec | UnOp::PostInc | UnOp::PostDec) {
                    target(a, out);
                }
                *cost += (op != UnOp::Plus) as u32;
                self.assigned_expr(a, out, cost);
            }
            Expr::Binary(op, a, b) => {
                *cost += (op != B::Comma) as u32;
                self.assigned_expr(a, out, cost);
                self.assigned_expr(b, out, cost);
            }
            Expr::Assign(op, l, r) => {
                target(l, out);
                *cost += op.is_some() as u32;
                self.assigned_expr(l, out, cost);
                self.assigned_expr(r, out, cost);
            }
            Expr::Ternary(c, a, b) => {
                *cost += 1;
                self.assigned_expr(c, out, cost);
                self.assigned_expr(a, out, cost);
                self.assigned_expr(b, out, cost);
            }
            Expr::Call(name, args) => {
                *cost += 1;
                // a function of the program may assign any global
                if self.by_name[name.0 as usize].1 != 0 {
                    out.push(GLOBALS);
                }
                for &a in u.args(args) {
                    // (it may be an out parameter)
                    target(a, out);
                    self.assigned_expr(a, out, cost);
                }
            }
            Expr::Method(_, all) => {
                let (obj, args) = u.method(all);
                *cost += 1;
                self.assigned_expr(obj, out, cost);
                for &a in u.args(args) {
                    target(a, out);
                    self.assigned_expr(a, out, cost);
                }
            }
            Expr::Construct(_, args) | Expr::InitList(args) => {
                for &a in u.args(args) {
                    self.assigned_expr(a, out, cost);
                }
            }
            Expr::Cast(_, a) | Expr::Member(a, _) => self.assigned_expr(a, out, cost),
            Expr::Index(a, i) => {
                self.assigned_expr(a, out, cost);
                self.assigned_expr(i, out, cost);
            }
        }
    }

    /// The value of an integer or boolean expression at the end of edge `ei` of the join
    /// `j`, when it follows from the constants variables are there: what DXC's jump
    /// threading finds for a phi of constants and the comparisons and operators on it. Only
    /// values from 0 to the largest int are followed, so that no width or sign matters.
    /// `unknown`: variables whose value at the edge is not their value any more.
    fn edge_int(&self, x: ExprId, j: JoinId, ei: usize, unknown: &[Sym]) -> Option<u64> {
        let v = match *self.u.expr(x) {
            Expr::Bool(b) => b as u64,
            Expr::Int(k, _) => k,
            Expr::Ident(s) => {
                let v = &self.vars[self.lookup(s)? as usize];
                let (VarKind::Ssa(slot), HTy::Num(k, 1, 1)) = (v.kind, v.ty) else { return None };
                if k.is_float() || unknown.contains(&s) || (v.frame == 0 && unknown.contains(&GLOBALS)) {
                    return None;
                }
                self.e.m.as_int(self.ssa.edge_value(&self.e, j, ei, slot))?
            }
            Expr::Unary(UnOp::Not, a) => (self.edge_int(a, j, ei, unknown)? == 0) as u64,
            Expr::Unary(UnOp::Plus, a) => self.edge_int(a, j, ei, unknown)?,
            Expr::Binary(op @ (B::LogOr | B::LogAnd), a, b) => {
                // the value that decides the operator alone
                let d = (op == B::LogOr) as u64;
                match self.edge_int(a, j, ei, unknown).map(|k| (k != 0) as u64) {
                    Some(k) if k == d => d,
                    Some(_) => (self.edge_int(b, j, ei, unknown)? != 0) as u64,
                    // an unknown left side that has no effect may as well not be evaluated
                    None if self.pure(a) && self.edge_int(b, j, ei, unknown).is_some_and(|k| (k != 0) as u64 == d) => d,
                    None => return None,
                }
            }
            Expr::Binary(op, a, b) => {
                let (a, b) = (self.edge_int(a, j, ei, unknown)?, self.edge_int(b, j, ei, unknown)?);
                match op {
                    B::Eq => (a == b) as u64,
                    B::Ne => (a != b) as u64,
                    B::Lt => (a < b) as u64,
                    B::Le => (a <= b) as u64,
                    B::Gt => (a > b) as u64,
                    B::Ge => (a >= b) as u64,
                    B::Add => a + b,
                    B::Sub => a.checked_sub(b)?,
                    B::Mul => a.checked_mul(b)?,
                    B::BitAnd => a & b,
                    B::BitOr => a | b,
                    B::BitXor => a ^ b,
                    _ => return None,
                }
            }
            _ => return None,
        };
        (v <= i32::MAX as u64).then_some(v)
    }
    /// An expression without calls and assignments.
    fn pure(&self, x: ExprId) -> bool {
        match *self.u.expr(x) {
            Expr::Int(..) | Expr::Float(..) | Expr::Bool(_) | Expr::Ident(_) => true,
            Expr::Unary(op, a) => !matches!(op, UnOp::PreInc | UnOp::PreDec | UnOp::PostInc | UnOp::PostDec) && self.pure(a),
            Expr::Binary(_, a, b) | Expr::Index(a, b) => self.pure(a) && self.pure(b),
            Expr::Ternary(c, a, b) => self.pure(c) && self.pure(a) && self.pure(b),
            Expr::Cast(_, a) | Expr::Member(a, _) => self.pure(a),
            _ => false,
        }
    }
}
