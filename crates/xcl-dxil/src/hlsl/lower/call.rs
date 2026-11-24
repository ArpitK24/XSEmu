//! Calls: user functions are resolved by name, argument count and types and lowered in
//! place (parameters are copied in and out, as HLSL defines them); `Interlocked*` on
//! `groupshared` memory are atomicrmw / cmpxchg; everything else is an intrinsic.

use super::super::ast::{qual, Range, Sym};
use super::super::types::{HTy, Sk, Val};
use super::expr::{Place, Pv};
use super::{Lower, RetCtx, VarKind};
use crate::ir::{RmwOp, Ty, Value};
use crate::{Error, Result};

enum Arg {
    Val(Val),
    /// an assignable argument (it may be bound to an `out` parameter) and its type
    Place(Place),
}

/// How far `from` is from a parameter of type `to`: 0 is the same type, None not convertible.
fn distance(from: HTy, to: HTy) -> Option<u32> {
    if from == to {
        return Some(0);
    }
    match (from, to) {
        (HTy::Num(fk, 1, fnn), HTy::Num(tk, 1, tn)) => {
            let shape = if fnn == tn {
                0
            } else if fnn == 1 {
                2
            } else if fnn > tn {
                16
            } else {
                return None;
            };
            let kind = match (fk, tk) {
                _ if fk == tk => 0,
                (Sk::LitInt, Sk::Int | Sk::Uint) => 1,
                (Sk::LitInt, Sk::Int64 | Sk::Uint64) => 2,
                (Sk::LitInt, Sk::Float | Sk::Double) => 3,
                (Sk::LitFloat, Sk::Float) => 1,
                (Sk::LitFloat, Sk::Double) => 2,
                (Sk::LitInt | Sk::LitFloat, _) => 5,
                (Sk::Int, Sk::Uint) | (Sk::Uint, Sk::Int) | (Sk::Int64, Sk::Uint64) | (Sk::Uint64, Sk::Int64) => 4,
                (Sk::Bool, _) => 6,
                (Sk::Int | Sk::Uint, Sk::Int64 | Sk::Uint64) | (Sk::Float, Sk::Double) => 6,
                _ => 8,
            };
            Some(shape + kind)
        }
        (HTy::HeapElem { sampler: false, .. }, HTy::Res(_)) | (HTy::HeapElem { sampler: true, .. }, HTy::Sampler { .. }) => Some(1),
        (HTy::Res(a), HTy::Res(b)) if a.kind == b.kind && a.uav == b.uav && a.elem == b.elem && a.elem_comps == b.elem_comps => Some(1),
        _ => None,
    }
}

impl Lower<'_> {
    pub(super) fn call(&mut self, name: Sym, args: Range) -> Result<Val> {
        let u = self.u;
        let argx = u.args(args);
        let (first, count) = self.by_name[name.0 as usize];
        if count == 0 {
            // no function of that name: a built-in, the hot ones without looking at the name
            match self.hot[name.0 as usize] {
                0 => {}
                hot => return self.hot_intrinsic(hot, name, args),
            }
        }
        let all = first as usize..(first + count) as usize;
        let fits = |l: &Self, k: usize| u.funcs[l.by_name_list[k] as usize].params.len() == argx.len();
        if !all.clone().any(|k| fits(self, k)) {
            let n = self.name(name);
            if n.starts_with("Interlocked") {
                return self.interlocked(n, args);
            }
            return self.intrinsic(name, args);
        }
        // an argument that some candidate takes as out / inout is a place
        let mut vals: Vec<Arg> = Vec::with_capacity(argx.len());
        let mut tys: Vec<HTy> = Vec::with_capacity(argx.len());
        for (i, &a) in argx.iter().enumerate() {
            let out = all.clone().any(|k| fits(self, k) && u.funcs[self.by_name_list[k] as usize].params[i].quals & qual::OUT != 0);
            if out {
                let p = self.place(a)?;
                tys.push(p.ty());
                vals.push(Arg::Place(p));
            } else {
                let v = self.expr(a)?;
                tys.push(v.ty);
                vals.push(Arg::Val(v));
            }
        }
        // the closest candidate
        let mut best: Option<(u32, u32, bool)> = None;
        for k in all {
            if !fits(self, k) {
                continue;
            }
            let f = self.by_name_list[k];
            let mut total = Some(0u32);
            for (p, &t) in u.funcs[f as usize].params.iter().zip(&tys) {
                let pt = self.decl_ty(p)?;
                let d = if p.quals & qual::IN != 0 || p.quals & qual::OUT == 0 { distance(t, pt) } else { distance(pt, t) };
                total = match (total, d) {
                    (Some(a), Some(b)) => Some(a + b),
                    _ => None,
                };
            }
            if let Some(d) = total {
                best = match best {
                    Some((bd, _, _)) if d > bd => best,
                    Some((bd, bf, _)) if d == bd => Some((bd, bf, true)),
                    _ => Some((d, f, false)),
                };
            }
        }
        match best {
            Some((_, f, false)) => self.inline(f, vals),
            Some(_) => self.unsupported(format!("ambiguous call of {}", self.name(name))),
            None => self.err(format!("no overload of {} takes these arguments", self.name(name))),
        }
    }

    /// Lower the body of function `f` here.
    fn inline(&mut self, f: u32, args: Vec<Arg>) -> Result<Val> {
        let u = self.u;
        let func = &u.funcs[f as usize];
        if std::mem::replace(&mut self.fn_busy[f as usize], true) {
            return self.unsupported("recursion");
        }
        // every call is lowered inside its caller: a chain of calls is as deep a recursion
        // here, and it ends before the stack does
        let here = 0u8;
        if self.stack_base.abs_diff(std::ptr::addr_of!(here) as usize) > super::STACK / 4 * 3 {
            return self.unsupported("calls nested too deep");
        }
        let Some(body) = func.body else { return Err(Error::Internal("call of a prototype".into())) };
        let line = self.e.line;
        let ret_ty = self.hty(func.ret)?;
        let scope = self.scope();
        let ret = self.blank(ret_ty, false)?;
        let ret_slot = self.ssa.len();
        for &c in ret.comps() {
            let t = self.e.ty_of(c);
            self.ssa.alloc(t, c);
        }
        // copy in; the parameters are locals of the new frame, declared after every
        // argument is read in the caller's
        let mut params = Vec::with_capacity(args.len());
        for (p, a) in func.params.iter().zip(&args) {
            let ty = self.decl_ty(p)?;
            let v = match a {
                Arg::Val(v) => self.coerce(v, ty)?,
                Arg::Place(pl) if p.quals & qual::IN != 0 || p.quals & qual::OUT == 0 => {
                    let v = self.load(pl)?;
                    self.coerce(&v, ty)?
                }
                Arg::Place(_) => self.blank(ty, false)?,
            };
            params.push((ty, v));
        }
        let caller = std::mem::replace(&mut self.frame, self.next_frame);
        self.next_frame += 1;
        let mut outs = Vec::new();
        for ((p, (ty, v)), a) in func.params.iter().zip(&params).zip(&args) {
            let i = match *ty {
                HTy::Res(_) | HTy::Sampler { .. } if p.quals & qual::OUT != 0 => return self.unsupported("an out parameter of a resource type"),
                HTy::Res(_) | HTy::Sampler { .. } => self.new_res_var(p.name, *ty, Some(v.one()))?,
                _ => self.new_ssa_var(p.name, *ty, v.comps()),
            };
            if let (Arg::Place(_), true) = (a, p.quals & qual::OUT != 0) {
                outs.push(i);
            }
        }
        let join = self.ssa.new_join(&self.e);
        self.ssa.open_region(&self.e);
        self.rets.push(RetCtx { join, slot: ret_slot, ty: ret_ty, depth: self.ssa.region_depth(), tail: false });
        // break and continue do not leave a function
        let (breaks, continues, loops) = (std::mem::take(&mut self.breaks), std::mem::take(&mut self.continues), std::mem::take(&mut self.loops));
        let lex = std::mem::replace(&mut self.dxb.fn_lex, self.an.fn_wave_lex(f));
        self.stmts(u.block(body), None)?;
        self.dxb.fn_lex = lex;
        (self.breaks, self.continues, self.loops) = (breaks, continues, loops);
        if !self.finish_body(join)? {
            return self.unsupported("a call that never returns");
        }
        self.frame = caller;
        self.e.line = line;
        let n = self.e.types.comps(ret_ty);
        let result = Val::new(ret_ty, &self.ssa.cur[ret_slot as usize..ret_slot as usize + n]);
        // copy out
        let mut k = 0;
        for (p, a) in func.params.iter().zip(&args) {
            if let (Arg::Place(pl), true) = (a, p.quals & qual::OUT != 0) {
                let var = &self.vars[outs[k] as usize];
                k += 1;
                let VarKind::Ssa(base) = var.kind else { continue };
                let ty = var.ty;
                let n = self.e.types.comps(ty);
                let v = Val::new(ty, &self.ssa.cur[base as usize..base as usize + n]);
                let v = self.coerce(&v, pl.ty())?;
                self.store(pl, &v)?;
            }
        }
        self.end_scope(scope);
        self.fn_busy[f as usize] = false;
        Ok(result)
    }

    /// `InterlockedAdd(shared[i], v, original)` and the rest on groupshared memory. On a
    /// resource element they are dx.ops of the resource, which are not lowered here.
    fn interlocked(&mut self, name: &str, args: Range) -> Result<Val> {
        let argx = self.u.args(args);
        if argx.len() < 2 {
            return self.err(format!("{name} arguments"));
        }
        let dest = match self.pv(argx[0])? {
            Pv::Place(p) => p,
            Pv::Val(_) => return self.err(format!("{name} needs a destination in memory")),
        };
        let (mem, index, off) = match &dest {
            Place::Mem { mem, index, ty: HTy::Num(Sk::Int | Sk::Uint, 1, 1), .. } => {
                let p = self.load_index(&dest);
                (*mem, *index, p)
            }
            Place::ResElem { oc, .. } => {
                // the dx.op of the resource
                let (mut places, vals) = self.builtin_args(&argx[1..])?;
                places.insert(0, None);
                let c = super::super::resources::index_atomic(&mut self.e, &oc.0, &oc.1, name, &vals)?;
                self.assign_outs(&places, &c.outs)?;
                return Ok(c.ret);
            }
            _ => return self.unsupported(format!("{name} on a variable that is not shared memory")),
        };
        let _ = index;
        let k = self.mems[mem as usize].elem;
        let kty = HTy::scalar(k);
        let base = self.mem_ptr(mem);
        let ptr = self.mem_elem_ptr(mem, base, off);
        let signed = k == Sk::Int;
        let op = match name {
            "InterlockedAdd" => Some(RmwOp::Add),
            "InterlockedAnd" => Some(RmwOp::And),
            "InterlockedOr" => Some(RmwOp::Or),
            "InterlockedXor" => Some(RmwOp::Xor),
            "InterlockedMin" => Some(if signed { RmwOp::Min } else { RmwOp::UMin }),
            "InterlockedMax" => Some(if signed { RmwOp::Max } else { RmwOp::UMax }),
            "InterlockedExchange" => Some(RmwOp::Xchg),
            "InterlockedCompareExchange" | "InterlockedCompareStore" => None,
            _ => return self.unsupported(format!("intrinsic {name}")),
        };
        let operand = |l: &mut Self, i: usize| -> Result<Value> {
            let v = l.expr(argx[i])?;
            Ok(l.coerce(&v, kty)?.one())
        };
        let (old, out) = match op {
            Some(op) => {
                if argx.len() > 3 || (op == RmwOp::Xchg && argx.len() != 3) {
                    return self.err(format!("{name} arguments"));
                }
                let v = operand(self, 1)?;
                self.e.touch(v);
                (self.e.body.atomic_rmw(self.e.bb, op, Ty::I32, ptr, v), argx.get(2))
            }
            None => {
                let store = name == "InterlockedCompareStore";
                if argx.len() != if store { 3 } else { 4 } {
                    return self.err(format!("{name} arguments"));
                }
                let (c, v) = (operand(self, 1)?, operand(self, 2)?);
                let pair = self.e.m.literal_struct(vec![Ty::I32, Ty::I1]);
                self.e.touch(c);
                self.e.touch(v);
                let r = self.e.body.cmpxchg(self.e.bb, pair, ptr, c, v);
                (self.e.extract(Ty::I32, r, 0), argx.get(3))
            }
        };
        if let Some(&o) = out {
            let p = self.place(o)?;
            let v = self.coerce(&Val::scalar(k, old), p.ty())?;
            self.store(&p, &v)?;
        }
        Ok(Val::void())
    }

    /// The flat index of a scalar place in memory.
    fn load_index(&mut self, p: &Place) -> Value {
        match p {
            Place::Mem { index, .. } => {
                let off = self.place_offset(p);
                let off = self.e.c_u32(off);
                self.e.bin(crate::ir::BinOp::Add, *index, off)
            }
            _ => Value::NONE,
        }
    }
}
