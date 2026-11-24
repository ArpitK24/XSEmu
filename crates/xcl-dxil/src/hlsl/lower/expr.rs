//! Expressions: values, places (what an lvalue expression denotes) and HLSL's operator
//! typing.

use super::super::ast::{BinOp as B, Expr, ExprId, FloatSuffix, IntSuffix, Range, Sym, UnOp};
use super::super::intrinsics;
use super::super::resources;
use super::super::stage::{Eval, ParamKind};
use super::super::types::{common_sk, HTy, Sk, Val};
use super::analysis::SSA_ARRAY_MAX;
use super::ssa::Term;
use super::{Lower, VarKind};
use crate::ir::{BinOp, FPred, IPred, Ty, Value};
use crate::Result;

/// Which components of a place an expression selects: all of them from `off` on, or a swizzle.
#[derive(Copy, Clone, Debug)]
pub(super) struct Sel {
    off: u32,
    swz: Option<([u8; 4], u8)>,
}

impl Sel {
    const ALL: Sel = Sel { off: 0, swz: None };
    fn at(&self, i: usize) -> u32 {
        match self.swz {
            Some((s, _)) => self.off + s[i] as u32,
            None => self.off + i as u32,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum Place {
    /// SSA components: component i is slot `base + sel.at(i)`
    Slots { ty: HTy, base: u32, sel: Sel },
    /// one of `count` elements (`stride` slots each, from `base`) of an SSA array or vector,
    /// chosen at run time: reads are select chains, writes selects per element
    Dyn { ty: HTy, base: u32, count: u32, stride: u32, sel: Sel, index: Value },
    /// scalars `index + sel.at(i)` of an array in memory
    Mem { ty: HTy, mem: u32, index: Value, sel: Sel },
    /// `texture[coord]`: the object and the coordinate (boxed: the rare place, and the
    /// only one that would make every place three times the size)
    ResElem { ty: HTy, oc: Box<(Val, Val)> },
    /// a resource variable (its slot holds a heap index, see `ResState`)
    ResVar { ty: HTy, slot: u32 },
    /// components `sel` of element `index` of a mesh output array (entry parameter
    /// `param`): write-only
    Stage { ty: HTy, param: u32, index: Value, sel: Sel },
}

impl Place {
    pub(super) fn ty(&self) -> HTy {
        match self {
            Place::Slots { ty, .. } | Place::Dyn { ty, .. } | Place::Mem { ty, .. } | Place::ResElem { ty, .. } | Place::Stage { ty, .. } | Place::ResVar { ty, .. } => *ty,
        }
    }
}

pub(super) enum Pv {
    Place(Place),
    Val(Val),
}

fn swizzle(name: &str, n: usize) -> Option<([u8; 4], u8)> {
    if name.is_empty() || name.len() > 4 {
        return None;
    }
    let mut out = [0u8; 4];
    let mut set = 0;
    for (i, c) in name.bytes().enumerate() {
        let (k, s) = match c {
            b'x' => (0, 1),
            b'y' => (1, 1),
            b'z' => (2, 1),
            b'w' => (3, 1),
            b'r' => (0, 2),
            b'g' => (1, 2),
            b'b' => (2, 2),
            b'a' => (3, 2),
            _ => return None,
        };
        if k as usize >= n || (set != 0 && set != s) {
            return None;
        }
        set = s;
        out[i] = k;
    }
    Some((out, name.len() as u8))
}

impl Lower<'_> {
    pub(super) fn expr(&mut self, x: ExprId) -> Result<Val> {
        match *self.u.expr(x) {
            Expr::Int(v, sfx) => {
                let (k, ok) = match sfx {
                    // an unsuffixed literal takes the type of what it meets; one that does
                    // not fit int is unsigned (DXC: 0x80000000 meets uint64_t as 2^31)
                    IntSuffix::None if v <= i32::MAX as u64 => (Sk::LitInt, true),
                    IntSuffix::None | IntSuffix::U if v <= u32::MAX as u64 => (Sk::Uint, true),
                    IntSuffix::None | IntSuffix::LL if v <= i64::MAX as u64 => (Sk::Int64, true),
                    IntSuffix::None | IntSuffix::U | IntSuffix::ULL | IntSuffix::LL => (Sk::Uint64, true),
                    // to DXC a long is 32 bits wide: `5l` is an int, `5ul` a uint
                    IntSuffix::L if v <= i32::MAX as u64 => (Sk::Int, true),
                    IntSuffix::UL if v <= u32::MAX as u64 => (Sk::Uint, true),
                    IntSuffix::L | IntSuffix::UL => (Sk::Int, false),
                };
                if !ok {
                    return self.unsupported("a long integer literal that does not fit 32 bits");
                }
                Ok(Val::scalar(k, self.e.c_int(k, v)))
            }
            Expr::Float(v, sfx) => Ok(match sfx {
                // a literal float is a double until it meets its type, as in DXC
                FloatSuffix::None => Val::scalar(Sk::LitFloat, self.e.c_f64(v)),
                FloatSuffix::F | FloatSuffix::H => Val::scalar(Sk::Float, self.e.c_f32(v as f32)),
                FloatSuffix::L => Val::scalar(Sk::Double, self.e.c_f64(v)),
            }),
            Expr::Bool(b) => Ok(Val::scalar(Sk::Bool, self.e.c_bool(b))),
            Expr::Ident(s) => {
                // the common leaf, a numeric local in SSA form: its components as they stand
                if let Some(i) = self.lookup(s) {
                    let v = &self.vars[i as usize];
                    if let (VarKind::Ssa(base), HTy::Num(k, r, c)) = (v.kind, v.ty) {
                        let n = r as usize * c as usize;
                        let val = Val::new(v.ty, &self.ssa.cur[base as usize..base as usize + n]);
                        if self.e.track && k == Sk::Bool {
                            self.dx_touch(&val);
                        }
                        return Ok(val);
                    }
                }
                match self.pv(x)? {
                    Pv::Place(p) => self.load(&p),
                    Pv::Val(v) => Ok(v),
                }
            }
            Expr::Member(..) | Expr::Index(..) => match self.pv(x)? {
                Pv::Place(p) => self.load(&p),
                Pv::Val(v) => Ok(v),
            },
            Expr::Unary(op, a) => self.unary(op, a),
            Expr::Binary(op, a, b) => match op {
                B::LogAnd | B::LogOr => self.logical(op == B::LogAnd, a, b),
                B::Comma => {
                    self.expr(a)?;
                    self.expr(b)
                }
                _ => {
                    let va = self.expr(a)?;
                    let vb = self.expr(b)?;
                    self.binary(op, &va, &vb)
                }
            },
            Expr::Assign(op, l, r) => {
                let rv = self.expr(r)?;
                let p = self.place(l)?;
                let v = match op {
                    None if rv.ty == p.ty() => rv,
                    None => self.coerce(&rv, p.ty())?,
                    Some(op) => {
                        let cur = self.load(&p)?;
                        let res = self.binary(op, &cur, &rv)?;
                        self.coerce(&res, p.ty())?
                    }
                };
                self.store(&p, &v)?;
                Ok(v)
            }
            Expr::Ternary(c, t, f) => self.ternary(c, t, f),
            Expr::Call(name, args) => self.call(name, args),
            Expr::Construct(ty, args) => {
                let ty = self.hty(ty)?;
                let HTy::Num(k, 1, n) = ty else { return self.unsupported("constructor of a non-vector type") };
                let n = n as usize;
                let mut out = [Value::NONE; 4];
                let mut have = 0;
                for &a in self.u.args(args) {
                    let v = self.expr(a)?;
                    let Some(from) = v.ty.sk() else { return self.err("constructor argument") };
                    for &c in v.comps() {
                        if have == n {
                            return self.err("too many constructor arguments");
                        }
                        out[have] = self.e.convert_scalar(c, from, k);
                        have += 1;
                    }
                }
                if have == 1 {
                    out = [out[0]; 4];
                } else if have != n {
                    return self.err("too few constructor arguments");
                }
                let val = Val::new(ty, &out[..n]);
                if self.e.track && n > 1 {
                    // DXC builds the vector with an insertelement per component
                    self.dx_touch(&val);
                }
                Ok(val)
            }
            Expr::Cast(ty, a) => {
                let ty = self.hty(ty)?;
                let v = self.expr(a)?;
                match (ty, v.ty) {
                    // (Struct)0
                    (HTy::Struct(_) | HTy::Array(_), HTy::Num(..)) if v.len() == 1 && self.e.m.as_int(v.one()) == Some(0) => self.blank(ty, true),
                    _ => self.coerce(&v, ty),
                }
            }
            Expr::Method(name, all) => {
                let (obj, args) = self.u.method(all);
                self.method(obj, name, args)
            }
            Expr::InitList(_) => self.err("an initialiser list outside a declaration"),
        }
    }

    /// The scalar boolean value of a condition.
    pub(super) fn cond(&mut self, x: ExprId) -> Result<Value> {
        let v = self.expr(x)?;
        self.truth(&v)
    }
    fn truth(&mut self, v: &Val) -> Result<Value> {
        match v.ty {
            HTy::Num(k, 1, 1) => Ok(self.e.convert_scalar(v.one(), k, Sk::Bool)),
            _ => self.err("a scalar condition is expected"),
        }
    }

    pub(super) fn place(&mut self, x: ExprId) -> Result<Place> {
        match self.pv(x)? {
            Pv::Place(p) => Ok(p),
            Pv::Val(_) => self.err("an assignable expression is expected"),
        }
    }

    /// An index as an i32 value.
    fn index_value(&mut self, i: ExprId) -> Result<Value> {
        let v = self.expr(i)?;
        match v.ty {
            HTy::Num(k, 1, 1) if !k.is_float() => Ok(self.e.convert_scalar(v.one(), k, Sk::Uint)),
            _ => self.err("an integer index is expected"),
        }
    }

    /// The entry parameter and element type, if `x` names an object of the stage.
    fn stage_obj(&self, x: ExprId) -> Option<(u32, HTy)> {
        let Expr::Ident(s) = *self.u.expr(x) else { return None };
        let v = &self.vars[self.lookup(s)? as usize];
        match v.kind {
            VarKind::Stage(p) => Some((p, v.ty)),
            _ => None,
        }
    }

    pub(super) fn pv(&mut self, x: ExprId) -> Result<Pv> {
        match *self.u.expr(x) {
            Expr::Ident(s) => {
                let Some(i) = self.lookup(s) else {
                    return match self.name(s) {
                        "ResourceDescriptorHeap" => Ok(Pv::Val(Val::new(HTy::ResHeap, &[]))),
                        "SamplerDescriptorHeap" => Ok(Pv::Val(Val::new(HTy::SamplerHeap, &[]))),
                        n => self.err(format!("undeclared identifier {n}")),
                    };
                };
                let v = &self.vars[i as usize];
                let ty = v.ty;
                Ok(match v.kind {
                    VarKind::Ssa(slot) if matches!(ty, HTy::Res(_) | HTy::Sampler { .. }) => Pv::Place(Place::ResVar { ty, slot }),
                    VarKind::Ssa(base) => Pv::Place(Place::Slots { ty, base, sel: Sel::ALL }),
                    VarKind::Mem(mem) => Pv::Place(Place::Mem { ty, mem, index: self.e.c_u32(0), sel: Sel::ALL }),
                    VarKind::Res(r) => Pv::Val(Val::new(ty, &[self.res_handle(r)?])),
                    VarKind::Stage(_) => return self.unsupported("a stage input array, stream or mesh output array used as a value"),
                })
            }
            Expr::Member(a, name) => {
                let base = self.pv(a)?;
                let ty = match &base {
                    Pv::Place(p) => p.ty(),
                    Pv::Val(v) => v.ty,
                };
                let name = self.name(name);
                match ty {
                    HTy::Num(k, 1, n) => {
                        let Some((swz, m)) = swizzle(name, n as usize) else { return self.err(format!("no member {name}")) };
                        if self.e.track && n > 1 {
                            // DXC extracts from (or inserts into) the whole vector: an
                            // instruction that reads every component (see `dxbreak`)
                            match &base {
                                Pv::Place(Place::Slots { base, sel, .. }) => {
                                    for j in 0..n as usize {
                                        let c = self.ssa.read(base + sel.at(j));
                                        self.e.touch(c);
                                    }
                                }
                                Pv::Val(v) => {
                                    for &c in v.comps() {
                                        self.e.touch(c);
                                    }
                                }
                                _ => {}
                            }
                        }
                        let ty = HTy::Num(k, 1, m);
                        let compose = |sel: Sel| {
                            let mut out = swz;
                            if let Some((inner, _)) = sel.swz {
                                for o in out.iter_mut().take(m as usize) {
                                    *o = inner[*o as usize];
                                }
                            }
                            Sel { off: sel.off, swz: Some((out, m)) }
                        };
                        Ok(match base {
                            Pv::Place(Place::Slots { base, sel, .. }) => Pv::Place(Place::Slots { ty, base, sel: compose(sel) }),
                            Pv::Place(Place::Dyn { base, count, stride, sel, index, .. }) => Pv::Place(Place::Dyn { ty, base, count, stride, sel: compose(sel), index }),
                            Pv::Place(Place::Mem { mem, index, sel, .. }) => Pv::Place(Place::Mem { ty, mem, index, sel: compose(sel) }),
                            Pv::Place(Place::Stage { param, index, sel, .. }) => Pv::Place(Place::Stage { ty, param, index, sel: compose(sel) }),
                            Pv::Place(p @ Place::ResElem { .. }) => {
                                let v = self.load(&p)?;
                                Pv::Val(self.swizzled(&v, ty, swz, m))
                            }
                            Pv::Place(Place::ResVar { .. }) => return self.err(format!("no member {name}")),
                            Pv::Val(v) => Pv::Val(self.swizzled(&v, ty, swz, m)),
                        })
                    }
                    HTy::Struct(s) => {
                        let Some((at, fty)) = self.e.types.field(s, name) else { return self.err(format!("no member {name}")) };
                        let n = self.e.types.comps(fty);
                        Ok(match base {
                            Pv::Place(Place::Slots { base, .. }) => Pv::Place(Place::Slots { ty: fty, base: base + at as u32, sel: Sel::ALL }),
                            Pv::Place(Place::Dyn { base, count, stride, sel, index, .. }) => {
                                Pv::Place(Place::Dyn { ty: fty, base, count, stride, sel: Sel { off: sel.off + at as u32, swz: None }, index })
                            }
                            Pv::Place(Place::Mem { mem, index, sel, .. }) => Pv::Place(Place::Mem { ty: fty, mem, index, sel: Sel { off: sel.off + at as u32, swz: None } }),
                            Pv::Place(Place::Stage { param, index, sel, .. }) => Pv::Place(Place::Stage { ty: fty, param, index, sel: Sel { off: sel.off + at as u32, swz: None } }),
                            Pv::Place(Place::ResElem { .. }) => return self.unsupported("a field of a resource element"),
                            Pv::Place(Place::ResVar { .. }) => return self.err(format!("no member {name}")),
                            Pv::Val(v) => Pv::Val(Val::new(fty, &v.comps()[at..at + n])),
                        })
                    }
                    // the `mips` and `sample` members of a texture, which DXC has
                    _ if matches!(name, "mips" | "sample") => self.unsupported(format!("the member {name}")),
                    _ => self.err(format!("no member {name}")),
                }
            }
            Expr::Index(a, i) => {
                if let Some((param, ety)) = self.stage_obj(a) {
                    let mut index = self.index_value(i)?;
                    self.e.define(std::slice::from_mut(&mut index));
                    let Some(io) = self.io.as_mut() else { return Err(crate::Error::Internal("stage object".into())) };
                    let len = match io.params()[param as usize].kind {
                        ParamKind::InputArray { len } | ParamKind::MeshVertices { len } | ParamKind::MeshPrimitives { len } | ParamKind::MeshIndices { len } => len,
                        _ => return self.err("a stream cannot be indexed"),
                    };
                    if self.e.m.as_int(index).is_some_and(|k| k >= len as u64) {
                        return self.err("index out of bounds");
                    }
                    if let ParamKind::InputArray { .. } = io.params()[param as usize].kind {
                        let comps = io.input_at(&mut self.e, param as usize, index)?;
                        return Ok(Pv::Val(Val::new(ety, &comps)));
                    }
                    return Ok(Pv::Place(Place::Stage { ty: ety, param, index, sel: Sel::ALL }));
                }
                let base = self.pv(a)?;
                let ty = match &base {
                    Pv::Place(p) => p.ty(),
                    Pv::Val(v) => v.ty,
                };
                match ty {
                    HTy::ResHeap | HTy::SamplerHeap => {
                        // NonUniformResourceIndex marks the handle, not the value
                        let (i, nonuniform) = match *self.u.expr(i) {
                            Expr::Call(f, r) if self.name(f) == "NonUniformResourceIndex" && r.len == 1 => (self.u.args(r)[0], true),
                            _ => (i, false),
                        };
                        let mut idx = self.index_value(i)?;
                        self.e.define(std::slice::from_mut(&mut idx));
                        Ok(Pv::Val(Val::new(HTy::HeapElem { sampler: ty == HTy::SamplerHeap, nonuniform }, &[idx])))
                    }
                    HTy::Res(rt) => {
                        let obj = match base {
                            Pv::Val(v) => v,
                            Pv::Place(p) => self.load(&p)?,
                        };
                        let coord = self.expr(i)?;
                        let mut coord = self.e.concrete(&coord, false)?;
                        self.e.define(coord.comps_mut());
                        Ok(Pv::Place(Place::ResElem { ty: HTy::Num(rt.elem, 1, rt.elem_comps), oc: Box::new((obj, coord)) }))
                    }
                    HTy::Array(_) | HTy::Num(_, 1, _) => {
                        let (ety, count) = match ty {
                            HTy::Num(k, 1, n) => (HTy::scalar(k), n as u32),
                            _ => self.e.types.elem(ty).unwrap_or((ty, 0)),
                        };
                        let stride = self.e.types.comps(ety) as u32;
                        let idx = self.index_value(i)?;
                        let konst = self.e.m.as_int(idx);
                        if let Some(k) = konst {
                            if k >= count as u64 {
                                return self.err("index out of bounds");
                            }
                        }
                        Ok(match (base, konst) {
                            (Pv::Place(Place::Slots { base, sel, .. }), Some(k)) => match sel.swz {
                                None => Pv::Place(Place::Slots { ty: ety, base: base + k as u32 * stride, sel: Sel::ALL }),
                                Some((s, _)) => Pv::Place(Place::Slots { ty: ety, base: base + s[k as usize] as u32, sel: Sel::ALL }),
                            },
                            (Pv::Place(Place::Slots { base, sel: Sel { swz: None, .. }, .. }), None) => Pv::Place(Place::Dyn { ty: ety, base, count, stride, sel: Sel::ALL, index: idx }),
                            (Pv::Place(Place::Dyn { base, count: outer, sel: Sel { off, swz: None }, index, stride: os, .. }), _) if off == 0 && os == count * stride => {
                                // an element of an element: one flat run-time index
                                let n = self.e.c_u32(count);
                                let scaled = self.e.bin(BinOp::Mul, index, n);
                                let index = self.e.bin(BinOp::Add, scaled, idx);
                                Pv::Place(Place::Dyn { ty: ety, base, count: outer * count, stride, sel: Sel::ALL, index })
                            }
                            (Pv::Place(Place::Mem { mem, index, sel: Sel { off, swz: None }, .. }), _) => {
                                let s = self.e.c_u32(stride);
                                let scaled = self.e.bin(BinOp::Mul, idx, s);
                                let index = self.e.bin(BinOp::Add, index, scaled);
                                Pv::Place(Place::Mem { ty: ety, mem, index, sel: Sel { off, swz: None } })
                            }
                            (Pv::Place(Place::Stage { param, index, sel, .. }), Some(k)) => {
                                let off = match sel.swz {
                                    None => sel.off + k as u32 * stride,
                                    Some((s, _)) => sel.off + s[k as usize] as u32,
                                };
                                Pv::Place(Place::Stage { ty: ety, param, index, sel: Sel { off, swz: None } })
                            }
                            (Pv::Place(Place::Stage { .. }), None) => return self.unsupported("a run-time index into a mesh output element"),
                            (Pv::Place(p), _) => {
                                // a swizzle indexed at run time and the like: a value only
                                let v = self.load(&p)?;
                                Pv::Val(self.index_val(&v, ety, count, stride, idx))
                            }
                            (Pv::Val(v), _) => Pv::Val(self.index_val(&v, ety, count, stride, idx)),
                        })
                    }
                    _ => self.err("this type cannot be indexed"),
                }
            }
            _ => Ok(Pv::Val(self.expr(x)?)),
        }
    }

    fn swizzled(&mut self, v: &Val, ty: HTy, swz: [u8; 4], n: u8) -> Val {
        let mut out = [Value::NONE; 4];
        for i in 0..n as usize {
            out[i] = v.comps()[swz[i] as usize];
        }
        Val::new(ty, &out[..n as usize])
    }

    /// Element `idx` of a value: the element itself for a constant index, else a select
    /// chain over the elements (the last one where no earlier index matches).
    fn index_val(&mut self, v: &Val, ety: HTy, count: u32, stride: u32, idx: Value) -> Val {
        let stride = stride as usize;
        let mut out = Vec::with_capacity(stride);
        for c in 0..stride {
            out.push(self.chain(idx, count, |k| v.comps()[k as usize * stride + c]));
        }
        Val::new(ety, &out)
    }
    fn chain(&mut self, idx: Value, count: u32, at: impl Fn(u32) -> Value) -> Value {
        if let Some(k) = self.e.m.as_int(idx) {
            return at((k as u32).min(count - 1));
        }
        let mut r = at(count - 1);
        for k in (0..count - 1).rev() {
            let kc = self.e.c_u32(k);
            let is = self.e.icmp(IPred::Eq, idx, kc);
            r = self.e.select(is, at(k), r);
        }
        r
    }

    /// The first scalar a place in memory selects, relative to its index.
    pub(super) fn place_offset(&self, p: &Place) -> u32 {
        match p {
            Place::Mem { sel, .. } => sel.at(0),
            _ => 0,
        }
    }

    pub(super) fn load(&mut self, p: &Place) -> Result<Val> {
        if let Place::ResVar { ty, slot } = *p {
            return self.res_var_load(slot, ty);
        }
        let n = self.e.types.comps(p.ty());
        let mut inline = [Value::NONE; 4];
        let mut more = Vec::new();
        for i in 0..n {
            let v = match p {
                Place::Slots { base, sel, .. } => self.ssa.read(base + sel.at(i)),
                Place::Dyn { base, count, stride, sel, index, .. } => {
                    let (base, stride, off) = (*base, *stride, sel.at(i));
                    let cur = &self.ssa.cur;
                    let at = |k: u32| cur[(base + k * stride + off) as usize];
                    // (the closure borrows the slots; the chain only emits)
                    if let Some(k) = self.e.m.as_int(*index) {
                        at((k as u32).min(count - 1))
                    } else {
                        let mut r = at(count - 1);
                        for k in (0..count - 1).rev() {
                            let kc = self.e.c_u32(k);
                            let is = self.e.icmp(IPred::Eq, *index, kc);
                            r = self.e.select(is, at(k), r);
                        }
                        r
                    }
                }
                Place::Mem { mem, index, sel, .. } => {
                    let off = self.e.c_u32(sel.at(i));
                    let idx = self.e.bin(BinOp::Add, *index, off);
                    self.mem_load(*mem, idx)
                }
                Place::ResElem { oc, .. } => return resources::index_load(&mut self.e, &oc.0, &oc.1),
                Place::Stage { .. } => return self.err("a mesh shader output cannot be read"),
                Place::ResVar { .. } => return Err(crate::Error::Internal("resource variable".into())),
            };
            if n <= 4 {
                inline[i] = v;
            } else {
                more.push(v);
            }
        }
        let val = if n <= 4 { Val::new(p.ty(), &inline[..n]) } else { Val::new(p.ty(), &more) };
        if self.e.track && Self::dx_instruction(p) {
            self.dx_touch(&val);
        }
        Ok(val)
    }

    /// Whether DXC has an instruction where this place is read or written whole, which a
    /// plain copy of a variable is not: a `bool` is converted from and to the i32 it is
    /// kept as, and components of a vector are extracted from it or inserted into it
    /// (see `dxbreak`).
    fn dx_instruction(p: &Place) -> bool {
        let part = match p {
            Place::Slots { sel, .. } | Place::Dyn { sel, .. } | Place::Mem { sel, .. } | Place::Stage { sel, .. } => sel.swz.is_some(),
            _ => false,
        };
        part || p.ty().sk() == Some(Sk::Bool)
    }
    pub(super) fn dx_touch(&mut self, v: &Val) {
        for &c in v.comps() {
            self.e.touch(c);
        }
    }

    /// Store `v` (already of the place's type).
    pub(super) fn store(&mut self, p: &Place, v: &Val) -> Result<()> {
        if let Place::ResVar { slot, .. } = *p {
            return self.res_var_store(slot, v.one());
        }
        if self.e.track && Self::dx_instruction(p) {
            self.dx_touch(v);
        }
        for (i, &c) in v.comps().iter().enumerate() {
            match p {
                Place::Slots { base, sel, .. } => self.ssa.write(base + sel.at(i), c),
                Place::Dyn { base, count, stride, sel, index, .. } => {
                    for k in 0..*count {
                        let slot = base + k * stride + sel.at(i);
                        let kc = self.e.c_u32(k);
                        let is = self.e.icmp(IPred::Eq, *index, kc);
                        let old = self.ssa.read(slot);
                        let new = self.e.select(is, c, old);
                        if new != old {
                            self.ssa.write(slot, new);
                        }
                    }
                }
                Place::Mem { mem, index, sel, .. } => {
                    if self.mems[*mem as usize].constant {
                        return self.err("assignment to a constant");
                    }
                    let off = self.e.c_u32(sel.at(i));
                    let idx = self.e.bin(BinOp::Add, *index, off);
                    let ptr = self.mem_ptr(*mem);
                    self.mem_store(*mem, ptr, idx, c);
                }
                Place::ResElem { oc, .. } => {
                    let mut v = v.clone();
                    self.e.define(v.comps_mut());
                    return resources::index_store(&mut self.e, &oc.0, &oc.1, &v);
                }
                Place::ResVar { .. } => {}
                Place::Stage { param, index, sel, .. } => {
                    let Some(io) = self.io.as_mut() else { return Err(crate::Error::Internal("stage object".into())) };
                    if sel.swz.is_none() {
                        return io.mesh_store(&mut self.e, *param as usize, *index, sel.off as usize, v.comps());
                    }
                    // a swizzle may reorder: one component at a time
                    io.mesh_store(&mut self.e, *param as usize, *index, sel.at(i) as usize, &[c])?;
                }
            }
        }
        Ok(())
    }

    fn unary(&mut self, op: UnOp, a: ExprId) -> Result<Val> {
        if matches!(op, UnOp::PreInc | UnOp::PreDec | UnOp::PostInc | UnOp::PostDec) {
            let p = self.place(a)?;
            let old = self.load(&p)?;
            let one = Val::scalar(Sk::LitInt, self.e.c_u32(1));
            let r = self.binary(if matches!(op, UnOp::PreInc | UnOp::PostInc) { B::Add } else { B::Sub }, &old, &one)?;
            let new = self.coerce(&r, p.ty())?;
            self.store(&p, &new)?;
            return Ok(if matches!(op, UnOp::PreInc | UnOp::PreDec) { new } else { old });
        }
        let v = self.expr(a)?;
        let HTy::Num(k, 1, n) = v.ty else { return self.err("operand type") };
        let mut out = [Value::NONE; 4];
        let (rk, v) = match op {
            UnOp::Not => (Sk::Bool, self.e.convert(&v, HTy::Num(Sk::Bool, 1, n))?),
            _ if k == Sk::Bool => (Sk::Int, self.e.convert(&v, HTy::Num(Sk::Int, 1, n))?),
            _ => (k, v),
        };
        for (i, &c) in v.comps().iter().enumerate() {
            out[i] = match op {
                UnOp::Plus => c,
                UnOp::Not => self.e.not(c),
                UnOp::Neg if rk.is_float() => {
                    let t = self.e.ty_of(c);
                    let z = if t == Ty::F64 { self.e.c_f64(-0.0) } else { self.e.c_f32(-0.0) };
                    self.e.bin(BinOp::FSub, z, c)
                }
                UnOp::Neg => {
                    let t = self.e.ty_of(c);
                    let z = self.e.zero_of(t);
                    self.e.bin(BinOp::Sub, z, c)
                }
                UnOp::BitNot if rk.is_float() => return self.err("~ of a float"),
                _ => {
                    let t = self.e.ty_of(c);
                    let ones = self.e.c_of(t, u64::MAX);
                    self.e.bin(BinOp::Xor, c, ones)
                }
            };
        }
        Ok(Val::new(HTy::Num(rk, 1, n), &out[..n as usize]))
    }

    /// A binary operator on two values, with HLSL's conversions: both operands to the common
    /// kind (shifts: the right one to the left one's), a scalar splat to the other's size,
    /// the longer vector cut to the shorter.
    pub(super) fn binary(&mut self, op: B, a: &Val, b: &Val) -> Result<Val> {
        let (HTy::Num(ka, 1, na), HTy::Num(kb, 1, nb)) = (a.ty, b.ty) else { return self.err("operand types") };
        let n = if na == 1 {
            nb
        } else if nb == 1 {
            na
        } else {
            na.min(nb)
        };
        let shift = matches!(op, B::Shl | B::Shr);
        let compare = matches!(op, B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge);
        let bitwise = matches!(op, B::BitAnd | B::BitOr | B::BitXor);
        let mut k = if shift || ka == kb { ka } else { common_sk(ka, kb) };
        if k == Sk::Bool && !compare && !bitwise {
            k = Sk::Int;
        }
        if shift && k == Sk::LitInt && kb != Sk::LitInt {
            k = Sk::Int;
        }
        if (shift || bitwise || op == B::Rem && false) && k.is_float() {
            return self.err("operand of a bit operation is a float");
        }
        let ty = HTy::Num(k, 1, n);
        let (ca, cb);
        let a = if a.ty == ty {
            a
        } else {
            ca = self.e.convert(a, ty)?;
            &ca
        };
        let b = if b.ty == ty {
            b
        } else {
            cb = self.e.convert(b, ty)?;
            &cb
        };
        let float = k.is_float();
        let signed = k.is_signed();
        let mut out = [Value::NONE; 4];
        for i in 0..n as usize {
            let (x, y) = (a.comp(i), b.comp(i));
            out[i] = match op {
                B::Add => self.e.bin(if float { BinOp::FAdd } else { BinOp::Add }, x, y),
                B::Sub => self.e.bin(if float { BinOp::FSub } else { BinOp::Sub }, x, y),
                B::Mul => self.e.bin(if float { BinOp::FMul } else { BinOp::Mul }, x, y),
                B::Div => self.e.bin(if float { BinOp::FDiv } else if signed { BinOp::SDiv } else { BinOp::UDiv }, x, y),
                B::Rem => self.e.bin(if float { BinOp::FRem } else if signed { BinOp::SRem } else { BinOp::URem }, x, y),
                B::BitAnd => self.e.bin(BinOp::And, x, y),
                B::BitOr => self.e.bin(BinOp::Or, x, y),
                B::BitXor => self.e.bin(BinOp::Xor, x, y),
                B::Shl | B::Shr => {
                    // DXC masks the amount to the width of the shifted type
                    let t = self.e.ty_of(x);
                    let bits = self.e.m.bits(t) as u64;
                    let mask = self.e.c_of(t, bits - 1);
                    let amount = self.e.bin(BinOp::And, y, mask);
                    self.e.bin(if op == B::Shl { BinOp::Shl } else if signed { BinOp::AShr } else { BinOp::LShr }, x, amount)
                }
                B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge if float => {
                    let p = match op {
                        B::Eq => FPred::Oeq,
                        B::Ne => FPred::Une,
                        B::Lt => FPred::Olt,
                        B::Le => FPred::Ole,
                        B::Gt => FPred::Ogt,
                        _ => FPred::Oge,
                    };
                    self.e.fcmp(p, x, y)
                }
                B::Eq => self.e.icmp(IPred::Eq, x, y),
                B::Ne => self.e.icmp(IPred::Ne, x, y),
                B::Lt => self.e.icmp(if signed { IPred::Slt } else { IPred::Ult }, x, y),
                B::Le => self.e.icmp(if signed { IPred::Sle } else { IPred::Ule }, x, y),
                B::Gt => self.e.icmp(if signed { IPred::Sgt } else { IPred::Ugt }, x, y),
                B::Ge => self.e.icmp(if signed { IPred::Sge } else { IPred::Uge }, x, y),
                B::LogAnd | B::LogOr | B::Comma => return self.err("operator"),
            };
        }
        Ok(Val::new(if compare { HTy::Num(Sk::Bool, 1, n) } else { ty }, &out[..n as usize]))
    }

    /// `a && b` / `a || b`: scalars only in HLSL 2021, and `b` runs only if `a` does not decide.
    /// `and` / `or` when `b` may run regardless, else a branch around it.
    fn logical(&mut self, and: bool, a: ExprId, b: ExprId) -> Result<Val> {
        let va = self.cond(a)?;
        if let Some(c) = self.e.m.as_int(va) {
            if (c != 0) != and {
                return Ok(Val::scalar(Sk::Bool, va));
            }
            let vb = self.cond(b)?;
            return Ok(Val::scalar(Sk::Bool, vb));
        }
        if self.an.speculatable(self.u, b) {
            // DXC has blocks here: the right-hand side in one, a phi of it in the next
            self.e.boundary();
            let vb = self.cond(b)?;
            self.e.boundary();
            self.e.touch(vb);
            let track = std::mem::replace(&mut self.e.track, false);
            let v = self.e.bin(if and { BinOp::And } else { BinOp::Or }, va, vb);
            self.e.track = track;
            return Ok(Val::scalar(Sk::Bool, v));
        }
        let head = self.e.bb;
        let join = self.ssa.new_join(&self.e);
        let rhs = self.e.body.new_block();
        self.ssa.add_edge(join, head, if and { Term::CondFalse { cond: va, other: rhs } } else { Term::CondTrue { cond: va, other: rhs } });
        self.ssa.open_region(&self.e);
        self.e.bb = rhs;
        self.e.boundary();
        let vb = self.cond(b)?;
        let end = self.e.bb;
        self.ssa.add_edge(join, end, Term::Br);
        self.ssa.undo_region(&mut self.e);
        self.ssa.resolve(&mut self.e, join);
        let phi = self.e.body.phi(self.e.bb, Ty::I1);
        let decided = self.e.c_bool(!and);
        self.e.body.phi_add(phi, decided, head);
        self.e.body.phi_add(phi, vb, end);
        Ok(Val::scalar(Sk::Bool, phi))
    }

    /// Whether the two blocks clang emits for the arms `t` and `f` of `?:` are still blocks
    /// when DXC decides its conditional breaks. At -O1 SimplifyCFG has run once by then, and
    /// it folds the arms into a select when each takes at most two instructions that may
    /// run unconditionally (`c ? i * 2u : 9u`: a load and a multiplication; measured:
    /// `gi * i + 1u` against `i + 2u` stays blocks). So the blocks stay when an arm calls
    /// something, divides or assigns, or takes more than two instructions by the count
    /// below, which is a lower bound: an arm that is miscounted is taken as folded, which
    /// errs towards keeping a break conditional.
    pub(super) fn cfg_keeps(&self, t: ExprId, f: ExprId) -> bool {
        match (self.spec_cost(t), self.spec_cost(f)) {
            (Some(a), Some(b)) => a > 2 || b > 2,
            _ => true,
        }
    }

    /// A lower bound of the instructions clang emits for `x` (a load per variable read, one
    /// per operation), or None when one of them cannot be run unconditionally.
    pub(super) fn spec_cost(&self, x: ExprId) -> Option<u32> {
        Some(match *self.u.expr(x) {
            Expr::Int(..) | Expr::Float(..) | Expr::Bool(_) => 0,
            Expr::Ident(s) => {
                let i = self.sym_top[s.0 as usize];
                (i != super::NONE && self.vars[i as usize].frame != 0) as u32
            }
            Expr::Unary(op, a) => match op {
                UnOp::PreInc | UnOp::PreDec | UnOp::PostInc | UnOp::PostDec => return None,
                UnOp::Plus => self.spec_cost(a)?,
                _ => self.spec_cost(a)? + 1,
            },
            Expr::Binary(op, a, b) => match op {
                B::Div | B::Rem | B::LogAnd | B::LogOr => return None,
                B::Comma => self.spec_cost(a)? + self.spec_cost(b)?,
                _ => self.spec_cost(a)? + self.spec_cost(b)? + 1,
            },
            Expr::Ternary(c, a, b) => self.spec_cost(c)? + self.spec_cost(a)? + self.spec_cost(b)? + 1,
            Expr::Cast(_, a) | Expr::Member(a, _) => self.spec_cost(a)?,
            Expr::Index(a, i) => self.spec_cost(a)? + self.spec_cost(i)?,
            Expr::Construct(_, args) | Expr::InitList(args) => {
                let mut n = 0;
                for &a in self.u.args(args) {
                    n += self.spec_cost(a)?;
                }
                n
            }
            Expr::Assign(..) | Expr::Call(..) | Expr::Method(..) => return None,
        })
    }

    /// clang's isCheapEnoughToEvaluateUnconditionally: a constant or a local variable.
    fn clang_cheap(&self, x: ExprId) -> bool {
        match *self.u.expr(x) {
            Expr::Int(..) | Expr::Float(..) | Expr::Bool(_) => true,
            Expr::Ident(s) => {
                let i = self.sym_top[s.0 as usize];
                i != super::NONE && self.vars[i as usize].frame != 0
            }
            Expr::Unary(UnOp::Neg | UnOp::Plus, a) | Expr::Cast(_, a) => matches!(*self.u.expr(a), Expr::Int(..) | Expr::Float(..) | Expr::Bool(_)),
            _ => false,
        }
    }

    /// The type both arms of `?:` are converted to.
    fn arm_type(&mut self, a: HTy, b: HTy) -> Result<HTy> {
        match (a, b) {
            _ if a == b => Ok(a),
            (HTy::Num(ka, 1, na), HTy::Num(kb, 1, nb)) => {
                let n = if na == 1 {
                    nb
                } else if nb == 1 {
                    na
                } else {
                    na.min(nb)
                };
                Ok(HTy::Num(common_sk(ka, kb), 1, n))
            }
            _ => self.err("the arms of ?: have different types"),
        }
    }

    /// `c ? t : f` with a scalar condition: only the chosen arm runs. A `select` when both
    /// arms may run regardless, else a branch.
    fn ternary(&mut self, c: ExprId, t: ExprId, f: ExprId) -> Result<Val> {
        let vc = self.cond(c)?;
        if let Some(k) = self.e.m.as_int(vc) {
            // the type is still the common type of both arms; the dead arm is not lowered,
            // so only a literal arm changes type
            let v = self.expr(if k != 0 { t } else { f })?;
            return Ok(match v.ty {
                HTy::Num(Sk::LitInt, 1, n) => self.e.convert(&v, HTy::Num(Sk::Int, 1, n))?,
                HTy::Num(Sk::LitFloat, 1, n) => self.e.convert(&v, HTy::Num(Sk::Float, 1, n))?,
                _ => v,
            });
        }
        if self.an.speculatable(self.u, t) && self.an.speculatable(self.u, f) {
            // clang emits a select too when both arms are a constant or a local variable
            // ("cheap enough to evaluate unconditionally"); else each arm has a block and
            // the result is a phi in a third
            let blocks = self.e.track && !(self.clang_cheap(t) && self.clang_cheap(f)) && self.cfg_keeps(t, f);
            if blocks {
                self.e.boundary();
            }
            let vt = self.expr(t)?;
            let vf = self.expr(f)?;
            let ty = self.arm_type(vt.ty, vf.ty)?;
            let (vt, vf) = (self.coerce(&vt, ty)?, self.coerce(&vf, ty)?);
            if blocks {
                self.e.boundary();
            }
            let mut out = Vec::with_capacity(vt.len());
            for i in 0..vt.len() {
                out.push(if blocks { self.e.select_merge(vc, vt.comp(i), vf.comp(i)) } else { self.e.select(vc, vt.comp(i), vf.comp(i)) });
            }
            return Ok(Val::new(ty, &out));
        }
        let head = self.e.bb;
        let join = self.ssa.new_join(&self.e);
        let tb = self.e.body.new_block();
        self.ssa.open_region(&self.e);
        self.e.bb = tb;
        self.e.boundary();
        let vt = self.expr(t)?;
        let t_end = self.e.bb;
        self.ssa.add_edge(join, t_end, Term::Br);
        self.ssa.undo_region(&mut self.e);
        let fb = self.e.body.new_block();
        self.e.body.cond_br(head, vc, tb, fb);
        self.ssa.open_region(&self.e);
        self.e.bb = fb;
        self.e.boundary();
        let vf = self.expr(f)?;
        let f_end = self.e.bb;
        self.ssa.add_edge(join, f_end, Term::Br);
        self.ssa.undo_region(&mut self.e);
        // the conversions to the common type belong to the arms: their blocks are still open
        let ty = self.arm_type(vt.ty, vf.ty)?;
        let mark = self.e.vn_mark();
        self.e.bb = t_end;
        let vt = self.coerce(&vt, ty)?;
        self.e.vn_release(mark);
        self.e.bb = f_end;
        let vf = self.coerce(&vf, ty)?;
        self.e.vn_release(mark);
        self.ssa.resolve(&mut self.e, join);
        let mut out = Vec::with_capacity(vt.len());
        for i in 0..vt.len() {
            let (x, y) = (vt.comp(i), vf.comp(i));
            out.push(if x == y {
                x
            } else {
                let t = self.e.ty_of(x);
                let phi = self.e.body.phi(self.e.bb, t);
                self.e.body.phi_add(phi, x, t_end);
                self.e.body.phi_add(phi, y, f_end);
                phi
            });
        }
        Ok(Val::new(ty, &out))
    }

    /// Arguments of an intrinsic or a method: places where the expression is one (it may be
    /// an `out` argument), and the values, literals made concrete.
    pub(super) fn builtin_args(&mut self, args: &[ExprId]) -> Result<(Vec<Option<Place>>, Vec<Val>)> {
        let (mut places, mut vals) = self.arg_pool.pop().unwrap_or_default();
        for &a in args {
            match self.pv(a)? {
                Pv::Place(p) => {
                    vals.push(self.load(&p)?);
                    places.push(Some(p));
                }
                Pv::Val(v) => {
                    vals.push(v);
                    places.push(None);
                }
            }
        }
        let double = vals.iter().any(|v| v.ty.sk() == Some(Sk::Double));
        self.ready_args(&mut vals, double)?;
        Ok((places, vals))
    }

    /// What every argument of a built-in goes through before the call.
    fn ready_args(&mut self, vals: &mut [Val], double: bool) -> Result<()> {
        for v in vals.iter_mut() {
            if matches!(v.ty, HTy::Num(Sk::LitFloat, ..)) {
                *v = self.e.concrete(v, double)?;
            }
            // a variable read before it is assigned is 0 once it reaches a dx.op
            self.e.define(v.comps_mut());
        }
        Ok(())
    }

    /// `select`, `asfloat`, `asuint`, `asint`, `and` and `or` (`hot`: 1 to 6) on scalars of
    /// the kinds they are declared with: most of what the translator's output is made of,
    /// and one instruction each, as [`intrinsics::call`] makes them too, but without the
    /// argument lists and the general conversions on the way. Vectors, literals and mixed
    /// kinds go there, with the arguments evaluated here.
    pub(super) fn hot_intrinsic(&mut self, hot: u8, name: Sym, args: Range) -> Result<Val> {
        let argx = self.u.args(args);
        let want = match hot {
            1 => 3,
            2..=4 => 1,
            _ => 2,
        };
        if argx.len() != want {
            return self.general_intrinsic(name, args);
        }
        let mut v = [Val::void(), Val::void(), Val::void()];
        for (v, &a) in v.iter_mut().zip(argx) {
            *v = self.expr(a)?;
        }
        let sk = |v: &Val| match v.ty {
            HTy::Num(k, 1, 1) => Some(k),
            _ => None,
        };
        let mut c = [v[0].comps().first().copied().unwrap_or(Value::NONE), v[1].comps().first().copied().unwrap_or(Value::NONE), v[2].comps().first().copied().unwrap_or(Value::NONE)];
        match (hot, sk(&v[0]), sk(&v[1]), sk(&v[2])) {
            (1, Some(Sk::Bool), Some(k), Some(k2)) if k == k2 && !matches!(k, Sk::LitInt | Sk::LitFloat) => {
                self.e.define(&mut c);
                return Ok(Val::scalar(k, self.e.select(c[0], c[1], c[2])));
            }
            (2..=4, Some(Sk::Float | Sk::Int | Sk::Uint), ..) => {
                let to = [Sk::Float, Sk::Uint, Sk::Int][hot as usize - 2];
                self.e.define(&mut c[..1]);
                return Ok(Val::scalar(to, self.e.cast(crate::ir::CastOp::BitCast, to.ir(), c[0])));
            }
            (5 | 6, Some(Sk::Bool), Some(Sk::Bool), _) => {
                self.e.define(&mut c[..2]);
                return Ok(Val::scalar(Sk::Bool, self.e.bin(if hot == 5 { BinOp::And } else { BinOp::Or }, c[0], c[1])));
            }
            _ => {}
        }
        let (places, mut vals) = self.arg_pool.pop().unwrap_or_default();
        vals.extend(v.into_iter().take(want));
        let double = vals.iter().any(|v| v.ty.sk() == Some(Sk::Double));
        self.ready_args(&mut vals, double)?;
        let name = self.name(name);
        match intrinsics::call(&mut self.e, name, &vals)? {
            Some(c) if c.outs.is_empty() => {
                self.recycle_args(places, vals);
                Ok(c.ret)
            }
            Some(_) => Err(crate::Error::Internal(format!("intrinsic {name}: an out argument"))),
            None => self.unsupported(format!("intrinsic {name}")),
        }
    }

    /// Hand the lists [`Self::builtin_args`] returned back for the next call to fill.
    pub(super) fn recycle_args(&mut self, mut places: Vec<Option<Place>>, mut vals: Vec<Val>) {
        places.clear();
        vals.clear();
        self.arg_pool.push((places, vals));
    }

    pub(super) fn assign_outs(&mut self, places: &[Option<Place>], outs: &[(usize, Val)]) -> Result<()> {
        for (i, v) in outs {
            let Some(Some(p)) = places.get(*i) else { return self.err("an out argument must be assignable") };
            let v = self.coerce(v, p.ty())?;
            self.store(p, &v)?;
        }
        Ok(())
    }

    /// The intrinsics that belong to the stage: `SetMeshOutputCounts`, `DispatchMesh`,
    /// `EvaluateAttributeCentroid` / `AtSample` / `Snapped`. None: not one of them.
    fn stage_call(&mut self, name: Sym, args: Range) -> Result<Option<Val>> {
        let argx = self.u.args(args);
        let uint = HTy::Num(Sk::Uint, 1, 1);
        let how = match (self.name(name), argx.len()) {
            ("SetMeshOutputCounts", 2) => {
                let mut c = [Value::NONE; 2];
                for (c, &a) in c.iter_mut().zip(argx) {
                    let v = self.expr(a)?;
                    *c = self.coerce(&v, uint)?.one();
                }
                self.e.define(&mut c);
                let Some(io) = self.io.as_mut() else { return Ok(None) };
                io.set_mesh_output_counts(&mut self.e, c[0], c[1])?;
                return Ok(Some(Val::void()));
            }
            ("DispatchMesh", 4) => {
                let mut c = [Value::NONE; 3];
                for (c, &a) in c.iter_mut().zip(argx) {
                    let v = self.expr(a)?;
                    *c = self.coerce(&v, uint)?.one();
                }
                self.e.define(&mut c);
                let payload = self.expr(argx[3])?;
                let HTy::Struct(s) = payload.ty else { return self.err("the payload of DispatchMesh must be a struct") };
                let u = self.u;
                let named = |t: &super::super::ast::TypeRef| matches!(t, super::super::ast::TypeRef::Named(n) if self.struct_of.get(n) == Some(&s));
                let Some(tid) = u.types.iter().position(named) else { return Err(crate::Error::Internal("payload type".into())) };
                let Some(io) = self.io.as_mut() else { return Ok(None) };
                io.dispatch_mesh(&mut self.e, c, super::super::ast::TypeId(tid as u32), payload.comps())?;
                return Ok(Some(Val::void()));
            }
            ("EvaluateAttributeCentroid", 1) => Eval::Centroid,
            ("EvaluateAttributeAtSample", 2) => Eval::AtSample,
            ("EvaluateAttributeSnapped", 2) => Eval::Snapped,
            _ => return Ok(None),
        };
        // DXC wants an input, or a field or swizzle of one
        let Pv::Place(Place::Slots { ty, base, sel }) = self.pv(argx[0])? else { return self.err("the argument of EvaluateAttribute must be an input") };
        let n = self.e.types.comps(ty);
        let first = base + sel.at(0);
        if (0..n).any(|i| base + sel.at(i) != first + i as u32) {
            return self.unsupported("EvaluateAttribute of a reordering swizzle");
        }
        let Some(&(pbase, _, param)) = self.stage_inputs.iter().find(|&&(b, len, _)| first >= b && first + n as u32 <= b + len) else {
            return self.err("the argument of EvaluateAttribute must be an input");
        };
        let arg = match how {
            Eval::Centroid => Val::void(),
            Eval::AtSample | Eval::Snapped => {
                let v = self.expr(argx[1])?;
                let mut v = self.coerce(&v, HTy::Num(Sk::Int, 1, if how == Eval::Snapped { 2 } else { 1 }))?;
                self.e.define(v.comps_mut());
                v
            }
        };
        let Some(io) = self.io.as_mut() else { return Ok(None) };
        let out = io.eval_attribute(&mut self.e, how, param as usize, (first - pbase) as usize, n, arg.comps())?;
        Ok(Some(Val::new(ty, &out)))
    }

    pub(super) fn intrinsic(&mut self, name: Sym, args: Range) -> Result<Val> {
        if self.io.is_some() {
            if let Some(v) = self.stage_call(name, args)? {
                return Ok(v);
            }
        }
        match self.hot[name.0 as usize] {
            0 => self.general_intrinsic(name, args),
            hot => self.hot_intrinsic(hot, name, args),
        }
    }

    fn general_intrinsic(&mut self, name: Sym, args: Range) -> Result<Val> {
        let (places, vals) = self.builtin_args(self.u.args(args))?;
        let name = self.name(name);
        match intrinsics::call(&mut self.e, name, &vals)? {
            Some(c) => {
                if !c.outs.is_empty() {
                    self.assign_outs(&places, &c.outs)?;
                }
                self.recycle_args(places, vals);
                Ok(c.ret)
            }
            None => self.unsupported(format!("intrinsic {name}")),
        }
    }

    fn method(&mut self, obj: ExprId, name: Sym, args: Range) -> Result<Val> {
        if let Some((param, ety)) = self.stage_obj(obj) {
            let argx = self.u.args(args);
            match (self.name(name), argx.len()) {
                ("Append", 1) => {
                    let v = self.expr(argx[0])?;
                    let v = self.coerce(&v, ety)?;
                    let Some(io) = self.io.as_mut() else { return Err(crate::Error::Internal("stage object".into())) };
                    io.append(&mut self.e, param as usize, v.comps())?;
                }
                ("RestartStrip", 0) => {
                    let Some(io) = self.io.as_mut() else { return Err(crate::Error::Internal("stage object".into())) };
                    io.restart_strip(&mut self.e, param as usize)?;
                }
                (n, _) => return self.err(format!("no method {n}")),
            }
            return Ok(Val::void());
        }
        let o = self.expr(obj)?;
        let name = self.name(name);
        if !matches!(o.ty, HTy::Res(_)) {
            return self.unsupported(format!("method {name} of a non-resource"));
        }
        let (places, vals) = self.builtin_args(self.u.args(args))?;
        let c = resources::method(&mut self.e, &o, name, &vals)?;
        if !c.outs.is_empty() {
            self.assign_outs(&places, &c.outs)?;
        }
        self.recycle_args(places, vals);
        Ok(c.ret)
    }
}

// the limit is part of the place rules above: arrays past it never are Slots
const _: () = assert!(SSA_ARRAY_MAX <= 64);
