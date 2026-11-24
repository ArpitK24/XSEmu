//! HLSL types and typed values, shared by [`super::lower`], [`super::intrinsics`] and
//! [`super::resources`]. Owner: the lowering subagent (the other two are written against
//! what is public here; extend, do not reshape).
//!
//! DXIL has no vectors, so a value of any numeric type is its scalar components
//! ([`Val::comps`]), row-major for matrices; a resource or sampler value is one annotated
//! `%dx.types.Handle`; structs and arrays held in SSA form are their fields' / elements'
//! components in order.

use crate::consts::{ComponentType, ResourceKind};
use crate::ir::{Ty, Value};

/// Scalar kinds after HLSL's rules without 16-bit types: `half` and `min16float` are float,
/// `min16int` / `min16uint` are int / uint (the dialect uses none of them for storage).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Sk {
    Bool,
    Int,
    Uint,
    Int64,
    Uint64,
    Float,
    Double,
    /// an unsuffixed integer literal (or an expression of only such): takes the type of the
    /// other operand, int where nothing decides
    LitInt,
    /// an unsuffixed float literal: float against anything but double
    LitFloat,
}

impl Sk {
    pub fn is_float(self) -> bool {
        matches!(self, Sk::Float | Sk::Double | Sk::LitFloat)
    }
    pub fn is_int(self) -> bool {
        matches!(self, Sk::Int | Sk::Uint | Sk::Int64 | Sk::Uint64 | Sk::LitInt)
    }
    /// Signed integer kind (LitInt counts as signed).
    pub fn is_signed(self) -> bool {
        matches!(self, Sk::Int | Sk::Int64 | Sk::LitInt)
    }
    pub fn is_64(self) -> bool {
        matches!(self, Sk::Int64 | Sk::Uint64 | Sk::Double)
    }
    /// The IR type values of this kind have in SSA form (bool is i1).
    pub fn ir(self) -> Ty {
        match self {
            Sk::Bool => Ty::I1,
            Sk::Int | Sk::Uint | Sk::LitInt => Ty::I32,
            Sk::Int64 | Sk::Uint64 => Ty::I64,
            Sk::Float | Sk::LitFloat => Ty::F32,
            Sk::Double => Ty::F64,
        }
    }
    /// The kind a literal kind becomes when nothing else decides.
    pub fn concrete(self) -> Sk {
        match self {
            Sk::LitInt => Sk::Int,
            Sk::LitFloat => Sk::Float,
            k => k,
        }
    }
    /// The DXIL component type of a resource element / signature element of this kind.
    pub fn component_type(self) -> ComponentType {
        match self.concrete() {
            Sk::Bool => ComponentType::I1,
            Sk::Int => ComponentType::I32,
            Sk::Uint => ComponentType::U32,
            Sk::Int64 => ComponentType::I64,
            Sk::Uint64 => ComponentType::U64,
            Sk::Float => ComponentType::F32,
            _ => ComponentType::F64,
        }
    }
}

/// The kind both operands of an arithmetic or comparison operator are converted to (HLSL's
/// usual arithmetic conversions): double > float > uint64 > int64 > uint > int > bool, with
/// literals giving way to the other operand's kind (bool with bool is int for arithmetic:
/// the caller decides that).
pub fn common_sk(a: Sk, b: Sk) -> Sk {
    fn rank(k: Sk) -> u32 {
        match k {
            Sk::Bool => 0,
            Sk::LitInt => 1,
            Sk::Int => 2,
            Sk::Uint => 3,
            Sk::Int64 => 4,
            Sk::Uint64 => 5,
            Sk::LitFloat => 6,
            Sk::Float => 7,
            Sk::Double => 8,
        }
    }
    match (a, b) {
        (Sk::LitFloat, k) | (k, Sk::LitFloat) if !k.is_float() => {
            if k == Sk::LitInt || k == Sk::LitFloat {
                Sk::LitFloat
            } else {
                Sk::Float
            }
        }
        _ => {
            if rank(a) >= rank(b) {
                a
            } else {
                b
            }
        }
    }
}

/// A resource type: what `annotateHandle` and the access operations need to know.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ResTy {
    pub kind: ResourceKind,
    /// read-write (`RW*`): a UAV, else an SRV
    pub uav: bool,
    /// element scalar kind of a typed resource (Uint for raw buffers)
    pub elem: Sk,
    /// element components, 1..4
    pub elem_comps: u8,
    /// `unorm` / `snorm` float elements are not in the dialect; samples of MS textures (0:
    /// not written)
    pub samples: u8,
    /// `globallycoherent`
    pub coherent: bool,
    /// bytes per element of a structured buffer, else 0
    pub stride: u32,
}

/// An HLSL type. `Copy`: struct and array shapes live in [`TypeTable`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum HTy {
    Void,
    /// scalar (1, 1), vector (1, n) or matrix (rows, cols)
    Num(Sk, u8, u8),
    /// index in [`TypeTable::structs`]
    Struct(u32),
    /// index in [`TypeTable::arrays`]
    Array(u32),
    Res(ResTy),
    Sampler { comparison: bool },
    /// `ResourceDescriptorHeap` itself (index it to get a [`HTy::HeapElem`])
    ResHeap,
    /// `SamplerDescriptorHeap` itself
    SamplerHeap,
    /// `ResourceDescriptorHeap[i]` / `SamplerDescriptorHeap[i]` before it meets the type it is
    /// assigned or passed to: one component, the i32 index; `sampler`: which heap;
    /// `nonuniform`: the index went through NonUniformResourceIndex
    HeapElem { sampler: bool, nonuniform: bool },
}

impl HTy {
    pub const BOOL: HTy = HTy::Num(Sk::Bool, 1, 1);
    pub const INT: HTy = HTy::Num(Sk::Int, 1, 1);
    pub const UINT: HTy = HTy::Num(Sk::Uint, 1, 1);
    pub const FLOAT: HTy = HTy::Num(Sk::Float, 1, 1);
    pub const DOUBLE: HTy = HTy::Num(Sk::Double, 1, 1);
    pub const UINT64: HTy = HTy::Num(Sk::Uint64, 1, 1);
    pub const INT64: HTy = HTy::Num(Sk::Int64, 1, 1);

    pub fn vec(k: Sk, n: u8) -> HTy {
        HTy::Num(k, 1, n)
    }
    pub fn scalar(k: Sk) -> HTy {
        HTy::Num(k, 1, 1)
    }
    pub fn sk(self) -> Option<Sk> {
        match self {
            HTy::Num(k, ..) => Some(k),
            _ => None,
        }
    }
    /// Components of a numeric type (rows × cols).
    pub fn num_comps(self) -> Option<usize> {
        match self {
            HTy::Num(_, r, c) => Some(r as usize * c as usize),
            _ => None,
        }
    }
    pub fn is_scalar(self) -> bool {
        matches!(self, HTy::Num(_, 1, 1))
    }
    pub fn is_vector(self) -> bool {
        matches!(self, HTy::Num(_, 1, n) if n > 1)
    }
    /// The same shape with another scalar kind.
    pub fn with_sk(self, k: Sk) -> HTy {
        match self {
            HTy::Num(_, r, c) => HTy::Num(k, r, c),
            t => t,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StructTy {
    pub name: String,
    /// (field name, type)
    pub fields: Vec<(String, HTy)>,
}

#[derive(Default, Debug)]
pub struct TypeTable {
    pub structs: Vec<StructTy>,
    /// (element type, length)
    pub arrays: Vec<(HTy, u32)>,
}

impl TypeTable {
    pub fn array(&mut self, of: HTy, n: u32) -> HTy {
        let i = self.arrays.iter().position(|&a| a == (of, n)).unwrap_or_else(|| {
            self.arrays.push((of, n));
            self.arrays.len() - 1
        });
        HTy::Array(i as u32)
    }
    /// Scalar components of a value of type `t` in SSA form.
    pub fn comps(&self, t: HTy) -> usize {
        match t {
            HTy::Void => 0,
            HTy::Num(_, r, c) => r as usize * c as usize,
            HTy::Struct(s) => self.structs[s as usize].fields.iter().map(|f| self.comps(f.1)).sum(),
            HTy::Array(a) => self.comps(self.arrays[a as usize].0) * self.arrays[a as usize].1 as usize,
            HTy::Res(_) | HTy::Sampler { .. } | HTy::HeapElem { .. } => 1,
            HTy::ResHeap | HTy::SamplerHeap => 0,
        }
    }
}

/// A typed value: its type and scalar components (inline up to four, the common case).
#[derive(Clone, Debug)]
pub struct Val {
    pub ty: HTy,
    n: u32,
    inline: [Value; 4],
    more: Option<Box<[Value]>>,
}

impl Val {
    pub fn new(ty: HTy, comps: &[Value]) -> Val {
        let mut inline = [Value::NONE; 4];
        if comps.len() <= 4 {
            inline[..comps.len()].copy_from_slice(comps);
            Val { ty, n: comps.len() as u32, inline, more: None }
        } else {
            Val { ty, n: comps.len() as u32, inline, more: Some(comps.into()) }
        }
    }
    pub fn scalar(k: Sk, v: Value) -> Val {
        Val::new(HTy::scalar(k), &[v])
    }
    pub fn void() -> Val {
        Val::new(HTy::Void, &[])
    }
    pub fn comps(&self) -> &[Value] {
        match &self.more {
            Some(m) => m,
            None => &self.inline[..self.n as usize],
        }
    }
    pub fn comps_mut(&mut self) -> &mut [Value] {
        match &mut self.more {
            Some(m) => m,
            None => &mut self.inline[..self.n as usize],
        }
    }
    /// The single component of a scalar, resource or sampler value.
    pub fn one(&self) -> Value {
        debug_assert_eq!(self.n, 1, "{:?}", self.ty);
        self.inline[0]
    }
    pub fn len(&self) -> usize {
        self.n as usize
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

impl TypeTable {
    /// Element type and length of an array type.
    pub fn elem(&self, t: HTy) -> Option<(HTy, u32)> {
        match t {
            HTy::Array(a) => Some(self.arrays[a as usize]),
            _ => None,
        }
    }
    /// A struct's field by name: (first component, type).
    pub fn field(&self, s: u32, name: &str) -> Option<(usize, HTy)> {
        let mut at = 0;
        for (n, t) in &self.structs[s as usize].fields {
            if n == name {
                return Some((at, *t));
            }
            at += self.comps(*t);
        }
        None
    }
    /// The IR type of every SSA component of a value of type `t`, appended to `out`;
    /// `handle` is `%dx.types.Handle` (resources and samplers).
    pub fn comp_tys(&self, t: HTy, handle: Ty, out: &mut Vec<Ty>) {
        match t {
            HTy::Void | HTy::ResHeap | HTy::SamplerHeap => {}
            HTy::Num(k, r, c) => out.extend(std::iter::repeat(k.ir()).take(r as usize * c as usize)),
            HTy::Struct(s) => {
                for i in 0..self.structs[s as usize].fields.len() {
                    self.comp_tys(self.structs[s as usize].fields[i].1, handle, out);
                }
            }
            HTy::Array(a) => {
                let (of, n) = self.arrays[a as usize];
                for _ in 0..n {
                    self.comp_tys(of, handle, out);
                }
            }
            HTy::Res(_) | HTy::Sampler { .. } => out.push(handle),
            HTy::HeapElem { .. } => out.push(Ty::I32),
        }
    }
    /// The scalar kind of every component of a numeric aggregate (struct / array of numeric
    /// types), appended to `out`; false when the type holds anything else.
    pub fn comp_sks(&self, t: HTy, out: &mut Vec<Sk>) -> bool {
        match t {
            HTy::Num(k, r, c) => {
                out.extend(std::iter::repeat(k).take(r as usize * c as usize));
                true
            }
            HTy::Struct(s) => (0..self.structs[s as usize].fields.len()).all(|i| self.comp_sks(self.structs[s as usize].fields[i].1, out)),
            HTy::Array(a) => {
                let (of, n) = self.arrays[a as usize];
                (0..n).all(|_| self.comp_sks(of, out))
            }
            _ => false,
        }
    }
}

impl Val {
    pub fn comp(&self, i: usize) -> Value {
        self.comps()[i]
    }
    /// The same components seen as another type (of the same component count).
    pub fn retyped(&self, ty: HTy) -> Val {
        let mut v = self.clone();
        v.ty = ty;
        v
    }
}
