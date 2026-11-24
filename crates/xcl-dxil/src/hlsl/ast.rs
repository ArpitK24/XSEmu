//! Syntax tree of the HLSL dialect, as index-based arenas (a 30 MB source has millions of
//! nodes). Purely syntactic: names are not resolved and nothing is typed. Owner: the parser
//! subagent; additions are fine, but [`super::lower`] is written against these shapes, so
//! change existing ones only together with it.
//!
//! What the tree means where the grammar leaves a choice ([`super::parse`] builds exactly
//! this, [`super::astprint`] prints it back):
//!
//! * **Preprocessor.** It runs inside the lexer. The tree holds only the code the `#if` /
//!   `#ifdef` lines select, with every macro use replaced by its expansion: no node says a
//!   macro or a conditional was there. A line number inside an expansion is the line of the
//!   macro's use.
//! * **Array dimensions.** [`VarDecl::dims`] holds only the `[n]` written after the name, in
//!   source order, which is outermost first: `uint x[4]` is `uint` with dims `[4]`, and
//!   `uint4 v[2][3]` is `uint4` with dims `[2, 3]` (2 arrays of 3 vectors, `v[i][j]` has
//!   `i < 2`). The components of a vector or matrix are part of the type ([`TypeRef::Num`]),
//!   never a dimension. `[]` without a size is not in the dialect (`Unsupported`).
//! * **Vector and matrix names.** `floatN` is `rows = 1, cols = N`; `floatRxC` is
//!   `rows = R, cols = C` for `R >= 2`. `float1` and `float1xC` (`vector<T, 1>`,
//!   `matrix<T, 1, c>`) would be indistinguishable from the scalar and the vectors here; the
//!   dialect has none of them and the parser rejects them (`Unsupported`). Synonyms are folded: `dword` / `uint32_t` are `Uint`, `int32_t` is
//!   `Int`, `float32_t` is `Float`, `float64_t` is `Double`; `vector<T, n>` and
//!   `matrix<T, r, c>` give the same [`TypeRef::Num`] as the short names. Types are interned:
//!   equal [`TypeRef`]s share one [`TypeId`].
//! * **Members.** `a.b.c` is `Member(Member(a, b), c)`; whether a name is a field or a
//!   swizzle is the lowering's call (it knows the type of the left side). `a.b(x)` is
//!   `Method(b, [a, x])`, with `a` any expression (`a.b.c(x)` is
//!   `Method(c, [Member(a, b), x])`).
//! * **Calls, constructors, casts.** `name(args)` with a numeric type name (`uint4(...)`,
//!   `float3(x)`, `uint(x)`) is [`Expr::Construct`], with any other name [`Expr::Call`]
//!   (user functions and intrinsics alike, `select` / `and` / `or` included). `(T)e` is
//!   [`Expr::Cast`] with `T` a numeric type or a declared struct (`(XclOut)0`) and `e` the
//!   whole unary expression after it: `(uint)x.y` casts `x.y`, `(uint)(x)` is the same tree as
//!   `(uint)x`, and `(uint)(x) + y` adds `y` to the cast. Parentheses leave no node: `(x)`
//!   is `x`.
//! * **Operators.** C precedence and associativity. `c ? a : b = d` assigns inside the third
//!   operand, as in C++ (and DXC). `-1` is `Unary(Neg, Int(1))`: literals carry no sign.
//! * **Literals.** [`Expr::Int`] holds the exact value (decimal, `0x` hexadecimal, or octal
//!   with a leading 0, as in C) and the suffix as written, case and order aside
//!   ([`IntSuffix`]: DXC makes `ll` / `ull` 64-bit and `l` / `ul` 32-bit). [`Expr::Float`]
//!   holds the value DXC's lexer gives the literal: with no suffix or `l` the `f64` nearest to
//!   the decimal text; with `f` or `h` the `f32` nearest to it, widened exactly (so `as f32`
//!   is exact and there is no double rounding: `1.00000005960464477539062500000000001f` is
//!   0x3F800001 in DXC, and 0x3F800000 without the `f`).
//! * **Declarations.** One [`VarDecl`] per declared name; `uint a = 1u, b[2];` in a function is
//!   one [`Stmt::Var`] over two consecutive entries of [`Unit::vars`] (same type and
//!   qualifiers), at top level two [`Global::Var`]s. `struct S { ... } a, b;` is
//!   [`Global::Struct`] followed by a [`Global::Var`] of type `Named(S)` per declarator.
//!   Structs are declared at top level only, and always named. An initialiser is an
//!   expression or an [`Expr::InitList`] (which appears nowhere else, and nests).
//! * **Names.** A name is any identifier, and also a type or qualifier word that DXC lets a
//!   program declare (`half`, `uint4`, `dword`, `Texture2D`, `sample`, `precise`, ...; not
//!   `float` / `int` / `bool` / `double`, nor `line` / `point` / `triangle` / `linear`). The
//!   translator's prelude has `uint64_t half = ...`: from that declaration to the end of
//!   its block (for a parameter, of the function) `half` is the variable and not the type.
//!   A global of such a name hides the type for the rest of the unit; a struct field hides
//!   nothing.
//! * **Qualifiers.** [`VarDecl::quals`] is the union of the words written before the type
//!   (order is not kept; `in out` is `inout`). On a function, `static` and `inline` are
//!   accepted and dropped (every function is inlined anyway); anything else there is
//!   `Unsupported`.
//! * **Attributes.** `[a] [b(1)]` before a function are its [`Func::attrs`]; before a
//!   statement they are in [`Unit::stmt_attrs`] under the id of that statement (the `for`,
//!   `if`, `switch`, ... itself, whatever its kind), in source order.
//! * **Statements.** Ids are post-order: a statement's children have smaller ids than it.
//!   `else if` is an [`Stmt::If`] in `els`. A `for` initialiser is a [`Stmt::Var`] or a
//!   [`Stmt::Expr`]. In a `switch` every label starts a new [`Case`], in source order
//!   (`default` wherever it was written); control falls from the end of a case's body into
//!   the next case unless the body ends the flow (`break`, `return`, ...).

use std::collections::HashMap;
use std::fmt;

/// An interned identifier.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Sym(pub u32);

/// One slot of the interner's open-addressing table.
#[derive(Copy, Clone)]
struct Slot {
    /// names of up to 16 bytes: the bytes themselves (little-endian, zero-padded), so a hit
    /// needs no second memory access; longer names: their hash, and 0
    key: (u64, u64),
    sym: u32,
    len: u32,
}

const NO_SYM: u32 = u32::MAX;
const HASH_MUL: u64 = 0x9E37_79B9_7F4A_7C15;
/// Which of the 16 bytes at the start of a name of 0 to 15 bytes are its own.
const LEN_MASK: [(u64, u64); 16] = len_masks();

const fn len_masks() -> [(u64, u64); 16] {
    let mut t = [(0u64, 0u64); 16];
    let mut n = 1;
    while n < 16 {
        t[n] = if n < 8 {
            ((1u64 << (8 * n)) - 1, 0)
        } else if n == 8 {
            (u64::MAX, 0)
        } else {
            (u64::MAX, (1u64 << (8 * (n - 8))) - 1)
        };
        n += 1;
    }
    t
}

/// The identifiers of a unit, numbered in order of first appearance (the parser interns its
/// keyword and type names first, so user names do not start at 0). The lexer interns every
/// identifier token once, straight from the source bytes: no `String` per token, and for the
/// names of up to 16 bytes, nearly all of a translated program, one probe of one table.
#[derive(Default, Clone)]
pub struct Interner {
    slots: Vec<Slot>,
    /// 64 - log2(slots.len())
    shift: u32,
    /// (offset, length) of each name in `text`
    spans: Vec<(u32, u32)>,
    text: String,
}

impl fmt::Debug for Interner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Interner").field("names", &self.spans.len()).finish()
    }
}

/// The 16 bytes at the start of a name as two words: of a name of up to 16 bytes, its key
/// once the bytes after it are masked off.
#[inline]
pub(crate) fn halves(a: &[u8; 16]) -> (u64, u64) {
    let v = u128::from_le_bytes(*a);
    (v as u64, (v >> 64) as u64)
}

/// The key of a name: see [`Slot::key`].
fn key_of(s: &[u8]) -> (u64, u64) {
    if s.len() > 16 {
        return (long_key(s), 0);
    }
    let mut a = [0u8; 16];
    a[..s.len()].copy_from_slice(s);
    halves(&a)
}

/// The hash of a name of more than 16 bytes.
fn long_key(s: &[u8]) -> u64 {
    let mut h = (s.len() as u64).wrapping_mul(HASH_MUL);
    let mut chunks = s.chunks_exact(8);
    for c in &mut chunks {
        let mut a = [0u8; 8];
        a.copy_from_slice(c);
        h = (h.rotate_left(23) ^ u64::from_le_bytes(a)).wrapping_mul(HASH_MUL);
    }
    // the last 8 bytes again (they overlap the chunks unless the length is a multiple of 8)
    let mut a = [0u8; 8];
    a.copy_from_slice(&s[s.len() - 8..]);
    (h.rotate_left(23) ^ u64::from_le_bytes(a)).wrapping_mul(HASH_MUL)
}

impl Interner {
    /// An interner with room for `names` identifiers before its table grows.
    pub fn with_capacity(names: usize) -> Self {
        let mut s = Self::default();
        s.spans.reserve(names);
        s.text.reserve(names.saturating_mul(8));
        s.resize((names.saturating_mul(2)).next_power_of_two().max(16));
        s
    }

    pub fn intern(&mut self, s: &str) -> Sym {
        let b = s.as_bytes();
        Sym(self.intern_at(b, 0, b.len()))
    }

    pub fn get(&self, s: &str) -> Option<Sym> {
        if self.slots.is_empty() {
            return None;
        }
        let b = s.as_bytes();
        let key = key_of(b);
        let mask = self.slots.len() - 1;
        let mut i = self.home(key, b.len()) & mask;
        loop {
            let slot = self.slots[i];
            if slot.sym == NO_SYM {
                return None;
            }
            if slot.key == key && slot.len as usize == b.len() && (b.len() <= 16 || self.name(Sym(slot.sym)).as_bytes() == b) {
                return Some(Sym(slot.sym));
            }
            i = (i + 1) & mask;
        }
    }

    pub fn name(&self, s: Sym) -> &str {
        let (at, len) = self.spans[s.0 as usize];
        &self.text[at as usize..(at + len) as usize]
    }

    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Where the probing for a name starts (to be masked by the caller if the table can be
    /// empty: its shift is 0 then).
    #[inline]
    fn home(&self, key: (u64, u64), len: usize) -> usize {
        (((key.0 ^ len as u64).wrapping_mul(HASH_MUL) ^ key.1).wrapping_mul(HASH_MUL) >> self.shift) as usize
    }

    /// Interns `src[start..end]`.
    pub(crate) fn intern_at(&mut self, src: &[u8], start: usize, end: usize) -> u32 {
        let name = &src[start..end];
        self.find_or_insert(key_of(name), name)
    }

    /// The symbol of the name of `len` bytes, 1 to 15, that the 16 bytes `head` ([`halves`])
    /// start with, if it has been interned: the lexer's path for a word whose end it found in
    /// the 16 bytes at its start. They are the key, so nothing is copied and nothing but the
    /// slot is read.
    #[inline]
    pub(crate) fn find_short(&self, head: (u64, u64), len: usize) -> Option<u32> {
        debug_assert!((1..16).contains(&len));
        let mask = LEN_MASK[len & 15];
        let key = (head.0 & mask.0, head.1 & mask.1);
        let slots = self.slots.len().wrapping_sub(1);
        let mut i = self.home(key, len);
        loop {
            let slot = self.slots.get(i & slots)?;
            if slot.sym == NO_SYM {
                return None;
            }
            if slot.key == key && slot.len as usize == len {
                return Some(slot.sym);
            }
            i += 1;
        }
    }

    #[inline]
    fn find_or_insert(&mut self, key: (u64, u64), name: &[u8]) -> u32 {
        let len = name.len();
        let mask = self.slots.len().wrapping_sub(1);
        let mut i = self.home(key, len);
        loop {
            let Some(&slot) = self.slots.get(i & mask) else {
                return self.first(key, name);
            };
            if slot.sym == NO_SYM {
                return self.insert(i & mask, key, name);
            }
            if slot.key == key && slot.len as usize == len && (len <= 16 || self.name(Sym(slot.sym)).as_bytes() == name) {
                return slot.sym;
            }
            i += 1;
        }
    }

    /// The first name of an interner made without a capacity.
    #[cold]
    fn first(&mut self, key: (u64, u64), name: &[u8]) -> u32 {
        self.resize(16);
        let at = self.home(key, name.len());
        self.insert(at, key, name)
    }

    #[cold]
    fn insert(&mut self, at: usize, key: (u64, u64), name: &[u8]) -> u32 {
        let sym = self.spans.len() as u32;
        self.spans.push((self.text.len() as u32, name.len() as u32));
        match std::str::from_utf8(name) {
            Ok(s) => self.text.push_str(s),
            // not reachable from `intern` or the lexer (identifier bytes are ASCII)
            Err(_) => self.text.extend(name.iter().map(|&b| if b < 0x80 { b as char } else { '?' })),
        }
        self.slots[at] = Slot { key, sym, len: name.len() as u32 };
        if self.spans.len() * 2 > self.slots.len() {
            self.resize(self.slots.len() * 2);
        }
        sym
    }

    fn resize(&mut self, slots: usize) {
        let old = std::mem::replace(&mut self.slots, vec![Slot { key: (0, 0), sym: NO_SYM, len: 0 }; slots]);
        self.shift = 64 - slots.trailing_zeros();
        let mask = slots - 1;
        for slot in old {
            if slot.sym == NO_SYM {
                continue;
            }
            let mut i = self.home(slot.key, slot.len as usize);
            while self.slots[i].sym != NO_SYM {
                i = (i + 1) & mask;
            }
            self.slots[i] = slot;
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ExprId(pub u32);
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct StmtId(pub u32);
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TypeId(pub u32);

/// A run of entries in one of the unit's list pools.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Range {
    pub start: u32,
    pub len: u32,
}

impl Range {
    pub fn of<'a, T>(&self, pool: &'a [T]) -> &'a [T] {
        &pool[self.start as usize..(self.start + self.len) as usize]
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum ScalarKind {
    Bool,
    Int,
    Uint,
    Int64,
    Uint64,
    /// `half`: a 32-bit float unless 16-bit types are enabled (they are not)
    Half,
    Float,
    Double,
    Min16Float,
    Min16Int,
    Min16Uint,
    Int16,
    Uint16,
    Float16,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum ResKind {
    Texture1D,
    Texture1DArray,
    Texture2D,
    Texture2DArray,
    Texture2DMS,
    Texture2DMSArray,
    Texture3D,
    TextureCube,
    TextureCubeArray,
    RWTexture1D,
    RWTexture1DArray,
    RWTexture2D,
    RWTexture2DArray,
    RWTexture3D,
    Buffer,
    RWBuffer,
    ByteAddressBuffer,
    RWByteAddressBuffer,
    StructuredBuffer,
    RWStructuredBuffer,
    AppendStructuredBuffer,
    ConsumeStructuredBuffer,
    ConstantBuffer,
    RaytracingAccelerationStructure,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum StreamKind {
    Point,
    Line,
    Triangle,
}

/// A type as written.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TypeRef {
    Void,
    /// scalar (rows = cols = 1), vector (rows = 1, cols = n: `uint4`, `vector<float, 3>`) or
    /// matrix (`float4x4`: rows = cols = 4)
    Num { kind: ScalarKind, rows: u8, cols: u8 },
    /// a struct (or typedef) name
    Named(Sym),
    /// `Texture2D<uint4>`, `RWByteAddressBuffer`, `RWStructuredBuffer<T>`,
    /// `Texture2DMS<float4, 4>`; `elem` None: none written (float4 for textures)
    Resource { kind: ResKind, elem: Option<TypeId>, samples: Option<u32> },
    Sampler { comparison: bool },
    /// `TriangleStream<T>` (geometry shader output)
    Stream { kind: StreamKind, elem: TypeId },
    /// `InputPatch<T, n>` / `OutputPatch<T, n>`
    Patch { output: bool, elem: TypeId, points: u32 },
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum IntSuffix {
    /// none: a literal int, which takes the type of what it meets
    None,
    U,
    /// `l`: to DXC a 32-bit `long` (`0x1FFFFFFFFl >> 32` is -1), not in the dialect
    L,
    /// `ul` or `lu`: 32-bit too, not in the dialect
    UL,
    /// `ll`: `int64_t`
    LL,
    /// `ull` or `llu`: `uint64_t` (the translator writes `0xFFFFFFFFull`, `1ull << n`)
    ULL,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FloatSuffix {
    /// none: a literal float
    None,
    F,
    H,
    L,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum UnOp {
    Neg,
    Plus,
    /// `!`
    Not,
    /// `~`
    BitNot,
    PreInc,
    PreDec,
    PostInc,
    PostDec,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Shl,
    Shr,
    BitAnd,
    BitOr,
    BitXor,
    /// `&&` (short-circuit for scalars in HLSL 2021)
    LogAnd,
    /// `||`
    LogOr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Comma,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Expr {
    Int(u64, IntSuffix),
    Float(f64, FloatSuffix),
    Bool(bool),
    Ident(Sym),
    Unary(UnOp, ExprId),
    Binary(BinOp, ExprId, ExprId),
    /// `l = r`, or `l op= r` with Some(op)
    Assign(Option<BinOp>, ExprId, ExprId),
    /// `c ? t : f` (short-circuit for a scalar condition in HLSL 2021)
    Ternary(ExprId, ExprId, ExprId),
    /// `name(args)`: a user function or an intrinsic; args in [`Unit::expr_lists`]
    Call(Sym, Range),
    /// `uint4(a, b, c, d)`, `float3(x)`: a constructor of a numeric type
    Construct(TypeId, Range),
    /// `(uint)x`
    Cast(TypeId, ExprId),
    /// `obj.Name(args)`: the object is the first entry of the list ([`Unit::method`]), which
    /// keeps every expression at 16 bytes
    Method(Sym, Range),
    /// `obj.name`: a swizzle or a field
    Member(ExprId, Sym),
    Index(ExprId, ExprId),
    /// `{ a, b, c }`
    InitList(Range),
}

/// Declaration qualifiers, as bits.
pub mod qual {
    pub const STATIC: u32 = 1 << 0;
    pub const CONST: u32 = 1 << 1;
    pub const GROUPSHARED: u32 = 1 << 2;
    pub const PRECISE: u32 = 1 << 3;
    pub const GLOBALLYCOHERENT: u32 = 1 << 4;
    pub const IN: u32 = 1 << 5;
    pub const OUT: u32 = 1 << 6;
    /// `inout` sets IN | OUT
    pub const INOUT: u32 = IN | OUT;
    pub const NOINTERPOLATION: u32 = 1 << 7;
    pub const LINEAR: u32 = 1 << 8;
    pub const CENTROID: u32 = 1 << 9;
    pub const NOPERSPECTIVE: u32 = 1 << 10;
    pub const SAMPLE: u32 = 1 << 11;
    pub const UNIFORM: u32 = 1 << 12;
    /// mesh shader outputs: `out vertices`, `out indices`, `out primitives`, `in payload`
    pub const VERTICES: u32 = 1 << 13;
    pub const INDICES: u32 = 1 << 14;
    pub const PRIMITIVES: u32 = 1 << 15;
    pub const PAYLOAD: u32 = 1 << 16;
    /// geometry shader input primitive: `point`, `line`, `triangle`, `lineadj`, `triangleadj`
    pub const GS_POINT: u32 = 1 << 17;
    pub const GS_LINE: u32 = 1 << 18;
    pub const GS_TRIANGLE: u32 = 1 << 19;
    pub const GS_LINEADJ: u32 = 1 << 20;
    pub const GS_TRIANGLEADJ: u32 = 1 << 21;
    pub const ROW_MAJOR: u32 = 1 << 22;
    pub const COLUMN_MAJOR: u32 = 1 << 23;
    pub const VOLATILE: u32 = 1 << 24;
    pub const EXTERN: u32 = 1 << 25;
    pub const INLINE: u32 = 1 << 26;
}

/// `: register(u3, space1)`
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Register {
    /// b't', b'u', b's' or b'b'
    pub class: u8,
    pub index: u32,
    pub space: u32,
}

/// One declared name: a global, a local, a parameter or a struct field.
#[derive(Clone, PartialEq, Debug)]
pub struct VarDecl {
    pub name: Sym,
    pub ty: TypeId,
    /// array dimensions as written after the name, outermost first (constant expressions;
    /// `[]` is not in the dialect): see the module documentation
    pub dims: Vec<ExprId>,
    pub init: Option<ExprId>,
    /// [`qual`] bits
    pub quals: u32,
    /// the semantic as written, with its index: `TEXCOORD3`, `SV_Target0`, `SV_Position`
    pub semantic: Option<Sym>,
    pub register: Option<Register>,
    pub line: u32,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Stmt {
    /// declarations of one statement (`uint a = 1u, b;`), in [`Unit::vars`]
    Var(Range),
    Expr(ExprId),
    /// `{ ... }`: statements in [`Unit::stmt_lists`]
    Block(Range),
    If { cond: ExprId, then: StmtId, els: Option<StmtId> },
    For { init: Option<StmtId>, cond: Option<ExprId>, step: Option<ExprId>, body: StmtId },
    While { cond: ExprId, body: StmtId },
    DoWhile { body: StmtId, cond: ExprId },
    /// cases in [`Unit::cases`]
    Switch { value: ExprId, cases: Range },
    Break,
    Continue,
    Discard,
    Return(Option<ExprId>),
    Empty,
}

/// One `case v:` / `default:` label and the statements up to the next label (a label that
/// directly follows another has an empty body and falls through, as do bodies that do not
/// end in `break`).
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Case {
    /// None: `default`
    pub value: Option<ExprId>,
    /// statements in [`Unit::stmt_lists`]
    pub body: Range,
}

#[derive(Clone, PartialEq, Debug)]
pub enum AttrArg {
    Int(u64),
    Float(f64),
    Str(String),
    Ident(Sym),
}

/// `[numthreads(32, 1, 1)]`, `[WaveSize(32)]`, `[outputtopology("triangle")]`, `[unroll]`
#[derive(Clone, PartialEq, Debug)]
pub struct Attr {
    pub name: Sym,
    pub args: Vec<AttrArg>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Func {
    pub name: Sym,
    pub ret: TypeId,
    /// `float4 main(...) : SV_Target`
    pub ret_semantic: Option<Sym>,
    pub params: Vec<VarDecl>,
    /// statements in [`Unit::stmt_lists`]; None: a prototype
    pub body: Option<Range>,
    pub attrs: Vec<Attr>,
    pub line: u32,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Struct {
    pub name: Sym,
    pub fields: Vec<VarDecl>,
}

/// A top-level declaration, in source order.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Global {
    /// index in [`Unit::vars`]
    Var(u32),
    /// index in [`Unit::funcs`]
    Func(u32),
    /// index in [`Unit::structs`]
    Struct(u32),
}

/// A parsed translation unit.
#[derive(Default, Debug)]
pub struct Unit {
    pub syms: Interner,
    pub types: Vec<TypeRef>,
    pub exprs: Vec<Expr>,
    pub stmts: Vec<Stmt>,
    /// source line of each statement (parallel to `stmts`), for error messages
    pub stmt_lines: Vec<u32>,
    pub expr_lists: Vec<ExprId>,
    pub stmt_lists: Vec<StmtId>,
    pub vars: Vec<VarDecl>,
    pub cases: Vec<Case>,
    pub funcs: Vec<Func>,
    pub structs: Vec<Struct>,
    pub globals: Vec<Global>,
    /// attributes written before a statement (`[unroll]`, `[loop]`, `[branch]`, `[flatten]`,
    /// `[allow_uav_condition]`, ...) by the id of that statement
    pub stmt_attrs: HashMap<u32, Vec<Attr>>,
}

impl Unit {
    pub fn expr(&self, e: ExprId) -> &Expr {
        &self.exprs[e.0 as usize]
    }
    pub fn stmt(&self, s: StmtId) -> &Stmt {
        &self.stmts[s.0 as usize]
    }
    pub fn ty(&self, t: TypeId) -> &TypeRef {
        &self.types[t.0 as usize]
    }
    pub fn name(&self, s: Sym) -> &str {
        self.syms.name(s)
    }
    pub fn args(&self, r: Range) -> &[ExprId] {
        r.of(&self.expr_lists)
    }
    /// The object and the arguments of an [`Expr::Method`] whose list is `all`.
    pub fn method(&self, all: Range) -> (ExprId, Range) {
        (self.expr_lists[all.start as usize], Range { start: all.start + 1, len: all.len - 1 })
    }
    pub fn block(&self, r: Range) -> &[StmtId] {
        r.of(&self.stmt_lists)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interner_numbers_names_in_order_of_first_appearance() {
        let mut i = Interner::default();
        assert!(i.is_empty() && i.get("a").is_none());
        // lengths around the 8 bytes a slot holds inline, and names that share a prefix
        let names = ["v0", "", "a", "abcdefg", "abcdefgh", "abcdefghi", "abcdefghij", "abcdefghijklmnop", "abcdefghijklmnopq", "abcdefgh\0", "t1_stride", "t2_stride"];
        for (n, name) in names.iter().enumerate() {
            assert_eq!(i.intern(name), Sym(n as u32), "{name}");
        }
        for (n, name) in names.iter().enumerate() {
            assert_eq!(i.intern(name), Sym(n as u32));
            assert_eq!(i.get(name), Some(Sym(n as u32)));
            assert_eq!(i.name(Sym(n as u32)), *name);
        }
        assert_eq!(i.len(), names.len());
        assert_eq!(i.get("abcdefghx"), None);
        assert_eq!(i.get("v"), None);
    }

    #[test]
    fn interner_grows() {
        let mut i = Interner::with_capacity(4);
        let name = |n: u32| if n % 3 == 0 { format!("t{n}") } else { format!("a_rather_long_name_{n}") };
        for n in 0..20_000 {
            assert_eq!(i.intern(&name(n)), Sym(n));
        }
        for n in (0..20_000).rev() {
            assert_eq!(i.get(&name(n)), Some(Sym(n)));
            assert_eq!(i.name(Sym(n)), name(n));
        }
        assert_eq!(i.len(), 20_000);
        // the lexer's path: a name read in place, at the very end of the text too
        let text = b"x t123 a_rather_long_name_124 t9";
        assert_eq!(i.intern_at(text, 2, 6), 123);
        assert_eq!(i.intern_at(text, 7, 29), 124);
        assert_eq!(i.intern_at(text, 30, 32), 9);
    }
}
