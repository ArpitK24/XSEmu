//! Tokens of the HLSL dialect, with the preprocessor lines the translator emits (owner: the
//! parser subagent).
//!
//! [`Lexer::next`] hands the parser one token at a time, straight from the source bytes: no
//! token array, no `String` per token, identifiers interned once ([`Interner`]). Keywords and
//! type names are interned before the source is read, so a word's symbol number says what it
//! is ([`Lexer::pre`]).
//!
//! The preprocessor is the subset `gpu::recomp::emit` produces: `#define` of object-like and
//! function-like macros (no `#` / `##`, no variadics, no line continuations), `#undef`,
//! `#ifdef` / `#ifndef` / `#if` / `#elif` / `#else` / `#endif` with C integer expressions and
//! `defined`. Macro uses are replaced by their expansion (arguments expanded first, a macro
//! never inside its own expansion, as in C) and unselected lines are skipped here, so the
//! parser never sees either. Anything else is `Error::Unsupported("preprocessor: ...")`.

use super::ast::{halves, qual, FloatSuffix, Interner, IntSuffix, ResKind, ScalarKind};
use crate::Error;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Tok {
    Eof,
    /// the lexer failed: [`Lexer::take_error`] has the reason, and every later token is `Eof`
    Err,
    /// `val`: the symbol; `aux`: 0, or 1 + the index in [`QUALS`] of a qualifier word
    Ident,
    /// `val`: the value; `aux`: an [`IntSuffix`] ([`int_suffix`])
    Int,
    /// `val`: the bits of the `f64`; `aux`: a [`FloatSuffix`] ([`float_suffix`])
    Float,
    /// `val`: start << 32 | length of the text between the quotes ([`Lexer::str_text`])
    Str,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Semi,
    Comma,
    Dot,
    Colon,
    Question,
    Tilde,
    Bang,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Amp,
    Pipe,
    Caret,
    Lt,
    Gt,
    Shl,
    Shr,
    Le,
    Ge,
    EqEq,
    Ne,
    AndAnd,
    OrOr,
    Assign,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    AmpEq,
    PipeEq,
    CaretEq,
    ShlEq,
    ShrEq,
    PlusPlus,
    MinusMinus,
    ColonColon,
    Arrow,
    /// `#` and `##` exist only on directive lines
    Hash,
    HashHash,
    // words: `val` is the symbol
    KwIf,
    KwElse,
    KwFor,
    KwWhile,
    KwDo,
    KwSwitch,
    KwCase,
    KwDefault,
    KwBreak,
    KwContinue,
    KwDiscard,
    KwReturn,
    KwStruct,
    KwTrue,
    KwFalse,
    KwVoid,
    KwRegister,
    KwPackoffset,
    KwVector,
    KwMatrix,
    /// a scalar, vector or matrix name; `aux`: see [`num_type`]
    TyNum,
    /// `aux`: index in [`RES_KINDS`]
    TyRes,
    /// `aux`: 1 for SamplerComparisonState
    TySampler,
    /// `aux`: 0 point, 1 line, 2 triangle
    TyStream,
    /// `aux`: 1 for OutputPatch
    TyPatch,
    /// a word of HLSL that is not in the dialect (`typedef`, `cbuffer`, `float1x3`, ...);
    /// `aux`: 1 if a declaration may use it as a name
    KwUnsupported,
    /// in a macro body: parameter `val`
    MacroParam,
}

pub const TOK_COUNT: usize = Tok::MacroParam as usize + 1;

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Token {
    pub kind: Tok,
    pub aux: u16,
    /// 1-based source line (of the macro use, for a token from an expansion)
    pub line: u32,
    pub val: u64,
}

pub fn int_suffix(aux: u16) -> IntSuffix {
    match aux {
        1 => IntSuffix::U,
        2 => IntSuffix::L,
        3 => IntSuffix::UL,
        4 => IntSuffix::LL,
        5 => IntSuffix::ULL,
        _ => IntSuffix::None,
    }
}

pub fn float_suffix(aux: u16) -> FloatSuffix {
    match aux {
        1 => FloatSuffix::F,
        2 => FloatSuffix::H,
        3 => FloatSuffix::L,
        _ => FloatSuffix::None,
    }
}

/// The qualifier words, their [`qual`] bits, and whether DXC reserves the word. The others
/// (`sample`, `precise`, ...) are ordinary identifiers to DXC wherever a name is declared
/// with them: see [`Lexer::can_be_declared`].
pub const QUALS: [(&str, u32, bool); 28] = [
    ("static", qual::STATIC, true),
    ("const", qual::CONST, true),
    ("groupshared", qual::GROUPSHARED, true),
    ("precise", qual::PRECISE, false),
    ("globallycoherent", qual::GLOBALLYCOHERENT, false),
    ("in", qual::IN, true),
    ("out", qual::OUT, true),
    ("inout", qual::INOUT, true),
    ("nointerpolation", qual::NOINTERPOLATION, true),
    ("uniform", qual::UNIFORM, true),
    ("row_major", qual::ROW_MAJOR, true),
    ("column_major", qual::COLUMN_MAJOR, true),
    ("volatile", qual::VOLATILE, true),
    ("extern", qual::EXTERN, true),
    ("inline", qual::INLINE, true),
    ("noperspective", qual::NOPERSPECTIVE, true),
    ("linear", qual::LINEAR, true),
    ("centroid", qual::CENTROID, true),
    ("sample", qual::SAMPLE, false),
    ("vertices", qual::VERTICES, false),
    ("indices", qual::INDICES, false),
    ("primitives", qual::PRIMITIVES, false),
    ("payload", qual::PAYLOAD, false),
    ("point", qual::GS_POINT, true),
    ("line", qual::GS_LINE, true),
    ("triangle", qual::GS_TRIANGLE, true),
    ("lineadj", qual::GS_LINEADJ, true),
    ("triangleadj", qual::GS_TRIANGLEADJ, true),
];

const KEYWORDS: [(&str, Tok); 20] = [
    ("if", Tok::KwIf),
    ("else", Tok::KwElse),
    ("for", Tok::KwFor),
    ("while", Tok::KwWhile),
    ("do", Tok::KwDo),
    ("switch", Tok::KwSwitch),
    ("case", Tok::KwCase),
    ("default", Tok::KwDefault),
    ("break", Tok::KwBreak),
    ("continue", Tok::KwContinue),
    ("discard", Tok::KwDiscard),
    ("return", Tok::KwReturn),
    ("struct", Tok::KwStruct),
    ("true", Tok::KwTrue),
    ("false", Tok::KwFalse),
    ("void", Tok::KwVoid),
    ("register", Tok::KwRegister),
    ("packoffset", Tok::KwPackoffset),
    ("vector", Tok::KwVector),
    ("matrix", Tok::KwMatrix),
];

/// Words of HLSL (DXC) that the translator never emits, and whether DXC lets a declaration
/// use the word as a name (the type names; the others are reserved).
const UNSUPPORTED_WORDS: [(&str, bool); 29] = [
    ("typedef", false),
    ("cbuffer", false),
    ("tbuffer", false),
    ("namespace", false),
    ("class", false),
    ("interface", false),
    ("template", false),
    ("typename", false),
    ("using", false),
    ("goto", false),
    ("enum", false),
    ("union", false),
    ("unsigned", false),
    ("signed", false),
    ("snorm", false),
    ("unorm", false),
    ("sizeof", false),
    ("operator", false),
    ("this", false),
    ("export", false),
    ("auto", false),
    ("technique", false),
    ("shared", false),
    ("asm", false),
    ("min10float", true),
    ("min12int", true),
    ("string", true),
    ("pass", true),
    ("compile", true),
];

/// Scalar type names; several are synonyms of one [`ScalarKind`].
const SCALARS: [(&str, ScalarKind); 19] = [
    ("bool", ScalarKind::Bool),
    ("int", ScalarKind::Int),
    ("uint", ScalarKind::Uint),
    ("dword", ScalarKind::Uint),
    ("half", ScalarKind::Half),
    ("float", ScalarKind::Float),
    ("double", ScalarKind::Double),
    ("min16float", ScalarKind::Min16Float),
    ("min16int", ScalarKind::Min16Int),
    ("min16uint", ScalarKind::Min16Uint),
    ("int16_t", ScalarKind::Int16),
    ("uint16_t", ScalarKind::Uint16),
    ("float16_t", ScalarKind::Float16),
    ("int32_t", ScalarKind::Int),
    ("uint32_t", ScalarKind::Uint),
    ("float32_t", ScalarKind::Float),
    ("int64_t", ScalarKind::Int64),
    ("uint64_t", ScalarKind::Uint64),
    ("float64_t", ScalarKind::Double),
];

/// Shapes a numeric type name can have: scalar, 3 vectors, 12 matrices with 2 or more rows.
const SHAPES: usize = 16;
pub const NUM_TYPES: usize = SCALARS.len() * SHAPES;

/// The (kind, rows, cols) of a [`Tok::TyNum`] token's `aux`.
pub fn num_type(aux: u16) -> (ScalarKind, u8, u8) {
    let kind = SCALARS[aux as usize / SHAPES].1;
    let shape = (aux as usize % SHAPES) as u8;
    match shape {
        0 => (kind, 1, 1),
        1..=3 => (kind, 1, shape + 1),
        _ => (kind, 2 + (shape - 4) / 4, 1 + (shape - 4) % 4),
    }
}

/// The `aux` of the scalar type a [`Tok::TyNum`] token's type is made of.
pub fn num_scalar(aux: u16) -> u16 {
    aux - aux % SHAPES as u16
}

/// The `aux` of the numeric type with the scalar of `aux` and this shape, if the tree can
/// say it (see the module documentation of [`super::ast`] on one-row matrices).
pub fn num_shaped(aux: u16, rows: u64, cols: u64) -> Option<u16> {
    let base = num_scalar(aux);
    match (rows, cols) {
        (1, 1) => Some(base),
        (1, 2..=4) => Some(base + cols as u16 - 1),
        (2..=4, 1..=4) => Some(base + 4 + (rows as u16 - 2) * 4 + cols as u16 - 1),
        _ => None,
    }
}

pub const RES_KINDS: [(&str, ResKind); 24] = [
    ("Texture1D", ResKind::Texture1D),
    ("Texture1DArray", ResKind::Texture1DArray),
    ("Texture2D", ResKind::Texture2D),
    ("Texture2DArray", ResKind::Texture2DArray),
    ("Texture2DMS", ResKind::Texture2DMS),
    ("Texture2DMSArray", ResKind::Texture2DMSArray),
    ("Texture3D", ResKind::Texture3D),
    ("TextureCube", ResKind::TextureCube),
    ("TextureCubeArray", ResKind::TextureCubeArray),
    ("RWTexture1D", ResKind::RWTexture1D),
    ("RWTexture1DArray", ResKind::RWTexture1DArray),
    ("RWTexture2D", ResKind::RWTexture2D),
    ("RWTexture2DArray", ResKind::RWTexture2DArray),
    ("RWTexture3D", ResKind::RWTexture3D),
    ("Buffer", ResKind::Buffer),
    ("RWBuffer", ResKind::RWBuffer),
    ("ByteAddressBuffer", ResKind::ByteAddressBuffer),
    ("RWByteAddressBuffer", ResKind::RWByteAddressBuffer),
    ("StructuredBuffer", ResKind::StructuredBuffer),
    ("RWStructuredBuffer", ResKind::RWStructuredBuffer),
    ("AppendStructuredBuffer", ResKind::AppendStructuredBuffer),
    ("ConsumeStructuredBuffer", ResKind::ConsumeStructuredBuffer),
    ("ConstantBuffer", ResKind::ConstantBuffer),
    ("RaytracingAccelerationStructure", ResKind::RaytracingAccelerationStructure),
];

const OTHER_TYPES: [(&str, Tok, u16); 7] = [
    ("SamplerState", Tok::TySampler, 0),
    ("SamplerComparisonState", Tok::TySampler, 1),
    ("PointStream", Tok::TyStream, 0),
    ("LineStream", Tok::TyStream, 1),
    ("TriangleStream", Tok::TyStream, 2),
    ("InputPatch", Tok::TyPatch, 0),
    ("OutputPatch", Tok::TyPatch, 1),
];

// The classes of bytes: one bit each, so that the hot path tests them one by one in the order
// of their frequency (a `match` on them becomes a jump table, which costs more than the two
// instructions of a test).
const C_OTHER: u8 = 0;
const C_WS: u8 = 1;
const C_NL: u8 = 2;
/// a token of its own wherever it stands: [`ONE`] has its kind
const C_ONE: u8 = 4;
const C_ID: u8 = 8;
const C_DIGIT: u8 = 16;
/// the first byte of the operators in [`OPS`]
const C_OP: u8 = 32;

const fn class_table() -> [u8; 256] {
    let mut t = [C_OTHER; 256];
    let mut i = 0;
    while i < 256 {
        let b = i as u8;
        t[i] = if b == b' ' || b == b'\t' || b == b'\r' || b == 0x0B || b == 0x0C {
            C_WS
        } else if b == b'\n' {
            C_NL
        } else if !matches!(one_table()[i], Tok::Eof) {
            C_ONE
        } else if !matches!(ops_table()[i].0, Tok::Eof) {
            C_OP
        } else if b.is_ascii_alphabetic() || b == b'_' {
            C_ID
        } else if b.is_ascii_digit() {
            C_DIGIT
        } else {
            C_OTHER
        };
        i += 1;
    }
    t
}

static CLASS: [u8; 256] = class_table();

const fn one_table() -> [Tok; 256] {
    let mut t = [Tok::Eof; 256];
    t[b'(' as usize] = Tok::LParen;
    t[b')' as usize] = Tok::RParen;
    t[b'{' as usize] = Tok::LBrace;
    t[b'}' as usize] = Tok::RBrace;
    t[b'[' as usize] = Tok::LBracket;
    t[b']' as usize] = Tok::RBracket;
    t[b';' as usize] = Tok::Semi;
    t[b',' as usize] = Tok::Comma;
    t[b'?' as usize] = Tok::Question;
    t[b'~' as usize] = Tok::Tilde;
    t
}

/// The token a [`C_ONE`] byte is.
static ONE: [Tok; 256] = one_table();

const fn ops_table() -> [(Tok, Tok, Tok); 256] {
    let mut t = [(Tok::Eof, Tok::Eof, Tok::Eof); 256];
    t[b'=' as usize] = (Tok::Assign, Tok::EqEq, Tok::Eof);
    t[b'!' as usize] = (Tok::Bang, Tok::Ne, Tok::Eof);
    t[b'+' as usize] = (Tok::Plus, Tok::PlusEq, Tok::PlusPlus);
    t[b'-' as usize] = (Tok::Minus, Tok::MinusEq, Tok::MinusMinus);
    t[b'*' as usize] = (Tok::Star, Tok::StarEq, Tok::Eof);
    t[b'%' as usize] = (Tok::Percent, Tok::PercentEq, Tok::Eof);
    t[b'^' as usize] = (Tok::Caret, Tok::CaretEq, Tok::Eof);
    t[b'&' as usize] = (Tok::Amp, Tok::AmpEq, Tok::AndAnd);
    t[b'|' as usize] = (Tok::Pipe, Tok::PipeEq, Tok::OrOr);
    t[b'<' as usize] = (Tok::Lt, Tok::Le, Tok::Shl);
    t[b'>' as usize] = (Tok::Gt, Tok::Ge, Tok::Shr);
    t[b'/' as usize] = (Tok::Slash, Tok::SlashEq, Tok::Err);
    t[b':' as usize] = (Tok::Colon, Tok::Eof, Tok::ColonColon);
    t[b'.' as usize] = (Tok::Dot, Tok::Eof, Tok::Eof);
    t
}

/// By the first byte of an operator: the token it is alone, with `=` after it, and twice
/// (`Eof`: that is not an operator; `Err`: it is something [`Lexer::operator`] leaves to
/// [`Lexer::scan_rest`]).
static OPS: [(Tok, Tok, Tok); 256] = ops_table();

#[inline]
fn is_word(b: u8) -> bool {
    CLASS[b as usize] & (C_ID | C_DIGIT) != 0
}

/// Index of the first `\n` at or after `from`, or the length: eight bytes a step.
fn find_newline(s: &[u8], from: usize) -> usize {
    let mut p = from;
    while let Some(c) = s.get(p..p + 8) {
        let mut a = [0u8; 8];
        a.copy_from_slice(c);
        let w = u64::from_le_bytes(a) ^ 0x0A0A_0A0A_0A0A_0A0A;
        let zero = w.wrapping_sub(0x0101_0101_0101_0101) & !w & 0x8080_8080_8080_8080;
        if zero != 0 {
            return p + (zero.trailing_zeros() / 8) as usize;
        }
        p += 8;
    }
    while p < s.len() && s[p] != b'\n' {
        p += 1;
    }
    p
}

struct Macro {
    name: u32,
    /// None: object-like
    params: Option<u32>,
    body: Vec<Token>,
    /// being expanded: its name is not a macro inside its own expansion
    active: bool,
}

/// One open `#if` group.
struct Cond {
    /// a branch of the group has been selected
    taken: bool,
    else_seen: bool,
}

const MAX_MACRO_DEPTH: u32 = 64;

pub struct Lexer<'a> {
    full: &'a [u8],
    /// `full`, or its part up to the end of the directive line being read
    src: &'a [u8],
    pos: usize,
    line: u32,
    /// what each pre-interned symbol is: its token kind and `aux`
    pre: Vec<(Tok, u16)>,
    /// 1 + index in `macros` by symbol, 0: not a macro (as long as the largest macro symbol)
    macro_of: Vec<u32>,
    macros: Vec<Macro>,
    /// the expansion being handed out
    pending: Vec<Token>,
    pending_pos: usize,
    conds: Vec<Cond>,
    in_directive: bool,
    /// reading macro arguments or a directive line: identifiers are not expanded
    no_expand: bool,
    err: Option<Error>,
    /// the token [`Lexer::advance`] read last
    pub tok: Token,
}

impl<'a> Lexer<'a> {
    /// `syms` must be empty: the keyword and type names take the first symbols.
    pub fn new(src: &'a str, syms: &mut Interner) -> Self {
        let mut pre = Vec::with_capacity(KEYWORDS.len() + UNSUPPORTED_WORDS.len() + QUALS.len() + SCALARS.len() * 21 + 40);
        let mut add = |syms: &mut Interner, name: &str, kind: Tok, aux: u16| {
            let s = syms.intern(name);
            if s.0 as usize == pre.len() {
                pre.push((kind, aux));
            }
        };
        for (name, kind) in KEYWORDS {
            add(syms, name, kind, 0);
        }
        for (name, free) in UNSUPPORTED_WORDS {
            add(syms, name, Tok::KwUnsupported, free as u16);
        }
        for (i, (name, _, _)) in QUALS.iter().enumerate() {
            add(syms, name, Tok::Ident, i as u16 + 1);
        }
        let mut name = String::with_capacity(16);
        for (i, (scalar, _)) in SCALARS.iter().enumerate() {
            let base = (i * SHAPES) as u16;
            add(syms, scalar, Tok::TyNum, base);
            for rows in 0..=4u16 {
                for cols in 1..=4u16 {
                    name.clear();
                    name.push_str(scalar);
                    if rows != 0 {
                        name.push((b'0' + rows as u8) as char);
                        name.push('x');
                    }
                    name.push((b'0' + cols as u8) as char);
                    let (kind, aux) = match (rows, cols) {
                        // `float1` and `float1xN`: see the module documentation of `ast`
                        (0, 1) | (1, _) => (Tok::KwUnsupported, 1),
                        (0, _) => (Tok::TyNum, base + cols - 1),
                        _ => (Tok::TyNum, base + 4 + (rows - 2) * 4 + cols - 1),
                    };
                    add(syms, &name, kind, aux);
                }
            }
        }
        for (i, (name, _)) in RES_KINDS.iter().enumerate() {
            add(syms, name, Tok::TyRes, i as u16);
        }
        for (name, kind, aux) in OTHER_TYPES {
            add(syms, name, kind, aux);
        }
        let full = src.as_bytes();
        // a UTF-8 byte order mark is not part of the text
        let pos = if full.starts_with(&[0xEF, 0xBB, 0xBF]) { 3 } else { 0 };
        Lexer {
            full,
            src: full,
            pos,
            line: 1,
            pre,
            macro_of: Vec::new(),
            macros: Vec::new(),
            pending: Vec::new(),
            pending_pos: 0,
            conds: Vec::new(),
            in_directive: false,
            no_expand: false,
            err: None,
            tok: Token { kind: Tok::Eof, aux: 0, line: 1, val: 0 },
        }
    }

    /// Whether `sym` is one of the words interned before the source (a keyword, a type name,
    /// a qualifier word, or a reserved word outside the dialect).
    pub fn pre(&self, sym: u32) -> Option<(Tok, u16)> {
        self.pre.get(sym as usize).copied()
    }

    /// Whether DXC lets a declaration name something with this word although it means
    /// something else until then: the type names other than the C keywords `bool`, `int`,
    /// `float`, `double` (to DXC they are predeclared identifiers: the translator's prelude
    /// has a local named `half`) and the qualifier words DXC does not reserve.
    pub fn can_be_declared(t: Token) -> bool {
        match t.kind {
            Tok::Ident => t.aux == 0 || !QUALS[t.aux as usize - 1].2,
            Tok::TyNum => t.aux % SHAPES as u16 != 0 || !matches!(SCALARS[t.aux as usize / SHAPES].0, "bool" | "int" | "float" | "double"),
            Tok::TyRes | Tok::TySampler | Tok::TyStream | Tok::TyPatch | Tok::KwVector | Tok::KwMatrix => true,
            Tok::KwUnsupported => t.aux != 0,
            _ => false,
        }
    }

    pub fn take_error(&mut self) -> Option<Error> {
        self.err.take()
    }

    pub fn has_error(&self) -> bool {
        self.err.is_some()
    }

    /// Records the first error; the lexer then only returns `Eof`.
    pub fn set_error(&mut self, e: Error) {
        if self.err.is_none() {
            self.err = Some(e);
        }
        self.pos = self.src.len();
        self.pending.clear();
        self.pending_pos = 0;
    }

    /// The text of a [`Tok::Str`] token.
    pub fn str_text(&self, t: Token) -> &'a str {
        let at = (t.val >> 32) as usize;
        let len = (t.val & 0xFFFF_FFFF) as usize;
        self.full.get(at..at + len).and_then(|b| std::str::from_utf8(b).ok()).unwrap_or("")
    }

    #[inline]
    pub fn next(&mut self, syms: &mut Interner) -> Token {
        self.advance(syms);
        self.tok
    }

    /// Reads the next token into [`Lexer::tok`].
    #[inline]
    pub fn advance(&mut self, syms: &mut Interner) {
        if self.pending_pos < self.pending.len() {
            self.tok = self.pending[self.pending_pos];
            self.pending_pos += 1;
            return;
        }
        self.scan_into(syms)
    }

    /// The next token of the source text (not of a pending expansion).
    fn scan(&mut self, syms: &mut Interner) -> Token {
        self.scan_into(syms);
        self.tok
    }

    #[cold]
    fn fail(&mut self, line: u32, e: Error) -> Token {
        self.set_error(e);
        Token { kind: Tok::Err, aux: 0, line, val: 0 }
    }

    #[cold]
    fn source(&mut self, line: u32, msg: &str) -> Token {
        self.fail(line, Error::Source { line, msg: msg.to_string() })
    }

    #[cold]
    fn unsupported(&mut self, line: u32, what: &str) -> Token {
        self.fail(line, Error::Unsupported(what.to_string()))
    }

    #[cold]
    fn at_eof(&mut self) -> Token {
        if !self.in_directive && !self.conds.is_empty() && self.err.is_none() {
            let line = self.line;
            return self.source(line, "#if without #endif");
        }
        Token { kind: Tok::Eof, aux: 0, line: self.line, val: 0 }
    }

    /// [`Lexer::scan`] into [`Lexer::tok`], fast.
    ///
    /// This function and the four it ends in ([`Lexer::known_word`], [`Lexer::newline`],
    /// [`Lexer::operator`], [`Lexer::small_int`]) read nearly every token of a translated
    /// program: blanks, the one-byte punctuation, words that have been seen before, operators
    /// and decimal integers. They are written to need no stack frame: few values at a time,
    /// and no call but to each other and to [`Lexer::scan_slow`] as their last statement. What
    /// they are not sure of they leave to [`Lexer::scan_any`], which reads everything, from
    /// where they stand.
    #[inline(never)]
    fn scan_into(&mut self, syms: &mut Interner) {
        let s = self.src;
        let mut p = self.pos;
        while let Some(&b) = s.get(p) {
            let class = CLASS[b as usize];
            if class & C_ONE != 0 {
                self.pos = p + 1;
                self.tok = Token { kind: ONE[b as usize], aux: 0, line: self.line, val: 0 };
                return;
            }
            if class & C_WS != 0 {
                p += 1;
                continue;
            }
            self.pos = p;
            if class & C_ID != 0 {
                // a word that ends in the 16 bytes at its start is looked up by them
                let Some(head) = s.get(p..).and_then(|w| w.first_chunk::<16>()) else { break };
                let mut n = 1;
                while n < 16 && is_word(head[n]) {
                    n += 1;
                }
                if n == 16 {
                    break;
                }
                let (lo, hi) = halves(head);
                return self.known_word(syms, lo, hi, n);
            }
            if class & C_NL != 0 {
                return self.newline(syms);
            }
            if class & C_OP != 0 {
                return self.operator(syms);
            }
            if class & C_DIGIT != 0 {
                return self.small_int(syms);
            }
            break;
        }
        self.pos = p;
        self.scan_slow(syms)
    }

    /// The word of `n` bytes (1 to 15) at the current position, where the 16 bytes `lo` and
    /// `hi` are.
    #[inline(never)]
    fn known_word(&mut self, syms: &mut Interner, lo: u64, hi: u64, n: usize) {
        let Some(sym) = syms.find_short((lo, hi), n) else {
            return self.scan_slow(syms);
        };
        let (mut kind, mut aux) = (Tok::Ident, 0);
        if let Some(&pre) = self.pre.get(sym as usize) {
            (kind, aux) = pre;
        }
        if kind == Tok::Ident && self.macro_of.get(sym as usize).is_some_and(|&m| m != 0) && !self.no_expand {
            return self.scan_slow(syms);
        }
        self.pos += n;
        self.tok = Token { kind, aux, line: self.line, val: sym as u64 };
    }

    /// The token after the line end at the current position.
    #[inline(never)]
    fn newline(&mut self, syms: &mut Interner) {
        let s = self.src;
        let mut p = self.pos + 1;
        self.line += 1;
        // indentation, eight spaces a step
        while let Some(c) = s.get(p..p + 8) {
            if c != b"        " {
                break;
            }
            p += 8;
        }
        if s.get(p..p + 4).is_some_and(|c| c == b"    ") {
            p += 4;
        }
        self.pos = p;
        self.scan_into(syms)
    }

    /// The operator at the current position, the first byte of which is a [`C_OP`] byte.
    #[inline(never)]
    fn operator(&mut self, syms: &mut Interner) {
        let p = self.pos;
        let Some(&[b, n, after]) = self.src.get(p..).and_then(|w| w.first_chunk::<3>()) else {
            return self.scan_slow(syms);
        };
        let (one, with_eq, twice) = OPS[b as usize];
        let (mut kind, mut len) = (one, 1);
        if n == b'=' {
            if with_eq != Tok::Eof {
                (kind, len) = (with_eq, 2);
            }
        } else if n == b {
            // a comment, or `<<=` or `>>=`
            if twice == Tok::Err || (after == b'=' && (b == b'<' || b == b'>')) {
                return self.scan_slow(syms);
            }
            if twice != Tok::Eof {
                (kind, len) = (twice, 2);
            }
        } else if (b == b'-' && n == b'>') || (b == b'/' && n == b'*') || (b == b'.' && n.is_ascii_digit()) {
            return self.scan_slow(syms);
        }
        self.pos = p + len;
        self.tok = Token { kind, aux: 0, line: self.line, val: 0 };
    }

    /// The number at the current position if it is a decimal integer with no suffix or `u`.
    #[inline(never)]
    fn small_int(&mut self, syms: &mut Interner) {
        let s = self.src;
        let start = self.pos;
        let mut p = start;
        let mut v = 0u64;
        while let Some(&c) = s.get(p) {
            if !c.is_ascii_digit() {
                break;
            }
            v = v.wrapping_mul(10).wrapping_add((c - b'0') as u64);
            p += 1;
        }
        let digits = p - start;
        // not octal, and 19 digits fit 64 bits whatever they are
        if digits <= 19 && (digits == 1 || s.get(start) != Some(&b'0')) {
            let word_at = |q: usize| matches!(s.get(q), Some(&c) if is_word(c));
            let c = s.get(p).copied().unwrap_or(b' ');
            let unsigned = matches!(c, b'u' | b'U') && !word_at(p + 1);
            if unsigned || !(is_word(c) || c == b'.') {
                self.pos = p + unsigned as usize;
                self.tok = Token { kind: Tok::Int, aux: unsigned as u16, line: self.line, val: v };
                return;
            }
        }
        self.scan_slow(syms)
    }

    #[inline(never)]
    fn scan_slow(&mut self, syms: &mut Interner) {
        self.tok = self.scan_any(syms);
    }

    /// The next token of the source text, whatever it is.
    #[inline(never)]
    fn scan_any(&mut self, syms: &mut Interner) -> Token {
        'again: loop {
            let s = self.src;
            let mut p = self.pos;
            loop {
                let Some(&b) = s.get(p) else {
                    self.pos = p;
                    return self.at_eof();
                };
                let class = CLASS[b as usize];
                if class == C_WS {
                    p += 1;
                    continue;
                }
                if class == C_ONE {
                    self.pos = p + 1;
                    return Token { kind: ONE[b as usize], aux: 0, line: self.line, val: 0 };
                }
                if class == C_ID {
                    let line = self.line;
                    let sym = self.word(p, syms);
                    let mut aux = 0;
                    if let Some(&(kind, a)) = self.pre.get(sym as usize) {
                        if kind != Tok::Ident {
                            return Token { kind, aux: a, line, val: sym as u64 };
                        }
                        aux = a;
                    }
                    if let Some(&m) = self.macro_of.get(sym as usize) {
                        if m != 0 && !self.no_expand {
                            match self.expand_use(m as usize - 1, line, syms) {
                                Some(t) => return t,
                                // an empty expansion
                                None => continue 'again,
                            }
                        }
                    }
                    return Token { kind: Tok::Ident, aux, line, val: sym as u64 };
                }
                if class == C_NL {
                    p += 1;
                    self.line += 1;
                    // indentation, eight spaces a step
                    while let Some(c) = s.get(p..p + 8) {
                        if c != b"        " {
                            break;
                        }
                        p += 8;
                    }
                    if s.get(p..p + 4).is_some_and(|c| c == b"    ") {
                        p += 4;
                    }
                    continue;
                }
                if class == C_DIGIT {
                    return self.number(p, self.line);
                }
                match self.scan_rest(p, b, syms) {
                    Some(t) => return t,
                    // a comment or a directive
                    None => continue 'again,
                }
            }
        }
    }

    /// The symbol of the word at `start`.
    fn word(&mut self, start: usize, syms: &mut Interner) -> u32 {
        let s = self.src;
        let mut p = start + 1;
        while p < s.len() && is_word(s[p]) {
            p += 1;
        }
        self.pos = p;
        syms.intern_at(s, start, p)
    }

    /// The token at `p`, whose first byte `b` is not a blank, a [`C_ONE`] byte, or the start of
    /// a word or a number; None after a comment or a directive: the next token is to be read.
    #[inline(never)]
    fn scan_rest(&mut self, p: usize, b: u8, syms: &mut Interner) -> Option<Token> {
        let s = self.src;
        let line = self.line;
        let n = s.get(p + 1).copied().unwrap_or(0);
        let (kind, len) = match b {
            b'.' => {
                if n.is_ascii_digit() {
                    return Some(self.number(p, line));
                }
                (Tok::Dot, 1)
            }
            b':' => {
                if n == b':' {
                    (Tok::ColonColon, 2)
                } else {
                    (Tok::Colon, 1)
                }
            }
            b'=' => {
                if n == b'=' {
                    (Tok::EqEq, 2)
                } else {
                    (Tok::Assign, 1)
                }
            }
            b'!' => {
                if n == b'=' {
                    (Tok::Ne, 2)
                } else {
                    (Tok::Bang, 1)
                }
            }
            b'+' => match n {
                b'+' => (Tok::PlusPlus, 2),
                b'=' => (Tok::PlusEq, 2),
                _ => (Tok::Plus, 1),
            },
            b'-' => match n {
                b'-' => (Tok::MinusMinus, 2),
                b'=' => (Tok::MinusEq, 2),
                b'>' => (Tok::Arrow, 2),
                _ => (Tok::Minus, 1),
            },
            b'*' => {
                if n == b'=' {
                    (Tok::StarEq, 2)
                } else {
                    (Tok::Star, 1)
                }
            }
            b'%' => {
                if n == b'=' {
                    (Tok::PercentEq, 2)
                } else {
                    (Tok::Percent, 1)
                }
            }
            b'^' => {
                if n == b'=' {
                    (Tok::CaretEq, 2)
                } else {
                    (Tok::Caret, 1)
                }
            }
            b'&' => match n {
                b'&' => (Tok::AndAnd, 2),
                b'=' => (Tok::AmpEq, 2),
                _ => (Tok::Amp, 1),
            },
            b'|' => match n {
                b'|' => (Tok::OrOr, 2),
                b'=' => (Tok::PipeEq, 2),
                _ => (Tok::Pipe, 1),
            },
            b'<' => match n {
                b'<' => {
                    if s.get(p + 2) == Some(&b'=') {
                        (Tok::ShlEq, 3)
                    } else {
                        (Tok::Shl, 2)
                    }
                }
                b'=' => (Tok::Le, 2),
                _ => (Tok::Lt, 1),
            },
            b'>' => match n {
                b'>' => {
                    if s.get(p + 2) == Some(&b'=') {
                        (Tok::ShrEq, 3)
                    } else {
                        (Tok::Shr, 2)
                    }
                }
                b'=' => (Tok::Ge, 2),
                _ => (Tok::Gt, 1),
            },
            b'/' => match n {
                b'/' => {
                    self.pos = find_newline(s, p + 2);
                    return None;
                }
                b'*' => {
                    let mut q = p + 2;
                    loop {
                        match s.get(q) {
                            None => {
                                return Some(if self.in_directive {
                                    self.unsupported(line, "preprocessor: comment spanning lines in a directive")
                                } else {
                                    self.source(line, "comment without */")
                                });
                            }
                            Some(b'*') if s.get(q + 1) == Some(&b'/') => break,
                            Some(b'\n') => self.line += 1,
                            _ => {}
                        }
                        q += 1;
                    }
                    self.pos = q + 2;
                    return None;
                }
                b'=' => (Tok::SlashEq, 2),
                _ => (Tok::Slash, 1),
            },
            b'"' => {
                let start = p + 1;
                let mut q = start;
                loop {
                    match s.get(q) {
                        None | Some(b'\n') => return Some(self.source(line, "string without closing quote")),
                        Some(b'\\') => return Some(self.unsupported(line, "escape in a string literal")),
                        Some(b'"') => break,
                        _ => q += 1,
                    }
                }
                self.pos = q + 1;
                return Some(Token { kind: Tok::Str, aux: 0, line, val: (start as u64) << 32 | (q - start) as u64 });
            }
            b'#' => {
                if self.in_directive {
                    if n == b'#' {
                        (Tok::HashHash, 2)
                    } else {
                        (Tok::Hash, 1)
                    }
                } else {
                    self.pos = p;
                    if self.directive(syms).is_none() {
                        return Some(Token { kind: Tok::Err, aux: 0, line, val: 0 });
                    }
                    return None;
                }
            }
            _ => {
                let msg = if b.is_ascii_graphic() { format!("unexpected character '{}'", b as char) } else { format!("unexpected byte 0x{b:02X}") };
                return Some(self.source(line, &msg));
            }
        };
        self.pos = p + len;
        Some(Token { kind, aux: 0, line, val: 0 })
    }

    /// A numeric literal at `start` (a digit, or `.` before a digit).
    fn number(&mut self, start: usize, line: u32) -> Token {
        let s = self.src;
        let mut p = start;
        if s[p] == b'0' && matches!(s.get(p + 1), Some(b'x' | b'X')) {
            p += 2;
            let digits = p;
            let mut v = 0u64;
            while let Some(&c) = s.get(p) {
                let d = match c {
                    b'0'..=b'9' => c - b'0',
                    b'a'..=b'f' => c - b'a' + 10,
                    b'A'..=b'F' => c - b'A' + 10,
                    _ => break,
                };
                if v >> 60 != 0 {
                    return self.source(line, "integer literal does not fit 64 bits");
                }
                v = v << 4 | d as u64;
                p += 1;
            }
            if p == digits {
                return self.source(line, "hexadecimal literal without digits");
            }
            return self.int_token(v, p, line);
        }
        let mut v = 0u64;
        while let Some(&c) = s.get(p) {
            if !c.is_ascii_digit() {
                break;
            }
            v = v.wrapping_mul(10).wrapping_add((c - b'0') as u64);
            p += 1;
        }
        // 19 digits fit 64 bits whatever they are; more may not
        let mut fits = true;
        if p - start > 19 {
            v = 0;
            for &c in &s[start..p] {
                match v.checked_mul(10).and_then(|x| x.checked_add((c - b'0') as u64)) {
                    Some(x) => v = x,
                    None => fits = false,
                }
            }
        }
        let int_end = p;
        let mut float = false;
        if s.get(p) == Some(&b'.') {
            let n = s.get(p + 1).copied().unwrap_or(0);
            if CLASS[n as usize] == C_ID && !matches!(n, b'e' | b'E' | b'f' | b'F' | b'h' | b'H' | b'l' | b'L') {
                return self.unsupported(line, "member access on a numeric literal");
            }
            float = true;
            p += 1;
            while matches!(s.get(p), Some(b'0'..=b'9')) {
                p += 1;
            }
        }
        if matches!(s.get(p), Some(b'e' | b'E')) {
            let mut q = p + 1;
            if matches!(s.get(q), Some(b'+' | b'-')) {
                q += 1;
            }
            if !matches!(s.get(q), Some(b'0'..=b'9')) {
                return self.source(line, "exponent without digits in a numeric literal");
            }
            while matches!(s.get(q), Some(b'0'..=b'9')) {
                q += 1;
            }
            p = q;
            float = true;
        }
        if !float {
            if int_end - start > 1 && s[start] == b'0' {
                // octal, as in C (and DXC)
                v = 0;
                for &c in &s[start..int_end] {
                    if c > b'7' {
                        return self.source(line, "digit 8 or 9 in an octal literal");
                    }
                    if v >> 61 != 0 {
                        return self.source(line, "integer literal does not fit 64 bits");
                    }
                    v = v << 3 | (c - b'0') as u64;
                }
            } else if !fits {
                return self.source(line, "integer literal does not fit 64 bits");
            }
            return self.int_token(v, p, line);
        }
        let text = std::str::from_utf8(&s[start..p]).unwrap_or("");
        let suffix = match s.get(p) {
            Some(b'f' | b'F') => 1,
            Some(b'h' | b'H') => 2,
            Some(b'l' | b'L') => 3,
            _ => 0,
        };
        if suffix != 0 {
            p += 1;
        }
        if matches!(s.get(p), Some(&c) if is_word(c)) {
            return self.source(line, "invalid suffix on a floating-point literal");
        }
        // `f` / `h`: clang converts the text with float semantics, one rounding; the others
        // are doubles. Rust's parsers round to nearest even, as APFloat does.
        let value = if suffix == 1 || suffix == 2 { text.parse::<f32>().map(|v| v as f64) } else { text.parse::<f64>() };
        let Ok(value) = value else {
            return self.source(line, "malformed floating-point literal");
        };
        self.pos = p;
        Token { kind: Tok::Float, aux: suffix, line, val: value.to_bits() }
    }

    fn int_token(&mut self, v: u64, mut p: usize, line: u32) -> Token {
        let s = self.src;
        // no suffix, or `u`: nearly every literal
        let word_at = |q: usize| matches!(s.get(q), Some(&c) if is_word(c));
        if !word_at(p) {
            self.pos = p;
            return Token { kind: Tok::Int, aux: 0, line, val: v };
        }
        if matches!(s[p], b'u' | b'U') && !word_at(p + 1) {
            self.pos = p + 1;
            return Token { kind: Tok::Int, aux: 1, line, val: v };
        }
        let start = p;
        while matches!(s.get(p), Some(b'u' | b'U' | b'l' | b'L')) && p - start < 4 {
            p += 1;
        }
        let mut letters = [0u8; 4];
        for (d, c) in letters.iter_mut().zip(&s[start..p]) {
            *d = c.to_ascii_lowercase();
        }
        let aux = match &letters[..p - start] {
            b"" => 0,
            b"u" => 1,
            b"l" => 2,
            b"ul" | b"lu" => 3,
            b"ll" => 4,
            b"ull" | b"llu" => 5,
            _ => return self.source(line, "invalid suffix on an integer literal"),
        };
        if matches!(s.get(p), Some(&c) if is_word(c)) {
            return self.source(line, "invalid suffix on an integer literal");
        }
        self.pos = p;
        Token { kind: Tok::Int, aux, line, val: v }
    }

    fn is_macro(&self, sym: u64) -> Option<usize> {
        match self.macro_of.get(sym as usize) {
            Some(&m) if m != 0 => Some(m as usize - 1),
            _ => None,
        }
    }

    /// A macro name was read from the source: reads its arguments, expands it into
    /// `pending` and returns the first token (None: the expansion is empty).
    #[cold]
    fn expand_use(&mut self, mi: usize, line: u32, syms: &mut Interner) -> Option<Token> {
        let err = Token { kind: Tok::Err, aux: 0, line, val: 0 };
        let mut args = Vec::new();
        if let Some(params) = self.macros[mi].params {
            self.no_expand = true;
            let open = self.scan(syms);
            if open.kind != Tok::LParen {
                self.no_expand = false;
                if open.kind != Tok::Err {
                    let what = format!("preprocessor: function-like macro {} used without arguments", syms.name(super::ast::Sym(self.macros[mi].name)));
                    self.unsupported(line, &what);
                }
                return Some(err);
            }
            let mut depth = 0u32;
            let mut arg = Vec::new();
            loop {
                let t = self.scan(syms);
                match t.kind {
                    Tok::Err => {
                        self.no_expand = false;
                        return Some(err);
                    }
                    Tok::Eof => {
                        self.no_expand = false;
                        self.source(line, "macro arguments without closing parenthesis");
                        return Some(err);
                    }
                    Tok::LParen => depth += 1,
                    Tok::RParen if depth == 0 => break,
                    Tok::RParen => depth -= 1,
                    Tok::Comma if depth == 0 => {
                        args.push(std::mem::take(&mut arg));
                        continue;
                    }
                    _ => {}
                }
                arg.push(t);
            }
            self.no_expand = false;
            if !(params == 0 && args.is_empty() && arg.is_empty()) {
                args.push(arg);
            }
            if args.len() != params as usize {
                self.source(line, "wrong number of macro arguments");
                return Some(err);
            }
        }
        let mut out = std::mem::take(&mut self.pending);
        out.clear();
        let ok = self.expand_macro(mi, args, &mut out, 0, syms);
        for t in &mut out {
            t.line = line;
        }
        if ok.is_none() {
            return Some(err);
        }
        self.pending = out;
        self.pending_pos = 1;
        self.pending.first().copied()
    }

    /// Appends the expansion of macro `mi` with these (unexpanded) arguments to `out`.
    fn expand_macro(&mut self, mi: usize, args: Vec<Vec<Token>>, out: &mut Vec<Token>, depth: u32, syms: &Interner) -> Option<()> {
        if depth > MAX_MACRO_DEPTH {
            self.unsupported(self.line, "preprocessor: macro expansion nested too deeply");
            return None;
        }
        // arguments first: they are expanded on their own, where this macro is still a macro
        let mut expanded = Vec::with_capacity(args.len());
        for a in &args {
            let mut e = Vec::new();
            self.expand_list(a, &mut e, depth + 1, syms)?;
            expanded.push(e);
        }
        // the body leaves the macro while it is rescanned (its name is no macro meanwhile)
        let body = std::mem::take(&mut self.macros[mi].body);
        self.macros[mi].active = true;
        let r = if expanded.is_empty() {
            self.expand_list(&body, out, depth + 1, syms)
        } else {
            let mut list = Vec::with_capacity(body.len());
            for t in &body {
                if t.kind == Tok::MacroParam {
                    list.extend_from_slice(&expanded[t.val as usize]);
                } else {
                    list.push(*t);
                }
            }
            self.expand_list(&list, out, depth + 1, syms)
        };
        self.macros[mi].active = false;
        self.macros[mi].body = body;
        r
    }

    /// Appends `list` with its macro uses expanded to `out`.
    fn expand_list(&mut self, list: &[Token], out: &mut Vec<Token>, depth: u32, syms: &Interner) -> Option<()> {
        let mut i = 0;
        while i < list.len() {
            let t = list[i];
            i += 1;
            let mi = match self.is_macro(t.val) {
                Some(mi) if t.kind == Tok::Ident && !self.macros[mi].active => mi,
                _ => {
                    out.push(t);
                    continue;
                }
            };
            let mut args = Vec::new();
            if let Some(params) = self.macros[mi].params {
                let name = syms.name(super::ast::Sym(self.macros[mi].name));
                if list.get(i).map(|t| t.kind) != Some(Tok::LParen) {
                    let what = format!("preprocessor: function-like macro {name} used without arguments");
                    self.unsupported(t.line, &what);
                    return None;
                }
                i += 1;
                let mut depth = 0u32;
                let mut arg = Vec::new();
                loop {
                    let Some(&a) = list.get(i) else {
                        let what = format!("preprocessor: arguments of macro {name} continue after the expansion they start in");
                        self.unsupported(t.line, &what);
                        return None;
                    };
                    i += 1;
                    match a.kind {
                        Tok::LParen => depth += 1,
                        Tok::RParen if depth == 0 => break,
                        Tok::RParen => depth -= 1,
                        Tok::Comma if depth == 0 => {
                            args.push(std::mem::take(&mut arg));
                            continue;
                        }
                        _ => {}
                    }
                    arg.push(a);
                }
                if !(params == 0 && args.is_empty() && arg.is_empty()) {
                    args.push(arg);
                }
                if args.len() != params as usize {
                    self.source(t.line, "wrong number of macro arguments");
                    return None;
                }
            }
            self.expand_macro(mi, args, out, depth + 1, syms)?;
        }
        Some(())
    }

    /// The tokens of a directive line from `from` to `eol`, unexpanded; leaves the lexer at
    /// `eol`.
    fn line_tokens(&mut self, from: usize, eol: usize, syms: &mut Interner) -> Option<Vec<Token>> {
        self.src = &self.full[..eol];
        self.pos = from;
        self.in_directive = true;
        self.no_expand = true;
        let mut v = Vec::new();
        let ok = loop {
            let t = self.scan(syms);
            match t.kind {
                Tok::Eof => break true,
                Tok::Err => break false,
                _ => v.push(t),
            }
        };
        self.src = self.full;
        self.in_directive = false;
        self.no_expand = false;
        if !ok {
            self.pos = self.src.len();
            return None;
        }
        self.pos = eol;
        Some(v)
    }

    /// The directive name at `p` (after `#` and blanks) and the position after it.
    fn directive_name(&self, mut p: usize) -> (&'a [u8], usize) {
        let s = self.full;
        while matches!(s.get(p), Some(b' ' | b'\t')) {
            p += 1;
        }
        let start = p;
        while matches!(s.get(p), Some(&c) if is_word(c)) {
            p += 1;
        }
        (&s[start..p], p)
    }

    /// Handles the directive whose `#` is at `pos`; leaves the lexer where reading goes on.
    #[cold]
    fn directive(&mut self, syms: &mut Interner) -> Option<()> {
        let s = self.full;
        let line = self.line;
        let hash = self.pos;
        let mut q = hash;
        while q > 0 && matches!(s[q - 1], b' ' | b'\t') {
            q -= 1;
        }
        if q > 0 && s[q - 1] != b'\n' && !(q == 3 && s.starts_with(&[0xEF, 0xBB, 0xBF])) {
            self.source(line, "# in the middle of a line");
            return None;
        }
        if self.no_expand {
            self.unsupported(line, "preprocessor: directive inside macro arguments");
            return None;
        }
        let eol = find_newline(s, hash);
        let mut e = eol;
        if e > hash && s[e - 1] == b'\r' {
            e -= 1;
        }
        if s[e - 1] == b'\\' {
            self.unsupported(line, "preprocessor: line continuation");
            return None;
        }
        let (name, p) = self.directive_name(hash + 1);
        match name {
            b"define" => self.define(p, eol, syms),
            b"undef" => {
                let toks = self.line_tokens(p, eol, syms)?;
                match toks[..] {
                    [t] if t.kind == Tok::Ident => {
                        if let Some(m) = self.macro_of.get_mut(t.val as usize) {
                            *m = 0;
                        }
                        Some(())
                    }
                    _ => {
                        self.source(line, "#undef takes one macro name");
                        None
                    }
                }
            }
            b"ifdef" | b"ifndef" => {
                let toks = self.line_tokens(p, eol, syms)?;
                let defined = match toks[..] {
                    [t] if is_name(t.kind) => self.is_macro(t.val).is_some() && t.kind == Tok::Ident,
                    _ => {
                        self.source(line, "#ifdef takes one macro name");
                        return None;
                    }
                };
                let taken = defined == (name == b"ifdef");
                self.conds.push(Cond { taken, else_seen: false });
                if taken {
                    Some(())
                } else {
                    self.skip_group(syms)
                }
            }
            b"if" => {
                let taken = self.eval_line(p, eol, syms)?;
                self.conds.push(Cond { taken, else_seen: false });
                if taken {
                    Some(())
                } else {
                    self.skip_group(syms)
                }
            }
            b"elif" | b"else" => {
                // reached from selected text: the rest of the group is not selected
                let Some(c) = self.conds.last_mut() else {
                    self.source(line, "#else without #if");
                    return None;
                };
                if c.else_seen {
                    self.source(line, "#else after #else");
                    return None;
                }
                c.else_seen = name == b"else";
                self.pos = eol;
                self.skip_group(syms)
            }
            b"endif" => {
                if self.conds.pop().is_none() {
                    self.source(line, "#endif without #if");
                    return None;
                }
                self.pos = eol;
                Some(())
            }
            // the null directive
            b"" if s[p..eol].iter().all(|&b| CLASS[b as usize] == C_WS) => {
                self.pos = eol;
                Some(())
            }
            _ => {
                let what = format!("preprocessor: #{}", String::from_utf8_lossy(name));
                self.unsupported(line, &what);
                None
            }
        }
    }

    fn define(&mut self, p: usize, eol: usize, syms: &mut Interner) -> Option<()> {
        let s = self.full;
        let line = self.line;
        let (name, mut p) = self.directive_name(p);
        if name.is_empty() || name[0].is_ascii_digit() {
            self.source(line, "#define without a macro name");
            return None;
        }
        let sym = syms.intern_at(s, p - name.len(), p);
        if matches!(self.pre.get(sym as usize), Some(&(kind, _)) if kind != Tok::Ident) || name == b"defined" {
            let what = format!("preprocessor: #define of the keyword {}", String::from_utf8_lossy(name));
            self.unsupported(line, &what);
            return None;
        }
        let mut params: Vec<u32> = Vec::new();
        let function_like = s.get(p) == Some(&b'(') && p < eol;
        if function_like {
            p += 1;
            loop {
                let (param, after) = self.directive_name(p);
                p = after;
                while matches!(s.get(p), Some(b' ' | b'\t')) && p < eol {
                    p += 1;
                }
                let c = if p < eol { s[p] } else { 0 };
                if param.first().is_some_and(u8::is_ascii_digit) {
                    self.source(line, "malformed macro parameter list");
                    return None;
                }
                if param.is_empty() {
                    if c == b')' && params.is_empty() {
                        p += 1;
                        break;
                    }
                    if c == b'.' {
                        self.unsupported(line, "preprocessor: variadic macro");
                    } else {
                        self.source(line, "malformed macro parameter list");
                    }
                    return None;
                }
                params.push(syms.intern_at(s, after - param.len(), after));
                p += 1;
                match c {
                    b')' => break,
                    b',' => {}
                    _ => {
                        self.source(line, "malformed macro parameter list");
                        return None;
                    }
                }
            }
        }
        let mut body = self.line_tokens(p, eol, syms)?;
        for t in &mut body {
            match t.kind {
                Tok::Hash | Tok::HashHash => {
                    self.unsupported(line, "preprocessor: # or ## in a macro");
                    return None;
                }
                Tok::Ident => {
                    if let Some(i) = params.iter().position(|&x| x as u64 == t.val) {
                        *t = Token { kind: Tok::MacroParam, aux: 0, line, val: i as u64 };
                    }
                }
                _ => {}
            }
        }
        if self.macro_of.len() <= sym as usize {
            self.macro_of.resize(sym as usize + 1, 0);
        }
        self.macros.push(Macro { name: sym, params: function_like.then_some(params.len() as u32), body, active: false });
        self.macro_of[sym as usize] = self.macros.len() as u32;
        Some(())
    }

    /// Skips the lines of an unselected group, from the end of the directive line at `pos` to
    /// the end of the `#elif` / `#else` line that selects what follows, or of the group's
    /// `#endif`.
    fn skip_group(&mut self, syms: &mut Interner) -> Option<()> {
        let s = self.full;
        let mut p = self.pos;
        let mut depth = 0u32;
        loop {
            // `p` is at the end of a line
            if p >= s.len() {
                self.source(self.line, "#if without #endif");
                return None;
            }
            p += 1;
            self.line += 1;
            let mut q = p;
            while matches!(s.get(q), Some(b' ' | b'\t')) {
                q += 1;
            }
            if s.get(q) == Some(&b'#') {
                let eol = find_newline(s, q);
                let (name, after) = self.directive_name(q + 1);
                match name {
                    b"if" | b"ifdef" | b"ifndef" => depth += 1,
                    b"endif" if depth > 0 => depth -= 1,
                    b"endif" => {
                        self.conds.pop();
                        self.pos = eol;
                        return Some(());
                    }
                    b"else" | b"elif" if depth == 0 => {
                        let line = self.line;
                        let Some(c) = self.conds.last_mut() else {
                            self.source(line, "#else without #if");
                            return None;
                        };
                        if c.else_seen {
                            self.source(line, "#else after #else");
                            return None;
                        }
                        c.else_seen = name == b"else";
                        if !c.taken {
                            let taken = name == b"else" || self.eval_line(after, eol, syms)?;
                            if taken {
                                if let Some(c) = self.conds.last_mut() {
                                    c.taken = true;
                                }
                                self.pos = eol;
                                return Some(());
                            }
                        }
                    }
                    _ => {}
                }
                p = eol;
                continue;
            }
            // an ordinary line: to its end, or to the end of the line a comment ends on
            loop {
                match s.get(q) {
                    None | Some(b'\n') => break,
                    Some(b'/') if s.get(q + 1) == Some(&b'/') => {
                        q = find_newline(s, q);
                        break;
                    }
                    Some(b'/') if s.get(q + 1) == Some(&b'*') => {
                        q += 2;
                        loop {
                            match s.get(q) {
                                None => break,
                                Some(b'*') if s.get(q + 1) == Some(&b'/') => {
                                    q += 2;
                                    break;
                                }
                                Some(b'\n') => self.line += 1,
                                _ => {}
                            }
                            q += 1;
                        }
                    }
                    _ => q += 1,
                }
            }
            p = q;
        }
    }

    /// The truth of the `#if` / `#elif` expression on the line from `from` to `eol`.
    fn eval_line(&mut self, from: usize, eol: usize, syms: &mut Interner) -> Option<bool> {
        let line = self.line;
        let raw = self.line_tokens(from, eol, syms)?;
        // `defined X` / `defined(X)` before any expansion
        let mut list = Vec::with_capacity(raw.len());
        let mut i = 0;
        while i < raw.len() {
            let t = raw[i];
            i += 1;
            if t.kind == Tok::Ident && syms.name(super::ast::Sym(t.val as u32)) == "defined" {
                let paren = raw.get(i).map(|t| t.kind) == Some(Tok::LParen);
                let name = raw.get(i + paren as usize).copied();
                let close = !paren || raw.get(i + 2).map(|t| t.kind) == Some(Tok::RParen);
                match name {
                    Some(n) if is_name(n.kind) && close => {
                        let v = n.kind == Tok::Ident && self.is_macro(n.val).is_some();
                        list.push(Token { kind: Tok::Int, aux: 0, line, val: v as u64 });
                        i += if paren { 3 } else { 1 };
                    }
                    _ => {
                        self.source(line, "defined takes one macro name");
                        return None;
                    }
                }
            } else {
                list.push(t);
            }
        }
        let mut toks = Vec::with_capacity(list.len());
        self.expand_list(&list, &mut toks, 0, syms)?;
        let mut e = CondExpr { toks: &toks, at: 0, depth: 0, bad: None };
        let v = e.ternary();
        if e.bad.is_none() && e.at != toks.len() {
            e.bad = Some("malformed #if expression");
        }
        if toks.is_empty() {
            e.bad = Some("#if without an expression");
        }
        if let Some(msg) = e.bad {
            self.source(line, msg);
            return None;
        }
        Some(v != 0)
    }
}

/// A token that is spelled like an identifier.
pub fn is_name(k: Tok) -> bool {
    k == Tok::Ident || (k as u8 >= Tok::KwIf as u8 && k as u8 <= Tok::KwUnsupported as u8)
}

/// An `#if` expression: C operators on 64-bit integers; a name that is not a macro is 0.
struct CondExpr<'t> {
    toks: &'t [Token],
    at: usize,
    depth: u32,
    bad: Option<&'static str>,
}

impl CondExpr<'_> {
    fn peek(&self) -> Tok {
        self.toks.get(self.at).map_or(Tok::Eof, |t| t.kind)
    }

    fn ternary(&mut self) -> i64 {
        let c = self.binary(1);
        if self.peek() != Tok::Question {
            return c;
        }
        self.at += 1;
        let a = self.ternary();
        if self.peek() != Tok::Colon {
            self.bad = Some("malformed #if expression");
            return 0;
        }
        self.at += 1;
        let b = self.ternary();
        if c != 0 {
            a
        } else {
            b
        }
    }

    fn binary(&mut self, min: u8) -> i64 {
        let mut l = self.unary();
        loop {
            let op = self.peek();
            let prec = match op {
                Tok::Star | Tok::Slash | Tok::Percent => 10,
                Tok::Plus | Tok::Minus => 9,
                Tok::Shl | Tok::Shr => 8,
                Tok::Lt | Tok::Le | Tok::Gt | Tok::Ge => 7,
                Tok::EqEq | Tok::Ne => 6,
                Tok::Amp => 5,
                Tok::Caret => 4,
                Tok::Pipe => 3,
                Tok::AndAnd => 2,
                Tok::OrOr => 1,
                _ => return l,
            };
            if prec < min || self.bad.is_some() {
                return l;
            }
            self.at += 1;
            let r = self.binary(prec + 1);
            l = match op {
                Tok::Star => l.wrapping_mul(r),
                Tok::Slash | Tok::Percent => {
                    if r == 0 {
                        self.bad = Some("division by zero in an #if expression");
                        0
                    } else if op == Tok::Slash {
                        l.wrapping_div(r)
                    } else {
                        l.wrapping_rem(r)
                    }
                }
                Tok::Plus => l.wrapping_add(r),
                Tok::Minus => l.wrapping_sub(r),
                Tok::Shl => l.wrapping_shl(r as u32),
                Tok::Shr => l.wrapping_shr(r as u32),
                Tok::Lt => (l < r) as i64,
                Tok::Le => (l <= r) as i64,
                Tok::Gt => (l > r) as i64,
                Tok::Ge => (l >= r) as i64,
                Tok::EqEq => (l == r) as i64,
                Tok::Ne => (l != r) as i64,
                Tok::Amp => l & r,
                Tok::Caret => l ^ r,
                Tok::Pipe => l | r,
                Tok::AndAnd => (l != 0 && r != 0) as i64,
                _ => (l != 0 || r != 0) as i64,
            };
        }
    }

    fn unary(&mut self) -> i64 {
        let Some(&t) = self.toks.get(self.at) else {
            self.bad = Some("malformed #if expression");
            return 0;
        };
        self.at += 1;
        self.depth += 1;
        if self.depth > 64 {
            self.bad = Some("#if expression nested too deeply");
            return 0;
        }
        let v = self.unary_of(t);
        self.depth -= 1;
        v
    }

    fn unary_of(&mut self, t: Token) -> i64 {
        match t.kind {
            Tok::Int => t.val as i64,
            Tok::KwTrue => 1,
            Tok::Bang => (self.unary() == 0) as i64,
            Tok::Minus => self.unary().wrapping_neg(),
            Tok::Plus => self.unary(),
            Tok::Tilde => !self.unary(),
            Tok::LParen => {
                let v = self.ternary();
                if self.peek() != Tok::RParen {
                    self.bad = Some("malformed #if expression");
                    return 0;
                }
                self.at += 1;
                v
            }
            k if is_name(k) => 0,
            _ => {
                self.bad = Some("malformed #if expression");
                0
            }
        }
    }
}

/// The scalar kind names as the printer writes them.
pub fn scalar_name(k: ScalarKind) -> &'static str {
    match k {
        ScalarKind::Bool => "bool",
        ScalarKind::Int => "int",
        ScalarKind::Uint => "uint",
        ScalarKind::Int64 => "int64_t",
        ScalarKind::Uint64 => "uint64_t",
        ScalarKind::Half => "half",
        ScalarKind::Float => "float",
        ScalarKind::Double => "double",
        ScalarKind::Min16Float => "min16float",
        ScalarKind::Min16Int => "min16int",
        ScalarKind::Min16Uint => "min16uint",
        ScalarKind::Int16 => "int16_t",
        ScalarKind::Uint16 => "uint16_t",
        ScalarKind::Float16 => "float16_t",
    }
}

/// The name of a resource kind.
pub fn res_name(k: ResKind) -> &'static str {
    RES_KINDS.iter().find(|(_, r)| *r == k).map_or("?", |(n, _)| n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> Result<Vec<Token>, Error> {
        let mut syms = Interner::default();
        let mut lx = Lexer::new(src, &mut syms);
        let mut out = Vec::new();
        loop {
            let t = lx.next(&mut syms);
            if let Some(e) = lx.take_error() {
                return Err(e);
            }
            if t.kind == Tok::Eof {
                return Ok(out);
            }
            out.push(t);
        }
    }

    fn kinds(src: &str) -> Vec<Tok> {
        lex(src).unwrap().iter().map(|t| t.kind).collect()
    }

    /// The text back from the tokens: names, numbers and punctuation, a space between.
    fn text(src: &str) -> String {
        let mut syms = Interner::default();
        let mut lx = Lexer::new(src, &mut syms);
        let mut out = String::new();
        loop {
            let t = lx.next(&mut syms);
            assert!(!lx.has_error(), "{:?}", lx.take_error());
            match t.kind {
                Tok::Eof => return out.trim_end().to_string(),
                Tok::Int => out.push_str(&t.val.to_string()),
                k if is_name(k) => out.push_str(syms.name(super::super::ast::Sym(t.val as u32))),
                k => out.push_str(&format!("{k:?}")),
            }
            out.push(' ');
        }
    }

    #[test]
    fn punctuation() {
        use Tok::*;
        assert_eq!(
            kinds("( ) { } [ ] ; , . : ? ~ ! + - * / % & | ^ < > << >> <= >= == != && || = += -= *= /= %= &= |= ^= <<= >>= ++ -- :: ->"),
            [
                LParen, RParen, LBrace, RBrace, LBracket, RBracket, Semi, Comma, Dot, Colon, Question, Tilde, Bang, Plus, Minus, Star, Slash, Percent, Amp,
                Pipe, Caret, Lt, Gt, Shl, Shr, Le, Ge, EqEq, Ne, AndAnd, OrOr, Assign, PlusEq, MinusEq, StarEq, SlashEq, PercentEq, AmpEq, PipeEq, CaretEq,
                ShlEq, ShrEq, PlusPlus, MinusMinus, ColonColon, Arrow
            ]
        );
        // the longest token wins, with no spaces to help
        assert_eq!(kinds("a>>=b>>c>=d>e"), [Ident, ShrEq, Ident, Shr, Ident, Ge, Ident, Gt, Ident]);
        assert_eq!(kinds("a+++b---c"), [Ident, PlusPlus, Plus, Ident, MinusMinus, Minus, Ident]);
        assert_eq!(kinds("x.5 .5 1..x"), [Ident, Float, Float, Float, Dot, Ident]);
    }

    #[test]
    fn words() {
        let t = lex("if uint uint4 float4x4 dword uint64_t3 half2x3 Texture2D RWByteAddressBuffer SamplerComparisonState TriangleStream sample foo float1 float1x4 typedef").unwrap();
        assert_eq!(t[0].kind, Tok::KwIf);
        let num = |i: usize| (t[i].kind, num_type(t[i].aux));
        assert_eq!(num(1), (Tok::TyNum, (ScalarKind::Uint, 1, 1)));
        assert_eq!(num(2), (Tok::TyNum, (ScalarKind::Uint, 1, 4)));
        assert_eq!(num(3), (Tok::TyNum, (ScalarKind::Float, 4, 4)));
        assert_eq!(num(4), (Tok::TyNum, (ScalarKind::Uint, 1, 1)));
        assert_eq!(num(5), (Tok::TyNum, (ScalarKind::Uint64, 1, 3)));
        assert_eq!(num(6), (Tok::TyNum, (ScalarKind::Half, 2, 3)));
        assert_eq!((t[7].kind, RES_KINDS[t[7].aux as usize].1), (Tok::TyRes, ResKind::Texture2D));
        assert_eq!((t[8].kind, RES_KINDS[t[8].aux as usize].1), (Tok::TyRes, ResKind::RWByteAddressBuffer));
        assert_eq!((t[9].kind, t[9].aux), (Tok::TySampler, 1));
        assert_eq!((t[10].kind, t[10].aux), (Tok::TyStream, 2));
        assert_eq!((t[11].kind, QUALS[t[11].aux as usize - 1].0), (Tok::Ident, "sample"));
        assert_eq!((t[12].kind, t[12].aux), (Tok::Ident, 0));
        assert_eq!([t[13].kind, t[14].kind, t[15].kind], [Tok::KwUnsupported; 3]);
        assert!(Lexer::can_be_declared(t[13]) && !Lexer::can_be_declared(t[15]));
        // every shape of every scalar has its own aux, and the shapes read back
        for (i, _) in SCALARS.iter().enumerate() {
            let base = (i * SHAPES) as u16;
            for (rows, cols) in [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (3, 4), (4, 4)] {
                let aux = num_shaped(base, rows, cols).unwrap();
                assert_eq!(num_type(aux), (SCALARS[i].1, rows as u8, cols as u8));
                assert_eq!(num_scalar(aux), base);
            }
            assert_eq!(num_shaped(base, 1, 5), None);
            assert_eq!(num_shaped(base, 5, 1), None);
        }
    }

    #[test]
    fn integer_literals() {
        let ints = |src: &str| lex(src).unwrap().iter().map(|t| (t.val, int_suffix(t.aux))).collect::<Vec<_>>();
        use IntSuffix::*;
        assert_eq!(
            ints("0 7 010 0777 0x1F 0XfF 18446744073709551615 0xFFFFFFFFFFFFFFFFull 01777777777777777777777"),
            [(0, None), (7, None), (8, None), (511, None), (31, None), (255, None), (u64::MAX, None), (u64::MAX, ULL), (u64::MAX, None)]
        );
        assert_eq!(ints("1u 1U 1l 1L 1ul 1LU 1ll 1LL 1ull 1llu 1ULL"), [(1, U), (1, U), (1, L), (1, L), (1, UL), (1, UL), (1, LL), (1, LL), (1, ULL), (1, ULL), (1, ULL)]);
        for bad in ["18446744073709551616", "0x10000000000000000", "02000000000000000000000", "09", "0x", "1uu", "1lul", "1ulll", "12ab", "0xfg", "1_0"] {
            assert!(matches!(lex(bad), Err(Error::Source { line: 1, .. })), "{bad}: {:?}", lex(bad));
        }
    }

    #[test]
    fn float_literals_are_exact() {
        let floats = |src: &str| lex(src).unwrap().iter().map(|t| (t.kind, f64::from_bits(t.val), float_suffix(t.aux))).collect::<Vec<_>>();
        use FloatSuffix::*;
        assert_eq!(
            floats("1.5 .5 1. 1e3 1E+3 1.5e-3 2.5h 1.5l 1.f 1e2F 0.0 00.5"),
            [1.5, 0.5, 1.0, 1e3, 1e3, 1.5e-3, 2.5, 1.5, 1.0, 100.0, 0.0, 0.5].into_iter().zip([None, None, None, None, None, None, H, L, F, F, None, None]).map(|(v, s)| (Tok::Float, v, s)).collect::<Vec<_>>()
        );
        // the nearest double to the text
        let max = format!("6.2831853 0.1 4294967295.0 1e-8 17976931348623157{}.0", "0".repeat(292));
        let v = floats(&max);
        assert_eq!(v.iter().map(|t| t.1.to_bits()).collect::<Vec<_>>(), [6.2831853f64.to_bits(), 0.1f64.to_bits(), 4294967295.0f64.to_bits(), 1e-8f64.to_bits(), f64::MAX.to_bits()]);
        // `f`: the nearest float to the text, in one rounding. Just above the midpoint of
        // 1.0 and its successor: through the nearest double (the midpoint itself) it would
        // round to even, 1.0
        let v = floats("1.00000005960464477539062500000000001f 1.00000005960464477539062500000000001 6.2831853f 0.1f 4294967295.0f 1e-8f 16777217.0f 1e-46f 1e39f");
        let single = |i: usize| {
            assert_eq!(v[i].1, v[i].1 as f32 as f64, "a float, widened");
            (v[i].1 as f32).to_bits()
        };
        assert_eq!(single(0), 0x3F80_0001);
        assert_eq!(v[1].1.to_bits(), 0x3FF0_0000_1000_0000);
        assert_eq!((v[1].1 as f32).to_bits(), 0x3F80_0000);
        assert_eq!(single(2), 6.2831853f32.to_bits());
        assert_eq!(single(3), 0.1f32.to_bits());
        assert_eq!(single(4), 0x4F80_0000);
        assert_eq!(single(5), 1e-8f32.to_bits());
        assert_eq!(single(6), 16777216.0f32.to_bits());
        assert_eq!(single(7), 0);
        assert_eq!(v[8].1, f64::INFINITY);
        for bad in ["1e", "1e+", "1.5q", "1.5ff", "1.0e5x"] {
            assert!(matches!(lex(bad), Err(Error::Source { .. })), "{bad}");
        }
        assert_eq!(lex("1.xxxx"), Err(Error::Unsupported("member access on a numeric literal".into())));
    }

    #[test]
    fn comments_strings_and_lines() {
        let t = lex("a // b c\n  /* d\n e */ f /**/ g\n\"s t\" h // end").unwrap();
        assert_eq!(t.iter().map(|t| (t.kind, t.line)).collect::<Vec<_>>(), [(Tok::Ident, 1), (Tok::Ident, 3), (Tok::Ident, 3), (Tok::Str, 4), (Tok::Ident, 4)]);
        let mut syms = Interner::default();
        let src = "[x(\"tri angle\")]";
        let mut lx = Lexer::new(src, &mut syms);
        let s = (0..4).map(|_| lx.next(&mut syms)).last().unwrap();
        assert_eq!((s.kind, lx.str_text(s)), (Tok::Str, "tri angle"));
        // a byte order mark, carriage returns, text that is not ASCII in a comment
        assert_eq!(kinds("\u{FEFF}a\r\nb // RDNA2 \u{2192} HLSL \u{b7}\r\nc"), [Tok::Ident; 3]);
        assert_eq!(lex("a /* b"), Err(Error::Source { line: 1, msg: "comment without */".into() }));
        assert_eq!(lex("a\n\"b\nc\""), Err(Error::Source { line: 2, msg: "string without closing quote".into() }));
        assert_eq!(lex("\n\na \u{2192} b"), Err(Error::Source { line: 3, msg: "unexpected byte 0xE2".into() }));
        assert_eq!(lex("a @ b"), Err(Error::Source { line: 1, msg: "unexpected character '@'".into() }));
        assert_eq!(lex("\"a\\n\""), Err(Error::Unsupported("escape in a string literal".into())));
    }

    #[test]
    fn macros() {
        assert_eq!(text("#define N 4u\n#define E\n#define P 1 + N\nx = P E * N;"), "x Assign 1 Plus 4 Star 4 Semi");
        assert_eq!(text("#define XS(i) lds[8u + (i)]\nXS(a) = XS(XS(1) + f(b, c));"), "lds LBracket 8 Plus LParen a RParen RBracket Assign lds LBracket 8 Plus LParen lds LBracket 8 Plus LParen 1 RParen RBracket Plus f LParen b Comma c RParen RParen RBracket Semi");
        assert_eq!(text("#define F(a, b) (b - a)\n#define G() 7\n#define H(x) x x\nF(1, (2, 3)) G() H(q) F\n(4,\n 5)"), "LParen LParen 2 Comma 3 RParen Minus 1 RParen 7 q q LParen 5 Minus 4 RParen");
        // a macro is not expanded inside its own expansion, also through another
        assert_eq!(text("#define A B + 1\n#define B A + 2\nA; B"), "A Plus 2 Plus 1 Semi B Plus 1 Plus 2");
        assert_eq!(text("#define M(x) M(x) + x\nM(M(2))"), "M LParen M LParen 2 RParen Plus 2 RParen Plus M LParen 2 RParen Plus 2");
        // redefinition, #undef, a name defined to itself, a qualifier word as a macro
        assert_eq!(text("#define N 1\n#define N 2\nN\n#undef N\nN\n#undef M\n#define sample sample\nsample"), "2 N sample");
        // a token of an expansion is on the line of the use
        let t = lex("#define TWO a\\\n").err();
        assert_eq!(t, Some(Error::Unsupported("preprocessor: line continuation".into())));
        let t = lex("#define TWO a b\n\n\nTWO\nc").unwrap();
        assert_eq!(t.iter().map(|t| t.line).collect::<Vec<_>>(), [4, 4, 5]);
        for (bad, what) in [
            ("#define F(x) x\nF + 1", "preprocessor: function-like macro F used without arguments"),
            ("#define F(x) x\n#define G F\nG(1)", "preprocessor: function-like macro F used without arguments"),
            ("#define F(x, ...) x\n", "preprocessor: variadic macro"),
            ("#define S(x) #x\n", "preprocessor: # or ## in a macro"),
            ("#define C(a, b) a ## b\n", "preprocessor: # or ## in a macro"),
            ("#define uint int\n", "preprocessor: #define of the keyword uint"),
            ("#define X 1 /* a\n b */\n", "preprocessor: comment spanning lines in a directive"),
            ("#define F(x) x\nF(\n#define Y 1\n)", "preprocessor: directive inside macro arguments"),
            ("#pragma once\n", "preprocessor: #pragma"),
            ("#include \"a.hlsl\"\n", "preprocessor: #include"),
            ("#line 3\n", "preprocessor: #line"),
            ("#error no\n", "preprocessor: #error"),
        ] {
            assert_eq!(lex(bad), Err(Error::Unsupported(what.into())), "{bad}");
        }
        for (bad, line) in [("#define F(x) x\nF(1, 2)", 2), ("#define F(x) x\n\nF(1", 3), ("#define\n", 1), ("#define F(x\n", 1), ("#define F(1) x\n", 1), ("#undef\n", 1), ("a #define X 1\n", 1)] {
            assert!(matches!(lex(bad), Err(Error::Source { line: l, .. }) if l == line), "{bad}: {:?}", lex(bad));
        }
    }

    #[test]
    fn conditionals() {
        assert_eq!(text("#define A 1\n#ifdef A\na\n#else\nb\n#endif\n#ifndef A\nc\n#else\nd\n#endif\n#ifdef B\ne\n#endif\nf"), "a d f");
        assert_eq!(text("#define SPIN 0\n#if SPIN\na\n#elif SPIN + 1 > 0\nb\n#elif 1\nc\n#else\nd\n#endif\ne"), "b e");
        assert_eq!(text("#if 0\na\n#elif 0\nb\n#else\nc\n#endif\n#if 1\nd\n#elif 1\ne\n#else\nf\n#endif"), "c d");
        // nested groups; directives and comments in skipped text
        assert_eq!(text("#if 0\n#if 1\na\n#else\nb\n#endif\n/* #endif\n#else */ c // #endif\n#pragma x\n#define Q 1\n#else\n#ifdef Q\nd\n#else\ne\n#endif\n#endif\n  #  if 1 // c\nf\n# endif"), "e f");
        // expressions
        for (expr, value) in [
            ("1", true), ("0", false), ("!0", true), ("-1", true), ("1 - 1", false), ("2 * 3 == 6", true), ("7 / 2 == 3 && 7 % 2 == 1", true), ("1 << 4 == 16", true),
            ("(1 | 2) == 3 && (3 & 2) == 2 && (3 ^ 1) == 2", true), ("1 ? 0 : 1", false), ("0 ? 0 : 1", true), ("defined(X) && defined Y", true), ("defined(Z)", false),
            ("X == 3 && Y", true), ("UNKNOWN", false), ("!defined(Z) || nothing", true), ("~0 == -1", true), ("2 > 1 && 1 >= 1 && 1 < 2 && 2 <= 2 && 1 != 2", true),
            ("1 || 0 && 0", true), ("(1 || 0) && 0", false), ("true", true), ("false", false), ("0x10u == 16", true), ("X > 2 ? Y : 0", true),
        ] {
            let src = format!("#define X 3\n#define Y 1\n#if {expr}\nyes\n#else\nno\n#endif");
            assert_eq!(text(&src), if value { "yes" } else { "no" }, "{expr}");
        }
        // lines go on counting through skipped text
        let t = lex("#if 0\na\n/* x\n y */\n#else\nb\n#endif\nc").unwrap();
        assert_eq!(t.iter().map(|t| t.line).collect::<Vec<_>>(), [6, 8]);
        for (bad, line) in [
            ("#if 1\na", 2), ("#if 0\na", 2), ("#endif", 1), ("a\n#else\n", 2), ("#if 1\n#else\n#else\n#endif", 3), ("#if 0\n#else\n#else\n#endif", 3), ("#if 0\n#else\n#elif 1\n#endif", 3),
            ("#if\n#endif", 1), ("#if 1 +\n#endif", 1), ("#if (1\n#endif", 1), ("#if 1 1\n#endif", 1), ("#if 1 / 0\n#endif", 1), ("#ifdef\n#endif", 1), ("#ifdef A B\n#endif", 1),
            ("#if defined\n#endif", 1), ("#if 0\n#elif )\n#endif", 2),
        ] {
            assert!(matches!(lex(bad), Err(Error::Source { line: l, .. }) if l == line), "{bad}: {:?}", lex(bad));
        }
        let deep = format!("#if {}1{}\n#endif", "(".repeat(100), ")".repeat(100));
        assert!(matches!(lex(&deep), Err(Error::Source { .. })));
    }
}
