//! The front end: the translator's HLSL dialect → IR.
//!
//! * [`lex`] / [`parse`]: text (with the few preprocessor lines the translator emits) → [`ast`],
//! * [`lower`]: AST → SSA IR in one walk: every user function is inlined into the entry
//!   point (DXIL has no calls to defined functions), `static` globals and locals become SSA
//!   values (Braun et al.'s on-the-fly construction, no allocas for scalars), arrays and
//!   groupshared memory stay memory,
//! * [`types`] / [`emit`]: HLSL types, typed values and instruction emission shared by
//!   [`lower`], [`intrinsics`] (HLSL intrinsic functions → dx.op / LLVM instructions) and
//!   [`resources`] (buffers, textures, samplers, descriptor heaps).

pub mod ast;
pub mod astprint;
pub mod emit;
pub mod intrinsics;
pub mod lex;
pub mod lower;
pub mod parse;
pub mod resources;
pub mod stage;
pub mod types;
