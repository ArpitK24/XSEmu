//! LLVM 3.7 bitcode → [`crate::ir::Module`], for round-trip tests against DXC's output (owner: the bitcode subagent).
//!
//! [`read`] takes the bitcode DXC writes (BLOCKINFO and local abbreviations, constants in
//! the module and per function, forward references, relative operand ids) and builds the
//! module the writer would be given for it, so that `read` → [`crate::bcwriter::write`] can
//! be validated and disassembled against the original. It covers what DXIL shader modules
//! contain and reports anything else as [`Error::Unsupported`] (constant expressions,
//! debug info, opaque or vector-valued code, partial fast-math flags), never by panicking.
//!
//! What the IR cannot hold is dropped and counted in [`Notes`]: metadata attachments other
//! than `!dx.precise` (DXC's `!tbaa` on loads and stores) and value names.

use crate::bcwriter::{align_of, code::*};
use crate::bitstream::{Entry, Reader};
use crate::ir::*;
use crate::{Error, Result};
use std::collections::HashMap;

fn bad(msg: impl Into<String>) -> Error {
    Error::Internal(format!("bcreader: {}", msg.into()))
}

fn u32_at(b: &[u8], at: usize) -> Result<u32> {
    b.get(at..at + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])).ok_or_else(|| bad("container truncated"))
}

/// Where the parts of a DXBC container are: (fourcc, offset of the part data, size).
pub fn container_parts(container: &[u8]) -> Result<Vec<([u8; 4], usize, usize)>> {
    if container.get(..4) != Some(b"DXBC") {
        return Err(bad("not a DXBC container"));
    }
    let count = u32_at(container, 28)? as usize;
    let mut parts = Vec::new();
    for k in 0..count {
        let at = u32_at(container, 32 + 4 * k)? as usize;
        let fourcc = container.get(at..at + 4).ok_or_else(|| bad("part offset out of range"))?;
        let size = u32_at(container, at + 4)? as usize;
        if at + 8 + size > container.len() {
            return Err(bad("part size out of range"));
        }
        parts.push(([fourcc[0], fourcc[1], fourcc[2], fourcc[3]], at + 8, size));
    }
    Ok(parts)
}

/// The LLVM bitcode inside the DXIL part of a container (after the program header).
pub fn container_bitcode(container: &[u8]) -> Result<&[u8]> {
    let parts = container_parts(container)?;
    let &(_, at, size) = parts.iter().find(|p| &p.0 == b"DXIL").ok_or_else(|| bad("no DXIL part"))?;
    // DxilProgramHeader: version, size in words, then the bitcode header: 'DXIL', version,
    // offset of the bitcode from the bitcode header, size
    if size < 24 || u32_at(container, at + 8)? != 0x4c495844 {
        return Err(bad("bad DXIL program header"));
    }
    let offset = u32_at(container, at + 16)? as usize;
    let len = u32_at(container, at + 20)? as usize;
    container.get(at + 8 + offset..at + 8 + offset + len).filter(|_| 8 + offset + len <= size).ok_or_else(|| bad("bitcode out of the DXIL part"))
}

/// A copy of `container` with the bitcode of its DXIL part replaced, sizes and part offsets
/// fixed up. The hash is left as it was (`dxv` recomputes it when it signs), and the other
/// parts are kept, so this wraps a module for the validator when it has the same signatures,
/// resources and flags as the one the container was made for.
pub fn replace_bitcode(container: &[u8], bitcode: &[u8]) -> Result<Vec<u8>> {
    let parts = container_parts(container)?;
    let mut out = container[..32 + 4 * parts.len()].to_vec();
    let mut offsets = Vec::new();
    for &(fourcc, at, size) in &parts {
        offsets.push(out.len() as u32);
        out.extend_from_slice(&fourcc);
        if &fourcc != b"DXIL" {
            out.extend_from_slice(&(size as u32).to_le_bytes());
            out.extend_from_slice(&container[at..at + size]);
            continue;
        }
        let padded = bitcode.len().div_ceil(4) * 4;
        out.extend_from_slice(&((24 + padded) as u32).to_le_bytes());
        out.extend_from_slice(&container[at..at + 4]);
        out.extend_from_slice(&((24 + padded) as u32 / 4).to_le_bytes());
        out.extend_from_slice(&container[at + 8..at + 16]);
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&(bitcode.len() as u32).to_le_bytes());
        out.extend_from_slice(bitcode);
        out.resize(out.len() + padded - bitcode.len(), 0);
    }
    for (k, o) in offsets.iter().enumerate() {
        out[32 + 4 * k..36 + 4 * k].copy_from_slice(&o.to_le_bytes());
    }
    let total = out.len() as u32;
    out[24..28].copy_from_slice(&total.to_le_bytes());
    Ok(out)
}

/// What [`read`] could not carry into the IR, and where the input differs from what the
/// writer derives.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Notes {
    /// metadata attachments dropped, by kind name (`tbaa`)
    pub dropped_attachments: Vec<(String, usize)>,
    /// loads, stores and allocas whose alignment is not the one [`align_of`] gives; they
    /// are in [`Body::aligns`]
    pub other_alignments: usize,
    /// named blocks and instructions (the IR has no names)
    pub local_names: usize,
    /// calls marked `tail`
    pub tail_calls: usize,
    /// calls with an attribute list of their own (DXC repeats `nounwind` on its
    /// `llvm.lifetime` calls), which the IR does not carry
    pub call_attributes: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Options {
    /// Give every instruction [`flag::EFFECT`], so that the writer keeps code DXC left dead
    /// (its -O0 output has unused loads and calls).
    pub keep_dead: bool,
}

fn unsupported<T>(what: impl Into<String>) -> Result<T> {
    Err(Error::Unsupported(format!("bcreader: {}", what.into())))
}

/// Values a function refers to before they are defined get this bit and their value id,
/// and are patched when the function ends.
const PLACEHOLDER: u32 = 0x2000_0000;

enum RawMd {
    Str(String),
    Val(Value),
    /// operand ids plus one, 0 for null
    Node(Vec<u64>),
    Distinct(Vec<u64>),
}

enum PendingConst {
    Done(Value),
    Aggregate(Ty, Vec<u64>),
    /// a constant expression: result type, the value ids of the base and the indices of a
    /// getelementptr, or the one of the operand of a bitcast
    Gep(Ty, Vec<u64>),
    Busy,
}

struct Rd<'a> {
    r: Reader<'a>,
    m: Module,
    opts: Options,
    notes: Notes,
    types: Vec<Ty>,
    struct_name: Option<String>,
    attr_groups: HashMap<u64, Vec<u64>>,
    /// the attribute lists; one the IR has no name for is its text
    attr_lists: Vec<std::result::Result<FnAttrs, String>>,
    /// names by module-level value id, from a first pass over the symbol table
    names: HashMap<u64, String>,
    /// every value by id: globals, functions, module constants, then the current function's
    vals: Vec<Value>,
    /// (global index, initialiser value id)
    inits: Vec<(usize, u64)>,
    /// function indices that have a body, in order; the next FUNCTION block is `defined[next_body]`
    defined: Vec<u32>,
    next_body: usize,
    /// the metadata records by id, and the node of the module each has become
    raw: Vec<RawMd>,
    mds: Vec<Option<Md>>,
    /// see check_align
    other_align: Option<u32>,
    kinds: HashMap<u64, String>,
}

fn string(ops: &[u64]) -> String {
    String::from_utf8_lossy(&ops.iter().map(|&c| c as u8).collect::<Vec<u8>>()).into_owned()
}

fn decode_signed(v: u64) -> i64 {
    if v & 1 == 0 {
        (v >> 1) as i64
    } else if v != 1 {
        -((v >> 1) as i64)
    } else {
        i64::MIN
    }
}

impl<'a> Rd<'a> {
    fn ty(&self, id: u64) -> Result<Ty> {
        self.types.get(id as usize).copied().ok_or_else(|| bad(format!("type {id} is not defined")))
    }

    fn value(&self, id: u64) -> Result<Value> {
        self.vals.get(id as usize).copied().ok_or_else(|| bad(format!("value {id} is not defined")))
    }

    fn op(&self, k: usize) -> Result<u64> {
        self.r.ops.get(k).copied().ok_or_else(|| bad("record too short"))
    }

    fn skip_block(&mut self, what: &str) -> Result<()> {
        // counted as dropped, like anything else the IR has no place for
        let _ = what;
        self.r.skip()
    }

    fn types_block(&mut self) -> Result<()> {
        loop {
            let code = match self.r.next()? {
                Entry::EndBlock => return Ok(()),
                Entry::Record(c) => c,
                Entry::SubBlock(_) | Entry::End => return Err(bad("unexpected block in the type table")),
            };
            let ops = std::mem::take(&mut self.r.ops);
            let t = match code {
                TYPE_NUMENTRY => {
                    self.r.ops = ops;
                    continue;
                }
                TYPE_VOID => Ty::VOID,
                TYPE_HALF => Ty::HALF,
                TYPE_FLOAT => Ty::F32,
                TYPE_DOUBLE => Ty::F64,
                TYPE_LABEL => Ty::LABEL,
                TYPE_METADATA => Ty::METADATA,
                TYPE_INTEGER => self.m.int(*ops.first().ok_or_else(|| bad("integer type without a width"))? as u32),
                TYPE_POINTER => {
                    let to = self.ty(*ops.first().ok_or_else(|| bad("short pointer type"))?)?;
                    self.m.ptr_in(to, ops.get(1).copied().unwrap_or(0) as u32)
                }
                TYPE_ARRAY | TYPE_VECTOR => {
                    if ops.len() < 2 {
                        return Err(bad("short array type"));
                    }
                    let of = self.ty(ops[1])?;
                    if code == TYPE_ARRAY {
                        self.m.array(ops[0], of)
                    } else {
                        self.m.vector(ops[0] as u32, of)
                    }
                }
                TYPE_FUNCTION => {
                    if ops.len() < 2 || ops[0] != 0 {
                        return unsupported("vararg function type");
                    }
                    let ret = self.ty(ops[1])?;
                    let params = ops[2..].iter().map(|&p| self.ty(p)).collect::<Result<Vec<_>>>()?;
                    self.m.func_ty(ret, params)
                }
                TYPE_STRUCT_NAME => {
                    self.struct_name = Some(string(&ops));
                    self.r.ops = ops;
                    continue;
                }
                TYPE_STRUCT_ANON | TYPE_STRUCT_NAMED => {
                    if ops.first() != Some(&0) {
                        return unsupported("packed struct type");
                    }
                    let fields = ops[1..].iter().map(|&p| self.ty(p)).collect::<Result<Vec<_>>>()?;
                    if code == TYPE_STRUCT_ANON {
                        self.m.literal_struct(fields)
                    } else {
                        let name = self.struct_name.take().ok_or_else(|| bad("named struct without a name"))?;
                        self.m.named_struct(&name, fields)
                    }
                }
                TYPE_OPAQUE => return unsupported("opaque struct type"),
                other => return unsupported(format!("type code {other}")),
            };
            self.types.push(t);
            self.r.ops = ops;
        }
    }

    fn attr_group_block(&mut self) -> Result<()> {
        loop {
            match self.r.next()? {
                Entry::EndBlock => return Ok(()),
                Entry::Record(PARAMATTR_GRP_ENTRY) => {
                    let ops = &self.r.ops;
                    if ops.len() < 2 {
                        return Err(bad("short attribute group"));
                    }
                    // the parameter index leads, so a list of groups reads as one sequence
                    let mut kinds = vec![ops[1]];
                    let mut k = 2;
                    while k < ops.len() {
                        if ops[k] != 0 || k + 1 >= ops.len() {
                            return unsupported("attribute with a value");
                        }
                        kinds.push(ops[k + 1]);
                        k += 2;
                    }
                    kinds[1..].sort_unstable();
                    self.attr_groups.insert(ops[0], kinds);
                }
                Entry::Record(_) => {}
                _ => return Err(bad("unexpected block in the attribute groups")),
            }
        }
    }

    fn attr_block(&mut self) -> Result<()> {
        loop {
            match self.r.next()? {
                Entry::EndBlock => return Ok(()),
                Entry::Record(PARAMATTR_ENTRY) => {
                    const FN: u64 = 0xffff_ffff;
                    let mut groups: Vec<&[u64]> = Vec::new();
                    for id in &self.r.ops {
                        groups.push(self.attr_groups.get(id).ok_or_else(|| bad("attribute group is not defined"))?);
                    }
                    groups.sort_unstable();
                    let a = match groups.as_slice() {
                        [[FN, ATTR_NO_UNWIND]] => Ok(FnAttrs::Nounwind),
                        [[FN, ATTR_NO_UNWIND, ATTR_READ_NONE]] => Ok(FnAttrs::NounwindReadnone),
                        [[FN, ATTR_NO_UNWIND, ATTR_READ_ONLY]] => Ok(FnAttrs::NounwindReadonly),
                        [[FN, ATTR_NO_RETURN, ATTR_NO_UNWIND]] => Ok(FnAttrs::NounwindNoreturn),
                        [[FN, ATTR_NO_DUPLICATE, ATTR_NO_UNWIND]] => Ok(FnAttrs::NounwindNoduplicate),
                        [[2, ATTR_NO_CAPTURE], [FN, ATTR_NO_UNWIND]] => Ok(FnAttrs::Lifetime),
                        // only an error where a function has them: DXC also gives calls lists
                        // of their own (nonnull on the payload of DispatchMesh), which are dropped
                        other => Err(format!("function attributes {other:?}")),
                    };
                    self.attr_lists.push(a);
                }
                Entry::Record(_) => return unsupported("old attribute record"),
                _ => return Err(bad("unexpected block in the attribute lists")),
            }
        }
    }

    fn resolve_const(&mut self, pending: &mut [PendingConst], base: usize, k: usize, depth: u32) -> Result<Value> {
        match std::mem::replace(&mut pending[k], PendingConst::Busy) {
            PendingConst::Done(v) => {
                pending[k] = PendingConst::Done(v);
                Ok(v)
            }
            PendingConst::Busy => Err(bad("constant refers to itself")),
            entry => {
                if depth > 1000 {
                    return Err(bad("constants nested too deeply"));
                }
                let (ty, gep, ids) = match entry {
                    PendingConst::Aggregate(ty, ids) => (ty, false, ids),
                    PendingConst::Gep(ty, ids) => (ty, true, ids),
                    _ => return Err(bad("constant in an impossible state")),
                };
                let mut fields = Vec::with_capacity(ids.len());
                for id in ids {
                    let id = id as usize;
                    let v = if id < base {
                        self.vals[id]
                    } else if id - base < pending.len() {
                        self.resolve_const(pending, base, id - base, depth + 1)?
                    } else {
                        return Err(bad("operand of a constant is not defined"));
                    };
                    fields.push(v);
                }
                let v = if gep {
                    if !(fields[0].is_const() || matches!(fields[0].kind(), ValueKind::Global(_))) || !fields[1..].iter().all(|f| f.is_const()) {
                        return Err(bad("getelementptr constant with an operand that is not a constant"));
                    }
                    if fields.len() == 1 {
                        self.m.const_bitcast(ty, fields[0])
                    } else {
                        self.m.const_gep(ty, fields[0], &fields[1..])
                    }
                } else {
                    if matches!(self.m.def(ty), TypeDef::Ptr { .. }) {
                        return Err(bad("aggregate constant of a pointer type"));
                    }
                    self.m.aggregate(ty, fields)
                };
                pending[k] = PendingConst::Done(v);
                Ok(v)
            }
        }
    }

    fn constants_block(&mut self) -> Result<()> {
        let base = self.vals.len();
        let mut pending: Vec<PendingConst> = Vec::new();
        let mut ty = Ty::I32;
        loop {
            let code = match self.r.next()? {
                Entry::EndBlock => break,
                Entry::Record(c) => c,
                _ => return Err(bad("unexpected block in constants")),
            };
            let ops = std::mem::take(&mut self.r.ops);
            let element = match self.m.def(ty) {
                TypeDef::Array { of, .. } => Some(*of),
                _ => None,
            };
            let c = match code {
                CST_SETTYPE => {
                    ty = self.ty(*ops.first().ok_or_else(|| bad("short SETTYPE"))?)?;
                    self.r.ops = ops;
                    continue;
                }
                CST_NULL => PendingConst::Done(self.m.zero(ty)),
                CST_UNDEF => PendingConst::Done(self.m.undef(ty)),
                CST_INTEGER => {
                    let v = decode_signed(*ops.first().ok_or_else(|| bad("short integer constant"))?);
                    if !matches!(self.m.def(ty), TypeDef::Int(b) if *b <= 64) {
                        return Err(bad("integer constant of a type that is not an integer"));
                    }
                    PendingConst::Done(self.m.const_int(ty, v as u64))
                }
                CST_FLOAT => {
                    if !self.m.is_float(ty) {
                        return unsupported("float constant of an unusual type");
                    }
                    PendingConst::Done(self.m.constant(ty, Const::Float(*ops.first().ok_or_else(|| bad("short float constant"))?)))
                }
                CST_AGGREGATE => PendingConst::Aggregate(ty, ops.clone()),
                CST_CE_GEP => return unsupported("getelementptr constant that is not inbounds"),
                CST_CE_CAST => {
                    if ops.len() != 3 || ops[0] != CAST_BITCAST as u64 || !matches!(self.m.def(ty), TypeDef::Ptr { .. }) {
                        return unsupported("constant cast other than a bitcast to a pointer");
                    }
                    PendingConst::Gep(ty, vec![ops[2]])
                }
                CST_CE_INBOUNDS_GEP => {
                    // [source element type (when the length is odd), (type, value)...]
                    let pairs = &ops[ops.len() % 2..];
                    if pairs.len() < 4 || !matches!(self.m.def(ty), TypeDef::Ptr { .. }) {
                        return Err(bad("getelementptr constant without a base or of a type that is not a pointer"));
                    }
                    PendingConst::Gep(ty, pairs.chunks(2).map(|p| p[1]).collect())
                }
                CST_DATA | CST_STRING | CST_CSTRING => {
                    let of = element.ok_or_else(|| bad("data constant of a type that is not an array"))?;
                    let mut fields = Vec::with_capacity(ops.len() + 1);
                    for &x in &ops {
                        fields.push(if self.m.is_float(of) { self.m.constant(of, Const::Float(x)) } else { self.m.const_int(of, x) });
                    }
                    if code == CST_CSTRING {
                        fields.push(self.m.const_int(of, 0));
                    }
                    PendingConst::Done(self.m.aggregate(ty, fields))
                }
                other => return unsupported(format!("constant code {other} (constant expression?)")),
            };
            pending.push(c);
            self.r.ops = ops;
        }
        for k in 0..pending.len() {
            let v = self.resolve_const(&mut pending, base, k, 0)?;
            self.vals.push(v);
        }
        Ok(())
    }

    /// A metadata node by id, built on first use: DXC's writer lets a node refer to one that
    /// comes later.
    fn md(&mut self, id: u64, depth: u32) -> Result<Md> {
        let k = id as usize;
        if let Some(Some(md)) = self.mds.get(k) {
            return Ok(*md);
        }
        if depth > 1000 {
            return Err(bad("metadata nested too deeply or referring to itself"));
        }
        let md = match self.raw.get(k).ok_or_else(|| bad("metadata operand is not defined"))? {
            RawMd::Str(s) => {
                let s = s.clone();
                self.m.md_str(&s)
            }
            RawMd::Val(v) => {
                let v = *v;
                self.m.md_val(v)
            }
            RawMd::Node(ops) => {
                let ops = ops.clone();
                let mut fields = Vec::with_capacity(ops.len());
                for o in ops {
                    fields.push(match o {
                        0 => None,
                        id => Some(self.md(id - 1, depth + 1)?),
                    });
                }
                self.m.md_tuple(fields)
            }
            // the IR cannot hold one: it may only be the target of an attachment
            RawMd::Distinct(_) => return unsupported("metadata that refers to a distinct node"),
        };
        self.mds[k] = Some(md);
        Ok(md)
    }

    fn metadata_block(&mut self) -> Result<()> {
        let mut name: Option<String> = None;
        let mut named: Vec<(String, Vec<u64>)> = Vec::new();
        loop {
            let code = match self.r.next()? {
                Entry::EndBlock => break,
                Entry::Record(c) => c,
                _ => return Err(bad("unexpected block in metadata")),
            };
            let ops = std::mem::take(&mut self.r.ops);
            match code {
                METADATA_STRING => self.raw.push(RawMd::Str(string(&ops))),
                METADATA_VALUE => {
                    if ops.len() != 2 {
                        return Err(bad("short metadata value"));
                    }
                    let v = self.value(ops[1])?;
                    self.raw.push(RawMd::Val(v));
                }
                METADATA_NODE => self.raw.push(RawMd::Node(ops.clone())),
                METADATA_DISTINCT_NODE => self.raw.push(RawMd::Distinct(ops.clone())),
                METADATA_NAME => name = Some(string(&ops)),
                METADATA_NAMED_NODE => named.push((name.take().ok_or_else(|| bad("named metadata without a name"))?, ops.clone())),
                METADATA_KIND => {
                    if ops.is_empty() {
                        return Err(bad("short metadata kind"));
                    }
                    self.kinds.insert(ops[0], string(&ops[1..]));
                }
                other => return unsupported(format!("metadata code {other} (debug info?)")),
            }
            self.r.ops = ops;
        }
        self.mds.resize(self.raw.len(), None);
        // in the order of the records, so that a module read and one built the same way hold
        // their metadata in the same order; what nothing names stays out of the module
        for (n, ops) in named {
            let mut nodes = Vec::with_capacity(ops.len());
            for id in ops {
                nodes.push(self.md(id, 0)?);
            }
            self.m.named(&n, nodes);
        }
        Ok(())
    }

    /// The hints of a `!dx.controlflow.hints` node: distinct !{itself, the name, i32 hint...}.
    fn hints_of(&self, node: u64) -> Option<u8> {
        let Some(RawMd::Distinct(ops)) = self.raw.get(node as usize) else { return None };
        if ops.len() < 3 || ops[0] != node + 1 || !matches!(self.raw.get((ops[1] as usize).wrapping_sub(1)), Some(RawMd::Str(s)) if s == "dx.controlflow.hints") {
            return None;
        }
        ops[2..].iter().try_fold(0u8, |set, id| match self.raw.get((*id as usize).wrapping_sub(1)) {
            Some(RawMd::Val(v)) => self.m.as_int(*v).filter(|h| (1..=6).contains(h)).map(|h| set | 1 << (h - 1)),
            _ => None,
        })
    }

    /// Whether a `!llvm.loop` node is distinct !{itself, !{!"llvm.loop.unroll.disable"}}.
    fn is_unroll_disable(&self, node: u64) -> bool {
        let Some(RawMd::Distinct(ops)) = self.raw.get(node as usize) else { return false };
        let inner = match ops.as_slice() {
            [me, inner] if *me == node + 1 => self.raw.get((*inner as usize).wrapping_sub(1)),
            _ => None,
        };
        let Some(RawMd::Node(inner)) = inner else { return false };
        inner.len() == 1 && matches!(self.raw.get((inner[0] as usize).wrapping_sub(1)), Some(RawMd::Str(s)) if s == "llvm.loop.unroll.disable")
    }

    fn global_var(&mut self) -> Result<()> {
        let ops = self.r.ops.clone();
        if ops.len() < 6 {
            return Err(bad("short global variable record"));
        }
        let (mut value_ty, mut space) = (self.ty(ops[0])?, (ops[1] >> 2) as u32);
        if ops[1] & 2 == 0 {
            match self.m.def(value_ty) {
                TypeDef::Ptr { to, space: s } => (value_ty, space) = (*to, *s),
                _ => return Err(bad("global variable whose type is not a pointer")),
            }
        }
        if ops[5] != 0 || ops.get(6).is_some_and(|&v| v != 0) || ops.get(7).is_some_and(|&v| v != 0) || !matches!(ops[3], 0 | 3) {
            return unsupported("global variable with a section, visibility, thread-local mode or unusual linkage");
        }
        let id = self.vals.len() as u64;
        let name = self.names.get(&id).cloned().unwrap_or_default();
        let v = self.m.global(&name, value_ty, space, None, ops[1] & 1 != 0);
        let g = self.m.globals.last_mut().ok_or_else(|| bad("global"))?;
        g.internal = ops[3] == 3;
        g.align = if ops[4] == 0 { 0 } else { 1 << (ops[4] - 1).min(31) };
        g.unnamed_addr = ops.get(8) == Some(&1);
        if ops[2] != 0 {
            self.inits.push((self.m.globals.len() - 1, ops[2] - 1));
        }
        self.vals.push(v);
        Ok(())
    }

    fn function(&mut self) -> Result<()> {
        let ops = self.r.ops.clone();
        if ops.len() < 5 {
            return Err(bad("short function record"));
        }
        let mut ty = self.ty(ops[0])?;
        if let TypeDef::Ptr { to, .. } = self.m.def(ty) {
            ty = *to;
        }
        if !matches!(self.m.def(ty), TypeDef::Func { .. }) {
            return Err(bad("function whose type is not a function type"));
        }
        if ops[1] != 0 || ops[3] != 0 {
            return unsupported("function with a calling convention or linkage");
        }
        let attrs = match ops[4] {
            0 => FnAttrs::None,
            n => match self.attr_lists.get(n as usize - 1).ok_or_else(|| bad("attribute list is not defined"))? {
                Ok(a) => *a,
                Err(what) => return unsupported(what.clone()),
            },
        };
        let id = self.vals.len() as u64;
        let name = self.names.get(&id).cloned().ok_or_else(|| bad("function without a name"))?;
        let before = self.m.funcs.len();
        let v = self.m.declare(&name, ty, attrs);
        if self.m.funcs.len() == before {
            return Err(bad(format!("two functions named {name}")));
        }
        if ops[2] == 0 {
            self.defined.push(before as u32);
        }
        self.vals.push(v);
        Ok(())
    }

    fn function_block(&mut self) -> Result<()> {
        let func = *self.defined.get(self.next_body).ok_or_else(|| bad("more function bodies than defined functions"))?;
        self.next_body += 1;
        if !matches!(self.m.def(self.m.funcs[func as usize].ty), TypeDef::Func { ret: Ty::VOID, params } if params.is_empty()) {
            return unsupported("defined function that is not void()");
        }
        let module_values = self.vals.len();
        let mut f = FnReader { body: Body::new(), bb: 0, placeholder_ty: HashMap::new(), value_inst: Vec::new() };
        let mut attachments: Vec<Vec<u64>> = Vec::new();
        loop {
            match self.r.next()? {
                Entry::EndBlock => break,
                Entry::SubBlock(CONSTANTS_BLOCK) => {
                    self.r.enter(CONSTANTS_BLOCK)?;
                    self.constants_block()?;
                }
                Entry::SubBlock(METADATA_ATTACHMENT_BLOCK) => {
                    self.r.enter(METADATA_ATTACHMENT_BLOCK)?;
                    loop {
                        match self.r.next()? {
                            Entry::EndBlock => break,
                            Entry::Record(METADATA_ATTACHMENT) => attachments.push(self.r.ops.clone()),
                            Entry::Record(_) => {}
                            _ => return Err(bad("unexpected block in metadata attachments")),
                        }
                    }
                }
                Entry::SubBlock(VALUE_SYMTAB_BLOCK) => {
                    self.r.enter(VALUE_SYMTAB_BLOCK)?;
                    loop {
                        match self.r.next()? {
                            Entry::EndBlock => break,
                            Entry::Record(_) => self.notes.local_names += 1,
                            _ => return Err(bad("unexpected block in a symbol table")),
                        }
                    }
                }
                Entry::SubBlock(USELIST_BLOCK) => self.skip_block("use list")?,
                Entry::SubBlock(METADATA_BLOCK) => return unsupported("function-local metadata (debug info?)"),
                Entry::SubBlock(other) => return unsupported(format!("block {other} in a function")),
                Entry::Record(code) => self.instruction(&mut f, code)?,
                Entry::End => return Err(bad("function block does not end")),
            }
        }
        if f.bb as usize != f.body.blocks.len() {
            return Err(bad("function declares more blocks than it has"));
        }
        // forward references: placeholder → the instruction that got that value id
        let first = module_values + (self.vals.len() - module_values - f.value_inst.len());
        let value_inst = &f.value_inst;
        let fix = |v: &mut Value| -> Result<()> {
            if v.0 >> 30 == 0 && v.0 & PLACEHOLDER != 0 {
                let id = (v.0 & (PLACEHOLDER - 1)) as usize;
                let inst = id.checked_sub(first).and_then(|k| value_inst.get(k)).ok_or_else(|| bad("forward reference to a value the function never defines"))?;
                *v = Value::inst(*inst);
            }
            Ok(())
        };
        if !f.placeholder_ty.is_empty() {
            for inst in f.body.insts.iter_mut() {
                match inst.op {
                    Op::Bin(_) | Op::ICmp(_) | Op::FCmp(_) | Op::Store | Op::AtomicRmw(_) => {
                        fix(&mut inst.a)?;
                        fix(&mut inst.b)?;
                    }
                    Op::Cast(_) | Op::Load | Op::ExtractValue | Op::CondBr | Op::Switch | Op::Gep => fix(&mut inst.a)?,
                    Op::Select | Op::CmpXchg => {
                        fix(&mut inst.a)?;
                        fix(&mut inst.b)?;
                        fix(&mut inst.c)?;
                    }
                    _ => {}
                }
            }
            for v in f.body.operands.iter_mut() {
                fix(v)?;
            }
            for list in f.body.phis.iter_mut() {
                for (v, _) in list.iter_mut() {
                    fix(v)?;
                }
            }
        }
        for rec in attachments {
            if rec.len() % 2 == 0 {
                return unsupported("metadata attached to a function");
            }
            let inst = f.body.insts.get_mut(rec[0] as usize).ok_or_else(|| bad("attachment to an instruction that does not exist"))?;
            for pair in rec[1..].chunks(2) {
                let kind = self.kinds.get(&pair[0]).cloned().unwrap_or_else(|| format!("kind {}", pair[0]));
                if kind == "dx.precise" && matches!(inst.op, Op::Call | Op::Alloca) {
                    inst.flags |= flag::PRECISE;
                    continue;
                }
                if kind == "dx.controlflow.hints" && matches!(inst.op, Op::CondBr | Op::Switch) {
                    if let Some(set) = self.hints_of(pair[1]) {
                        inst.flags |= set;
                        continue;
                    }
                }
                if kind == "llvm.loop" && matches!(inst.op, Op::Br | Op::CondBr) && self.is_unroll_disable(pair[1]) {
                    inst.flags |= flag::LOOP_UNROLL_DISABLE;
                    continue;
                }
                match self.notes.dropped_attachments.iter_mut().find(|(k, _)| *k == kind) {
                    Some((_, n)) => *n += 1,
                    None => self.notes.dropped_attachments.push((kind, 1)),
                }
            }
        }
        if self.opts.keep_dead {
            for inst in f.body.insts.iter_mut() {
                inst.flags |= flag::EFFECT;
            }
        }
        self.m.funcs[func as usize].body = Some(f.body);
        self.vals.truncate(module_values);
        Ok(())
    }

    /// The type of a value of the function being read.
    fn value_ty(&mut self, f: &FnReader, v: Value) -> Result<Ty> {
        if v.0 >> 30 == 0 && v.0 & PLACEHOLDER != 0 {
            return f.placeholder_ty.get(&(v.0 & (PLACEHOLDER - 1))).copied().ok_or_else(|| bad("forward reference without a type"));
        }
        Ok(self.m.type_of(&f.body, v))
    }

    /// LLVM's getValueTypePair: a relative id, followed by a type when it is a forward
    /// reference.
    fn typed(&mut self, f: &mut FnReader, k: &mut usize) -> Result<(Value, Ty)> {
        let cur = self.vals.len() as u32;
        let id = cur.wrapping_sub(self.op(*k)? as u32);
        *k += 1;
        if id < cur {
            let v = self.vals[id as usize];
            return Ok((v, self.value_ty(f, v)?));
        }
        let ty = self.ty(self.op(*k)?)?;
        *k += 1;
        Ok((f.placeholder(id, ty)?, ty))
    }

    /// LLVM's getValue: a relative id whose type the record implies.
    fn untyped(&mut self, f: &mut FnReader, k: &mut usize, ty: Ty) -> Result<Value> {
        let cur = self.vals.len() as u32;
        let id = cur.wrapping_sub(self.op(*k)? as u32);
        *k += 1;
        if id < cur {
            return Ok(self.vals[id as usize]);
        }
        f.placeholder(id, ty)
    }

    fn block(&self, f: &FnReader, k: usize) -> Result<Value> {
        let b = self.op(k)?;
        if b as usize >= f.body.blocks.len() {
            return Err(bad("branch to a block the function does not declare"));
        }
        Ok(Value(b as u32))
    }

    fn pointee(&self, ptr: Ty) -> Result<(Ty, u32)> {
        match self.m.def(ptr) {
            TypeDef::Ptr { to, space } => Ok((*to, *space)),
            _ => Err(bad("address operand that is not a pointer")),
        }
    }

    /// Note the alignment of the instruction being read if it is not its type's.
    fn check_align(&mut self, field: u64, ty: Ty) {
        let align = if field == 0 { 0 } else { 1u32 << (field - 1).min(30) };
        if align != align_of(&self.m, ty) {
            self.notes.other_alignments += 1;
            self.other_align = Some(align);
        }
    }

    fn instruction(&mut self, f: &mut FnReader, code: u32) -> Result<()> {
        if code == FUNC_DECLAREBLOCKS {
            let n = self.op(0)?;
            if n == 0 || n > u32::MAX as u64 / 2 {
                return Err(bad("bad block count"));
            }
            for _ in 1..n {
                f.body.new_block();
            }
            return Ok(());
        }
        if f.bb as usize >= f.body.blocks.len() {
            return Err(bad("instruction after the last declared block"));
        }
        let n = self.r.ops.len();
        let none = Value::NONE;
        let mut k = 0usize;
        let inst = match code {
            FUNC_BINOP => {
                let (a, ty) = self.typed(f, &mut k)?;
                let b = self.untyped(f, &mut k, ty)?;
                let opc = self.op(k)?;
                let float = self.m.is_float(ty);
                let op = match (opc, float) {
                    (0, false) => BinOp::Add,
                    (0, true) => BinOp::FAdd,
                    (1, false) => BinOp::Sub,
                    (1, true) => BinOp::FSub,
                    (2, false) => BinOp::Mul,
                    (2, true) => BinOp::FMul,
                    (3, false) => BinOp::UDiv,
                    (4, false) => BinOp::SDiv,
                    (4, true) => BinOp::FDiv,
                    (5, false) => BinOp::URem,
                    (6, false) => BinOp::SRem,
                    (6, true) => BinOp::FRem,
                    (7, false) => BinOp::Shl,
                    (8, false) => BinOp::LShr,
                    (9, false) => BinOp::AShr,
                    (10, false) => BinOp::And,
                    (11, false) => BinOp::Or,
                    (12, false) => BinOp::Xor,
                    _ => return Err(bad(format!("binary opcode {opc}"))),
                };
                let raw = self.r.ops.get(k + 1).copied().unwrap_or(0);
                let flags = match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl => (if raw & 1 != 0 { flag::NUW } else { 0 }) | (if raw & 2 != 0 { flag::NSW } else { 0 }),
                    BinOp::UDiv | BinOp::SDiv | BinOp::LShr | BinOp::AShr => {
                        if raw & 1 != 0 {
                            flag::EXACT
                        } else {
                            0
                        }
                    }
                    _ if float => match raw {
                        0 => 0,
                        31 => flag::FAST,
                        _ => return unsupported("fast-math flags other than none or fast"),
                    },
                    _ => 0,
                };
                Inst { op: Op::Bin(op), flags, ty, a, b, c: none }
            }
            FUNC_CAST => {
                let (a, _) = self.typed(f, &mut k)?;
                let ty = self.ty(self.op(k)?)?;
                let op = match self.op(k + 1)? {
                    0 => CastOp::Trunc,
                    1 => CastOp::ZExt,
                    2 => CastOp::SExt,
                    3 => CastOp::FPToUI,
                    4 => CastOp::FPToSI,
                    5 => CastOp::UIToFP,
                    6 => CastOp::SIToFP,
                    7 => CastOp::FPTrunc,
                    8 => CastOp::FPExt,
                    11 => CastOp::BitCast,
                    other => return unsupported(format!("cast opcode {other}")),
                };
                Inst { op: Op::Cast(op), flags: 0, ty, a, b: none, c: none }
            }
            FUNC_CMP2 | 9 => {
                let (a, ty) = self.typed(f, &mut k)?;
                let b = self.untyped(f, &mut k, ty)?;
                let pred = self.op(k)?;
                let fast = match self.r.ops.get(k + 1) {
                    None | Some(0) => 0,
                    Some(31) => flag::FAST,
                    Some(_) => return unsupported("fast-math flags other than none or fast"),
                };
                let op = if self.m.is_float(ty) {
                    Op::FCmp(match pred {
                        1 => FPred::Oeq,
                        2 => FPred::Ogt,
                        3 => FPred::Oge,
                        4 => FPred::Olt,
                        5 => FPred::Ole,
                        6 => FPred::One,
                        7 => FPred::Ord,
                        8 => FPred::Uno,
                        9 => FPred::Ueq,
                        10 => FPred::Ugt,
                        11 => FPred::Uge,
                        12 => FPred::Ult,
                        13 => FPred::Ule,
                        14 => FPred::Une,
                        other => return unsupported(format!("float predicate {other}")),
                    })
                } else {
                    Op::ICmp(match pred {
                        32 => IPred::Eq,
                        33 => IPred::Ne,
                        34 => IPred::Ugt,
                        35 => IPred::Uge,
                        36 => IPred::Ult,
                        37 => IPred::Ule,
                        38 => IPred::Sgt,
                        39 => IPred::Sge,
                        40 => IPred::Slt,
                        41 => IPred::Sle,
                        other => return Err(bad(format!("integer predicate {other}"))),
                    })
                };
                Inst { op, flags: fast, ty: Ty::I1, a, b, c: none }
            }
            FUNC_VSELECT => {
                let (t, ty) = self.typed(f, &mut k)?;
                let e = self.untyped(f, &mut k, ty)?;
                let (c, _) = self.typed(f, &mut k)?;
                Inst { op: Op::Select, flags: 0, ty, a: c, b: t, c: e }
            }
            FUNC_EXTRACTVAL => {
                let (a, agg) = self.typed(f, &mut k)?;
                if n != k + 1 {
                    return unsupported("extractvalue with several indices");
                }
                let index = self.op(k)?;
                let ty = match self.m.def(agg) {
                    TypeDef::Struct { fields, .. } => fields.get(index as usize).copied(),
                    TypeDef::Array { of, .. } => Some(*of),
                    _ => None,
                }
                .ok_or_else(|| bad("extractvalue index out of the aggregate"))?;
                Inst { op: Op::ExtractValue, flags: 0, ty, a, b: Value(index as u32), c: none }
            }
            FUNC_CALL => {
                if n < 4 {
                    return Err(bad("short call"));
                }
                let cc = self.op(1)?;
                if cc & !(1 << 15 | 1) != 0 {
                    return unsupported("call with a calling convention");
                }
                match self.op(0)? {
                    0 => {}
                    // the IR has no call-site attributes (DXC repeats nounwind on its
                    // llvm.lifetime calls)
                    list if list as usize <= self.attr_lists.len() => self.notes.call_attributes += 1,
                    _ => return Err(bad("call with an attribute list that is not defined")),
                }
                if cc & 1 != 0 {
                    self.notes.tail_calls += 1;
                }
                k = 2;
                let mut explicit = None;
                if cc & (1 << 15) != 0 {
                    explicit = Some(self.ty(self.op(k)?)?);
                    k += 1;
                }
                let (callee, _) = self.typed(f, &mut k)?;
                let ValueKind::Func(fi) = callee.kind() else { return unsupported("indirect call") };
                let function = self.m.funcs.get(fi as usize).filter(|_| !callee.is_none()).ok_or_else(|| bad("call of a function that does not exist"))?;
                let fty = explicit.unwrap_or(function.ty);
                let effect = !matches!(function.attrs, FnAttrs::NounwindReadnone | FnAttrs::NounwindReadonly);
                let (ret, params) = match self.m.def(fty) {
                    TypeDef::Func { ret, params } => (*ret, params.clone()),
                    _ => return Err(bad("call through a type that is not a function type")),
                };
                if n != k + params.len() {
                    return Err(bad("call with the wrong number of arguments"));
                }
                let first = f.body.operands.len() as u32;
                for p in &params {
                    let v = self.untyped(f, &mut k, *p)?;
                    f.body.operands.push(v);
                }
                Inst { op: Op::Call, flags: if effect { flag::EFFECT } else { 0 }, ty: ret, a: callee, b: Value(first), c: Value(params.len() as u32) }
            }
            FUNC_LOAD => {
                let (a, ptr) = self.typed(f, &mut k)?;
                let ty = if n == k + 3 {
                    k += 1;
                    self.ty(self.op(k - 1)?)?
                } else {
                    self.pointee(ptr)?.0
                };
                if self.op(k + 1)? != 0 {
                    return unsupported("volatile load");
                }
                let align = self.op(k)?;
                self.check_align(align, ty);
                Inst { op: Op::Load, flags: 0, ty, a, b: none, c: none }
            }
            FUNC_STORE => {
                let (p, _) = self.typed(f, &mut k)?;
                let (v, ty) = self.typed(f, &mut k)?;
                if self.op(k + 1)? != 0 {
                    return unsupported("volatile store");
                }
                let align = self.op(k)?;
                self.check_align(align, ty);
                Inst { op: Op::Store, flags: flag::EFFECT, ty: Ty::VOID, a: p, b: v, c: none }
            }
            FUNC_GEP => {
                let inbounds = self.op(0)? == 1;
                let mut at = self.ty(self.op(1)?)?;
                k = 2;
                let (base, base_ty) = self.typed(f, &mut k)?;
                let space = self.pointee(base_ty)?.1;
                let first = f.body.operands.len() as u32;
                let mut count = 0u32;
                while k < n {
                    let (v, _) = self.typed(f, &mut k)?;
                    if count > 0 {
                        at = match self.m.def(at) {
                            TypeDef::Array { of, .. } | TypeDef::Vector { of, .. } => *of,
                            TypeDef::Struct { fields, .. } => {
                                let index = self.m.as_int(v).ok_or_else(|| bad("struct index that is not a constant"))?;
                                *fields.get(index as usize).ok_or_else(|| bad("struct index out of range"))?
                            }
                            _ => return Err(bad("getelementptr into a type that is not an aggregate")),
                        };
                    }
                    f.body.operands.push(v);
                    count += 1;
                }
                let ty = self.m.ptr_in(at, space);
                Inst { op: Op::Gep, flags: if inbounds { 0 } else { flag::NO_INBOUNDS }, ty, a: base, b: Value(first), c: Value(count) }
            }
            FUNC_ALLOCA => {
                if n != 4 {
                    return Err(bad("short alloca"));
                }
                let record = self.op(3)?;
                let mut allocated = self.ty(self.op(0)?)?;
                if record & (1 << 6) == 0 {
                    allocated = self.pointee(allocated)?.0;
                }
                let count = self.value(self.op(2)?)?;
                if self.m.as_int(count) != Some(1) || record & (1 << 5) != 0 {
                    return unsupported("alloca of a count or inalloca");
                }
                self.check_align(record & 31, allocated);
                let ty = self.m.ptr(allocated);
                Inst { op: Op::Alloca, flags: 0, ty, a: none, b: none, c: none }
            }
            FUNC_PHI => {
                if n % 2 != 1 {
                    return Err(bad("phi with an odd operand list"));
                }
                let ty = self.ty(self.op(0)?)?;
                let cur = self.vals.len() as u32;
                let mut list = Vec::with_capacity(n / 2);
                for p in (1..n).step_by(2) {
                    let id = (cur as i64 - decode_signed(self.op(p)?)) as u32;
                    let v = if id < cur { self.vals[id as usize] } else { f.placeholder(id, ty)? };
                    let from = self.block(f, p + 1)?;
                    list.push((v, Block(from.0)));
                }
                f.body.phis.push(list.into());
                Inst { op: Op::Phi, flags: 0, ty, a: Value(f.body.phis.len() as u32 - 1), b: none, c: none }
            }
            FUNC_ATOMICRMW => {
                let (p, ptr) = self.typed(f, &mut k)?;
                let ty = self.pointee(ptr)?.0;
                let v = self.untyped(f, &mut k, ty)?;
                let op = match self.op(k)? {
                    0 => RmwOp::Xchg,
                    1 => RmwOp::Add,
                    2 => RmwOp::Sub,
                    3 => RmwOp::And,
                    4 => RmwOp::Nand,
                    5 => RmwOp::Or,
                    6 => RmwOp::Xor,
                    7 => RmwOp::Max,
                    8 => RmwOp::Min,
                    9 => RmwOp::UMax,
                    10 => RmwOp::UMin,
                    other => return Err(bad(format!("atomicrmw operation {other}"))),
                };
                if self.op(k + 1)? != 0 || self.op(k + 2)? != 6 || self.op(k + 3)? != 1 {
                    return unsupported("atomicrmw that is volatile or not seq_cst");
                }
                Inst { op: Op::AtomicRmw(op), flags: flag::EFFECT, ty, a: p, b: v, c: none }
            }
            FUNC_CMPXCHG => {
                let (p, _) = self.typed(f, &mut k)?;
                let (e, ty) = self.typed(f, &mut k)?;
                let v = self.untyped(f, &mut k, ty)?;
                if self.op(k)? != 0 || self.op(k + 1)? != 6 || self.op(k + 2)? != 1 || self.r.ops.get(k + 3).is_some_and(|&o| o != 6) || self.r.ops.get(k + 4).is_some_and(|&o| o != 0) {
                    return unsupported("cmpxchg that is volatile, weak or not seq_cst");
                }
                let pair = self.m.literal_struct(vec![ty, Ty::I1]);
                Inst { op: Op::CmpXchg, flags: flag::EFFECT, ty: pair, a: p, b: e, c: v }
            }
            FUNC_BR => {
                if n == 1 {
                    Inst { op: Op::Br, flags: flag::EFFECT, ty: Ty::VOID, a: self.block(f, 0)?, b: none, c: none }
                } else {
                    let (t, e) = (self.block(f, 0)?, self.block(f, 1)?);
                    k = 2;
                    let c = self.untyped(f, &mut k, Ty::I1)?;
                    Inst { op: Op::CondBr, flags: flag::EFFECT, ty: Ty::VOID, a: c, b: t, c: e }
                }
            }
            FUNC_SWITCH => {
                if n < 3 || n % 2 != 1 || self.op(0)? >> 16 == 0x4B5 {
                    return unsupported("switch record of another layout");
                }
                let ty = self.ty(self.op(0)?)?;
                k = 1;
                let v = self.untyped(f, &mut k, ty)?;
                let default = self.block(f, 2)?;
                let mut cases = Vec::with_capacity(n / 2 - 1);
                for p in (3..n).step_by(2) {
                    let case = self.value(self.op(p)?)?;
                    let bits = self.m.as_int(case).ok_or_else(|| bad("switch case that is not an integer constant"))?;
                    cases.push((bits, Block(self.block(f, p + 1)?.0)));
                }
                f.body.switches.push(cases);
                Inst { op: Op::Switch, flags: flag::EFFECT, ty: Ty::VOID, a: v, b: default, c: Value(f.body.switches.len() as u32 - 1) }
            }
            FUNC_RET => {
                if n != 0 {
                    return unsupported("ret with a value");
                }
                Inst { op: Op::Ret, flags: flag::EFFECT, ty: Ty::VOID, a: none, b: none, c: none }
            }
            FUNC_UNREACHABLE => Inst { op: Op::Unreachable, flags: flag::EFFECT, ty: Ty::VOID, a: none, b: none, c: none },
            other => return unsupported(format!("instruction code {other}")),
        };
        let index = f.body.insts.len() as u32;
        if index >= PLACEHOLDER {
            return unsupported("function with too many instructions");
        }
        let block = &mut f.body.blocks[f.bb as usize];
        if inst.op == Op::Phi {
            if !block.insts.is_empty() {
                return Err(bad("phi after other instructions of its block"));
            }
            block.phis.push(index);
        } else {
            block.insts.push(index);
        }
        if inst.op.is_terminator() {
            f.bb += 1;
        }
        if inst.ty != Ty::VOID {
            self.vals.push(Value::inst(index));
            f.value_inst.push(index);
        }
        f.body.insts.push(inst);
        if let Some(align) = self.other_align.take() {
            f.body.aligns.insert(index, align);
        }
        Ok(())
    }

    fn module(&mut self) -> Result<()> {
        loop {
            match self.r.next()? {
                Entry::EndBlock => break,
                Entry::End => return Err(bad("module block does not end")),
                Entry::SubBlock(id) => match id {
                    0 => self.r.enter(0)?,
                    PARAMATTR_GROUP_BLOCK => {
                        self.r.enter(id)?;
                        self.attr_group_block()?;
                    }
                    PARAMATTR_BLOCK => {
                        self.r.enter(id)?;
                        self.attr_block()?;
                    }
                    TYPE_BLOCK_NEW => {
                        self.r.enter(id)?;
                        self.types_block()?;
                    }
                    CONSTANTS_BLOCK => {
                        self.r.enter(id)?;
                        self.constants_block()?;
                        for (g, init) in std::mem::take(&mut self.inits) {
                            self.m.globals[g].init = Some(self.value(init)?);
                        }
                    }
                    METADATA_BLOCK => {
                        self.r.enter(id)?;
                        self.metadata_block()?;
                    }
                    FUNCTION_BLOCK => {
                        self.r.enter(id)?;
                        self.function_block()?;
                    }
                    VALUE_SYMTAB_BLOCK => self.skip_block("symbol table (read in the first pass)")?,
                    USELIST_BLOCK => self.skip_block("use list")?,
                    other => return unsupported(format!("block {other} in the module")),
                },
                Entry::Record(code) => match code {
                    MODULE_VERSION => {
                        if self.op(0)? != 1 {
                            return unsupported("module version 0 (absolute operand ids)");
                        }
                    }
                    MODULE_TRIPLE | MODULE_DATALAYOUT => {
                        // the IR has one of each
                        let (text, ours) = (string(&self.r.ops), if code == MODULE_TRIPLE { crate::irprint::TRIPLE } else { crate::irprint::DATALAYOUT });
                        if text != ours {
                            return unsupported(format!("target {text} (a module compiled with -enable-16bit-types?)"));
                        }
                    }
                    MODULE_GLOBALVAR => self.global_var()?,
                    MODULE_FUNCTION => self.function()?,
                    other => return unsupported(format!("module record {other}")),
                },
            }
        }
        if !self.inits.is_empty() {
            return Err(bad("global initialiser without a constants block"));
        }
        if self.next_body != self.defined.len() {
            return Err(bad("defined function without a body"));
        }
        Ok(())
    }
}

struct FnReader {
    body: Body,
    /// the block instructions are being appended to
    bb: u32,
    /// forward-referenced value ids and the types they were referenced with
    placeholder_ty: HashMap<u32, Ty>,
    /// the instruction index of each value the function has defined so far
    value_inst: Vec<u32>,
}

impl FnReader {
    fn placeholder(&mut self, id: u32, ty: Ty) -> Result<Value> {
        if id >= PLACEHOLDER {
            return Err(bad("forward reference out of range"));
        }
        self.placeholder_ty.insert(id, ty);
        Ok(Value(PLACEHOLDER | id))
    }
}

/// The names of the module-level values, by value id (the symbol table comes after the
/// records it names).
fn symbol_names(bitcode: &[u8]) -> Result<HashMap<u64, String>> {
    let mut r = Reader::new(bitcode);
    let mut names = HashMap::new();
    r.read(32)?;
    match r.next()? {
        Entry::SubBlock(MODULE_BLOCK) => r.enter(MODULE_BLOCK)?,
        _ => return Err(bad("no module block")),
    }
    loop {
        match r.next()? {
            Entry::EndBlock | Entry::End => return Ok(names),
            Entry::SubBlock(0) => r.enter(0)?,
            Entry::SubBlock(VALUE_SYMTAB_BLOCK) => {
                r.enter(VALUE_SYMTAB_BLOCK)?;
                loop {
                    match r.next()? {
                        Entry::EndBlock => break,
                        Entry::Record(VST_ENTRY) if !r.ops.is_empty() => {
                            names.insert(r.ops[0], string(&r.ops[1..]));
                        }
                        Entry::Record(_) => {}
                        _ => return Err(bad("unexpected block in the symbol table")),
                    }
                }
            }
            Entry::SubBlock(_) => r.skip()?,
            Entry::Record(_) => {}
        }
    }
}

/// Read a module, keeping DXC's dead code dead (the writer will drop it).
pub fn read(bitcode: &[u8]) -> Result<(Module, Notes)> {
    read_with(bitcode, Options::default())
}

pub fn read_with(bitcode: &[u8], opts: Options) -> Result<(Module, Notes)> {
    let names = symbol_names(bitcode)?;
    let mut rd = Rd {
        r: Reader::new(bitcode),
        m: Module::new(),
        opts,
        notes: Notes::default(),
        types: Vec::new(),
        struct_name: None,
        attr_groups: HashMap::new(),
        attr_lists: Vec::new(),
        names,
        vals: Vec::new(),
        inits: Vec::new(),
        defined: Vec::new(),
        next_body: 0,
        raw: Vec::new(),
        mds: Vec::new(),
        other_align: None,
        kinds: HashMap::new(),
    };
    if rd.r.read(32)? != 0xdec0_4342 {
        return Err(bad("not LLVM bitcode"));
    }
    match rd.r.next()? {
        Entry::SubBlock(MODULE_BLOCK) => rd.r.enter(MODULE_BLOCK)?,
        _ => return Err(bad("no module block")),
    }
    rd.module()?;
    Ok((rd.m, rd.notes))
}
