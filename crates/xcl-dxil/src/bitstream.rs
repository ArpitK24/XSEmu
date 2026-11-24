//! LLVM bitstream container: the bit-level writer [`crate::bcwriter`] emits through and the
//! reader [`crate::bcreader`] parses with (owner: the bitcode subagent).
//!
//! The format is LLVM 3.7's (`Bitcode/BitstreamWriter.h`, `BitstreamReader.h`): a stream of
//! little-endian 32-bit words read from the low bit up, nested blocks that carry their length
//! in words, records of VBR6 operands, and abbreviations that give a record a fixed layout.

use crate::{Error, Result};
use std::collections::HashMap;
use std::rc::Rc;

/// Abbreviation ids every block has.
pub const END_BLOCK: u32 = 0;
pub const ENTER_SUBBLOCK: u32 = 1;
pub const DEFINE_ABBREV: u32 = 2;
pub const UNABBREV_RECORD: u32 = 3;
pub const FIRST_APPLICATION_ABBREV: u32 = 4;

pub const BLOCKINFO_BLOCK_ID: u32 = 0;
const BLOCKINFO_CODE_SETBID: u32 = 1;

/// One operand of an abbreviation.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AbbrevOp {
    Literal(u64),
    Fixed(u32),
    Vbr(u32),
    /// the next operand is the element encoding
    Array,
    Char6,
    Blob,
}

pub fn is_char6(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'.' || c == b'_'
}

fn encode_char6(c: u8) -> u32 {
    match c {
        b'a'..=b'z' => (c - b'a') as u32,
        b'A'..=b'Z' => (c - b'A') as u32 + 26,
        b'0'..=b'9' => (c - b'0') as u32 + 52,
        b'.' => 62,
        _ => 63,
    }
}

fn decode_char6(v: u64) -> u64 {
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._"[(v & 63) as usize] as u64
}

/// The bit writer. Hot records are emitted field by field ([`Writer::emit`], [`Writer::vbr`])
/// by code that knows the abbreviation; [`Writer::record`] and [`Writer::abbreviated`] are the
/// general forms.
pub struct Writer {
    out: Vec<u8>,
    /// where the stream starts in `out`
    base: usize,
    cur: u64,
    nbits: u32,
    abbrev_width: u32,
    /// (abbreviation width of the parent, byte offset of this block's length word)
    blocks: Vec<(u32, usize)>,
    /// the block the last SETBID in BLOCKINFO named
    blockinfo_bid: u32,
}

impl Writer {
    /// `capacity`: expected size in bytes.
    pub fn new(capacity: usize) -> Writer {
        Writer::appending(Vec::new(), capacity)
    }

    /// A writer whose stream starts at the end of `out` (what [`Writer::finish`] returns then
    /// starts with those bytes); `capacity`: expected size of the stream in bytes.
    pub fn appending(mut out: Vec<u8>, capacity: usize) -> Writer {
        out.reserve_exact(capacity);
        Writer { base: out.len(), out, cur: 0, nbits: 0, abbrev_width: 2, blocks: Vec::new(), blockinfo_bid: u32::MAX }
    }

    /// `width` bits of `v` (at most 32; `v` must fit).
    #[inline(always)]
    pub fn emit(&mut self, v: u32, width: u32) {
        debug_assert!(width <= 32 && (width == 32 || v >> width == 0), "{v} in {width} bits");
        self.cur |= (v as u64) << self.nbits;
        self.nbits += width;
        if self.nbits >= 32 {
            self.out.extend_from_slice(&(self.cur as u32).to_le_bytes());
            self.cur >>= 32;
            self.nbits -= 32;
        }
    }

    /// VBR6, the encoding of every operand of an unabbreviated record: up to five chunks
    /// (values below 2^25, which is every value id of a module of 30 million values) go out
    /// as one field, built with the continuation bits in place.
    #[inline(always)]
    fn vbr6(&mut self, v: u32) {
        const MORE: u32 = 32;
        if v < 1 << 5 {
            self.emit(v, 6);
        } else if v < 1 << 10 {
            self.emit((v & 31) | MORE | (v >> 5) << 6, 12);
        } else if v < 1 << 15 {
            self.emit((v & 31) | MORE | ((v >> 5 & 31) | MORE) << 6 | (v >> 10) << 12, 18);
        } else if v < 1 << 20 {
            self.emit((v & 31) | MORE | ((v >> 5 & 31) | MORE) << 6 | ((v >> 10 & 31) | MORE) << 12 | (v >> 15) << 18, 24);
        } else if v < 1 << 25 {
            self.emit((v & 31) | MORE | ((v >> 5 & 31) | MORE) << 6 | ((v >> 10 & 31) | MORE) << 12 | ((v >> 15 & 31) | MORE) << 18 | (v >> 20) << 24, 30);
        } else {
            self.vbr_chunks(v, 6);
        }
    }

    #[inline(always)]
    pub fn vbr(&mut self, v: u32, n: u32) {
        if n == 6 {
            self.vbr6(v);
        } else {
            self.vbr_chunks(v, n);
        }
    }

    #[inline(always)]
    fn vbr_chunks(&mut self, mut v: u32, n: u32) {
        let hi = 1u32 << (n - 1);
        while v >= hi {
            self.emit((v & (hi - 1)) | hi, n);
            v >>= n - 1;
        }
        self.emit(v, n);
    }

    #[inline(always)]
    pub fn vbr64(&mut self, mut v: u64, n: u32) {
        if v >> 32 == 0 {
            return self.vbr(v as u32, n);
        }
        let hi = 1u64 << (n - 1);
        while v >= hi {
            self.emit(((v & (hi - 1)) | hi) as u32, n);
            v >>= n - 1;
        }
        self.emit(v as u32, n);
    }

    /// An abbreviation id in the current block's width.
    #[inline(always)]
    pub fn code(&mut self, abbrev_id: u32) {
        self.emit(abbrev_id, self.abbrev_width);
    }

    fn align32(&mut self) {
        if self.nbits != 0 {
            self.out.extend_from_slice(&(self.cur as u32).to_le_bytes());
            self.cur = 0;
            self.nbits = 0;
        }
    }

    pub fn enter_block(&mut self, id: u32, abbrev_width: u32) {
        self.code(ENTER_SUBBLOCK);
        self.vbr(id, 8);
        self.vbr(abbrev_width, 4);
        self.align32();
        self.blocks.push((self.abbrev_width, self.out.len()));
        self.out.extend_from_slice(&[0; 4]);
        self.abbrev_width = abbrev_width;
    }

    pub fn exit_block(&mut self) {
        self.code(END_BLOCK);
        self.align32();
        let Some((width, at)) = self.blocks.pop() else { return };
        let words = ((self.out.len() - at - 4) / 4) as u32;
        self.out[at..at + 4].copy_from_slice(&words.to_le_bytes());
        self.abbrev_width = width;
    }

    /// An unabbreviated record.
    pub fn record(&mut self, code: u32, ops: &[u64]) {
        self.code(UNABBREV_RECORD);
        self.vbr(code, 6);
        self.vbr(ops.len() as u32, 6);
        for &op in ops {
            self.vbr64(op, 6);
        }
    }

    /// The head of an unabbreviated record of `n` operands; the caller emits each as VBR6.
    #[inline(always)]
    pub fn record_head(&mut self, code: u32, n: u32) {
        if code < 1 << 10 && n < 32 && self.abbrev_width <= 14 {
            // one field: the abbreviation id, the code in one or two chunks, the count
            let (code, bits) = if code < 32 { (code, 6) } else { ((code & 31) | 32 | (code >> 5) << 6, 12) };
            let w = self.abbrev_width;
            self.emit(UNABBREV_RECORD | code << w | n << (w + bits), w + bits + 6);
        } else {
            self.code(UNABBREV_RECORD);
            self.vbr(code, 6);
            self.vbr(n, 6);
        }
    }

    /// DEFINE_ABBREV in the current block. The id it gets is the caller's to count: the
    /// block's BLOCKINFO abbreviations first, then the block's own, from 4.
    pub fn define_abbrev(&mut self, ops: &[AbbrevOp]) {
        self.code(DEFINE_ABBREV);
        self.vbr(ops.len() as u32, 5);
        for op in ops {
            match *op {
                AbbrevOp::Literal(v) => {
                    self.emit(1, 1);
                    self.vbr64(v, 8);
                }
                AbbrevOp::Fixed(w) => {
                    self.emit(0, 1);
                    self.emit(1, 3);
                    self.vbr(w, 5);
                }
                AbbrevOp::Vbr(w) => {
                    self.emit(0, 1);
                    self.emit(2, 3);
                    self.vbr(w, 5);
                }
                AbbrevOp::Array => {
                    self.emit(0, 1);
                    self.emit(3, 3);
                }
                AbbrevOp::Char6 => {
                    self.emit(0, 1);
                    self.emit(4, 3);
                }
                AbbrevOp::Blob => {
                    self.emit(0, 1);
                    self.emit(5, 3);
                }
            }
        }
    }

    /// Start the BLOCKINFO block (abbreviation width 2, as LLVM's writer).
    pub fn enter_blockinfo(&mut self) {
        self.enter_block(BLOCKINFO_BLOCK_ID, 2);
        self.blockinfo_bid = u32::MAX;
    }

    /// Inside BLOCKINFO: an abbreviation every block `block_id` starts with.
    pub fn blockinfo_abbrev(&mut self, block_id: u32, ops: &[AbbrevOp]) {
        if self.blockinfo_bid != block_id {
            self.record(BLOCKINFO_CODE_SETBID, &[block_id as u64]);
            self.blockinfo_bid = block_id;
        }
        self.define_abbrev(ops);
    }

    fn scalar(&mut self, op: AbbrevOp, v: u64) {
        match op {
            AbbrevOp::Fixed(w) => {
                if w > 32 {
                    self.emit(v as u32, 32);
                    self.emit((v >> 32) as u32, w - 32);
                } else {
                    self.emit(v as u32, w);
                }
            }
            AbbrevOp::Vbr(w) => self.vbr64(v, w),
            AbbrevOp::Char6 => self.emit(encode_char6(v as u8), 6),
            AbbrevOp::Literal(_) | AbbrevOp::Array | AbbrevOp::Blob => {}
        }
    }

    /// A record through abbreviation `id` whose layout is `abbrev`; `vals` starts with the
    /// record code, as in LLVM's `EmitRecordWithAbbrev`. Blobs are not supported.
    pub fn abbreviated(&mut self, id: u32, abbrev: &[AbbrevOp], vals: &[u64]) {
        self.code(id);
        let mut k = 0;
        let mut i = 0;
        while i < abbrev.len() {
            match abbrev[i] {
                AbbrevOp::Literal(_) => k += 1,
                AbbrevOp::Array => {
                    let elt = abbrev.get(i + 1).copied().unwrap_or(AbbrevOp::Vbr(6));
                    let rest = vals.get(k..).unwrap_or(&[]);
                    self.vbr(rest.len() as u32, 6);
                    for &v in rest {
                        self.scalar(elt, v);
                    }
                    k = vals.len();
                    i += 1;
                }
                op => {
                    self.scalar(op, vals.get(k).copied().unwrap_or(0));
                    k += 1;
                }
            }
            i += 1;
        }
    }

    /// Bits written so far.
    pub fn bit_len(&self) -> u64 {
        (self.out.len() - self.base) as u64 * 8 + self.nbits as u64
    }

    /// The finished stream (padded to a 32-bit word).
    pub fn finish(mut self) -> Vec<u8> {
        self.align32();
        self.out
    }
}

fn bad(msg: impl Into<String>) -> Error {
    Error::Internal(format!("bitstream: {}", msg.into()))
}

/// What [`Reader::next`] found.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Entry {
    /// the current block ended (the reader is back in its parent)
    EndBlock,
    /// a sub-block starts: [`Reader::enter`] or [`Reader::skip`] it
    SubBlock(u32),
    /// a record: its code; the operands are in [`Reader::ops`]
    Record(u32),
    /// the stream ended at the top level
    End,
}

struct Scope {
    abbrev_width: u32,
    abbrevs: Vec<Rc<Vec<AbbrevOp>>>,
    block_id: u32,
    /// bit position of the end of the block
    end: u64,
}

/// A pull reader over a bitstream: records with abbreviations expanded, BLOCKINFO applied.
pub struct Reader<'a> {
    data: &'a [u8],
    pos: u64,
    scopes: Vec<Scope>,
    blockinfo: HashMap<u32, Vec<Rc<Vec<AbbrevOp>>>>,
    /// operands of the last record
    pub ops: Vec<u64>,
    /// abbreviation id the last record was written with
    pub abbrev_id: u32,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Reader<'a> {
        let top = Scope { abbrev_width: 2, abbrevs: Vec::new(), block_id: u32::MAX, end: data.len() as u64 * 8 };
        Reader { data, pos: 0, scopes: vec![top], blockinfo: HashMap::new(), ops: Vec::new(), abbrev_id: 0 }
    }

    pub fn bit_pos(&self) -> u64 {
        self.pos
    }

    pub fn read(&mut self, width: u32) -> Result<u64> {
        if width == 0 {
            return Ok(0);
        }
        if width > 64 || self.pos + width as u64 > self.data.len() as u64 * 8 {
            return Err(bad("read past the end"));
        }
        let mut v = 0u64;
        let mut got = 0;
        while got < width {
            let byte = self.data[(self.pos >> 3) as usize] as u64;
            let off = (self.pos & 7) as u32;
            let take = (8 - off).min(width - got);
            v |= ((byte >> off) & ((1u64 << take) - 1)) << got;
            got += take;
            self.pos += take as u64;
        }
        Ok(v)
    }

    pub fn vbr(&mut self, n: u32) -> Result<u64> {
        if !(2..=32).contains(&n) {
            return Err(bad("bad VBR width"));
        }
        let hi = 1u64 << (n - 1);
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let piece = self.read(n)?;
            if shift < 64 {
                v |= (piece & (hi - 1)) << shift;
            }
            if piece & hi == 0 {
                return Ok(v);
            }
            shift += n - 1;
            if shift > 70 {
                return Err(bad("VBR too long"));
            }
        }
    }

    fn align32(&mut self) {
        self.pos = self.pos.div_ceil(32) * 32;
    }

    fn read_abbrev(&mut self) -> Result<Vec<AbbrevOp>> {
        let n = self.vbr(5)?;
        let mut ops = Vec::new();
        let mut i = 0;
        while i < n {
            i += 1;
            if self.read(1)? == 1 {
                ops.push(AbbrevOp::Literal(self.vbr(8)?));
                continue;
            }
            ops.push(match self.read(3)? {
                1 => match self.vbr(5)? as u32 {
                    0 => AbbrevOp::Literal(0),
                    w => AbbrevOp::Fixed(w),
                },
                2 => match self.vbr(5)? as u32 {
                    0 => AbbrevOp::Literal(0),
                    w => AbbrevOp::Vbr(w),
                },
                3 => AbbrevOp::Array,
                4 => AbbrevOp::Char6,
                5 => AbbrevOp::Blob,
                e => return Err(bad(format!("abbreviation encoding {e}"))),
            });
        }
        Ok(ops)
    }

    fn scalar(&mut self, op: AbbrevOp) -> Result<u64> {
        match op {
            AbbrevOp::Literal(v) => Ok(v),
            AbbrevOp::Fixed(w) => self.read(w),
            AbbrevOp::Vbr(w) => self.vbr(w),
            AbbrevOp::Char6 => Ok(decode_char6(self.read(6)?)),
            AbbrevOp::Array | AbbrevOp::Blob => Err(bad("nested array in an abbreviation")),
        }
    }

    /// The block the reader is in (`u32::MAX` at the top level).
    pub fn block_id(&self) -> u32 {
        self.scopes.last().map_or(u32::MAX, |s| s.block_id)
    }

    /// The next thing in the current block. DEFINE_ABBREV is consumed here.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Entry> {
        loop {
            let width = self.scopes.last().map_or(2, |s| s.abbrev_width);
            if self.scopes.len() == 1 && self.pos + width as u64 > self.data.len() as u64 * 8 {
                return Ok(Entry::End);
            }
            let id = self.read(width)? as u32;
            match id {
                END_BLOCK => {
                    if self.scopes.len() <= 1 {
                        return Err(bad("END_BLOCK at the top level"));
                    }
                    self.align32();
                    let scope = self.scopes.pop();
                    if scope.is_some_and(|s| s.end != self.pos) {
                        return Err(bad("block length does not match its END_BLOCK"));
                    }
                    return Ok(Entry::EndBlock);
                }
                ENTER_SUBBLOCK => return Ok(Entry::SubBlock(self.vbr(8)? as u32)),
                DEFINE_ABBREV => {
                    let a = Rc::new(self.read_abbrev()?);
                    if let Some(s) = self.scopes.last_mut() {
                        s.abbrevs.push(a);
                    }
                }
                UNABBREV_RECORD => {
                    let code = self.vbr(6)? as u32;
                    let n = self.vbr(6)?;
                    self.ops.clear();
                    for _ in 0..n {
                        let v = self.vbr(6)?;
                        self.ops.push(v);
                    }
                    self.abbrev_id = id;
                    return Ok(Entry::Record(code));
                }
                _ => {
                    let abbrev = self.scopes.last().and_then(|s| s.abbrevs.get((id - FIRST_APPLICATION_ABBREV) as usize)).cloned().ok_or_else(|| bad(format!("unknown abbreviation {id}")))?;
                    self.ops.clear();
                    let mut i = 0;
                    while i < abbrev.len() {
                        match abbrev[i] {
                            AbbrevOp::Array => {
                                let elt = *abbrev.get(i + 1).ok_or_else(|| bad("array without an element encoding"))?;
                                let n = self.vbr(6)?;
                                for _ in 0..n {
                                    let v = self.scalar(elt)?;
                                    self.ops.push(v);
                                }
                                i += 1;
                            }
                            AbbrevOp::Blob => {
                                let n = self.vbr(6)?;
                                self.align32();
                                for _ in 0..n {
                                    let v = self.read(8)?;
                                    self.ops.push(v);
                                }
                                self.align32();
                            }
                            op => {
                                let v = self.scalar(op)?;
                                self.ops.push(v);
                            }
                        }
                        i += 1;
                    }
                    if self.ops.is_empty() {
                        return Err(bad("abbreviation without a record code"));
                    }
                    let code = self.ops.remove(0) as u32;
                    self.abbrev_id = id;
                    return Ok(Entry::Record(code));
                }
            }
        }
    }

    /// Enter the sub-block [`Reader::next`] just announced. BLOCKINFO is parsed whole here
    /// (the reader stays in the parent).
    pub fn enter(&mut self, block_id: u32) -> Result<()> {
        let abbrev_width = self.vbr(4)? as u32;
        self.align32();
        let words = self.read(32)?;
        let end = self.pos + words * 32;
        if end > self.data.len() as u64 * 8 || abbrev_width == 0 {
            return Err(bad("bad block header"));
        }
        let abbrevs = self.blockinfo.get(&block_id).cloned().unwrap_or_default();
        self.scopes.push(Scope { abbrev_width, abbrevs, block_id, end });
        if block_id == BLOCKINFO_BLOCK_ID {
            self.parse_blockinfo()?;
        }
        Ok(())
    }

    fn parse_blockinfo(&mut self) -> Result<()> {
        let mut target: Option<u32> = None;
        loop {
            let width = self.scopes.last().map_or(2, |s| s.abbrev_width);
            match self.read(width)? as u32 {
                END_BLOCK => {
                    self.align32();
                    self.scopes.pop();
                    return Ok(());
                }
                ENTER_SUBBLOCK => return Err(bad("block inside BLOCKINFO")),
                DEFINE_ABBREV => {
                    let a = Rc::new(self.read_abbrev()?);
                    let t = target.ok_or_else(|| bad("abbreviation before SETBID"))?;
                    self.blockinfo.entry(t).or_default().push(a);
                }
                UNABBREV_RECORD => {
                    let code = self.vbr(6)? as u32;
                    let n = self.vbr(6)?;
                    let mut first = 0;
                    for k in 0..n {
                        let v = self.vbr(6)?;
                        if k == 0 {
                            first = v;
                        }
                    }
                    if code == BLOCKINFO_CODE_SETBID {
                        target = Some(first as u32);
                    }
                }
                id => return Err(bad(format!("abbreviated record {id} in BLOCKINFO"))),
            }
        }
    }

    /// Skip the sub-block [`Reader::next`] just announced.
    pub fn skip(&mut self) -> Result<()> {
        self.vbr(4)?;
        self.align32();
        let words = self.read(32)?;
        let end = self.pos + words * 32;
        if end > self.data.len() as u64 * 8 {
            return Err(bad("bad block header"));
        }
        self.pos = end;
        Ok(())
    }
}

/// Every block and record of a stream as text (`<block 12> { [code: ops...] ... }`), with the
/// abbreviation id each record used; for looking at what a writer produced.
pub fn dump(data: &[u8], max_ops: usize) -> Result<String> {
    use std::fmt::Write;
    let mut r = Reader::new(data);
    let mut out = String::new();
    let magic = r.read(32)?;
    let _ = writeln!(out, "magic {magic:#010x}");
    let mut depth = 0usize;
    loop {
        match r.next()? {
            Entry::End => break,
            Entry::EndBlock => {
                depth = depth.saturating_sub(1);
                let _ = writeln!(out, "{}}}", "  ".repeat(depth));
            }
            Entry::SubBlock(id) => {
                let at = r.bit_pos();
                r.enter(id)?;
                if id == BLOCKINFO_BLOCK_ID {
                    let _ = writeln!(out, "{}<blockinfo> {} bits", "  ".repeat(depth), r.bit_pos() - at);
                    let mut ids: Vec<_> = r.blockinfo.iter().collect();
                    ids.sort_by_key(|(k, _)| **k);
                    for (bid, abbrevs) in ids {
                        for (k, a) in abbrevs.iter().enumerate() {
                            let _ = writeln!(out, "{}  block {bid} abbrev {}: {:?}", "  ".repeat(depth), k + 4, a);
                        }
                    }
                    continue;
                }
                let s = r.scopes.last().map(|s| (s.abbrev_width, (s.end - r.bit_pos()) / 8)).unwrap_or((0, 0));
                let _ = writeln!(out, "{}<block {id}> width {} bytes {} {{", "  ".repeat(depth), s.0, s.1);
                depth += 1;
            }
            Entry::Record(code) => {
                let _ = write!(out, "{}[{code}@{}:", "  ".repeat(depth), r.abbrev_id);
                for (k, v) in r.ops.iter().enumerate() {
                    if k == max_ops {
                        let _ = write!(out, " ...({})", r.ops.len());
                        break;
                    }
                    let _ = write!(out, " {v}");
                }
                out.push_str("]\n");
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read() {
        let mut w = Writer::new(64);
        w.emit(0xdec04342, 32);
        w.enter_blockinfo();
        let abbrev = [AbbrevOp::Literal(7), AbbrevOp::Fixed(3), AbbrevOp::Vbr(6), AbbrevOp::Array, AbbrevOp::Char6];
        w.blockinfo_abbrev(9, &abbrev);
        w.exit_block();
        w.enter_block(9, 4);
        w.record(1, &[1, 2, 300, u64::MAX]);
        w.abbreviated(4, &abbrev, &[7, 5, 1000, b'a' as u64, b'Z' as u64, b'_' as u64]);
        let local = [AbbrevOp::Literal(2), AbbrevOp::Array, AbbrevOp::Fixed(8)];
        w.define_abbrev(&local);
        w.abbreviated(5, &local, &[2, 0, 255, 17]);
        w.enter_block(10, 3);
        w.record(3, &[]);
        w.exit_block();
        w.exit_block();
        let bytes = w.finish();
        assert_eq!(bytes.len() % 4, 0);
        let mut r = Reader::new(&bytes);
        assert_eq!(r.read(32).unwrap(), 0xdec04342);
        assert_eq!(r.next().unwrap(), Entry::SubBlock(0));
        r.enter(0).unwrap();
        assert_eq!(r.next().unwrap(), Entry::SubBlock(9));
        r.enter(9).unwrap();
        assert_eq!(r.next().unwrap(), Entry::Record(1));
        assert_eq!(r.ops, [1, 2, 300, u64::MAX]);
        assert_eq!(r.next().unwrap(), Entry::Record(7));
        assert_eq!((r.abbrev_id, r.ops.as_slice()), (4, &[5, 1000, b'a' as u64, b'Z' as u64, b'_' as u64][..]));
        assert_eq!(r.next().unwrap(), Entry::Record(2));
        assert_eq!(r.ops, [0, 255, 17]);
        assert_eq!(r.next().unwrap(), Entry::SubBlock(10));
        r.enter(10).unwrap();
        assert_eq!(r.next().unwrap(), Entry::Record(3));
        assert_eq!(r.next().unwrap(), Entry::EndBlock);
        assert_eq!(r.next().unwrap(), Entry::EndBlock);
        assert_eq!(r.next().unwrap(), Entry::End);
        assert!(dump(&bytes, 8).unwrap().contains("[7@4: 5 1000 97 90 95]"));
    }

    #[test]
    fn truncated_input_is_an_error() {
        let mut w = Writer::new(64);
        w.enter_block(8, 3);
        w.record(1, &[1; 40]);
        w.exit_block();
        let bytes = w.finish();
        for cut in 0..bytes.len() {
            let mut r = Reader::new(&bytes[..cut]);
            let mut steps = 0;
            loop {
                steps += 1;
                match r.next() {
                    Ok(Entry::SubBlock(id)) => {
                        if r.enter(id).is_err() {
                            break;
                        }
                    }
                    Ok(Entry::End) | Err(_) => break,
                    Ok(_) => {}
                }
                assert!(steps < 100);
            }
        }
    }
}
