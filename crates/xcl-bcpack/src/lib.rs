//! BCPack: the console's lossless packing of BC1–BC7 texture blocks, which its DirectStorage
//! hardware unpacks on the way into memory. This crate only unpacks. BC2 is not supported
//! (no known title data uses it).
//!
//! A packed stream has no header and carries no sizes; the destination length gives the
//! block count. The stream is split over 12 or 16 interleaved rANS lanes ([`lanes`]). Each
//! lane is a run of fragments (a header byte picking one of 128 static models, a length,
//! then the rANS payload), and a fixed schedule decides which lane receives the next input
//! byte. Every lane contributes one or two symbols to each block's record. A record holds two
//! back references, one for the endpoints and one for the indices, each pointing at one of
//! the last 256 blocks or at a per-format dictionary entry, and the residuals to add to them.
//!
//! The models, rank tables, BC7 control words and dictionaries are constants of the format.
//! They are not part of this crate: [`Tables::load`] reads them from a 77,982-byte file.

mod bc6;
mod bc7;
mod lanes;
mod plain;
mod shared;
mod tables;
#[cfg(test)]
mod tests;

pub use tables::{Tables, TABLES_BYTES};

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The request's BcpackMode is not BC1, BC3, BC4, BC5, BC6H or BC7.
    UnsupportedMode(u8),
    /// The destination is empty or not a whole number of blocks.
    DestinationSize(usize),
    /// The lanes ran out of input, or the stream is corrupt in a way that stops them.
    Truncated,
    /// A BC6H record names a mode outside 1–14.
    Bc6Mode(u8),
    /// A BC7 record selects an unused control word.
    Bc7Control(usize),
    /// The table file does not have [`TABLES_BYTES`] bytes.
    TableSize(usize),
    /// A model in the table file has an empty bucket or does not sum to 65536.
    TableModel(usize),
    /// The table file could not be read.
    TableRead(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnsupportedMode(m) => write!(f, "BCPack mode {m} is not supported"),
            Error::DestinationSize(n) => write!(f, "destination of {n} bytes is not a whole number of blocks"),
            Error::Truncated => write!(f, "the stream ends early or is corrupt"),
            Error::Bc6Mode(m) => write!(f, "BC6H mode {m} is out of range"),
            Error::Bc7Control(c) => write!(f, "BC7 control code {c} is unused"),
            Error::TableSize(n) => write!(f, "table file has {n} bytes, expected {TABLES_BYTES}"),
            Error::TableModel(m) => write!(f, "table model {m} is malformed"),
            Error::TableRead(e) => write!(f, "table file: {e}"),
        }
    }
}

impl std::error::Error for Error {}

/// The block formats BCPack carries, with the lane layout of each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Format {
    Bc1,
    Bc3,
    Bc4,
    Bc5,
    Bc6h,
    Bc7,
}

impl Format {
    fn from_mode(mode: u8) -> Result<Format, Error> {
        Ok(match mode {
            1 => Format::Bc1,
            3 => Format::Bc3,
            4 => Format::Bc4,
            5 => Format::Bc5,
            6 => Format::Bc6h,
            7 => Format::Bc7,
            _ => return Err(Error::UnsupportedMode(mode)),
        })
    }

    fn block_bytes(self) -> usize {
        match self {
            Format::Bc1 | Format::Bc4 => 8,
            _ => 16,
        }
    }

    /// Per lane, the symbol width in bits minus one at the lane's full alphabet.
    pub(crate) fn lane_widths(self) -> &'static [u8] {
        match self {
            Format::Bc1 => &[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7],
            Format::Bc3 => &[3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7],
            Format::Bc4 => &[7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5],
            Format::Bc5 => &[3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5],
            Format::Bc6h => &[3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4],
            Format::Bc7 => &[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4],
        }
    }

    /// Symbols each lane contributes to one block.
    pub(crate) fn symbols_per_lane(self) -> usize {
        match self {
            Format::Bc1 | Format::Bc4 => 1,
            _ => 2,
        }
    }

    /// Symbols in one block's record: lane `j` owns `j * per .. (j + 1) * per`.
    pub(crate) fn record_len(self) -> usize {
        self.lane_widths().len() * self.symbols_per_lane()
    }
}

/// Unpacks `src` into `dst` for BcpackMode `mode` (1 and 3–7 for BC1 and BC3–BC7). The
/// length of `dst` gives the block count. Input past what the lanes need is ignored; input
/// that runs out first is an error. On error `dst` holds partial output.
pub fn decode(mode: u8, src: &[u8], dst: &mut [u8], tables: &Tables) -> Result<(), Error> {
    let format = Format::from_mode(mode)?;
    let block = format.block_bytes();
    if dst.is_empty() || !dst.len().is_multiple_of(block) {
        return Err(Error::DestinationSize(dst.len()));
    }
    let blocks = dst.len() / block;
    let mut records = vec![0u8; blocks * format.record_len()];
    let mut input = src.iter().copied();
    lanes::decode(tables, format, blocks, |_| input.next(), &mut records)?;
    let records = records.chunks_exact(format.record_len());
    let out = dst.chunks_exact_mut(block);
    match format {
        Format::Bc6h => bc6::unpack(records, out, tables),
        Format::Bc7 => bc7::unpack(records, out, tables),
        plain => plain::unpack(plain, records, out, tables),
    }
}
