//! BC7. A record selects a control word (mode, rotation, index mode, subsets, partition)
//! from the table's 512, then carries endpoint residuals per subset and index residuals.
//!
//! Control word bits: 0–2 mode, 3–4 rotation, 5 index selection (mode 4), 6–7 subset
//! count, 8–13 partition.

use crate::shared::{add_index, back_references, index_pair, BitWriter, History};
use crate::tables::Tables;
use crate::Error;

/// Endpoints per subset as R0 R1 G0 G1 B0 B1 A0 A1, each left-aligned in 16 bits.
#[derive(Clone, Copy, Default)]
pub(crate) struct Bc7Endpoints {
    pub(crate) control: u16,
    pub(crate) rows: [[u16; 8]; 3],
}

/// The primary and secondary index sets, each with its width in bits.
#[derive(Clone, Copy, Default)]
pub(crate) struct Bc7Indices {
    pub(crate) bits: [u8; 2],
    pub(crate) values: [[i16; 16]; 2],
}

/// The public BC7 mode table.
struct Mode {
    /// Endpoint bits of R, G, B and A, without p-bits.
    channel_bits: [u8; 4],
    /// Bits of the primary and secondary index sets.
    index_bits: [u8; 2],
    /// Per-endpoint p-bits (modes 0, 3, 6, 7) or per-subset p-bits (mode 1).
    p_bits: bool,
    partition_bits: u32,
}

const fn mode(rgb: u8, a: u8, index_bits: [u8; 2], p_bits: bool, partition_bits: u32) -> Mode {
    Mode { channel_bits: [rgb, rgb, rgb, a], index_bits, p_bits, partition_bits }
}

const MODES: [Mode; 8] = [
    mode(4, 0, [3, 0], true, 4),
    mode(6, 0, [3, 0], true, 6),
    mode(5, 0, [2, 0], false, 6),
    mode(7, 0, [2, 0], true, 6),
    mode(5, 6, [3, 2], false, 0),
    mode(7, 8, [2, 2], false, 0),
    mode(7, 7, [4, 0], true, 0),
    mode(5, 5, [2, 0], true, 6),
];

/// Anchor index of the second subset of two-subset partitions.
const ANCHOR_2_OF_2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, //
    15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2, //
    15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, //
    6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
];
/// Anchor indices of the second and third subsets of three-subset partitions.
const ANCHOR_2_OF_3: [u8; 64] = [
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, //
    3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15, //
    8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, //
    3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3,
];
const ANCHOR_3_OF_3: [u8; 64] = [
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, //
    15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8, //
    15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, //
    15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];

struct Control {
    word: u16,
}

impl Control {
    fn mode(&self) -> usize {
        usize::from(self.word & 7)
    }
    fn rotation(&self) -> usize {
        usize::from(self.word >> 3 & 3)
    }
    fn subsets(&self) -> usize {
        subsets(self.word)
    }
    fn partition(&self) -> usize {
        usize::from(self.word >> 8 & 63)
    }

    /// Stored bits per channel. The p-bit counts as red's lowest bit; a rotation swaps
    /// alpha's width with the rotated channel's.
    fn precision(&self) -> [u8; 4] {
        let m = &MODES[self.mode()];
        let mut p = m.channel_bits;
        if self.rotation() != 0 {
            p.swap(3, self.rotation() - 1);
        } else if m.p_bits {
            p[0] += 1;
        }
        p
    }
}

fn subsets(control: u16) -> usize {
    usize::from((control & 0xff) >> 6)
}

/// Record symbols holding each index set's residuals, two symbols per offset.
const INDEX_SYMBOLS: [[usize; 4]; 2] = [[22, 24, 26, 28], [12, 14, 16, 20]];

pub(crate) fn unpack<'a>(
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    t: &Tables,
) -> Result<(), Error> {
    let mut history = History::new(&t.dict_bc7, (Bc7Endpoints::default(), Bc7Indices::default()));
    for (i, (s, block)) in records.zip(out).enumerate() {
        let (ep_back, ix_back) = back_references(s, true);
        let code = usize::from(s[31] & 15) << 5 | usize::from(s[30] & 31);
        let control = Control { word: t.bc7_controls[code] };
        if control.word == 0 {
            return Err(Error::Bc7Control(code));
        }
        let subsets = control.subsets();
        let precision = control.precision();

        // Residual symbols per subset row and endpoint column. Symbols 10, 11, 18 and 19
        // carry alpha with one or two subsets, the third subset's G and B with three.
        let three = subsets == 3;
        let alpha = |k: usize| if three { 0 } else { s[k] };
        let third = |k: usize| if three { s[k] } else { 0 };
        let stored = [
            [s[4], s[5], s[6], s[7], s[8], s[9], alpha(10), alpha(11)],
            [s[12], s[13], s[14], s[15], s[16], s[17], alpha(18), alpha(19)],
            [s[20], s[21], third(10), third(11), third(18), third(19), 0, 0],
        ];
        let reference = &history.get(i, ep_back).0;
        let mut ends = Bc7Endpoints { control: control.word, ..Default::default() };
        // Subsets the reference lacks build on its first subset.
        let shared_rows = subsets.min(self::subsets(reference.control));
        let columns = if precision[3] > 0 { 8 } else { 6 };
        for column in 0..columns {
            let bits = precision[column / 2];
            if bits == 0 {
                continue;
            }
            let shift = 16 - u32::from(bits);
            for (row, symbols) in stored.iter().enumerate().take(subsets) {
                let v = symbols[column];
                let residual = (u16::from(v >> 1) ^ u16::from(v & 1).wrapping_neg()) << shift;
                let base = reference.rows[if row < shared_rows { row } else { 0 }][column] >> shift << shift;
                ends.rows[row][column] = base.wrapping_add(residual);
            }
        }

        let index_bits = MODES[control.mode()].index_bits;
        let ref_ix = &history.get(i, ix_back).1;
        let mut ix = Bc7Indices { bits: index_bits, ..Default::default() };
        for (set, symbols) in INDEX_SYMBOLS.iter().enumerate() {
            if index_bits[set] == 0 {
                continue;
            }
            for (k, &first) in symbols.iter().enumerate() {
                for (b, &byte) in s[first..first + 2].iter().enumerate() {
                    for (h, v) in index_pair(byte, k < 3).into_iter().enumerate() {
                        let p = 4 * k + 2 * b + h;
                        ix.values[set][p] = add_index(v, index_bits[set], ref_ix.values[set][p], ref_ix.bits[set]);
                    }
                }
            }
        }
        history.record(i, (ends, ix));
        write_block(&control, &ends, &ix, precision, block);
    }
    Ok(())
}

fn write_block(control: &Control, ends: &Bc7Endpoints, ix: &Bc7Indices, precision: [u8; 4], out: &mut [u8]) {
    let mode = control.mode();
    let m = &MODES[mode];
    let subsets = control.subsets();
    let partition = control.partition();
    let rotation = control.rotation();
    let mut w = BitWriter::default();
    w.put(1 << mode, mode as u32 + 1);
    w.put(partition as u64, m.partition_bits);
    if mode == 4 || mode == 5 {
        w.put(rotation as u64, 2);
    }
    if mode == 4 {
        w.put(u64::from(control.word >> 5 & 1), 1);
    }

    let endpoints = 2 * subsets;
    let field = |row: usize, column: usize, bits: u8| u64::from(ends.rows[row][column] >> (16 - bits));
    let mut channels = [0, 1, 2, 3];
    if rotation != 0 {
        channels.swap(3, rotation - 1);
    }
    for channel in channels {
        let bits = precision[channel];
        if bits == 0 {
            continue;
        }
        let p_bit = u8::from(channel == 0 && m.p_bits);
        for e in 0..endpoints {
            w.put(field(e / 2, 2 * channel + e % 2, bits) >> p_bit, u32::from(bits - p_bit));
        }
    }
    if mode == 1 {
        for row in 0..subsets {
            w.put(field(row, 0, precision[0]) & 1, 1);
        }
    } else if m.p_bits {
        for e in 0..endpoints {
            w.put(field(e / 2, e % 2, precision[0]) & 1, 1);
        }
    }

    let anchors: &[u8] = match subsets {
        2 => &[0, ANCHOR_2_OF_2[partition]],
        3 => &[0, ANCHOR_2_OF_3[partition], ANCHOR_3_OF_3[partition]],
        _ => &[0],
    };
    let mut put_indices = |set: usize| {
        let bits = u32::from(ix.bits[set]);
        for (p, &v) in ix.values[set].iter().enumerate() {
            w.put(u64::from(v as u16), bits - u32::from(anchors.contains(&(p as u8))));
        }
    };
    match mode {
        4 => {
            put_indices(1);
            put_indices(0);
        }
        5 => {
            put_indices(0);
            put_indices(1);
        }
        _ => put_indices(0),
    }
    w.finish(out);
}
