//! BC6H. A record names the mode (1–14 in the public numbering) and carries endpoint
//! residuals for endpoint 0 at the mode's base precision and for the other endpoints at
//! their delta precision, plus index residuals. Endpoints are kept left-aligned in 16 bits
//! with the top bit flipped.

use crate::shared::{add_index, back_references, index_pair, interleave8, unzigzag16, BitWriter, History};
use crate::tables::Tables;
use crate::Error;

/// Endpoints 0–3 as R G B. One-region modes keep endpoint 2 as endpoint 0 at the delta
/// precision and endpoint 3 as a copy of endpoint 1.
#[derive(Clone, Copy, Default)]
pub(crate) struct Bc6Endpoints {
    pub(crate) values: [[u16; 3]; 4],
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Bc6Indices {
    pub(crate) mode: u8,
    pub(crate) values: [i16; 16],
}

/// A run of header bits: `len` bits of `endpoint`'s `channel` from bit `low` up, or the
/// partition when `channel` is `PARTITION`.
#[derive(Clone, Copy)]
struct Field {
    channel: u8,
    endpoint: u8,
    low: u8,
    len: u8,
}

const PARTITION: u8 = 3;
const MAX_FIELDS: usize = 24;

struct Layout {
    fields: [Field; MAX_FIELDS],
    count: usize,
}

/// Parses a header layout written in the notation of the public BC6H tables: `rw[9:0]` is
/// bits 9 down to 0 of endpoint w's red (w x y z are endpoints 0–3), `gy4` a single bit,
/// `d[4:0]` the partition. Each token is one run, stored low bit first.
const fn layout(spec: &str) -> Layout {
    let s = spec.as_bytes();
    let mut fields = [Field { channel: 0, endpoint: 0, low: 0, len: 0 }; MAX_FIELDS];
    let mut count = 0;
    let mut i = 0;
    while i < s.len() {
        if s[i] == b' ' {
            i += 1;
            continue;
        }
        let channel = match s[i] {
            b'r' => 0,
            b'g' => 1,
            b'b' => 2,
            b'd' => PARTITION,
            _ => panic!("bad channel"),
        };
        i += 1;
        let mut endpoint = 0;
        if channel != PARTITION {
            endpoint = match s[i] {
                b'w' => 0,
                b'x' => 1,
                b'y' => 2,
                b'z' => 3,
                _ => panic!("bad endpoint"),
            };
            i += 1;
        }
        let (high, low);
        if s[i] == b'[' {
            let (h, next) = number(s, i + 1);
            assert!(s[next] == b':');
            let (l, next) = number(s, next + 1);
            assert!(s[next] == b']');
            (high, low) = (h, l);
            i = next + 1;
        } else {
            let (bit, next) = number(s, i);
            (high, low) = (bit, bit);
            i = next;
        }
        assert!(count < MAX_FIELDS && high >= low);
        fields[count] = Field { channel, endpoint, low, len: high - low + 1 };
        count += 1;
    }
    Layout { fields, count }
}

const fn number(s: &[u8], mut i: usize) -> (u8, usize) {
    let mut n = 0;
    assert!(i < s.len() && s[i].is_ascii_digit());
    while i < s.len() && s[i].is_ascii_digit() {
        n = n * 10 + (s[i] - b'0');
        i += 1;
    }
    (n, i)
}

struct Mode {
    /// Mode bits and their count.
    code: u64,
    code_bits: u32,
    /// Bits of endpoint 0, then of the other endpoints' R, G and B.
    base_bits: u8,
    delta_bits: [u8; 3],
    layout: Layout,
}

const fn mode(code: u64, code_bits: u32, base_bits: u8, delta_bits: [u8; 3], spec: &str) -> Mode {
    Mode { code, code_bits, base_bits, delta_bits, layout: layout(spec) }
}

/// Modes 1–14 (index 0 unused). Modes 13 and 14 store the high bits of endpoint 0 as one
/// ascending run, as the console's packer writes them.
const MODES: [Mode; 15] = [
    mode(0, 0, 0, [0; 3], ""),
    mode(0, 2, 10, [5, 5, 5], "gy4 by4 bz4 rw[9:0] gw[9:0] bw[9:0] rx[4:0] gz4 gy[3:0] gx[4:0] bz0 gz[3:0] bx[4:0] bz1 by[3:0] ry[4:0] bz2 rz[4:0] bz3 d[4:0]"),
    mode(1, 2, 7, [6, 6, 6], "gy5 gz4 gz5 rw[6:0] bz0 bz1 by4 gw[6:0] by5 bz2 gy4 bw[6:0] bz3 bz5 bz4 rx[5:0] gy[3:0] gx[5:0] gz[3:0] bx[5:0] by[3:0] ry[5:0] rz[5:0] d[4:0]"),
    mode(2, 5, 11, [5, 4, 4], "rw[9:0] gw[9:0] bw[9:0] rx[4:0] rw10 gy[3:0] gx[3:0] gw10 bz0 gz[3:0] bx[3:0] bw10 bz1 by[3:0] ry[4:0] bz2 rz[4:0] bz3 d[4:0]"),
    mode(6, 5, 11, [4, 5, 4], "rw[9:0] gw[9:0] bw[9:0] rx[3:0] rw10 gz4 gy[3:0] gx[4:0] gw10 gz[3:0] bx[3:0] bw10 bz1 by[3:0] ry[3:0] bz0 bz2 rz[3:0] gy4 bz3 d[4:0]"),
    mode(10, 5, 11, [4, 4, 5], "rw[9:0] gw[9:0] bw[9:0] rx[3:0] rw10 by4 gy[3:0] gx[3:0] gw10 bz0 gz[3:0] bx[4:0] bw10 by[3:0] ry[3:0] bz1 bz2 rz[3:0] bz4 bz3 d[4:0]"),
    mode(14, 5, 9, [5, 5, 5], "rw[8:0] by4 gw[8:0] gy4 bw[8:0] bz4 rx[4:0] gz4 gy[3:0] gx[4:0] bz0 gz[3:0] bx[4:0] bz1 by[3:0] ry[4:0] bz2 rz[4:0] bz3 d[4:0]"),
    mode(18, 5, 8, [6, 5, 5], "rw[7:0] gz4 by4 gw[7:0] bz2 gy4 bw[7:0] bz3 bz4 rx[5:0] gy[3:0] gx[4:0] bz0 gz[3:0] bx[4:0] bz1 by[3:0] ry[5:0] rz[5:0] d[4:0]"),
    mode(22, 5, 8, [5, 6, 5], "rw[7:0] bz0 by4 gw[7:0] gy5 gy4 bw[7:0] gz5 bz4 rx[4:0] gz4 gy[3:0] gx[5:0] gz[3:0] bx[4:0] bz1 by[3:0] ry[4:0] bz2 rz[4:0] bz3 d[4:0]"),
    mode(26, 5, 8, [5, 5, 6], "rw[7:0] bz1 by4 gw[7:0] by5 gy4 bw[7:0] bz5 bz4 rx[4:0] gz4 gy[3:0] gx[4:0] bz0 gz[3:0] bx[5:0] by[3:0] ry[4:0] bz2 rz[4:0] bz3 d[4:0]"),
    mode(30, 5, 6, [6, 6, 6], "rw[5:0] gz4 bz0 bz1 by4 gw[5:0] gy5 by5 bz2 gy4 bw[5:0] gz5 bz3 bz5 bz4 rx[5:0] gy[3:0] gx[5:0] gz[3:0] bx[5:0] by[3:0] ry[5:0] rz[5:0] d[4:0]"),
    mode(3, 5, 10, [10, 10, 10], "rw[9:0] gw[9:0] bw[9:0] rx[9:0] gx[9:0] bx[9:0]"),
    mode(7, 5, 11, [9, 9, 9], "rw[9:0] gw[9:0] bw[9:0] rx[8:0] rw10 gx[8:0] gw10 bx[8:0] bw10"),
    mode(11, 5, 12, [8, 8, 8], "rw[9:0] gw[9:0] bw[9:0] rx[7:0] rw[11:10] gx[7:0] gw[11:10] bx[7:0] bw[11:10]"),
    mode(15, 5, 16, [4, 4, 4], "rw[9:0] gw[9:0] bw[9:0] rx[3:0] rw[15:10] gx[3:0] gw[15:10] bx[3:0] bw[15:10]"),
];

/// Modes 1–10 have two regions (four endpoints), 11–14 one.
fn two_regions(mode: u8) -> bool {
    mode < 11
}

fn index_bits(mode: u8) -> u8 {
    match mode {
        1..=10 => 3,
        11..=14 => 4,
        _ => 0,
    }
}

/// Second-region anchor index per partition, shared with BC7's two-subset partitions.
const ANCHOR: [u8; 32] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, //
    15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
];

pub(crate) fn unpack<'a>(
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    t: &Tables,
) -> Result<(), Error> {
    let none = (Bc6Endpoints::default(), Bc6Indices { mode: 11, ..Default::default() });
    let mut history = History::new(&t.dict_bc6, none);
    for (i, (s, block)) in records.zip(out).enumerate() {
        let (ep_back, ix_back) = back_references(s, true);
        let mode = s[30].wrapping_add(1);
        if !(1..=14).contains(&mode) {
            return Err(Error::Bc6Mode(mode));
        }
        let m = &MODES[usize::from(mode)];
        let two = two_regions(mode);
        // Left shifts that align endpoint 0, and the other endpoints' R, G, B, to 16 bits.
        let base_shift = 16 - u32::from(m.base_bits);
        let delta_shift = m.delta_bits.map(|b| 16 - u32::from(b));

        let wide = |k: usize| unzigzag16(interleave8(s[k], s[k + 1]));
        let narrow = |k: usize| unzigzag16(s[k].into());
        let residuals = [
            [wide(4), wide(6), wide(8)],
            [wide(10), wide(12), wide(14)],
            [narrow(16), narrow(18), narrow(20)],
            [narrow(17), narrow(19), narrow(21)],
        ];
        let reference = &history.get(i, ep_back).0;
        let mut ends = Bc6Endpoints::default();
        // The sum is cut to the endpoint's precision, so the reference's low bits drop out.
        let add = |r: u16, v: u16, shift: u32| r.wrapping_add(v << shift) & (0xffff << shift);
        for (e, residual) in residuals.iter().enumerate().take(if two { 4 } else { 2 }) {
            for c in 0..3 {
                let shift = if e == 0 { base_shift } else { delta_shift[c] };
                ends.values[e][c] = add(reference.values[e][c], residual[c], shift);
            }
        }
        if !two {
            for c in 0..3 {
                ends.values[2][c] = (ends.values[0][c] >> base_shift) << delta_shift[0];
            }
            ends.values[3] = ends.values[1];
        }

        let ref_ix = &history.get(i, ix_back).1;
        let bits = index_bits(mode);
        let mut ix = Bc6Indices { mode, ..Default::default() };
        for k in 0..8 {
            for (h, v) in index_pair(s[22 + k], k < 6).into_iter().enumerate() {
                let p = 2 * k + h;
                ix.values[p] = add_index(v, bits, ref_ix.values[p], index_bits(ref_ix.mode));
            }
        }
        history.record(i, (ends, ix));

        let mut w = BitWriter::default();
        w.put(m.code, m.code_bits);
        let partition = s[31] & 31;
        let value = |e: usize, c: usize| {
            let shift = if e == 0 { base_shift } else { delta_shift[c] };
            u64::from((ends.values[e][c] ^ 0x8000) >> shift)
        };
        for f in &m.layout.fields[..m.layout.count] {
            let v = if f.channel == PARTITION {
                u64::from(partition)
            } else {
                value(f.endpoint.into(), f.channel.into()) >> f.low
            };
            w.put(v, f.len.into());
        }
        let anchor = if two { usize::from(ANCHOR[usize::from(partition)]) } else { 0 };
        for (p, &v) in ix.values.iter().enumerate() {
            let anchored = p == 0 || p == anchor;
            let biased = (v as u16).wrapping_add(1 << (bits - 1));
            w.put(biased.into(), u32::from(bits) - u32::from(anchored));
        }
        w.finish(block);
    }
    Ok(())
}
