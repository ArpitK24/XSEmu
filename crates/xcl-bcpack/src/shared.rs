//! Pieces every block format shares: back references, zigzag residuals and the bit writer.

/// Blocks a back reference can reach.
const HISTORY: usize = 256;

/// The values of the last 256 blocks, with the format's dictionary standing in for the
/// blocks before the first.
pub(crate) struct History<'d, T> {
    ring: Vec<T>,
    dictionary: &'d [T],
    none: T,
}

impl<'d, T: Copy> History<'d, T> {
    /// `none` is what reference 0 means.
    pub(crate) fn new(dictionary: &'d [T], none: T) -> Self {
        History { ring: vec![none; HISTORY], dictionary, none }
    }

    /// Block `block`'s reference `back`: 0 is none, else `back` blocks earlier.
    pub(crate) fn get(&self, block: usize, back: usize) -> &T {
        match back {
            0 => &self.none,
            b if b <= block => &self.ring[(block - b) % HISTORY],
            b => &self.dictionary[b - block - 1],
        }
    }

    pub(crate) fn record(&mut self, block: usize, value: T) {
        self.ring[block % HISTORY] = value;
    }
}

/// A record's endpoint and index back references. The index reference is coded directly,
/// the endpoint reference as a zigzag difference from it. Formats with two symbols per lane
/// split each reference over two 4-bit symbols.
pub(crate) fn back_references(record: &[u8], split: bool) -> (usize, usize) {
    let (diff, index) = if split {
        (interleave4(record[0], record[1]), interleave4(record[2], record[3]))
    } else {
        (record[0], record[1])
    };
    let diff = (diff >> 1) ^ (diff & 1).wrapping_neg();
    (usize::from(index.wrapping_add(diff)), usize::from(index))
}

/// Merges two 4-bit values bit by bit, `even` on the even bits.
fn interleave4(even: u8, odd: u8) -> u8 {
    (0..4).fold(0, |acc, k| acc | (even >> k & 1) << (2 * k) | (odd >> k & 1) << (2 * k + 1))
}

/// Merges two bytes bit by bit into 16 bits, `even` on the even bits.
pub(crate) fn interleave8(even: u8, odd: u8) -> u16 {
    (0..8).fold(0, |acc, k| acc | u16::from(even >> k & 1) << (2 * k) | u16::from(odd >> k & 1) << (2 * k + 1))
}

/// Zigzag (0, -1, 1, -2, …) decoded modulo `2^bits`.
pub(crate) fn unzigzag_bits(v: u8, bits: u32) -> u8 {
    let mask = ((1u32 << bits) - 1) as u8;
    ((v >> 1) ^ if v & 1 != 0 { mask } else { 0 }) & mask
}

/// Zigzag decoded to 16 bits.
pub(crate) fn unzigzag16(v: u16) -> u16 {
    (v >> 1) ^ (v & 1).wrapping_neg()
}

/// A byte as two zigzag-coded 4-bit index residuals, one on its even bits and one on its odd
/// bits. `odd_first` gives the odd one first.
pub(crate) fn index_pair(b: u8, odd_first: bool) -> [i16; 2] {
    let gather = |v: u8| (0..4).fold(0u8, |acc, k| acc | (v >> (2 * k) & 1) << k);
    let unzig = |u: u8| i16::from(u >> 1) ^ -i16::from(u & 1);
    let (even, odd) = (unzig(gather(b)), unzig(gather(b >> 1)));
    if odd_first {
        [odd, even]
    } else {
        [even, odd]
    }
}

/// Adds an index residual of `bits` bits to a reference index of `ref_bits` bits: both are
/// scaled to the top of 16 bits, summed, and the sum cut back to `bits` bits (signed).
pub(crate) fn add_index(residual: i16, bits: u8, reference: i16, ref_bits: u8) -> i16 {
    let scaled = |v: i16, b: u8| if b == 0 || b > 16 { 0 } else { (v as u16) << (16 - b) };
    if bits == 0 {
        return 0;
    }
    (scaled(residual, bits).wrapping_add(scaled(reference, ref_bits)) as i16) >> (16 - bits)
}

/// Writes fields low bit first into one 128-bit block; bits past the block are dropped.
#[derive(Default)]
pub(crate) struct BitWriter {
    bits: u128,
    at: u32,
}

impl BitWriter {
    pub(crate) fn put(&mut self, value: u64, width: u32) {
        if width > 0 && self.at < 128 {
            self.bits |= (u128::from(value) & ((1u128 << width) - 1)) << self.at;
        }
        self.at += width;
    }

    pub(crate) fn finish(self, out: &mut [u8]) {
        out.copy_from_slice(&self.bits.to_le_bytes());
    }
}
