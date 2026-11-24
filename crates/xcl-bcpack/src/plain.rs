//! BC1, BC3, BC4 and BC5: endpoints and indices are small unsigned values, and a block is
//! its reference plus the residual, element by element, modulo each element's width.

use crate::shared::{back_references, unzigzag_bits, History};
use crate::tables::Tables;
use crate::{Error, Format};

/// A block of the plain formats before packing. BC3 keeps colour endpoints in 0–5 and alpha
/// endpoints in 6–7, alpha indices in 0–15 and colour indices in 16–31; BC5 keeps the red
/// channel's endpoints and indices before the green channel's.
#[derive(Clone, Copy, Default)]
pub(crate) struct PlainBlock {
    pub(crate) endpoints: [u8; 8],
    pub(crate) indices: [u8; 32],
}

/// Element widths in bits.
struct Widths {
    endpoints: [u8; 8],
    indices: [u8; 32],
}

const fn widths(endpoints: [u8; 8], three_bit: usize, two_bit: usize) -> Widths {
    let mut indices = [0; 32];
    let mut k = 0;
    while k < three_bit + two_bit {
        indices[k] = if k < three_bit { 3 } else { 2 };
        k += 1;
    }
    Widths { endpoints, indices }
}

const RGB565_PAIR: [u8; 6] = [5, 6, 5, 5, 6, 5];
const BC1: Widths = widths([5, 6, 5, 5, 6, 5, 0, 0], 0, 16);
const BC3: Widths = widths([5, 6, 5, 5, 6, 5, 8, 8], 16, 16);
const BC4: Widths = widths([8, 8, 0, 0, 0, 0, 0, 0], 16, 0);
const BC5: Widths = widths([8, 8, 8, 8, 0, 0, 0, 0], 32, 0);

pub(crate) fn unpack<'a>(
    format: Format,
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    t: &Tables,
) -> Result<(), Error> {
    let (w, dictionary) = match format {
        Format::Bc1 => (&BC1, &t.dict_bc1),
        Format::Bc3 => (&BC3, &t.dict_bc3),
        Format::Bc4 => (&BC4, &t.dict_bc4),
        _ => (&BC5, &t.dict_bc5),
    };
    let mut history = History::new(dictionary, PlainBlock::default());
    for (i, (s, block)) in records.zip(out).enumerate() {
        let (ep_back, ix_back) = back_references(s, format != Format::Bc1 && format != Format::Bc4);
        let mut v = residual(format, s, t);
        let ep_ref = &history.get(i, ep_back).endpoints;
        for ((v, r), bits) in v.endpoints.iter_mut().zip(ep_ref).zip(w.endpoints) {
            *v = v.wrapping_add(*r) & ((1u16 << bits) - 1) as u8;
        }
        let ix_ref = &history.get(i, ix_back).indices;
        for ((v, r), bits) in v.indices.iter_mut().zip(ix_ref).zip(w.indices) {
            *v = v.wrapping_add(*r) & ((1u8 << bits) - 1);
        }
        history.record(i, v);
        match format {
            Format::Bc1 => bc1_color(&v.endpoints, &v.indices, block),
            Format::Bc3 => {
                bc4_channel(v.endpoints[6], v.endpoints[7], &v.indices[..16], &mut block[..8]);
                bc1_color(&v.endpoints, &v.indices[16..], &mut block[8..]);
            }
            Format::Bc4 => bc4_channel(v.endpoints[0], v.endpoints[1], &v.indices, block),
            _ => {
                bc4_channel(v.endpoints[0], v.endpoints[1], &v.indices[..16], &mut block[..8]);
                bc4_channel(v.endpoints[2], v.endpoints[3], &v.indices[16..], &mut block[8..]);
            }
        }
    }
    Ok(())
}

/// The record's residuals. Record symbol `k` comes from lane `k / per`; the first lanes
/// carry the back references.
fn residual(format: Format, s: &[u8], t: &Tables) -> PlainBlock {
    let mut v = PlainBlock::default();
    match format {
        Format::Bc1 => {
            for (k, bits) in RGB565_PAIR.into_iter().enumerate() {
                v.endpoints[k] = unzigzag_bits(s[2 + k], bits.into());
            }
            quad_indices(&t.rank_bc1, &s[8..12], &mut v.indices[..16]);
        }
        Format::Bc3 => {
            // Each colour lane carries one channel's two endpoints.
            for (k, (&sym, bits)) in [s[14], s[16], s[18], s[15], s[17], s[19]].iter().zip(RGB565_PAIR).enumerate() {
                v.endpoints[k] = unzigzag_bits(sym, bits.into());
            }
            v.endpoints[6] = unzigzag_bits(s[4], 8);
            v.endpoints[7] = unzigzag_bits(s[5], 8);
            pair_indices(&t.rank_bc3_alpha, &s[6..14], &mut v.indices[..16]);
            quad_indices(&t.rank_bc3_color, &s[20..24], &mut v.indices[16..]);
        }
        Format::Bc4 => {
            v.endpoints[0] = unzigzag_bits(s[2], 8);
            v.endpoints[1] = unzigzag_bits(s[3], 8);
            pair_indices(&t.rank_bc4, &s[4..12], &mut v.indices[..16]);
        }
        _ => {
            for k in 0..4 {
                v.endpoints[k] = unzigzag_bits(s[4 + k], 8);
            }
            pair_indices(&t.rank_bc5, &s[8..24], &mut v.indices);
        }
    }
    v
}

/// 2-bit index residuals: each symbol ranks the four indices of one 2×2 quad.
fn quad_indices(rank: &[u8; 256], symbols: &[u8], indices: &mut [u8]) {
    for (q, &sym) in symbols.iter().enumerate() {
        let packed = rank[usize::from(sym)];
        let corner = (q >> 1) * 8 + (q & 1) * 2;
        for (k, offset) in [0, 1, 4, 5].into_iter().enumerate() {
            indices[corner + offset] = packed >> (2 * k) & 3;
        }
    }
}

/// 3-bit index residuals: each symbol ranks two neighbouring indices in raster order.
fn pair_indices(rank: &[u8; 64], symbols: &[u8], indices: &mut [u8]) {
    for (k, &sym) in symbols.iter().enumerate() {
        let packed = rank[usize::from(sym & 63)];
        indices[2 * k] = packed & 7;
        indices[2 * k + 1] = packed >> 4 & 7;
    }
}

/// Indices count along the line from endpoint 0 to endpoint 1; BC1 and BC4 number their
/// palette with both endpoints first.
const BC1_PALETTE: [u32; 4] = [0, 2, 3, 1];
const BC4_PALETTE: [u64; 8] = [0, 2, 3, 4, 5, 6, 7, 1];

fn bc1_color(endpoints: &[u8], indices: &[u8], out: &mut [u8]) {
    let rgb565 = |e: &[u8]| u16::from(e[0]) << 11 | u16::from(e[1]) << 5 | u16::from(e[2]);
    let selectors = (0..16).fold(0, |acc, p| acc | BC1_PALETTE[usize::from(indices[p] & 3)] << (2 * p));
    out[..2].copy_from_slice(&rgb565(&endpoints[..3]).to_le_bytes());
    out[2..4].copy_from_slice(&rgb565(&endpoints[3..6]).to_le_bytes());
    out[4..8].copy_from_slice(&selectors.to_le_bytes());
}

/// One BC4 channel. Endpoints are kept 128 below their stored value.
fn bc4_channel(e0: u8, e1: u8, indices: &[u8], out: &mut [u8]) {
    let selectors = (0..16).fold(0u64, |acc, p| acc | BC4_PALETTE[usize::from(indices[p] & 7)] << (16 + 3 * p));
    let v = u64::from(e0 ^ 0x80) | u64::from(e1 ^ 0x80) << 8 | selectors;
    out[..8].copy_from_slice(&v.to_le_bytes());
}
