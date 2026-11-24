//! The two hashes of a DXIL container, both MD5 with DXC's block layout
//! (lib/DxilHash/DxilHash.cpp): plain MD5 for the HASH part (the bitcode's "shader hash"), and
//! the container digest in the header, which D3D12 checks before it accepts a shader (an
//! unsigned or wrongly signed container is rejected at pipeline creation).
//!
//! The digest is MD5's compression function over the container from its version field on,
//! with a different last block: where MD5 appends 0x80, zero padding and the bit length in the
//! last two words, the digest's final block starts with the bit length, then the remaining
//! data and the padding, and ends with `(length << 1) | 1` (DXC's `ComputeHashRetail`).

const INIT: [u32; 4] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];

/// MD5's compression function: one 64-byte block into the state. A container is hashed
/// twice (the bitcode for HASH, the whole of it for the digest), 13 MB each for the largest
/// programs, so this is written out: the 64 steps unrolled with
/// their constants (floor(2^32 * abs(sin(i + 1))), RFC 1321's table) and rotations in place,
/// and the round functions in the forms with the shortest dependency chains
/// (F = d ^ (b & (c ^ d)), G = c ^ (d & (b ^ c))).
#[inline(always)]
fn block(state: &mut [u32; 4], bytes: &[u8; 64]) {
    let [mut a, mut b, mut c, mut d] = *state;
    // word k of the block, read where it lies
    macro_rules! x {
        ($k:expr) => {
            u32::from_le_bytes([bytes[4 * $k], bytes[4 * $k + 1], bytes[4 * $k + 2], bytes[4 * $k + 3]])
        };
    }
    macro_rules! step {
        (f, $a:ident, $b:ident, $c:ident, $d:ident, $k:expr, $s:expr, $t:expr) => {
            $a = $a.wrapping_add($d ^ ($b & ($c ^ $d))).wrapping_add(x!($k)).wrapping_add($t).rotate_left($s).wrapping_add($b)
        };
        (g, $a:ident, $b:ident, $c:ident, $d:ident, $k:expr, $s:expr, $t:expr) => {
            $a = $a.wrapping_add($c ^ ($d & ($b ^ $c))).wrapping_add(x!($k)).wrapping_add($t).rotate_left($s).wrapping_add($b)
        };
        (h, $a:ident, $b:ident, $c:ident, $d:ident, $k:expr, $s:expr, $t:expr) => {
            $a = $a.wrapping_add($b ^ $c ^ $d).wrapping_add(x!($k)).wrapping_add($t).rotate_left($s).wrapping_add($b)
        };
        (i, $a:ident, $b:ident, $c:ident, $d:ident, $k:expr, $s:expr, $t:expr) => {
            $a = $a.wrapping_add($c ^ ($b | !$d)).wrapping_add(x!($k)).wrapping_add($t).rotate_left($s).wrapping_add($b)
        };
    }
    step!(f, a, b, c, d,  0,  7, 0xd76aa478);
    step!(f, d, a, b, c,  1, 12, 0xe8c7b756);
    step!(f, c, d, a, b,  2, 17, 0x242070db);
    step!(f, b, c, d, a,  3, 22, 0xc1bdceee);
    step!(f, a, b, c, d,  4,  7, 0xf57c0faf);
    step!(f, d, a, b, c,  5, 12, 0x4787c62a);
    step!(f, c, d, a, b,  6, 17, 0xa8304613);
    step!(f, b, c, d, a,  7, 22, 0xfd469501);
    step!(f, a, b, c, d,  8,  7, 0x698098d8);
    step!(f, d, a, b, c,  9, 12, 0x8b44f7af);
    step!(f, c, d, a, b, 10, 17, 0xffff5bb1);
    step!(f, b, c, d, a, 11, 22, 0x895cd7be);
    step!(f, a, b, c, d, 12,  7, 0x6b901122);
    step!(f, d, a, b, c, 13, 12, 0xfd987193);
    step!(f, c, d, a, b, 14, 17, 0xa679438e);
    step!(f, b, c, d, a, 15, 22, 0x49b40821);

    step!(g, a, b, c, d,  1,  5, 0xf61e2562);
    step!(g, d, a, b, c,  6,  9, 0xc040b340);
    step!(g, c, d, a, b, 11, 14, 0x265e5a51);
    step!(g, b, c, d, a,  0, 20, 0xe9b6c7aa);
    step!(g, a, b, c, d,  5,  5, 0xd62f105d);
    step!(g, d, a, b, c, 10,  9, 0x02441453);
    step!(g, c, d, a, b, 15, 14, 0xd8a1e681);
    step!(g, b, c, d, a,  4, 20, 0xe7d3fbc8);
    step!(g, a, b, c, d,  9,  5, 0x21e1cde6);
    step!(g, d, a, b, c, 14,  9, 0xc33707d6);
    step!(g, c, d, a, b,  3, 14, 0xf4d50d87);
    step!(g, b, c, d, a,  8, 20, 0x455a14ed);
    step!(g, a, b, c, d, 13,  5, 0xa9e3e905);
    step!(g, d, a, b, c,  2,  9, 0xfcefa3f8);
    step!(g, c, d, a, b,  7, 14, 0x676f02d9);
    step!(g, b, c, d, a, 12, 20, 0x8d2a4c8a);

    step!(h, a, b, c, d,  5,  4, 0xfffa3942);
    step!(h, d, a, b, c,  8, 11, 0x8771f681);
    step!(h, c, d, a, b, 11, 16, 0x6d9d6122);
    step!(h, b, c, d, a, 14, 23, 0xfde5380c);
    step!(h, a, b, c, d,  1,  4, 0xa4beea44);
    step!(h, d, a, b, c,  4, 11, 0x4bdecfa9);
    step!(h, c, d, a, b,  7, 16, 0xf6bb4b60);
    step!(h, b, c, d, a, 10, 23, 0xbebfbc70);
    step!(h, a, b, c, d, 13,  4, 0x289b7ec6);
    step!(h, d, a, b, c,  0, 11, 0xeaa127fa);
    step!(h, c, d, a, b,  3, 16, 0xd4ef3085);
    step!(h, b, c, d, a,  6, 23, 0x04881d05);
    step!(h, a, b, c, d,  9,  4, 0xd9d4d039);
    step!(h, d, a, b, c, 12, 11, 0xe6db99e5);
    step!(h, c, d, a, b, 15, 16, 0x1fa27cf8);
    step!(h, b, c, d, a,  2, 23, 0xc4ac5665);

    step!(i, a, b, c, d,  0,  6, 0xf4292244);
    step!(i, d, a, b, c,  7, 10, 0x432aff97);
    step!(i, c, d, a, b, 14, 15, 0xab9423a7);
    step!(i, b, c, d, a,  5, 21, 0xfc93a039);
    step!(i, a, b, c, d, 12,  6, 0x655b59c3);
    step!(i, d, a, b, c,  3, 10, 0x8f0ccc92);
    step!(i, c, d, a, b, 10, 15, 0xffeff47d);
    step!(i, b, c, d, a,  1, 21, 0x85845dd1);
    step!(i, a, b, c, d,  8,  6, 0x6fa87e4f);
    step!(i, d, a, b, c, 15, 10, 0xfe2ce6e0);
    step!(i, c, d, a, b,  6, 15, 0xa3014314);
    step!(i, b, c, d, a, 13, 21, 0x4e0811a1);
    step!(i, a, b, c, d,  4,  6, 0xf7537e82);
    step!(i, d, a, b, c, 11, 10, 0xbd3af235);
    step!(i, c, d, a, b,  2, 15, 0x2ad7d2bb);
    step!(i, b, c, d, a,  9, 21, 0xeb86d391);
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
}

/// Every whole 64-byte block of `data` into the state; returns what is left over (under 64
/// bytes). No copy: the blocks are read where they lie.
fn blocks<'a>(state: &mut [u32; 4], data: &'a [u8]) -> &'a [u8] {
    let mut s = *state;
    let mut chunks = data.chunks_exact(64);
    for c in &mut chunks {
        if let Ok(c) = <&[u8; 64]>::try_from(c) {
            block(&mut s, c);
        }
    }
    *state = s;
    chunks.remainder()
}

fn finish(state: [u32; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (o, w) in out.chunks_exact_mut(4).zip(state) {
        o.copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// RFC 1321 MD5: what DXC stores in the HASH part for the bitcode (`llvm::MD5`).
pub fn md5(data: &[u8]) -> [u8; 16] {
    let mut state = INIT;
    let rest = blocks(&mut state, data);
    let mut tail = [0u8; 128];
    tail[..rest.len()].copy_from_slice(rest);
    tail[rest.len()] = 0x80;
    let n = if rest.len() < 56 { 64 } else { 128 };
    tail[n - 8..n].copy_from_slice(&((data.len() as u64) << 3).to_le_bytes());
    blocks(&mut state, &tail[..n]);
    finish(state)
}

/// DXC's `ComputeHashRetail`: the digest of a container, over `data` = everything after the
/// digest field. The lengths are DXC's 32-bit ones (a container's size field is 32 bits).
pub fn digest(data: &[u8]) -> [u8; 16] {
    let mut state = INIT;
    let rest = blocks(&mut state, data);
    let len = data.len() as u32;
    let mut last = [0u8; 64];
    last[..4].copy_from_slice(&(len << 3).to_le_bytes());
    last[60..].copy_from_slice(&((len << 1) | 1).to_le_bytes());
    if rest.len() < 56 {
        last[4..4 + rest.len()].copy_from_slice(rest);
        last[4 + rest.len()] = 0x80;
    } else {
        // no room for the two length words: the data and the 0x80 marker fill a block of
        // their own, and the last block has only the lengths
        let mut full = [0u8; 64];
        full[..rest.len()].copy_from_slice(rest);
        full[rest.len()] = 0x80;
        block(&mut state, &full);
    }
    block(&mut state, &last);
    finish(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: [u8; 16]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// RFC 1321, appendix A.5.
    #[test]
    fn md5_rfc_vectors() {
        assert_eq!(hex(md5(b"")), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(md5(b"a")), "0cc175b9c0f1b6a831c399e269772661");
        assert_eq!(hex(md5(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(hex(md5(b"message digest")), "f96b697d7cb7938d525a2f31aaf161d0");
        assert_eq!(hex(md5(b"abcdefghijklmnopqrstuvwxyz")), "c3fcd3d76192e4007dfb496cca67e13b");
        assert_eq!(hex(md5(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")), "d174ab98d277d9f5a5611c2c9f419d9f");
        assert_eq!(hex(md5(b"12345678901234567890123456789012345678901234567890123456789012345678901234567890")), "57edf4a22be3c955ac49da2e2107b67a");
    }

    /// Lengths around the block and padding boundaries against the definition (padding
    /// written out by hand, one block at a time).
    #[test]
    fn md5_padding_boundaries() {
        for n in [55usize, 56, 57, 63, 64, 65, 119, 120, 121, 128, 1000] {
            let data: Vec<u8> = (0..n).map(|i| (i * 7 + 3) as u8).collect();
            let mut padded = data.clone();
            padded.push(0x80);
            while padded.len() % 64 != 56 {
                padded.push(0);
            }
            padded.extend_from_slice(&((n as u64) * 8).to_le_bytes());
            let mut state = INIT;
            assert!(blocks(&mut state, &padded).is_empty());
            assert_eq!(md5(&data), finish(state), "length {n}");
        }
    }

    fn words(d: [u8; 16]) -> [u32; 4] {
        [0, 4, 8, 12].map(|i| u32::from_le_bytes([d[i], d[i + 1], d[i + 2], d[i + 3]]))
    }

    /// DXC's own vectors (unittests/DxilHash/DxilHashTest.cpp, FixHashValueTest).
    #[test]
    fn digest_dxc_vectors() {
        assert_eq!(words(digest(b"")), [0xf6600d14, 0xbae275b7, 0xd4be4a4e, 0xa1e9b201]);
        let rep = |tails: [&str; 5]| tails.iter().map(|t| format!("abcdefghijklmnopqrstuvwxyz{t}")).collect::<String>();
        let data2 = rep(["abcdef", "ghijkl", "mnopqr", "stuvwx", "yzabcd"]);
        assert_eq!(data2.len(), 160);
        assert_eq!(words(digest(data2.as_bytes())), [0x00830fe6, 0xd8d8a035, 0xe6d62794, 0x016629df]);
        assert_eq!(words(digest("a".repeat(160).as_bytes())), [0xa5e5a0bb, 0xd96dd9f8, 0x0b4b7191, 0xd63aa54a]);
        assert_eq!(words(digest("z".repeat(160).as_bytes())), [0x58ed1b7a, 0x90ede58b, 0xf9ad857c, 0x5440e613]);
        assert_eq!(words(digest("ab".repeat(80).as_bytes())), [0xf4fc06fc, 0x0bbd9ef7, 0x765ae0f7, 0x52a55925]);
    }

    /// The two-block ending (56..=63 bytes left over) against DXC's layout written out by
    /// hand: the data and 0x80 in a block of their own, then lengths around 56 zero bytes.
    #[test]
    fn digest_two_block_ending() {
        for n in [56usize, 60, 63, 120, 127] {
            let data: Vec<u8> = (0..n).map(|i| (i * 13 + 1) as u8).collect();
            let mut padded = data.clone();
            padded.push(0x80);
            while padded.len() % 64 != 0 {
                padded.push(0);
            }
            padded.extend_from_slice(&((n as u32) << 3).to_le_bytes());
            padded.extend_from_slice(&[0u8; 56]);
            padded.extend_from_slice(&(((n as u32) << 1) | 1).to_le_bytes());
            let mut state = INIT;
            assert!(blocks(&mut state, &padded).is_empty());
            assert_eq!(digest(&data), finish(state), "length {n}");
        }
    }
}
