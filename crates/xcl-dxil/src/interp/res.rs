//! The resources a run sees: buffers and textures behind register bindings and the
//! descriptor heap, generated from a seed, with every write and atomic recorded.
//!
//! Nothing here is stored until it is written: a buffer word or a texel is a hash of the
//! seed, the resource and the address, shaped by [`biased`] into values that make translated
//! programs do something (small integers, word-aligned addresses, floats with their special
//! cases). Writes go to pages (buffers) or a sparse map (textures) laid over that. D3D12
//! semantics for what both programs must agree on: reads out of bounds give 0, writes out
//! of bounds are dropped, a null descriptor reads 0 and drops writes.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// A multiplicative hasher for the integer keys used here (the default SipHash costs more
/// than the interpreter's instruction).
#[derive(Default, Clone, Copy)]
pub struct IntHasher(u64);

impl Hasher for IntHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn write_u32(&mut self, x: u32) {
        self.write_u64(x as u64)
    }
    fn write_u64(&mut self, x: u64) {
        let h = (self.0 ^ x).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = h ^ (h >> 29);
    }
}

pub type IntMap<K, V> = HashMap<K, V, BuildHasherDefault<IntHasher>>;

pub fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

const FLOATS: [u32; 32] = [
    0x0000_0000, // 0
    0x8000_0000, // -0
    0x3f80_0000, // 1
    0xbf80_0000, // -1
    0x3f00_0000, // 0.5
    0x4000_0000, // 2
    0x3e80_0000, // 0.25
    0xc000_0000, // -2
    0x4049_0fdb, // pi
    0x3dcc_cccd, // 0.1
    0x4120_0000, // 10
    0x437f_0000, // 255
    0x477f_ff00, // 65535
    0x4f80_0000, // 2^32
    0xcf00_0000, // -2^31
    0x4f00_0000, // 2^31
    0x7f80_0000, // inf
    0xff80_0000, // -inf
    0x7fc0_0000, // NaN
    0xffc0_0000, // -NaN
    0x7f80_0001, // signalling NaN
    0x7fc1_2345, // NaN with a payload
    0x0000_0001, // smallest denormal
    0x007f_ffff, // largest denormal
    0x0080_0000, // smallest normal
    0x8000_0001, // negative denormal
    0x7f7f_ffff, // largest finite
    0xff7f_ffff, // most negative finite
    0x3f7f_ffff, // just below 1
    0x3f80_0001, // just above 1
    0x3eaa_aaab, // 1/3
    0x4b80_0000, // 2^24
];

/// A hash shaped into a word a translated program is likely to do something with.
pub fn biased(h: u64) -> u32 {
    let r = (h >> 8) as u32;
    match h & 0xff {
        0..=59 => r & 15,
        60..=89 => r & 255,
        90..=113 => (r & 0x3fff) << 2,
        114..=129 => 0,
        130..=145 => match r & 7 {
            0 => u32::MAX,
            1 => 0x8000_0000,
            2 => 0x7fff_ffff,
            3 => 1 << ((r >> 3) & 31),
            4 => (1u32 << ((r >> 3) & 31)).wrapping_sub(1),
            5 => 0xffff,
            6 => 0xff,
            _ => !(1 << ((r >> 3) & 31)),
        },
        146..=175 => FLOATS[(r & 31) as usize],
        176..=199 => match r & 3 {
            // small integers and fractions as floats
            0 => (((r >> 2) & 63) as f32).to_bits(),
            1 => (((r >> 2) & 255) as f32 / 16.0 - 8.0).to_bits(),
            2 => (((r >> 2) & 0xffff) as f32 / 65535.0).to_bits(),
            _ => (((r >> 2) & 0xfff) as f32 * 0.37 - 700.0).to_bits(),
        },
        _ => (h >> 32) as u32,
    }
}

/// A float texel component: mostly a colour in [0, 1], sometimes anything.
fn biased_texel(h: u64) -> u32 {
    let r = (h >> 8) as u32;
    match h & 0xff {
        0..=159 => ((r & 255) as f32 / 255.0).to_bits(),
        160..=199 => ((r & 0xffff) as f32 / 4096.0 - 4.0).to_bits(),
        200..=239 => FLOATS[(r & 31) as usize],
        _ => (h >> 32) as u32,
    }
}

const PAGE_WORDS: usize = 1024;

struct Page {
    words: Box<[u32; PAGE_WORDS]>,
    written: [u64; PAGE_WORDS / 64],
    /// who wrote each word last and when ([`Clock`])
    owner: Box<[u32; PAGE_WORDS]>,
    stamp: Box<[u32; PAGE_WORDS]>,
}

/// Who is accessing memory and when, for the race detector. Two lanes of a wave race when
/// one writes what the other reads or writes and the order of the two accesses is not
/// fixed by the program: they ran in different block executions since the wave's lanes
/// were last all together (which side of a branch runs first is the scheduler's choice,
/// and differs between two shapes of one program). Two waves race when one writes what
/// the other reads or writes with no group barrier between the two accesses, unless both
/// are atomics: only atomics are scheduling points, and a compiler may move any other
/// access across one that it proves to be elsewhere.
#[derive(Default, Clone, Copy, Debug)]
pub struct Clock {
    /// (wave + 1) << 8 | lane, with [ATOMIC_OWNER] while an atomic runs
    pub me: u32,
    /// counts block executions and memory-clobbering calls
    pub tick: u32,
    /// the tick the running block execution began at
    pub block_start: u32,
    /// the tick of the wave's last block execution with all its lanes
    pub region_start: u32,
    /// the tick the group began at or last passed a group barrier at
    pub epoch_start: u32,
}

/// In an owner: the word was written by an atomic.
pub const ATOMIC_OWNER: u32 = 1 << 31;

impl Clock {
    #[inline]
    pub fn same_wave_other_lane(&self, owner: u32) -> bool {
        let (o, m) = (owner & !ATOMIC_OWNER, self.me & !ATOMIC_OWNER);
        o != 0 && o != m && o >> 8 == m >> 8
    }
    /// Whether an access now conflicts with a write by `owner` at `stamp`.
    #[inline]
    pub fn races(&self, owner: u32, stamp: u32) -> bool {
        let (o, m) = (owner & !ATOMIC_OWNER, self.me & !ATOMIC_OWNER);
        if o == 0 || o == m {
            false
        } else if o >> 8 == m >> 8 {
            stamp >= self.region_start && stamp < self.block_start
        } else {
            stamp >= self.epoch_start && owner & self.me & ATOMIC_OWNER == 0
        }
    }
}

pub struct Buffer {
    pub ident: u32,
    /// bytes; 0 for the null view
    pub size: u64,
    pages: IntMap<u32, u32>,
    store: Vec<Page>,
    /// the page touched last and its index in store
    last: (u32, u32),
    pub counter: u32,
}

pub struct Texture {
    pub ident: u32,
    /// width, height, depth or array size, mip levels; all 0 for the null view
    pub dims: [u32; 4],
}

/// One atomic operation, for the order-independent comparison.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Atomic {
    /// resource identity; [`SHARED`] for groupshared memory
    pub res: u32,
    pub addr: u64,
    /// the DXIL AtomicBinOpCode, or [`ATOMIC_CMPXCHG`] / [`ATOMIC_COUNTER`]
    pub op: u8,
    pub a: u64,
    pub b: u64,
}

pub const SHARED: u32 = u32::MAX;
/// In a [Written] output: the pixel was discarded (and has no other outputs).
pub const DISCARDED: u32 = u32::MAX;
/// DXIL SemanticKind values input generation knows.
pub const SEM_VERTEX_ID: u8 = 1;
pub const SEM_INSTANCE_ID: u8 = 2;
pub const SEM_POSITION: u8 = 3;
pub const SEM_PRIMITIVE_ID: u8 = 10;
pub const SEM_SAMPLE_INDEX: u8 = 12;
pub const SEM_IS_FRONT_FACE: u8 = 13;
pub const SEM_COVERAGE: u8 = 14;
/// What the hash of an input is taken under: no resource has this identity.
const IDENT_INPUT: u32 = IDENT_BOUND | 3 << 28 | 0xf_ffff;
pub const ATOMIC_CMPXCHG: u8 = 100;
pub const ATOMIC_COUNTER: u8 = 101;
const ATOMIC_SAMPLE: usize = 1 << 16;

/// The multiset of atomics of a run: a commutative hash and a count, and the first ones
/// kept for the report.
#[derive(Default, Clone, PartialEq, Eq, Debug)]
pub struct Atomics {
    pub count: u64,
    pub hash: u64,
    /// the hash with every NaN operand replaced by one NaN ([any_nan])
    pub hash_any_nan: u64,
    pub sample: Vec<Atomic>,
}

/// A value with the payload of a NaN removed: a 32-bit pattern that is a NaN as a float,
/// or a 64-bit one that is a NaN as a double, becomes the quiet NaN of its width.
pub fn any_nan(v: u64) -> u64 {
    if v >> 32 == 0 && v & 0x7fff_ffff > 0x7f80_0000 {
        0x7fc0_0000
    } else if v & (u64::MAX >> 1) > 0x7ff0_0000_0000_0000 {
        0x7ff8_0000_0000_0000
    } else {
        v
    }
}

impl Atomics {
    pub fn record(&mut self, a: Atomic) {
        self.count += 1;
        let hash = |x: u64, y: u64| mix(mix(mix(a.res as u64 ^ (a.op as u64) << 32) ^ a.addr) ^ x).wrapping_add(mix(y ^ 0x5555));
        self.hash = self.hash.wrapping_add(hash(a.a, a.b));
        self.hash_any_nan = self.hash_any_nan.wrapping_add(hash(any_nan(a.a), any_nan(a.b)));
        if self.sample.len() < ATOMIC_SAMPLE {
            self.sample.push(a);
        }
    }
}

/// A written cell of the final state: (0 buffer word / 1 texel / 2 output of a vertex or
/// pixel, resource, address, value). An output's resource is the id of its signature
/// element ([DISCARDED] for the mark of a discarded pixel) and its address is
/// lane << 16 | row << 8 | column.
pub type Written = (u8, u32, u64, [u32; 4]);

#[derive(Default, Clone, Copy, Debug)]
pub struct Access {
    pub buffer_reads: u64,
    pub buffer_writes: u64,
    pub texture_reads: u64,
    pub texture_writes: u64,
    pub samples: u64,
    pub out_of_bounds: u64,
    pub null: u64,
}

/// Resource identities: the tag in the top two bits. A bound resource is its class, space
/// and register (the same in two compilations of one program); a heap one its index.
pub const IDENT_NULL: u32 = 0;
pub const IDENT_BOUND: u32 = 1 << 30;
pub const IDENT_HEAP: u32 = 2 << 30;
pub const IDENT_SAMPLER_HEAP: u32 = 3 << 30;
/// Descriptors of the modelled heap (the D3D12 limit for a CBV / SRV / UAV heap).
pub const HEAP_SIZE: u32 = 1_000_000;
/// The translator's `io` buffer: a read-write buffer at u0, space 0.
pub const IDENT_IO: u32 = IDENT_BOUND | 1 << 28;
/// Words of `io` (`gpu::recomp::emit`): the first of the heap slots of the memory windows
/// (a 64-bit address is a window, by its high half, and an offset in it), the high half of
/// the first window's addresses, and the mask of the lanes that execute.
const IO_UMA_BASE: u64 = 112;
const IO_MEM_VIEWS: u64 = 113;
const IO_EXEC_LO: u64 = 126;
const IO_EXEC_HI: u64 = 127;
/// Memory windows a program can address (the last takes every other high half).
const MEM_WINDOWS: u32 = 5;

pub struct World {
    pub seed: u64,
    buffers: Vec<Buffer>,
    buffer_ix: IntMap<u32, u32>,
    last_buffer: (u32, u32),
    textures: Vec<Texture>,
    texture_ix: IntMap<u32, u32>,
    last_texture: (u32, u32),
    texels: IntMap<u64, ([u32; 4], u32, u32)>,
    pub atomics: Atomics,
    pub access: Access,
    /// the heap slot of the first memory window ([`IO_MEM_VIEWS`])
    views: u32,
    pub clock: Clock,
    /// accesses that raced (see [`Clock`])
    pub races: u64,
    /// which blocks of the program ran (by the interpreter's block number)
    pub covered: Vec<bool>,
    /// generated buffer words replaced by the input ((resource << 32 | word) → value)
    pub patches: IntMap<u64, u32>,
    /// when exploring: what the program read and compared
    pub trace: Option<Trace>,
    /// outputs of vertices and pixels: element << 32 | lane << 16 | row << 8 | column
    pub outputs: IntMap<u64, u32>,
    /// some ballots without a set bit are given set bits ([`super::Config::force_ballots`])
    pub force_ballots: bool,
}

/// What a run looked at, for [`super::explore`] to change.
#[derive(Default, Debug, Clone)]
pub struct Trace {
    seen: IntMap<u64, ()>,
    /// generated buffer words read: (resource << 32 | word, value), first reads first
    pub reads: Vec<(u64, u32)>,
    /// operand pairs of comparisons that were not equal, and of switches and their cases:
    /// the last ones of the run (where it stopped going further), oldest first once
    /// [Trace::finish] has run
    pub compares: Vec<(u64, u64)>,
    /// where the next pair goes once `compares` is full
    next: usize,
    last: (u64, u64),
}

const TRACE_READS: usize = 1 << 16;
const TRACE_COMPARES: usize = 1 << 12;

impl Trace {
    pub fn compare(&mut self, x: u64, y: u64) {
        if x == y || self.last == (x, y) {
            return;
        }
        self.last = (x, y);
        if self.compares.len() < TRACE_COMPARES {
            self.compares.push((x, y));
        } else {
            self.compares[self.next] = (x, y);
            self.next = (self.next + 1) % TRACE_COMPARES;
        }
    }

    /// Put the pairs in the order they were met.
    pub fn finish(&mut self) {
        self.compares.rotate_left(self.next);
        self.next = 0;
    }
}

impl World {
    pub fn new(seed: u64) -> World {
        let seed = mix(seed ^ 0x7863_6c5f_6478_696c);
        World {
            seed,
            views: (mix(seed ^ 0x7669_6577) % (HEAP_SIZE - MEM_WINDOWS) as u64) as u32,
            buffers: Vec::new(),
            buffer_ix: IntMap::default(),
            last_buffer: (IDENT_NULL, u32::MAX),
            textures: Vec::new(),
            texture_ix: IntMap::default(),
            last_texture: (IDENT_NULL, u32::MAX),
            texels: IntMap::default(),
            atomics: Atomics::default(),
            access: Access::default(),
            clock: Clock::default(),
            races: 0,
            covered: Vec::new(),
            patches: IntMap::default(),
            trace: None,
            outputs: IntMap::default(),
            force_ballots: false,
        }
    }

    fn hash(&self, ident: u32, addr: u64, salt: u64) -> u64 {
        mix(self.seed ^ mix((ident as u64) << 32 ^ salt.wrapping_mul(0x9e37_79b9)) ^ addr.wrapping_mul(0xd6e8_feb8_6659_fd93))
    }

    /// Whether the descriptor at a heap slot is the null view (bound resources and the
    /// memory windows never are).
    fn is_null(&self, ident: u32) -> bool {
        match ident >> 30 {
            0 => true,
            1 => false,
            _ => !self.is_window(ident) && self.hash(ident, 0, 1) % 32 == 0,
        }
    }

    fn is_window(&self, ident: u32) -> bool {
        ident >> 30 == 2 && (ident & 0x3fff_ffff).wrapping_sub(self.views) < MEM_WINDOWS
    }

    /// The generated content of a buffer word. The translator's programs read their state
    /// from `io` (u0): scalar registers in the first words, then 256 words of vector
    /// registers per lane, and a few words that everything else hangs on, which get values
    /// that let the program run: the heap slot of the memory windows is a valid one, the
    /// execution mask is often full.
    fn generated(&self, ident: u32, w: u64) -> u32 {
        if !self.patches.is_empty() {
            if let Some(&v) = self.patches.get(&((ident as u64) << 32 | w)) {
                return v;
            }
        }
        let h = self.hash(ident, w, 0);
        if ident == IDENT_IO {
            match w {
                IO_MEM_VIEWS => return self.views,
                IO_UMA_BASE => return (h >> 8) as u32 & 7,
                IO_EXEC_LO | IO_EXEC_HI if h & 0x300 != 0 => return u32::MAX,
                _ => {}
            }
        }
        biased(h)
    }

    fn buffer(&mut self, ident: u32) -> usize {
        if self.last_buffer.0 == ident && self.last_buffer.1 != u32::MAX {
            return self.last_buffer.1 as usize;
        }
        let ix = match self.buffer_ix.get(&ident) {
            Some(&ix) => ix,
            None => {
                let size = if self.is_null(ident) {
                    0
                } else if ident >> 30 == 1 {
                    1 << 20
                } else if self.is_window(ident) {
                    1 << 32
                } else {
                    match self.hash(ident, 0, 2) % 16 {
                        0 => 256,
                        1 => 4096,
                        2 => 1 << 16,
                        3 => 1 << 24,
                        _ => 1 << 32,
                    }
                };
                self.buffers.push(Buffer { ident, size, pages: IntMap::default(), store: Vec::new(), last: (u32::MAX, 0), counter: 0 });
                let ix = self.buffers.len() as u32 - 1;
                self.buffer_ix.insert(ident, ix);
                ix
            }
        };
        self.last_buffer = (ident, ix);
        ix as usize
    }

    pub fn buffer_size(&mut self, ident: u32) -> u64 {
        let b = self.buffer(ident);
        self.buffers[b].size
    }

    fn word(&mut self, b: usize, w: u64) -> u32 {
        let buf = &self.buffers[b];
        let page = (w / PAGE_WORDS as u64) as u32;
        let at = w as usize % PAGE_WORDS;
        let p = if buf.last.0 == page {
            Some(buf.last.1)
        } else if buf.pages.is_empty() {
            None
        } else {
            buf.pages.get(&page).copied()
        };
        let (v, generated) = match p {
            None => (self.generated(buf.ident, w), true),
            Some(p) => {
                let data = &buf.store[p as usize];
                if self.clock.races(data.owner[at], data.stamp[at]) {
                    self.races += 1;
                }
                (data.words[at], data.written[at / 64] >> (at % 64) & 1 == 0)
            }
        };
        if generated {
            if let Some(trace) = &mut self.trace {
                let key = (self.buffers[b].ident as u64) << 32 | w;
                if trace.reads.len() < TRACE_READS && trace.seen.insert(key, ()).is_none() {
                    trace.reads.push((key, v));
                }
            }
        }
        v
    }

    fn word_mut(&mut self, b: usize, w: u64) -> &mut u32 {
        let page = (w / PAGE_WORDS as u64) as u32;
        let at = w as usize % PAGE_WORDS;
        if self.buffers[b].last.0 != page {
            let p = match self.buffers[b].pages.get(&page) {
                Some(&p) => p,
                None => {
                    let mut words = Box::new([0u32; PAGE_WORDS]);
                    let ident = self.buffers[b].ident;
                    let base = page as u64 * PAGE_WORDS as u64;
                    for (k, word) in words.iter_mut().enumerate() {
                        *word = self.generated(ident, base + k as u64);
                    }
                    let buf = &mut self.buffers[b];
                    buf.store.push(Page { words, written: [0; PAGE_WORDS / 64], owner: Box::new([0; PAGE_WORDS]), stamp: Box::new([0; PAGE_WORDS]) });
                    let p = buf.store.len() as u32 - 1;
                    buf.pages.insert(page, p);
                    p
                }
            };
            self.buffers[b].last = (page, p);
        }
        let buf = &mut self.buffers[b];
        let page = &mut buf.store[buf.last.1 as usize];
        page.written[at / 64] |= 1 << (at % 64);
        if self.clock.races(page.owner[at], page.stamp[at]) {
            self.races += 1;
        }
        page.owner[at] = self.clock.me;
        page.stamp[at] = self.clock.tick;
        &mut page.words[at]
    }

    /// Read `bytes` (1 to 8) at a byte address; 0 when any of them is out of bounds.
    pub fn read(&mut self, ident: u32, addr: u64, bytes: u32) -> u64 {
        let b = self.buffer(ident);
        self.access.buffer_reads += 1;
        let size = self.buffers[b].size;
        if addr.checked_add(bytes as u64).map_or(true, |end| end > size) {
            if size == 0 {
                self.access.null += 1;
            } else {
                self.access.out_of_bounds += 1;
            }
            return 0;
        }
        if addr & 3 == 0 && bytes == 4 {
            return self.word(b, addr >> 2) as u64;
        }
        let mut v = 0u64;
        for k in 0..bytes as u64 {
            let a = addr + k;
            v |= ((self.word(b, a >> 2) >> ((a & 3) * 8)) as u64 & 0xff) << (k * 8);
        }
        v
    }

    /// Write `bytes` (1 to 8) at a byte address; dropped when any of them is out of bounds.
    /// Returns whether it was written.
    pub fn write(&mut self, ident: u32, addr: u64, bytes: u32, v: u64) -> bool {
        let b = self.buffer(ident);
        self.access.buffer_writes += 1;
        let size = self.buffers[b].size;
        if addr.checked_add(bytes as u64).map_or(true, |end| end > size) {
            if size == 0 {
                self.access.null += 1;
            } else {
                self.access.out_of_bounds += 1;
            }
            return false;
        }
        if addr & 3 == 0 && bytes == 4 {
            *self.word_mut(b, addr >> 2) = v as u32;
            return true;
        }
        for k in 0..bytes as u64 {
            let a = addr + k;
            let shift = (a & 3) * 8;
            let word = self.word_mut(b, a >> 2);
            *word = *word & !(0xff << shift) | (((v >> (k * 8)) & 0xff) as u32) << shift;
        }
        true
    }

    /// The hidden counter of a structured buffer: add `delta` and return (before, after).
    pub fn counter(&mut self, ident: u32, delta: i32) -> (u32, u32) {
        let b = self.buffer(ident);
        let before = self.buffers[b].counter;
        let after = before.wrapping_add(delta as u32);
        if self.buffers[b].size != 0 {
            self.buffers[b].counter = after;
        }
        (before, after)
    }

    /// A word of a constant buffer (never written).
    pub fn constant(&mut self, ident: u32, word: u64) -> u32 {
        if self.is_null(ident) {
            return 0;
        }
        biased(self.hash(ident, word, 3))
    }

    fn texture(&mut self, ident: u32) -> usize {
        if self.last_texture.0 == ident && self.last_texture.1 != u32::MAX {
            return self.last_texture.1 as usize;
        }
        let ix = match self.texture_ix.get(&ident) {
            Some(&ix) => ix,
            None => {
                let dims = if self.is_null(ident) {
                    [0; 4]
                } else {
                    let h = self.hash(ident, 0, 4);
                    let w = 1u32 << (h % 7);
                    let ht = 1u32 << ((h >> 8) % 7);
                    let d = 1u32 << ((h >> 16) % 4);
                    let mips = 1 + ((h >> 24) % (1 + w.max(ht).trailing_zeros() as u64)) as u32;
                    [w, ht, d, mips]
                };
                self.textures.push(Texture { ident, dims });
                let ix = self.textures.len() as u32 - 1;
                self.texture_ix.insert(ident, ix);
                ix
            }
        };
        self.last_texture = (ident, ix);
        ix as usize
    }

    /// (width, height, depth or array size, mip levels) at mip 0.
    pub fn texture_dims(&mut self, ident: u32) -> [u32; 4] {
        let t = self.texture(ident);
        self.textures[t].dims
    }

    /// The texel key if (x, y, z, mip) is inside the texture. `dims`: how many of x, y, z the
    /// texture's kind uses; the others are ignored.
    fn texel_key(&mut self, ident: u32, coord: [u32; 3], mip: u32, used: [bool; 3]) -> Option<(usize, u64)> {
        let t = self.texture(ident);
        let d = self.textures[t].dims;
        if d[3] == 0 {
            self.access.null += 1;
            return None;
        }
        let mut key = (t as u64) << 44 | (mip as u64 & 15) << 40;
        let mut inside = mip < d[3];
        for k in 0..3 {
            if !used[k] {
                continue;
            }
            // only width and height shrink with the mip level here (array slices do not, and
            // a volume's depth is modelled alike)
            let extent = if k < 2 { (d[k] >> mip.min(31)).max(1) } else { d[2] };
            inside &= coord[k] < extent;
            key |= (coord[k] as u64 & 0x3fff) << (14 * k);
        }
        if !inside {
            self.access.out_of_bounds += 1;
            return None;
        }
        Some((t, key))
    }

    /// Load a texel: raw bits of four components (0 out of bounds). `float`: the view's
    /// component type, which shapes generated values.
    pub fn texel(&mut self, ident: u32, coord: [u32; 3], mip: u32, used: [bool; 3], float: bool) -> [u32; 4] {
        self.access.texture_reads += 1;
        let Some((_, key)) = self.texel_key(ident, coord, mip, used) else { return [0; 4] };
        if !self.texels.is_empty() {
            if let Some(&(v, owner, stamp)) = self.texels.get(&key) {
                if self.clock.races(owner, stamp) {
                    self.races += 1;
                }
                return v;
            }
        }
        let mut out = [0; 4];
        for (c, o) in out.iter_mut().enumerate() {
            let h = self.hash(ident, key & ((1 << 44) - 1), 8 + c as u64);
            *o = if float { biased_texel(h) } else { biased(h) };
        }
        out
    }

    /// Store a texel (dropped out of bounds). Returns whether it was written.
    pub fn set_texel(&mut self, ident: u32, coord: [u32; 3], mip: u32, used: [bool; 3], v: [u32; 4]) -> bool {
        self.access.texture_writes += 1;
        let Some((_, key)) = self.texel_key(ident, coord, mip, used) else { return false };
        if let Some((_, owner, stamp)) = self.texels.insert(key, (v, self.clock.me, self.clock.tick)) {
            if self.clock.races(owner, stamp) {
                self.races += 1;
            }
        }
        true
    }

    /// The result of a sample, gather or LOD query: a function of the texture, the sampler
    /// and every operand's bits (filtering is not modelled, and stores do not feed back).
    pub fn sample(&mut self, texture: u32, sampler: u32, op: u32, operands: &[u64]) -> [u32; 4] {
        self.access.samples += 1;
        let t = self.texture(texture);
        if self.textures[t].dims[3] == 0 {
            self.access.null += 1;
            return [0; 4];
        }
        let mut h = self.hash(texture, sampler as u64, 16 + op as u64);
        for &o in operands {
            h = mix(h ^ o);
        }
        let mut out = [0; 4];
        for (c, o) in out.iter_mut().enumerate() {
            let x = mix(h ^ (c as u64 + 1) << 56);
            *o = ((x >> 40) as f32 / (1u64 << 24) as f32).to_bits();
        }
        out
    }

    /// The value of a component of an input signature element for a lane (a vertex, or a
    /// pixel of the quad lane / 4 at (lane & 1, lane >> 1 & 1) in it), as raw bits.
    ///
    /// A vertex gets words of its own, but for SV_VertexID (consecutive) and
    /// SV_InstanceID (one for the wave). In a pixel program SV_Position is the centre of
    /// the pixel, with a depth in [0, 1) and a w in [0.5, 1.5) that are those of the quad;
    /// an integer or a `flat` (nointerpolation) element is the same in the four pixels of
    /// a quad, as on one primitive; any other float lies on a plane through the quad, as
    /// an interpolated attribute does, so derivatives are small and consistent.
    #[allow(clippy::too_many_arguments)]
    pub fn input(&self, pixel: bool, kind: u8, flat: bool, element: u32, row: u32, col: u32, lane: u32, float: bool) -> u32 {
        let addr = (element as u64) << 16 | (row as u64 & 0xff) << 8 | col as u64 & 0xff;
        let small = |h: u64| match h & 7 {
            0 => (h >> 32) as u32,
            1 => 0,
            _ => (h >> 8) as u32 & 0x3f,
        };
        if !pixel {
            let h = self.hash(IDENT_INPUT, addr, 1 << 32 | lane as u64);
            let of_wave = self.hash(IDENT_INPUT, addr, 0);
            return match kind {
                SEM_VERTEX_ID => (of_wave % 4096) as u32 + lane,
                SEM_INSTANCE_ID => (of_wave % 64) as u32,
                _ if float => biased(h),
                _ => small(h),
            };
        }
        let (quad, q) = ((lane / 4) as u64, lane % 4);
        let (qx, qy) = ((q & 1) as f32, (q >> 1) as f32);
        let h = self.hash(IDENT_INPUT, addr, 2 << 32 | quad);
        if !float {
            return match kind {
                SEM_IS_FRONT_FACE => h as u32 & 1,
                SEM_SAMPLE_INDEX => 0,
                SEM_COVERAGE => 1,
                _ => small(h),
            };
        }
        if kind == SEM_POSITION {
            let unit = (h >> 40) as f32 / (1u64 << 24) as f32;
            let at = |axis: u64| ((self.hash(IDENT_INPUT, axis, 3 << 32 | quad) % 512) * 2) as f32;
            return match col {
                0 => at(0) + qx + 0.5,
                1 => at(1) + qy + 0.5,
                2 => unit,
                _ => 0.5 + unit,
            }
            .to_bits();
        }
        let centre = f32::from_bits(biased(h));
        if flat {
            return centre.to_bits();
        }
        let slope = |axis: u64| {
            let s = self.hash(IDENT_INPUT, addr, (4 + axis) << 32 | quad);
            if s & 3 == 0 {
                0.0
            } else {
                ((s >> 8) % 33) as f32 / 64.0 - 0.25
            }
        };
        (centre + qx * slope(0) + qy * slope(1)).to_bits()
    }

    /// Record an output of a lane (the last store of a component stands).
    pub fn store_output(&mut self, element: u32, row: u32, col: u32, lane: u32, v: u32) {
        self.outputs.insert((element as u64) << 32 | (lane as u64 & 0xffff) << 16 | (row as u64 & 0xff) << 8 | col as u64 & 0xff, v);
    }

    /// A pixel was discarded: it has no outputs but the mark of it.
    pub fn discard(&mut self, lane: u32) {
        let lane = lane as u64 & 0xffff;
        self.outputs.retain(|key, _| key >> 16 & 0xffff != lane);
        self.outputs.insert((DISCARDED as u64) << 32 | lane << 16, 1);
    }

    /// Every written cell, sorted: the final state the comparison is about.
    pub fn written(&self) -> Vec<Written> {
        let mut out = Vec::new();
        for (&key, &v) in &self.outputs {
            out.push((2, (key >> 32) as u32, key & 0xffff_ffff, [v, 0, 0, 0]));
        }
        for buf in &self.buffers {
            for (&page, &p) in &buf.pages {
                let page_data = &buf.store[p as usize];
                for (k, bits) in page_data.written.iter().enumerate() {
                    let mut bits = *bits;
                    while bits != 0 {
                        let at = k * 64 + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        out.push((0, buf.ident, (page as u64 * PAGE_WORDS as u64 + at as u64) * 4, [page_data.words[at], 0, 0, 0]));
                    }
                }
            }
        }
        for (&key, v) in &self.texels {
            let t = &self.textures[(key >> 44) as usize];
            out.push((1, t.ident, key & ((1 << 44) - 1), v.0));
        }
        out.sort_unstable();
        out
    }
}

/// How a written cell's address reads in a report.
pub fn describe(w: &Written) -> String {
    if w.0 == 2 {
        let lane = w.2 >> 16;
        return if w.1 == DISCARDED { format!("discard of lane {lane}") } else { format!("output {} row {} column {} of lane {lane}", w.1, w.2 >> 8 & 0xff, w.2 & 0xff) };
    }
    let res = match w.1 >> 30 {
        1 => format!("{}{}{}", ["t", "u", "b", "s"][(w.1 >> 28 & 3) as usize], w.1 & 0xf_ffff, if w.1 >> 20 & 0xff != 0 { format!(" space {}", w.1 >> 20 & 0xff) } else { String::new() }),
        2 => format!("heap [{}]", w.1 & 0x3fff_ffff),
        3 => format!("sampler heap [{}]", w.1 & 0x3fff_ffff),
        _ => "null".to_string(),
    };
    if w.0 == 0 {
        format!("buffer {res} +{:#x}", w.2)
    } else {
        format!("texture {res} mip {} ({}, {}, {})", w.2 >> 40, w.2 & 0x3fff, (w.2 >> 14) & 0x3fff, (w.2 >> 28) & 0xfff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_are_stable_and_writes_stick() {
        let mut w = World::new(7);
        let id = IDENT_BOUND;
        let a = w.read(id, 64, 4);
        assert_eq!(w.read(id, 64, 4), a);
        assert!(w.write(id, 64, 4, 0xdead_beef));
        assert_eq!(w.read(id, 64, 4), 0xdead_beef);
        // neighbours keep their generated values after the page is materialised
        let mut fresh = World::new(7);
        assert_eq!(w.read(id, 68, 4), fresh.read(id, 68, 4));
        assert_eq!(w.written(), vec![(0, id, 64, [0xdead_beef, 0, 0, 0])]);
    }

    #[test]
    fn unaligned_and_wide_accesses() {
        let mut w = World::new(1);
        let id = IDENT_BOUND;
        w.write(id, 0, 4, 0x4433_2211);
        w.write(id, 4, 4, 0x8877_6655);
        assert_eq!(w.read(id, 1, 4), 0x5544_3322);
        assert_eq!(w.read(id, 0, 8), 0x8877_6655_4433_2211);
        assert_eq!(w.read(id, 2, 2), 0x4433);
        w.write(id, 3, 2, 0xbbaa);
        assert_eq!(w.read(id, 0, 8), 0x8877_66bb_aa33_2211);
    }

    #[test]
    fn out_of_bounds_reads_zero_and_drops_writes() {
        let mut w = World::new(3);
        let id = IDENT_BOUND;
        let size = w.buffer_size(id);
        assert_eq!(w.read(id, size - 2, 4), 0);
        assert!(!w.write(id, size, 4, 1));
        assert_eq!(w.read(IDENT_NULL, 0, 4), 0);
        assert!(!w.write(IDENT_NULL, 0, 4, 1));
        assert!(w.written().is_empty());
    }

    #[test]
    fn texels_round_trip() {
        let mut w = World::new(5);
        let id = (0..64).map(|k| IDENT_HEAP | k).find(|id| w.texture_dims(*id)[3] != 0).unwrap();
        let used = [true, true, false];
        let before = w.texel(id, [0, 0, 0], 0, used, true);
        assert_eq!(w.texel(id, [0, 0, 9], 0, used, true), before);
        assert!(w.set_texel(id, [0, 0, 0], 0, used, [1, 2, 3, 4]));
        assert_eq!(w.texel(id, [0, 0, 0], 0, used, true), [1, 2, 3, 4]);
        assert_eq!(w.texel(id, [1000, 0, 0], 0, used, true), [0; 4]);
        assert!(!w.set_texel(id, [1000, 0, 0], 0, used, [9; 4]));
        assert_eq!(w.written().len(), 1);
    }
}
