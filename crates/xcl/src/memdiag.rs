//! Where the process's memory is, logged every 10 s as `memdiag:` lines (`XCL_MEMDIAG=0`: off).
//!
//! Every 10 s: the process counters (commit, working set, private and shareable), XCL's own Rust
//! heap (counted by the global allocator: live bytes, the peak since the last line as seen at
//! allocations of 64 KiB and more, the allocations of 64 MiB and more), the Windows heaps (HeapSummary: the process heap, which
//! XCL's Rust code, DXC and the CRT share, and every other heap, the driver's among them), the
//! D3D12 objects XCL saw the title make, by heap type (heaps and committed resources made on the
//! PC, buffer-only heaps given a unified-memory block as well), the title page segments the GPU
//! imports, the title's physical pages (allocated, live, committed, committed in freed pages) and
//! the entry counts of XCL's long-lived maps.
//!
//! Every 30 s (XCL_MEMDIAG_WALK=0: off) also a walk of the address space: committed private
//! memory by owner, mapped memory split into pagefile-backed sections, the driver's shader cache
//! files, the pipeline library and other files, and images; with an estimate of what of each is
//! resident and of that locked, which Windows cannot page out (QueryWorkingSetEx on one page in every
//! 256 KiB, XCL_MEMDIAG_RESIDENT=0: off). Each resident page sampled is also read, where its
//! protection allows (ReadProcessMemory on this process: a page gone meanwhile fails the read, and
//! a page not resident is never read, so nothing is paged in or made), and counted when all of it
//! is zeros (XCL_MEMDIAG_ZERO=0: off): committed, touched and empty memory. A page with more than
//! one mapping is counted too: the working set counts it once for each.
//!
//! Every second (XCL_MEMDIAG_POLL=0: off) the process heap less the Rust heap is noted under the
//! driver compiles running then, and the 10 s line gives its least, mean and most over the run
//! with no compile running, with no giant, with one and with more: what the heap holds when the
//! driver compiles nothing, and what a giant adds. The same line gives the Rust heap by
//! allocation size, the console shader copies alive and the pipeline stand-ins whose build (the
//! description and shaders it owns) is kept.
//! Nothing here changes what XCL does.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, Ordering::Relaxed};

const BIG: usize = 64 << 20;
/// allocations from which the peak is taken (one contended update per allocation, not two)
const PEAK_FROM: usize = 64 << 10;

/// The global allocator: System, with the bytes it holds counted.
pub struct Counting;

/// The live bytes, and of them by allocation size (`SIZES`), on one cache line: an allocation
/// writes two of them.
#[repr(C, align(64))]
struct Live {
    all: AtomicI64,
    sized: [AtomicI64; 5],
}

static LIVE: Live = Live { all: AtomicI64::new(0), sized: [const { AtomicI64::new(0) }; 5] };
static PEAK: AtomicI64 = AtomicI64::new(0);

const SIZES: [&str; 5] = ["under 1 KiB", "1-64 KiB", "64 KiB-1 MiB", "1-64 MiB", "64 MiB+"];

fn sized(n: usize) -> usize {
    match n {
        0..=0x3ff => 0,
        0x400..=0xffff => 1,
        0x1_0000..=0xf_ffff => 2,
        0x10_0000..=0x3ff_ffff => 3,
        _ => 4,
    }
}
static BIG_LIVE: AtomicI64 = AtomicI64::new(0);
static BIG_COUNT: AtomicI64 = AtomicI64::new(0);

fn grew(n: usize) {
    let now = LIVE.all.fetch_add(n as i64, Relaxed) + n as i64;
    LIVE.sized[sized(n)].fetch_add(n as i64, Relaxed);
    if n >= PEAK_FROM {
        PEAK.fetch_max(now, Relaxed);
    }
    if n >= BIG {
        BIG_LIVE.fetch_add(n as i64, Relaxed);
        BIG_COUNT.fetch_add(1, Relaxed);
    }
}

fn shrank(n: usize) {
    LIVE.all.fetch_sub(n as i64, Relaxed);
    LIVE.sized[sized(n)].fetch_sub(n as i64, Relaxed);
    if n >= BIG {
        BIG_LIVE.fetch_sub(n as i64, Relaxed);
        BIG_COUNT.fetch_sub(1, Relaxed);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        System.dealloc(p, layout);
        shrank(layout.size());
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let q = System.realloc(p, layout, size);
        if !q.is_null() {
            shrank(layout.size());
            grew(size);
        }
        q
    }
}

/// A count of objects and the bytes they hold.
pub struct Tally {
    count: AtomicI64,
    bytes: AtomicI64,
}

impl Tally {
    pub const fn new() -> Tally {
        Tally { count: AtomicI64::new(0), bytes: AtomicI64::new(0) }
    }

    pub fn add(&self, bytes: u64) {
        self.count.fetch_add(1, Relaxed);
        self.bytes.fetch_add(bytes as i64, Relaxed);
    }

    pub fn sub(&self, bytes: u64) {
        self.count.fetch_sub(1, Relaxed);
        self.bytes.fetch_sub(bytes as i64, Relaxed);
    }

    fn show(&self) -> String {
        format!("{}/{} MB", self.count.load(Relaxed), self.bytes.load(Relaxed) >> 20)
    }
}

impl Default for Tally {
    fn default() -> Tally {
        Tally::new()
    }
}

const CLASSES: [&str; 6] = ["DEFAULT", "UPLOAD", "READBACK", "CUSTOM L0", "CUSTOM L1", "other"];

/// Heaps the title made, by heap_class (made, never counted down: XCL sees no heap's release).
pub static TITLE_HEAPS: [Tally; 6] = [const { Tally::new() }; 6];
/// Of them, heaps that hold only buffers and were also given a block of unified memory, where
/// their buffers are placed (real::heap_block).
pub static BUFFER_HEAPS_IN_UMA: Tally = Tally::new();
/// Committed resources the title made that reached the PC as committed resources, by class
/// (made, never counted down).
pub static TITLE_COMMITTED: [Tally; 6] = [const { Tally::new() }; 6];
/// Title page segments imported as D3D12 heaps and alive now (physical_import).
pub static IMPORTED_SEGMENTS: Tally = Tally::new();

/// TITLE_HEAPS / TITLE_COMMITTED index of props: DEFAULT, UPLOAD, READBACK, CUSTOM in L0, in L1,
/// anything else.
pub fn heap_class(props: &windows::Win32::Graphics::Direct3D12::D3D12_HEAP_PROPERTIES) -> usize {
    use windows::Win32::Graphics::Direct3D12::*;
    match props.Type {
        D3D12_HEAP_TYPE_DEFAULT => 0,
        D3D12_HEAP_TYPE_UPLOAD => 1,
        D3D12_HEAP_TYPE_READBACK => 2,
        D3D12_HEAP_TYPE_CUSTOM if props.MemoryPoolPreference == D3D12_MEMORY_POOL_L1 => 4,
        D3D12_HEAP_TYPE_CUSTOM => 3,
        _ => 5,
    }
}

fn on(name: &str) -> bool {
    std::env::var(name).map_or(true, |v| v.trim() != "0")
}

/// Starts the thread that logs the lines (once).
pub fn start() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    if !on("XCL_MEMDIAG") {
        return;
    }
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-memdiag".into()).spawn(|| {
            let poll = on("XCL_MEMDIAG_POLL");
            let mut seen = [Seen::default(); 4];
            for tick in 1u64.. {
                for _ in 0..10 {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    if poll {
                        note_heap(&mut seen);
                    }
                }
                report(poll.then_some(&seen));
                if tick % 3 == 0 && on("XCL_MEMDIAG_WALK") {
                    walk();
                }
            }
        });
    });
}

fn mb(b: u64) -> u64 {
    b >> 20
}

/// The process heap less the Rust heap as seen with some number of driver compiles running.
#[derive(Default, Clone, Copy)]
struct Seen {
    samples: u64,
    least: u64,
    most: u64,
    sum: u64,
}

const RUNNING: [&str; 4] = ["none", "no giant", "1 giant", "2+ giants"];

/// Committed bytes of the process heap.
fn process_heap() -> u64 {
    use windows_sys::Win32::System::Memory::{GetProcessHeap, HeapSummary, HEAP_SUMMARY};
    let mut s: HEAP_SUMMARY = unsafe { std::mem::zeroed() };
    s.cb = std::mem::size_of::<HEAP_SUMMARY>() as u32;
    if unsafe { HeapSummary(GetProcessHeap(), 0, &mut s) } == 0 {
        return 0;
    }
    s.cbCommitted as u64
}

fn note_heap(seen: &mut [Seen; 4]) {
    let (running, giants, _) = crate::hle::d3d12x::real::diag_driver_compiles();
    let heap = process_heap().saturating_sub(LIVE.all.load(Relaxed).max(0) as u64);
    let s = &mut seen[match (running, giants) {
        (0, _) => 0,
        (_, 0) => 1,
        (_, 1) => 2,
        _ => 3,
    }];
    s.least = if s.samples == 0 { heap } else { s.least.min(heap) };
    s.most = s.most.max(heap);
    s.sum += heap;
    s.samples += 1;
}

fn report(seen: Option<&[Seen; 4]>) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX2};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut p = PROCESS_MEMORY_COUNTERS_EX2 { cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX2>() as u32, ..Default::default() };
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), std::ptr::from_mut(&mut p).cast(), p.cb) };
    let live = LIVE.all.load(Relaxed).max(0) as u64;
    let peak = PEAK.swap(live as i64, Relaxed).max(0) as u64;
    let (process_heap, others) = heaps();
    let largest: Vec<String> = others.iter().take(3).map(|b| format!("{} MB", mb(*b))).collect();
    crate::xlog_info!(
        "memdiag: process commit {} MB, working set {} MB ({} MB private, {} MB shareable); XCL Rust heap {} MB live (peak since the last line, at allocations of 64 KB+, {} MB; {} allocations of 64 MB+ holding {} MB); Windows heaps committed: process heap {} MB, {} other heaps {} MB (largest {})",
        mb(p.PrivateUsage as u64),
        mb(p.WorkingSetSize as u64),
        mb(p.PrivateWorkingSetSize as u64),
        mb(p.WorkingSetSize.saturating_sub(p.PrivateWorkingSetSize) as u64),
        mb(live),
        mb(peak),
        BIG_COUNT.load(Relaxed),
        BIG_LIVE.load(Relaxed) >> 20,
        mb(process_heap),
        others.len(),
        mb(others.iter().sum()),
        if largest.is_empty() { "none".to_string() } else { largest.join(", ") }
    );
    let classes = |t: &[Tally; 6]| CLASSES.iter().zip(t).map(|(n, t)| format!("{n} {}", t.show())).collect::<Vec<_>>().join(", ");
    let (local, non_local) = crate::hle::d3d12x::real::video_memory().unwrap_or((0, 0));
    crate::xlog_info!(
        "memdiag: D3D12: video memory {} MB local, {} MB non-local; title heaps made (count/MB): {}; buffer-only heaps also given unified memory {}; title committed resources made on the PC: {}; imported title page segments alive {}; {}",
        mb(local),
        mb(non_local),
        classes(&TITLE_HEAPS),
        BUFFER_HEAPS_IN_UMA.show(),
        classes(&TITLE_COMMITTED),
        IMPORTED_SEGMENTS.show(),
        crate::gpu::rebar::summary()
    );
    let (live_psos, idle, again) = crate::hle::d3d12x::pso_async::live();
    crate::xlog_info!(
        "memdiag: title physical pages: {}; caches: compute blobs {} MB; pipelines live {live_psos} (released unbound {idle}, built again {again}); XCL maps (entries): {}",
        crate::hle::kernelx::physical_page_summary(),
        crate::hle::d3d12x::real::resident_compute_blob_bytes() >> 20,
        crate::hle::d3d12x::real::diag_map_sizes()
    );
    let (running, giants, reserved) = crate::hle::d3d12x::real::diag_driver_compiles();
    let by_running = seen.map_or_else(String::new, |seen| {
        let shown: Vec<String> = RUNNING
            .iter()
            .zip(seen)
            .map(|(name, s)| match s.samples {
                0 => format!("{name} -"),
                n => format!("{name} {}/{}/{} ({n})", mb(s.least), mb(s.sum / n), mb(s.most)),
            })
            .collect();
        format!("; process heap less the Rust heap by driver compiles running, MB least/mean/most over the run (1 s samples): {}", shown.join(", "))
    });
    let sizes: Vec<String> = SIZES.iter().zip(&LIVE.sized).map(|(name, bytes)| format!("{name} {} MB", bytes.load(Relaxed).max(0) >> 20)).collect();
    let (copies, copy_bytes) = crate::hle::d3d12x::real::diag_shader_copies();
    let (standins, kept, bound) = crate::hle::d3d12x::pso_async::diag_builds();
    crate::xlog_info!(
        "memdiag: driver compiles running {running} ({giants} giants, {} MiB reserved){by_running}; Rust heap by allocation size: {}; console shader copies {copies}/{} MB; pipeline stand-ins {standins} ({kept} with their build kept, {bound} built and bound)",
        reserved >> 20,
        sizes.join(", "),
        mb(copy_bytes)
    );
}

/// Committed bytes of the process heap and of every other heap, the others largest first.
fn heaps() -> (u64, Vec<u64>) {
    use windows_sys::Win32::System::Memory::{GetProcessHeap, GetProcessHeaps, HeapSummary, HEAP_SUMMARY};
    let mut handles = vec![std::ptr::null_mut(); 256];
    let n = unsafe { GetProcessHeaps(handles.len() as u32, handles.as_mut_ptr()) } as usize;
    handles.truncate(n.min(256));
    let main = unsafe { GetProcessHeap() };
    let (mut process_heap, mut others) = (0u64, Vec::new());
    for h in handles {
        let mut s: HEAP_SUMMARY = unsafe { std::mem::zeroed() };
        s.cb = std::mem::size_of::<HEAP_SUMMARY>() as u32;
        if unsafe { HeapSummary(h, 0, &mut s) } == 0 {
            continue;
        }
        if h == main {
            process_heap = s.cbCommitted as u64;
        } else {
            others.push(s.cbCommitted as u64);
        }
    }
    others.sort_unstable_by(|a, b| b.cmp(a));
    (process_heap, others)
}

#[derive(Default, Clone, Copy)]
struct Part {
    committed: u64,
    sampled: u64,
    resident: u64,
    /// resident and locked: pinned (non-local video memory made resident is), so Windows cannot page it
    locked: u64,
    /// resident with more than one mapping (of this process or others): the working set counts each
    shared: u64,
    /// resident and read
    read: u64,
    /// of those, all zeros
    zero: u64,
}

const STRIDE: usize = 256 << 10;

const NAMES: [&str; 11] = [
    "private: coherent window",
    "private: sparse windows",
    "private: title reservations",
    "private: write-combined (behind video memory)",
    "private: other (heaps, driver, title)",
    "mapped: sections (title pages, video memory)",
    "mapped: driver shader cache",
    "mapped: pipeline library",
    "mapped: other files",
    "mapped: views under 1 MB",
    "images",
];

/// The address-space walk: committed and (sampled) resident bytes by owner.
fn walk() {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_IMAGE, MEM_MAPPED, MEM_PRIVATE};
    use windows_sys::Win32::System::ProcessStatus::K32GetMappedFileNameW;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let started = std::time::Instant::now();
    let uma = crate::gpu::uma::get();
    let resident = on("XCL_MEMDIAG_RESIDENT");
    let mut parts = [Part::default(); NAMES.len()];
    let zero = resident && on("XCL_MEMDIAG_ZERO");
    let mut samples: Vec<(usize, usize, bool)> = Vec::new();
    let mut at = 0x1_0000usize;
    while at < 0x7FFF_FFFF_0000 {
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(at as *const std::ffi::c_void, &mut m, std::mem::size_of_val(&m)) } == 0 || m.RegionSize == 0 {
            break;
        }
        let (base, size) = (m.BaseAddress as usize, m.RegionSize);
        at = base + size;
        if m.State != MEM_COMMIT {
            continue;
        }
        let part = match m.Type {
            MEM_PRIVATE if uma.is_some_and(|u| u.is_sparse(base)) => 1,
            MEM_PRIVATE if uma.is_some_and(|u| u.contains(base)) => 0,
            MEM_PRIVATE if crate::hle::kernelx::in_graphics_reservation(base) || crate::hle::kernelx::in_reservation(base) || crate::memmap::reservation(base).is_some() => 2,
            MEM_PRIVATE if m.Protect & 0x600 != 0 => 3,
            MEM_PRIVATE => 4,
            MEM_MAPPED if size < 1 << 20 => 9,
            MEM_MAPPED => {
                let mut name = [0u16; 512];
                let n = unsafe { K32GetMappedFileNameW(GetCurrentProcess(), base as *const _, name.as_mut_ptr(), name.len() as u32) } as usize;
                if n == 0 {
                    5
                } else {
                    let name = String::from_utf16_lossy(&name[..n]).to_ascii_lowercase();
                    if name.contains("dxcache") || name.contains("nv_cache") || name.contains("glcache") {
                        6
                    } else if name.contains("\\library\\") && name.contains("library-") {
                        7
                    } else {
                        8
                    }
                }
            }
            MEM_IMAGE => 10,
            _ => 4,
        };
        parts[part].committed += size as u64;
        if resident {
            // (readable, and no guard page: a read would take the guard away)
            let readable = zero && m.Protect & 0x100 == 0 && m.Protect & 0xEE != 0;
            let mut a = base.next_multiple_of(STRIDE);
            while a < base + size {
                samples.push((a, part, readable));
                a += STRIDE;
            }
        }
    }
    if resident {
        sample_resident(&samples, &mut parts);
    }
    let shown: Vec<String> = NAMES
        .iter()
        .zip(parts)
        .filter(|(_, p)| p.committed > 0)
        .map(|(n, p)| {
            if p.sampled > 0 {
                let resident = p.committed * p.resident / p.sampled;
                let zero = if p.read > 0 { format!(", zero pages ~{} MB of it", mb(resident * p.zero / p.read)) } else { String::new() };
                let shared = if p.shared > 0 { format!(", ~{} MB of it mapped more than once", mb(p.committed * p.shared / p.sampled)) } else { String::new() };
                format!("{n} {} MB (resident ~{} MB{zero}{shared}, locked ~{} MB)", mb(p.committed), mb(resident), mb(p.committed * p.locked / p.sampled))
            } else {
                format!("{n} {} MB", mb(p.committed))
            }
        })
        .collect();
    let (process_heap, others) = heaps();
    let heaps = process_heap + others.iter().sum::<u64>();
    crate::xlog_info!(
        "memdiag: address space (committed{}{}): {}; of private other: process heap {} MB, other heaps {} MB, the rest {} MB; walk {} ms",
        if resident { ", resident sampled every 256 KiB" } else { "" },
        if zero { ", zero pages among the resident ones read" } else { "" },
        shown.join("; "),
        mb(process_heap),
        mb(heaps - process_heap),
        mb(parts[4].committed.saturating_sub(heaps)),
        started.elapsed().as_millis()
    );
}

fn sample_resident(samples: &[(usize, usize, bool)], parts: &mut [Part]) {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut page = [0u8; 4096];
    for chunk in samples.chunks(16384) {
        let mut info: Vec<PSAPI_WORKING_SET_EX_INFORMATION> = chunk
            .iter()
            .map(|&(a, _, _)| {
                let mut i: PSAPI_WORKING_SET_EX_INFORMATION = unsafe { std::mem::zeroed() };
                i.VirtualAddress = a as *mut _;
                i
            })
            .collect();
        let bytes = (info.len() * std::mem::size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32;
        if unsafe { K32QueryWorkingSetEx(GetCurrentProcess(), info.as_mut_ptr().cast(), bytes) } == 0 {
            return;
        }
        for (i, &(a, part, readable)) in info.iter().zip(chunk) {
            parts[part].sampled += 1;
            let flags = unsafe { i.VirtualAttributes.Flags };
            if flags & 1 != 0 {
                parts[part].resident += 1;
                if flags & (1 << 22) != 0 {
                    parts[part].locked += 1;
                }
                // (ShareCount, bits 1 to 3)
                if (flags >> 1) & 7 > 1 {
                    parts[part].shared += 1;
                }
                let mut done = 0usize;
                if readable && unsafe { ReadProcessMemory(GetCurrentProcess(), a as *const _, page.as_mut_ptr().cast(), page.len(), &mut done) } != 0 && done == page.len() {
                    parts[part].read += 1;
                    if page.iter().all(|&b| b == 0) {
                        parts[part].zero += 1;
                    }
                }
            }
        }
    }
}
