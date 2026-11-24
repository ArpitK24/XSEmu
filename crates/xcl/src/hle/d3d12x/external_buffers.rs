use std::collections::HashMap;
use std::ffi::c_void;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use windows_sys::Win32::System::Memory::{
    VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_READWRITE, PAGE_WRITECOPY,
};

use super::title_textures::{GetWriteWatch, REFRESH};

const PAGE: usize = 4096;

#[derive(Clone, Copy)]
pub(super) struct Backing {
    pub title: usize,
    pub mirror: usize,
    pub bytes: usize,
    /// outside write-watched memory (views of physical pages): what the title's bytes held at the
    /// last look (0: watched)
    seen: usize,
    /// where the next look's comparison starts
    cursor: usize,
}

/// What each look compares of a buffer outside write-watched memory (XCL's own writes there are
/// reported at once: `physical::note_written`; the title's are found within a few looks)
const LOOK_BYTES: usize = 1 << 20;

static BUFFERS: Mutex<Option<HashMap<usize, Backing>>> = Mutex::new(None);
static GENERATION: AtomicU64 = AtomicU64::new(0);

pub(super) fn generation() -> u64 {
    GENERATION.load(Ordering::Acquire)
}

pub(super) fn ranges() -> Vec<(usize, usize)> {
    BUFFERS.lock().unwrap().as_ref().map_or_else(Vec::new, |b| {
        b.values().map(|b| (b.title & !(PAGE - 1), b.title + b.bytes)).collect()
    })
}

pub(super) fn get(res: usize) -> Option<Backing> {
    BUFFERS.lock().unwrap().as_ref()?.get(&res).copied()
}

/// Why a title buffer outside unified memory cannot be mirrored (None: it can).
pub(super) fn unsupported(title: usize, bytes: usize) -> Option<&'static str> {
    let Some(end) = title.checked_add(bytes) else {
        return Some("its end overflows");
    };
    if bytes == 0 {
        return Some("it is empty");
    }
    crate::hle::kernelx::lazy::ensure(title, bytes);
    if !writable(title, end) {
        return Some("its memory is not committed read-write");
    }
    None
}

fn watched(title: usize, end: usize) -> bool {
    let mut covered = title;
    for (base, size) in crate::hle::kernelx::graphics_reservations() {
        if base + size <= covered {
            continue;
        }
        if base > covered {
            return false;
        }
        covered = base + size;
        if covered >= end {
            return true;
        }
    }
    false
}

fn writable(mut start: usize, end: usize) -> bool {
    if let Some(a) = crate::memmap::access(start, end.saturating_sub(start)) {
        return a == crate::memmap::Access::Write;
    }
    while start < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        if unsafe {
            VirtualQuery(
                start as *const c_void,
                &mut info,
                std::mem::size_of_val(&info),
            )
        } == 0
            || info.State != MEM_COMMIT
            || info.Protect & PAGE_GUARD != 0
            || !matches!(info.Protect & 0xFF, PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)
        {
            return false;
        }
        start = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

pub(super) unsafe fn register(res: usize, title: usize, mirror: usize, bytes: usize) -> bool {
    let mut backing = Backing { title, mirror, bytes, seen: 0, cursor: 0 };
    let mut g = BUFFERS.lock().unwrap();
    let buffers = g.get_or_insert_with(HashMap::new);
    if buffers
        .values()
        .any(|b| b.title < backing.title + backing.bytes && backing.title < b.title + b.bytes)
    {
        return false;
    }
    std::ptr::copy_nonoverlapping(
        backing.title as *const u8,
        backing.mirror as *mut u8,
        backing.bytes,
    );
    if !watched(title, title + bytes) {
        let seen = std::slice::from_raw_parts(title as *const u8, bytes).to_vec().into_boxed_slice();
        backing.seen = Box::into_raw(seen) as *mut u8 as usize;
    }
    buffers.insert(res, backing);
    GENERATION.fetch_add(1, Ordering::Release);
    true
}

pub(super) fn forget(res: usize) {
    let Some(b) = BUFFERS.lock().unwrap().as_mut().and_then(|b| b.remove(&res)) else { return };
    if b.seen != 0 {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(b.seen as *mut u8, b.bytes)) });
    }
    GENERATION.fetch_add(1, Ordering::Release);
}

/// Buffers outside write-watched memory: their pages whose bytes changed since the last look
/// are taken as written.
pub(super) unsafe fn unwatched_dirty(pages: &mut Vec<usize>) {
    let mut g = BUFFERS.lock().unwrap();
    let Some(buffers) = g.as_mut() else { return };
    for b in buffers.values_mut() {
        b.changed_pages(LOOK_BYTES, pages);
    }
}

impl Backing {
    unsafe fn changed_pages(&mut self, budget: usize, out: &mut Vec<usize>) {
        let end = self.title + self.bytes;
        if self.seen == 0 || !writable(self.title, end) {
            return;
        }
        let mut at = self.title + self.cursor;
        let mut left = budget.min(self.bytes);
        while left > 0 {
            if at >= end {
                at = self.title;
            }
            let next = ((at & !(PAGE - 1)) + PAGE).min(end);
            left = left.saturating_sub(next - at);
            let now = std::slice::from_raw_parts(at as *const u8, next - at);
            let seen = std::slice::from_raw_parts_mut((self.seen + (at - self.title)) as *mut u8, next - at);
            if now != seen {
                seen.copy_from_slice(now);
                out.push(at & !(PAGE - 1));
            }
            at = next;
        }
        self.cursor = if at >= end { 0 } else { at - self.title };
    }

    unsafe fn sync_page(self, page: usize, granularity: usize) -> Option<Range<usize>> {
        let start = page.max(self.title);
        let end = (page + granularity).min(self.title + self.bytes);
        if start >= end || !writable(start, end) {
            return None;
        }
        let dst = self.mirror + (start - self.title);
        let len = end - start;
        std::ptr::copy_nonoverlapping(start as *const u8, dst as *mut u8, len);
        Some(dst..dst + len)
    }

    pub(super) unsafe fn read(mut self, begin: usize, end: usize) -> bool {
        let _refresh = REFRESH.lock().unwrap();
        let start = (self.title + begin) & !(PAGE - 1);
        let stop = self.title + end;
        if !writable(start, stop) {
            return false;
        }
        for (base, size) in crate::hle::kernelx::graphics_reservations() {
            let (lo, hi) = (start.max(base), stop.min(base + size));
            if lo >= hi {
                continue;
            }
            let mut pages = vec![std::ptr::null_mut::<c_void>(); (hi - lo).div_ceil(PAGE)];
            let mut count = pages.len();
            let mut granularity = 0u32;
            if GetWriteWatch(
                0,
                lo as *const c_void,
                hi - lo,
                pages.as_mut_ptr(),
                &mut count,
                &mut granularity,
            ) == 0
            {
                for &p in &pages[..count] {
                    self.sync_page(p as usize, granularity as usize);
                }
            }
        }
        let mut changed = Vec::new();
        self.changed_pages(self.bytes, &mut changed);
        for p in changed {
            self.sync_page(p, PAGE);
        }
        super::sparse::readback_now(self.mirror + begin, end - begin);
        std::ptr::copy_nonoverlapping(
            (self.mirror + begin) as *const u8,
            (self.title + begin) as *mut u8,
            end - begin,
        );
        if self.seen != 0 {
            std::ptr::copy_nonoverlapping((self.mirror + begin) as *const u8, (self.seen + begin) as *mut u8, end - begin);
        }
        true
    }
}

pub(super) unsafe fn sync_dirty(pages: &mut Vec<usize>, granularity: usize) {
    let g = BUFFERS.lock().unwrap();
    let Some(buffers) = g.as_ref() else { return };
    let original = pages.len();
    for b in buffers.values() {
        let start = pages[..original].partition_point(|&p| p + granularity <= b.title);
        let end = pages[..original].partition_point(|&p| p < b.title + b.bytes);
        for i in start..end {
            if let Some(dst) = b.sync_page(pages[i], granularity) {
                pages.extend(((dst.start & !(PAGE - 1))..dst.end).step_by(PAGE));
            }
        }
    }
    if pages.len() != original {
        pages.sort_unstable();
    }
}
