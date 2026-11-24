//! XCL's own record of the title's memory: what XCL reserved and committed for it – the
//! title's allocations through kernelx, physical pages and their views, unified memory and its
//! sparse windows – so the hot paths that ask whether title memory is there (`gpu::shader::
//! readable` per draw, texture uploads, DirectStorage destinations) need no system call.
//! VirtualQuery is a system call that walks the address space's tree and sizes a region by
//! walking its pages: with hundreds of thousands of views and placeholders mapped, and views of
//! gigabytes, one call took milliseconds. Addresses outside everything recorded (host DLLs, heaps,
//! stacks) are still asked of Windows (`access` returns None).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_NOACCESS, PAGE_READWRITE, PAGE_WRITECOPY};

/// What the memory allows, least first.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Access {
    /// not committed, or no access
    None,
    Read,
    Write,
}

impl Access {
    /// From a Windows page protection.
    pub fn of(protect: u32) -> Access {
        if protect == 0 || protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 {
            Access::None
        } else if protect & (PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY) != 0 {
            Access::Write
        } else {
            Access::Read
        }
    }
}

#[derive(Default)]
struct Map {
    /// address space XCL reserved or handed out: start → end (allocations, not merged)
    known: BTreeMap<usize, usize>,
    /// committed memory: start → (end, access), merged where the access is the same
    committed: BTreeMap<usize, (usize, Access)>,
}

static MAP: RwLock<Map> = RwLock::new(Map { known: BTreeMap::new(), committed: BTreeMap::new() });

/// calls answered from the record, and asked of Windows
static ANSWERED: AtomicU64 = AtomicU64::new(0);
static ASKED: AtomicU64 = AtomicU64::new(0);

impl Map {
    fn cut(&mut self, a: usize, b: usize) {
        let hit: Vec<(usize, (usize, Access))> = self.committed.range(..b).rev().take_while(|(_, &(e, _))| e > a).map(|(&s, &v)| (s, v)).collect();
        for (s, (e, acc)) in hit {
            self.committed.remove(&s);
            if s < a {
                self.committed.insert(s, (a, acc));
            }
            if e > b {
                self.committed.insert(b, (e, acc));
            }
        }
    }

    fn put(&mut self, mut a: usize, mut b: usize, acc: Access) {
        self.cut(a, b);
        if let Some((&s, &(e, x))) = self.committed.range(..a).next_back() {
            if e == a && x == acc {
                self.committed.remove(&s);
                a = s;
            }
        }
        if let Some(&(e, x)) = self.committed.get(&b) {
            if x == acc {
                self.committed.remove(&b);
                b = e;
            }
        }
        self.committed.insert(a, (b, acc));
    }

    fn known(&self, at: usize) -> bool {
        self.known.range(..=at).next_back().is_some_and(|(_, &e)| at < e) || crate::gpu::uma::get().is_some_and(|u| u.is_sparse(at))
    }
}

fn write() -> std::sync::RwLockWriteGuard<'static, Map> {
    MAP.write().unwrap_or_else(|e| e.into_inner())
}

fn read() -> std::sync::RwLockReadGuard<'static, Map> {
    MAP.read().unwrap_or_else(|e| e.into_inner())
}

/// `[start, start + len)` is address space XCL reserved (or handed out) for the title.
pub fn reserved(start: usize, len: usize) {
    if len > 0 {
        write().known.insert(start, start + len);
    }
}

/// The reservation at `start` is gone, with everything committed in it.
pub fn released(start: usize) {
    let mut g = write();
    if let Some(end) = g.known.remove(&start) {
        g.cut(start, end);
    }
}

/// The reservation holding `at` (start, end), if XCL made it.
pub fn reservation(at: usize) -> Option<(usize, usize)> {
    read().known.range(..=at).next_back().filter(|(_, &e)| at < e).map(|(&s, &e)| (s, e))
}

/// `[start, start + len)` is committed with `protect`.
pub fn committed(start: usize, len: usize, protect: u32) {
    if len > 0 {
        write().put(start & !0xFFF, (start + len + 0xFFF) & !0xFFF, Access::of(protect));
    }
}

/// `[start, start + len)` is decommitted.
pub fn decommitted(start: usize, len: usize) {
    if len > 0 {
        write().cut(start & !0xFFF, (start + len + 0xFFF) & !0xFFF);
    }
}

/// The committed parts of `[start, start + len)` now have `protect`.
pub fn protected(start: usize, len: usize, protect: u32) {
    let (a, b) = (start & !0xFFF, (start + len + 0xFFF) & !0xFFF);
    let mut g = write();
    let parts: Vec<(usize, usize)> = g.committed.range(..b).rev().take_while(|(_, &(e, _))| e > a).map(|(&s, &(e, _))| (s.max(a), e.min(b))).collect();
    for (s, e) in parts {
        g.put(s, e, Access::of(protect));
    }
}

/// The least access over `[p, p + len)`, or None when part of it is outside what XCL recorded
/// (ask Windows).
pub fn access(p: usize, len: usize) -> Option<Access> {
    let end = p.checked_add(len.max(1))?;
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(p, len.max(1))) {
        return Some(Access::Write);
    }
    let g = read();
    let mut least = Access::Write;
    let mut cur = p;
    while cur < end {
        match g.committed.range(..=cur).next_back() {
            Some((_, &(e, acc))) if e > cur => {
                least = least.min(acc);
                cur = e;
            }
            _ if g.known(cur) => {
                ANSWERED.fetch_add(1, Ordering::Relaxed);
                return Some(Access::None);
            }
            _ => {
                ASKED.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        }
    }
    ANSWERED.fetch_add(1, Ordering::Relaxed);
    Some(least)
}

/// The committed, readable runs (start, end) of `[p, p + len)`, or None when part of it is
/// outside what XCL recorded.
pub fn readable_runs(p: usize, len: usize) -> Option<Vec<(usize, usize)>> {
    let end = p.checked_add(len)?;
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(p, len)) {
        return Some(vec![(p, end)]);
    }
    let g = read();
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut cur = p;
    while cur < end {
        match g.committed.range(..=cur).next_back() {
            Some((_, &(e, acc))) if e > cur => {
                let hi = e.min(end);
                if acc >= Access::Read {
                    match out.last_mut() {
                        Some(last) if last.1 == cur => last.1 = hi,
                        _ => out.push((cur, hi)),
                    }
                }
                cur = hi;
            }
            _ if g.known(cur) => {
                // to the next committed run, or the end
                cur = g.committed.range(cur..end).next().map_or(end, |(&s, _)| s);
            }
            _ => return None,
        }
    }
    Some(out)
}

/// (answered from the record, asked of Windows) since the start
pub fn counts() -> (u64, u64) {
    (ANSWERED.load(Ordering::Relaxed), ASKED.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commits_merge_split_and_answer() {
        let base = 0x7000_0000_0000usize;
        reserved(base, 0x10_0000);
        assert_eq!(access(base, 0x1000), Some(Access::None), "reserved, not committed");
        committed(base + 0x1000, 0x3000, PAGE_READWRITE);
        committed(base + 0x4000, 0x1000, PAGE_READWRITE);
        assert_eq!(access(base + 0x1000, 0x4000), Some(Access::Write));
        protected(base + 0x2000, 0x1000, 2); // PAGE_READONLY
        assert_eq!(access(base + 0x1000, 0x4000), Some(Access::Read));
        decommitted(base + 0x3000, 0x1000);
        assert_eq!(access(base + 0x1000, 0x4000), Some(Access::None));
        assert_eq!(readable_runs(base, 0x6000), Some(vec![(base + 0x1000, base + 0x3000), (base + 0x4000, base + 0x5000)]));
        assert_eq!(access(base + 0x20_0000, 4), None, "outside the record");
        released(base);
        assert_eq!(access(base + 0x1000, 4), None);
    }
}
