//! `XCL_TRACE_WAITS=1`: every wait the title's GPU work or threads can stand still on, with what
//! should have ended it. Fence waits (a queue's Wait, a fence's SetEventOnCompletion) are followed
//! until the fence gets there: each that takes 5 ms or more is logged (throttled), each still
//! pending after 2 s always, with the fence's completed value and its last signal (who, what
//! value, how long ago) – a fence whose signal never comes. Memory waits (Wait32/64BitValueX)
//! are timed where the CPU resolves them, with the last value XCL's CPU writes put there.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::ID3D12Fence;

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_TRACE_WAITS").is_ok_and(|v| v != "0"))
}

/// fence → (value, who, when) of its last signal
static SIGNALS: Mutex<Option<HashMap<usize, (u64, &'static str, Instant)>>> = Mutex::new(None);
/// address → (value, when) of XCL's last CPU write there
static WRITES: Mutex<Option<HashMap<usize, (u64, Instant)>>> = Mutex::new(None);

struct Pending {
    fence: usize,
    value: u64,
    what: &'static str,
    since: Instant,
    completed: u64,
    reported: bool,
}
static PENDING: Mutex<Vec<Pending>> = Mutex::new(Vec::new());
static LOGGED: AtomicU64 = AtomicU64::new(0);

/// first 300, then every 100th
fn throttled() -> bool {
    let n = LOGGED.fetch_add(1, Ordering::Relaxed);
    n < 300 || n % 100 == 0
}

unsafe fn completed(fence: usize) -> Option<u64> {
    let p = fence as *mut c_void;
    ID3D12Fence::from_raw_borrowed(&p).map(|f| f.GetCompletedValue())
}

fn last_signal(fence: usize) -> String {
    match SIGNALS.lock().unwrap().as_ref().and_then(|m| m.get(&fence).copied()) {
        Some((v, who, t)) => format!("last signal {v:#x} by {who} {:?} ago", t.elapsed()),
        None => "never signalled".into(),
    }
}

pub(super) fn signal(fence: usize, value: u64, who: &'static str) {
    if !enabled() {
        return;
    }
    SIGNALS.lock().unwrap().get_or_insert_with(HashMap::new).insert(fence, (value, who, Instant::now()));
}

/// The title waits for `fence` to reach `value` (`what`: a queue's Wait, a CPU wait).
pub(super) unsafe fn wait(fence: usize, value: u64, what: &'static str) {
    if !enabled() {
        return;
    }
    let Some(now) = completed(fence) else { return };
    if now >= value {
        return;
    }
    static MONITOR: OnceLock<()> = OnceLock::new();
    MONITOR.get_or_init(|| {
        let _ = std::thread::Builder::new().name("xcl-wait-trace".into()).spawn(|| loop {
            std::thread::sleep(Duration::from_millis(20));
            unsafe { follow() };
        });
    });
    PENDING.lock().unwrap().push(Pending { fence, value, what, since: Instant::now(), completed: now, reported: false });
}

unsafe fn follow() {
    let mut g = PENDING.lock().unwrap();
    g.retain_mut(|w| {
        let now = completed(w.fence).unwrap_or(u64::MAX);
        let took = w.since.elapsed();
        if now >= w.value {
            if took >= Duration::from_millis(5) && (w.reported || throttled()) {
                crate::xlog_info!("wait trace: {} on fence {:#x} for {:#x} took {took:?} (completed {:#x} when it began; {})", w.what, w.fence, w.value, w.completed, last_signal(w.fence));
            }
            return false;
        }
        if !w.reported && took >= Duration::from_secs(2) {
            w.reported = true;
            crate::xlog_warn!("wait trace: {} on fence {:#x} for {:#x} still pending after {took:?}: completed {now:#x}; {}", w.what, w.fence, w.value, last_signal(w.fence));
        }
        true
    });
}

/// A blocking wait the caller timed itself (SetEventOnCompletion without an event).
pub(super) fn waited(fence: usize, value: u64, what: &'static str, took: Duration) {
    if enabled() && took >= Duration::from_millis(5) && (took >= Duration::from_secs(2) || throttled()) {
        crate::xlog_info!("wait trace: {what} on fence {fence:#x} for {value:#x} took {took:?} ({})", last_signal(fence));
    }
}

pub(super) fn written(va: usize, words: &[u32]) {
    if !enabled() || words.is_empty() {
        return;
    }
    let value = words[0] as u64 | words.get(1).map_or(0, |&w| (w as u64) << 32);
    let mut g = WRITES.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if m.len() > 65536 {
        m.clear();
    }
    m.insert(va, (value, Instant::now()));
}

/// A Wait32/64BitValueX the CPU resolved before a submit.
pub(super) fn memory(what: &str, va: usize, comparison: u32, value: u64, took: Duration, held: bool, now: Option<u64>) {
    if !enabled() || (held && took < Duration::from_millis(1)) || (held && !throttled()) {
        return;
    }
    let last = match WRITES.lock().unwrap().as_ref().and_then(|m| m.get(&va).copied()) {
        Some((v, t)) => format!("last CPU write {v:#x} {:?} ago", t.elapsed()),
        None => "no CPU write by XCL there".into(),
    };
    crate::xlog_info!("wait trace: {what} at {va:#x} (op {comparison}, want {value:#x}) {} after {took:?}; memory holds {now:x?}; {last}", if held { "held" } else { "TIMED OUT" });
}
