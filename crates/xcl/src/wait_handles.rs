//! `XCL_TRACE_HANDLES=1`: which handle each thread waits on, who made it and who last set it,
//! shown in the thread dump.
//!
//! A load that stops with every title thread parked in `WaitForSingleObjectEx` says nothing
//! by itself: the stack shows the wait, not the object. Here the kernelbase wait, event and
//! semaphore functions record the handle a thread waits on, the code that created each event
//! or semaphore, and the code that last set or released it (or that nobody did).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::hooks::hook;

/// Whether `XCL_TRACE_HANDLES` is on.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_TRACE_HANDLES").is_ok_and(|v| v != "0"))
}

fn now_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Return addresses above a hook: its caller and the two before.
type Callers = [usize; 3];

fn callers() -> Callers {
    let mut frames = [std::ptr::null_mut::<c_void>(); 3];
    unsafe { windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace(2, 3, frames.as_mut_ptr(), std::ptr::null_mut()) };
    frames.map(|f| f as usize)
}

/// The first caller outside the Windows libraries.
fn describe(c: &Callers) -> String {
    let all: Vec<String> = c.iter().filter(|&&a| a != 0).map(|&a| crate::modules::describe_address(a)).collect();
    let system = |s: &&String| {
        let l = s.to_ascii_lowercase();
        l.starts_with("kernelbase") || l.starts_with("kernel32") || l.starts_with("ntdll")
    };
    all.iter().find(|s| !system(s)).or(all.first()).cloned().unwrap_or_else(|| "?".into())
}

struct Object {
    kind: &'static str,
    name: String,
    made_by: Callers,
    sets: u64,
    last_set_by: Callers,
    last_set_ms: u64,
}

static OBJECTS: Mutex<Option<HashMap<usize, Object>>> = Mutex::new(None);

unsafe fn made(h: *mut c_void, kind: &'static str, name: *const u16) {
    if h.is_null() {
        return;
    }
    let o = Object { kind, name: crate::util::from_wide_ptr(name).unwrap_or_default(), made_by: callers(), sets: 0, last_set_by: [0; 3], last_set_ms: 0 };
    OBJECTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(h as usize, o);
}

fn set(h: *mut c_void) {
    let by = callers();
    if let Some(o) = OBJECTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(h as usize))) {
        o.sets += 1;
        o.last_set_by = by;
        o.last_set_ms = now_ms();
    }
}

/// One thread's current wait, in a table indexed by its id.
struct Wait {
    tid: AtomicU32,
    handle: AtomicUsize,
    count: AtomicU32,
    since_ms: AtomicU64,
}

const WAIT_SLOTS: usize = 4096;
static WAITS: [Wait; WAIT_SLOTS] = [const { Wait { tid: AtomicU32::new(0), handle: AtomicUsize::new(0), count: AtomicU32::new(0), since_ms: AtomicU64::new(0) } }; WAIT_SLOTS];

fn slot(tid: u32) -> &'static Wait {
    &WAITS[(tid as usize / 4) % WAIT_SLOTS]
}

fn begin_wait(handle: usize, count: u32) -> u32 {
    let tid = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
    let w = slot(tid);
    w.tid.store(tid, Ordering::Relaxed);
    w.count.store(count, Ordering::Relaxed);
    w.since_ms.store(now_ms(), Ordering::Relaxed);
    w.handle.store(handle, Ordering::Release);
    tid
}

fn end_wait(tid: u32) {
    let w = slot(tid);
    if w.tid.load(Ordering::Relaxed) == tid {
        w.handle.store(0, Ordering::Release);
    }
}

/// What thread `tid` waits on, for the thread dump.
pub fn describe_wait(tid: u32) -> Option<String> {
    if !enabled() {
        return None;
    }
    let w = slot(tid);
    let h = w.handle.load(Ordering::Acquire);
    if h == 0 || w.tid.load(Ordering::Relaxed) != tid {
        return None;
    }
    let now = now_ms();
    let waited = now.saturating_sub(w.since_ms.load(Ordering::Relaxed)) as f64 / 1000.0;
    let count = w.count.load(Ordering::Relaxed);
    let of = if count > 1 { format!(" (first of {count})") } else { String::new() };
    let state = unsafe { object_state(h) };
    let what = match OBJECTS.lock().unwrap().as_ref().and_then(|m| m.get(&h)) {
        None => format!("{state}, not made through the hooked calls"),
        Some(o) => {
            let name = if o.name.is_empty() { String::new() } else { format!(" '{}'", o.name) };
            let set = if o.sets == 0 {
                "never set".to_string()
            } else {
                format!("set {} times, last by {} {:.1} s ago", o.sets, describe(&o.last_set_by), now.saturating_sub(o.last_set_ms) as f64 / 1000.0)
            };
            format!("{}{name} ({state}) made by {}, {set}", o.kind, describe(&o.made_by))
        }
    };
    Some(format!("waits {waited:.1} s on {h:#x}{of}: {what}"))
}

/// The kernel object behind `h`: its type, and for an event or semaphore its state now.
unsafe fn object_state(h: usize) -> String {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    type Query = unsafe extern "system" fn(usize, u32, *mut c_void, u32, *mut u32) -> i32;
    let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr() as *const u8);
    let proc = |name: &std::ffi::CStr| GetProcAddress(ntdll, name.as_ptr() as *const u8).map(|f| std::mem::transmute::<_, Query>(f));
    let Some(query_object) = proc(c"NtQueryObject") else { return "?".into() };
    // PUBLIC_OBJECT_TYPE_INFORMATION: the type name (UNICODE_STRING) first
    #[repr(C, align(8))]
    struct Buf([u8; 512]);
    let mut buf = Buf([0; 512]);
    if query_object(h, 2, buf.0.as_mut_ptr() as *mut c_void, 512, std::ptr::null_mut()) < 0 {
        return "not a handle now".into();
    }
    let len = u16::from_ne_bytes([buf.0[0], buf.0[1]]) as usize / 2;
    let text = *(buf.0.as_ptr().add(8) as *const *const u16);
    let kind = if text.is_null() { "?".to_string() } else { String::from_utf16_lossy(std::slice::from_raw_parts(text, len)) };
    let mut basic = [0i32; 2];
    let detail = match kind.as_str() {
        // EVENT_BASIC_INFORMATION { EventType, EventState }
        "Event" => proc(c"NtQueryEvent").filter(|q| q(h, 0, basic.as_mut_ptr() as *mut c_void, 8, std::ptr::null_mut()) >= 0).map(|_| {
            format!(", {} {}", if basic[0] == 0 { "manual" } else { "auto" }, if basic[1] != 0 { "signalled" } else { "not signalled" })
        }),
        // SEMAPHORE_BASIC_INFORMATION { CurrentCount, MaximumCount }
        "Semaphore" => proc(c"NtQuerySemaphore").filter(|q| q(h, 0, basic.as_mut_ptr() as *mut c_void, 8, std::ptr::null_mut()) >= 0).map(|_| format!(", count {} of {}", basic[0], basic[1])),
        _ => None,
    };
    format!("{kind}{}", detail.unwrap_or_default())
}

macro_rules! orig {
    ($slot:ident as $ty:ty) => {
        std::mem::transmute::<usize, $ty>($slot.load(Ordering::Acquire))
    };
}

type Handle = *mut c_void;
type FnWaitOne = unsafe extern "system" fn(Handle, u32, i32) -> u32;
type FnWaitMany = unsafe extern "system" fn(u32, *const Handle, i32, u32, i32) -> u32;
type FnCreateEventW = unsafe extern "system" fn(*const c_void, i32, i32, *const u16) -> Handle;
type FnCreateEventExW = unsafe extern "system" fn(*const c_void, *const u16, u32, u32) -> Handle;
type FnCreateSemaphoreW = unsafe extern "system" fn(*const c_void, i32, i32, *const u16) -> Handle;
type FnCreateSemaphoreExW = unsafe extern "system" fn(*const c_void, i32, i32, *const u16, u32, u32) -> Handle;
type FnSetEvent = unsafe extern "system" fn(Handle) -> i32;
type FnReleaseSemaphore = unsafe extern "system" fn(Handle, i32, *mut i32) -> i32;
type FnCloseHandle = unsafe extern "system" fn(Handle) -> i32;

static O_WAIT_ONE: AtomicUsize = AtomicUsize::new(0);
static O_WAIT_MANY: AtomicUsize = AtomicUsize::new(0);
static O_CREATE_EVENT_W: AtomicUsize = AtomicUsize::new(0);
static O_CREATE_EVENT_EX_W: AtomicUsize = AtomicUsize::new(0);
static O_CREATE_SEMAPHORE_W: AtomicUsize = AtomicUsize::new(0);
static O_CREATE_SEMAPHORE_EX_W: AtomicUsize = AtomicUsize::new(0);
static O_SET_EVENT: AtomicUsize = AtomicUsize::new(0);
static O_RELEASE_SEMAPHORE: AtomicUsize = AtomicUsize::new(0);
static O_CLOSE_HANDLE: AtomicUsize = AtomicUsize::new(0);

unsafe extern "system" fn hk_wait_one(h: Handle, ms: u32, alertable: i32) -> u32 {
    let tid = begin_wait(h as usize, 1);
    let r = orig!(O_WAIT_ONE as FnWaitOne)(h, ms, alertable);
    end_wait(tid);
    r
}

unsafe extern "system" fn hk_wait_many(n: u32, hs: *const Handle, all: i32, ms: u32, alertable: i32) -> u32 {
    let first = if n == 0 || hs.is_null() { 0 } else { *hs as usize };
    let tid = begin_wait(first, n);
    let r = orig!(O_WAIT_MANY as FnWaitMany)(n, hs, all, ms, alertable);
    end_wait(tid);
    r
}

unsafe extern "system" fn hk_create_event_w(sa: *const c_void, manual: i32, initial: i32, name: *const u16) -> Handle {
    let h = orig!(O_CREATE_EVENT_W as FnCreateEventW)(sa, manual, initial, name);
    made(h, "event", name);
    h
}

unsafe extern "system" fn hk_create_event_ex_w(sa: *const c_void, name: *const u16, flags: u32, access: u32) -> Handle {
    let h = orig!(O_CREATE_EVENT_EX_W as FnCreateEventExW)(sa, name, flags, access);
    made(h, "event", name);
    h
}

unsafe extern "system" fn hk_create_semaphore_w(sa: *const c_void, initial: i32, max: i32, name: *const u16) -> Handle {
    let h = orig!(O_CREATE_SEMAPHORE_W as FnCreateSemaphoreW)(sa, initial, max, name);
    made(h, "semaphore", name);
    h
}

unsafe extern "system" fn hk_create_semaphore_ex_w(sa: *const c_void, initial: i32, max: i32, name: *const u16, flags: u32, access: u32) -> Handle {
    let h = orig!(O_CREATE_SEMAPHORE_EX_W as FnCreateSemaphoreExW)(sa, initial, max, name, flags, access);
    made(h, "semaphore", name);
    h
}

unsafe extern "system" fn hk_set_event(h: Handle) -> i32 {
    set(h);
    orig!(O_SET_EVENT as FnSetEvent)(h)
}

unsafe extern "system" fn hk_release_semaphore(h: Handle, n: i32, prev: *mut i32) -> i32 {
    set(h);
    orig!(O_RELEASE_SEMAPHORE as FnReleaseSemaphore)(h, n, prev)
}

unsafe extern "system" fn hk_close_handle(h: Handle) -> i32 {
    if let Some(m) = OBJECTS.lock().unwrap().as_mut() {
        m.remove(&(h as usize));
    }
    orig!(O_CLOSE_HANDLE as FnCloseHandle)(h)
}

/// The hooks, with `XCL_TRACE_HANDLES` on.
pub fn install() -> Result<(), String> {
    if !enabled() {
        return Ok(());
    }
    now_ms();
    let list: &[(&str, usize, &AtomicUsize)] = &[
        ("CreateEventW", hk_create_event_w as usize, &O_CREATE_EVENT_W),
        ("CreateEventExW", hk_create_event_ex_w as usize, &O_CREATE_EVENT_EX_W),
        ("CreateSemaphoreW", hk_create_semaphore_w as usize, &O_CREATE_SEMAPHORE_W),
        ("CreateSemaphoreExW", hk_create_semaphore_ex_w as usize, &O_CREATE_SEMAPHORE_EX_W),
        ("SetEvent", hk_set_event as usize, &O_SET_EVENT),
        ("ReleaseSemaphore", hk_release_semaphore as usize, &O_RELEASE_SEMAPHORE),
        ("CloseHandle", hk_close_handle as usize, &O_CLOSE_HANDLE),
        ("WaitForSingleObjectEx", hk_wait_one as usize, &O_WAIT_ONE),
        ("WaitForMultipleObjectsEx", hk_wait_many as usize, &O_WAIT_MANY),
    ];
    let errors: Vec<String> = list.iter().filter_map(|&(f, detour, orig)| unsafe { hook("kernelbase.dll", f, detour, orig) }.err()).collect();
    crate::xlog_info!("XCL_TRACE_HANDLES: {} of {} wait / event / semaphore hooks active", list.len() - errors.len(), list.len());
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}
