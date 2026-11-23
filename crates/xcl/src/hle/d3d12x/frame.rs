//! Console frame pacing: the device's frame clock (slots 78–81) and the flip tracking
//! `PresentX` feeds.
//!
//! A console title does not wait on a swap chain: it sets a frame interval
//! (`SetFrameIntervalX`, e.g. 16 667 µs), schedules frame events at an offset into each
//! interval (`ScheduleFrameEventX`: type 0 = the frame origin, 1 and 2 = further points of the
//! frame) and blocks on them (`WaitFrameEventX`, which hands out a frame token at the origin).
//! `PresentX` names the token of the frame it flips, `GetFrameStatisticsX` reports when that
//! frame's origin and flip happened. Semantics as observed on Fortnite (UE) – arguments,
//! return codes, the ancillary handle block `{ u32 count; HANDLE* list @8; i32* signaled @0x10 }`.
//!
//! XCL runs the clock on a thread of its own (high-resolution waitable timer, the last
//! ~0.6 ms spun): at each event it releases the type's internal semaphore (origin) or sets its
//! event, and signals the title's ancillary handles. Not locked to the monitor's vblank.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::Foundation::{CloseHandle, DuplicateHandle, DUPLICATE_SAME_ACCESS, HANDLE, WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows_sys::Win32::System::Threading::{
    CreateEventW, CreateSemaphoreW, CreateWaitableTimerExW, GetCurrentProcess, GetCurrentThread, ReleaseSemaphore, SetEvent, SetThreadPriority,
    SetWaitableTimer, WaitForMultipleObjects, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, INFINITE, THREAD_PRIORITY_TIME_CRITICAL, TIMER_ALL_ACCESS,
};

use crate::util::{HRESULT, S_OK};

const S_FALSE: HRESULT = 1;
/// what ScheduleFrameEventX returns when it starts the clock
const STARTED_CLOCK: HRESULT = 0x1000_0000;
const E_FAIL: HRESULT = 0x8000_4005u32 as i32;
const E_INVALIDARG: HRESULT = 0x8007_0057u32 as i32;
/// frame event types: 0 origin, 1 and 2 further points of the frame
const TYPES: usize = 3;
const ORIGIN_SEMAPHORE_MAX: i32 = 4;
/// WaitFrameEventX flags
const WAIT_ALL: u32 = 1;
const TOKEN_ON_TIMEOUT: u32 = 2;
/// ScheduleFrameEventX flags
const UNSCHEDULE: u32 = 1;
const TOKEN_MASK: u64 = 0xFFFF_FFFF_FFFF;
const KNOWN_TOKENS: usize = 64;
const FLIPPED_KEEP: usize = 64;

/// Frame intervals the console accepts (µs, with the rounding a title may use); the 120 Hz
/// ones (8 333, 25 000) only on a display that runs at 120 Hz.
pub fn interval_valid(len_us: u32, high_refresh: bool) -> bool {
    // either side of the nominal length: 1_000_000 / 60 = 16 666 and 16 667 are both 60 Hz
    let near = |base: u32, span: u32| len_us.abs_diff(base) <= span;
    if near(8333, 9) || near(25000, 25) {
        return high_refresh;
    }
    near(16667, 16) || near(33333, 34) || near(50000, 49) || near(66667, 67)
}

/// The first deadline `base + offset + k·interval` after `after`.
pub fn next_deadline(base: i64, interval: i64, offset: i64, after: i64) -> i64 {
    if interval <= 0 {
        return i64::MAX;
    }
    let first = base + offset;
    if after < first {
        return first;
    }
    first + ((after - first) / interval + 1) * interval
}

struct Kind {
    /// semaphore (origin) or auto-reset event the clock signals
    internal: HANDLE,
    offset_us: u32,
    /// the title's ancillary handles, duplicated
    extra: Vec<HANDLE>,
    scheduled: bool,
}

struct Clock {
    interval_us: u32,
    active: bool,
    kinds: [Kind; TYPES],
    /// bumped on every change: the clock thread re-plans
    generation: u64,
    base_qpc: i64,
}

unsafe impl Send for Clock {}

static CLOCK: Mutex<Option<Clock>> = Mutex::new(None);
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);
static ISSUED: Mutex<Vec<u64>> = Mutex::new(Vec::new());
/// (token, origin qpc, flip qpc)
static FLIPPED: Mutex<Vec<(u64, i64, i64)>> = Mutex::new(Vec::new());
static LAST_ORIGIN: AtomicI64 = AtomicI64::new(0);
static THREAD: AtomicBool = AtomicBool::new(false);

pub fn qpc() -> i64 {
    let mut v = 0i64;
    unsafe { QueryPerformanceCounter(&mut v) };
    v
}

fn qpf() -> i64 {
    static F: OnceLock<i64> = OnceLock::new();
    *F.get_or_init(|| {
        let mut v = 0i64;
        unsafe { QueryPerformanceFrequency(&mut v) };
        v.max(1)
    })
}

fn ticks(us: i64) -> i64 {
    us * qpf() / 1_000_000
}

/// the monitor runs at 120 Hz or more
fn high_refresh() -> bool {
    use windows::Win32::Graphics::Gdi::{EnumDisplaySettingsW, DEVMODEW, ENUM_CURRENT_SETTINGS};
    static HIGH: OnceLock<bool> = OnceLock::new();
    *HIGH.get_or_init(|| unsafe {
        let mut dm = DEVMODEW { dmSize: std::mem::size_of::<DEVMODEW>() as u16, ..Default::default() };
        EnumDisplaySettingsW(windows::core::PCWSTR::null(), ENUM_CURRENT_SETTINGS, &mut dm).as_bool() && dm.dmDisplayFrequency >= 119
    })
}

fn wake() -> HANDLE {
    static W: OnceLock<usize> = OnceLock::new();
    *W.get_or_init(|| unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) } as usize) as HANDLE
}

fn with<R>(f: impl FnOnce(&mut Clock) -> R) -> R {
    let mut g = CLOCK.lock().unwrap();
    let c = g.get_or_insert_with(|| {
        let kind = |i: usize| Kind {
            internal: unsafe {
                if i == 0 {
                    CreateSemaphoreW(std::ptr::null(), 0, ORIGIN_SEMAPHORE_MAX, std::ptr::null())
                } else {
                    CreateEventW(std::ptr::null(), 0, 0, std::ptr::null())
                }
            },
            offset_us: 0,
            extra: Vec::new(),
            scheduled: false,
        };
        Clock { interval_us: 0, active: false, kinds: [kind(0), kind(1), kind(2)], generation: 0, base_qpc: 0 }
    });
    f(c)
}

/// device #78 SetFrameIntervalX(output, length µs, periodicity, flags)
pub unsafe extern "system" fn set_frame_interval_x(_this: *mut std::ffi::c_void, _output: *mut std::ffi::c_void, len_us: u32, periodicity: u32, flags: u32) -> HRESULT {
    if len_us != 0 && !interval_valid(len_us, high_refresh()) {
        crate::xlog_warn!("D3D12.X: SetFrameIntervalX({len_us} µs) – not an interval of this display");
        return E_INVALIDARG;
    }
    with(|c| {
        if c.interval_us != len_us {
            c.kinds[0].scheduled = false;
            if c.active && len_us != 0 {
                c.base_qpc = qpc();
            }
        }
        c.interval_us = len_us;
        if len_us == 0 {
            c.active = false;
            for k in c.kinds.iter_mut() {
                k.scheduled = false;
            }
        }
        c.generation += 1;
    });
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| crate::xlog_info!("D3D12.X: frame interval {len_us} µs (periodicity {periodicity}, flags {flags:#x})"));
    SetEvent(wake());
    S_FALSE
}

/// the handles of an ancillary block `{ u32 count; HANDLE* list @8; i32* signaled @0x10 }`
unsafe fn ancillary(block: *const u8) -> Result<Vec<HANDLE>, HRESULT> {
    if block.is_null() {
        return Ok(Vec::new());
    }
    if !crate::gpu::shader::readable(block as usize, 16) {
        return Err(E_INVALIDARG);
    }
    let count = (block as *const u32).read_unaligned() as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    let list = (block.add(8) as *const usize).read_unaligned();
    if count > 63 || !crate::gpu::shader::readable(list, count * 8) {
        return Err(E_INVALIDARG);
    }
    Ok((0..count).map(|i| (list as *const HANDLE).add(i).read_unaligned()).collect())
}

unsafe fn note_signaled(block: *const u8, index: i32) {
    if block.is_null() || !crate::gpu::shader::readable(block as usize, 0x18) {
        return;
    }
    let out = (block.add(0x10) as *const usize).read_unaligned();
    if crate::gpu::shader::readable(out, 4) {
        (out as *mut i32).write_unaligned(index);
    }
}

/// device #79 ScheduleFrameEventX(type, offset µs, ancillary, flags)
pub unsafe extern "system" fn schedule_frame_event_x(_this: *mut std::ffi::c_void, ty: u32, offset_us: u32, block: *const u8, flags: u32) -> HRESULT {
    let unschedule = flags & UNSCHEDULE != 0;
    let invalid = with(|c| c.interval_us == 0 || ty as usize >= TYPES || (!c.active && ty != 0) || (unschedule && (ty == 0 || !c.kinds[0].scheduled)));
    if invalid {
        return E_INVALIDARG;
    }
    let handles = match ancillary(block) {
        Ok(h) => h,
        Err(e) => return e,
    };
    let mut dups = Vec::with_capacity(handles.len());
    if !unschedule {
        for h in handles {
            let mut d: HANDLE = std::ptr::null_mut();
            if DuplicateHandle(GetCurrentProcess(), h, GetCurrentProcess(), &mut d, 0, 0, DUPLICATE_SAME_ACCESS) == 0 {
                dups.into_iter().for_each(|d| {
                    CloseHandle(d);
                });
                return E_FAIL;
            }
            dups.push(d);
        }
    }
    let hr = with(|c| {
        let mut hr = S_FALSE;
        if !c.active {
            c.active = true;
            c.base_qpc = qpc();
            hr = STARTED_CLOCK;
        }
        let interval = c.interval_us.max(1);
        let k = &mut c.kinds[ty as usize];
        for h in k.extra.drain(..) {
            CloseHandle(h);
        }
        if unschedule {
            k.scheduled = false;
        } else {
            k.offset_us = offset_us % interval;
            k.extra = dups;
            k.scheduled = true;
        }
        c.generation += 1;
        hr
    });
    if !THREAD.swap(true, Ordering::SeqCst) {
        let _ = std::thread::Builder::new().name("xcl-frame-clock".into()).spawn(clock_thread);
    }
    SetEvent(wake());
    hr
}

/// `XCL_ACTIVITY`: WaitFrameEventX calls, their timeouts, GetFrameStatisticsX calls, and those
/// that found no flip yet
static ACTIVITY: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

pub fn activity() -> [u64; 4] {
    ACTIVITY.each_ref().map(|a| a.load(Ordering::Relaxed))
}

/// device #80 WaitFrameEventX(type, timeout ms, ancillary, flags, token in/out)
pub unsafe extern "system" fn wait_frame_event_x(_this: *mut std::ffi::c_void, ty: u32, timeout_ms: u32, block: *const u8, flags: u32, token: *mut u64) -> HRESULT {
    let internal = with(|c| ((ty as usize) < TYPES && c.kinds[ty as usize].scheduled).then(|| c.kinds[ty as usize].internal as usize));
    let Some(internal) = internal else { return E_INVALIDARG };
    let token_in = if token.is_null() { 0 } else { *token };
    if ty == 0 && token_in != 0 && (!ISSUED.lock().unwrap().contains(&token_in) || flags & TOKEN_ON_TIMEOUT != 0) {
        return E_INVALIDARG;
    }
    let extra = match ancillary(block) {
        Ok(h) => h,
        Err(e) => return e,
    };
    let mut hs = vec![internal as HANDLE];
    hs.extend(extra);
    let r = WaitForMultipleObjects(hs.len() as u32, hs.as_ptr(), (flags & WAIT_ALL != 0) as i32, timeout_ms);
    ACTIVITY[0].fetch_add(1, Ordering::Relaxed);
    if r == WAIT_TIMEOUT {
        ACTIVITY[1].fetch_add(1, Ordering::Relaxed);
    }
    let (hr, index) = if r == WAIT_TIMEOUT {
        (S_FALSE, -1)
    } else if ((r - WAIT_OBJECT_0) as usize) < hs.len() {
        // -2: the frame event itself, else the index of the ancillary handle
        (S_OK, if r == WAIT_OBJECT_0 { -2 } else { (r - WAIT_OBJECT_0) as i32 - 1 })
    } else {
        (E_FAIL, -1)
    };
    note_signaled(block, index);
    if hr == E_FAIL {
        return hr;
    }
    let issue = index == -2 || (index == -1 && flags & TOKEN_ON_TIMEOUT != 0);
    if issue && ty == 0 && token_in == 0 {
        let mut t = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed) & TOKEN_MASK;
        if t == 0 {
            t = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed) & TOKEN_MASK;
        }
        {
            let mut issued = ISSUED.lock().unwrap();
            if issued.len() >= KNOWN_TOKENS {
                issued.remove(0);
            }
            issued.push(t);
        }
        if !token.is_null() {
            *token = t;
        }
    }
    hr
}

/// `PresentX` flipped the frame of `token`.
pub fn mark_flipped(token: u64) {
    if token == 0 {
        return;
    }
    let mut f = FLIPPED.lock().unwrap();
    f.push((token, LAST_ORIGIN.load(Ordering::Relaxed), qpc()));
    if f.len() > FLIPPED_KEEP {
        let n = f.len() - FLIPPED_KEEP;
        f.drain(..n);
    }
}

/// device #81 GetFrameStatisticsX(token, type, count in/out, statistics): one record
/// `{ u32 type; …; i64 origin qpc @0x10; i64 flip qpc @0x18 }` once the frame was flipped.
pub unsafe extern "system" fn frame_statistics_x(_this: *mut std::ffi::c_void, token: u64, ty: u32, count: *mut u32, stats: *mut u8) -> HRESULT {
    if count.is_null() || stats.is_null() {
        return E_INVALIDARG;
    }
    let flipped = FLIPPED.lock().unwrap().iter().rev().find(|f| f.0 == token).map(|f| (f.1, f.2));
    ACTIVITY[2].fetch_add(1, Ordering::Relaxed);
    let Some((origin, flip)) = flipped else {
        ACTIVITY[3].fetch_add(1, Ordering::Relaxed);
        *count = 0;
        return E_FAIL;
    };
    if *count < 1 {
        return E_INVALIDARG;
    }
    *count = 1;
    (stats as *mut u32).write_unaligned(ty);
    (stats.add(0x10) as *mut i64).write_unaligned(if origin != 0 && origin <= flip { origin } else { flip });
    (stats.add(0x18) as *mut i64).write_unaligned(flip);
    S_OK
}

/// sleep until `deadline` (qpc); false: woken by a change of the schedule
unsafe fn sleep_until(timer: HANDLE, deadline: i64) -> bool {
    const SPIN_US: i64 = 600;
    loop {
        let left = deadline - qpc();
        if left <= 0 {
            return true;
        }
        let left_us = left * 1_000_000 / qpf();
        if left_us > SPIN_US {
            let due = -((left_us - SPIN_US) * 10); // relative, 100 ns units
            SetWaitableTimer(timer, &due, 0, None, std::ptr::null(), 0);
            let hs = [timer, wake()];
            if WaitForMultipleObjects(2, hs.as_ptr(), 0, INFINITE) == WAIT_OBJECT_0 + 1 {
                return false;
            }
        } else {
            std::hint::spin_loop();
        }
    }
}

fn clock_thread() {
    unsafe {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        let mut timer = CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if timer.is_null() {
            timer = CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), 0, TIMER_ALL_ACCESS);
        }
        let mut last = [i64::MIN / 2; TYPES];
        loop {
            // the next event of the current schedule
            let plan = with(|c| {
                if !c.active || c.interval_us == 0 {
                    return None;
                }
                let interval = ticks(c.interval_us as i64);
                let now = qpc();
                c.kinds
                    .iter()
                    .enumerate()
                    .filter(|(_, k)| k.scheduled)
                    .map(|(i, k)| {
                        let mut d = next_deadline(c.base_qpc, interval, ticks(k.offset_us as i64), now - 1);
                        if d - last[i] < interval / 2 {
                            d += interval; // this point of the frame already fired
                        }
                        (d, i, k.internal as usize, k.extra.iter().map(|&h| h as usize).collect::<Vec<_>>())
                    })
                    .min_by_key(|e| e.0)
                    .map(|e| (c.generation, e))
            });
            let Some((generation, (deadline, ty, internal, extra))) = plan else {
                WaitForMultipleObjects(1, &wake(), 0, INFINITE);
                continue;
            };
            if !sleep_until(timer, deadline) || !with(|c| c.generation == generation) {
                continue;
            }
            last[ty] = deadline;
            if ty == 0 {
                LAST_ORIGIN.store(qpc(), Ordering::Relaxed);
                ReleaseSemaphore(internal as HANDLE, 1, std::ptr::null_mut());
            } else {
                SetEvent(internal as HANDLE);
            }
            for h in extra {
                if SetEvent(h as HANDLE) == 0 {
                    ReleaseSemaphore(h as HANDLE, 1, std::ptr::null_mut());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn console_intervals_and_deadlines() {
        assert!(interval_valid(16667, false) && interval_valid(16666 + 16, false) && interval_valid(33333, false));
        assert!(!interval_valid(16000, false) && !interval_valid(8333, false) && interval_valid(8333, true));
        assert!(interval_valid(16666, false) && interval_valid(66666, false) && interval_valid(33333 - 34, false));
        assert_eq!(next_deadline(100, 10, 3, 50), 103);
        assert_eq!(next_deadline(100, 10, 3, 103), 113);
        assert_eq!(next_deadline(100, 10, 3, 120), 123);
    }

    /// The origin event of a 60 Hz clock: waits return about one interval apart with fresh
    /// tokens; a flipped token has statistics, an unknown one does not.
    #[test]
    fn origin_events_issue_tokens_and_flips_have_statistics() {
        unsafe {
            let dev = std::ptr::null_mut();
            assert_eq!(set_frame_interval_x(dev, std::ptr::null_mut(), 16667, 1, 0), S_FALSE);
            assert_eq!(wait_frame_event_x(dev, 0, 10, std::ptr::null(), 0, std::ptr::null_mut()), E_INVALIDARG, "not scheduled");
            let hr = schedule_frame_event_x(dev, 0, 0, std::ptr::null(), 0);
            assert!(hr == STARTED_CLOCK || hr == S_FALSE, "{hr:#x}");
            let mut tokens = Vec::new();
            let t0 = std::time::Instant::now();
            for _ in 0..4 {
                let mut t = 0u64;
                assert_eq!(wait_frame_event_x(dev, 0, 200, std::ptr::null(), 0, &mut t), S_OK);
                tokens.push(t);
            }
            let ms = t0.elapsed().as_millis();
            assert!((25..=200).contains(&ms), "4 origins at 60 Hz took {ms} ms");
            assert!(tokens.windows(2).all(|w| w[1] > w[0]), "{tokens:?}");
            mark_flipped(tokens[2]);
            let (mut n, mut st) = (1u32, [0u8; 0x40]);
            assert_eq!(frame_statistics_x(dev, tokens[2], 0, &mut n, st.as_mut_ptr()), S_OK);
            let (origin, flip) = (i64::from_le_bytes(st[0x10..0x18].try_into().unwrap()), i64::from_le_bytes(st[0x18..0x20].try_into().unwrap()));
            assert!(n == 1 && origin > 0 && origin <= flip);
            assert_eq!(frame_statistics_x(dev, 0xDEAD, 0, &mut n, st.as_mut_ptr()), E_FAIL);
            assert_eq!(set_frame_interval_x(dev, std::ptr::null_mut(), 16000, 1, 0), E_INVALIDARG);
            assert_eq!(set_frame_interval_x(dev, std::ptr::null_mut(), 0, 0, 0), S_FALSE, "stops the clock");
        }
    }
}
