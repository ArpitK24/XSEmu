//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.
//!
//! A lazy pipeline (`lazy`) is not built in the call: it is built on the background workers when
//! the title makes it (`preload`: all the cores but two, below normal priority, in the order of
//! `pso_sched::Backlog`), and a bind waits for it, its build going ahead of all the others. A big
//! driver compile that does not fit the memory is put off and its worker goes on (`put_off`). One nobody binds is released again after a while
//! (`idle_limit`) and built again at its first bind. With `XCL_PSO_PRELOAD=0` it is built
//! when first bound, its shaders prepared before that on background threads
//! (`prepare_threads`): translated and compiled into the disk caches with the driver left out
//! (`preparing`). Without either the title's binding threads did every translation and DXC
//! compile themselves, one after another: a retail game showed one compile running on a
//! 24-thread PC while thousands of its pipelines waited. `XCL_PREPARE_PSO=0` turns that off.

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock, Weak};

/// The priority of a preload worker: below the title's threads, which keep their cores while
/// every other core builds pipelines (at `THREAD_PRIORITY_LOWEST` the workers were also behind
/// every other process's threads).
const PRELOAD_PRIORITY: i32 = windows_sys::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL;
/// … and while the title loads (`pso_sched::Tempo::low`): behind title threads that run below
/// normal themselves, as loaders do.
const LOADING_PRIORITY: i32 = windows_sys::Win32::System::Threading::THREAD_PRIORITY_LOWEST;
/// the priority a preload worker has now while no bind waits for what it builds (`pace`)
static REST_PRIORITY: AtomicI32 = AtomicI32::new(PRELOAD_PRIORITY);

/// Refcounted priority inheritance for a preload worker (`REST_PRIORITY`): while a guard
/// lives (a bind waits for the pipeline the worker builds), its OS thread runs at normal
/// priority, whatever the pace of the others; the last guard dropped restores its own.
struct BoostToken {
    tid: u32,
    count: Mutex<usize>,
}

struct BoostGuard(Arc<BoostToken>);

/// Whether preload workers run at `PRELOAD_PRIORITY` and are raised again on demand. Under
/// Wine lowering a thread's priority raises its Linux nice value (always allowed), while raising
/// it back needs `RLIMIT_NICE` / `CAP_SYS_NICE` and fails for a normal user: there the workers
/// stay at normal priority.
fn can_lower_preload_priority() -> bool {
    !crate::util::under_wine()
}

fn set_thread_priority_by_id(tid: u32, prio: i32) {
    if !can_lower_preload_priority() {
        return;
    }
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenThread, SetThreadPriority, THREAD_SET_LIMITED_INFORMATION};
    unsafe {
        let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, tid);
        if !h.is_null() {
            SetThreadPriority(h, prio);
            CloseHandle(h);
        }
    }
}

impl BoostToken {
    fn for_current_thread() -> Arc<Self> {
        let tid = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
        Arc::new(Self { tid, count: Mutex::new(0) })
    }

    #[cfg(test)]
    fn is_boosted(&self) -> bool {
        *self.count.lock().unwrap() > 0
    }

    fn boost(self: &Arc<Self>) -> BoostGuard {
        let mut c = self.count.lock().unwrap();
        if *c == 0 {
            set_thread_priority_by_id(self.tid, windows_sys::Win32::System::Threading::THREAD_PRIORITY_NORMAL);
        }
        *c += 1;
        BoostGuard(self.clone())
    }

    /// The workers' priority is `prio` from now on: this one's too, unless a bind waits for
    /// what it builds (then when that is built).
    fn rest(&self, prio: i32) {
        let c = self.count.lock().unwrap();
        if *c == 0 {
            set_thread_priority_by_id(self.tid, prio);
        }
    }
}

impl Drop for BoostGuard {
    fn drop(&mut self) {
        let mut c = self.0.count.lock().unwrap();
        *c = c.saturating_sub(1);
        if *c == 0 {
            set_thread_priority_by_id(self.0.tid, REST_PRIORITY.load(Ordering::Acquire));
        }
    }
}

/// Who builds a pipeline in the background ahead of its first bind, and whether a bind needs it.
#[derive(Default)]
struct PreloadOwner {
    /// the preload worker building it now
    worker: Option<Arc<BoostToken>>,
    /// a bind waits for it (`promote_preload`): a preload worker that has not started it skips it
    demanded: bool,
    /// the worker's raised priority while it builds a pipeline a bind waits for
    boost: Option<BoostGuard>,
}

struct PreloadJobGuard<'a>(&'a Pending);

impl Drop for PreloadJobGuard<'_> {
    fn drop(&mut self) {
        let mut owner = self.0.preload.lock().unwrap_or_else(|e| e.into_inner());
        owner.worker = None;
        owner.boost = None;
    }
}

struct Pending {
    /// the build, until a full-quality build of it finished
    job: Mutex<Option<Arc<JobFn>>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// a preload worker's build of it, and the bind that waits for it
    preload: Mutex<PreloadOwner>,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
    /// a full-quality build started (one runs: `build_full`)
    full_started: std::sync::atomic::AtomicBool,
    /// a build that may take the fast tier started (one runs: `build_tiered`)
    tiered_started: std::sync::atomic::AtomicBool,
    /// `real` came from the fast tier: a full-quality build replaces it
    real_fast: std::sync::atomic::AtomicBool,
    /// a bind started a build of its own (`resolve_wait`)
    urgent: std::sync::atomic::AtomicBool,
    /// counted in `COMPUTE_BOUND_PENDING` until its build is done
    counted: std::sync::atomic::AtomicBool,
    /// the fast-tier pipeline the optimized one replaced, kept until the stand-in goes (lists
    /// recorded with it may still run)
    retired: AtomicUsize,
    /// a bind (or a lookup made for one) took it: its real pipeline stays (`release_idle`)
    bound: std::sync::atomic::AtomicBool,
    /// when its real pipeline was built (`since_start`, ms; 0: not built)
    built_at: std::sync::atomic::AtomicU64,
    /// its real pipeline was released, nothing having bound it (`release_idle`)
    released_idle: std::sync::atomic::AtomicBool,
    /// a bind asked for it again after that: its next build is a rebuild
    rebuild: std::sync::atomic::AtomicBool,
    /// what its last build waited for by a bind was (`note_build`)
    note: Mutex<Option<String>>,
    /// its shaders, for the log (stage and hash of each)
    what: String,
    /// a draw was skipped for want of it (`backstop`): later binds do not wait for it either
    skipped: std::sync::atomic::AtomicBool,
    /// a build a bind waited for was refused for memory (`wait_for_memory`): binds take the
    /// stand-in at once until a background worker has built it
    starved: std::sync::atomic::AtomicBool,
    /// the title's root signature of the pipeline (0: unknown): what is bound next is likely of
    /// the group of what was bound last (`pso_sched::Backlog`)
    group: u64,
    preload_seq: AtomicU64,
}

impl Drop for Pending {
    fn drop(&mut self) {
        release(*self.retired.get_mut());
    }
}

impl Pending {
    /// A bind waits for this pipeline. True if a preload worker is building it already: that
    /// worker is raised to normal priority until it is done, and no second build of the same
    /// program is started (its driver compile takes 1.4 GB; `Governor` counts it as waited for).
    /// False: no worker has it, and a worker that takes it later skips it.
    fn promote_preload(&self) -> bool {
        let mut owner = self.preload.lock().unwrap_or_else(|e| e.into_inner());
        owner.demanded = true;
        let Some(worker) = owner.worker.clone() else { return false };
        if owner.boost.is_none() {
            owner.boost = Some(worker.boost());
        }
        true
    }

    /// A bind (or a lookup for one) takes this pipeline: from now on its real pipeline stays.
    fn note_bind(&self) {
        self.bound.store(true, Ordering::SeqCst);
        note_bound_group(self.group);
        if self.released_idle.swap(false, Ordering::AcqRel) {
            REBUILT.fetch_add(1, Ordering::Relaxed);
            self.rebuild.store(true, Ordering::Release);
        }
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
        }
    }
}

/// the groups (`Pending::group`) of the pipelines bound last, and how many were noted: the last
/// one is at that number less one, modulo the length
static RECENT_GROUPS: [AtomicU64; super::pso_sched::RECENT_GROUPS] = [const { AtomicU64::new(0) }; super::pso_sched::RECENT_GROUPS];
static RECENT_NOTED: AtomicUsize = AtomicUsize::new(0);

/// A pipeline of `group` is bound (every bind comes here: no lock, and a group bound again and
/// again is noted once).
fn note_bound_group(group: u64) {
    let noted = RECENT_NOTED.load(Ordering::Relaxed);
    if group == 0 || RECENT_GROUPS[noted.wrapping_sub(1) % RECENT_GROUPS.len()].load(Ordering::Relaxed) == group {
        return;
    }
    RECENT_GROUPS[noted % RECENT_GROUPS.len()].store(group, Ordering::Relaxed);
    RECENT_NOTED.store(noted.wrapping_add(1), Ordering::Relaxed);
}

/// The groups of the pipelines bound last, the last one first (0: none).
fn recent_groups() -> [u64; super::pso_sched::RECENT_GROUPS] {
    let noted = RECENT_NOTED.load(Ordering::Relaxed);
    std::array::from_fn(|k| RECENT_GROUPS[noted.wrapping_sub(1 + k) % RECENT_GROUPS.len()].load(Ordering::Relaxed))
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);

/// A build finished (`real`: its pipeline, 0 = failed; `fast`: one of the fast tier,
/// `dxc::fast_tier`). The first pipeline is used at once; an optimized one replaces a fast one.
/// A failed fast build leaves the pipeline waiting for its full-quality build.
fn complete(p: &Pending, real: usize, fast: bool) {
    let drop_now = {
        let mut done = p.done.lock().unwrap();
        let mut drop_now = 0;
        if p.orphan.load(Ordering::Acquire) {
            drop_now = real;
        } else if real != 0 {
            match p.real.load(Ordering::Acquire) {
                0 => {
                    BUILT.fetch_add(1, Ordering::Relaxed);
                    p.real.store(real, Ordering::SeqCst);
                    p.real_fast.store(fast, Ordering::Release);
                    p.built_at.store(since_start().max(1), Ordering::Release);
                }
                old if !fast && p.real_fast.swap(false, Ordering::AcqRel) => {
                    p.real.store(real, Ordering::Release);
                    drop_now = p.retired.swap(old, Ordering::AcqRel);
                }
                _ => drop_now = real,
            }
        } else if !fast && !*done {
            FAILED.fetch_add(1, Ordering::Relaxed);
        }
        if real != 0 || !fast {
            if p.counted.swap(false, Ordering::AcqRel) {
                COMPUTE_BOUND_PENDING.fetch_sub(1, Ordering::AcqRel);
            }
            *done = true;
            p.cv.notify_all();
        }
        drop_now
    };
    // outside `done`: the Release override takes PENDING (`counts` takes PENDING, then `done`)
    release(drop_now);
    // (one nothing bound keeps its build: its pipeline may be released and built again)
    if !fast && p.bound.load(Ordering::SeqCst) {
        *p.job.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

/// Milliseconds since XCL first asked.
fn since_start() -> u64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// How long a pipeline built when the title made it may wait for its first bind before its real
/// pipeline is released (`XCL_PSO_IDLE`: seconds, 0 = never): its build is kept, and a bind
/// builds it again from the disk caches (the translation's DXIL, the driver's blob – ~0.1 s).
/// A retail game makes ~51 000 pipelines while it loads and binds few; all of them built and
/// kept ran the process out of commit.
fn idle_limit() -> Option<std::time::Duration> {
    static L: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *L.get_or_init(|| {
        let secs = std::env::var("XCL_PSO_IDLE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(30u64);
        (secs > 0).then(|| std::time::Duration::from_secs(secs))
    })
}

/// pipelines released after waiting `idle_limit` for a first bind, and of those built again
static IDLE_RELEASED: AtomicUsize = AtomicUsize::new(0);
static REBUILT: AtomicUsize = AtomicUsize::new(0);

/// Release `p`'s real pipeline if nothing bound it in the `limit` since it was built (true:
/// released). Its stand-in and build stay; it waits for a bind again, as before it was built.
///
/// (A bind marks `bound` before it reads `real` or `done`, and this checks `bound` after it
/// took `real` away, all `SeqCst` and under `done`: either the bind finds the pipeline gone and
/// waits for it to be built again, or this finds the bind and puts the pipeline back.)
fn release_idle(p: &Pending, limit: std::time::Duration) -> bool {
    let built_at = p.built_at.load(Ordering::Acquire);
    if built_at == 0 || since_start().saturating_sub(built_at) < limit.as_millis() as u64 || p.bound.load(Ordering::SeqCst) || p.orphan.load(Ordering::Acquire) {
        return false;
    }
    if job_of(p).is_none() {
        return false;
    }
    let real = {
        let mut done = p.done.lock().unwrap();
        if !*done {
            return false;
        }
        let real = p.real.swap(0, Ordering::SeqCst);
        if real == 0 {
            return false;
        }
        if p.bound.load(Ordering::SeqCst) {
            p.real.store(real, Ordering::SeqCst);
            return false;
        }
        *done = false;
        p.built_at.store(0, Ordering::Release);
        p.full_started.store(false, Ordering::Release);
        p.tiered_started.store(false, Ordering::Release);
        p.urgent.store(false, Ordering::Release);
        p.preload.lock().unwrap_or_else(|e| e.into_inner()).demanded = false;
        p.sent.store(false, Ordering::Release);
        p.released_idle.store(true, Ordering::Release);
        real
    };
    release(real);
    IDLE_RELEASED.fetch_add(1, Ordering::Relaxed);
    true
}

/// How many built pipelines nothing has bound yet are kept at most (`XCL_PSO_IDLE_KEEP`, 4096 by
/// default: ~1.6 GB in an NVIDIA driver): beyond it the oldest are released before `idle_limit`.
/// Workers on every core build a retail game's pipelines faster than 30 s release them.
fn idle_keep() -> usize {
    static K: OnceLock<usize> = OnceLock::new();
    *K.get_or_init(|| std::env::var("XCL_PSO_IDLE_KEEP").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4096))
}

/// pipelines built ahead since the idle thread last looked, and its wake-up
static BUILT_SINCE_LOOK: Mutex<usize> = Mutex::new(0);
static LOOK: Condvar = Condvar::new();

/// A preload worker built a pipeline: the idle thread looks again once a quarter of `idle_keep`
/// were built since it last did, without waiting out its 5 s.
fn note_built_ahead() {
    let mut n = BUILT_SINCE_LOOK.lock().unwrap();
    *n += 1;
    if *n >= (idle_keep() / 4).max(1) {
        LOOK.notify_one();
    }
}

/// The thread that releases pipelines nothing bound (`idle_limit`, `idle_keep`), every 5 s or sooner
/// (`note_built_ahead`); it also drops the builds of bound ones (never released, never built again).
fn start_idle_release() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    let Some(limit) = idle_limit() else { return };
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-pso-idle".into()).spawn(move || loop {
            {
                let n = BUILT_SINCE_LOOK.lock().unwrap();
                let (mut n, _) = LOOK.wait_timeout_while(n, std::time::Duration::from_secs(5), |n| *n < (idle_keep() / 4).max(1)).unwrap();
                *n = 0;
            }
            let all: Vec<Arc<Pending>> = PENDING.lock().unwrap().as_ref().map_or_else(Vec::new, |m| m.values().cloned().collect());
            let mut unbound = Vec::new();
            for (at, p) in all.iter().enumerate() {
                if p.bound.load(Ordering::SeqCst) {
                    if *p.done.lock().unwrap() {
                        *p.job.lock().unwrap_or_else(|e| e.into_inner()) = None;
                    }
                } else {
                    match p.built_at.load(Ordering::Acquire) {
                        0 => {}
                        built_at => unbound.push((at, built_at)),
                    }
                }
            }
            let (kept, mut released, mut early) = (unbound.len(), 0, 0);
            for at in super::pso_sched::idle_to_release(&mut unbound, since_start(), limit.as_millis() as u64, idle_keep()) {
                let old = since_start().saturating_sub(all[at].built_at.load(Ordering::Acquire)) >= limit.as_millis() as u64;
                if release_idle(&all[at], if old { limit } else { std::time::Duration::ZERO }) {
                    released += 1;
                    early += !old as usize;
                }
            }
            if released > 0 {
                let (live, _, _) = live();
                crate::xlog_info!(
                    "D3D12.X: {released} pipelines nobody bound released ({} of them within {} s of their build: {kept} were kept, {} at most; {live} live; they are built again at their first bind)",
                    early,
                    limit.as_secs(),
                    idle_keep()
                );
            }
        });
    });
}

fn job_of(p: &Pending) -> Option<Arc<JobFn>> {
    p.job.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// The full-quality build (one runs).
fn build_full(p: &Pending) {
    if p.orphan.load(Ordering::Acquire) || p.full_started.swap(true, Ordering::AcqRel) {
        return;
    }
    if let Some(job) = job_of(p) {
        complete(p, run(&*job), false);
    }
}

/// The build something waits for: a big program the cache has no optimized DXIL for compiled
/// at the fast tier first (`dxc::fast_tier`), its full-quality build queued behind everything
/// else and swapped in when done.
fn build_tiered(p: &Arc<Pending>) {
    if p.orphan.load(Ordering::Acquire) || *p.done.lock().unwrap() || p.tiered_started.swap(true, Ordering::AcqRel) {
        return;
    }
    let Some(job) = job_of(p) else { return };
    // (no fast tier: -O0 DXIL took an NVIDIA driver longer than -O1 – 182 s against 123 s on a
    // retail game's biggest compute program – and DXC only ~half the time)
    STARVED.with(|s| s.set(None));
    let (real, fast) = (run(&*job), false);
    if STARVED.with(|s| s.take()).filter(|_| real == 0).is_some_and(|dxil_len| wait_for_memory(p, dxil_len)) {
        return;
    }
    if !fast {
        p.full_started.store(true, Ordering::Release);
    }
    complete(p, real, fast);
    if fast && !p.full_started.load(Ordering::Acquire) {
        enqueue_upgrade(p.clone());
    }
}

thread_local! {
    /// the driver compile (bytes of DXIL) the build this thread runs for a bind was refused for
    /// memory (`no_memory`)
    static STARVED: Cell<Option<usize>> = const { Cell::new(None) };
}

/// The build this thread runs for a bind was refused its driver compile of `dxil_len` bytes of
/// DXIL: it does not fit the memory even with no other compile running (`real::driver_slot`).
pub fn no_memory(dxil_len: usize) {
    STARVED.with(|s| s.set(Some(dxil_len)));
}

/// A build a bind waited for was refused for memory. Its bind goes without it (the stand-in, as
/// for a failed build), and so do the binds after it, at once – but it is not failed: it waits
/// with the builds put off for memory and is built by a background worker when its compile
/// fits, and the binds from then on use it. (Four of a retail game's giants were refused with
/// 4.6 GB available and their dispatches skipped for the rest of the run.) False: it is not to
/// be built again, and fails as before (released, or no background workers).
fn wait_for_memory(p: &Arc<Pending>, dxil_len: usize) -> bool {
    let Some(q) = preload() else { return false };
    if !mark_starved(p, dxil_len) {
        return false;
    }
    park(q, dxil_len, p);
    q.cv.notify_one();
    true
}

/// `p`'s driver compile of `dxil_len` bytes of DXIL waits for memory (`pso_sched::Backlog::park`).
fn park(q: &Preload, dxil_len: usize, p: &Arc<Pending>) {
    let mut list = q.list.lock().unwrap();
    list.park(dxil_len, p.starved.load(Ordering::Acquire), p.group, p.preload_seq.load(Ordering::Relaxed), Arc::downgrade(p));
    LEAST_PARKED.store(list.least_parked(), Ordering::Release);
}

/// `wait_for_memory`'s change to `p`, refused for memory just now: not done, waiting to be built
/// again, and the binds waiting for it woken (`resolve_wait`).
fn mark_starved(p: &Pending, dxil_len: usize) -> bool {
    if preload().is_none() {
        return false;
    }
    {
        let done = p.done.lock().unwrap();
        if *done || p.orphan.load(Ordering::Acquire) || job_of(p).is_none() {
            return false;
        }
        p.starved.store(true, Ordering::Release);
        p.full_started.store(false, Ordering::Release);
        p.tiered_started.store(false, Ordering::Release);
        p.urgent.store(false, Ordering::Release);
        p.preload.lock().unwrap_or_else(|e| e.into_inner()).demanded = false;
        if p.counted.swap(false, Ordering::AcqRel) {
            COMPUTE_BOUND_PENDING.fetch_sub(1, Ordering::AcqRel);
        }
        p.cv.notify_all();
    }
    crate::xlog_warn!(
        "D3D12.X: a {} pipeline's driver compile of {dxil_len} B of DXIL does not fit the memory now ({}): binds go without it until a background worker has built it, when it fits",
        if p.compute { "compute" } else { "graphics" },
        if p.what.is_empty() { "shaders unknown" } else { &p.what }
    );
    true
}

static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);
/// binds waiting right now for a pipeline still being built (`title_held`)
static HELD: AtomicUsize = AtomicUsize::new(0);

/// Whether a bind is waiting for a pipeline still being built: one of the title's threads is
/// held where a console title never waits (`kernelx::Sleep` does not count that time).
pub fn title_held() -> bool {
    HELD.load(Ordering::Acquire) != 0
}

/// A bind's wait for a pipeline still being built (`title_held`).
struct Held;

impl Held {
    fn new() -> Held {
        HELD.fetch_add(1, Ordering::AcqRel);
        Held
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        HELD.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

/// Builds the real pipeline: its raw pointer, 0 = failed. Called again for the pipeline once
/// its shaders were prepared (`preparing`).
pub type Job = Box<JobFn>;
type JobFn = dyn Fn() -> usize + Send + Sync;

/// Run a job; a panic in it is a failed build (a worker must not die with it, and whoever
/// waits for the pipeline must be woken).
fn run(job: &JobFn) -> usize {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).unwrap_or(0)
}

thread_local! {
    static PREPARING: Cell<bool> = const { Cell::new(false) };
}

/// Inside a job run to prepare its shaders: the translation and DXC run, the driver does not
/// (creating the pipeline fails at once, and what fails is not reported – the real build
/// reports it).
pub fn preparing() -> bool {
    PREPARING.with(|p| p.get())
}

/// `f` with the driver, inside a job preparing its shaders: the driver's compile of a big
/// program is worth having in its disk cache before the program's first use.
pub fn with_driver<R>(f: impl FnOnce() -> R) -> R {
    let was = PREPARING.with(|p| p.replace(false));
    let r = f();
    PREPARING.with(|p| p.set(was));
    r
}

/// Background threads that prepare lazy pipelines' shaders: half the compile threads
/// (`XCL_PREPARE_THREADS` to choose), so a title thread's own compile always finds a free DXC
/// worker and the title keeps half the CPU. 0: none.
pub fn prepare_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        if std::env::var("XCL_PREPARE_PSO").is_ok_and(|v| v == "0") {
            return 0;
        }
        let n = crate::gpu::recomp::dxc::compile_threads();
        std::env::var("XCL_PREPARE_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(n.div_ceil(2)).min(n)
    })
}

/// Prepare `p`'s shaders, unless it was built, started or released already.
fn prepare(p: &Pending) {
    if p.orphan.load(Ordering::Acquire) || p.full_started.load(Ordering::Acquire) || *p.done.lock().unwrap() {
        return;
    }
    let Some(job) = job_of(p) else { return };
    PREPARING.with(|f| f.set(true));
    let made = run(&*job);
    PREPARING.with(|f| f.set(false));
    release(made);
    let n = PREPARED.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() && n >= 64 || n % 2000 == 0 {
        let (running, compiled) = crate::gpu::recomp::dxc::compile_counts();
        crate::xlog_info!("D3D12.X: shaders of {n} pipelines prepared in the background ({compiled} DXC compiles so far, {running} running)");
    }
}

static PREPARED: AtomicUsize = AtomicUsize::new(0);

fn prepare_queue() -> Option<&'static Mutex<mpsc::Sender<Weak<Pending>>>> {
    static Q: OnceLock<Option<Mutex<mpsc::Sender<Weak<Pending>>>>> = OnceLock::new();
    Q.get_or_init(|| {
        let n = prepare_threads();
        if n == 0 {
            return None;
        }
        let (tx, rx) = mpsc::channel::<Weak<Pending>>();
        let rx = Arc::new(Mutex::new(rx));
        for k in 0..n {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-prep-{k}")).stack_size(64 << 20).spawn(move || loop {
                let next = rx.lock().unwrap().recv();
                let Ok(p) = next else { return };
                let _task = crate::progress::PIPELINES.take();
                // (None: the stand-in was released)
                if let Some(p) = p.upgrade() {
                    prepare(&p);
                }
            });
        }
        crate::xlog_info!("D3D12.X: shaders of pipelines made for later are prepared on {n} background threads");
        Some(Mutex::new(tx))
    })
    .as_ref()
}

#[derive(Default)]
struct BuildQueue {
    graphics: VecDeque<Arc<Pending>>,
    compute: VecDeque<Arc<Pending>>,
    /// optimized builds replacing fast-tier pipelines: after everything else
    upgrades: VecDeque<Arc<Pending>>,
    /// builds a bind waits for, with when the bind asked: before everything else
    urgent: VecDeque<(Arc<Pending>, std::time::Instant)>,
}

impl BuildQueue {
    fn push(&mut self, p: Arc<Pending>) {
        if p.compute {
            self.compute.push_back(p);
        } else {
            self.graphics.push_back(p);
        }
    }

    fn pop(&mut self) -> Option<Arc<Pending>> {
        self.graphics.pop_front().or_else(|| self.compute.pop_front())
    }
}

static QUEUE: Mutex<BuildQueue> = Mutex::new(BuildQueue {
    graphics: VecDeque::new(),
    compute: VecDeque::new(),
    upgrades: VecDeque::new(),
    urgent: VecDeque::new(),
});

/// threads that build only what binds wait for, besides the workers (which take it first too)
const URGENT_THREADS: usize = 4;
static READY: Condvar = Condvar::new();

fn start_workers() {
    static WORKERS: std::sync::Once = std::sync::Once::new();
    WORKERS.call_once(|| {
        let n = crate::gpu::recomp::dxc::compile_threads();
        for k in 0..n + URGENT_THREADS {
            let urgent_only = k >= n;
            let name = if urgent_only { format!("xcl-pso-urgent-{}", k - n) } else { format!("xcl-pso-{k}") };
            let _ = std::thread::Builder::new().name(name).stack_size(64 << 20).spawn(move || loop {
                let (p, upgrade, asked) = {
                    let mut queue = QUEUE.lock().unwrap();
                    loop {
                        let next = queue.urgent.pop_front().map(|(p, at)| (p, false, Some(at)));
                        let next = if urgent_only { next } else { next.or_else(|| queue.pop().map(|p| (p, false, None))).or_else(|| queue.upgrades.pop_front().map(|p| (p, true, None))) };
                        if let Some(next) = next {
                            break next;
                        }
                        queue = READY.wait(queue).unwrap();
                    }
                };
                let _task = crate::progress::PIPELINES.take();
                BUILDING.with(|b| *b.borrow_mut() = Some(p.clone()));
                let started = std::time::Instant::now();
                if upgrade {
                    build_full(&p);
                } else {
                    build_tiered(&p);
                }
                BUILDING.with(|b| *b.borrow_mut() = None);
                if let Some(at) = asked {
                    let (queued, built) = (started.duration_since(at), started.elapsed());
                    if queued + built >= std::time::Duration::from_millis(100) {
                        crate::xlog_info!(
                            "D3D12.X: a waited-for {} pipeline: {} ms in the queue, {} ms to build{}; {}",
                            if p.compute { "compute" } else { "graphics" },
                            queued.as_millis(),
                            built.as_millis(),
                            if p.rebuild.load(Ordering::Acquire) { " (a rebuild of one released unbound)" } else { "" },
                            p.note.lock().unwrap().as_deref().unwrap_or("no build details")
                        );
                    }
                }
            });
        }
    });
}

thread_local! {
    /// the pipeline this worker builds now (`waited_on`)
    static BUILDING: std::cell::RefCell<Option<Arc<Pending>>> = const { std::cell::RefCell::new(None) };
}

/// What the pipeline this worker builds for a waiting bind was made of and how (`f`: shader
/// hashes, DXIL sizes, translation and driver times; logged with the wait).
pub fn note_build(f: impl FnOnce() -> String) {
    BUILDING.with(|b| {
        if let Some(p) = b.borrow().as_ref().filter(|p| p.urgent.load(Ordering::Acquire)) {
            *p.note.lock().unwrap() = Some(f());
        }
    });
}

/// builds a bind started, the ms binds waited, and draws skipped (`backstop`), since `take_window`
static URGENT_BUILDS: AtomicUsize = AtomicUsize::new(0);
static WAIT_MS: AtomicUsize = AtomicUsize::new(0);
static SKIPS: AtomicUsize = AtomicUsize::new(0);
static WINDOW: Mutex<(usize, usize, usize, usize)> = Mutex::new((0, 0, 0, 0));

/// (builds a bind started, binds that waited, ms they waited in all, draws skipped) since the
/// last call
pub fn take_window() -> (usize, usize, usize, usize) {
    let (bind, used, _) = waits();
    let now = (URGENT_BUILDS.load(Ordering::Relaxed), bind + used, WAIT_MS.load(Ordering::Relaxed), SKIPS.load(Ordering::Relaxed));
    let last = std::mem::replace(&mut *WINDOW.lock().unwrap(), now);
    (now.0 - last.0, now.1 - last.1, now.2 - last.2, now.3 - last.3)
}

/// draws skipped so far (`backstop`)
pub fn skips() -> usize {
    SKIPS.load(Ordering::Relaxed)
}

#[cfg(test)]
thread_local! {
    /// (tests) this thread's backstop instead of `XCL_BIND_BACKSTOP_MS`
    pub(super) static BACKSTOP: Cell<Option<Option<std::time::Duration>>> = const { Cell::new(None) };
    /// (tests) this thread's frame wait budget instead of `XCL_FRAME_STALL_BUDGET_MS`
    pub(super) static FRAME_BUDGET: Cell<Option<Option<std::time::Duration>>> = const { Cell::new(None) };
}

thread_local! {
    /// this thread's bind waits in the frame being made: (frames shown when it began, time waited)
    static FRAME_WAIT: Cell<(u64, std::time::Duration)> = const { Cell::new((u64::MAX, std::time::Duration::ZERO)) };
}

/// draws skipped at once because their thread had used up its frame's wait budget
static BUDGET_SKIPS: AtomicUsize = AtomicUsize::new(0);

/// How long one thread's binds may wait for pipelines in one frame (until the next present)
/// before its draws with graphics pipelines still being built are skipped at once
/// (`XCL_FRAME_STALL_BUDGET_MS`, 1000 by default; 0: no budget). Dispatches always wait. A run
/// of new pipelines waited for one after another would otherwise hold a frame past the title's
/// GPU watchdog.
fn frame_budget() -> Option<std::time::Duration> {
    #[cfg(test)]
    if let Some(b) = FRAME_BUDGET.with(Cell::get) {
        return b;
    }
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| {
        let ms = std::env::var("XCL_FRAME_STALL_BUDGET_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1000u64);
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    })
}

/// What this thread's binds have waited in the frame being made (a new frame starts at zero).
fn frame_waited() -> std::time::Duration {
    let frame = super::present::presents();
    FRAME_WAIT.with(|w| match w.get() {
        (f, waited) if f == frame => waited,
        _ => {
            w.set((frame, std::time::Duration::ZERO));
            std::time::Duration::ZERO
        }
    })
}

fn add_frame_wait(waited: std::time::Duration) {
    FRAME_WAIT.with(|w| {
        let (f, total) = w.get();
        w.set((f, total + waited));
    });
}

/// draws skipped at once for the frame wait budget so far (`frame_budget`)
pub fn budget_skips() -> usize {
    BUDGET_SKIPS.load(Ordering::Relaxed)
}

/// How long a draw waits for its graphics pipeline before it is skipped for that bind
/// (`XCL_BIND_BACKSTOP_MS`, 3000 by default; 0: for ever). The build goes on ahead of the others
/// and a later bind uses it. A dispatch always waits: a skipped one could leave what the title
/// computes wrong (it is logged once it waited this long).
fn backstop() -> Option<std::time::Duration> {
    #[cfg(test)]
    if let Some(b) = BACKSTOP.with(Cell::get) {
        return b;
    }
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| {
        let ms = std::env::var("XCL_BIND_BACKSTOP_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(3000u64);
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    })
}

/// A bind waits for the pipeline this thread builds now (it may have started as background work
/// before): its driver compile goes ahead of the memory governor's limits.
pub fn waited_on() -> bool {
    // (`starved`: a bind asked for it and went without – it is built as what binds wait for are)
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.urgent.load(Ordering::Acquire) || p.starved.load(Ordering::Acquire)))
}

fn enqueue(p: Arc<Pending>) {
    start_workers();
    crate::progress::PIPELINES.queue();
    QUEUE.lock().unwrap().push(p);
    READY.notify_one();
}

fn enqueue_urgent(p: Arc<Pending>) {
    URGENT_BUILDS.fetch_add(1, Ordering::Relaxed);
    start_workers();
    crate::progress::PIPELINES.queue();
    QUEUE.lock().unwrap().urgent.push_back((p, std::time::Instant::now()));
    READY.notify_all();
}

fn enqueue_upgrade(p: Arc<Pending>) {
    start_workers();
    crate::progress::PIPELINES.queue();
    QUEUE.lock().unwrap().upgrades.push_back(p);
    READY.notify_one();
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true, 0, String::new(), 0, 0);
}

/// `submit` with what the pipeline is made of (`what`: its shaders, for the log).
pub fn submit_described(standin: usize, compute: bool, job: Job, what: String) {
    submit_as(standin, compute, job, true, 0, what, 0, 0);
}

/// Lazy pipeline creation (`XCL_LAZY_PSO=0`: off): a title's pipelines with console shaders get
/// their stand-in when made and are built in the background (`submit_lazy`), not in the call. A
/// retail game makes ~51 000 pipelines in its first 30 s and uses few; each one built stays in
/// the driver (~400 KB), and the process ran out of commit – so one nobody binds is released
/// again (`idle_limit`).
pub fn lazy() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// `submit`, built by the background workers (`preload`) below normal priority, in the order of
/// `pso_sched::Backlog`: the ones with the most console code first, then the ones of shaders an
/// earlier run used (`key`, 0 = none; `used`), then compute, then graphics. A bind sends it
/// ahead of all of them. (`XCL_PSO_PRELOAD=0`: built at its first `resolve`, its shaders prepared
/// before.)
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    submit_as(standin, compute, job, false, key, String::new(), 0, 0);
}

/// `submit_lazy` with what the pipeline is made of (`what`: its shaders, for the log;
/// `shader_bytes`: its console shaders' size).
pub fn submit_lazy_described(standin: usize, compute: bool, job: Job, key: u64, what: String, shader_bytes: usize, group: u64) {
    submit_as(standin, compute, job, false, key, what, shader_bytes, group);
}

#[allow(clippy::too_many_arguments)]
fn submit_as(standin: usize, compute: bool, job: Job, send: bool, key: u64, what: String, shader_bytes: usize, group: u64) {
    let p = Arc::new(Pending {
        job: Mutex::new(Some(Arc::from(job))),
        sent: std::sync::atomic::AtomicBool::new(send),
        preload: Default::default(),
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        key,
        used: Default::default(),
        orphan: Default::default(),
        full_started: Default::default(),
        tiered_started: Default::default(),
        real_fast: Default::default(),
        urgent: Default::default(),
        counted: Default::default(),
        retired: Default::default(),
        bound: Default::default(),
        built_at: Default::default(),
        released_idle: Default::default(),
        rebuild: Default::default(),
        note: Default::default(),
        what,
        skipped: Default::default(),
        starved: Default::default(),
        group,
        preload_seq: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        enqueue(p);
    } else {
        start_idle_release();
        if let Some(q) = preload() {
            crate::progress::PIPELINES.queue();
            let mut list = q.list.lock().unwrap();
            let seq = list.push(shader_bytes, used::known(key), compute, group, Arc::downgrade(&p));
            p.preload_seq.store(seq, Ordering::Relaxed);
            drop(list);
            q.cv.notify_one();
        } else if let Some(q) = prepare_queue() {
            crate::progress::PIPELINES.queue();
            if q.lock().unwrap().send(Arc::downgrade(&p)).is_err() {
                drop(crate::progress::PIPELINES.take());
            }
        }
    }
}

/// The pipelines earlier runs used, by the hash of their shaders (`xcl warm-up` builds them first).
pub fn used_earlier() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(used::path()).unwrap_or_default())
}

/// The pipelines a draw of an earlier run was skipped for (`backstop`; the warm-up builds them
/// before the others an earlier run bound).
pub fn skipped_earlier() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(used::skipped_path()).unwrap_or_default())
}

/// The pipelines a title used, by a hash of their shaders, kept between runs
/// (`used-pipelines.txt` beside the DXIL cache, a key a line).
///
/// A lazy pipeline is built at its first use: until then a graphics one draws nothing, and a
/// compute one is waited for on the title's thread. A retail title's first frames of its world
/// use hundreds of pipelines for the first time at once. The pipelines of an earlier run are
/// built while the title loads instead (`preload`): measured on a build that skips a dispatch
/// instead of waiting for its pipeline, ~2 170 dispatches were skipped in those frames without
/// this and 94 – 150 with it.
mod used {
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock, RwLock};

    struct Used {
        earlier: RwLock<HashSet<u64>>,
        now: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
    }

    pub fn path() -> std::path::PathBuf {
        crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt")
    }

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).filter(|&k| k != 0).collect()
    }

    fn get() -> &'static Used {
        static U: OnceLock<Used> = OnceLock::new();
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used { earlier: RwLock::new(HashSet::new()), now: Mutex::new((HashSet::new(), None)) };
            }
            let path = path();
            let earlier = parse(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
            Used { now: Mutex::new((earlier.clone(), file)), earlier: RwLock::new(earlier) }
        })
    }

    /// An earlier run used a pipeline of these shaders.
    pub fn known(key: u64) -> bool {
        key != 0 && get().earlier.read().unwrap().contains(&key)
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        if key == 0 {
            return;
        }
        let mut g = get().now.lock().unwrap();
        if g.0.insert(key) {
            if let Some(f) = g.1.as_mut() {
                let _ = writeln!(f, "{key:016x}");
            }
        }
    }

    /// `skipped-pipelines.txt` beside `used-pipelines.txt`: the pipelines a draw was skipped for
    /// (`backstop`), which the next warm-up builds first.
    pub fn skipped_path() -> std::path::PathBuf {
        crate::gpu::recomp::dxc::cache_dir().join("skipped-pipelines.txt")
    }

    /// A draw was skipped for want of a pipeline of these shaders: kept for the next warm-up
    /// (once a run; a test writes nothing).
    pub fn note_skipped(key: u64) {
        static NOTED: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
        if cfg!(test) || !NOTED.lock().unwrap().get_or_insert_with(HashSet::new).insert(key) {
            return;
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(skipped_path()) {
            let _ = writeln!(f, "{key:016x}");
        }
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.lock().unwrap().0.contains(&key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        get().earlier.write().unwrap().insert(key);
    }
}

/// The workers that build every pipeline the title makes ahead of its first use
/// (`XCL_PSO_PRELOAD=0`: off), in the order of `pso_sched::Backlog`. Apart from the ones a bind
/// sends to (`queue`) and below normal priority: a pipeline that is bound is not held up by the ones
/// made ahead (whoever takes `job` first builds it), and the title's own threads are not either.
/// Their big driver compiles are governed (`real::DRIVER_COMPILES`), and none of them waits there:
/// one that does not fit the memory now is put off (`put_off`) and taken up again when it fits.
/// (Two workers that waited in the governor – one 278 s for a 1.9 MB program, one 324 s for a 0.6 MB
/// one, with 4 GB available and 6 GB asked for – built 256 of a retail game's 87 000 pipelines in
/// 14 minutes, and every bind compiled its own.)
struct Preload {
    list: Mutex<super::pso_sched::Backlog<Weak<Pending>>>,
    cv: Condvar,
}

/// How many preload workers there are (`XCL_PSO_PRELOAD_THREADS`; `pso_sched::background_workers`
/// of the cores); how many of them build at once is `pace`'s. Under Wine they cannot run below
/// normal priority and stay few.
fn preload_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        if let Some(n) = std::env::var("XCL_PSO_PRELOAD_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
            return n.clamp(1, 64);
        }
        if !can_lower_preload_priority() {
            return (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        }
        super::pso_sched::background_workers(std::thread::available_parallelism().map_or(4, |n| n.get()), physical_cores())
    })
}

/// What a preload worker's build may do about a big driver compile that does not fit the memory
#[derive(Clone, Copy, PartialEq, Eq)]
enum PutOff {
    /// wait for the memory (every other thread)
    No,
    /// leave it (`may_put_off`)
    Allowed,
    /// it left one of so many bytes of DXIL (`put_off`)
    Left(usize),
}

thread_local! {
    static PUT_OFF: Cell<PutOff> = const { Cell::new(PutOff::No) };
}

/// On a preload worker, in a build nothing has to wait for: a big driver compile that does not
/// fit the memory now is not waited for (`real::driver_slot`).
pub fn may_put_off() -> bool {
    PUT_OFF.with(|d| d.get()) != PutOff::No
}

/// The build this worker runs leaves its driver compile of `dxil_len` bytes of DXIL for later: it
/// did not fit the memory. The build fails without an error, and its pipeline waits to be built
/// when the compile fits (`pso_sched::Backlog::park`) or a bind asks for it.
pub fn put_off(dxil_len: usize) {
    PUT_OFF.with(|d| {
        if d.get() != PutOff::No {
            d.set(PutOff::Left(dxil_len));
        }
    });
}

/// The full-quality build on a preload worker (one runs). Some(DXIL bytes): its big driver
/// compile did not fit the memory and nothing waits for it – not built, to be parked. A bind that
/// came for it meanwhile (`promote_preload` found this worker) gets it built here and now, waiting
/// for memory like every build a bind waits for.
fn build_ahead(p: &Pending) -> Option<usize> {
    if p.orphan.load(Ordering::Acquire) || p.full_started.swap(true, Ordering::AcqRel) {
        return None;
    }
    let job = job_of(p)?;
    PUT_OFF.with(|d| d.set(PutOff::Allowed));
    let mut real = run(&*job);
    if let (0, PutOff::Left(dxil_len)) = (real, PUT_OFF.with(|d| d.replace(PutOff::No))) {
        {
            let mut owner = p.preload.lock().unwrap_or_else(|e| e.into_inner());
            if !owner.demanded {
                // (under `preload`: a bind from here on finds no worker and starts its own build)
                owner.worker = None;
                p.full_started.store(false, Ordering::Release);
                return Some(dxil_len);
            }
        }
        // (`None`: built here it may wait for memory and be refused it, as on a bind's own thread)
        STARVED.with(|s| s.set(None));
        real = run(&*job);
        if let Some(dxil_len) = STARVED.with(|s| s.take()).filter(|_| real == 0 && mark_starved(p, dxil_len)) {
            return Some(dxil_len);
        }
    }
    complete(p, real, false);
    None
}

/// pipelines the preload workers built, and builds they put off for memory (`build_ahead`)
static AHEAD: AtomicUsize = AtomicUsize::new(0);
static PUT_OFFS: AtomicUsize = AtomicUsize::new(0);

/// (tests) the DXIL lengths there is no memory for, instead of asking the governor
#[cfg(test)]
static NO_ROOM: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// (tests) Whether a driver compile of so many bytes of DXIL would start now.
#[cfg(test)]
fn room_for(dxil_len: usize) -> bool {
    !NO_ROOM.lock().unwrap().contains(&dxil_len)
}

/// the smallest driver compile parked for memory, in bytes of DXIL (0: none parked)
static LEAST_PARKED: AtomicUsize = AtomicUsize::new(0);

/// How often the workers ask whether memory has come for the parked builds: it comes back
/// unannounced (the game's own, another process's).
const PARKED_POLL: std::time::Duration = std::time::Duration::from_millis(200);

/// Which parked driver compiles (by their bytes of DXIL, and whether a bind asked for them)
/// would start now, with the memory as it is now – asked once every `PARKED_POLL` by whichever worker comes first, outside the
/// backlog's lock (the title's threads queue their new pipelines under it). None: nothing is
/// parked, or another worker asked just now.
fn parked_room() -> Option<Box<dyn Fn(usize, bool) -> bool>> {
    let least = LEAST_PARKED.load(Ordering::Acquire);
    if least == 0 {
        return None;
    }
    static ASKED: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    {
        let mut asked = ASKED.lock().unwrap();
        if asked.is_some_and(|t| t.elapsed() < PARKED_POLL) {
            return None;
        }
        *asked = Some(std::time::Instant::now());
    }
    #[cfg(test)]
    {
        Some(Box::new(|dxil_len, _| room_for(dxil_len)))
    }
    #[cfg(not(test))]
    {
        Some(Box::new(super::real::background_room(least)))
    }
}

/// Cores of the PC (its hardware threads when the system does not tell them apart).
fn physical_cores() -> usize {
    use windows_sys::Win32::System::SystemInformation::{GetLogicalProcessorInformation, RelationProcessorCore, SYSTEM_LOGICAL_PROCESSOR_INFORMATION};
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut bytes = 0u32;
    unsafe { GetLogicalProcessorInformation(std::ptr::null_mut(), &mut bytes) };
    let each = std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>();
    let mut info: Vec<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> = vec![unsafe { std::mem::zeroed() }; bytes as usize / each];
    let mut bytes = (info.len() * each) as u32;
    if info.is_empty() || unsafe { GetLogicalProcessorInformation(info.as_mut_ptr(), &mut bytes) } == 0 {
        return threads;
    }
    match info[..(bytes as usize / each).min(info.len())].iter().filter(|i| i.Relationship == RelationProcessorCore).count() {
        0 => threads,
        cores => cores.min(threads),
    }
}

fn filetime(t: &windows_sys::Win32::Foundation::FILETIME) -> u64 {
    (t.dwHighDateTime as u64) << 32 | t.dwLowDateTime as u64
}

/// CPU time this process has used, in 100 ns.
fn process_cpu() -> u64 {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let mut t = [windows_sys::Win32::Foundation::FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 }; 4];
    let [made, gone, kernel, user] = &mut t;
    if unsafe { GetProcessTimes(GetCurrentProcess(), made, gone, kernel, user) } == 0 {
        return 0;
    }
    filetime(kernel) + filetime(user)
}

/// CPU time thread `tid` has used, in 100 ns (0: gone).
fn thread_cpu(tid: u32) -> u64 {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{GetThreadTimes, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
    let mut t = [windows_sys::Win32::Foundation::FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 }; 4];
    let [made, gone, kernel, user] = &mut t;
    unsafe {
        let h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid);
        if h.is_null() {
            return 0;
        }
        let ok = GetThreadTimes(h, made, gone, kernel, user);
        CloseHandle(h);
        if ok == 0 {
            return 0;
        }
    }
    filetime(kernel) + filetime(user)
}

/// How often the pace of the preload workers is set again (`pso_sched::Pace::step`), and how
/// long a worker that may not build now rests before it asks again.
#[cfg(not(test))]
const PACE_STEP: std::time::Duration = std::time::Duration::from_millis(500);
const PACE_REST: std::time::Duration = std::time::Duration::from_millis(100);

/// preload workers that may build at once now (`pace`; all of them until it has started), the
/// ones building, and the CPU the rest of the process used in the last step (thousandths of a
/// thread)
static PACE_WORKERS: AtomicUsize = AtomicUsize::new(usize::MAX);
static BUILDING_AHEAD: AtomicUsize = AtomicUsize::new(0);
static PACE_OTHERS: AtomicU64 = AtomicU64::new(0);
/// the title loads through its storage API (`pso_sched::loading`, set by `pace`; until it has
/// run it does: a title starts by loading)
static TITLE_LOADING: AtomicBool = AtomicBool::new(true);

/// Whether the title loads now, as far as its storage API tells.
pub fn title_loading() -> bool {
    TITLE_LOADING.load(Ordering::Relaxed)
}

/// The pace of the `pool` preload workers, set every `PACE_STEP` (`pso_sched::Pace`): how many
/// build at once – from the CPU the process uses apart from them, and whether the title loads
/// through DirectStorage (`pso_sched::loading`) – and their priority. A worker over the number finishes what it
/// builds and takes nothing new; one a bind waits on keeps normal priority (`BoostToken`), and a
/// bind that finds no worker on its pipeline builds it on a thread of its own (`resolve_wait`),
/// so nothing a bind waits for is held up here. Each step also samples the memory for the
/// floor of the background compiles (`real::background_floor`).
#[cfg(not(test))]
fn pace(pool: usize, workers: mpsc::Receiver<Arc<BoostToken>>) {
    let workers: Vec<Arc<BoostToken>> = workers.into_iter().take(pool).collect();
    let mut pace = super::pso_sched::Pace::new(pool, physical_cores());
    let cpu = |workers: &[Arc<BoostToken>]| (process_cpu(), workers.iter().map(|w| thread_cpu(w.tid)).sum::<u64>());
    let storage = || {
        let (requests, bytes, _) = crate::hle::dstorage::counters();
        (requests, bytes)
    };
    let (mut at, mut before, mut ran) = (std::time::Instant::now(), cpu(&workers), storage());
    super::real::sample_background_headroom();
    loop {
        std::thread::sleep(PACE_STEP);
        let (now, after, read) = (std::time::Instant::now(), cpu(&workers), storage());
        let wall = ((now - at).as_nanos() / 100).max(1);
        let others = ((after.0.saturating_sub(before.0)).saturating_sub(after.1.saturating_sub(before.1)) as u128 * 1000 / wall) as u64;
        let loading = super::pso_sched::loading(read.0.saturating_sub(ran.0), read.1.saturating_sub(ran.1), (now - at).as_millis() as u64);
        TITLE_LOADING.store(loading, Ordering::Relaxed);
        let tempo = pace.step(others, loading);
        let rest = if tempo.low { LOADING_PRIORITY } else { PRELOAD_PRIORITY };
        if REST_PRIORITY.swap(rest, Ordering::AcqRel) != rest {
            for w in &workers {
                w.rest(rest);
            }
        }
        PACE_WORKERS.store(tempo.workers, Ordering::Release);
        PACE_OTHERS.store(others, Ordering::Relaxed);
        super::real::sample_background_headroom();
        (at, before, ran) = (now, after, read);
    }
}

/// For the 30 s line: what the preload workers have built, what waits for them, what waits for
/// memory, and their pace now.
pub fn ahead_summary() -> String {
    let (waiting, parked) = Q.get().and_then(|q| q.as_ref()).map_or((0, 0), |q| q.list.lock().unwrap().waiting());
    let pool = if Q.get().is_some_and(|q| q.is_some()) { preload_threads() } else { 0 };
    format!(
        "built ahead {} on {} workers ({} waiting, {parked} of them for memory; put off {} times; now {} of up to {} building{}, beside {:.1} threads of other work, {}{})",
        AHEAD.load(Ordering::Relaxed),
        pool,
        waiting,
        PUT_OFFS.load(Ordering::Relaxed),
        BUILDING_AHEAD.load(Ordering::Relaxed),
        PACE_WORKERS.load(Ordering::Relaxed).min(pool),
        if REST_PRIORITY.load(Ordering::Relaxed) == LOADING_PRIORITY { " at the lowest priority" } else { "" },
        PACE_OTHERS.load(Ordering::Relaxed) as f64 / 1000.0,
        super::real::background_memory_summary(),
        if title_loading() { ", the title loads" } else { "" }
    )
}

static Q: OnceLock<Option<Preload>> = OnceLock::new();

fn preload() -> Option<&'static Preload> {
    static STARTED: std::sync::Once = std::sync::Once::new();
    let q = Q
        .get_or_init(|| {
            if std::env::var("XCL_PSO_PRELOAD").is_ok_and(|v| v.trim() == "0") {
                return None;
            }
            Some(Preload { list: Mutex::new(Default::default()), cv: Condvar::new() })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        let n = preload_threads();
        crate::xlog_info!(
            "D3D12.X: pipelines are built when the title makes them on {n} background threads ({} cores): as many at once as the title's own threads leave cores for, fewer and at the lowest priority while it loads",
            physical_cores()
        );
        let (tokens, started) = mpsc::channel();
        #[cfg(not(test))]
        {
            // (from the least: a title starts by loading)
            PACE_WORKERS.store(super::pso_sched::least_workers(n), Ordering::Release);
            let _ = std::thread::Builder::new().name("xcl-pace".into()).spawn(move || pace(n, started));
        }
        #[cfg(test)]
        drop(started);
        for k in 0..n {
            let tokens = tokens.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
                if can_lower_preload_priority() {
                    unsafe {
                        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority};
                        SetThreadPriority(GetCurrentThread(), REST_PRIORITY.load(Ordering::Acquire));
                    }
                }
                let token = BoostToken::for_current_thread();
                let _ = tokens.send(token.clone());
                drop(tokens);
                loop {
                    let (p, lane) = loop {
                        // (not on `cv`: a pipeline queued must wake a worker that may take it)
                        if BUILDING_AHEAD.load(Ordering::Acquire) >= PACE_WORKERS.load(Ordering::Acquire) {
                            std::thread::sleep(PACE_REST);
                            continue;
                        }
                        let room = parked_room();
                        let mut g = q.list.lock().unwrap();
                        // (under the backlog's lock, as every take is: never more than the pace allows)
                        if BUILDING_AHEAD.load(Ordering::Acquire) >= PACE_WORKERS.load(Ordering::Acquire) {
                            continue;
                        }
                        let wanted = |p: &Weak<Pending>| p.upgrade().is_some_and(|p| !p.orphan.load(Ordering::Acquire) && !p.full_started.load(Ordering::Acquire) && !*p.done.lock().unwrap());
                        let next = g.pop(n, &recent_groups(), wanted, |dxil_len, asked| room.as_ref().is_some_and(|fits| fits(dxil_len, asked)));
                        LEAST_PARKED.store(g.least_parked(), Ordering::Release);
                        if let Some(next) = next {
                            BUILDING_AHEAD.fetch_add(1, Ordering::AcqRel);
                            break next;
                        }
                        if g.has_parked() {
                            drop(q.cv.wait_timeout(g, PARKED_POLL).unwrap());
                        } else {
                            drop(q.cv.wait(g).unwrap());
                        }
                    };
                    struct Taken(&'static Preload, super::pso_sched::Lane);
                    impl Drop for Taken {
                        fn drop(&mut self) {
                            self.0.list.lock().unwrap_or_else(|e| e.into_inner()).done(self.1);
                            BUILDING_AHEAD.fetch_sub(1, Ordering::AcqRel);
                        }
                    }
                    let _taken = Taken(q, lane);
                    // (a parked one was counted as taken when it was first taken)
                    let _task = (lane != super::pso_sched::Lane::Parked).then(|| crate::progress::PIPELINES.take());
                    let Some(p) = p.upgrade() else { continue };
                    if p.orphan.load(Ordering::Acquire) {
                        continue;
                    }
                    if *p.done.lock().unwrap() || p.full_started.load(Ordering::Acquire) {
                        continue;
                    }
                    {
                        let mut owner = p.preload.lock().unwrap_or_else(|e| e.into_inner());
                        if owner.demanded {
                            // a bind's own build of it is queued or running (`resolve_wait`)
                            continue;
                        }
                        owner.worker = Some(token.clone());
                    }
                    let _owned = PreloadJobGuard(&p);
                    BUILDING.with(|b| *b.borrow_mut() = Some(p.clone()));
                    let left = build_ahead(&p);
                    BUILDING.with(|b| *b.borrow_mut() = None);
                    if let Some(dxil_len) = left {
                        let times = PUT_OFFS.fetch_add(1, Ordering::Relaxed) + 1;
                        if times.is_power_of_two() {
                            crate::xlog_info!(
                                "D3D12.X: a background build put off: its driver compile of {dxil_len} B of DXIL does not fit the memory now ({}; #{times}) – built when it fits, or when a bind asks for it",
                                if p.what.is_empty() { "shaders unknown" } else { &p.what }
                            );
                        }
                        park(q, dxil_len, &p);
                        continue;
                    }
                    let n = AHEAD.fetch_add(1, Ordering::Relaxed) + 1;
                    if n.is_power_of_two() {
                        crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
                    }
                    note_built_ahead();
                }
            });
        }
    });
    Some(q)
}

#[cfg(test)]
mod used_tests {
    use super::*;

    /// While a bind waits for a pipeline still being built, the title is held (`title_held`):
    /// `kernelx::Sleep` does not count that time.
    #[test]
    fn a_bind_waiting_for_a_build_holds_the_title() {
        let standin = 0x7E57_B002usize;
        let (started, building) = mpsc::channel();
        let (go, finish) = mpsc::channel::<()>();
        let (started, finish) = (Mutex::new(started), Mutex::new(finish));
        submit_lazy(
            standin,
            true,
            Box::new(move || {
                let _ = started.lock().unwrap().send(());
                let _ = finish.lock().unwrap().recv();
                0x5252
            }),
            0,
        );
        let bind = std::thread::spawn(move || resolve_wait(standin));
        building.recv_timeout(std::time::Duration::from_secs(10)).expect("the pipeline is being built");
        let t = std::time::Instant::now();
        while !title_held() && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(title_held(), "a bind waits for a build, but the title is not held");
        go.send(()).unwrap();
        assert_eq!(bind.join().unwrap(), 0x5252);
        let _ = released(standin);
    }

    fn thread_priority_of(tid: u32) -> i32 {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{GetThreadPriority, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
        unsafe {
            let h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid);
            assert!(!h.is_null(), "OpenThread({tid}) failed");
            let p = GetThreadPriority(h);
            CloseHandle(h);
            p
        }
    }

    /// The pace's change of priority reaches a worker at once, unless a bind waits for what it
    /// builds: that one stays at normal priority until it is built.
    #[test]
    fn a_worker_takes_the_priority_of_the_pace_unless_a_bind_waits_on_it() {
        use windows_sys::Win32::System::Threading::THREAD_PRIORITY_NORMAL;
        if !can_lower_preload_priority() {
            return;
        }
        let (tok_tx, tok_rx) = mpsc::channel();
        let (end_tx, end_rx) = mpsc::channel::<()>();
        let t = std::thread::spawn(move || {
            tok_tx.send(BoostToken::for_current_thread()).unwrap();
            let _ = end_rx.recv();
        });
        let tok: Arc<BoostToken> = tok_rx.recv().unwrap();
        tok.rest(LOADING_PRIORITY);
        assert_eq!(thread_priority_of(tok.tid), LOADING_PRIORITY);
        tok.rest(PRELOAD_PRIORITY);
        assert_eq!(thread_priority_of(tok.tid), PRELOAD_PRIORITY);
        let waited_on = tok.boost();
        tok.rest(LOADING_PRIORITY);
        assert_eq!(thread_priority_of(tok.tid), THREAD_PRIORITY_NORMAL, "a bind waits on it: not lowered");
        drop(waited_on);
        assert_eq!(thread_priority_of(tok.tid), REST_PRIORITY.load(Ordering::Acquire), "the workers' priority once its bind has it");
        drop(end_tx);
        t.join().unwrap();
    }

    /// What the pace is set from: the CPU a thread used is in the CPU the process used, and the
    /// PC has no more cores than hardware threads.
    #[test]
    fn the_cpu_of_a_thread_is_part_of_the_cpu_of_the_process() {
        let (tid_tx, tid_rx) = mpsc::channel();
        let (p0, started) = (process_cpu(), std::time::Instant::now());
        let t = std::thread::spawn(move || {
            tid_tx.send(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }).unwrap();
            let mut x = 1u64;
            while started.elapsed() < std::time::Duration::from_millis(300) {
                x = std::hint::black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1));
            }
            // (its times are read while it lives)
            std::thread::sleep(std::time::Duration::from_millis(200));
        });
        let tid = tid_rx.recv().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(350));
        let (thread, process) = (thread_cpu(tid), process_cpu() - p0);
        t.join().unwrap();
        // (100 ns units; a busy PC may give the thread less than its 300 ms)
        assert!(thread >= 500_000, "the thread's CPU time is read: {thread}");
        assert!(process >= thread * 9 / 10, "process {process}, thread {thread}");
        assert_eq!(thread_cpu(u32::MAX - 3), 0, "no such thread");
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        assert!((1..=threads).contains(&physical_cores()), "{} cores, {threads} threads", physical_cores());
    }

    #[test]
    fn boost_token_refcounts_nested_guards() {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_NORMAL};
        let check_os = can_lower_preload_priority();
        let (tok_tx, tok_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let h = std::thread::spawn(move || {
            if check_os {
                unsafe { SetThreadPriority(GetCurrentThread(), PRELOAD_PRIORITY) };
            }
            tok_tx.send(BoostToken::for_current_thread()).unwrap();
            let _ = done_rx.recv();
        });
        let tok = tok_rx.recv().unwrap();
        assert!(!tok.is_boosted());
        let g1 = tok.boost();
        assert!(tok.is_boosted());
        if check_os {
            assert_eq!(thread_priority_of(tok.tid), THREAD_PRIORITY_NORMAL);
        }
        let g2 = tok.boost();
        drop(g1);
        assert!(tok.is_boosted(), "still boosted while the second guard lives");
        if check_os {
            assert_eq!(thread_priority_of(tok.tid), THREAD_PRIORITY_NORMAL);
        }
        drop(g2);
        assert!(!tok.is_boosted(), "unboosted once the last guard drops");
        if check_os {
            assert_eq!(thread_priority_of(tok.tid), PRELOAD_PRIORITY, "restored once the last guard drops");
        }
        let _ = done_tx.send(());
        h.join().unwrap();
    }

    /// A bind that waits for a pipeline a preload worker is building raises that worker to normal
    /// priority until it is done (and starts no second build); a pipeline a bind waited for
    /// before a preload worker took it is skipped by that worker.
    #[test]
    fn resolve_wait_boosts_the_preload_worker_building_it_and_restores_it() {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadPriority, THREAD_PRIORITY_NORMAL};
        let check_os = can_lower_preload_priority();
        let (standin, key) = (0x7E57_7011usize, 0x5EED_0011u64);
        used::earlier_run_used(key);
        let builds = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = mpsc::channel();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let go_rx = Mutex::new(go_rx);
        let counted = builds.clone();
        submit_lazy(
            standin,
            true,
            Box::new(move || {
                counted.fetch_add(1, Ordering::SeqCst);
                let tok = entry(standin).unwrap().preload.lock().unwrap().worker.clone().expect("a preload worker builds it");
                let tid = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
                started_tx.send((tid, tok.clone(), unsafe { GetThreadPriority(GetCurrentThread()) })).unwrap();
                let _ = go_rx.lock().unwrap().recv();
                assert!(tok.is_boosted(), "raised while a bind waits for it");
                0x5678
            }),
            key,
        );
        let (tid, tok, before) = started_rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap();
        assert_eq!(before, if check_os { PRELOAD_PRIORITY } else { THREAD_PRIORITY_NORMAL });
        let waiter = std::thread::spawn(move || resolve_wait(standin));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !tok.is_boosted() || (check_os && thread_priority_of(tid) != THREAD_PRIORITY_NORMAL) {
            assert!(std::time::Instant::now() < deadline, "the preload worker was not raised");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        go_tx.send(()).unwrap();
        assert_eq!(waiter.join().unwrap(), 0x5678);
        assert_eq!(builds.load(Ordering::SeqCst), 1, "the bind did not start a second build");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while tok.is_boosted() || (check_os && thread_priority_of(tid) != PRELOAD_PRIORITY) {
            assert!(std::time::Instant::now() < deadline, "the worker was not lowered again after the build");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        released(standin);
    }

    #[test]
    fn the_keys_of_a_file_are_read_and_other_lines_passed_over() {
        let keys = used::parse("00000000000000ff\n\nnot a key\n  1a2b \n0\n");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0xff) && keys.contains(&0x1a2b));
    }

    /// Every lazy pipeline is built when it is made, with no bind – those of shaders an earlier
    /// run used and new ones alike – and a first use is kept for the next run.
    #[test]
    fn a_pipeline_is_built_when_made_and_its_first_use_kept() {
        let (known, unknown) = (0x7E57_7001usize, 0x7E57_7002usize);
        let (known_key, unknown_key) = (0x5EED_0001u64, 0x5EED_0002u64);
        used::earlier_run_used(known_key);
        submit_lazy(known, false, Box::new(|| if preparing() { 0 } else { 0x4321 }), known_key);
        submit_lazy(unknown, false, Box::new(|| if preparing() { 0 } else { 0x4322 }), unknown_key);
        let start = std::time::Instant::now();
        while (is_pending(known) || is_pending(unknown)) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(known) && !is_pending(unknown), "built by the background workers, with no bind");
        assert!(!used::noted(unknown_key), "built is not used");
        assert_eq!(resolve(known), 0x4321);
        assert_eq!(resolve_wait(unknown), 0x4322);
        assert!(used::noted(unknown_key) && used::noted(known_key), "a first use is kept for the next run");
        released(known);
        released(unknown);
    }

    /// A draw whose graphics pipeline is not built within the backstop is skipped (the stand-in,
    /// no hang) and draws after it are not held up; the build goes on, and once it is done a bind
    /// takes the real pipeline. A dispatch is never skipped.
    #[test]
    fn a_slow_graphics_pipeline_skips_its_draw_and_a_later_bind_uses_it() {
        BACKSTOP.with(|b| b.set(Some(Some(std::time::Duration::from_millis(150)))));
        let (standin, compute) = (0x7E57_7101usize, 0x7E57_7102usize);
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let (g, gc) = (gate.clone(), gate.clone());
        let held = move |g: &(Mutex<bool>, Condvar), real: usize| {
            if preparing() {
                return 0;
            }
            let mut open = g.0.lock().unwrap();
            while !*open {
                open = g.1.wait(open).unwrap();
            }
            real
        };
        submit_lazy_described(standin, false, Box::new(move || held(&g, 0x4331)), 0, "VS 1 PS 2".into(), 0, 0);
        submit_lazy_described(compute, true, Box::new(move || held(&gc, 0x4332)), 0, "CS 3".into(), 0, 0);
        let skipped_before = skips();
        let t0 = std::time::Instant::now();
        assert_eq!(resolve_wait(standin), standin, "the draw is skipped: the stand-in");
        assert!(t0.elapsed() < std::time::Duration::from_secs(5), "no hang");
        let t1 = std::time::Instant::now();
        assert_eq!(resolve_wait(standin), standin, "skipped again while it is built");
        assert!(t1.elapsed() < std::time::Duration::from_millis(100), "a later draw does not wait again");
        assert!(skips() >= skipped_before + 2);
        let waiter = std::thread::spawn(move || {
            BACKSTOP.with(|b| b.set(Some(Some(std::time::Duration::from_millis(150)))));
            resolve_wait(compute)
        });
        std::thread::sleep(std::time::Duration::from_millis(400));
        assert!(!waiter.is_finished(), "a dispatch waits past the backstop");
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        assert_eq!(waiter.join().unwrap(), 0x4332);
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait(standin), 0x4331, "built: a bind uses it");
        released(standin);
        released(compute);
    }

    /// A COM object whose Release counts (a pipeline `release` may drop).
    fn counted_object(releases: &'static AtomicUsize) -> usize {
        unsafe extern "system" fn query(_: *mut std::ffi::c_void, _: *const windows::core::GUID, _: *mut *mut std::ffi::c_void) -> windows::core::HRESULT {
            windows::Win32::Foundation::E_NOINTERFACE
        }
        unsafe extern "system" fn add_ref(_: *mut std::ffi::c_void) -> u32 {
            1
        }
        unsafe extern "system" fn release(this: *mut std::ffi::c_void) -> u32 {
            (*(this as *const Counted)).1.fetch_add(1, Ordering::Relaxed);
            0
        }
        #[repr(C)]
        struct Vtable(
            unsafe extern "system" fn(*mut std::ffi::c_void, *const windows::core::GUID, *mut *mut std::ffi::c_void) -> windows::core::HRESULT,
            unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
            unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
        );
        #[repr(C)]
        struct Counted(&'static Vtable, &'static AtomicUsize);
        static VTABLE: Vtable = Vtable(query, add_ref, release);
        Box::into_raw(Box::new(Counted(&VTABLE, releases))) as usize
    }

    /// A pipeline nobody bound is released after its time (`release_idle`) and built again when
    /// a bind wants it; one that was bound is never released.
    #[test]
    fn a_pipeline_nobody_binds_is_released_and_built_again_at_its_first_bind() {
        static RELEASES: AtomicUsize = AtomicUsize::new(0);
        static BUILDS: AtomicUsize = AtomicUsize::new(0);
        let standin = 0x7E57_7004usize;
        let (first, second) = (counted_object(&RELEASES), counted_object(&RELEASES));
        submit_lazy(standin, false, Box::new(move || if BUILDS.fetch_add(1, Ordering::SeqCst) == 0 { first } else { second }), 0);
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let p = entry(standin).unwrap();
        assert_eq!(p.real.load(Ordering::SeqCst), first, "built when made");
        assert!(!release_idle(&p, std::time::Duration::from_secs(3600)), "not idle long enough");
        assert!(release_idle(&p, std::time::Duration::ZERO));
        assert_eq!(RELEASES.load(Ordering::SeqCst), 1, "its real pipeline released");
        assert!(is_pending(standin), "waiting for a bind again");
        assert!(!p.sent.load(Ordering::Acquire), "not counted as building (`counts`)");
        assert_eq!(resolve_wait(standin), second, "a bind builds it again");
        assert_eq!(BUILDS.load(Ordering::SeqCst), 2);
        assert!(!release_idle(&p, std::time::Duration::ZERO), "bound: kept");
        assert_eq!(released(standin), second);
        assert_eq!(RELEASES.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_new_compute_pipeline_is_built_without_a_bind() {
        let compute = 0x7E57_7003;
        submit_lazy(compute, true, Box::new(|| 0x4323), 0);
        let start = std::time::Instant::now();
        while is_pending(compute) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(compute));
        assert_eq!(released(compute), 0x4323);
    }
}

/// The preload workers' scheduling with the real threads: a build put off for memory, and the
/// binds that come for it. (The order of the backlog and the governor's admission are
/// `pso_sched`'s tests.)
#[cfg(test)]
mod ahead_tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn wait_until(what: &str, f: impl Fn() -> bool) {
        let t = Instant::now();
        while !f() {
            assert!(t.elapsed() < Duration::from_secs(10), "{what}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn a_passive_bind_leaves_pending_compute_to_the_first_use() {
        let standin = 0x7E57_D001usize;
        let real = 0x4701usize;
        let (finish, gate) = mpsc::channel();
        let gate = Mutex::new(gate);
        submit_lazy(standin, true, Box::new(move || {
            gate.lock().unwrap().recv().unwrap();
            real
        }), 0);
        let pending = entry(standin).unwrap();
        assert_eq!(resolve_bind(standin, true), standin);
        assert!(!pending.urgent.load(Ordering::Acquire));
        assert!(!pending.bound.load(Ordering::Acquire));

        let used = std::thread::spawn(move || {
            let count = AtomicUsize::new(0);
            (resolve_wait_from(standin, &count), count.load(Ordering::Relaxed))
        });
        wait_until("compute use makes the unfinished build urgent", || pending.urgent.load(Ordering::Acquire));
        assert!(!used.is_finished());
        finish.send(()).unwrap();
        assert_eq!(used.join().unwrap(), (real, 1));
        assert_eq!(resolve_bind(standin, true), real);
        assert_eq!(released(standin), real);
    }

    /// A build with a big driver compile of `dxil_len` bytes, as `real::driver_slot` treats it:
    /// with no memory for it (`NO_ROOM`) a preload worker nothing waits on leaves it; any other
    /// thread would wait for the memory (here: builds it).
    fn big(dxil_len: usize, real: usize, runs: Arc<AtomicUsize>) -> Job {
        Box::new(move || {
            runs.fetch_add(1, Ordering::SeqCst);
            if !room_for(dxil_len) && may_put_off() && !waited_on() {
                put_off(dxil_len);
                return 0;
            }
            real
        })
    }

    fn parked(standin: usize) -> bool {
        let p = entry(standin).unwrap();
        let owner = p.preload.lock().unwrap();
        owner.worker.is_none() && !p.full_started.load(Ordering::Acquire) && !*p.done.lock().unwrap()
    }

    /// A build whose driver compile does not fit the memory is put off: the pipelines made
    /// after it are all built meanwhile (two workers that waited in the governor held a whole
    /// run's builds behind them), it stays waiting to be built – not failed – and is built once it fits.
    #[test]
    fn a_build_put_off_for_memory_holds_nothing_up_and_is_built_when_it_fits() {
        let len = 0x7_0001;
        NO_ROOM.lock().unwrap().push(len);
        let runs = Arc::new(AtomicUsize::new(0));
        let (big_one, small) = (0x7E57_A001usize, 0x7E57_A100usize);
        submit_lazy_described(big_one, true, big(len, 0x4401, runs.clone()), 0, "CS big".into(), super::super::pso_sched::HEAVY_SHADER_BYTES, 0);
        for k in 0..64 {
            submit_lazy(small + k, k % 2 == 0, Box::new(move || 0x4500 + k), 0);
        }
        wait_until("the pipelines behind a build that waits for memory are built", || (0..64).all(|k| !is_pending(small + k)));
        wait_until("the big build was put off", || runs.load(Ordering::SeqCst) >= 1 && parked(big_one));
        std::thread::sleep(Duration::from_millis(500));
        assert!(is_pending(big_one), "not built while it does not fit – and not failed: it still waits to be built");
        assert_eq!(runs.load(Ordering::SeqCst), 1, "and not tried again while it does not fit");
        NO_ROOM.lock().unwrap().retain(|l| *l != len);
        wait_until("built once it fits", || !is_pending(big_one));
        assert_eq!(released(big_one), 0x4401);
        for k in 0..64 {
            assert_eq!(released(small + k), 0x4500 + k);
        }
    }

    /// The groups of the pipelines bound last are kept, the last one first; one bound again and
    /// again is noted once, and a pipeline of no group changes nothing.
    #[test]
    fn the_groups_bound_last_are_kept_the_last_one_first() {
        // (the other tests' pipelines have no group: only this one notes any)
        let (a, b, c) = (0x7E57_0000_0000_00A1u64, 0x7E57_0000_0000_00B2u64, 0x7E57_0000_0000_00C3u64);
        for group in [a, a, 0, a, b, 0, b, a, c, c] {
            note_bound_group(group);
        }
        assert_eq!(recent_groups()[..4], [c, a, b, a]);
        for _ in 0..super::super::pso_sched::RECENT_GROUPS {
            note_bound_group(b);
            note_bound_group(c);
        }
        assert!(recent_groups().iter().all(|g| *g == b || *g == c), "the older ones are gone");
        assert_eq!(recent_groups()[..2], [c, b]);
    }

    /// No more background workers build at once than the pace allows (`PACE_WORKERS`): the
    /// others take nothing until one is done.
    #[test]
    fn the_workers_build_no_more_at_once_than_the_pace_allows() {
        struct Restore;
        impl Drop for Restore {
            fn drop(&mut self) {
                PACE_WORKERS.store(usize::MAX, Ordering::Release);
            }
        }
        let (at_once, most) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let first = 0x7E57_C100usize;
        let _restore = Restore;
        PACE_WORKERS.store(2, Ordering::Release);
        for k in 0..24 {
            let (at_once, most) = (at_once.clone(), most.clone());
            submit_lazy(
                first + k,
                k % 2 == 0,
                Box::new(move || {
                    most.fetch_max(at_once.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(15));
                    at_once.fetch_sub(1, Ordering::SeqCst);
                    0x4600 + k
                }),
                0,
            );
        }
        wait_until("all of them are built, two at a time", || (0..24).all(|k| !is_pending(first + k)));
        assert!((1..=2).contains(&most.load(Ordering::SeqCst)), "{} at once", most.load(Ordering::SeqCst));
        for k in 0..24 {
            assert_eq!(released(first + k), 0x4600 + k);
        }
    }

    /// A bind promotes a build that waits for memory: its build starts at once on a thread of
    /// its own, ahead of everything queued, and waits for the memory itself.
    #[test]
    fn a_bind_takes_a_build_put_off_for_memory() {
        let len = 0x7_0002;
        NO_ROOM.lock().unwrap().push(len);
        let runs = Arc::new(AtomicUsize::new(0));
        let standin = 0x7E57_A002usize;
        submit_lazy_described(standin, true, big(len, 0x4402, runs.clone()), 0, "CS big".into(), 0, 0);
        wait_until("the build was put off", || runs.load(Ordering::SeqCst) >= 1 && parked(standin));
        let urgent_before = URGENT_BUILDS.load(Ordering::SeqCst);
        assert_eq!(resolve_wait(standin), 0x4402, "the bind's own build");
        assert!(URGENT_BUILDS.load(Ordering::SeqCst) > urgent_before, "started by the bind, ahead of the queue");
        assert_eq!(runs.load(Ordering::SeqCst), 2);
        NO_ROOM.lock().unwrap().retain(|l| *l != len);
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(runs.load(Ordering::SeqCst), 2, "the parked build is dropped, not built again");
        assert_eq!(released(standin), 0x4402);
    }

    /// A build a bind waited for and the memory could not hold is not failed for good: that
    /// bind and the ones after it go on with the stand-in without waiting, and a background
    /// worker builds the pipeline once its compile fits.
    #[test]
    fn a_build_refused_for_memory_is_built_when_it_fits_and_its_binds_do_not_wait() {
        let len = 0x7_0004;
        NO_ROOM.lock().unwrap().push(len);
        let standin = 0x7E57_A004usize;
        let runs = Arc::new(AtomicUsize::new(0));
        let ran = runs.clone();
        // (as `real::driver_slot`: put off on a preload worker, refused on a bind's own thread)
        let job = move || {
            ran.fetch_add(1, Ordering::SeqCst);
            if room_for(len) {
                return 0x4404;
            }
            if may_put_off() && !waited_on() {
                put_off(len);
            } else {
                no_memory(len);
            }
            0
        };
        submit_lazy_described(standin, true, Box::new(job), 0, "CS giant".into(), 0, 0);
        wait_until("the build was put off", || runs.load(Ordering::SeqCst) >= 1 && parked(standin));
        assert_eq!(resolve_wait(standin), standin, "refused for memory: the stand-in");
        let t = Instant::now();
        assert_eq!(resolve_wait(standin), standin);
        assert!(t.elapsed() < Duration::from_millis(100), "the binds after it do not wait");
        assert!(is_pending(standin), "not failed: it waits to be built");
        assert!(!compute_bound_pending() || COMPUTE_BOUND_PENDING.load(Ordering::SeqCst) < 64, "not counted as a dispatch waiting");
        NO_ROOM.lock().unwrap().retain(|l| *l != len);
        wait_until("built once it fits", || !is_pending(standin));
        assert_eq!(resolve_wait(standin), 0x4404, "and bound from then on");
        assert_eq!(released(standin), 0x4404);
    }

    /// A bind that comes while a preload worker is putting its build off is not left waiting
    /// for a build nobody runs: the worker builds it itself, now.
    #[test]
    fn a_bind_that_comes_while_its_build_is_put_off_gets_it_built() {
        let len = 0x7_0003;
        let standin = 0x7E57_A003usize;
        let (started_tx, started_rx) = mpsc::channel();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let go_rx = Mutex::new(go_rx);
        let threads = Arc::new(Mutex::new(Vec::new()));
        let ran_on = threads.clone();
        submit_lazy_described(
            standin,
            true,
            Box::new(move || {
                let first = {
                    let mut t = ran_on.lock().unwrap();
                    t.push(std::thread::current().id());
                    t.len() == 1
                };
                if !first {
                    return 0x4403;
                }
                assert!(may_put_off(), "on a preload worker");
                started_tx.send(()).unwrap();
                let _ = go_rx.lock().unwrap().recv();
                put_off(len);
                0
            }),
            0,
            "CS big".into(),
            0,
            0,
        );
        started_rx.recv_timeout(Duration::from_secs(10)).expect("a preload worker builds it");
        let waiter = std::thread::spawn(move || resolve_wait(standin));
        wait_until("the bind found the worker", || entry(standin).unwrap().preload.lock().unwrap().demanded);
        go_tx.send(()).unwrap();
        assert_eq!(waiter.join().unwrap(), 0x4403);
        let threads = threads.lock().unwrap();
        assert_eq!(threads.len(), 2, "built once more, and only once");
        assert_eq!(threads[0], threads[1], "by the worker that had it");
        assert_eq!(released(standin), 0x4403);
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// The pipeline to use for `pso` now: the real one once built, else `pso` itself.
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_bind();
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue(p);
            }
            pso
        }
        r => r,
    }
}

/// A bind waits until its pipeline is built, graphics and compute alike: a stand-in bound
/// instead skips the draw or dispatch, and what it would have written stays unwritten (with
/// stand-ins bound for hundreds of pipelines a retail game crashed or hung the GPU while it
/// loaded). Waits this long or longer are logged.
const LONG_WAIT: std::time::Duration = std::time::Duration::from_secs(8);

/// the longest a bind waited so far (ms)
static WORST_WAIT: AtomicUsize = AtomicUsize::new(0);

/// `resolve`, waiting for a pipeline still being built: its build starts now on a thread of its
/// own (`build_tiered`; a queue of a title's hundreds of pipelines must not hold up the one it
/// binds now), and the bind waits until it is built (`LONG_WAIT`). A build that failed leaves
/// the stand-in (its failure is logged where it failed).
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_from(pso, &USE_WAITS)
}

pub(super) fn resolve_wait_at_bind(pso: usize) -> usize {
    resolve_wait_from(pso, &BIND_WAITS)
}

fn resolve_wait_from(pso: usize, site_waits: &AtomicUsize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_bind();
    if p.starved.load(Ordering::Acquire) {
        // refused for memory once: no bind waits for it again; a background worker builds it
        return match p.real.load(Ordering::Acquire) {
            0 => pso,
            r => r,
        };
    }
    if !*p.done.lock().unwrap() {
        // (the title's thread is held from here until the pipeline is built or the bind gives up)
        let _held = Held::new();
        if !p.urgent.swap(true, Ordering::AcqRel) {
            if p.compute && !*p.done.lock().unwrap() && !p.counted.swap(true, Ordering::AcqRel) {
                COMPUTE_BOUND_PENDING.fetch_add(1, Ordering::AcqRel);
            }
            if !p.promote_preload() {
                enqueue_urgent(p.clone());
            }
        }
        let t0 = std::time::Instant::now();
        let frame_waited = frame_waited();
        let over_budget = !p.compute && frame_budget().is_some_and(|b| frame_waited >= b);
        let limit = if (p.skipped.load(Ordering::Acquire) || over_budget) && !p.compute { Some(std::time::Duration::ZERO) } else { backstop() };
        let (mut done, mut warned) = (p.done.lock().unwrap(), false);
        // (`starved`: refused for memory while this bind waited – it goes on with the stand-in)
        while !*done && !p.starved.load(Ordering::Acquire) {
            let Some(left) = limit.map(|l| l.saturating_sub(t0.elapsed())) else {
                done = p.cv.wait(done).unwrap();
                continue;
            };
            if !left.is_zero() {
                done = p.cv.wait_timeout(done, left).unwrap().0;
            } else if !p.compute {
                drop(done);
                add_frame_wait(t0.elapsed());
                let ms = t0.elapsed().as_millis() as usize;
                site_waits.fetch_add(1, Ordering::Relaxed);
                WAIT_MS.fetch_add(ms, Ordering::Relaxed);
                SKIPS.fetch_add(1, Ordering::Relaxed);
                if over_budget && !p.skipped.load(Ordering::Acquire) {
                    let n = BUDGET_SKIPS.fetch_add(1, Ordering::Relaxed) + 1;
                    if n <= 8 || n.is_power_of_two() {
                        crate::xlog_warn!(
                            "D3D12.X: a draw skipped at once: graphics pipeline {pso:#x} ({}) not built, and this thread's binds have waited {} ms for pipelines since the last present (XCL_FRAME_STALL_BUDGET_MS) (#{n})",
                            if p.what.is_empty() { "shaders unknown" } else { &p.what },
                            frame_waited.as_millis()
                        );
                    }
                    p.skipped.store(true, Ordering::Release);
                    if p.key != 0 {
                        used::note_skipped(p.key);
                    }
                    return pso;
                }
                // the first skip of a pipeline is logged (and kept for the next warm-up); later
                // draws with it are skipped without a wait until it is built, and counted
                if !p.skipped.swap(true, Ordering::AcqRel) {
                    if p.key != 0 {
                        used::note_skipped(p.key);
                    }
                    crate::xlog_warn!(
                        "D3D12.X: a draw skipped: graphics pipeline {pso:#x} ({}) not built after {ms} ms of waiting (XCL_BIND_BACKSTOP_MS) – its build goes on ahead of the others; draws with it are skipped until it is built, then use it",
                        if p.what.is_empty() { "shaders unknown" } else { &p.what }
                    );
                }
                return pso;
            } else {
                if !std::mem::replace(&mut warned, true) {
                    crate::xlog_warn!(
                        "D3D12.X: a dispatch has waited {} ms for compute pipeline {pso:#x} ({}) – still waiting: a skipped dispatch could leave the title's state wrong",
                        t0.elapsed().as_millis(),
                        if p.what.is_empty() { "shaders unknown" } else { &p.what }
                    );
                }
                done = p.cv.wait(done).unwrap();
            }
        }
        drop(done);
        let waited = t0.elapsed();
        add_frame_wait(waited);
        let kind = if p.compute { "compute" } else { "graphics" };
        site_waits.fetch_add(1, Ordering::Relaxed);
        if waited >= LONG_WAIT {
            crate::xlog_warn!("D3D12.X: a bind waited {:.1} s for {kind} pipeline {pso:#x} (binds wait for every pipeline)", waited.as_secs_f32());
        }
        let ms = waited.as_millis() as usize;
        WAIT_MS.fetch_add(ms, Ordering::Relaxed);
        if ms >= 250 {
            crate::xlog_info!(
                "D3D12.X: a bind waited {ms} ms for {kind} pipeline {pso:#x}{}; {}",
                if p.rebuild.load(Ordering::Acquire) { " (a rebuild of one released unbound)" } else { "" },
                p.note.lock().unwrap().as_deref().unwrap_or("no build details")
            );
        }
        if WORST_WAIT.fetch_max(ms, Ordering::Relaxed) < ms && ms >= 250 {
            crate::xlog_info!("D3D12.X: longest bind wait so far: {ms} ms ({kind} pipeline {pso:#x})");
        }
    }
    match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    }
}

/// binds that waited for a pipeline still being built
static BIND_WAITS: AtomicUsize = AtomicUsize::new(0);
static USE_WAITS: AtomicUsize = AtomicUsize::new(0);

/// (bind-time waits, use-time waits, the longest wait in ms) so far
pub fn waits() -> (usize, usize, usize) {
    (BIND_WAITS.load(Ordering::Relaxed), USE_WAITS.load(Ordering::Relaxed), WORST_WAIT.load(Ordering::Relaxed))
}

pub(super) fn passive_bind() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| lazy() && std::env::var("XCL_PSO_PASSIVE_BIND").is_ok_and(|v| v.trim() == "1"))
}

/// Resolve without enqueuing or pinning a pipeline for a bookkeeping-only lookup.
pub(super) fn resolve_if_ready(pso: usize) -> usize {
    if !passive_bind() {
        return resolve(pso);
    }
    entry(pso).map_or(pso, |p| match p.real.load(Ordering::Acquire) {
        0 => pso,
        real => real,
    })
}

/// Opt-in passive binds keep the stand-in until an actual draw or dispatch asks for the PSO.
pub fn resolve_for_bind(pso: usize) -> usize {
    resolve_bind(pso, passive_bind())
}

fn resolve_bind(pso: usize, passive: bool) -> usize {
    if !passive {
        return resolve_wait_at_bind(pso);
    }
    let Some(p) = entry(pso) else { return pso };
    if p.real.load(Ordering::Acquire) != 0 {
        p.bound.store(true, Ordering::SeqCst);
        match p.real.load(Ordering::SeqCst) {
            0 => (),
            real => return real,
        }
    }
    let seq = p.preload_seq.load(Ordering::Relaxed);
    if seq != 0 {
        if let Some(Some(q)) = Q.get() {
            q.list.lock().unwrap().promote(seq);
            q.cv.notify_one();
        }
    }
    pso
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    let done = p.done.lock().unwrap();
    // a build still running (or an optimized one to come) releases what it makes
    p.orphan.store(true, Ordering::Release);
    if *done {
        p.real.swap(0, Ordering::AcqRel)
    } else {
        0
    }
}

/// compute pipelines a bind is waiting for (bound, not built yet)
static COMPUTE_BOUND_PENDING: AtomicUsize = AtomicUsize::new(0);

/// Whether a dispatch is waiting for its compute pipeline now.
pub fn compute_bound_pending() -> bool {
    COMPUTE_BOUND_PENDING.load(Ordering::Acquire) > 0
}

/// The shaders a console pipeline is made of, when known.
pub fn what(pso: usize) -> Option<String> {
    entry(pso).map(|p| p.what.clone()).filter(|w| !w.is_empty())
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// (live real pipelines, released as nobody bound them, built again at a bind) so far
pub fn live() -> (usize, usize, usize) {
    let live = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| p.real.load(Ordering::Acquire) != 0).count());
    (live, IDLE_RELEASED.load(Ordering::Relaxed), REBUILT.load(Ordering::Relaxed))
}

/// (built in the background, failed, still building): lazy pipelines count as building once
/// they were wanted (`resolve`), not while they wait for their first use.
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| p.sent.load(Ordering::Acquire) && !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Once a thread's binds have waited the frame budget, its draws with pipelines still
    /// being built are skipped at once, while its dispatches still wait; a new frame starts the
    /// budget again.
    #[test]
    fn draws_past_the_frame_wait_budget_are_skipped_at_once_and_dispatches_still_wait() {
        let (first, second, compute) = (0x7E57_7211usize, 0x7E57_7212usize, 0x7E57_7213usize);
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let held = |g: Arc<(Mutex<bool>, Condvar)>, real: usize| -> Job {
            Box::new(move || {
                if preparing() {
                    return 0;
                }
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                real
            })
        };
        submit_lazy(first, false, held(gate.clone(), 0x1251), 0);
        submit_lazy(second, false, held(gate.clone(), 0x1252), 0);
        submit_lazy(compute, true, held(gate.clone(), 0x1253), 0);
        BACKSTOP.with(|b| b.set(Some(Some(Duration::from_millis(300)))));
        FRAME_BUDGET.with(|b| b.set(Some(Some(Duration::from_millis(200)))));
        FRAME_WAIT.with(|w| w.set((u64::MAX, Duration::ZERO)));

        assert_eq!(resolve_wait(first), first, "skipped at the backstop");
        assert!(frame_waited() >= Duration::from_millis(300));
        let (skips, t0) = (budget_skips(), Instant::now());
        assert_eq!(resolve_wait(second), second);
        assert!(t0.elapsed() < Duration::from_millis(150), "skipped at once: {:?}", t0.elapsed());
        assert_eq!(budget_skips(), skips + 1);

        let opener = {
            let g = gate.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(500));
                *g.0.lock().unwrap() = true;
                g.1.notify_all();
            })
        };
        let t0 = Instant::now();
        assert_eq!(resolve_wait(compute), 0x1253, "a dispatch waits past the budget and the backstop");
        assert!(t0.elapsed() >= Duration::from_millis(400));
        opener.join().unwrap();

        FRAME_WAIT.with(|w| w.set((super::super::present::presents().wrapping_add(1), Duration::from_secs(5))));
        assert_eq!(frame_waited(), Duration::ZERO, "a new frame starts the budget again");
        for p in [first, second, compute] {
            released(p);
        }
    }

    #[test]
    fn build_queue_prefers_graphics_and_keeps_order_within_each_class() {
        let pending = |compute| Arc::new(Pending {
            job: Mutex::new(None),
            sent: Default::default(),
            preload: Default::default(),
            real: Default::default(),
            done: Default::default(),
            cv: Default::default(),
            compute,
            key: 0,
            used: Default::default(),
            orphan: Default::default(),
            full_started: Default::default(),
            tiered_started: Default::default(),
            real_fast: Default::default(),
            urgent: Default::default(),
            counted: Default::default(),
            retired: Default::default(),
            bound: Default::default(),
            built_at: Default::default(),
            released_idle: Default::default(),
            rebuild: Default::default(),
            note: Default::default(),
            what: String::new(),
            skipped: Default::default(),
            starved: Default::default(),
            group: 0,
            preload_seq: Default::default(),
        });
        let (c1, c2, g1, g2) = (pending(true), pending(true), pending(false), pending(false));
        let mut queue = BuildQueue::default();
        for p in [&c1, &g1, &c2, &g2] {
            queue.push(Arc::clone(p));
        }
        for expected in [&g1, &g2, &c1, &c2] {
            assert!(Arc::ptr_eq(&queue.pop().unwrap(), expected));
        }
        assert!(queue.pop().is_none());
    }

    /// A lazy pipeline's job runs once to prepare its shaders, before anything binds it, and
    /// once more – for the driver – when it is first bound.
    #[test]
    fn lazy_pipelines_are_prepared_before_their_first_use() {
        // (only without the background workers, `XCL_PSO_PRELOAD=0`: with them it is built when
        // made, `a_pipeline_is_built_when_made_and_its_first_use_kept`)
        if prepare_threads() == 0 || preload().is_some() {
            return;
        }
        let (standin, real) = (0x7300_0010usize, 0x7300_0020usize);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let seen = calls.clone();
        submit_lazy(
            standin,
            false,
            Box::new(move || {
                seen.lock().unwrap().push(preparing());
                if preparing() { 0 } else { real }
            }),
            0,
        );
        let t0 = Instant::now();
        while calls.lock().unwrap().is_empty() && t0.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(*calls.lock().unwrap(), [true], "prepared in the background, the driver left out");
        assert!(is_pending(standin), "preparing does not build the pipeline");
        assert!(!entry(standin).unwrap().sent.load(Ordering::Acquire), "nor counted as building (`counts`)");
        assert_eq!(resolve_wait(standin), real);
        assert_eq!(*calls.lock().unwrap(), [true, false]);
        assert!(!preparing());
        assert_eq!(released(standin), real);
    }

    /// `XCL_BENCH_HLSL=<folder of dumped HLSL, *_{cs,ps,vs}.fssm*.hlsl>`: lazy pipelines whose
    /// builds compile real programs with DXC (into a fresh disk cache), all bound one after
    /// another by one thread right after they were made – a title's loading at its worst.
    /// Run once with `XCL_PREPARE_PSO=0` (each bind compiles its own) and once without.
    #[test]
    #[ignore]
    fn lazy_pipelines_bound_one_after_another() {
        let Some(dir) = std::env::var_os("XCL_BENCH_HLSL") else { return };
        let n: usize = std::env::var("XCL_BENCH_N").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
        let max: u64 = std::env::var("XCL_BENCH_MAX_KB").ok().and_then(|v| v.parse().ok()).unwrap_or(1024) << 10;
        let mut files: Vec<(u64, std::path::PathBuf, &'static str)> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let profile = [("_cs.fssm", "cs_6_6"), ("_ps.fssm", "ps_6_6"), ("_vs.fssm", "vs_6_6")].iter().find(|(s, _)| name.contains(s) && name.ends_with(".hlsl"))?.1;
                let len = e.metadata().ok()?.len();
                (len <= max).then(|| (len, e.path(), profile))
            })
            .collect();
        files.sort();
        let take = n.min(files.len());
        let cache = std::env::temp_dir().join(format!("xcl-lazy-bench-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cache);
        let mut standins = Vec::new();
        let t0 = Instant::now();
        for k in 0..take {
            let (_, path, profile) = &files[k * files.len() / take];
            let (src, profile, cache) = (std::fs::read_to_string(path).unwrap(), *profile, cache.clone());
            let (standin, real) = (0x7400_0000 + 16 * k, 0x7500_0000 + 16 * k);
            submit_lazy(standin, false, Box::new(move || crate::gpu::recomp::dxc::compile_cached_in(&cache, &src, "main", profile).map_or(0, |_| if preparing() { 0 } else { real })), 0);
            standins.push(standin);
        }
        let built = standins.iter().filter(|&&s| resolve_wait(s) != s).count();
        let secs = t0.elapsed().as_secs_f64();
        let (_, compiles) = crate::gpu::recomp::dxc::compile_counts();
        println!("{take} lazy pipelines ({built} built, {compiles} DXC compiles), {} prepare threads: all bound in {secs:.1} s", prepare_threads());
        for &s in &standins {
            released(s);
        }
        let _ = std::fs::remove_dir_all(&cache);
    }

    /// A build that panics is a failed build: whoever waits for it is woken, the worker lives.
    #[test]
    fn a_panicking_build_fails_instead_of_hanging() {
        let (standin, again) = (0x7300_0110usize, 0x7300_0120usize);
        submit(standin, true, Box::new(|| panic!("synthetic build failure")));
        assert_eq!(resolve_wait(standin), standin);
        submit(again, true, Box::new(|| 0x7300_0130));
        assert_eq!(resolve_wait(again), 0x7300_0130);
        assert_eq!(released(standin), 0);
        assert_eq!(released(again), 0x7300_0130);
    }
}
