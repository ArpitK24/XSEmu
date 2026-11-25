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
//! the title makes it (`preload`, governed like every big driver compile), and a bind waits for
//! it, its build going ahead of all the others. One nobody binds is released again after a while
//! (`idle_limit`) and built again at its first bind. With `XCL_PSO_PRELOAD=0` it is built
//! when first bound, its shaders prepared before that on background threads
//! (`prepare_threads`): translated and compiled into the disk caches with the driver left out
//! (`preparing`). Without either the title's binding threads did every translation and DXC
//! compile themselves, one after another: a retail game showed one compile running on a
//! 24-thread PC while thousands of its pipelines waited. `XCL_PREPARE_PSO=0` turns that off.

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock, Weak};

/// Refcounted priority inheritance for a `THREAD_PRIORITY_LOWEST` preload worker: while a guard
/// lives (a bind waits for the pipeline the worker builds), its OS thread runs at normal
/// priority; the last guard dropped restores `LOWEST`.
struct BoostToken {
    tid: u32,
    count: Mutex<usize>,
}

struct BoostGuard(Arc<BoostToken>);

/// Whether preload workers run at `THREAD_PRIORITY_LOWEST` and are raised again on demand. Under
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
}

impl Drop for BoostGuard {
    fn drop(&mut self) {
        let mut c = self.0.count.lock().unwrap();
        *c = c.saturating_sub(1);
        if *c == 0 {
            set_thread_priority_by_id(self.0.tid, windows_sys::Win32::System::Threading::THREAD_PRIORITY_LOWEST);
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
        if self.released_idle.swap(false, Ordering::AcqRel) {
            REBUILT.fetch_add(1, Ordering::Relaxed);
            self.rebuild.store(true, Ordering::Release);
        }
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
        }
    }
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

/// The thread that releases pipelines nothing bound (`idle_limit`), every 5 s; it also drops the
/// builds of bound ones (never released, never built again).
fn start_idle_release() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    let Some(limit) = idle_limit() else { return };
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-pso-idle".into()).spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let all: Vec<Arc<Pending>> = PENDING.lock().unwrap().as_ref().map_or_else(Vec::new, |m| m.values().cloned().collect());
            let mut released = 0;
            for p in &all {
                if p.bound.load(Ordering::SeqCst) {
                    if *p.done.lock().unwrap() {
                        *p.job.lock().unwrap_or_else(|e| e.into_inner()) = None;
                    }
                } else if release_idle(p, limit) {
                    released += 1;
                }
            }
            if released > 0 {
                let (live, _, _) = live();
                crate::xlog_info!("D3D12.X: {released} pipelines nobody bound in {} s released ({live} live; they are built again at their first bind)", limit.as_secs());
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
    let (real, fast) = (run(&*job), false);
    if !fast {
        p.full_started.store(true, Ordering::Release);
    }
    complete(p, real, fast);
    if fast && !p.full_started.load(Ordering::Acquire) {
        enqueue_upgrade(p.clone());
    }
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
    let now = (URGENT_BUILDS.load(Ordering::Relaxed), WAITS.load(Ordering::Relaxed), WAIT_MS.load(Ordering::Relaxed), SKIPS.load(Ordering::Relaxed));
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
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.urgent.load(Ordering::Acquire)))
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
    submit_as(standin, compute, job, true, 0, String::new());
}

/// `submit` with what the pipeline is made of (`what`: its shaders, for the log).
pub fn submit_described(standin: usize, compute: bool, job: Job, what: String) {
    submit_as(standin, compute, job, true, 0, what);
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

/// `submit`, built by the background workers (`preload`) at the lowest priority: first the ones of
/// shaders an earlier run used (`key`, 0 = none; `used`), then compute, then graphics. A bind
/// sends it ahead of all of them. (`XCL_PSO_PRELOAD=0`: built at its first `resolve`, its shaders
/// prepared before.)
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    submit_as(standin, compute, job, false, key, String::new());
}

/// `submit_lazy` with what the pipeline is made of (`what`: its shaders, for the log).
pub fn submit_lazy_described(standin: usize, compute: bool, job: Job, key: u64, what: String) {
    submit_as(standin, compute, job, false, key, what);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool, key: u64, what: String) {
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
            let mut jobs = q.list.lock().unwrap();
            if used::known(key) {
                jobs.known.push_back(Arc::downgrade(&p));
            } else if compute {
                jobs.compute.push_back(Arc::downgrade(&p));
            } else {
                jobs.graphics.push_back(Arc::downgrade(&p));
            }
            drop(jobs);
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

/// The workers that build every pipeline the title makes ahead of its first use, those of an
/// earlier run first (`XCL_PSO_PRELOAD=0`: off). Apart from the ones a bind sends to (`queue`)
/// and at the lowest priority: a pipeline that is bound is not held up by the ones made ahead
/// (whoever takes `job` first builds it), and the title's own threads are not either. Their big
/// driver compiles are governed (`real::DRIVER_COMPILES`).
struct Preload {
    list: Mutex<PreloadJobs>,
    cv: Condvar,
}

#[derive(Default)]
struct PreloadJobs {
    known: VecDeque<Weak<Pending>>,
    compute: VecDeque<Weak<Pending>>,
    graphics: VecDeque<Weak<Pending>>,
}

fn preload() -> Option<&'static Preload> {
    static Q: OnceLock<Option<Preload>> = OnceLock::new();
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
        let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        for k in 0..n {
            let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
                if can_lower_preload_priority() {
                    unsafe {
                        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                    }
                }
                let token = BoostToken::for_current_thread();
                static DONE: AtomicUsize = AtomicUsize::new(0);
                loop {
                    let p = {
                        let mut g = q.list.lock().unwrap();
                        loop {
                            if let Some(p) = g.known.pop_front().or_else(|| g.compute.pop_front()).or_else(|| g.graphics.pop_front()) {
                                break p;
                            }
                            g = q.cv.wait(g).unwrap();
                        }
                    };
                    let _task = crate::progress::PIPELINES.take();
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
                    build_full(&p);
                    BUILDING.with(|b| *b.borrow_mut() = None);
                    let n = DONE.fetch_add(1, Ordering::Relaxed) + 1;
                    if n.is_power_of_two() {
                        crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
                    }
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

    #[test]
    fn boost_token_refcounts_nested_guards() {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_NORMAL};
        let check_os = can_lower_preload_priority();
        let (tok_tx, tok_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let h = std::thread::spawn(move || {
            if check_os {
                unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) };
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
            assert_eq!(thread_priority_of(tok.tid), THREAD_PRIORITY_LOWEST, "restored once the last guard drops");
        }
        let _ = done_tx.send(());
        h.join().unwrap();
    }

    /// A bind that waits for a pipeline a preload worker is building raises that worker to normal
    /// priority until it is done (and starts no second build); a pipeline a bind waited for
    /// before a preload worker took it is skipped by that worker.
    #[test]
    fn resolve_wait_boosts_the_preload_worker_building_it_and_restores_it() {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadPriority, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_NORMAL};
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
        assert_eq!(before, if check_os { THREAD_PRIORITY_LOWEST } else { THREAD_PRIORITY_NORMAL });
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
        while tok.is_boosted() || (check_os && thread_priority_of(tid) != THREAD_PRIORITY_LOWEST) {
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
        submit_lazy_described(standin, false, Box::new(move || held(&g, 0x4331)), 0, "VS 1 PS 2".into());
        submit_lazy_described(compute, true, Box::new(move || held(&gc, 0x4332)), 0, "CS 3".into());
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
    let Some(p) = entry(pso) else { return pso };
    p.note_bind();
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
        while !*done {
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
                WAITS.fetch_add(1, Ordering::Relaxed);
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
        WAITS.fetch_add(1, Ordering::Relaxed);
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
static WAITS: AtomicUsize = AtomicUsize::new(0);

/// (binds that waited, the longest wait in ms) so far
pub fn waits() -> (usize, usize) {
    (WAITS.load(Ordering::Relaxed), WORST_WAIT.load(Ordering::Relaxed))
}

/// For binding: every pipeline is waited for (`resolve_wait`).
pub fn resolve_for_bind(pso: usize) -> usize {
    resolve_wait(pso)
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
