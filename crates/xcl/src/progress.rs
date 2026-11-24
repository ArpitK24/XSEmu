//! Progress of the shader work a title waits for, shown in the XCL window so a cold start is
//! not minutes of black screen: the window's title says what is being built and how far it got
//! (`Status::title`), and a thin bar runs along the bottom edge of the frame while anything is
//! pending (`bar`, drawn by `present`). `xcl warm-up` prints the same counts on its console.
//!
//! The pipeline code counts its work here: pipelines queued for the background workers
//! (`PIPELINES`), console shader translations (`TRANSLATIONS`), driver compiles of big programs
//! (`DRIVER`, the giants among them in `LARGE_DRIVER`) and the warm-up's done/total; DXC compiles
//! come from `dxc::compile_counts` (counting them in `gpu/recomp` would re-key the translated
//! shader cache, see build.rs). `XCL_PROGRESS=0` turns the display off; counting stays on.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Work of one kind: waiting or running (`pending`) and finished.
pub struct Counter {
    pending: AtomicUsize,
    finished: AtomicUsize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub pending: usize,
    pub finished: usize,
}

impl Counter {
    pub const fn new() -> Counter {
        Counter { pending: AtomicUsize::new(0), finished: AtomicUsize::new(0) }
    }

    /// One more is waiting (a queue took it); `take` it when its handling starts.
    pub fn queue(&'static self) {
        self.pending.fetch_add(1, Ordering::Relaxed);
    }

    /// The handling of one `queue`d: finished when the task is dropped.
    pub fn take(&'static self) -> Task {
        Task(self)
    }

    /// One more, running until the task is dropped.
    pub fn begin(&'static self) -> Task {
        self.queue();
        self.take()
    }

    pub fn counts(&self) -> Counts {
        Counts { pending: self.pending.load(Ordering::Relaxed), finished: self.finished.load(Ordering::Relaxed) }
    }
}

/// One piece of work of a `Counter`: finished when dropped.
pub struct Task(&'static Counter);

impl Drop for Task {
    fn drop(&mut self) {
        self.0.finished.fetch_add(1, Ordering::Relaxed);
        self.0.pending.fetch_sub(1, Ordering::Relaxed);
    }
}

/// pipelines queued for the background workers (`pso_async`: prepared, built ahead, built)
pub static PIPELINES: Counter = Counter::new();
/// console shaders being translated for a pipeline (cache lookups included)
pub static TRANSLATIONS: Counter = Counter::new();
/// driver compiles of big programs running (`real::DRIVER_COMPILES`)
pub static DRIVER: Counter = Counter::new();
/// … of them the giants (1 MB of DXIL and more, ~1.4 GB of RAM each in an NVIDIA driver)
pub static LARGE_DRIVER: Counter = Counter::new();
static WARM_UP: [AtomicUsize; 2] = [const { AtomicUsize::new(0) }; 2];

/// `xcl warm-up` built `done` of its `total` driver blobs.
pub fn warm_up(done: usize, total: usize) {
    WARM_UP[1].store(total, Ordering::Relaxed);
    WARM_UP[0].store(done, Ordering::Relaxed);
}

/// The display is on (`XCL_PROGRESS`, default on, `0` turns it off).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| on(std::env::var("XCL_PROGRESS").ok().as_deref()))
}

fn on(setting: Option<&str>) -> bool {
    setting.map_or(true, |v| v.trim() != "0")
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    pub warm_up: Counts,
    pub pipelines: Counts,
    pub dxc: Counts,
    pub translations: Counts,
    pub driver: Counts,
    pub large: usize,
}

pub fn snapshot() -> Snapshot {
    let (running, compiled) = crate::gpu::recomp::dxc::compile_counts();
    let (done, total) = (WARM_UP[0].load(Ordering::Relaxed), WARM_UP[1].load(Ordering::Relaxed));
    Snapshot {
        warm_up: Counts { pending: total.saturating_sub(done), finished: done },
        pipelines: PIPELINES.counts(),
        dxc: Counts { pending: running, finished: compiled },
        translations: TRANSLATIONS.counts(),
        driver: DRIVER.counts(),
        large: LARGE_DRIVER.counts().pending,
    }
}

/// What the window shows: the first kind of work with something pending, in this order.
const KINDS: [(&str, fn(&Snapshot) -> Counts); 5] = [
    ("warming the driver cache", |s| s.warm_up),
    ("building shaders", |s| s.pipelines),
    ("compiling shaders", |s| s.dxc),
    ("translating shaders", |s| s.translations),
    ("building driver pipelines", |s| s.driver),
];
/// `KINDS`' DXC entry
const DXC_KIND: usize = 2;

/// A kind's work counts from its last idle moment: once nothing of it was pending for `GRACE`,
/// the next work starts at 0 of its own total (a title thread's compiles come one by one, so a
/// short gap does not start over).
const GRACE: Duration = Duration::from_secs(2);

#[derive(Default)]
pub struct Tracker {
    /// finished when the batch started
    base: [usize; KINDS.len()],
    idle_since: [Option<Instant>; KINDS.len()],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Status {
    pub label: &'static str,
    pub done: usize,
    pub total: usize,
    /// DXC compiles running besides (0 when DXC is what `label` counts)
    pub dxc: usize,
    pub large: usize,
}

impl Tracker {
    pub fn update(&mut self, s: &Snapshot, now: Instant) -> Option<Status> {
        let mut shown = None;
        for (k, &(label, counts)) in KINDS.iter().enumerate() {
            let c = counts(s);
            if c.pending > 0 {
                self.idle_since[k] = None;
            } else if now.duration_since(*self.idle_since[k].get_or_insert(now)) >= GRACE {
                self.base[k] = c.finished;
            }
            if shown.is_none() && c.pending > 0 {
                let done = c.finished.saturating_sub(self.base[k]);
                shown = Some(Status { label, done, total: done + c.pending, dxc: if k == DXC_KIND { 0 } else { s.dxc.pending }, large: s.large });
            }
        }
        shown
    }
}

/// What to show now, `None` while nothing is pending or with the display off.
pub fn current() -> Option<Status> {
    static TRACKER: Mutex<Option<Tracker>> = Mutex::new(None);
    if !enabled() {
        return None;
    }
    let s = snapshot();
    TRACKER.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Tracker::default).update(&s, Instant::now())
}

impl Status {
    pub fn fraction(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.done as f32 / self.total as f32 }
    }

    /// "xcl: building shaders 1234/2600 (47%), 12 DXC running, 3 large in flight"
    pub fn title(&self) -> String {
        let mut t = format!("xcl: {} {}/{} ({}%)", self.label, self.done, self.total, self.done * 100 / self.total.max(1));
        if self.dxc > 0 {
            t += &format!(", {} DXC running", self.dxc);
        }
        if self.large > 0 {
            t += &format!(", {} large in flight", self.large);
        }
        t
    }
}

/// (left, top, right, bottom)
pub type Rect = [i32; 4];

/// The bar along the bottom edge of a `w`×`h` frame: the done part, then the rest of its track,
/// each with its colour (linear RGBA). A few pixels high: 1/180 of the height, 3 to 8.
pub fn bar(w: u32, h: u32, fraction: f32) -> [(Rect, [f32; 4]); 2] {
    let (w, h) = (w as i32, h as i32);
    let height = (h / 180).clamp(3, 8).min(h);
    let split = ((w as f32 * fraction.clamp(0.0, 1.0)).round() as i32).clamp(0, w);
    [([0, h - height, split, h], [1.0, 0.72, 0.18, 1.0]), ([split, h - height, w, h], [0.06, 0.06, 0.06, 1.0])]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_task_is_pending_until_dropped() {
        static C: Counter = Counter::new();
        C.queue();
        C.queue();
        assert_eq!(C.counts(), Counts { pending: 2, finished: 0 });
        drop(C.take());
        assert_eq!(C.counts(), Counts { pending: 1, finished: 1 });
        let running = C.begin();
        assert_eq!(C.counts(), Counts { pending: 2, finished: 1 });
        drop(running);
        drop(C.take());
        assert_eq!(C.counts(), Counts { pending: 0, finished: 3 });
    }

    #[test]
    fn the_setting_turns_the_display_off_only_with_0() {
        assert!(on(None));
        assert!(on(Some("1")));
        assert!(!on(Some("0")));
        assert!(!on(Some(" 0 ")));
    }

    #[test]
    fn the_title_says_what_is_built_and_how_far() {
        let s = Status { label: "building shaders", done: 1234, total: 2600, dxc: 0, large: 3 };
        assert_eq!(s.title(), "xcl: building shaders 1234/2600 (47%), 3 large in flight");
        let s = Status { label: "compiling shaders", done: 0, total: 5, dxc: 0, large: 0 };
        assert_eq!(s.title(), "xcl: compiling shaders 0/5 (0%)");
        let s = Status { label: "building shaders", done: 10, total: 20, dxc: 12, large: 2 };
        assert_eq!(s.title(), "xcl: building shaders 10/20 (50%), 12 DXC running, 2 large in flight");
        assert_eq!(s.fraction(), 0.5);
    }

    #[test]
    fn work_counts_from_its_last_idle_moment() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        let mut t = Tracker::default();
        let mut s = Snapshot { pipelines: Counts { pending: 0, finished: 500 }, ..Default::default() };
        assert_eq!(t.update(&s, at(0)), None, "nothing pending: nothing shown");
        t.update(&s, at(3));
        s.pipelines = Counts { pending: 30, finished: 510 };
        let shown = t.update(&s, at(4)).unwrap();
        assert_eq!((shown.label, shown.done, shown.total), ("building shaders", 10, 40));
        // a short gap keeps counting the same batch
        s.pipelines = Counts { pending: 0, finished: 540 };
        assert_eq!(t.update(&s, at(5)), None);
        s.pipelines = Counts { pending: 4, finished: 541 };
        assert_eq!(t.update(&s, at(6)).map(|s| (s.done, s.total)), Some((41, 45)));
        // idle past the grace: the next work starts over
        s.pipelines = Counts { pending: 0, finished: 545 };
        t.update(&s, at(7));
        t.update(&s, at(10));
        s.pipelines = Counts { pending: 8, finished: 547 };
        assert_eq!(t.update(&s, at(11)).map(|s| (s.done, s.total)), Some((2, 10)));
    }

    #[test]
    fn the_warm_up_comes_first_and_dxc_shows_beside_other_work() {
        let mut t = Tracker::default();
        let s = Snapshot {
            warm_up: Counts { pending: 60, finished: 40 },
            pipelines: Counts { pending: 3, finished: 0 },
            dxc: Counts { pending: 2, finished: 9 },
            large: 3,
            ..Default::default()
        };
        assert_eq!(t.update(&s, Instant::now()), Some(Status { label: "warming the driver cache", done: 40, total: 100, dxc: 2, large: 3 }));
        let s = Snapshot { dxc: Counts { pending: 2, finished: 9 }, ..Default::default() };
        let shown = Tracker::default().update(&s, Instant::now()).unwrap();
        assert_eq!((shown.label, shown.done, shown.total, shown.dxc), ("compiling shaders", 9, 11, 0));
    }

    #[test]
    fn the_bar_runs_along_the_bottom_edge() {
        let [(done, _), (rest, _)] = bar(1920, 1080, 0.25);
        assert_eq!(done, [0, 1074, 480, 1080]);
        assert_eq!(rest, [480, 1074, 1920, 1080]);
        assert_eq!(bar(640, 360, 2.0)[0].0, [0, 357, 640, 360], "3 px at least, full at most");
        assert_eq!(bar(3840, 2160, 0.0)[0].0[2], 0);
        assert_eq!(bar(3840, 2160, 0.0)[1].0[1], 2160 - 8, "8 px at most");
    }
}
