//! What the console shows of a running title besides milestones, warnings and errors (with
//! `log::verbose` off): every 10 s one status line (frames, pipelines, memory) and, while work is
//! stuck, one line with how much is stuck and its oldest wait, at most every 30 s. Both go to
//! xcl.log too; the per-job detail (`live stall:`, `async pipelines:`) stays there alone.

use std::time::{Duration, Instant};

use crate::live_stall::Sample;

const EVERY: Duration = Duration::from_secs(10);
const STUCK_EVERY: Duration = Duration::from_secs(30);
/// a thread waiting this long is stuck
const STUCK_WAIT: Duration = Duration::from_secs(5);
/// a pipeline wanted and not built for this long is stuck
const STUCK_PIPELINE: Duration = Duration::from_secs(30);

pub fn start() {
    if crate::log::verbose() {
        return;
    }
    super::pso_async::start_status();
    let _ = std::thread::Builder::new().name("xcl-status".into()).spawn(|| {
        let mut last = (Instant::now(), super::present::counters().1);
        let mut stuck_shown: Option<Instant> = None;
        loop {
            std::thread::sleep(EVERY);
            let now = Instant::now();
            let frames = super::present::counters().1;
            let fps = frames.saturating_sub(last.1) as f64 / now.duration_since(last.0).as_secs_f64().max(1e-3);
            last = (now, frames);
            let vram = super::real::video_memory_info().map(|[local, _]| (local.CurrentUsage, local.Budget));
            crate::xlog_console!("status {:.0} s: {}", crate::log::uptime(), status_line(fps, super::pso_async::status_counts().0, memory(), vram));
            match stuck(now) {
                Some(line) if stuck_shown.is_none_or(|t| now.duration_since(t) >= STUCK_EVERY) => {
                    crate::xlog_console!("{line}");
                    stuck_shown = Some(now);
                }
                Some(_) => {}
                None => stuck_shown = None,
            }
        }
    });
}

/// (bytes the process has committed, physical memory available)
fn memory() -> (u64, Option<u64>) {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    let free = (unsafe { GlobalMemoryStatusEx(&mut m) } != 0).then_some(m.ullAvailPhys);
    (crate::util::process_commit_bytes(), free)
}

/// Pipelines (built, failed, pending; `pso_async::status_counts`) as the console and overlay show them.
pub(super) fn pipelines((built, failed, pending): (usize, usize, usize)) -> String {
    format!("pipelines {built} of {} built, {pending} pending, {failed} failed", built + failed + pending)
}

fn status_line(fps: f64, counts: (usize, usize, usize), (commit, free): (u64, Option<u64>), vram: Option<(u64, u64)>) -> String {
    let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
    format!(
        "{fps:.1} fps | {} | RAM {:.1} GB committed, {} free | VRAM {}",
        pipelines(counts),
        gb(commit),
        free.map_or_else(|| "?".into(), |f| format!("{:.1} GB", gb(f))),
        vram.map_or_else(|| "n/a".into(), |(used, budget)| format!("{:.1} of {:.1} GB", gb(used), gb(budget))),
    )
}

/// One line for what is stuck at `now`: the threads waiting `STUCK_WAIT` or longer and the pipelines
/// wanted and not built for `STUCK_PIPELINE` or longer, each as a count and its oldest wait.
pub(super) fn stuck(now: Instant) -> Option<String> {
    stuck_line(now, &super::live_status::observations(), super::pso_async::status_counts().1)
}

fn stuck_line(now: Instant, spans: &[Sample], (pending, oldest): (usize, Option<Duration>)) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(age) = oldest.filter(|&age| pending != 0 && age >= STUCK_PIPELINE) {
        parts.push(format!("{pending} pipelines wanted and not built, the oldest for {:.0} s", age.as_secs_f64()));
    }
    let waits: Vec<_> = spans.iter().filter(|s| s.stage.blocking() && now.saturating_duration_since(s.since) >= STUCK_WAIT).collect();
    if let Some(first) = waits.iter().min_by_key(|s| s.since) {
        parts.push(format!("{} thread wait(s), the oldest a {} for {:.1} s", waits.len(), first.stage.label(), now.saturating_duration_since(first.since).as_secs_f64()));
    }
    (!parts.is_empty()).then(|| format!("stuck: {}", parts.join("; ")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_stall::Stage;

    fn span(stage: Stage, since: Instant) -> Sample {
        Sample { stage, since, tid: 7, pipeline: 0x1234, shader: 0x5678, target: None, queue: None }
    }

    #[test]
    fn the_status_line_names_frames_pipelines_and_memory() {
        let line = status_line(29.96, (5400, 2, 1843), (9 << 30, Some(7 << 30)), Some((5 << 30, 7 << 30)));
        assert_eq!(line, "30.0 fps | pipelines 5400 of 7245 built, 1843 pending, 2 failed | RAM 9.0 GB committed, 7.0 GB free | VRAM 5.0 of 7.0 GB");
        assert!(status_line(0.0, (0, 0, 0), (0, None), None).ends_with("RAM 0.0 GB committed, ? free | VRAM n/a"));
    }

    /// One line for many stuck jobs: their count and the oldest wait, with no job's TID, PSO or
    /// shader key; nothing while every wait is short.
    #[test]
    fn stuck_work_is_one_line_with_its_count_and_oldest_wait() {
        let now = Instant::now();
        let ago = |s: u64| now - Duration::from_secs(s);
        let spans = [span(Stage::PipelineWait, ago(12)), span(Stage::FenceWait, ago(6)), span(Stage::FenceWait, ago(1)), span(Stage::DriverSlot, ago(40))];
        let line = stuck_line(now, &spans, (1843, Some(Duration::from_secs(245)))).unwrap();
        assert_eq!(line, "stuck: 1843 pipelines wanted and not built, the oldest for 245 s; 2 thread wait(s), the oldest a pending pipeline wait for 12.0 s");
        assert!(!line.contains("TID") && !line.contains("PSO") && !line.contains("shader key"));
        assert_eq!(stuck_line(now, &spans[2..], (12, Some(Duration::from_secs(29)))), None, "short waits and a young backlog are not stuck");
        assert_eq!(stuck_line(now, &[], (0, None)), None);
    }
}
