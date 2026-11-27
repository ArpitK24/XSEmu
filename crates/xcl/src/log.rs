//! Process-wide logger. The file handle is opened once at startup (before any file
//! hooks are installed) so logging never re-enters the VFS.
//!
//! xcl.log gets every line of the level asked for. The console (stderr) gets warnings and errors,
//! each call site's first `FIRST` and then one every `EVERY` (`Repeats`), and the lines written for it
//! (`xlog_console!`: milestones, the status line); a message of several lines shows its first.
//! `XCL_VERBOSE=1`: every info line too, whole, and no repeat limit, as before.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Level {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
}

impl Level {
    pub fn parse(s: &str) -> Level {
        match s.to_ascii_lowercase().as_str() {
            "error" => Level::Error,
            "warn" | "warning" => Level::Warn,
            "debug" => Level::Debug,
            "trace" => Level::Trace,
            _ => Level::Info,
        }
    }
    fn tag(self) -> &'static str {
        match self {
            Level::Error => "ERROR",
            Level::Warn => "WARN ",
            Level::Info => "INFO ",
            Level::Debug => "DEBUG",
            Level::Trace => "TRACE",
        }
    }
}

static FILE: Mutex<Option<File>> = Mutex::new(None);
static LEVEL: AtomicU8 = AtomicU8::new(Level::Info as u8);
static ECHO: AtomicBool = AtomicBool::new(true);
static START: OnceLock<Instant> = OnceLock::new();
static REPEATS: Mutex<Option<Repeats>> = Mutex::new(None);

/// `XCL_VERBOSE=1`: the console shows every info line, and warnings and errors without a limit.
pub fn verbose() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_VERBOSE").is_ok_and(|v| v.trim() == "1"))
}

/// a call site's warnings and errors shown on the console before the limit
const FIRST: u32 = 3;
/// then one every
const EVERY: Duration = Duration::from_secs(30);

/// The warnings and errors each call site showed on the console: (shown, when last, held back since).
#[derive(Default)]
struct Repeats {
    sites: HashMap<&'static str, (u32, Option<Instant>, u64)>,
}

impl Repeats {
    /// Whether a line from `site` is shown at `now`: how many from it were held back since the
    /// last one shown.
    fn admit(&mut self, site: &'static str, now: Instant) -> Option<u64> {
        let (shown, last, held) = self.sites.entry(site).or_default();
        if *shown >= FIRST && last.is_some_and(|t| now.saturating_duration_since(t) < EVERY) {
            *held += 1;
            return None;
        }
        *shown = shown.saturating_add(1);
        *last = Some(now);
        Some(std::mem::take(held))
    }
}

/// Seconds since the log started (the host's start) – the times log lines carry.
pub fn uptime() -> f64 {
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

pub fn init(path: Option<&Path>, level: Level, echo_stderr: bool) {
    START.get_or_init(Instant::now);
    LEVEL.store(level as u8, Ordering::Relaxed);
    ECHO.store(echo_stderr, Ordering::Relaxed);
    if let Some(p) = path {
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(f) = File::create(p) {
            if let Ok(mut g) = FILE.lock() {
                *g = Some(f);
            }
        }
    }
}

pub fn enabled(level: Level) -> bool {
    level as u8 <= LEVEL.load(Ordering::Relaxed)
}

/// A line of `level` from `site` (file:line, for a warning or error); `console`: one written for the
/// console as well.
pub fn write(level: Level, site: &'static str, console: bool, args: std::fmt::Arguments<'_>) {
    if !enabled(level) {
        return;
    }
    let t = START.get_or_init(Instant::now).elapsed().as_secs_f64();
    let tid = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
    let line = format!("[{t:9.3}][{tid:5}][{}] {args}\n", level.tag());
    if let Ok(mut g) = FILE.try_lock().or_else(|_| FILE.lock()) {
        if let Some(f) = g.as_mut() {
            let _ = f.write_all(line.as_bytes());
            if level <= Level::Warn {
                let _ = f.flush();
            }
        }
    }
    if !ECHO.load(Ordering::Relaxed) {
        return;
    }
    let held = if verbose() {
        (level <= Level::Info).then_some(0)
    } else if console {
        Some(0)
    } else if level <= Level::Warn {
        REPEATS.try_lock().map_or(Some(0), |mut r| r.get_or_insert_with(Repeats::default).admit(site, Instant::now()))
    } else {
        None
    };
    let Some(held) = held else { return };
    let mut text = args.to_string();
    if !verbose() {
        if let Some((first, rest)) = text.split_once('\n') {
            text = format!("{first} (+{} lines in xcl.log)", rest.lines().count());
        }
    }
    let more = if held == 0 { String::new() } else { format!(" ({held} more from here not shown; xcl.log has them)") };
    let _ = std::io::stderr().write_all(format!("xcl: {text}{more}\n").as_bytes());
}

pub fn flush() {
    if let Ok(mut g) = FILE.lock() {
        if let Some(f) = g.as_mut() {
            let _ = f.flush();
        }
    }
}

pub(crate) fn access_violation_notice(tid: u32, unhandled: bool) {
    if let Ok(mut g) = FILE.try_lock() {
        if let Some(f) = g.as_mut() {
            let kind = if unhandled { "unhandled exception" } else { "title dump requested for exception" };
            let _ = writeln!(f, "[{:9.3}][{tid:5}][ERROR] {kind} 0xc0000005; raw diagnostics are local only", uptime());
        }
    }
}

#[macro_export]
macro_rules! xlog_error { ($($a:tt)*) => { $crate::log::write($crate::log::Level::Error, concat!(file!(), ":", line!()), false, format_args!($($a)*)) } }
#[macro_export]
macro_rules! xlog_warn { ($($a:tt)*) => { $crate::log::write($crate::log::Level::Warn, concat!(file!(), ":", line!()), false, format_args!($($a)*)) } }
#[macro_export]
macro_rules! xlog_info { ($($a:tt)*) => { $crate::log::write($crate::log::Level::Info, "", false, format_args!($($a)*)) } }
/// An info line for the console as well as xcl.log: a milestone or the status line.
#[macro_export]
macro_rules! xlog_console { ($($a:tt)*) => { $crate::log::write($crate::log::Level::Info, "", true, format_args!($($a)*)) } }
#[macro_export]
macro_rules! xlog_debug { ($($a:tt)*) => { if $crate::log::enabled($crate::log::Level::Debug) { $crate::log::write($crate::log::Level::Debug, "", false, format_args!($($a)*)) } } }
#[macro_export]
macro_rules! xlog_trace { ($($a:tt)*) => { if $crate::log::enabled($crate::log::Level::Trace) { $crate::log::write($crate::log::Level::Trace, "", false, format_args!($($a)*)) } } }

#[cfg(test)]
mod tests {
    use super::*;

    /// A call site's first lines reach the console, then one every EVERY carrying the count
    /// held back since; another call site has its own count.
    #[test]
    fn repeated_warnings_are_limited_per_call_site() {
        let mut r = Repeats::default();
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        assert_eq!((0..FIRST).map(|k| r.admit("a.rs:1", at(k as u64))).collect::<Vec<_>>(), vec![Some(0); FIRST as usize]);
        assert_eq!(r.admit("a.rs:1", at(3)), None);
        assert_eq!(r.admit("a.rs:1", at(4)), None);
        assert_eq!(r.admit("b.rs:9", at(4)), Some(0), "another call site");
        assert_eq!(r.admit("a.rs:1", at(2 + EVERY.as_secs())), Some(2), "one every EVERY, with the count held back");
        assert_eq!(r.admit("a.rs:1", at(3 + EVERY.as_secs())), None);
    }
}
