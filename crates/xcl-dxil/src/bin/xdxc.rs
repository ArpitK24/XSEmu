//! `xdxc`: the backend as a command, and the corpus runner the build is measured with.
//!
//! ```text
//! xdxc <in.hlsl> [-T <profile>] [-E <entry>] [-o out.dxil] [--emit-ir out.ll] [--time] [--stats] [--stack-mb N]
//!      [--stop-after parse|lower] [--lane-check]
//! xdxc --batch <dir or file>... [-j N] [--out-dir DIR] [--ir-dir DIR] [--tsv FILE] [--timeout SECONDS] [--lane-check]
//! ```
//!
//! One file: exit code 0 compiled, 2 unsupported (the caller falls back to DXC), 1 any other
//! error, 101 a panic (a bug in the crate). Without `-T` / `-E` the profile and entry point come
//! from the first line of the source when it is `// xdxc: -E main -T cs_6_6` (what the corpus
//! dump writes), else from a `<name>.<profile>.hlsl` file name; the entry defaults to `main`.
//! `--stats` prints one tab-separated `key=value` line on stdout whatever the outcome (see
//! [`Report::line`]); `--emit-ir` writes the lowered function as LLVM-style text
//! ([`xcl_dxil::irprint`]), which works before the bitcode writer does. `--stop-after` runs the front
//! end only, to tell which phase a crash or a slowdown belongs to; `--stack-mb` runs the compile on
//! a thread with that much stack, to measure how deep a nesting it survives.
//!
//! `--lane-check` turns on the lowering's lane-uniformity check (`XCL_DXIL_LANE_CHECK=2`, off in the
//! game for its cost): a program whose lowering needs lanes to leave a loop around a wave operation
//! together, where the check cannot prove they do, ends as "lane-check refused" (exit code 2, like
//! unsupported) with the chain of variables that makes the exit vary as its message.
//!
//! `--batch` runs every `.hlsl` under the given folders, one child process per file on all cores
//! but one, so that a panic, a stack overflow or a hang in one file is an outcome of that file
//! and not the end of the run. It prints a summary grouped by outcome, the unsupported / error /
//! panic texts grouped with numbers and numbered identifiers normalised, each with the agent
//! that most likely owns it. `-T` / `-E` there are the defaults for files without a first-line
//! comment (and, for `-T`, without a profile in their name). Exit code 1 when any file ended in an internal error, a panic, a crash or a timeout.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use xcl_dxil::{Error, Stats};

const USAGE: &str = "usage: xdxc <in.hlsl> [-T <profile>] [-E main] [-o out.dxil] [--emit-ir out.ll] [--time] [--stats] [--stack-mb N] [--stop-after parse|lower] [--lane-check]
       xdxc --batch <dir or file>... [-j N] [--out-dir DIR] [--ir-dir DIR] [--tsv FILE] [--timeout SECONDS] [--lane-check]";

/// The start of the lowering's `Error::Unsupported` text when the lane check refuses a program;
/// with `XCL_DXIL_LANE_CHECK=2` the chain follows after `: `.
const LANES_APART: &str = "a loop around a wave operation whose exit can differ between lanes";

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Outcome {
    Ok,
    Unsupported,
    /// Unsupported because `--lane-check` could not prove that lanes leave a loop together.
    LaneRefused,
    Source,
    Internal,
    Panic,
    /// The process died (stack overflow, abort): only seen from the batch parent.
    Crash,
    Timeout,
    /// The file could not be read or names no profile.
    Io,
}

impl Outcome {
    const ALL: [Outcome; 9] =
        [Outcome::Ok, Outcome::Unsupported, Outcome::LaneRefused, Outcome::Source, Outcome::Internal, Outcome::Panic, Outcome::Crash, Outcome::Timeout, Outcome::Io];

    fn key(self) -> &'static str {
        match self {
            Outcome::Ok => "ok",
            Outcome::Unsupported => "unsupported",
            Outcome::LaneRefused => "lane-refused",
            Outcome::Source => "source",
            Outcome::Internal => "internal",
            Outcome::Panic => "panic",
            Outcome::Crash => "crash",
            Outcome::Timeout => "timeout",
            Outcome::Io => "io",
        }
    }

    fn title(self) -> &'static str {
        match self {
            Outcome::LaneRefused => "lane-check refused",
            Outcome::Source => "source error",
            Outcome::Internal => "internal error",
            Outcome::Crash => "crash",
            Outcome::Io => "not run",
            o => o.key(),
        }
    }

    fn from_key(s: &str) -> Option<Outcome> {
        Outcome::ALL.into_iter().find(|o| o.key() == s)
    }

    fn exit_code(self) -> i32 {
        match self {
            Outcome::Ok => 0,
            Outcome::Unsupported | Outcome::LaneRefused => 2,
            Outcome::Panic => 101,
            _ => 1,
        }
    }

    /// A bug in the crate, whatever the input.
    fn is_bug(self) -> bool {
        matches!(self, Outcome::Internal | Outcome::Panic | Outcome::Crash | Outcome::Timeout)
    }
}

/// What happened to one file.
#[derive(Clone, Debug)]
struct Report {
    file: String,
    outcome: Outcome,
    profile: String,
    src_bytes: usize,
    /// seconds in [`xcl_dxil::compile_timed`], up to the error when there is one
    total: f64,
    stats: Stats,
    /// where a failure happened: parse, lower, backend (flags, metadata, bitcode, container)
    phase: String,
    /// the agent whose code most likely has to change (a guess from the phase and the text)
    owner: String,
    peak_rss_kb: u64,
    /// seconds reading the source
    read: f64,
    /// seconds from the start of `main` to printing the `--stats` line
    main: f64,
    msg: String,
}

impl Report {
    fn new(file: &str) -> Report {
        Report {
            file: file.to_string(),
            outcome: Outcome::Io,
            profile: String::new(),
            src_bytes: 0,
            total: 0.0,
            stats: Stats::default(),
            phase: String::new(),
            owner: String::new(),
            peak_rss_kb: 0,
            read: 0.0,
            main: 0.0,
            msg: String::new(),
        }
    }

    /// The `--stats` line: `xdxc`, then tab-separated `key=value` fields, `msg` last (it may
    /// hold any character but tab and newline). Times are seconds.
    fn line(&self) -> String {
        let s = &self.stats;
        let mut l = format!(
            "xdxc\tfile={}\toutcome={}\tprofile={}\tsrc_bytes={}\ttotal_s={:.6}\tparse_s={:.6}\tlower_s={:.6}\twrite_s={:.6}\tcontainer_s={:.6}\tinstructions={}\tblocks={}\tdxil_bytes={}\tpeak_rss_kb={}\tread_s={:.6}\tfree_s={:.6}\tmain_s={:.6}",
            self.file,
            self.outcome.key(),
            self.profile,
            self.src_bytes,
            self.total,
            s.parse,
            s.lower,
            s.write,
            s.container,
            s.instructions,
            s.blocks,
            s.dxil_bytes,
            self.peak_rss_kb,
            self.read,
            s.free,
            self.main
        );
        if self.outcome != Outcome::Ok {
            let msg: String = self.msg.chars().map(|c| if c == '\t' || c == '\n' || c == '\r' { ' ' } else { c }).take(2000).collect();
            l.push_str(&format!("\tphase={}\towner={}\tnorm={}\tmsg={}", self.phase, self.owner, normalise(&msg), msg));
        }
        l
    }

    fn from_line(line: &str) -> Option<Report> {
        let mut fields = line.split('\t');
        if fields.next() != Some("xdxc") {
            return None;
        }
        let mut r = Report::new("");
        let rest = line.split_once("\tmsg=").map(|(_, m)| m);
        for f in fields {
            let Some((k, v)) = f.split_once('=') else { continue };
            let num = || v.parse::<f64>().unwrap_or(0.0);
            match k {
                "file" => r.file = v.to_string(),
                "outcome" => r.outcome = Outcome::from_key(v)?,
                "profile" => r.profile = v.to_string(),
                "src_bytes" => r.src_bytes = num() as usize,
                "total_s" => r.total = num(),
                "parse_s" => r.stats.parse = num(),
                "lower_s" => r.stats.lower = num(),
                "write_s" => r.stats.write = num(),
                "container_s" => r.stats.container = num(),
                "instructions" => r.stats.instructions = num() as usize,
                "blocks" => r.stats.blocks = num() as usize,
                "dxil_bytes" => r.stats.dxil_bytes = num() as usize,
                "peak_rss_kb" => r.peak_rss_kb = num() as u64,
                "read_s" => r.read = num(),
                "free_s" => r.stats.free = num(),
                "main_s" => r.main = num(),
                "phase" => r.phase = v.to_string(),
                "owner" => r.owner = v.to_string(),
                "msg" => break,
                _ => {}
            }
        }
        r.msg = rest.unwrap_or("").to_string();
        Some(r)
    }
}

/// The agent that owns the file a panic came from (`tools/owners.txt`).
fn owner_of_path(path: &str) -> &'static str {
    let p = path.replace('\\', "/");
    let p = p.rsplit_once("src/").map_or(p.as_str(), |(_, r)| r);
    const TABLE: [(&str, &str); 18] = [
        ("hlsl/lex", "parser"),
        ("hlsl/parse", "parser"),
        ("hlsl/ast", "parser"),
        ("hlsl/lower", "lower"),
        ("hlsl/types", "lower"),
        ("hlsl/emit", "lower"),
        ("hlsl/intrinsics", "intrinsics"),
        ("hlsl/resources", "resources"),
        ("hlsl/stage", "stage"),
        ("bitstream", "bitcode"),
        ("bcwriter", "bitcode"),
        ("bcreader", "bitcode"),
        ("dxop", "bitcode"),
        ("irprint", "bitcode"),
        ("container", "container"),
        ("meta", "meta"),
        ("flags", "meta"),
        ("interp", "interp"),
    ];
    TABLE.iter().find(|(prefix, _)| p.starts_with(prefix)).map_or("lead", |&(_, owner)| owner)
}

/// A guess at the owner of an error from the phase it came out of and its text: the lowering
/// calls into the intrinsic, resource and stage modules, and the back end is three agents.
fn owner_of_error(phase: &str, msg: &str) -> &'static str {
    let m = msg.to_ascii_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| m.contains(w));
    match phase {
        "parse" => "parser",
        "lower" => {
            if m.starts_with("stage") || m.starts_with("profile") || has(&["semantic", "signature", "sv_", "entry point", "patch", "payload"]) {
                "stage"
            } else if has(&["intrinsic"]) {
                "intrinsics"
            } else if has(&["texture", "sampler", "buffer", "resource", "descriptorheap", "getdimensions"]) {
                "resources"
            } else {
                "lower"
            }
        }
        _ => {
            if has(&["bcwriter", "bitstream", "bitcode", "dx.op", "abbrev"]) {
                "bitcode"
            } else if has(&["container", "psv", "isg1", "osg1", "psg1", "sfi0", "hash"]) {
                "container"
            } else if has(&["meta", "flag", "dx."]) {
                "meta"
            } else {
                "backend"
            }
        }
    }
}

/// Type names, profiles and LLVM types keep their digits when messages are grouped.
fn is_type_like(t: &str) -> bool {
    const STEMS: [&str; 20] = [
        "bool", "int", "uint", "float", "double", "half", "dword", "min16float", "min16int", "min16uint", "min10float", "min12int",
        "int64_t", "uint64_t", "int32_t", "uint32_t", "int16_t", "uint16_t", "float16_t", "float64_t",
    ];
    let b = t.as_bytes();
    let dim = |c: u8| (b'1'..=b'4').contains(&c);
    let n = b.len();
    let stem = if n >= 4 && dim(b[n - 1]) && b[n - 2] == b'x' && dim(b[n - 3]) {
        &t[..n - 3]
    } else if n >= 2 && dim(b[n - 1]) {
        &t[..n - 1]
    } else {
        t
    };
    if STEMS.contains(&t) || STEMS.contains(&stem) {
        return true;
    }
    // i1, i32, f64
    if n >= 2 && (b[0] == b'i' || b[0] == b'f') && b[1..].iter().all(u8::is_ascii_digit) {
        return true;
    }
    // cs_6_6, lib_6_8
    if let Some((stage, version)) = t.split_once('_') {
        if stage.len() <= 3 && stage.bytes().all(|c| c.is_ascii_lowercase()) && version.len() == 3 && version.as_bytes()[1] == b'_' {
            return true;
        }
    }
    // Texture2D, RWTexture3D, Texture2DMSArray, ...
    t.contains("1D") || t.contains("2D") || t.contains("3D")
}

/// The text of a message with what varies from program to program taken out: numbers become
/// `N`, and the digit runs of identifiers too (`v12` → `vN`) unless the identifier is a type.
fn normalise(msg: &str) -> String {
    let b = msg.as_bytes();
    let mut out = String::with_capacity(msg.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let tok = &msg[start..i];
            if !tok.bytes().any(|c| c.is_ascii_digit()) || is_type_like(tok) {
                out.push_str(tok);
            } else {
                let mut in_digits = false;
                for ch in tok.chars() {
                    if ch.is_ascii_digit() {
                        if !in_digits {
                            out.push('N');
                        }
                        in_digits = true;
                    } else {
                        in_digits = false;
                        out.push(ch);
                    }
                }
            }
        } else if c.is_ascii_digit() {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'.') {
                i += 1;
            }
            out.push('N');
        } else {
            let start = i;
            i += 1;
            while i < b.len() && (b[i] & 0xC0) == 0x80 {
                i += 1;
            }
            out.push_str(&msg[start..i]);
        }
    }
    out
}

/// `-E` and `-T` from a first line `// xdxc: -E main -T cs_6_6 ...`.
fn header(src: &str) -> (Option<String>, Option<String>) {
    let first = src.lines().next().unwrap_or("");
    let Some(args) = first.trim_start().strip_prefix("// xdxc:") else { return (None, None) };
    let (mut entry, mut profile) = (None, None);
    let mut words = args.split_whitespace();
    while let Some(w) = words.next() {
        match w {
            "-E" => entry = words.next().map(str::to_string),
            "-T" => profile = words.next().map(str::to_string),
            _ => {}
        }
    }
    (entry, profile)
}

/// The profile of a `<name>.<profile>.hlsl` file name.
fn profile_of_name(path: &str) -> Option<String> {
    let name = Path::new(path).file_name()?.to_str()?;
    let mut parts = name.rsplit('.');
    let (_ext, profile) = (parts.next()?, parts.next()?);
    parts.next()?;
    let b = profile.as_bytes();
    (b.len() == 6 && &b[2..4] == b"_6" && b[4] == b'_' && b[5].is_ascii_digit()).then(|| profile.to_string())
}

fn peak_rss_kb() -> u64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else { return 0 };
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .unwrap_or(0)
}

/// Where the last panic happened and what it said, set by the hook [`main`] installs.
static PANIC: Mutex<Option<(String, String)>> = Mutex::new(None);

/// When `main` started, for `main_s` of the `--stats` line.
static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

fn take_panic() -> (String, String) {
    PANIC.lock().ok().and_then(|mut p| p.take()).unwrap_or_else(|| ("?".into(), "panic".into()))
}

#[derive(Default)]
struct Opts {
    inputs: Vec<String>,
    out: Option<String>,
    emit_ir: Option<String>,
    entry: Option<String>,
    profile: Option<String>,
    time: bool,
    stats: bool,
    stack_mb: Option<usize>,
    stop_after: Option<String>,
    batch: bool,
    jobs: Option<usize>,
    out_dir: Option<String>,
    ir_dir: Option<String>,
    tsv: Option<String>,
    timeout: Option<f64>,
    lane_check: bool,
}

/// Compile one file in this process. The second value is the container.
fn run_one(o: &Opts, input: &str) -> (Report, Option<Vec<u8>>) {
    let mut r = Report::new(input);
    let t = Instant::now();
    let read = std::fs::read_to_string(input);
    r.read = t.elapsed().as_secs_f64();
    let src = match read {
        Ok(s) => s,
        Err(e) => {
            r.msg = e.to_string();
            return (r, None);
        }
    };
    r.src_bytes = src.len();
    let (h_entry, h_profile) = header(&src);
    let entry = o.entry.clone().or(h_entry).unwrap_or_else(|| "main".to_string());
    let Some(profile) = o.profile.clone().or(h_profile).or_else(|| profile_of_name(input)) else {
        r.msg = "no profile: pass -T, or start the source with // xdxc: -E main -T <profile>".into();
        return (r, None);
    };
    r.profile = profile.clone();

    let t = Instant::now();
    let result = catch_unwind(AssertUnwindSafe(|| match o.stop_after.as_deref() {
        // the front end alone: no container, the time of the phases that ran
        Some(phase) => {
            let mut stats = Stats::default();
            let unit = xcl_dxil::hlsl::parse::parse(&src)?;
            stats.parse = t.elapsed().as_secs_f64();
            if phase == "lower" {
                xcl_dxil::hlsl::lower::lower(&unit, &entry, &profile)?;
                stats.lower = t.elapsed().as_secs_f64() - stats.parse;
            }
            Ok((Vec::new(), stats))
        }
        None => xcl_dxil::compile_timed(&src, &entry, &profile),
    }));
    r.total = t.elapsed().as_secs_f64();
    let mut dxil = None;
    match result {
        Ok(Ok((c, stats))) => {
            r.outcome = Outcome::Ok;
            r.stats = stats;
            dxil = Some(c);
        }
        Ok(Err(e)) => {
            // which phase: run the front end alone again (cheap next to finding it by hand)
            let phase = catch_unwind(AssertUnwindSafe(|| match xcl_dxil::hlsl::parse::parse(&src) {
                Err(_) => "parse",
                Ok(unit) => match xcl_dxil::hlsl::lower::lower(&unit, &entry, &profile) {
                    Err(_) => "lower",
                    Ok(_) => "backend",
                },
            }))
            .unwrap_or("?");
            r.phase = phase.to_string();
            let (outcome, msg) = match e {
                Error::Unsupported(what) => match what.strip_prefix(LANES_APART) {
                    Some(chain) => (Outcome::LaneRefused, chain.strip_prefix(": ").unwrap_or(LANES_APART).to_string()),
                    None => (Outcome::Unsupported, what),
                },
                Error::Source { line, msg } => (Outcome::Source, format!("line {line}: {msg}")),
                Error::Internal(what) => (Outcome::Internal, what),
            };
            r.outcome = outcome;
            r.owner = owner_of_error(phase, &msg).to_string();
            r.msg = msg;
        }
        Err(_) => {
            let (at, msg) = take_panic();
            r.outcome = Outcome::Panic;
            r.phase = "panic".into();
            r.owner = owner_of_path(&at).to_string();
            r.msg = format!("{at}: {msg}");
        }
    }
    if let Some(path) = &o.emit_ir {
        let ir = catch_unwind(AssertUnwindSafe(|| {
            let unit = xcl_dxil::hlsl::parse::parse(&src)?;
            let (module, _) = xcl_dxil::hlsl::lower::lower(&unit, &entry, &profile)?;
            Ok::<String, Error>(xcl_dxil::irprint::module(&module))
        }));
        match ir {
            Ok(Ok(text)) => {
                if let Err(e) = std::fs::write(path, text) {
                    eprintln!("{path}: {e}");
                }
            }
            // the compile above already reported why there is no IR
            Ok(Err(_)) => {}
            Err(_) => {
                let (at, msg) = take_panic();
                if r.outcome != Outcome::Panic {
                    eprintln!("{input}: --emit-ir: panic at {at}: {msg}");
                }
            }
        }
    }
    r.peak_rss_kb = peak_rss_kb();
    (r, dxil)
}

fn single(o: &Opts) -> i32 {
    let input = &o.inputs[0];
    let (r, dxil) = match o.stack_mb {
        // the compile on a thread with a stack of that size, to find out how much it needs
        Some(mb) => {
            let run = std::thread::scope(|s| {
                std::thread::Builder::new().stack_size(mb << 20).spawn_scoped(s, || run_one(o, input)).map(|h| h.join())
            });
            match run {
                Ok(Ok(x)) => x,
                _ => {
                    eprintln!("{input}: the compile thread did not start or died");
                    return 1;
                }
            }
        }
        None => run_one(o, input),
    };
    if o.stats {
        let mut r = r.clone();
        r.main = START.get().map_or(0.0, |s| s.elapsed().as_secs_f64());
        println!("{}", r.line());
    }
    match r.outcome {
        Outcome::Ok => {
            let s = &r.stats;
            if o.time {
                eprintln!(
                    "{input}: {:.3} s (parse {:.3}, lower {:.3}, write {:.3}, container {:.3}); {} instructions, {} blocks, {} bytes",
                    r.total, s.parse, s.lower, s.write, s.container, s.instructions, s.blocks, s.dxil_bytes
                );
            }
            if let (Some(out), Some(dxil), None) = (&o.out, &dxil, &o.stop_after) {
                if let Err(e) = std::fs::write(out, dxil) {
                    eprintln!("{out}: {e}");
                    return 1;
                }
            }
        }
        Outcome::Unsupported => eprintln!("{input}: unsupported: {}", r.msg),
        Outcome::LaneRefused => eprintln!("{input}: lane-check refused: {}", r.msg),
        Outcome::Internal => eprintln!("{input}: internal error: {}", r.msg),
        Outcome::Panic => eprintln!("{input}: panic at {}", r.msg),
        _ => eprintln!("{input}: {}", r.msg),
    }
    r.outcome.exit_code()
}

fn collect(inputs: &[String]) -> Vec<String> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "hlsl") {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    for i in inputs {
        let p = PathBuf::from(i);
        if p.is_dir() {
            let mut found = Vec::new();
            walk(&p, &mut found);
            found.sort();
            files.extend(found);
        } else {
            files.push(p);
        }
    }
    files.iter().map(|p| p.to_string_lossy().into_owned()).collect()
}

/// The path of `file`'s output in `dir`: its name with `.hlsl` replaced by `ext`.
fn sibling(dir: &str, file: &str, ext: &str) -> String {
    let name = Path::new(file).file_name().and_then(|n| n.to_str()).unwrap_or("out");
    let stem = name.strip_suffix(".hlsl").unwrap_or(name);
    Path::new(dir).join(format!("{stem}.{ext}")).to_string_lossy().into_owned()
}

/// Run one file in a child `xdxc --stats` and classify how it ended.
fn run_child(o: &Opts, exe: &Path, file: &str, timeout: f64) -> Report {
    let mut r = Report::new(file);
    r.src_bytes = std::fs::metadata(file).map(|m| m.len() as usize).unwrap_or(0);
    let mut cmd = Command::new(exe);
    cmd.arg(file).arg("--stats");
    // -T / -E are defaults here, for files whose first line and name say nothing
    let head = {
        let mut buf = vec![0u8; 256];
        let n = std::fs::File::open(file).and_then(|mut f| f.read(&mut buf)).unwrap_or(0);
        String::from_utf8_lossy(&buf[..n]).into_owned()
    };
    let (h_entry, h_profile) = header(&head);
    if let (None, None, Some(p)) = (&h_profile, profile_of_name(file), &o.profile) {
        cmd.arg("-T").arg(p);
    }
    if let (None, Some(e)) = (&h_entry, &o.entry) {
        cmd.arg("-E").arg(e);
    }
    if let Some(dir) = &o.out_dir {
        cmd.arg("-o").arg(sibling(dir, file, "dxil"));
    }
    if let Some(dir) = &o.ir_dir {
        cmd.arg("--emit-ir").arg(sibling(dir, file, "ll"));
    }
    if let Some(mb) = o.stack_mb {
        cmd.arg("--stack-mb").arg(mb.to_string());
    }
    if o.lane_check {
        cmd.arg("--lane-check");
    }
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let t = Instant::now();
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            r.msg = format!("cannot start {}: {e}", exe.display());
            return r;
        }
    };
    // drain both pipes on their own threads: a child that prints a lot must not look hung
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut text = Vec::new();
            if let Some(mut p) = pipe {
                let _ = p.read_to_end(&mut text);
            }
            String::from_utf8_lossy(&text).into_owned()
        })
    };
    let out = drain(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let err = drain(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if t.elapsed().as_secs_f64() > timeout => {
                timed_out = true;
                let _ = child.kill();
                break child.wait().ok();
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(2)),
            Err(_) => break None,
        }
    };
    let (out, err) = (out.join().unwrap_or_default(), err.join().unwrap_or_default());
    if timed_out {
        r.outcome = Outcome::Timeout;
        r.total = t.elapsed().as_secs_f64();
        r.owner = "?".into();
        r.msg = format!("no result after {timeout} s");
        return r;
    }
    if let Some(parsed) = out.lines().find_map(Report::from_line) {
        return parsed;
    }
    // no stats line: the process died before it could print one
    let last = err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string();
    r.outcome = Outcome::Crash;
    r.total = t.elapsed().as_secs_f64();
    r.phase = "?".into();
    r.owner = "?".into();
    r.msg = match status {
        Some(s) => match s.code() {
            Some(code) => format!("exit code {code}: {last}"),
            None => format!("killed ({s}): {last}"),
        },
        None => format!("lost the child: {last}"),
    };
    r
}

fn batch(o: &Opts) -> i32 {
    let files = collect(&o.inputs);
    if files.is_empty() {
        eprintln!("xdxc --batch: no .hlsl file under {}", o.inputs.join(", "));
        return 1;
    }
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("xdxc --batch: cannot find my own executable: {e}");
            return 1;
        }
    };
    for dir in [&o.out_dir, &o.ir_dir].into_iter().flatten() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("{dir}: {e}");
            return 1;
        }
    }
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let jobs = o.jobs.unwrap_or(cores.saturating_sub(1)).clamp(1, files.len());
    let timeout = o.timeout.unwrap_or(120.0);
    let next = AtomicUsize::new(0);
    let reports: Mutex<Vec<Option<Report>>> = Mutex::new(vec![None; files.len()]);
    let t = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(file) = files.get(i) else { break };
                let r = run_child(o, &exe, file, timeout);
                if let Ok(mut all) = reports.lock() {
                    all[i] = Some(r);
                }
            });
        }
    });
    let wall = t.elapsed().as_secs_f64();
    let reports: Vec<Report> = reports.into_inner().unwrap_or_default().into_iter().flatten().collect();

    if let Some(path) = &o.tsv {
        let text: String = reports.iter().map(|r| r.line() + "\n").collect();
        if let Err(e) = std::fs::write(path, text) {
            eprintln!("{path}: {e}");
            return 1;
        }
    }

    let stdout = std::io::stdout();
    let mut w = stdout.lock();
    let ok: Vec<&Report> = reports.iter().filter(|r| r.outcome == Outcome::Ok).collect();
    let _ = writeln!(w, "xdxc batch: {} files, {} at a time, {:.2} s", reports.len(), jobs, wall);
    for outcome in Outcome::ALL {
        let of: Vec<&Report> = reports.iter().filter(|r| r.outcome == outcome).collect();
        if of.is_empty() && matches!(outcome, Outcome::Crash | Outcome::Timeout | Outcome::Io) {
            continue;
        }
        let _ = writeln!(w, "{:<15}{:>5}", outcome.title(), of.len());
        if outcome == Outcome::Ok {
            continue;
        }
        // (owner, normalised text) → the files
        let mut groups: BTreeMap<(String, String), Vec<&Report>> = BTreeMap::new();
        for r in of {
            groups.entry((r.owner.clone(), normalise(&r.msg))).or_default().push(r);
        }
        let mut groups: Vec<_> = groups.into_iter().collect();
        groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));
        for ((owner, text), of) in groups {
            let names: Vec<&str> =
                of.iter().take(3).map(|r| Path::new(&r.file).file_name().and_then(|n| n.to_str()).unwrap_or(&r.file)).collect();
            let more = if of.len() > names.len() { ", ..." } else { "" };
            let _ = writeln!(w, "  {:>5}  [{}] {}   ({}{})", of.len(), owner, text, names.join(", "), more);
        }
    }
    if !ok.is_empty() {
        let bytes: usize = ok.iter().map(|r| r.src_bytes).sum();
        let seconds: f64 = ok.iter().map(|r| r.total).sum();
        let (insts, dxil): (usize, usize) = ok.iter().fold((0, 0), |a, r| (a.0 + r.stats.instructions, a.1 + r.stats.dxil_bytes));
        let _ = writeln!(
            w,
            "compiled: {:.2} MB of source in {:.3} s ({:.1} MB/s), {} instructions, {} container bytes",
            bytes as f64 / 1e6,
            seconds,
            bytes as f64 / 1e6 / seconds.max(1e-9),
            insts,
            dxil
        );
    }
    i32::from(reports.iter().any(|r| r.outcome.is_bug()))
}

fn main() {
    START.get_or_init(Instant::now);
    let mut o = Opts::default();
    let mut args = std::env::args().skip(1);
    let mut bad = false;
    while let Some(a) = args.next() {
        let mut number = |into: &mut Option<f64>| match args.next().and_then(|v| v.parse::<f64>().ok()) {
            Some(n) => *into = Some(n),
            None => bad = true,
        };
        let mut n = None;
        match a.as_str() {
            "-T" => o.profile = args.next(),
            "-E" => o.entry = args.next(),
            "-o" | "-Fo" => o.out = args.next(),
            "--emit-ir" => o.emit_ir = args.next(),
            "--time" => o.time = true,
            "--stats" => o.stats = true,
            "--stop-after" => {
                o.stop_after = args.next();
                bad |= !matches!(o.stop_after.as_deref(), Some("parse" | "lower"));
            }
            "--batch" => o.batch = true,
            "--lane-check" => o.lane_check = true,
            "--out-dir" => o.out_dir = args.next(),
            "--ir-dir" => o.ir_dir = args.next(),
            "--tsv" => o.tsv = args.next(),
            "--timeout" => number(&mut o.timeout),
            "--stack-mb" => {
                number(&mut n);
                o.stack_mb = n.map(|n| n as usize);
            }
            "-j" => {
                number(&mut n);
                o.jobs = n.map(|n| n as usize);
            }
            _ if a.starts_with("--") => bad = true,
            _ => o.inputs.push(a),
        }
    }
    if bad || o.inputs.is_empty() || (!o.batch && o.inputs.len() != 1) {
        eprintln!("{USAGE}");
        std::process::exit(1);
    }
    if o.lane_check {
        std::env::set_var("XCL_DXIL_LANE_CHECK", "2");
    }
    // a panic is an outcome to report with its place, not a message in the middle of the output
    std::panic::set_hook(Box::new(|info| {
        let at = info.location().map_or_else(|| "?".to_string(), |l| format!("{}:{}", l.file(), l.line()));
        let payload = info.payload();
        let msg = payload.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| payload.downcast_ref::<String>().cloned());
        if let Ok(mut p) = PANIC.lock() {
            *p = Some((at, msg.unwrap_or_else(|| "panic".to_string())));
        }
    }));
    let code = if o.batch { batch(&o) } else { single(&o) };
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_group_by_shape() {
        assert_eq!(normalise("line 12: unknown identifier v123 in s4_x"), "line N: unknown identifier vN in sN_x");
        assert_eq!(normalise("no overload of mul for float3x3 and uint4 (cs_6_6, i32)"), "no overload of mul for float3x3 and uint4 (cs_6_6, i32)");
        assert_eq!(normalise("Texture2D<float4> at 0x3F800000u, 1.5e3"), "Texture2D<float4> at N, N");
        assert_eq!(normalise("intrinsic frexp"), "intrinsic frexp");
    }

    #[test]
    fn header_and_name() {
        assert_eq!(header("// xdxc: -E main -T ps_6_6\nfloat x;"), (Some("main".into()), Some("ps_6_6".into())));
        assert_eq!(header("// generated\n"), (None, None));
        assert_eq!(profile_of_name("a/0123.ms_6_8.hlsl"), Some("ms_6_8".into()));
        assert_eq!(profile_of_name("a/f1dc.hlsl"), None);
    }

    #[test]
    fn stats_line_round_trips() {
        let mut r = Report::new("a b.hlsl");
        r.outcome = Outcome::Unsupported;
        r.profile = "cs_6_6".into();
        r.phase = "lower".into();
        r.owner = "intrinsics".into();
        r.msg = "intrinsic frexp = 3".into();
        (r.read, r.stats.free, r.main, r.peak_rss_kb) = (0.25, 0.5, 0.75, 7);
        let back = Report::from_line(&r.line()).unwrap();
        assert_eq!((back.outcome, back.file.as_str(), back.owner.as_str(), back.msg.as_str()), (Outcome::Unsupported, "a b.hlsl", "intrinsics", "intrinsic frexp = 3"));
        assert_eq!((back.read, back.stats.free, back.main, back.peak_rss_kb), (0.25, 0.5, 0.75, 7));
    }

    #[test]
    fn panic_owners() {
        assert_eq!(owner_of_path("src/hlsl/lower/expr.rs"), "lower");
        assert_eq!(owner_of_path("src\\container\\psv.rs"), "container");
        assert_eq!(owner_of_path("src/ir.rs"), "lead");
    }
}
