//! The whole pipeline on the HLSL corpora: every program either becomes a container the DXIL
//! validator accepts, or is [`Error::Unsupported`] (the caller falls back to DXC). A panic, an
//! internal error, a source error on a program DXC accepts, or a container `dxv` rejects fails
//! the test, and so does compiling a program DXC rejects (a `<name>.dxc-error.txt` beside it).
//!
//! The programs are not in the repository. The tests run when `XCL_DXIL_DXC` names a DXC folder
//! (`bin/dxv`, `lib/`) and the corpus exists: `XCL_DXIL_CORPUS`, or `~/.capy/work/dxil/corpus`
//! with one folder per corpus (`tests` and `tests-v2`: dumped from the GPU test suite,
//! `game-sample`, `synthetic`: shapes the translator is about to emit, and its whole prelude with
//! every function called). Without either, or for a folder that is absent, they pass with a note.
//! `tools/corpus-run.sh` is the same run with the comparison against DXC and the report.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use xcl_dxil::Error;

/// The DXC folder and the `.hlsl` files of one corpus folder, or None (with a note) when either is
/// missing.
fn setup(corpus: &str) -> Option<(PathBuf, Vec<PathBuf>)> {
    let Some(dxc) = std::env::var_os("XCL_DXIL_DXC").map(PathBuf::from) else {
        eprintln!("XCL_DXIL_DXC is not set: skipped");
        return None;
    };
    let root = match std::env::var_os("XCL_DXIL_CORPUS") {
        Some(c) => PathBuf::from(c),
        None => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".capy/work/dxil/corpus"),
    };
    let dir = root.join(corpus);
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
    walk(&dir, &mut files);
    files.sort();
    if files.is_empty() {
        eprintln!("no .hlsl under {}: skipped", dir.display());
        return None;
    }
    Some((dxc, files))
}

/// Entry point and profile: the first line `// xdxc: -E main -T cs_6_6` the dump writes, else a
/// `<name>.<profile>.hlsl` file name, else a compute shader.
fn entry_and_profile(path: &Path, src: &str) -> (String, String) {
    let (mut entry, mut profile) = ("main".to_string(), None);
    if let Some(args) = src.lines().next().and_then(|l| l.strip_prefix("// xdxc:")) {
        let mut words = args.split_whitespace();
        while let Some(w) = words.next() {
            match w {
                "-E" => entry = words.next().unwrap_or("main").to_string(),
                "-T" => profile = words.next().map(str::to_string),
                _ => {}
            }
        }
    }
    let from_name = || {
        let name = path.file_name()?.to_str()?.strip_suffix(".hlsl")?;
        let p = name.rsplit_once('.')?.1;
        (p.len() == 6 && p[2..].starts_with("_6_")).then(|| p.to_string())
    };
    (entry, profile.or_else(from_name).unwrap_or_else(|| "cs_6_6".to_string()))
}

/// None when the file is fine, else what is wrong with it.
fn check(dxc: &Path, tmp: &Path, path: &Path) -> Option<String> {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => return Some(e.to_string()),
    };
    let (entry, profile) = entry_and_profile(path, &src);
    let dxc_rejects = path.with_extension("dxc-error.txt").exists();
    let container = match catch_unwind(AssertUnwindSafe(|| xcl_dxil::compile(&src, &entry, &profile))) {
        Err(panic) => {
            let msg = panic.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| panic.downcast_ref::<String>().cloned());
            return Some(format!("panic: {}", msg.unwrap_or_default()));
        }
        Ok(Err(Error::Unsupported(_))) => return None,
        Ok(Err(Error::Source { .. })) if dxc_rejects => return None,
        Ok(Err(e)) => return Some(e.to_string()),
        Ok(Ok(_)) if dxc_rejects => return Some("compiles, but DXC rejects this source".to_string()),
        Ok(Ok(c)) => c,
    };
    let out = tmp.join(path.file_name().unwrap_or_default()).with_extension("dxil");
    if let Err(e) = std::fs::write(&out, &container) {
        return Some(format!("{}: {e}", out.display()));
    }
    let run = Command::new(dxc.join("bin").join("dxv")).arg(&out).env("LD_LIBRARY_PATH", dxc.join("lib")).output();
    let _ = std::fs::remove_file(&out);
    match run {
        Err(e) => Some(format!("cannot run dxv: {e}")),
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout).into_owned() + &String::from_utf8_lossy(&o.stderr);
            if o.status.success() && text.contains("Validation succeeded") {
                return None;
            }
            let first = text.lines().map(str::trim).find(|l| l.contains("error") && *l != "error: validation errors");
            Some(format!("dxv: {}", first.unwrap_or("validation failed")))
        }
    }
}

fn run(corpus: &str) {
    let Some((dxc, files)) = setup(corpus) else { return };
    let tmp = std::env::temp_dir().join(format!("xcl-dxil-corpus-{}-{corpus}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("temp folder");
    let next = AtomicUsize::new(0);
    let failed = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(1, |n| n.get()) {
            s.spawn(|| loop {
                let Some(path) = files.get(next.fetch_add(1, Ordering::Relaxed)) else { break };
                if let Some(why) = check(&dxc, &tmp, path) {
                    // the name only: a program's text or path beyond it never goes into a log
                    let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
                    failed.lock().unwrap().push(format!("{name}: {why}"));
                }
            });
        }
    });
    let _ = std::fs::remove_dir_all(&tmp);
    let mut failed = failed.into_inner().unwrap();
    failed.sort();
    assert!(failed.is_empty(), "{} of {} files of corpus/{corpus}:\n{}", failed.len(), files.len(), failed.join("\n"));
}

/// The sources of the GPU test suite, every stage.
#[test]
fn corpus_tests() {
    run("tests");
}

/// The same suite dumped with the translator's newer output (heap handles in locals, window
/// accessors, no comments); absent until that dump exists.
#[test]
fn corpus_tests_v2() {
    run("tests-v2");
}

/// Two large real compute programs.
#[test]
fn corpus_game_sample() {
    run("game-sample");
}

/// Shapes the translator is about to emit (resource-typed locals across control flow, ...).
#[test]
fn corpus_synthetic() {
    run("synthetic");
}
