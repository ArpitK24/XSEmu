//! `XCL_TITLE_CORPUS=<folder of dumped console shaders (*_cs.bin, *_ps.bin, *_vs.bin)>`: every
//! shader is built the way a run on a GPU with 32-lane waves builds it (4 memory windows, split
//! wave64, the configured DXIL backend with its DXC fallback). One line per shader as it
//! finishes, so a run that is stopped early still counts what it got through. The folder holds
//! title data and stays on the machine with the title; without it the test does nothing.
//! `XCL_DXIL_CACHE` must name a scratch folder, so the run never writes the emulator's caches.
//! `XCL_TITLE_CORPUS_JOBS`: shaders built at once (default 4).

use super::recomp::pso::{console_to_dxil_in, Stage};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

#[test]
fn title_shaders_build() {
    let Some(dir) = std::env::var_os("XCL_TITLE_CORPUS") else {
        println!("XCL_TITLE_CORPUS not set");
        return;
    };
    assert!(std::env::var_os("XCL_DXIL_CACHE").is_some(), "set XCL_DXIL_CACHE to a scratch folder: this test must not write the emulator's DXIL cache");
    let mut files: Vec<(std::path::PathBuf, String, Stage)> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let stage = match name.rsplit_once('_')?.1 {
                "cs.bin" => Stage::Compute,
                "ps.bin" => Stage::Pixel,
                "vs.bin" => Stage::Vertex,
                _ => return None,
            };
            Some((e.path(), name, stage))
        })
        .collect();
    files.sort_by(|a, b| a.1.cmp(&b.1));
    assert!(!files.is_empty(), "no console shaders in {dir:?}");
    let jobs: usize = std::env::var("XCL_TITLE_CORPUS_JOBS").ok().and_then(|v| v.parse().ok()).unwrap_or(4).max(1);
    println!("title-corpus: {} shaders, {jobs} at once", files.len());
    let next = AtomicUsize::new(0);
    let failed = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..jobs.min(files.len()) {
            s.spawn(|| loop {
                let Some((path, name, stage)) = files.get(next.fetch_add(1, Ordering::Relaxed)) else { break };
                let t0 = Instant::now();
                let r = std::fs::read(path).map_err(|e| e.to_string()).and_then(|blob| console_to_dxil_in(&blob, *stage, 0, true, 4));
                let secs = t0.elapsed().as_secs_f64();
                let line = match &r {
                    Ok(c) => format!("title-corpus: {name} ok {} KB DXIL {secs:.1} s", c.dxil.len() >> 10),
                    Err(e) => {
                        let first = e.lines().next().unwrap_or_default().to_string();
                        failed.lock().unwrap().push(format!("{name}: {first}"));
                        format!("title-corpus: {name} FAILED {secs:.1} s: {first}")
                    }
                };
                let mut out = std::io::stdout().lock();
                let _ = writeln!(out, "{line}");
                let _ = out.flush();
            });
        }
    });
    let failed = failed.into_inner().unwrap();
    println!("title-corpus: {} built, {} failed", files.len() - failed.len(), failed.len());
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
