//! The lowering's lane-uniformity check (`xdxc --lane-check`, `XCL_DXIL_LANE_CHECK=2`) over the
//! translator's output. The lowering matches DXC only where lanes leave every loop around a wave
//! operation together, which the translator guarantees per console wave; the check is too slow to
//! run in the game, so it runs here, and a translator change that breaks the guarantee fails this
//! test instead of a game run.
//!
//! Every file dumped from the GPU test suite (`tests`, `tests-v2`, and any later `tests-*` folder)
//! must pass the check. In `synthetic`, the files in [`DIVERGENT`] exit a loop on a per-lane value
//! by design and must be refused; any other refused file fails with the chain the check prints.
//!
//! The programs are not in the repository: the corpus is `XCL_DXIL_CORPUS`, or
//! `~/.capy/work/dxil/corpus`. Without it, or for an absent folder, the tests pass with a note.

use std::path::PathBuf;
use std::process::Command;

/// Synthetic files (name up to the first `.`) whose loop exit depends on a per-lane value by
/// design. A name that is not in the folder is skipped.
const DIVERGENT: [&str; 2] = ["thr_continue", "thr_while_flag"];

fn root() -> PathBuf {
    match std::env::var_os("XCL_DXIL_CORPUS") {
        Some(c) => PathBuf::from(c),
        None => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".capy/work/dxil/corpus"),
    }
}

/// `xdxc --batch --lane-check` over one corpus folder: (file name, outcome, message) per file, or
/// None (with a note) when the folder holds no `.hlsl`.
fn lane_check(corpus: &str) -> Option<Vec<(String, String, String)>> {
    let dir = root().join(corpus);
    let has_hlsl = std::fs::read_dir(&dir).is_ok_and(|d| d.flatten().any(|e| e.path().extension().is_some_and(|x| x == "hlsl")));
    if !has_hlsl {
        eprintln!("no .hlsl under {}: skipped", dir.display());
        return None;
    }
    let tsv = std::env::temp_dir().join(format!("xcl-dxil-lanes-{}-{corpus}.tsv", std::process::id()));
    let out = Command::new(env!("CARGO_BIN_EXE_xdxc"))
        .arg("--batch")
        .arg(&dir)
        .args(["--lane-check", "-T", "cs_6_6", "--tsv"])
        .arg(&tsv)
        .output()
        .expect("xdxc runs");
    let text = std::fs::read_to_string(&tsv)
        .unwrap_or_else(|e| panic!("xdxc --batch wrote no {}: {e}\n{}", tsv.display(), String::from_utf8_lossy(&out.stderr)));
    let _ = std::fs::remove_file(&tsv);
    let rows: Vec<(String, String, String)> = text
        .lines()
        .map(|l| {
            let field = |k: &str| l.split('\t').find_map(|f| f.strip_prefix(k)).unwrap_or("").to_string();
            let file = field("file=");
            let name = file.rsplit(['/', '\\']).next().unwrap_or(&file).to_string();
            (name, field("outcome="), l.split_once("\tmsg=").map_or(String::new(), |(_, m)| m.to_string()))
        })
        .collect();
    assert!(!rows.is_empty(), "xdxc --batch --lane-check gave no result for corpus/{corpus}");
    Some(rows)
}

fn refused(rows: &[(String, String, String)]) -> Vec<String> {
    rows.iter().filter(|r| r.1 == "lane-refused").map(|r| format!("{}: {}", r.0, r.2)).collect()
}

/// The GPU test suite's sources, every dump of them (`tests`, `tests-v2`, `tests-v3`, ...).
#[test]
fn corpus_lanes_dumps() {
    let mut dumps: Vec<String> = std::fs::read_dir(root())
        .map(|d| d.flatten().filter(|e| e.path().is_dir()).filter_map(|e| e.file_name().into_string().ok()).collect())
        .unwrap_or_default();
    dumps.retain(|d| d == "tests" || d.starts_with("tests-"));
    dumps.sort();
    if dumps.is_empty() {
        eprintln!("no tests folder under {}: skipped", root().display());
    }
    let mut failed = Vec::new();
    for corpus in dumps {
        let Some(rows) = lane_check(&corpus) else { continue };
        let bad = refused(&rows);
        eprintln!("corpus/{corpus}: {} of {} files refused by the lane check", bad.len(), rows.len());
        failed.extend(bad.into_iter().map(|b| format!("{corpus}/{b}")));
    }
    assert!(failed.is_empty(), "the lane check refuses {} translator-derived files:\n{}", failed.len(), failed.join("\n"));
}

/// Shapes the translator is about to emit, and the variations with a per-lane loop exit.
#[test]
fn corpus_lanes_synthetic() {
    let Some(rows) = lane_check("synthetic") else { return };
    let mut failed = Vec::new();
    for (name, outcome, msg) in &rows {
        let divergent = DIVERGENT.contains(&name.split('.').next().unwrap_or(name));
        match (divergent, outcome == "lane-refused") {
            (true, false) => failed.push(format!("{name}: diverges by design but is not refused ({outcome})")),
            (false, true) => failed.push(format!("{name}: refused: {msg}")),
            _ => {}
        }
    }
    eprintln!("corpus/synthetic: {} of {} files refused by the lane check", refused(&rows).len(), rows.len());
    failed.sort();
    assert!(failed.is_empty(), "{} of {} files of corpus/synthetic:\n{}", failed.len(), rows.len(), failed.join("\n"));
}
