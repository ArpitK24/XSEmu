//! The parser on real translator output. The programs are not in the repository: they are
//! looked for in the folder `XCL_DXIL_CORPUS` names, or in `~/.capy/work/dxil/corpus` when the
//! variable is not set (`.hlsl` files, searched recursively). Where the folder does not exist
//! these tests pass with a note.

use std::path::{Path, PathBuf};
use std::time::Instant;

use xcl_dxil::hlsl::{astprint, parse};

fn corpus_root() -> Option<PathBuf> {
    let root = match std::env::var_os("XCL_DXIL_CORPUS") {
        Some(root) => PathBuf::from(root),
        None => PathBuf::from(std::env::var_os("HOME")?).join(".capy/work/dxil/corpus"),
    };
    if !root.is_dir() {
        eprintln!("no corpus at {}: skipped", root.display());
        return None;
    }
    Some(root)
}

fn hlsl_files(root: &Path) -> Vec<PathBuf> {
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
    walk(root, &mut files);
    files.sort();
    files
}

fn corpus() -> Option<Vec<PathBuf>> {
    Some(hlsl_files(&corpus_root()?))
}

/// Every program of the translator's test suite (the `tests` folder of the corpus) parses,
/// those DXC rejects for what they mean (a `.dxc-error.txt` beside them) included. Each
/// program that does not is reported with its error.
#[test]
fn test_suite_parses() {
    let Some(root) = corpus_root() else { return };
    let dir = root.join("tests");
    if !dir.is_dir() {
        eprintln!("no corpus at {}: skipped", dir.display());
        return;
    }
    let files = hlsl_files(&dir);
    assert!(!files.is_empty(), "no .hlsl files in {}", dir.display());
    let mut failed = Vec::new();
    for f in &files {
        let name = f.strip_prefix(&dir).unwrap().display();
        match std::fs::read_to_string(f) {
            Ok(src) => {
                if let Err(e) = parse::parse(&src) {
                    failed.push(format!("{name}: {e}"));
                }
            }
            Err(e) => failed.push(format!("{name}: cannot be read: {e}")),
        }
    }
    eprintln!("{}: {} of {} programs parse", dir.display(), files.len() - failed.len(), files.len());
    assert!(failed.is_empty(), "{} of {} programs in {} do not parse:\n{}", failed.len(), files.len(), dir.display(), failed.join("\n"));
}

#[test]
fn corpus_parses() {
    let Some(files) = corpus() else { return };
    let mut failed = Vec::new();
    let (mut bytes, mut seconds) = (0usize, 0f64);
    for f in &files {
        let src = std::fs::read_to_string(f).unwrap();
        let t = Instant::now();
        let r = parse::parse(&src);
        seconds += t.elapsed().as_secs_f64();
        bytes += src.len();
        if let Err(e) = r {
            failed.push(format!("{}: {e}", f.display()));
        }
    }
    eprintln!("{} files, {:.1} MB, {:.0} MB/s", files.len(), bytes as f64 / 1e6, bytes as f64 / 1e6 / seconds.max(1e-9));
    assert!(failed.is_empty(), "{} of {} files do not parse:\n{}", failed.len(), files.len(), failed.join("\n"));
}

/// parse(print(parse(x))) is the tree of parse(x), for every program of the corpus.
#[test]
fn corpus_round_trips() {
    let Some(files) = corpus() else { return };
    let mut failed = Vec::new();
    for f in &files {
        let src = std::fs::read_to_string(f).unwrap();
        // the big ones are covered by the release-mode speed test's sibling below
        if src.len() > 2_000_000 && cfg!(debug_assertions) && std::env::var_os("XCL_DXIL_CORPUS_ALL").is_none() {
            continue;
        }
        let Ok(a) = parse::parse(&src) else { continue };
        let printed = astprint::unit(&a);
        match parse::parse(&printed) {
            Ok(b) => {
                if astprint::tree(&a) != astprint::tree(&b) {
                    failed.push(format!("{}: the printed text parses to another tree", f.display()));
                }
            }
            Err(e) => failed.push(format!("{}: the printed text does not parse: {e}", f.display())),
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Lexing + parsing speed on the largest program of the corpus (ten runs) and on a 30 MB
/// source made of its entry point's body repeated. `cargo test --release -- --ignored speed`
#[test]
#[ignore]
fn speed() {
    let Some(files) = corpus() else { return };
    let Some(big) = files.iter().max_by_key(|f| std::fs::metadata(f).map_or(0, |m| m.len())) else { return };
    let src = std::fs::read_to_string(big).unwrap();
    let mb = |n: usize| n as f64 / 1e6;
    let mut best = f64::MAX;
    let t = Instant::now();
    for _ in 0..10 {
        let t = Instant::now();
        let u = parse::parse(&src).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
        std::hint::black_box(&u);
    }
    let total = t.elapsed().as_secs_f64();
    let u = parse::parse(&src).unwrap();
    println!(
        "{:.1} MB x 10: {:.0} MB/s mean (with freeing the trees), {:.0} MB/s best; {} exprs, {} stmts, {} vars, {} names",
        mb(src.len()),
        mb(src.len()) * 10.0 / total,
        mb(src.len()) / best,
        u.exprs.len(),
        u.stmts.len(),
        u.vars.len(),
        u.syms.len()
    );
    drop(u);

    // the body of the last function (the entry point), repeated as nested blocks
    let open = src.rfind("\nvoid main(").and_then(|m| src[m..].find('{').map(|b| m + b + 1));
    let (Some(open), Some(close)) = (open, src.rfind('}')) else {
        println!("no entry point found: no synthetic source");
        return;
    };
    let body = &src[open..close];
    let copies = (30_000_000 / body.len()).max(1);
    for rename in [false, true] {
        let mut big = String::with_capacity(src.len() + copies * (body.len() + 16));
        big.push_str(&src[..open]);
        for k in 0..copies {
            big.push_str("{\n");
            if rename {
                // every copy with its own temporaries (`t123` becomes `t123_k`), as a real
                // program of this size has: the names do not repeat
                let b = body.as_bytes();
                let mut i = 0;
                let mut from = 0;
                while i < b.len() {
                    let word_start = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
                    if word_start && b[i] == b't' && b.get(i + 1).is_some_and(|c| c.is_ascii_digit()) {
                        let mut j = i + 1;
                        while b.get(j).is_some_and(|c| c.is_ascii_digit()) {
                            j += 1;
                        }
                        big.push_str(&body[from..j]);
                        big.push_str(&format!("_{k}"));
                        from = j;
                        i = j;
                    } else {
                        i += 1;
                    }
                }
                big.push_str(&body[from..]);
            } else {
                big.push_str(body);
            }
            big.push_str("}\n");
        }
        big.push_str("}\n");
        let mut best = f64::MAX;
        let mut names = 0;
        for _ in 0..3 {
            let t = Instant::now();
            let u = parse::parse(&big).unwrap();
            best = best.min(t.elapsed().as_secs_f64());
            names = u.syms.len();
        }
        println!(
            "synthetic {:.1} MB ({copies} copies of the body{}): {:.0} MB/s, {:.0} ms, {names} names",
            mb(big.len()),
            if rename { ", temporaries renamed per copy" } else { "" },
            mb(big.len()) / best,
            best * 1e3
        );
    }
}

#[test]
#[ignore]
fn lex_only() {
    let Some(files) = corpus() else { return };
    let Some(big) = files.iter().max_by_key(|f| std::fs::metadata(f).map_or(0, |m| m.len())) else { return };
    let src = std::fs::read_to_string(big).unwrap();
    for _ in 0..5 {
        let t = Instant::now();
        let mut syms = xcl_dxil::hlsl::ast::Interner::with_capacity(src.len() / 128 + 1024);
        let mut lx = xcl_dxil::hlsl::lex::Lexer::new(&src, &mut syms);
        let mut n = 0usize;
        loop {
            let t = lx.next(&mut syms);
            if matches!(t.kind, xcl_dxil::hlsl::lex::Tok::Eof | xcl_dxil::hlsl::lex::Tok::Err) {
                break;
            }
            n += 1;
        }
        let s = t.elapsed().as_secs_f64();
        println!("lex only: {n} tokens, {:.0} MB/s", src.len() as f64 / 1e6 / s);
        let t = Instant::now();
        let u = parse::parse(&src).unwrap();
        let s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        drop(u);
        println!("parse: {:.0} MB/s, drop {:.1} ms", src.len() as f64 / 1e6 / s, t.elapsed().as_secs_f64() * 1e3);
    }
}

/// Writes the printed form of every corpus program to `XCL_DXIL_PRINT_TO` (for comparing DXC's
/// output for the original and the printed text by hand).
#[test]
#[ignore]
fn print_corpus() {
    let (Some(files), Some(to)) = (corpus(), std::env::var_os("XCL_DXIL_PRINT_TO")) else { return };
    std::fs::create_dir_all(&to).unwrap();
    for f in &files {
        let src = std::fs::read_to_string(f).unwrap();
        if let Ok(u) = parse::parse(&src) {
            std::fs::write(Path::new(&to).join(f.file_name().unwrap()), astprint::unit(&u)).unwrap();
        }
    }
}
