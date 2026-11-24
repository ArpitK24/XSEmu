//! A development aid: XCL_LOWER_SRC=file.hlsl [XCL_LOWER_PROFILE=cs_6_6] [XCL_LOWER_PRINT=1]
//! cargo test --release --test lower_dump -- --nocapture
//! lowers the file and prints timings, counts of what the writer would keep, and optionally
//! the IR. Without the variable it does nothing.

use xcl_dxil::hlsl;
use xcl_dxil::ir::Op;

#[test]
fn dump() {
    let Ok(path) = std::env::var("XCL_LOWER_SRC") else { return };
    let profile = std::env::var("XCL_LOWER_PROFILE").unwrap_or_else(|_| "cs_6_6".into());
    let src = std::fs::read_to_string(&path).expect("read");
    let t = std::time::Instant::now();
    let unit = hlsl::parse::parse(&src).expect("parse");
    let parse = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let (m, info) = match hlsl::lower::lower(&unit, "main", &profile) {
        Ok(x) => x,
        Err(e) => {
            println!("{path}: {e}");
            return;
        }
    };
    let lower = t.elapsed().as_secs_f64();
    let body = m.funcs.iter().find_map(|f| f.body.as_ref()).expect("body");
    let live = xcl_dxil::bcwriter::liveness(body);
    let mut counts = std::collections::BTreeMap::new();
    let (mut n, mut blocks) = (0, 0);
    for (bi, b) in body.blocks.iter().enumerate() {
        if !live.blocks[bi] {
            continue;
        }
        blocks += 1;
        for &i in b.phis.iter().chain(&b.insts) {
            if live.insts[i as usize] {
                n += 1;
                let name = match body.insts[i as usize].op {
                    Op::Bin(_) => "bin",
                    Op::Cast(_) => "cast",
                    Op::ICmp(_) | Op::FCmp(_) => "cmp",
                    Op::Select => "select",
                    Op::ExtractValue => "extract",
                    Op::Call => "call",
                    Op::Load => "load",
                    Op::Store => "store",
                    Op::Gep => "gep",
                    Op::Alloca => "alloca",
                    Op::Phi => "phi",
                    Op::AtomicRmw(_) | Op::CmpXchg => "atomic",
                    Op::Br | Op::CondBr => "br",
                    Op::Switch => "switch",
                    Op::Ret | Op::Unreachable => "ret",
                };
                *counts.entry(name).or_insert(0usize) += 1;
            }
        }
    }
    println!("{path}: parse {parse:.3} s, lower {lower:.3} s; {} instructions emitted, {n} live in {blocks} blocks (of {}), {} resources", body.insts.len(), body.blocks.len(), info.resources.len());
    println!("{counts:?}");
    if std::env::var("XCL_LOWER_PRINT").is_ok() {
        println!("{}", xcl_dxil::irprint::function(&m, body));
    }
}

/// XCL_LOWER_DIR=dir: lower every *.hlsl there (profile from the "// xdxc: -E main -T x" line)
/// and print what failed, grouped by message.
#[test]
fn sweep() {
    let Ok(dir) = std::env::var("XCL_LOWER_DIR") else { return };
    let mut by_msg: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    let (mut ok, mut total, mut time) = (0, 0, 0.0);
    let mut files: Vec<_> = std::fs::read_dir(&dir).expect("dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "hlsl")).collect();
    files.sort();
    for p in files {
        let src = std::fs::read_to_string(&p).expect("read");
        let profile = src.lines().next().and_then(|l| l.split("-T ").nth(1)).map(|s| s.trim().to_string()).unwrap_or_else(|| "cs_6_6".into());
        total += 1;
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let t = std::time::Instant::now();
        let r = hlsl::parse::parse(&src).and_then(|u| hlsl::lower::lower(&u, "main", &profile));
        time += t.elapsed().as_secs_f64();
        match r {
            Ok(_) => ok += 1,
            Err(e) => {
                let mut msg = e.to_string();
                if let Some(rest) = msg.strip_prefix("line ") {
                    msg = rest.split_once(": ").map(|x| x.1.to_string()).unwrap_or(msg.clone());
                }
                by_msg.entry(msg).or_default().push(name);
            }
        }
    }
    println!("{ok} of {total} lowered in {time:.3} s");
    for (m, f) in by_msg {
        println!("{:4}  {m}   e.g. {}", f.len(), f[0]);
    }
}
