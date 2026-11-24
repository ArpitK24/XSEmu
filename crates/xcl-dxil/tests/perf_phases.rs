//! Where the time of one compile goes, phase by phase, for a program on disk: the tool to check
//! the one-second budget on a real program, on the machine it has to hold on. Ignored by
//! default; run with
//!
//! ```text
//! XCL_DXIL_PERF_FILE=big.hlsl cargo test --release --test perf_phases -- --ignored --nocapture
//! ```
//!
//! `XCL_DXIL_PERF_PROFILE` (default `cs_6_6`), `XCL_DXIL_PERF_ENTRY` (default `main`) and
//! `XCL_DXIL_PERF_RUNS` (default 5) say how. Each run goes through the same steps as
//! [`xcl_dxil::compile_timed`], timed one by one, and has to give the container
//! `compile_timed` gives; the table shows the best time of each step over the runs and the
//! run with the best total. The last lines are the sizes that decide the memory a compile
//! touches (every fresh page is a fault: on Linux they are a sixth of the wall clock).

use std::time::Instant;

use xcl_dxil::{bcwriter, container, hlsl, ir, meta};

const STEPS: [&str; 10] = ["parse", "lower", "seal (liveness)", "flags, view id", "metadata", "write bitcode", "container", "drop syntax tree", "drop the rest", "total"];

#[test]
#[ignore = "needs XCL_DXIL_PERF_FILE"]
fn time_per_phase() {
    let Some(path) = std::env::var_os("XCL_DXIL_PERF_FILE") else {
        eprintln!("XCL_DXIL_PERF_FILE is not set: nothing to time");
        return;
    };
    let var = |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.to_string());
    let (profile, entry) = (var("XCL_DXIL_PERF_PROFILE", "cs_6_6"), var("XCL_DXIL_PERF_ENTRY", "main"));
    let runs: usize = var("XCL_DXIL_PERF_RUNS", "5").parse().unwrap_or(5).max(1);
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.to_string_lossy()));
    let (expected, stats) = xcl_dxil::compile_timed(&src, &entry, &profile).unwrap_or_else(|e| panic!("compile: {e}"));

    let mut best = [f64::INFINITY; STEPS.len()];
    let mut best_run = [f64::INFINITY; STEPS.len()];
    let mut sizes = String::new();
    for _ in 0..runs {
        let mut t = [0f64; STEPS.len()];
        let start = Instant::now();
        let mut at = start;
        let mut lap = |k: usize| {
            let now = Instant::now();
            t[k] = (now - at).as_secs_f64();
            at = now;
        };
        let unit = hlsl::parse::parse(&src).unwrap();
        lap(0);
        let (module, mut info) = hlsl::lower::lower(&unit, &entry, &profile).unwrap();
        lap(1);
        let mut module = ir::Sealed::new(module).unwrap();
        lap(2);
        meta::prepare_sealed(&module, &mut info);
        lap(3);
        module.extend(|m| meta::emit(m, &info)).unwrap();
        lap(4);
        let (c, counts) = bcwriter::write_sealed_into(&module, container::open(&info).unwrap()).unwrap();
        lap(5);
        let c = container::close(c).unwrap();
        lap(6);
        if sizes.is_empty() {
            let m = module.module();
            let (insts, listed): (usize, usize) = m.funcs.iter().filter_map(|f| f.body.as_ref()).fold((0, 0), |(a, b), body| (a + body.insts.len(), b + body.blocks.iter().map(|x| x.phis.len() + x.insts.len()).sum::<usize>()));
            sizes = format!(
                "source {} bytes; syntax tree: {} expressions of {} bytes, {} statements of {} bytes, {} names\ncode: {} instructions of {} bytes built, {} in blocks, {} written in {} blocks; {} constants, {} types\nbitcode {} bytes, container {} bytes",
                src.len(),
                unit.exprs.len(),
                std::mem::size_of::<hlsl::ast::Expr>(),
                unit.stmts.len(),
                std::mem::size_of::<hlsl::ast::Stmt>(),
                unit.syms.len(),
                insts,
                std::mem::size_of::<ir::Inst>(),
                listed,
                counts.instructions,
                counts.blocks,
                m.consts.len(),
                m.types.len(),
                bitcode_len(&c),
                c.len()
            );
        }
        if std::env::var_os("XCL_DXIL_PERF_DEAD").is_some() && !sizes.contains("dead") {
            sizes.push_str(&dead_summary(&module));
        }
        assert!(c == expected, "the steps give another container than compile_timed");
        drop(unit);
        lap(7);
        drop((module, c));
        lap(8);
        t[9] = start.elapsed().as_secs_f64();
        for k in 0..STEPS.len() {
            best[k] = best[k].min(t[k]);
        }
        if t[9] < best_run[9] {
            best_run = t;
        }
    }
    println!("{}: {profile}, {runs} runs; seconds", path.to_string_lossy());
    println!("{:<18} {:>9} {:>9}", "step", "best", "best run");
    for k in 0..STEPS.len() {
        println!("{:<18} {:>9.4} {:>9.4}", STEPS[k], best[k], best_run[k]);
    }
    println!("compile_timed: parse {:.4}, lower (with seal, flags, metadata) {:.4}, write {:.4}, container {:.4}, free {:.4}", stats.parse, stats.lower, stats.write, stats.container, stats.free);
    println!("{sizes}");
}

/// `XCL_DXIL_PERF_DEAD=1`: what the instructions are that the lowering built and the writer
/// drops (by operation, and for a phi by its number of incoming values), and how many phis have
/// how many incoming values: every dead instruction costs its 20 bytes and a visit in each pass.
fn dead_summary(module: &ir::Sealed) -> String {
    use std::collections::BTreeMap;
    let mut dead: BTreeMap<String, usize> = BTreeMap::new();
    let mut incoming: BTreeMap<usize, usize> = BTreeMap::new();
    let m = module.module();
    for (f, live) in m.funcs.iter().zip(module.lives()) {
        let (Some(body), Some(live)) = (&f.body, live) else { continue };
        for (i, inst) in body.insts.iter().enumerate() {
            if inst.op == ir::Op::Phi {
                *incoming.entry(body.phi_incoming(ir::Value::inst(i as u32)).len().min(5)).or_default() += 1;
            }
            if live.insts[i] {
                continue;
            }
            let what = match inst.op {
                ir::Op::Call => format!("call {}", m.function(inst.a).name),
                ir::Op::Bin(_) => "binary".to_string(),
                ir::Op::Cast(_) => "cast".to_string(),
                ir::Op::ICmp(_) | ir::Op::FCmp(_) => "compare".to_string(),
                op => format!("{op:?}").to_lowercase(),
            };
            *dead.entry(what).or_default() += 1;
        }
    }
    let mut dead: Vec<(String, usize)> = dead.into_iter().collect();
    dead.sort_by(|a, b| b.1.cmp(&a.1));
    let dead: Vec<String> = dead.iter().take(12).map(|(what, n)| format!("{what} {n}")).collect();
    format!("\ndead instructions: {}\nphis by incoming values (5: five or more): {incoming:?}", dead.join(", "))
}

/// The length of the bitcode in a container: the size field of the DXIL part's program header.
fn bitcode_len(c: &[u8]) -> usize {
    let word = |at: usize| u32::from_le_bytes([c[at], c[at + 1], c[at + 2], c[at + 3]]) as usize;
    let parts = word(28);
    let dxil = word(32 + 4 * (parts - 1));
    word(dxil + 28)
}

/// The passes that read the whole module, each timed alone over and over on the module of one
/// compile (`XCL_DXIL_PERF_FILE`, `XCL_DXIL_PERF_RUNS` repetitions, default 9; best and mean):
/// a change to one of them shows here above the noise of whole compiles.
#[test]
#[ignore = "needs XCL_DXIL_PERF_FILE"]
fn time_passes() {
    let Some(path) = std::env::var_os("XCL_DXIL_PERF_FILE") else {
        eprintln!("XCL_DXIL_PERF_FILE is not set: nothing to time");
        return;
    };
    let var = |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.to_string());
    let (profile, entry) = (var("XCL_DXIL_PERF_PROFILE", "cs_6_6"), var("XCL_DXIL_PERF_ENTRY", "main"));
    let runs: usize = var("XCL_DXIL_PERF_RUNS", "9").parse().unwrap_or(9).max(1);
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.to_string_lossy()));
    let time = |name: &str, f: &mut dyn FnMut()| {
        let mut all = Vec::with_capacity(runs);
        for _ in 0..runs {
            let t = Instant::now();
            f();
            all.push(t.elapsed().as_secs_f64());
        }
        let best = all.iter().copied().fold(f64::INFINITY, f64::min);
        println!("{name:<18} best {best:.4}  mean {:.4}", all.iter().sum::<f64>() / runs as f64);
    };
    time("parse", &mut || drop(hlsl::parse::parse(&src).unwrap()));
    let unit = hlsl::parse::parse(&src).unwrap();
    let (module, mut info) = hlsl::lower::lower(&unit, &entry, &profile).unwrap();
    drop(unit);
    let body = module.funcs.iter().find_map(|f| f.body.as_ref()).unwrap();
    time("liveness", &mut || drop(bcwriter::liveness(body)));
    let mut module = ir::Sealed::new(module).unwrap();
    time("flags, view id", &mut || {
        let mut i = info.clone();
        meta::prepare_sealed(&module, &mut i);
    });
    meta::prepare_sealed(&module, &mut info);
    module.extend(|m| meta::emit(m, &info)).unwrap();
    time("write bitcode", &mut || drop(bcwriter::write_sealed(&module).unwrap()));
    let open = container::open(&info).unwrap();
    let (c, _) = bcwriter::write_sealed_into(&module, open.clone()).unwrap();
    time("container", &mut || drop(container::close(c.clone()).unwrap()));
}
