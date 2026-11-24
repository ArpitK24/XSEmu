//! The interpreter's semantics on small programs (owner: the interp subagent): wave
//! operations in diverged control flow and loops, the group barrier and atomics across the
//! waves of a group, what a run reports as undefined, and the inputs, outputs, derivatives
//! and discard of vertex and pixel programs.
//!
//! Each program is compiled by this crate. With the oracle (the XCL_DXIL_DXC environment
//! variable names DXC's folder, as in the brief) it is also compiled by DXC at -O0 and at
//! -O1, and the three containers must agree with each other: for these programs the
//! expected values are in the source semantics, so all three must give them.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use xcl_dxil::interp::{self, exec, Config, Verdict};

const HEAD: &str = "RWByteAddressBuffer res : register(u1);\n";

/// Lanes leave the loop one group of eight after another: the count taken in iteration i
/// must be that of the lanes still in the loop, and after the loop all lanes are together.
const LOOP: &str = r#"
[WaveSize(32)]
[numthreads(32, 1, 1)]
void main(uint gi : SV_GroupIndex) {
    uint n = 0u;
    uint rounds = gi & 3u;
    [loop] for (uint i = 0u; i < 4u; i++) {
        if (i > rounds) break;
        n += WaveActiveCountBits(true);
    }
    uint after = WaveActiveCountBits(true);
    res.Store(gi * 8u, n);
    res.Store(gi * 8u + 4u, after);
}
"#;

/// A wave operation on the way out of a loop belongs to the iteration it leaves in: DXC
/// keeps its block in the loop with a dx.break condition, and the count is that of the
/// eight lanes leaving together, not of all the lanes that have left.
const BREAK: &str = r#"
[WaveSize(32)]
[numthreads(32, 1, 1)]
void main(uint gi : SV_GroupIndex) {
    uint n = 0u;
    uint rounds = gi & 3u;
    [loop] for (uint i = 0u; i < 4u; i++) {
        if (i == rounds) {
            n = WaveActiveCountBits(true) + 100u * i;
            break;
        }
    }
    res.Store(gi * 4u, n);
}
"#;

/// The two sides of a branch each see their own lanes, the code after it sees all.
const BRANCH: &str = r#"
[WaveSize(32)]
[numthreads(32, 1, 1)]
void main(uint gi : SV_GroupIndex) {
    uint side;
    if (gi % 3u == 0u) { side = WaveActiveSum(gi); } else { side = WaveActiveCountBits(true); }
    uint4 all = WaveActiveBallot(true);
    uint first = WaveReadLaneFirst(gi);
    uint before = WavePrefixCountBits((gi & 1u) != 0u);
    res.Store(gi * 16u, side);
    res.Store(gi * 16u + 4u, all.x);
    res.Store(gi * 16u + 8u, first);
    res.Store(gi * 16u + 12u, before);
}
"#;

/// Two waves: every thread's write must be there for every thread's read after the barrier.
const BARRIER: &str = r#"
groupshared uint cells[64];
[WaveSize(32)]
[numthreads(64, 1, 1)]
void main(uint gi : SV_GroupIndex) {
    cells[gi] = gi + 1u;
    GroupMemoryBarrierWithGroupSync();
    res.Store(gi * 4u, cells[63u - gi]);
}
"#;

/// Two waves count themselves with an atomic: each thread gets its own old value.
const ATOMIC: &str = r#"
groupshared uint counter;
groupshared uint seen[64];
[WaveSize(32)]
[numthreads(64, 1, 1)]
void main(uint gi : SV_GroupIndex) {
    if (gi == 0u) counter = 0u;
    GroupMemoryBarrierWithGroupSync();
    uint old;
    InterlockedAdd(counter, 1u, old);
    seen[old & 63u] = 1u;
    GroupMemoryBarrierWithGroupSync();
    uint total = 0u;
    [unroll] for (uint i = 0u; i < 64u; i++) total += seen[i];
    res.Store(gi * 8u, counter);
    res.Store(gi * 8u + 4u, total);
}
"#;

/// The order of the two sides of a branch is not the program's to rely on.
const RACE: &str = r#"
groupshared uint cell;
[WaveSize(32)]
[numthreads(32, 1, 1)]
void main(uint gi : SV_GroupIndex) {
    if ((gi & 1u) != 0u) { cell = 1u; } else { cell = 2u; }
    GroupMemoryBarrierWithGroupSync();
    res.Store(gi * 4u, cell);
}
"#;

/// Pixels of a quad differ by one in x and in y, so the derivatives of SV_Position are
/// exact; the right column of every quad is discarded and keeps no output.
const PIXEL: &str = r#"
float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0, nointerpolation uint id : ID) : SV_Target {
    float steps = ddx(pos.x) + 2.0 * ddy_fine(pos.y) + 4.0 * ddx_fine(pos.x);
    float flat = ddx((float)id) + ddy(pos.x);
    if ((((uint)pos.x) & 1u) != 0u) discard;
    return float4(steps, flat, uv.x, (float)id);
}
"#;

const VERTEX: &str = r#"
struct Out { float4 pos : SV_Position; uint id : ID; };
Out main(float3 p : POSITION, uint vid : SV_VertexID) {
    Out o;
    o.pos = float4(p * 2.0, 1.0);
    o.id = vid;
    return o;
}
"#;

fn ours(body: &str) -> Vec<u8> {
    xcl_dxil::compile(&format!("{HEAD}{body}"), "main", "cs_6_6").expect("this crate compiles the program")
}

/// DXC's container at an optimisation level, when the oracle is there.
fn dxc(name: &str, body: &str, level: &str) -> Option<Vec<u8>> {
    dxc_as(name, &format!("{HEAD}{body}"), level, "cs_6_6")
}

fn dxc_as(name: &str, source: &str, level: &str, profile: &str) -> Option<Vec<u8>> {
    let dir = PathBuf::from(std::env::var_os("XCL_DXIL_DXC").filter(|v| !v.is_empty())?);
    let tmp = std::env::temp_dir().join(format!("xcl-interp-{}-{name}{level}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let (src, out) = (tmp.join("p.hlsl"), tmp.join("p.dxil"));
    std::fs::write(&src, source).unwrap();
    let exe = dir.join("bin").join(if cfg!(windows) { "dxc.exe" } else { "dxc" });
    let status = Command::new(exe)
        .env("LD_LIBRARY_PATH", dir.join("lib"))
        .args(["-E", "main", "-T", profile, "-HV", "2021", "-Gis", level, "-Fo"])
        .arg(&out)
        .arg(&src)
        .output()
        .expect("dxc runs");
    assert!(status.status.success(), "dxc {level} on {name}: {}", String::from_utf8_lossy(&status.stderr));
    let bytes = std::fs::read(&out).unwrap();
    std::fs::remove_dir_all(&tmp).ok();
    Some(bytes)
}

/// Run a program of this crate's and return the words it stored to u1 by byte offset; with
/// the oracle, DXC's two containers must first agree with it.
fn stored(name: &str, body: &str) -> (BTreeMap<u64, u32>, u32) {
    let cfg = Config::default();
    let p = interp::load(&ours(body)).unwrap();
    for level in ["-O0", "-O1"] {
        let Some(bytes) = dxc(name, body, level) else {
            eprintln!("{name}: no oracle (XCL_DXIL_DXC), DXC not compared");
            break;
        };
        let q = interp::load(&bytes).unwrap();
        for seed in 1..=4 {
            let v = interp::compare(&interp::run(&q, seed, &cfg), &interp::run(&p, seed, &cfg));
            assert!(matches!(v, Verdict::Equal) || name == "race" && matches!(v, Verdict::Undefined(_)), "{name}: DXC {level} / ours, seed {seed}: {v:?}");
        }
    }
    words(name, &p)
}

/// The words a program stores to u1 by byte offset, and the undefined behaviour it met.
fn words(name: &str, p: &interp::Program) -> (BTreeMap<u64, u32>, u32) {
    let run = interp::run(p, 1, &Config::default());
    assert!(run.stop.is_none(), "{name}: {:?}", run.stop);
    let mut words = BTreeMap::new();
    for w in &run.written {
        let text = interp::res::describe(w);
        let offset = text.strip_prefix("buffer u1 +0x").unwrap_or_else(|| panic!("{name}: wrote {text}"));
        words.insert(u64::from_str_radix(offset, 16).unwrap(), w.3[0]);
    }
    (words, run.ub)
}

#[test]
fn wave_ops_in_a_loop_see_the_lanes_of_their_iteration() {
    let (w, ub) = stored("loop", LOOP);
    assert_eq!(ub, 0);
    for gi in 0..32u64 {
        // iteration i runs for the lanes with gi & 3 >= i: 32, 24, 16, 8 of them
        let expected: u32 = (0..=(gi & 3) as u32).map(|i| 32 - 8 * i).sum();
        assert_eq!(w[&(gi * 8)], expected, "lane {gi}");
        assert_eq!(w[&(gi * 8 + 4)], 32, "lane {gi} after the loop");
    }
}

fn leaves_in_eights(w: &BTreeMap<u64, u32>) {
    for gi in 0..32u64 {
        assert_eq!(w[&(gi * 4)], 8 + 100 * (gi & 3) as u32, "lane {gi}");
    }
}

#[test]
fn dxc_keeps_a_wave_op_before_a_break_in_its_iteration() {
    for level in ["-O0", "-O1"] {
        let Some(bytes) = dxc("break", BREAK, level) else {
            eprintln!("break: no oracle (XCL_DXIL_DXC), nothing checked");
            return;
        };
        let (w, ub) = words("break", &interp::load(&bytes).unwrap());
        assert_eq!(ub, 0);
        leaves_in_eights(&w);
    }
}

#[test]
fn a_wave_op_before_a_break_sees_the_lanes_leaving_with_it() {
    let (w, ub) = stored("break", BREAK);
    assert_eq!(ub, 0);
    leaves_in_eights(&w);
}

#[test]
fn wave_ops_in_a_branch_see_their_side() {
    let (w, ub) = stored("branch", BRANCH);
    assert_eq!(ub, 0);
    let thirds: u32 = (0..32).filter(|g| g % 3 == 0).sum();
    for gi in 0..32u64 {
        assert_eq!(w[&(gi * 16)], if gi % 3 == 0 { thirds } else { 21 }, "lane {gi}");
        assert_eq!(w[&(gi * 16 + 4)], u32::MAX);
        assert_eq!(w.get(&(gi * 16 + 8)).copied().unwrap_or(0), 0);
        assert_eq!(w.get(&(gi * 16 + 12)).copied().unwrap_or(0), (gi / 2) as u32, "lane {gi}");
    }
}

#[test]
fn the_barrier_holds_every_wave() {
    let (w, ub) = stored("barrier", BARRIER);
    assert_eq!(ub, 0);
    for gi in 0..64u64 {
        assert_eq!(w[&(gi * 4)], 64 - gi as u32, "thread {gi}");
    }
}

#[test]
fn atomics_count_across_waves() {
    let (w, ub) = stored("atomic", ATOMIC);
    assert_eq!(ub, 0);
    for gi in 0..64u64 {
        assert_eq!(w[&(gi * 8)], 64, "thread {gi}");
        assert_eq!(w[&(gi * 8 + 4)], 64, "thread {gi}: every old value once");
    }
}

#[test]
fn an_unordered_write_is_reported() {
    let (_, ub) = stored("race", RACE);
    assert_ne!(ub & exec::UB_RACE, 0);
}

/// The outputs of a vertex or pixel program of this crate's on seed 1, by (lane, element,
/// row, column); with the oracle, DXC's two containers must first agree with it.
fn outputs(name: &str, source: &str, profile: &str) -> BTreeMap<(u64, u32, u64, u64), u32> {
    let cfg = Config::default();
    let p = interp::load(&xcl_dxil::compile(source, "main", profile).expect("this crate compiles the program")).unwrap();
    for level in ["-O0", "-O1"] {
        let Some(bytes) = dxc_as(name, source, level, profile) else {
            eprintln!("{name}: no oracle (XCL_DXIL_DXC), DXC not compared");
            break;
        };
        let q = interp::load(&bytes).unwrap();
        for seed in 1..=4 {
            let v = interp::compare(&interp::run(&q, seed, &cfg), &interp::run(&p, seed, &cfg));
            assert!(matches!(v, Verdict::Equal), "{name}: DXC {level} / ours, seed {seed}: {v:?}");
        }
    }
    let run = interp::run(&p, 1, &cfg);
    assert!(run.stop.is_none() && run.ub == 0, "{name}: {:?}, undefined {}", run.stop, run.ub);
    run.written.iter().filter(|w| w.0 == 2).map(|w| ((w.2 >> 16, w.1, w.2 >> 8 & 0xff, w.2 & 0xff), w.3[0])).collect()
}

#[test]
fn a_pixel_program_runs_in_quads() {
    let out = outputs("pixel", PIXEL, "ps_6_6");
    for lane in 0..32u64 {
        if lane & 1 != 0 {
            // discarded: the mark and nothing else
            assert_eq!(out.range((lane, 0, 0, 0)..(lane + 1, 0, 0, 0)).map(|(k, v)| (k.1, *v)).collect::<Vec<_>>(), [(interp::res::DISCARDED, 1)], "lane {lane}");
            continue;
        }
        let of = |col: u64| f32::from_bits(out[&(lane, 0, 0, col)]);
        assert_eq!(of(0), 7.0, "lane {lane}: d/dx of x is 1, d/dy of y is 1");
        assert_eq!(of(1), 0.0, "lane {lane}: a flat input and x along y do not change");
        // the flat input is the quad's
        assert_eq!(out[&(lane, 0, 0, 3)], out[&(lane & !3, 0, 0, 3)], "lane {lane}");
    }
    // interpolated inputs lie on a plane through the quad: not all quads have it level
    assert!((0..32).step_by(4).any(|quad| out[&(quad, 0, 0, 2)] != out[&(quad + 2, 0, 0, 2)]));
}

#[test]
fn a_vertex_program_runs_a_wave_of_vertices() {
    let out = outputs("vertex", VERTEX, "vs_6_6");
    assert_eq!(out.len(), 32 * 5);
    for lane in 0..32u64 {
        let id = |l: u64| out.iter().find(|(k, _)| k.0 == l && k.1 == 1).map(|(_, v)| *v).unwrap();
        assert_eq!(id(lane), id(0) + lane as u32, "SV_VertexID counts up");
        assert_eq!(out[&(lane, 0, 0, 3)], 1f32.to_bits());
    }
}
