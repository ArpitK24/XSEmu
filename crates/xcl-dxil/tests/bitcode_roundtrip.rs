//! DXC's own output through the reader and the writer: every module is read, written again,
//! put into DXC's container in place of its bitcode, validated with dxv, and its
//! `dxc -dumpbin` text compared with that of the container DXC made.
//!
//! The oracle is the DXC folder `XCL_DXIL_DXC` names (as in the other oracle tests); without
//! it the tests pass with a note. `stages_round_trip` compiles the shaders below, one or
//! more of every stage. `containers_round_trip` takes containers DXC already made: every
//! `.dxil` file of the folders `XCL_DXIL_CONTAINERS` lists (separated by `:`).
//!
//! The two texts are equal but for what [`normalise`] removes:
//! * the `; shader hash` line (the container's digest covers the bitcode);
//! * `!tbaa`, `!alias.scope` and `!noalias` attachments and their nodes, which the IR does not
//!   carry ([`bcreader::Notes::dropped_attachments`] counts them);
//! * the `!llvm.loop` mark of an `[unroll]` loop that DXC's -O0 leaves in place
//!   (`llvm.loop.unroll.full`; at -O1 the loop is unrolled and the mark is gone). The mark of
//!   a `[loop]` loop is carried ([`xcl_dxil::ir::flag::LOOP_UNROLL_DISABLE`]);
//! * code DXC left dead (its -O0 output is full of loads of `@dx.nothing.a`): the writer drops
//!   it. Where the texts differ for that, the round trip is made again with
//!   [`bcreader::Options::keep_dead`] and the module is listed as `dead code kept`;
//! * the numbers of metadata nodes: a reference is replaced by what the node holds. DXC makes
//!   one `!dx.controlflow.hints` node for every statement with a hint, the writer one for
//!   every set of hints;
//! * call-site attributes (`#N` after an `llvm.lifetime` call, `nonnull` on the payload of
//!   DispatchMesh), which the IR does not carry;
//! * the comment DXC's disassembler puts after a dx.op call.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use xcl_dxil::{bcreader, bcwriter};

struct Case {
    name: &'static str,
    profile: &'static str,
    args: &'static [&'static str],
    source: &'static str,
}

fn oracle() -> Option<PathBuf> {
    match std::env::var_os("XCL_DXIL_DXC") {
        Some(dir) => Some(PathBuf::from(dir)),
        None => {
            eprintln!("XCL_DXIL_DXC is not set: skipping the round trips of DXC's output");
            None
        }
    }
}

fn tool(dxc: &Path, name: &str) -> Command {
    let mut c = Command::new(dxc.join("bin").join(name));
    c.env("LD_LIBRARY_PATH", dxc.join("lib"));
    c
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xcl-dxil-bitcode-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn numbered(line: &str) -> Option<(&str, &str)> {
    let (n, def) = line.strip_prefix('!')?.split_once(" = ")?;
    (!n.is_empty() && n.bytes().all(|c| c.is_ascii_digit())).then_some((n, def))
}

/// Copy `s`, every reference to a numbered metadata node replaced by the node's text.
fn expand<'a>(s: &'a str, nodes: &HashMap<&'a str, &'a str>, out: &mut String, stack: &mut Vec<&'a str>) {
    let b = s.as_bytes();
    let (mut i, mut start, mut quoted) = (0, 0, false);
    while i < b.len() {
        if b[i] == b'"' {
            quoted = !quoted;
        } else if b[i] == b'!' && !quoted && b.get(i + 1).is_some_and(u8::is_ascii_digit) {
            let end = i + 1 + b[i + 1..].iter().take_while(|c| c.is_ascii_digit()).count();
            let n = &s[i + 1..end];
            out.push_str(&s[start..i]);
            match (stack.iter().position(|&x| x == n), nodes.get(n)) {
                // a distinct node names itself
                (Some(at), _) => {
                    let _ = write!(out, "!<up {}>", stack.len() - at);
                }
                (None, Some(def)) => {
                    stack.push(n);
                    expand(def, nodes, out, stack);
                    stack.pop();
                }
                (None, None) => out.push_str(&s[i..end]),
            }
            (i, start) = (end, end);
            continue;
        }
        i += 1;
    }
    out.push_str(&s[start..]);
}

/// The disassembly without what a round trip changes (see the top of this file).
fn normalise(text: &str) -> String {
    let nodes: HashMap<&str, &str> = text.lines().filter_map(numbered).collect();
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        if line.starts_with("; shader hash") || numbered(line).is_some() {
            continue;
        }
        if !line.starts_with("  ") {
            if line.starts_with('!') {
                expand(line, &nodes, &mut out, &mut Vec::new());
            } else {
                out.push_str(line);
            }
            out.push('\n');
            continue;
        }
        let mut line = line.to_string();
        if let Some(at) = line[2..].find("  ; ") {
            line.truncate(at + 2);
        }
        for kind in [", !tbaa !", ", !alias.scope !", ", !noalias !"] {
            while let Some(at) = line.find(kind) {
                let end = at + kind.len() + line[at + kind.len()..].bytes().take_while(u8::is_ascii_digit).count();
                line.replace_range(at..end, "");
            }
        }
        if line.contains(" @llvm.lifetime.") {
            if let Some(at) = line.rfind(") #") {
                if line[at + 3..].bytes().all(|c| c.is_ascii_digit()) {
                    line.truncate(at + 1);
                }
            }
        }
        if line.contains(" @dx.op.dispatchMesh.") {
            line = line.replace("* nonnull %", "* %");
        }
        let from = out.len();
        expand(&line, &nodes, &mut out, &mut Vec::new());
        if let Some(at) = out[from..].find(UNROLL_FULL) {
            out.replace_range(from + at..from + at + UNROLL_FULL.len(), "");
        }
        out.push('\n');
    }
    out
}

const UNROLL_FULL: &str = ", !llvm.loop distinct !{!<up 1>, !{!\"llvm.loop.unroll.full\"}}";

struct Outcome {
    dxc: usize,
    ours: usize,
    instructions: usize,
    write: f64,
    notes: bcreader::Notes,
    /// the text is DXC's only with the dead code kept
    kept_dead: bool,
}

fn run(mut c: Command) -> String {
    let out = c.output().expect("run the oracle");
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// Read, write, swap, validate and compare one container DXC made. `to` is where the
/// container with the rewritten bitcode goes.
fn round_trip(dxc: &Path, from: &Path, to: &Path) -> Result<Outcome, String> {
    match attempt(dxc, from, to, false) {
        Err(e) if e.starts_with("the text differs") => attempt(dxc, from, to, true).map_err(|kept| format!("{e}
and with the dead code kept: {kept}")),
        other => other,
    }
}

fn attempt(dxc: &Path, from: &Path, to: &Path, keep_dead: bool) -> Result<Outcome, String> {
    let bytes = std::fs::read(from).map_err(|e| format!("read: {e}"))?;
    let bitcode = bcreader::container_bitcode(&bytes).map_err(|e| format!("container: {e}"))?;
    let (module, notes) = bcreader::read_with(bitcode, bcreader::Options { keep_dead }).map_err(|e| format!("read: {e}"))?;
    let t = std::time::Instant::now();
    let (ours, counts) = bcwriter::write(&module).map_err(|e| format!("write: {e}"))?;
    let write = t.elapsed().as_secs_f64();
    let swapped = bcreader::replace_bitcode(&bytes, &ours).map_err(|e| format!("swap: {e}"))?;
    std::fs::write(to, swapped).map_err(|e| format!("write: {e}"))?;
    let mut dxv = tool(dxc, "dxv");
    dxv.arg(to);
    let verdict = run(dxv);
    if !verdict.contains("Validation succeeded") {
        return Err(format!("dxv: {}", verdict.trim()));
    }
    let text = |path: &Path| {
        let mut c = tool(dxc, "dxc");
        c.arg("-dumpbin").arg(path);
        normalise(&run(c))
    };
    let (a, b) = (text(from), text(to));
    if a != b {
        let at = a.lines().zip(b.lines()).position(|(x, y)| x != y).unwrap_or(a.lines().count().min(b.lines().count()));
        let line = |t: &str| t.lines().nth(at).unwrap_or("<end>").chars().take(300).collect::<String>();
        return Err(format!("the text differs at line {}:\n  dxc:  {}\n  ours: {}", at + 1, line(&a), line(&b)));
    }
    if a.lines().count() < 20 || !a.contains("define void @") {
        return Err(format!("dxc -dumpbin gave no module: {}", a.chars().take(300).collect::<String>()));
    }
    Ok(Outcome { dxc: bitcode.len(), ours: ours.len(), instructions: counts.instructions, write, notes, kept_dead: keep_dead })
}

/// Print one line per module and fail if any did not make it.
fn report(results: Vec<(String, Result<Outcome, String>)>) {
    let (mut dxc, mut ours, mut failed) = (0, 0, Vec::new());
    for (name, r) in &results {
        match r {
            Ok(o) => {
                dxc += o.dxc;
                ours += o.ours;
                println!("{name:28} dxc {:8} ours {:8} ({:+5.2}%) {:7} instructions, written in {:.4} s; {:?}{}", o.dxc, o.ours, (o.ours as f64 / o.dxc as f64 - 1.0) * 100.0, o.instructions, o.write, o.notes, if o.kept_dead { "; dead code kept" } else { "" });
            }
            Err(e) => failed.push(format!("{name}: {e}")),
        }
    }
    println!("{} of {} modules round trip; bitcode bytes: dxc {dxc}, ours {ours} ({:+.2}%)", results.len() - failed.len(), results.len(), (ours as f64 / dxc.max(1) as f64 - 1.0) * 100.0);
    assert!(failed.is_empty(), "{} of {} failed:\n{}", failed.len(), results.len(), failed.join("\n"));
}

#[test]
fn normalise_compares_metadata_by_content() {
    let a = "; shader hash: 12\n  %1 = load i32, i32* %0, align 4, !tbaa !7\n  %2 = call float @dx.op.unary.f32(i32 6, float %x), !dx.precise !3  ; Saturate(value)\n  br i1 %c, label %3, label %4, !dx.controlflow.hints !4\n!dx.version = !{!0}\n!0 = !{i32 1, i32 6}\n!3 = !{i32 1}\n!4 = distinct !{!4, !\"dx.controlflow.hints\", i32 1}\n!7 = !{!8, !8, i64 0}\n!8 = !{!\"int\"}\n";
    let b = "  %1 = load i32, i32* %0, align 4\n  %2 = call float @dx.op.unary.f32(i32 6, float %x), !dx.precise !1\n  br i1 %c, label %3, label %4, !dx.controlflow.hints !2\n!dx.version = !{!0}\n!0 = !{i32 1, i32 6}\n!1 = !{i32 1}\n!2 = distinct !{!2, !\"dx.controlflow.hints\", i32 1}\n";
    assert_eq!(normalise(a), normalise(b));
    assert!(normalise(a).contains("!dx.controlflow.hints distinct !{!<up 1>, !\"dx.controlflow.hints\", i32 1}"), "{}", normalise(a));
    assert_ne!(normalise(a), normalise(&b.replace("i32 6}", "i32 7}")));
    assert_ne!(normalise(a), normalise(&b.replace(", !dx.precise !1", "")));
}

#[test]
fn stages_round_trip() {
    let Some(dxc) = oracle() else { return };
    let dir = scratch("stages");
    let mut results = Vec::new();
    for case in CASES {
        let (src, from, to) = (dir.join(format!("{}.hlsl", case.name)), dir.join(format!("{}.dxil", case.name)), dir.join(format!("{}.rt.dxil", case.name)));
        std::fs::write(&src, case.source).unwrap();
        // as XCL runs DXC
        let mut c = tool(&dxc, "dxc");
        c.args(["-E", "main", "-T", case.profile, "-HV", "2021", "-Gis", "-O1"]).args(case.args).arg("-Fo").arg(&from).arg(&src);
        let log = run(c);
        let r = if from.exists() { round_trip(&dxc, &from, &to) } else { Err(format!("dxc: {log}")) };
        results.push((case.name.to_string(), r));
    }
    let _ = std::fs::remove_dir_all(&dir);
    report(results);
}

#[test]
fn containers_round_trip() {
    let (Some(dxc), Some(list)) = (oracle(), std::env::var_os("XCL_DXIL_CONTAINERS")) else { return };
    let mut files: Vec<PathBuf> = Vec::new();
    for folder in std::env::split_paths(&list) {
        let entries = std::fs::read_dir(&folder).unwrap_or_else(|e| panic!("{}: {e}", folder.display()));
        files.extend(entries.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "dxil") && !p.to_string_lossy().ends_with(".rt.dxil")));
    }
    files.sort();
    let dir = scratch("containers");
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            s.spawn(|| loop {
                let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(from) = files.get(k) else { break };
                let r = round_trip(&dxc, from, &dir.join(format!("{k}.rt.dxil")));
                let name = from.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                results.lock().unwrap().push((k, name, r));
            });
        }
    });
    let _ = std::fs::remove_dir_all(&dir);
    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|r| r.0);
    report(results.into_iter().map(|(_, name, r)| (name, r)).collect());
}

#[rustfmt::skip]
const CASES: &[Case] = &[
    Case {
        name: "as_cull",
        profile: "as_6_6",
        args: &[],
        source: r#"struct Payload { float3 origin; uint count; };
cbuffer C : register(b0) { float4 planes[6]; float3 base; uint total; };
bool visible(float3 p) {
  for (uint k = 0; k < 6; k++) if (dot(planes[k], float4(p, 1)) < 0) return false;
  return true;
}
[numthreads(8, 1, 1)]
void main(uint3 did : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
  Payload pl;
  pl.origin = base + float3(did.x * 4, 0, 0);
  pl.count = total;
  bool show = did.x < total && visible(pl.origin);
  uint n = WaveActiveCountBits(show);
  DispatchMesh(show ? n : 0, 1, 1, pl);
}
"#,
    },
    Case {
        name: "cs_atomics",
        profile: "cs_6_6",
        args: &[],
        source: r#"
RWByteAddressBuffer io : register(u0);
RWTexture2D<float4> rwt : register(u1);
RWTexture2D<uint> rwu : register(u2);
RWStructuredBuffer<float4> rwsb : register(u3);
groupshared uint gs_counter;
groupshared float gs_arr[16];
static const float TBL[4] = { 1.0, 2.0, 0.1, 4.0 };
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint gi : SV_GroupIndex, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
  uint v = io.Load(tid.x * 4);
  float f = asfloat(v);
  uint orig;
  InterlockedAdd(gs_counter, v, orig);
  InterlockedCompareExchange(gs_counter, orig, v + 1, orig);
  io.InterlockedAdd(8, 1, orig);
  io.InterlockedCompareExchange(12, orig, 5, orig);
  InterlockedOr(rwu[tid.xy], 1);
  gs_arr[gi & 15] = f;
  GroupMemoryBarrierWithGroupSync();
  f += gs_arr[(gi + 1) & 15] + TBL[gi & 3];
  f += WaveActiveSum(f) + WaveActiveMax(f) + WavePrefixSum(f) + WaveReadLaneFirst(f) + WaveReadLaneAt(f, 0);
  v += WaveActiveSum(v) + WaveGetLaneCount() + WaveGetLaneIndex() + WaveActiveCountBits(v > 3) + WavePrefixCountBits(v > 1);
  uint4 ballot = WaveActiveBallot(v > 2);
  v += ballot.x + (WaveIsFirstLane() ? 1 : 0) + (WaveActiveAnyTrue(v > 7) ? 2 : 0) + (WaveActiveAllTrue(v > 8) ? 3 : 0) + (WaveActiveAllEqual(v) ? 1 : 0);
  v += WaveActiveBitAnd(v) + WaveActiveBitOr(v) + WaveActiveBitXor(v) + WaveActiveMin(v) + WavePrefixProduct(v);
  uint64_t big = (uint64_t)v * 0x100000001ULL + io.Load<uint64_t>(32);
  big = (big << (gi & 63)) ^ (big >> 3);
  int64_t sbig = (int64_t)big / 7;
  double d = (double)sbig * 0.5 + (double)f;
  switch (gi & 7) {
    case 0: v += 1; break;
    case 1: v *= 3; break;
    case 2: v ^= gid.x; break;
    case 5: v -= gtid.y; break;
    default: v = v / 3 + v % 5; break;
  }
  for (uint i = 0; i < (v & 15); i++) { f = f * 1.5 + rwt[uint2(i, tid.y)].x; if (f > 100.0) break; }
  int si = (int)v; si = si / 3 + si % 7 + (si >> 2); si = max(si, -5); si = clamp(si, -100, 100);
  uint a, b2; a = mul(v, 3u); 
  float4 r = float4(f, (float)d, (float)big, si);
  rwt[tid.xy] = r;
  rwsb[tid.x] = r;
  rwsb.IncrementCounter();
  io.Store(tid.x * 4, v);
  io.Store<float>(64, f);
  io.Store<uint64_t>(128, big);
  io.Store<double>(136, d);
}
"#,
    },
    Case {
        name: "cs_heap",
        profile: "cs_6_6",
        args: &[],
        source: r#"cbuffer C : register(b0) { uint tex_index; uint buf_index; uint sampler_index; };
[numthreads(4, 4, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  Texture2D<float4> t = ResourceDescriptorHeap[tex_index];
  Texture2D<float4> t2 = ResourceDescriptorHeap[NonUniformResourceIndex(tex_index + tid.z)];
  RWByteAddressBuffer b = ResourceDescriptorHeap[buf_index];
  RWTexture2D<float4> o = ResourceDescriptorHeap[buf_index + 1];
  SamplerState s = SamplerDescriptorHeap[sampler_index];
  float4 c = t.SampleLevel(s, tid.xy / 4.0, 0) + t2.Load(int3(tid.xy, 0));
  uint prev; b.InterlockedExchange(tid.x * 4, asuint(c.x), prev);
  b.InterlockedMax(64, prev); b.InterlockedMin(68, prev); b.InterlockedAnd(72, prev); b.InterlockedXor(76, prev);
  o[tid.xy] = c + f16tof32(prev) + (WaveIsFirstLane() ? QuadReadAcrossX(c.x) * 0 : 0);
}
"#,
    },
    Case {
        name: "cs_loops",
        profile: "cs_6_6",
        args: &[],
        source: r#"RWStructuredBuffer<uint> data : register(u0);
ByteAddressBuffer words : register(t0);
cbuffer C : register(b0) { uint count; uint mode; int bias; float scale; };
static const uint PRIMES[8] = { 2, 3, 5, 7, 11, 13, 17, 19 };
static const float WEIGHTS[4] = { 0.125, 0.375, 0.375, 0.125 };
uint hash(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16); }
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint acc = tid.x;
  uint local[8];
  for (uint i = 0; i < 8; i++) local[i] = hash(tid.x + PRIMES[i]);
  [loop] for (uint a = 0; a < count; a++) {
    if (a == 3) continue;
    uint v = words.Load((a & 255) * 4);
    uint b = 0;
    while (b < (v & 7)) {
      acc += local[(a + b) & 7];
      if (acc > 0x80000000u) break;
      b++;
    }
    do { acc = acc * 5 + 1; v >>= 1; } while (v > 1000);
    [forcecase] switch (mode) {
      case 0: acc ^= v; break;
      case 1:
      case 2: acc += PRIMES[v & 7]; break;
      case 7: for (uint k = 0; k < 4; k++) acc += (uint)(WEIGHTS[k] * scale * local[k]); break;
      case 100: return;
      default: acc = hash(acc); break;
    }
    [branch] if ((int)acc + bias < 0) { acc = ~acc; } else [flatten] if (acc & 1) { acc >>= 1; }
  }
  int s = (int)acc;
  s = s < 0 ? -s : s;
  s = (s / 3) * (s % 5 + 1) + (s >> 3) - (s << 2);
  data[tid.x] = acc + (uint)s + (uint)local[acc & 7];
}
"#,
    },
    Case {
        name: "cs_structs",
        profile: "cs_6_6",
        args: &[],
        source: r#"struct Particle { float3 pos; float life; float3 vel; uint kind; float4 colour; int2 cell; };
struct Emitter { float4x4 world; float3 gravity; float dt; uint first; uint count; };
RWStructuredBuffer<Particle> particles : register(u0);
AppendStructuredBuffer<uint> dead : register(u1);
ConsumeStructuredBuffer<uint> free : register(u2);
StructuredBuffer<Emitter> emitters : register(t0);
ConstantBuffer<Emitter> main_emitter : register(b0);
groupshared Particle cache[16];
groupshared uint alive;
Particle step(Particle p, Emitter e) {
  p.vel += e.gravity * e.dt;
  p.pos += p.vel * e.dt;
  p.life -= e.dt;
  p.cell = int2(floor(p.pos.xz));
  p.colour = lerp(p.colour, float4(0, 0, 0, 0), saturate(1 - p.life));
  return p;
}
[numthreads(16, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
  Emitter e = emitters[tid.x >> 8];
  if (gi == 0) alive = 0;
  GroupMemoryBarrierWithGroupSync();
  Particle p = particles[e.first + tid.x];
  p = step(p, e);
  p = step(p, main_emitter);
  cache[gi] = p;
  GroupMemoryBarrierWithGroupSync();
  Particle q = cache[(gi + 1) & 15];
  if (distance(p.pos, q.pos) < 0.1 && p.kind == q.kind) p.vel = reflect(p.vel, normalize(p.pos - q.pos + 0.001));
  if (p.life <= 0) {
    dead.Append(e.first + tid.x);
    uint slot = free.Consume();
    p.pos = mul(e.world, float4(0, 0, 0, 1)).xyz;
    p.life = 1 + (slot & 3);
  } else {
    InterlockedAdd(alive, 1);
  }
  particles[e.first + tid.x] = p;
  GroupMemoryBarrierWithGroupSync();
  if (gi == 0) particles.IncrementCounter();
  if (gi == 15 && alive == 0) particles.DecrementCounter();
}
"#,
    },
    Case {
        name: "cs_textures",
        profile: "cs_6_6",
        args: &[],
        source: r#"Texture2D<float4> colour : register(t0);
Texture2D<float> depth : register(t1);
Texture2DArray<uint> ids : register(t2);
Texture3D<float4> lut : register(t3);
TextureCubeArray<float4> probes : register(t4);
Texture1D<float4> ramp : register(t5);
Texture2D<float4> many[4] : register(t0, space1);
RWTexture2D<float4> out_colour : register(u0);
RWTexture3D<float> out_volume : register(u1);
RWTexture2DArray<uint2> out_ids : register(u2);
RWBuffer<uint> histogram : register(u3);
SamplerState lin : register(s0);
SamplerComparisonState cmp : register(s1);
cbuffer C : register(b0) { float2 inv_size; uint slice; float bias; };
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  float2 uv = (tid.xy + 0.5) * inv_size;
  uint w, h, levels; colour.GetDimensions(0, w, h, levels);
  uint aw, ah, layers; ids.GetDimensions(aw, ah, layers);
  float4 c = colour.SampleLevel(lin, uv, 0) + colour.SampleLevel(lin, uv, 1, int2(1, -1));
  c += colour.GatherRed(lin, uv) + colour.GatherAlpha(lin, uv, int2(2, 2));
  c.x += depth.SampleCmpLevelZero(cmp, uv, bias) + depth.GatherCmp(cmp, uv, bias).z;
  c += colour.Load(int3(tid.xy, 0), int2(-1, 0)) + colour[tid.xy] + colour.mips[1][tid.xy / 2];
  c += lut.SampleLevel(lin, c.xyz, 0) + lut.Load(int4(tid % 16, 0));
  c += probes.SampleLevel(lin, float4(c.xyz, slice), 2) + ramp.SampleLevel(lin, uv.x, 0) + ramp[tid.x & 255];
  c += many[tid.z & 3].SampleLevel(lin, uv, 0) + many[NonUniformResourceIndex(tid.x & 3)].Load(int3(tid.xy, 0));
  uint id = ids.Load(int4(tid.xy % uint2(aw, ah), slice % layers, 0));
  out_colour[tid.xy] = c + out_colour[tid.xy ^ 1];
  out_volume[tid] = c.x;
  out_ids[uint3(tid.xy, slice)] = uint2(id, levels + w * h);
  histogram[id & 255] += 1;
}
"#,
    },
    Case {
        name: "cs_wide",
        profile: "cs_6_6",
        args: &[],
        source: r#"RWByteAddressBuffer io : register(u0);
RWStructuredBuffer<double> doubles : register(u1);
cbuffer C : register(b0) { uint64_t seed; int64_t offset; double rate; min16int mi; min16uint mu; };
static const uint64_t K[3] = { 0x9e3779b97f4a7c15ULL, 0xbf58476d1ce4e5b9ULL, 0x94d049bb133111ebULL };
uint64_t mix(uint64_t z) { z = (z ^ (z >> 30)) * K[1]; z = (z ^ (z >> 27)) * K[2]; return z ^ (z >> 31); }
[numthreads(16, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint64_t x = mix(seed + K[0] * tid.x);
  int64_t s = (int64_t)x + offset;
  s = s / 3 - s % 7 + (s >> 5) + abs(s) + max(s, offset) + min(s, -offset);
  uint64_t u = x / 10 + x % 10 + (x << (tid.x & 63)) + (x >> 7) + countbits(x) + firstbithigh(x) + firstbitlow(x) + reversebits(x);
  uint lo = (uint)u, hi = (uint)(u >> 32);
  uint carry; uint sum = lo + hi; carry = sum < lo ? 1 : 0;
  double d = (double)s * rate + (double)u + asdouble(lo, hi);
  d = d * d - fma(d, rate, 1.0) + abs(d) + max(d, rate) + min(d, 2.0) + (d > rate ? 1.0 : -1.0);
  d = sqrt((float)d) + (double)(float)d / 3.0;
  uint dlo, dhi; asuint(d, dlo, dhi);
  min16int w = mi * (min16int)3 + (min16int)tid.x; min16uint uw = mu ^ (min16uint)dlo; uw = (uw << 3) | (uw >> 2);
  io.Store<uint64_t>(tid.x * 32, u ^ (uint64_t)s);
  io.Store<double>(tid.x * 32 + 8, d);
  io.Store<float>(tid.x * 32 + 16, (float)w);
  io.Store2(tid.x * 32 + 20, uint2(dlo + carry, dhi + sum + w + uw));
  doubles[tid.x] = d + doubles[tid.x ^ 1];
  uint64_t old = io.Load<uint64_t>(4096);
  io.Store<uint64_t>(4096, old + x);
}
"#,
    },
    Case {
        name: "ds_tri",
        profile: "ds_6_6",
        args: &[],
        source: r#"cbuffer C : register(b0) { float4x4 vp; float height; };
Texture2D<float> heights : register(t0);
SamplerState lin : register(s0);
struct Cp { float3 pos : POSITION; float2 uv : TEXCOORD0; };
struct Pc { float edges[3] : SV_TessFactor; float inside : SV_InsideTessFactor; float3 centre : CENTRE; };
struct Out { float4 pos : SV_Position; float2 uv : TEXCOORD0; float lift : LIFT; };
[domain("tri")]
Out main(Pc pc, float3 bary : SV_DomainLocation, const OutputPatch<Cp, 3> p, uint id : SV_PrimitiveID) {
  Out o;
  float3 pos = p[0].pos * bary.x + p[1].pos * bary.y + p[2].pos * bary.z;
  o.uv = p[0].uv * bary.x + p[1].uv * bary.y + p[2].uv * bary.z;
  o.lift = heights.SampleLevel(lin, o.uv, 0) * height * pc.inside;
  pos += normalize(pos - pc.centre) * o.lift;
  if (id == 0) pos.y += pc.edges[1];
  o.pos = mul(vp, float4(pos, 1));
  return o;
}
"#,
    },
    Case {
        name: "gs_points",
        profile: "gs_6_6",
        args: &[],
        source: r#"struct GsIn { float4 pos : SV_Position; float size : PSIZE; };
struct Sprite { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct Debug { float4 pos : SV_Position; uint prim : PRIM; uint view : SV_ViewportArrayIndex; };
[maxvertexcount(5)]
void main(point GsIn i[1], uint prim : SV_PrimitiveID, inout PointStream<Sprite> sprites, inout PointStream<Debug> debug) {
  for (uint k = 0; k < 4; k++) {
    Sprite s;
    s.uv = float2(k & 1, k >> 1);
    s.pos = i[0].pos + float4((s.uv - 0.5) * i[0].size, 0, 0);
    sprites.Append(s);
  }
  Debug d;
  d.pos = i[0].pos;
  d.prim = prim;
  d.view = prim & 3;
  debug.Append(d);
}
"#,
    },
    Case {
        name: "gs_strip",
        profile: "gs_6_6",
        args: &[],
        source: r#"cbuffer C : register(b0) { float4x4 vp; float size; };
struct GsIn { float4 pos : SV_Position; float3 colour : COLOR; };
struct GsOut { float4 pos : SV_Position; float3 colour : COLOR; uint layer : SV_RenderTargetArrayIndex; };
[maxvertexcount(6)]
[instance(2)]
void main(triangle GsIn i[3], uint prim : SV_PrimitiveID, uint inst : SV_GSInstanceID, inout TriangleStream<GsOut> tris) {
  float3 centre = (i[0].pos.xyz + i[1].pos.xyz + i[2].pos.xyz) / 3;
  for (uint pass = 0; pass < 2; pass++) {
    if (pass == 1 && (prim & 1) == 0) break;
    for (uint k = 0; k < 3; k++) {
      GsOut o;
      o.pos = mul(vp, float4(lerp(centre, i[k].pos.xyz, size * (pass + 1)), 1));
      o.colour = i[k].colour * (inst + 1);
      o.layer = inst + pass;
      tris.Append(o);
    }
    tris.RestartStrip();
  }
}
"#,
    },
    Case {
        name: "hs_patch",
        profile: "hs_6_6",
        args: &[],
        source: r#"cbuffer C : register(b0) { float3 eye; float factor; };
struct Cp { float3 pos : POSITION; float2 uv : TEXCOORD0; };
struct Pc { float edges[3] : SV_TessFactor; float inside : SV_InsideTessFactor; float3 centre : CENTRE; };
Pc patch(InputPatch<Cp, 3> p, uint id : SV_PrimitiveID) {
  Pc o;
  o.centre = (p[0].pos + p[1].pos + p[2].pos) / 3;
  float d = distance(o.centre, eye);
  for (uint k = 0; k < 3; k++) o.edges[k] = clamp(factor / max(d, 0.001) + (id & 3), 1, 64);
  o.inside = (o.edges[0] + o.edges[1] + o.edges[2]) / 3;
  return o;
}
[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("patch")]
[maxtessfactor(64.0)]
Cp main(InputPatch<Cp, 3> p, uint k : SV_OutputControlPointID) {
  Cp o;
  o.pos = p[k].pos + normalize(p[k].pos - eye) * 0.01;
  o.uv = p[k].uv;
  return o;
}
"#,
    },
    Case {
        name: "ms_grid",
        profile: "ms_6_6",
        args: &[],
        source: r#"struct Payload { float3 origin; uint count; };
struct Vert { float4 pos : SV_Position; float3 colour : COLOR; };
struct Prim { uint layer : SV_RenderTargetArrayIndex; bool cull : SV_CullPrimitive; float3 flat : FLAT; };
cbuffer C : register(b0) { float4x4 vp; };
[outputtopology("triangle")]
[numthreads(4, 4, 1)]
void main(uint3 tid : SV_GroupThreadID, uint gi : SV_GroupIndex, in payload Payload pl, out vertices Vert verts[16], out indices uint3 tris[18], out primitives Prim prims[18]) {
  SetMeshOutputCounts(16, 18);
  verts[gi].pos = mul(vp, float4(pl.origin + float3(tid.xy, 0), 1));
  verts[gi].colour = float3(tid.xy / 3.0, pl.count);
  if (tid.x < 3 && tid.y < 3) {
    uint q = tid.y * 3 + tid.x;
    tris[q * 2] = uint3(gi, gi + 1, gi + 4);
    tris[q * 2 + 1] = uint3(gi + 1, gi + 5, gi + 4);
    for (uint k = 0; k < 2; k++) {
      prims[q * 2 + k].layer = pl.count & 1;
      prims[q * 2 + k].cull = (q + k) % 5 == 0;
      prims[q * 2 + k].flat = float3(q, k, 0);
    }
  }
}
"#,
    },
    Case {
        name: "ps_discard",
        profile: "ps_6_6",
        args: &[],
        source: r#"
Texture2D<float4> tex : register(t0);
Texture2D<uint4> utex : register(t1);
SamplerState ss : register(s0);
SamplerComparisonState cs : register(s1);
Buffer<float4> buf : register(t2);
ByteAddressBuffer bab : register(t3);
StructuredBuffer<float4> sb : register(t4);
cbuffer C : register(b0) { float4 k; double dk; int ik; };
struct PSIn { float4 pos : SV_Position; float2 uv : TEXCOORD0; nointerpolation uint id : ID; float3 n : NORMAL; float3 bary : SV_Barycentrics; uint cov : SV_Coverage; };
float4 main(PSIn i, out float depth : SV_Depth) : SV_Target {
  float4 c = tex.Sample(ss, i.uv);
  c += tex.SampleLevel(ss, i.uv, 1.0);
  c += tex.SampleBias(ss, i.uv, 0.5);
  c += tex.SampleGrad(ss, i.uv, ddx(i.uv), ddy(i.uv));
  c.x += tex.SampleCmp(cs, i.uv, 0.5);
  c.y += tex.SampleCmpLevelZero(cs, i.uv, 0.5);
  c += tex.Gather(ss, i.uv);
  c += tex.GatherCmp(cs, i.uv, 0.25);
  c += tex.Load(int3(i.id, 2, 0));
  c += (float4)utex.Load(int3(1, 2, 0));
  c += buf.Load(i.id);
  c += bab.Load<float4>(16);
  c += sb[i.id];
  c.x += tex.CalculateLevelOfDetail(ss, i.uv);
  c.y += tex.CalculateLevelOfDetailUnclamped(ss, i.uv);
  uint w, h, l; tex.GetDimensions(0, w, h, l); c.z += w + h + l;
  c += k * sqrt(abs(c)) + sin(c) * cos(c) + exp(c) + log(max(c, 0.001));
  c += fmod(c, k) + frac(c) + floor(c) + ceil(c) + round(c) + trunc(c) + rsqrt(c);
  c.x += dot(i.n, c.xyz) + length(i.n) + (float)dk + ik;
  c = saturate(c); c = min(c, k); c = mad(c, k, c); c = lerp(c, k, 0.5); c = pow(abs(c), k);
  c.x += ddx_coarse(c.y) + ddy_fine(c.z);
  c.x += isnan(c.y) ? 1 : 0; c.y += isinf(c.x) ? 1 : 0;
  c.z += f16tof32(i.id); c.w += f32tof16(c.x);
  c.x += asfloat(i.id); c.y += asuint(c.z);
  c.z += countbits(i.id) + firstbithigh(i.id) + reversebits(i.id);
  c.xyz += i.bary;
  c.x += EvaluateAttributeAtSample(i.uv, 0).x + EvaluateAttributeCentroid(i.uv).y + EvaluateAttributeSnapped(i.uv, int2(1,1)).x;
  c.x += GetRenderTargetSampleCount() + GetRenderTargetSamplePosition(0).x;
  c.x += atan2(c.y, c.z) + tan(c.x) + asin(c.y) + acos(c.z) + atan(c.w) + sinh(c.x) + cosh(c.y) + tanh(c.z);
  c.x += smoothstep(0, 1, c.y) + step(c.z, c.w) + sign(c.x) + exp2(c.y) + log2(abs(c.z)+1) + log10(abs(c.w)+1) + ldexp(c.x, c.y);
  double d = dk * c.x; d = sqrt((float)d) + fma(d, dk, d); c.y += (float)d;
  uint lo, hi; asuint(dk, lo, hi); c.z += lo + hi; c.w += (float)asdouble(lo, hi);
  if (c.w < 0.1) discard;
  depth = c.x;
  return c;
}
"#,
    },
    Case {
        name: "ps_targets",
        profile: "ps_6_6",
        args: &[],
        source: r#"Texture2DArray<float4> layers : register(t0);
TextureCube<float4> sky : register(t1);
Texture3D<float> volume : register(t2);
Texture2DMS<float4> ms : register(t3);
SamplerState lin : register(s0);
struct Light { float3 dir; float power; float4 colour; };
StructuredBuffer<Light> lights : register(t4);
ConstantBuffer<Light> sun : register(b0);
struct In { float4 pos : SV_Position; centroid float3 dir : DIR; noperspective float2 uv : UV; sample float fog : FOG; bool front : SV_IsFrontFace; uint sample_id : SV_SampleIndex; };
struct Out { float4 a : SV_Target0; uint4 b : SV_Target1; float depth : SV_DepthLessEqual; uint mask : SV_Coverage; };
Out main(In i) {
  Out o;
  float4 c = layers.Sample(lin, float3(i.uv, 2)) + sky.SampleLevel(lin, i.dir, 0);
  c.x += volume.Sample(lin, i.dir, int3(1, 0, -1));
  c += ms.Load(int2(i.pos.xy), i.sample_id);
  uint n, stride; lights.GetDimensions(n, stride);
  [loop] for (uint k = 0; k < n; k++) {
    Light l = lights[k];
    float d = dot(l.dir, i.dir);
    [branch] if (d > 0) c += l.colour * d * l.power; else c *= 0.99;
  }
  [flatten] if (i.front) c += sun.colour * sun.power;
  clip(c.w - 0.01);
  o.a = c * i.fog;
  o.b = uint4(c * 255.0) ^ uint4(i.sample_id, n, stride, 7);
  o.depth = i.pos.z * 0.5;
  o.mask = 1u << i.sample_id;
  return o;
}
"#,
    },
    Case {
        name: "vs_transform",
        profile: "vs_6_6",
        args: &[],
        source: r#"cbuffer Frame : register(b0) { float4x4 view_proj; float4 tint; float time; uint flags; };
cbuffer Bones : register(b1) { float4x4 bones[4]; };
struct VsIn { float3 pos : POSITION; float3 normal : NORMAL; float2 uv : TEXCOORD0; uint4 bone : BLENDINDICES; float4 weight : BLENDWEIGHT; uint vid : SV_VertexID; uint iid : SV_InstanceID; };
struct VsOut { float4 pos : SV_Position; float3 normal : NORMAL; float2 uv : TEXCOORD0; nointerpolation uint id : ID; float clip : SV_ClipDistance0; };
VsOut main(VsIn i) {
  VsOut o;
  float4 p = float4(0, 0, 0, 0);
  for (uint k = 0; k < 4; k++) p += mul(bones[i.bone[k] & 3], float4(i.pos, 1)) * i.weight[k];
  if (flags & 1) p.xyz += i.normal * sin(time + i.vid);
  o.pos = mul(view_proj, p);
  o.normal = normalize(mul((float3x3)bones[0], i.normal));
  o.uv = i.uv * tint.xy + float2(i.iid, i.vid & 1);
  o.id = i.iid * 16 + (i.vid % 3);
  o.clip = dot(p, tint);
  return o;
}
"#,
    },
];
