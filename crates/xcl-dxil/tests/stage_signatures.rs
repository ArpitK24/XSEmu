//! The entry signatures and stage state [`xcl_dxil::hlsl::stage`] builds, against what DXC
//! builds for the same source.
//!
//! The cases are in `src/hlsl/stage/golden.txt`: an HLSL snippet in one of the shapes the
//! translator emits (and the corners around them) and the canonical text of what DXC 1.9
//! (`-HV 2021 -Gis -O1`) wrote in `dx.entryPoints` for it: every signature element with its
//! register and column, and the stage's properties. Without DXC the golden copy is the
//! reference; with `XCL_DXIL_DXC` (the `dxc-linux` folder) set, the golden copy is first
//! checked against DXC itself (`XCL_DXIL_BLESS=1` rewrites it). With `XCL_DXIL_CORPUS` as well,
//! every graphics program of the corpus is compared the same way.
//!
//! Two comparisons per case: [`stage::begin`] alone (everything but the usage masks, which
//! depend on the code), and the whole front end ([`xcl_dxil::hlsl::lower::lower`]) with the
//! masks, for the cases it lowers. The masks state which components the code that was
//! generated reads and writes. The written ones are DXC's; of the inputs DXC's optimiser can
//! prove more reads dead than the lowering does, so there the masks must cover DXC's.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use xcl_dxil::hlsl::emit::Emit;
use xcl_dxil::hlsl::{lower, parse, stage};
use xcl_dxil::shader::{parse_profile, ShaderInfo, SigElement, StageProps};
use xcl_dxil::Error;

const GOLDEN: &str = include_str!("../src/hlsl/stage/golden.txt");

struct Case {
    name: String,
    profile: String,
    /// `begin` is expected to return `Unsupported` (the golden text is still DXC's)
    unsupported: bool,
    hlsl: String,
    expected: String,
}

fn cases() -> (String, Vec<Case>) {
    let (header, rest) = GOLDEN.split_once("=== ").expect("golden.txt has cases");
    let cases = rest
        .split("\n=== ")
        .map(|c| {
            let (head, body) = c.split_once('\n').unwrap();
            let mut words = head.split_whitespace();
            let (name, profile) = (words.next().unwrap().to_string(), words.next().unwrap().to_string());
            let unsupported = words.next() == Some("unsupported");
            let (hlsl, expected) = body.split_once("\n---\n").unwrap_or_else(|| (body.strip_suffix("\n---").unwrap_or(body), ""));
            Case { name, profile, unsupported, hlsl: format!("{hlsl}\n"), expected: expected.trim_end().to_string() }
        })
        .collect();
    (header.to_string(), cases)
}

// ---- canonical text ----

fn element_line(sig: &str, e: &SigElement, masks: bool) -> String {
    let mut s = format!(
        "{sig} {} idx={:?} ct={} kind={} interp={} rows={} cols={} at={},{} stream={}",
        e.name, e.indices, e.comp_type as u32, e.kind as u32, e.interp as u32, e.rows, e.cols, e.start_row, e.start_col, e.stream
    );
    if masks {
        let _ = write!(s, " dyn={} used={}", e.dynamic_index_mask, e.usage_mask);
    }
    s
}

fn props_line(p: &StageProps) -> String {
    match p {
        StageProps::Vertex => "props vs".into(),
        StageProps::Pixel { early_depth_stencil } => format!("props ps eds={}", *early_depth_stencil as u32),
        StageProps::Geometry { input_primitive, max_vertex_count, active_streams, stream_topology, instance_count } => {
            format!("props gs prim={} maxv={max_vertex_count} streams={active_streams} topo={} inst={instance_count}", *input_primitive as u32, *stream_topology as u32)
        }
        StageProps::Compute { numthreads: [x, y, z], wave_size } => format!("props cs threads={x},{y},{z} wave={}", wave_size.unwrap_or(0)),
        StageProps::Mesh { numthreads: [x, y, z], max_vertices, max_primitives, topology, payload_bytes } => {
            format!("props ms threads={x},{y},{z} maxv={max_vertices} maxp={max_primitives} topo={} payload={payload_bytes}", *topology as u32)
        }
        StageProps::Amplification { numthreads: [x, y, z], payload_bytes } => format!("props as threads={x},{y},{z} payload={payload_bytes}"),
        other => format!("props {other:?}"),
    }
}

fn info_text(info: &ShaderInfo, masks: bool) -> String {
    let mut lines = Vec::new();
    for (sig, elements) in [("in", &info.inputs), ("out", &info.outputs), ("pc", &info.patch_or_prim)] {
        lines.extend(elements.iter().map(|e| element_line(sig, e, masks)));
    }
    lines.push(props_line(&info.props));
    lines.join("\n")
}

/// Whether `ours` (with masks) is DXC's `theirs`: every line equal, but an input may be
/// read in more components than DXC's code reads.
fn masks_agree(ours: &str, theirs: &str) -> bool {
    let masks = |l: &str| -> Option<(u32, u32)> {
        let (_, m) = l.split_once(" dyn=")?;
        let (d, u) = m.split_once(" used=")?;
        Some((d.parse().ok()?, u.parse().ok()?))
    };
    ours.lines().count() == theirs.lines().count()
        && ours.lines().zip(theirs.lines()).all(|(a, b)| {
            a == b
                || (a.starts_with("in ")
                    && a.split(" dyn=").next() == b.split(" dyn=").next()
                    && matches!((masks(a), masks(b)), (Some((ad, au)), Some((bd, bu))) if ad & bd == bd && au & bu == bu))
        })
}

/// The golden text without the usage masks.
fn without_masks(text: &str) -> String {
    text.lines().map(|l| l.split(" dyn=").next().unwrap()).collect::<Vec<_>>().join("\n")
}

// ---- our side ----

/// The signatures and properties from [`stage::begin`] alone.
fn ours(hlsl: &str, profile: &str) -> Result<String, Error> {
    let unit = parse::parse(hlsl)?;
    let (kind, ..) = parse_profile(profile).unwrap();
    let entry = unit.funcs.iter().rfind(|f| f.body.is_some() && unit.name(f.name) == "main").expect("an entry function");
    let mut e = Emit::new();
    let io = stage::begin(&mut e, &unit, entry, kind)?;
    let mut info = ShaderInfo::new(kind, "main", StageProps::Vertex);
    io.declared(&mut info);
    Ok(info_text(&info, false))
}

/// The same from the whole front end, with the usage masks.
fn lowered(hlsl: &str, profile: &str) -> Result<String, Error> {
    let unit = parse::parse(hlsl)?;
    let (_, info) = lower::lower(&unit, "main", profile)?;
    Ok(info_text(&info, true))
}

// ---- DXC's side ----

fn dxc_dir() -> Option<PathBuf> {
    std::env::var_os("XCL_DXIL_DXC").map(PathBuf::from)
}

/// A metadata operand of DXC's listing.
#[derive(Debug, Clone)]
enum Md {
    Null,
    Int(i64),
    Str(String),
    Node(Vec<Md>),
    Other,
}

impl Md {
    fn int(&self) -> i64 {
        match self {
            Md::Int(v) => *v,
            other => panic!("an integer is expected: {other:?}"),
        }
    }
    fn list(&self) -> &[Md] {
        match self {
            Md::Node(v) => v,
            _ => &[],
        }
    }
}

/// Split at the commas outside brackets.
fn operands(s: &str) -> Vec<&str> {
    let (mut out, mut depth, mut start, mut quoted) = (Vec::new(), 0i32, 0usize, false);
    for (i, c) in s.char_indices() {
        match c {
            '"' => quoted = !quoted,
            '[' | '{' | '(' if !quoted => depth += 1,
            ']' | '}' | ')' if !quoted => depth -= 1,
            ',' if depth == 0 && !quoted => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    if !s[start..].trim().is_empty() {
        out.push(s[start..].trim());
    }
    out
}

fn md(nodes: &std::collections::HashMap<u32, String>, op: &str) -> Md {
    if op == "null" {
        return Md::Null;
    }
    if let Some(rest) = op.strip_prefix('!') {
        if let Some(text) = rest.strip_prefix('"') {
            return Md::Str(text.trim_end_matches('"').to_string());
        }
        if let Some(body) = rest.strip_prefix('{') {
            return Md::Node(operands(body.trim_end_matches('}')).into_iter().map(|o| md(nodes, o)).collect());
        }
        if let Some(body) = rest.parse::<u32>().ok().and_then(|n| nodes.get(&n)) {
            return Md::Node(operands(body).into_iter().map(|o| md(nodes, o)).collect());
        }
        return Md::Other;
    }
    match op.split_once(' ') {
        Some((ty, v)) if ty.starts_with('i') => v.parse().map_or(Md::Other, Md::Int),
        _ => Md::Other,
    }
}

/// The canonical text of a DXC listing (`-Fc`): its dx.entryPoints record.
fn listing_text(ll: &str, profile: &str) -> String {
    let mut nodes = std::collections::HashMap::new();
    let mut entry_points = None;
    for line in ll.lines() {
        let Some((name, body)) = line.strip_prefix('!').and_then(|l| l.split_once(" = !{")) else { continue };
        let body = body.trim_end().trim_end_matches('}');
        if name == "dx.entryPoints" {
            entry_points = Some(body.to_string());
        } else if let Ok(n) = name.parse::<u32>() {
            nodes.insert(n, body.to_string());
        }
    }
    let entry = md(&nodes, operands(&entry_points.expect("dx.entryPoints"))[0]);
    let entry = entry.list();
    let mut lines = Vec::new();
    for (sig, elements) in ["in", "out", "pc"].iter().zip(entry[2].list()) {
        for e in elements.list() {
            let e = e.list();
            let Md::Str(name) = &e[1] else { panic!("element name") };
            let indices: Vec<i64> = e[4].list().iter().map(Md::int).collect();
            let (mut stream, mut dynamic, mut used) = (0, 0, 0);
            for pair in e[10].list().chunks(2) {
                match pair[0].int() {
                    0 => stream = pair[1].int(),
                    2 => dynamic = pair[1].int(),
                    3 => used = pair[1].int(),
                    _ => {}
                }
            }
            lines.push(format!(
                "{sig} {name} idx={indices:?} ct={} kind={} interp={} rows={} cols={} at={},{} stream={stream} dyn={dynamic} used={used}",
                e[2].int(),
                e[3].int(),
                e[5].int(),
                e[6].int(),
                e[7].int(),
                e[8].int(),
                e[9].int()
            ));
        }
    }
    let (mut flags, mut state, mut threads, mut wave) = (0, Vec::new(), Vec::new(), 0);
    for pair in entry[4].list().chunks(2) {
        match pair[0].int() {
            0 => flags = pair[1].int(),
            1 | 9 | 10 => state = pair[1].list().to_vec(),
            4 => threads = pair[1].list().iter().map(Md::int).collect(),
            11 => wave = pair[1].list().first().map_or(0, Md::int),
            _ => {}
        }
    }
    let ints = |m: &Md| m.list().iter().map(|x| x.int().to_string()).collect::<Vec<_>>().join(",");
    lines.push(match &profile[..2] {
        "vs" => "props vs".to_string(),
        // DXIL shader flag bit 3: ForceEarlyDepthStencil
        "ps" => format!("props ps eds={}", flags >> 3 & 1),
        "gs" => format!("props gs prim={} maxv={} streams={} topo={} inst={}", state[0].int(), state[1].int(), state[2].int(), state[3].int(), state[4].int()),
        "cs" => format!("props cs threads={} wave={wave}", threads.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")),
        "ms" => format!("props ms threads={} maxv={} maxp={} topo={} payload={}", ints(&state[0]), state[1].int(), state[2].int(), state[3].int(), state[4].int()),
        "as" => format!("props as threads={} payload={}", ints(&state[0]), state[1].int()),
        other => format!("props {other}"),
    });
    lines.join("\n")
}

/// Compile with DXC as XCL does; the canonical text, or "error" when DXC rejects the source.
fn dxc_text(dxc: &Path, hlsl: &str, profile: &str, tag: &str) -> String {
    let dir = std::env::temp_dir().join(format!("xcl-dxil-stage-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (src, ll) = (dir.join(format!("{tag}.hlsl")), dir.join(format!("{tag}.ll")));
    std::fs::write(&src, hlsl).unwrap();
    let _ = std::fs::remove_file(&ll);
    let out = Command::new(dxc.join("bin/dxc"))
        .env("LD_LIBRARY_PATH", dxc.join("lib"))
        .args(["-E", "main", "-T", profile, "-HV", "2021", "-Gis", "-O1", "-Fc"])
        .arg(&ll)
        .arg(&src)
        .output()
        .expect("dxc runs");
    let text = std::fs::read_to_string(&ll).ok().filter(|_| out.status.success());
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&ll);
    match text {
        Some(text) => listing_text(&text, profile),
        None => "error".to_string(),
    }
}

// ---- the tests ----

/// The golden copy is what DXC builds (needs DXC; `XCL_DXIL_BLESS=1` rewrites the file).
#[test]
fn golden_is_dxc() {
    let Some(dxc) = dxc_dir() else {
        eprintln!("stage_signatures: XCL_DXIL_DXC is not set, the golden copy is not checked against DXC");
        return;
    };
    let (header, cases) = cases();
    let mut file = header;
    let mut stale = Vec::new();
    for c in &cases {
        let text = dxc_text(&dxc, &c.hlsl, &c.profile, &c.name);
        if text != c.expected {
            stale.push(format!("{}:\n{text}\n-- golden:\n{}", c.name, c.expected));
        }
        let _ = write!(file, "=== {} {}{}\n{}---\n{text}\n", c.name, c.profile, if c.unsupported { " unsupported" } else { "" }, c.hlsl);
    }
    if std::env::var_os("XCL_DXIL_BLESS").is_some() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/hlsl/stage/golden.txt");
        std::fs::write(&path, file).unwrap();
        eprintln!("stage_signatures: wrote {} ({} cases, {} changed)", path.display(), cases.len(), stale.len());
        return;
    }
    assert!(stale.is_empty(), "{} golden cases differ from DXC:\n{}", stale.len(), stale.join("\n\n"));
}

/// [`stage::begin`] builds DXC's signatures: elements, their order (the ids), semantic kinds,
/// component types, interpolation modes, registers and columns, streams; and the properties.
#[test]
fn signatures_match_golden() {
    let (_, cases) = cases();
    let mut failed = Vec::new();
    for c in &cases {
        let got = ours(&c.hlsl, &c.profile);
        let ok = match (&got, c.expected.as_str()) {
            (Err(Error::Unsupported(_)), _) => c.unsupported,
            (_, _) if c.unsupported => false,
            // what DXC rejects is an error of the source here too: of the declaration, or (an
            // output the code does not write) of the lowered function
            (Err(Error::Source { .. }), "error") => true,
            (Ok(_), "error") => matches!(lowered(&c.hlsl, &c.profile), Err(Error::Source { .. })),
            (Ok(text), expected) => *text == without_masks(expected),
            _ => false,
        };
        if !ok {
            failed.push(format!("{} ({}):\n{}\n-- DXC:\n{}", c.name, c.profile, got.unwrap_or_else(|e| format!("{e:?}")), without_masks(&c.expected)));
        }
    }
    assert!(failed.is_empty(), "{} of {} cases differ:\n{}", failed.len(), cases.len(), failed.join("\n\n"));
}

/// The whole front end on the same cases, usage masks included, for the ones it lowers.
#[test]
fn lowered_masks_match_golden() {
    let (_, cases) = cases();
    let (mut failed, mut skipped, mut checked, mut exact) = (Vec::new(), Vec::new(), 0, 0);
    for c in cases.iter().filter(|c| c.expected != "error" && !c.unsupported) {
        match lowered(&c.hlsl, &c.profile) {
            Err(Error::Unsupported(what)) => skipped.push(format!("{} ({what})", c.name)),
            Ok(text) if masks_agree(&text, &c.expected) => {
                checked += 1;
                exact += (text == c.expected) as u32;
            }
            got => failed.push(format!("{} ({}):\n{}\n-- DXC:\n{}", c.name, c.profile, got.unwrap_or_else(|e| format!("{e:?}")), c.expected)),
        }
    }
    eprintln!("stage_signatures: {checked} cases lowered and equal to DXC ({exact} with DXC's input masks exactly); not lowered: {}", if skipped.is_empty() { "none".to_string() } else { skipped.join(", ") });
    assert!(failed.is_empty(), "{} cases differ:\n{}", failed.len(), failed.join("\n\n"));
}

/// Every graphics program of the corpus (`XCL_DXIL_CORPUS`), against DXC directly.
#[test]
fn corpus_signatures_match_dxc() {
    let (Some(dxc), Some(corpus)) = (dxc_dir(), std::env::var_os("XCL_DXIL_CORPUS").map(PathBuf::from)) else {
        eprintln!("stage_signatures: XCL_DXIL_DXC and XCL_DXIL_CORPUS are not both set: skipping the corpus");
        return;
    };
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "hlsl") {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    walk(&corpus, &mut files);
    files.sort();
    let (mut failed, mut checked, mut lowered_ok, mut masks_equal, mut not_lowered) = (Vec::new(), 0, 0, 0, 0);
    for f in &files {
        let src = std::fs::read_to_string(f).unwrap();
        // the first line names the profile: "// xdxc: -E main -T ps_6_6"
        let Some(profile) = src.lines().next().and_then(|l| l.strip_prefix("// xdxc: -E main -T ")).map(str::trim) else { continue };
        if profile.starts_with("cs") || !profile.ends_with("_6_6") {
            continue;
        }
        let name = f.file_name().unwrap().to_string_lossy().replace('.', "_");
        let expected = dxc_text(&dxc, &src, profile, &name);
        let got = ours(&src, profile);
        let ok = match (&got, expected.as_str()) {
            (Err(Error::Source { .. }), "error") => true,
            (Ok(text), expected) => *text == without_masks(expected),
            _ => false,
        };
        if !ok {
            failed.push(format!("{}:\n{}\n-- DXC:\n{}", f.display(), got.unwrap_or_else(|e| format!("{e:?}")), without_masks(&expected)));
            continue;
        }
        checked += 1;
        match lowered(&src, profile) {
            Ok(text) if masks_agree(&text, &expected) => {
                lowered_ok += 1;
                masks_equal += (text == expected) as u32;
            }
            Ok(text) => failed.push(format!("{} (lowered):\n{text}\n-- DXC:\n{expected}", f.display())),
            Err(Error::Unsupported(_)) => not_lowered += 1,
            Err(_) if expected == "error" => {}
            Err(e) => failed.push(format!("{} (lowered): {e:?}", f.display())),
        }
    }
    eprintln!("stage_signatures: {checked} corpus programs with DXC's signatures; {lowered_ok} lowered with masks that agree ({masks_equal} exactly), {not_lowered} not lowered");
    assert!(failed.is_empty(), "{} corpus programs differ:\n{}", failed.len(), failed.join("\n\n"));
}

/// A small deterministic generator (the cases are the same on every run).
struct Rng(u64);

impl Rng {
    fn next(&mut self, n: u32) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32) % n
    }
    fn pick<'a>(&mut self, of: &[&'a str]) -> &'a str {
        of[self.next(of.len() as u32) as usize]
    }
}

/// A random struct of signature fields in the generator's vocabulary and around it: float,
/// int and uint vectors of every width with every interpolation modifier, small arrays, clip
/// and cull distances, and the system values valid at `point` ("vsout", "psin", "gs", "msv",
/// "msp").
fn random_fields(r: &mut Rng, point: &str) -> String {
    let mut s = String::new();
    if point != "msp" && (point != "psin" || r.next(8) != 0) {
        s.push_str("    float4 pos : SV_Position;\n");
    }
    let (mut clip, mut cull, mut used) = (0, 0, std::collections::HashSet::new());
    for k in 0..r.next(12) {
        let width = r.next(4) + 1;
        let n = |ty: &str| if width == 1 { ty.to_string() } else { format!("{ty}{width}") };
        let choice = r.next(if point == "msp" { 10 } else { 16 });
        match choice {
            0..=5 => {
                let m = if point == "msp" { "" } else { r.pick(&["", "", "", "nointerpolation ", "noperspective ", "centroid ", "sample ", "noperspective centroid ", "noperspective sample ", "linear "]) };
                let dim = if r.next(6) == 0 { format!("[{}]", r.next(2) + 2) } else { String::new() };
                let _ = writeln!(s, "    {m}{} f{k}{dim} : F{k}_;", n("float"));
            }
            6..=8 => {
                let m = if r.next(2) == 0 { "nointerpolation " } else { "" };
                let _ = writeln!(s, "    {m}{} i{k} : I{k}_;", n(r.pick(&["uint", "int"])));
            }
            9 => {
                let sv = r.pick(&["SV_RenderTargetArrayIndex", "SV_ViewportArrayIndex", "SV_PrimitiveID", "SV_IsFrontFace", "SV_InstanceID", "SV_SampleIndex", "SV_Coverage", "SV_CullPrimitive"]);
                let valid = match (sv, point) {
                    ("SV_RenderTargetArrayIndex" | "SV_ViewportArrayIndex", p) => p != "msv",
                    ("SV_PrimitiveID", p) => matches!(p, "psin" | "gs" | "msp"),
                    ("SV_IsFrontFace", p) => matches!(p, "psin" | "gs"),
                    ("SV_InstanceID", p) => p != "msp" && p != "msv",
                    ("SV_SampleIndex" | "SV_Coverage", p) => p == "psin",
                    (_, p) => p == "msp",
                };
                if valid && used.insert(sv) {
                    let ty = if matches!(sv, "SV_IsFrontFace" | "SV_CullPrimitive") { "bool" } else { "uint" };
                    let _ = writeln!(s, "    {ty} s{k} : {sv};");
                }
            }
            _ => {
                let (name, index) = if choice % 2 == 0 { ("SV_ClipDistance", &mut clip) } else { ("SV_CullDistance", &mut cull) };
                if r.next(4) == 0 {
                    let rows = r.next(2) + 2;
                    let _ = writeln!(s, "    float d{k}[{rows}] : {name}{index};");
                    *index += rows;
                } else {
                    let _ = writeln!(s, "    {} d{k} : {name}{index};", n("float"));
                    *index += 1;
                }
            }
        }
    }
    s
}

fn random_case(r: &mut Rng) -> (String, &'static str) {
    match r.next(4) {
        0 => (format!("struct O {{\n{}}};\nO main(float4 p : POSITION) {{ O o = (O)0; return o; }}\n", random_fields(r, "vsout")), "vs_6_6"),
        1 => (format!("struct I {{\n{}}};\nfloat4 main(I i) : SV_Target {{ return 0; }}\n", random_fields(r, "psin")), "ps_6_6"),
        2 => (format!("struct V {{\n{}}};\n[maxvertexcount(1)]\nvoid main(point V v[1], inout PointStream<V> s) {{ s.Append(v[0]); }}\n", random_fields(r, "gs")), "gs_6_6"),
        _ => (
            format!(
                "struct V {{\n{}}};\nstruct P {{\n    uint id : ID;\n{}}};\n[outputtopology(\"triangle\")] [numthreads(1, 1, 1)]\nvoid main(out vertices V v[3], out primitives P p[1], out indices uint3 t[1]) {{ SetMeshOutputCounts(3, 1); V o = (V)0; v[0] = o; P q = (P)0; p[0] = q; t[0] = uint3(0, 1, 2); }}\n",
                random_fields(r, "msv"),
                random_fields(r, "msp")
            ),
            "ms_6_6",
        ),
    }
}

/// Random signatures against DXC itself (needs DXC): registers and columns must be DXC's
/// whatever the mix of widths, interpolation modes, arrays, clip / cull distances and system
/// values, and what DXC rejects must not compile here either. `XCL_DXIL_STAGE_FUZZ` sets the
/// number of cases (400 without it).
#[test]
fn random_signatures_match_dxc() {
    let Some(dxc) = dxc_dir() else {
        eprintln!("stage_signatures: XCL_DXIL_DXC is not set: skipping the random signatures");
        return;
    };
    let n: u32 = std::env::var("XCL_DXIL_STAGE_FUZZ").ok().and_then(|s| s.parse().ok()).unwrap_or(400);
    let mut r = Rng(0x5EED);
    let (mut failed, mut rejected) = (Vec::new(), 0);
    for i in 0..n {
        let (hlsl, profile) = random_case(&mut r);
        let expected = dxc_text(&dxc, &hlsl, profile, &format!("fuzz{i}"));
        let got = ours(&hlsl, profile);
        let ok = match (&got, expected.as_str()) {
            (Err(Error::Source { .. }), "error") => {
                rejected += 1;
                true
            }
            (Ok(text), expected) => *text == without_masks(expected),
            _ => false,
        };
        if !ok && failed.len() < 6 {
            failed.push(format!("case {i} ({profile}):\n{hlsl}{}\n-- DXC:\n{}", got.unwrap_or_else(|e| format!("{e:?}")), without_masks(&expected)));
        } else if !ok {
            failed.push(format!("case {i}"));
        }
    }
    eprintln!("stage_signatures: {n} random signatures, {rejected} rejected by DXC and by us");
    assert!(failed.is_empty(), "{} of {n} random signatures differ:\n{}", failed.len(), failed.join("\n\n"));
}
