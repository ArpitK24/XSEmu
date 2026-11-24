//! Fixture tests for [`xcl_dxil::hlsl::resources`]: every case is one resource access written in a
//! small notation (see [`cases`]). From it the test derives
//!
//! * an HLSL snippet, which DXC (the oracle, `-HV 2021 -Gis -O1`) compiles when `XCL_DXIL_DXC`
//!   points at the `dxc-linux` folder, and
//! * the same access built through `Emit` and the resources API,
//!
//! and compares both in a canonical form: the calls with an effect or a result that is read
//! from the device (loads, samples, stores, atomics) in order, operands expanded as
//! expression trees, so that value numbering, the place of pure instructions and common
//! subexpression elimination do not matter but every dx.op, operand, constant and conversion
//! does. DXC's canonical output is kept in `src/hlsl/resources/golden.txt` so the tests run
//! without DXC; with DXC present the golden copy is checked against it (`XCL_DXIL_BLESS=1`
//! rewrites the file).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use xcl_dxil::hlsl::ast::{Register, ResKind};
use xcl_dxil::hlsl::emit::Emit;
use xcl_dxil::hlsl::resources;
use xcl_dxil::hlsl::types::{HTy, ResTy, Sk, Val};
use xcl_dxil::ir::{flag, BinOp, Body, CastOp, Const, Module, Op, Ty, TypeDef, Value, ValueKind};

const GOLDEN: &str = include_str!("../src/hlsl/resources/golden.txt");

// ---------------------------------------------------------------------------------------
// the case notation

#[derive(Clone, Debug, PartialEq)]
enum DeclTy {
    Res(ResTy),
    Sampler { comparison: bool },
}

#[derive(Clone, Debug)]
struct Decl {
    name: String,
    /// the HLSL type as written (with `globallycoherent`)
    hlsl: String,
    ty: DeclTy,
    /// None: from the descriptor heap
    register: Option<Register>,
    nonuniform: bool,
}

#[derive(Clone, Debug)]
enum Arg {
    /// an input of `n` components read from `io`
    In(Sk, u8),
    /// a declared resource or sampler
    Obj(usize),
    /// an `out` argument
    Out(Sk, u8),
    /// a constant: constructor text and values
    Const(Sk, Vec<f64>, String),
}

#[derive(Clone, Debug)]
enum Stmt {
    /// `ret = Obj.Method(args)`
    Method { ret: Option<(Sk, u8)>, obj: usize, name: String, args: Vec<Arg> },
    /// `ret = Obj[coord]`
    IndexLoad { ret: (Sk, u8), obj: usize, coord: Arg },
    /// `Obj[coord] = value`
    IndexStore { obj: usize, coord: Arg, value: Arg },
    /// `Interlocked*(Obj[coord], args)`
    IndexAtomic { obj: usize, coord: Arg, name: String, args: Vec<Arg> },
}

#[derive(Clone, Debug)]
struct Case {
    text: String,
    pixel: bool,
    decls: Vec<Decl>,
    stmt: Stmt,
}

fn split_top(s: &str, sep: char) -> Vec<String> {
    let (mut depth, mut out, mut cur) = (0i32, Vec::new(), String::new());
    for c in s.chars() {
        match c {
            '(' | '<' | '[' | '{' => depth += 1,
            ')' | '>' | ']' | '}' => depth -= 1,
            _ => {}
        }
        if c == sep && depth == 0 {
            out.push(cur.trim().to_string());
            cur.clear();
        } else {
            cur.push(c);
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn parse_num_ty(s: &str) -> Option<(Sk, u8)> {
    let (k, rest) = if let Some(r) = s.strip_prefix("sl") {
        (Sk::Int64, r)
    } else if let Some(r) = s.strip_prefix('l') {
        (Sk::Uint64, r)
    } else if let Some(r) = s.strip_prefix('f') {
        (Sk::Float, r)
    } else if let Some(r) = s.strip_prefix('i') {
        (Sk::Int, r)
    } else if let Some(r) = s.strip_prefix('u') {
        (Sk::Uint, r)
    } else {
        return None;
    };
    let n: u8 = rest.parse().ok()?;
    (1..=4).contains(&n).then_some((k, n))
}

fn hlsl_scalar(k: Sk) -> &'static str {
    match k {
        Sk::Float => "float",
        Sk::Int => "int",
        Sk::Uint => "uint",
        Sk::Int64 => "int64_t",
        Sk::Uint64 => "uint64_t",
        _ => panic!("no fixture type for {k:?}"),
    }
}

fn hlsl_ty(k: Sk, n: u8) -> String {
    if n == 1 {
        hlsl_scalar(k).to_string()
    } else {
        format!("{}{n}", hlsl_scalar(k))
    }
}

/// `Texture2DMS<float4, 4>` → the ResTy, through [`resources::res_ty`].
fn parse_res_ty(s: &str, coherent: bool) -> DeclTy {
    match s {
        "SamplerState" => return DeclTy::Sampler { comparison: false },
        "SamplerComparisonState" => return DeclTy::Sampler { comparison: true },
        _ => {}
    }
    let (base, targs) = match s.find('<') {
        Some(i) => (&s[..i], Some(&s[i + 1..s.len() - 1])),
        None => (s, None),
    };
    let kind = match base {
        "Texture1D" => ResKind::Texture1D,
        "Texture1DArray" => ResKind::Texture1DArray,
        "Texture2D" => ResKind::Texture2D,
        "Texture2DArray" => ResKind::Texture2DArray,
        "Texture2DMS" => ResKind::Texture2DMS,
        "Texture2DMSArray" => ResKind::Texture2DMSArray,
        "Texture3D" => ResKind::Texture3D,
        "TextureCube" => ResKind::TextureCube,
        "TextureCubeArray" => ResKind::TextureCubeArray,
        "RWTexture1D" => ResKind::RWTexture1D,
        "RWTexture1DArray" => ResKind::RWTexture1DArray,
        "RWTexture2D" => ResKind::RWTexture2D,
        "RWTexture2DArray" => ResKind::RWTexture2DArray,
        "RWTexture3D" => ResKind::RWTexture3D,
        "Buffer" => ResKind::Buffer,
        "RWBuffer" => ResKind::RWBuffer,
        "ByteAddressBuffer" => ResKind::ByteAddressBuffer,
        "RWByteAddressBuffer" => ResKind::RWByteAddressBuffer,
        "StructuredBuffer" => ResKind::StructuredBuffer,
        "RWStructuredBuffer" => ResKind::RWStructuredBuffer,
        other => panic!("fixture resource type {other}"),
    };
    let (mut elem, mut samples) = (None, None);
    if let Some(t) = targs {
        let parts = split_top(t, ',');
        let e = parts[0].as_str();
        let (name, n) = match e.chars().last() {
            Some(d @ '1'..='4') if !e.ends_with("_t") => (&e[..e.len() - 1], d as u8 - b'0'),
            _ => (e, 1),
        };
        let k = match name {
            "float" => Sk::Float,
            "int" => Sk::Int,
            "uint" => Sk::Uint,
            "uint64_t" => Sk::Uint64,
            "int64_t" => Sk::Int64,
            "double" => Sk::Double,
            other => panic!("fixture element type {other}"),
        };
        elem = Some(HTy::vec(k, n));
        samples = parts.get(1).map(|p| p.parse().expect("sample count"));
    }
    let e = Emit::new();
    DeclTy::Res(resources::res_ty(kind, elem, samples, coherent, &e.types).unwrap_or_else(|err| panic!("res_ty({s}): {err}")))
}

fn parse_case(text: &str) -> Case {
    let parts = split_top(text, ';');
    let pixel = match parts[0].as_str() {
        "cs" => false,
        "ps" => true,
        other => panic!("stage {other} in {text}"),
    };
    let mut decls = Vec::new();
    for d in &parts[1..parts.len() - 1] {
        let (lhs, ty) = d.split_once('=').unwrap_or_else(|| panic!("declaration {d}"));
        let (name, register) = match lhs.trim().split_once('@') {
            Some((n, r)) => {
                let (r, space) = match r.split_once(',') {
                    Some((r, s)) => (r, s.parse().expect("space")),
                    None => (r, 0),
                };
                (n, Some(Register { class: r.as_bytes()[0], index: r[1..].parse().expect("register"), space }))
            }
            None => (lhs.trim(), None),
        };
        let ty = ty.trim();
        let (nonuniform, ty) = match ty.strip_prefix("uniform ") {
            Some(t) => (false, t),
            None => (true, ty),
        };
        let coherent = ty.starts_with("globallycoherent ");
        decls.push(Decl {
            name: name.to_string(),
            hlsl: ty.to_string(),
            ty: parse_res_ty(ty.trim_start_matches("globallycoherent "), coherent),
            register,
            nonuniform,
        });
    }
    let obj = |n: &str| decls.iter().position(|d| d.name == n).unwrap_or_else(|| panic!("no declaration {n} in {text}"));
    let arg = |a: &str| -> Arg {
        if let Some(o) = a.strip_prefix("out:") {
            let (k, n) = parse_num_ty(o).unwrap_or_else(|| panic!("out argument {a}"));
            return Arg::Out(k, n);
        }
        if let Some(c) = a.strip_prefix("c:") {
            let open = c.find('(').expect("constant constructor");
            let name = &c[..open];
            let name = name.trim_end_matches(|ch: char| ch.is_ascii_digit());
            let k = match name {
                "float" => Sk::Float,
                "int" => Sk::Int,
                "uint" => Sk::Uint,
                other => panic!("constant type {other}"),
            };
            let vals = c[open + 1..c.len() - 1].split(',').map(|v| v.trim().trim_end_matches(['f', 'u']).parse().expect("constant")).collect();
            return Arg::Const(k, vals, c.to_string());
        }
        if let Some((k, n)) = parse_num_ty(a) {
            return Arg::In(k, n);
        }
        Arg::Obj(obj(a))
    };
    let s = parts.last().unwrap().as_str();
    let stmt = if s.starts_with("Interlocked") {
        let open = s.find('(').unwrap();
        let args = split_top(&s[open + 1..s.len() - 1], ',');
        let (o, c) = args[0].split_once('[').unwrap();
        Stmt::IndexAtomic { obj: obj(o), coord: arg(c.trim_end_matches(']')), name: s[..open].to_string(), args: args[1..].iter().map(|a| arg(a)).collect() }
    } else {
        let (lhs, rhs) = s.split_once(" = ").unwrap_or_else(|| panic!("statement {s}"));
        if let Some((o, c)) = lhs.split_once('[') {
            Stmt::IndexStore { obj: obj(o), coord: arg(c.trim_end_matches(']')), value: arg(rhs) }
        } else if let Some(open) = rhs.find('(') {
            let (o, name) = rhs[..open].split_once('.').unwrap();
            Stmt::Method { ret: parse_num_ty(lhs), obj: obj(o), name: name.to_string(), args: split_top(&rhs[open + 1..rhs.len() - 1], ',').iter().map(|a| arg(a)).collect() }
        } else {
            let (o, c) = rhs.split_once('[').unwrap();
            Stmt::IndexLoad { ret: parse_num_ty(lhs).expect("result type"), obj: obj(o), coord: arg(c.trim_end_matches(']')) }
        }
    };
    Case { text: text.to_string(), pixel, decls, stmt }
}

/// Byte offset in `io` of input `k` / in `res` of `out` argument `k` (the result is at 0).
fn slot(k: usize) -> u32 {
    64 + 16 * k as u32
}

// ---------------------------------------------------------------------------------------
// the cases

/// Every access under test, in the notation `stage; declarations; statement`:
///
/// * stage: `cs` or `ps` (implicit-derivative sampling),
/// * a declaration `T=Texture2D<float4>` is taken from the descriptor heap with a
///   `NonUniformResourceIndex` (`T=uniform ...`: without); `T@t0=...` or `S@s1,1=...` is
///   bound to a register (and space),
/// * arguments: `f2` / `i3` / `u1` / `l1` / `sl1` are float / int / uint / uint64 / int64
///   inputs of that many components, `out:u1` an out argument, `c:int2(1, -2)` a constant, a
///   declared name the resource or sampler.
fn cases() -> Vec<String> {
    let mut c: Vec<String> = Vec::new();
    let mut add = |s: String| c.push(s);

    // --- raw buffers: the dialect's memory (heap, non-uniform, globally coherent) and io / res (bound)
    let mem = "B=globallycoherent RWByteAddressBuffer";
    for decl in [mem, "B=RWByteAddressBuffer", "B=uniform RWByteAddressBuffer", "B=uniform globallycoherent RWByteAddressBuffer", "B@u2=RWByteAddressBuffer", "B@u3,2=globallycoherent RWByteAddressBuffer", "B=ByteAddressBuffer", "B@t4=ByteAddressBuffer"] {
        add(format!("cs; {decl}; u1 = B.Load(u1)"));
    }
    for (n, name) in [(2, "Load2"), (3, "Load3"), (4, "Load4")] {
        add(format!("cs; {mem}; u{n} = B.{name}(u1)"));
    }
    add(format!("cs; {mem}; u1 = B.Load(i1)"));
    add(format!("cs; {mem}; u1 = B.Load(c:uint(64))"));
    add(format!("cs; {mem}; u1 = B.Load(u1, out:u1)"));
    add(format!("cs; {mem}; u4 = B.Load4(u1, out:u1)"));
    for (n, name) in [(1, "Store"), (2, "Store2"), (3, "Store3"), (4, "Store4")] {
        add(format!("cs; {mem}; void = B.{name}(u1, u{n})"));
    }
    add("cs; B@u2=RWByteAddressBuffer; void = B.Store(u1, u1)".to_string());
    add(format!("cs; {mem}; void = B.Store(u1, i1)"));
    add(format!("cs; {mem}; void = B.Store(u1, f1)"));
    add(format!("cs; {mem}; void = B.Store(c:uint(8), c:uint(7))"));
    add(format!("cs; {mem}; void = B.GetDimensions(out:u1)"));
    for op in ["Add", "And", "Or", "Xor", "Min", "Max"] {
        add(format!("cs; {mem}; void = B.Interlocked{op}(u1, u1)"));
        add(format!("cs; {mem}; void = B.Interlocked{op}(u1, u1, out:u1)"));
        add(format!("cs; {mem}; void = B.Interlocked{op}64(u1, l1)"));
        add(format!("cs; {mem}; void = B.Interlocked{op}64(u1, l1, out:l1)"));
    }
    for op in ["Min", "Max"] {
        add(format!("cs; {mem}; void = B.Interlocked{op}(u1, i1)"));
        add(format!("cs; {mem}; void = B.Interlocked{op}(u1, i1, out:i1)"));
        add(format!("cs; {mem}; void = B.Interlocked{op}64(u1, sl1)"));
        add(format!("cs; {mem}; void = B.Interlocked{op}64(u1, sl1, out:sl1)"));
    }
    add(format!("cs; {mem}; void = B.InterlockedExchange(u1, u1, out:u1)"));
    add(format!("cs; {mem}; void = B.InterlockedExchange64(u1, l1, out:l1)"));
    add(format!("cs; {mem}; void = B.InterlockedCompareExchange(u1, u1, u1, out:u1)"));
    add(format!("cs; {mem}; void = B.InterlockedCompareExchange64(u1, l1, l1, out:l1)"));
    add(format!("cs; {mem}; void = B.InterlockedCompareStore(u1, u1, u1)"));
    add(format!("cs; {mem}; void = B.InterlockedCompareStore64(u1, l1, l1)"));
    add("cs; B@u2=RWByteAddressBuffer; void = B.InterlockedAdd(u1, u1, out:u1)".to_string());

    // --- structured and typed buffers
    for decl in ["B@u2=RWStructuredBuffer<uint>", "B=RWStructuredBuffer<uint>", "B=StructuredBuffer<uint>"] {
        add(format!("cs; {decl}; u1 = B[u1]"));
    }
    add("cs; B@u2=RWStructuredBuffer<uint>; B[u1] = u1".to_string());
    add("cs; B=RWStructuredBuffer<float4>; B[u1] = f4".to_string());
    add("cs; B=RWStructuredBuffer<int2>; B[i1] = i2".to_string());
    add("cs; B=StructuredBuffer<float3>; f3 = B[u1]".to_string());
    add("cs; B=StructuredBuffer<float3>; f3 = B.Load(i1)".to_string());
    add("cs; B=StructuredBuffer<uint2>; u2 = B.Load(i1, out:u1)".to_string());
    add("cs; B=StructuredBuffer<uint64_t>; l1 = B[u1]".to_string());
    add("cs; B=RWStructuredBuffer<uint64_t>; B[u1] = l1".to_string());
    add("cs; B=RWStructuredBuffer<float2>; void = B.GetDimensions(out:u1, out:u1)".to_string());
    add("cs; B=RWStructuredBuffer<uint>; InterlockedAdd(B[u1], u1, out:u1)".to_string());
    add("cs; B=RWStructuredBuffer<int>; InterlockedMin(B[u1], i1, out:i1)".to_string());
    add("cs; B=RWStructuredBuffer<uint>; InterlockedCompareExchange(B[u1], u1, u1, out:u1)".to_string());
    for (ty, r) in [("float4", "f4"), ("uint", "u1"), ("int2", "i2"), ("float3", "f3")] {
        add(format!("cs; B=Buffer<{ty}>; {r} = B.Load(i1)"));
        add(format!("cs; B=Buffer<{ty}>; {r} = B[u1]"));
        add(format!("cs; B=RWBuffer<{ty}>; {r} = B[u1]"));
        add(format!("cs; B=RWBuffer<{ty}>; B[u1] = {r}"));
    }
    add("cs; B=Buffer<float4>; f4 = B.Load(i1, out:u1)".to_string());
    add("cs; B@t3=Buffer<uint4>; u4 = B.Load(i1)".to_string());
    add("cs; B=Buffer<float4>; void = B.GetDimensions(out:u1)".to_string());
    add("cs; B=RWBuffer<uint>; InterlockedAdd(B[u1], u1, out:u1)".to_string());
    add("cs; B=RWBuffer<int>; InterlockedMax(B[u1], i1)".to_string());
    add("cs; B=RWBuffer<uint>; InterlockedCompareStore(B[u1], u1, u1)".to_string());

    // --- sampling: (kind, coordinates, an offset constant, gradient components, sizes of GetDimensions)
    let kinds = [
        ("Texture1D", 1, Some("c:int(1)"), 1, 1),
        ("Texture1DArray", 2, Some("c:int(-2)"), 1, 2),
        ("Texture2D", 2, Some("c:int2(1, -2)"), 2, 2),
        ("Texture2DArray", 3, Some("c:int2(-8, 7)"), 2, 3),
        ("Texture3D", 3, Some("c:int3(1, 2, 3)"), 3, 3),
        ("TextureCube", 3, None, 3, 2),
        ("TextureCubeArray", 4, None, 3, 3),
    ];
    let ss = "S=SamplerState";
    let sc = "S=SamplerComparisonState";
    for (kind, nc, offset, ng, sizes) in kinds {
        let t = format!("T={kind}<float4>");
        let with = |args: &str, extra: &[&str]| {
            let mut s = args.to_string();
            for x in extra {
                s.push_str(", ");
                s.push_str(x);
            }
            s
        };
        // the optional tails: offset (none for cubes), clamp, status
        let off: Vec<&str> = offset.into_iter().collect();
        let zero = match kind {
            "Texture1D" | "Texture1DArray" => vec!["c:int(0)"],
            "Texture2D" | "Texture2DArray" => vec!["c:int2(0, 0)"],
            "Texture3D" => vec!["c:int3(0, 0, 0)"],
            _ => vec![],
        };
        let mut off_clamp = off.clone();
        off_clamp.push("f1");
        let mut zero_clamp = zero.clone();
        zero_clamp.push("f1");
        let mut off_clamp_status = off_clamp.clone();
        off_clamp_status.push("out:u1");
        let mut off_status = off.clone();
        off_status.push("out:u1");

        add(format!("ps; {t}; {ss}; f4 = T.Sample({})", with(&format!("S, f{nc}"), &[])));
        add(format!("ps; {t}; {ss}; f4 = T.Sample({})", with(&format!("S, f{nc}"), &zero_clamp)));
        add(format!("ps; {t}; {ss}; f4 = T.Sample({})", with(&format!("S, f{nc}"), &off_clamp_status)));
        add(format!("ps; {t}; {ss}; f4 = T.SampleBias({})", with(&format!("S, f{nc}, f1"), &[])));
        add(format!("ps; {t}; {ss}; f4 = T.SampleBias({})", with(&format!("S, f{nc}, f1"), &zero_clamp)));
        add(format!("cs; {t}; {ss}; f4 = T.SampleLevel({})", with(&format!("S, f{nc}, f1"), &[])));
        add(format!("cs; {t}; {ss}; f4 = T.SampleLevel({})", with(&format!("S, f{nc}, c:float(0.0f)"), &[])));
        add(format!("cs; {t}; {ss}; f4 = T.SampleLevel({})", with(&format!("S, f{nc}, f1"), &off_status)));
        add(format!("cs; {t}; {ss}; f4 = T.SampleGrad({})", with(&format!("S, f{nc}, f{ng}, f{ng}"), &[])));
        add(format!("cs; {t}; {ss}; f4 = T.SampleGrad({})", with(&format!("S, f{nc}, f{ng}, f{ng}"), &zero_clamp)));
        if !off.is_empty() {
            add(format!("ps; {t}; {ss}; f4 = T.Sample({})", with(&format!("S, f{nc}"), &off)));
            add(format!("ps; {t}; {ss}; f4 = T.SampleBias({})", with(&format!("S, f{nc}, f1"), &off)));
            add(format!("cs; {t}; {ss}; f4 = T.SampleLevel({})", with(&format!("S, f{nc}, f1"), &off)));
            add(format!("cs; {t}; {ss}; f4 = T.SampleGrad({})", with(&format!("S, f{nc}, f{ng}, f{ng}"), &off)));
        }
        if kind != "Texture3D" {
            add(format!("ps; {t}; {sc}; f1 = T.SampleCmp({})", with(&format!("S, f{nc}, f1"), &[])));
            add(format!("ps; {t}; {sc}; f1 = T.SampleCmp({})", with(&format!("S, f{nc}, f1"), &zero_clamp)));
            add(format!("cs; {t}; {sc}; f1 = T.SampleCmpLevelZero({})", with(&format!("S, f{nc}, f1"), &[])));
            add(format!("cs; {t}; {sc}; f1 = T.SampleCmpLevelZero({})", with(&format!("S, f{nc}, f1"), &off_status)));
        }
        add(format!("ps; {t}; {ss}; f1 = T.CalculateLevelOfDetail(S, f{ng})"));
        add(format!("ps; {t}; {ss}; f1 = T.CalculateLevelOfDetailUnclamped(S, f{ng})"));

        // GetDimensions: (mip, sizes.., levels) and (sizes..), uint and float outs
        let outs = |k: &str, n: usize| vec![format!("out:{k}1"); n].join(", ");
        add(format!("cs; {t}; void = T.GetDimensions(u1, {})", outs("u", sizes + 1)));
        add(format!("cs; {t}; void = T.GetDimensions(c:uint(0), {})", outs("u", sizes + 1)));
        add(format!("cs; {t}; void = T.GetDimensions(u1, {})", outs("f", sizes + 1)));
        add(format!("cs; {t}; void = T.GetDimensions({})", outs("u", sizes)));
        add(format!("cs; {t}; void = T.GetDimensions({})", outs("f", sizes)));

        // gathers
        if matches!(kind, "Texture2D" | "Texture2DArray" | "TextureCube" | "TextureCubeArray") {
            for ch in ["", "Red", "Green", "Blue", "Alpha"] {
                add(format!("cs; {t}; {ss}; f4 = T.Gather{ch}(S, f{nc})"));
                add(format!("cs; {t}; {sc}; f4 = T.GatherCmp{ch}(S, f{nc}, f1)"));
            }
            add(format!("cs; T={kind}<uint4>; {ss}; u4 = T.GatherRed(S, f{nc})"));
            add(format!("cs; T={kind}<int4>; {ss}; i4 = T.GatherAlpha(S, f{nc})"));
            add(format!("cs; {t}; {ss}; f4 = T.GatherGreen({})", with(&format!("S, f{nc}"), &off_status)));
            add(format!("cs; {t}; {sc}; f4 = T.GatherCmpBlue({})", with(&format!("S, f{nc}, f1"), &off_status)));
            if !off.is_empty() {
                add(format!("cs; {t}; {ss}; f4 = T.Gather({})", with(&format!("S, f{nc}"), &off)));
                add(format!("cs; {t}; {ss}; f4 = T.GatherRed({})", with(&format!("S, f{nc}"), &off)));
                add(format!("cs; {t}; {sc}; f4 = T.GatherCmp({})", with(&format!("S, f{nc}, f1"), &off)));
                let four = "c:int2(0, 0), c:int2(1, 0), c:int2(0, 1), c:int2(1, 1)";
                add(format!("cs; {t}; {ss}; f4 = T.GatherRed(S, f{nc}, {four})"));
                add(format!("cs; {t}; {ss}; f4 = T.GatherAlpha(S, f{nc}, {four}, out:u1)"));
                add(format!("cs; {t}; {sc}; f4 = T.GatherCmpGreen(S, f{nc}, f1, {four})"));
            }
        }

        // loads (not cubes): location = coordinates + mip
        if !kind.starts_with("TextureCube") {
            for (ty, r) in [("float4", "f4"), ("uint4", "u4"), ("int4", "i4")] {
                add(format!("cs; T={kind}<{ty}>; {r} = T.Load(i{})", nc + 1));
                add(format!("cs; T={kind}<{ty}>; {r} = T[i{nc}]"));
            }
            add(format!("cs; T={kind}<float>; f1 = T.Load(u{})", nc + 1));
            add(format!("cs; T={kind}<uint2>; u2 = T.Load(i{})", nc + 1));
            add(format!("cs; T={kind}<float3>; f3 = T.Load(i{})", nc + 1));
            add(format!("cs; {t}; f4 = T.Load({})", with(&format!("i{}", nc + 1), &off)));
            add(format!("cs; {t}; f4 = T.Load({})", with(&format!("i{}", nc + 1), &off_status)));
        }
    }
    // the dialect's own shapes: an untemplated bound texture with a bound sampler, samplers of both
    // kinds bound in a space and from the heap without NonUniformResourceIndex
    add("ps; T@t0=Texture2D; S@s0=SamplerState; f4 = T.SampleLevel(S, f2, c:int(0))".to_string());
    add("cs; T=Texture2D<float4>; S@s3,1=SamplerState; f4 = T.SampleLevel(S, f2, f1)".to_string());
    add("cs; T=Texture2D<float4>; S@s2,1=SamplerComparisonState; f1 = T.SampleCmpLevelZero(S, f2, f1)".to_string());
    add("cs; T=uniform Texture2D<float4>; S=uniform SamplerState; f4 = T.SampleLevel(S, f2, f1)".to_string());
    add("cs; T@t7,3=Texture2DArray<uint4>; u4 = T.Load(i4)".to_string());
    add("ps; T=Texture2D<float2>; S=SamplerState; f2 = T.Sample(S, f2)".to_string());
    add("ps; T=Texture2D<float>; S=SamplerState; f1 = T.SampleBias(S, f2, c:float(20.0f))".to_string());
    add("ps; T=Texture2D<float>; S=SamplerState; f1 = T.SampleBias(S, f2, c:float(-17.5f))".to_string());

    // --- multisampled textures
    for (kind, nc) in [("Texture2DMS", 2), ("Texture2DMSArray", 3)] {
        for (ty, r) in [("float4", "f4"), ("uint4", "u4"), ("int4", "i4")] {
            add(format!("cs; T={kind}<{ty}>; {r} = T.Load(i{nc}, i1)"));
        }
        add(format!("cs; T={kind}<float4>; f4 = T.Load(i{nc}, c:int(0))"));
        add(format!("cs; T={kind}<float4>; f4 = T.Load(i{nc}, i1, c:int2(1, -1))"));
        add(format!("cs; T={kind}<float4>; f4 = T.Load(i{nc}, i1, c:int2(0, 0), out:u1)"));
        add(format!("cs; T={kind}<float4>; f4 = T[i{nc}]"));
        add(format!("cs; T={kind}<float4, 4>; f4 = T.Load(i{nc}, i1)"));
        let outs = |k: &str| vec![format!("out:{k}1"); nc + 1].join(", ");
        add(format!("cs; T={kind}<float4>; void = T.GetDimensions({})", outs("u")));
        add(format!("cs; T={kind}<float4>; void = T.GetDimensions({})", outs("f")));
    }
    add("cs; T@t2=Texture2DMS<float4, 8>; f4 = T.Load(i2, i1)".to_string());

    // --- read-write textures: loads, stores, dimensions, texel atomics
    for (kind, nc) in [("RWTexture1D", 1), ("RWTexture1DArray", 2), ("RWTexture2D", 2), ("RWTexture2DArray", 3), ("RWTexture3D", 3)] {
        for (ty, r) in [("float4", "f4"), ("uint4", "u4"), ("int4", "i4"), ("uint", "u1"), ("int", "i1"), ("float", "f1"), ("float2", "f2"), ("uint3", "u3")] {
            add(format!("cs; T={kind}<{ty}>; T[i{nc}] = {r}"));
            add(format!("cs; T={kind}<{ty}>; {r} = T[i{nc}]"));
        }
        add(format!("cs; T={kind}<float4>; f4 = T.Load(i{nc})"));
        add(format!("cs; T={kind}<uint4>; u4 = T.Load(i{nc}, out:u1)"));
        add(format!("cs; T={kind}<uint4>; T[u{nc}] = u4"));
        add(format!("cs; T={kind}<float4>; void = T.GetDimensions({})", vec!["out:u1"; nc].join(", ")));
        add(format!("cs; T={kind}<float4>; void = T.GetDimensions({})", vec!["out:f1"; nc].join(", ")));
        for op in ["Add", "And", "Or", "Xor", "Min", "Max", "Exchange"] {
            add(format!("cs; T={kind}<uint>; Interlocked{op}(T[i{nc}], u1, out:u1)"));
            add(format!("cs; T={kind}<int>; Interlocked{op}(T[i{nc}], i1, out:i1)"));
            if op != "Exchange" {
                add(format!("cs; T={kind}<uint>; Interlocked{op}(T[i{nc}], u1)"));
            }
        }
        add(format!("cs; T={kind}<uint>; InterlockedCompareExchange(T[i{nc}], u1, u1, out:u1)"));
        add(format!("cs; T={kind}<int>; InterlockedCompareExchange(T[i{nc}], i1, i1, out:i1)"));
        add(format!("cs; T={kind}<uint>; InterlockedCompareStore(T[i{nc}], u1, u1)"));
    }
    add("cs; T@u2=RWTexture2D<float4>; T[i2] = f4".to_string());
    add("cs; T@u2=RWTexture2D<uint>; InterlockedAdd(T[i2], u1, out:u1)".to_string());
    c
}

// ---------------------------------------------------------------------------------------
// the HLSL of a case

fn hlsl_of(c: &Case) -> String {
    let mut h = String::from("RWByteAddressBuffer io : register(u0);\nRWByteAddressBuffer res : register(u1);\n");
    for d in &c.decls {
        if let Some(r) = d.register {
            let _ = writeln!(h, "{} {} : register({}{}, space{});", d.hlsl, d.name, r.class as char, r.index, r.space);
        }
    }
    h.push_str(if c.pixel { "void main() {\n" } else { "[numthreads(1, 1, 1)]\nvoid main() {\n" });
    for (j, d) in c.decls.iter().enumerate() {
        if d.register.is_some() {
            continue;
        }
        let heap = if d.ty == (DeclTy::Sampler { comparison: false }) || d.ty == (DeclTy::Sampler { comparison: true }) { "SamplerDescriptorHeap" } else { "ResourceDescriptorHeap" };
        let index = format!("io.Load({}u)", 4 * j);
        let index = if d.nonuniform { format!("NonUniformResourceIndex({index})") } else { index };
        let _ = writeln!(h, "    {} {} = {heap}[{index}];", d.hlsl, d.name);
    }
    let mut stores = String::new();
    let mut k = 0usize;
    let mut arg = |a: &Arg, h: &mut String, stores: &mut String| -> String {
        let me = k;
        k += 1;
        match a {
            Arg::Obj(o) => c.decls[*o].name.clone(),
            Arg::Const(_, _, text) => text.clone(),
            Arg::In(sk, n) => {
                let ty = hlsl_ty(*sk, *n);
                let off = slot(me);
                let load = |o: u32| if *n == 1 || sk.is_64() { format!("io.Load({o}u)") } else { format!("io.Load{n}({o}u)") };
                let init = match sk {
                    Sk::Float => format!("asfloat({})", load(off)),
                    Sk::Int => format!("({ty}){}", load(off)),
                    Sk::Uint => load(off),
                    _ => {
                        assert_eq!(*n, 1, "64-bit fixture inputs are scalars");
                        format!("({ty})((uint64_t){} | ((uint64_t){} << 32))", load(off), load(off + 4))
                    }
                };
                let _ = writeln!(h, "    {ty} a{me} = {init};");
                format!("a{me}")
            }
            Arg::Out(sk, n) => {
                let _ = writeln!(h, "    {} o{me};", hlsl_ty(*sk, *n));
                stores.push_str(&store_hlsl(slot(me), &format!("o{me}"), *sk, *n));
                format!("o{me}")
            }
        }
    };
    let obj_name = |o: usize| c.decls[o].name.clone();
    match &c.stmt {
        Stmt::Method { ret, obj, name, args } => {
            let a: Vec<String> = args.iter().map(|a| arg(a, &mut h, &mut stores)).collect();
            let call = format!("{}.{name}({})", obj_name(*obj), a.join(", "));
            match ret {
                Some((sk, n)) => {
                    let _ = writeln!(h, "    {} r = {call};", hlsl_ty(*sk, *n));
                    h.push_str(&store_hlsl(0, "r", *sk, *n));
                }
                None => {
                    let _ = writeln!(h, "    {call};");
                }
            }
        }
        Stmt::IndexLoad { ret, obj, coord } => {
            let a = arg(coord, &mut h, &mut stores);
            let _ = writeln!(h, "    {} r = {}[{a}];", hlsl_ty(ret.0, ret.1), obj_name(*obj));
            h.push_str(&store_hlsl(0, "r", ret.0, ret.1));
        }
        Stmt::IndexStore { obj, coord, value } => {
            let a = arg(coord, &mut h, &mut stores);
            let v = arg(value, &mut h, &mut stores);
            let _ = writeln!(h, "    {}[{a}] = {v};", obj_name(*obj));
        }
        Stmt::IndexAtomic { obj, coord, name, args } => {
            let a = arg(coord, &mut h, &mut stores);
            let rest: Vec<String> = args.iter().map(|a| arg(a, &mut h, &mut stores)).collect();
            let _ = writeln!(h, "    {name}({}[{a}], {});", obj_name(*obj), rest.join(", "));
        }
    }
    h.push_str(&stores);
    h.push_str("}\n");
    h
}

fn store_hlsl(off: u32, v: &str, sk: Sk, n: u8) -> String {
    let suffix = if n == 1 { String::new() } else { n.to_string() };
    match sk {
        Sk::Float => format!("    res.Store{suffix}({off}u, asuint({v}));\n"),
        Sk::Int => format!("    res.Store{suffix}({off}u, ({}){v});\n", hlsl_ty(Sk::Uint, n)),
        Sk::Uint => format!("    res.Store{suffix}({off}u, {v});\n"),
        _ => format!("    res.Store2({off}u, uint2((uint){v}, (uint)({v} >> 32)));\n"),
    }
}

// ---------------------------------------------------------------------------------------
// the same access through Emit

struct Built {
    e: Emit,
}

fn rw_bab() -> ResTy {
    let e = Emit::new();
    resources::res_ty(ResKind::RWByteAddressBuffer, None, None, false, &e.types).unwrap()
}

fn build(c: &Case) -> xcl_dxil::Result<Built> {
    let mut e = Emit::new();
    let mut records = Vec::new();
    let bab = rw_bab();
    let io_i = resources::declare(&mut records, "io", HTy::Res(bab), Some(&Register { class: b'u', index: 0, space: 0 }))?;
    let res_i = resources::declare(&mut records, "res", HTy::Res(bab), Some(&Register { class: b'u', index: 1, space: 0 }))?;
    let io = Val::new(HTy::Res(bab), &[resources::bound_handle(&mut e, &records[io_i], &bab)?]);
    let res = Val::new(HTy::Res(bab), &[resources::bound_handle(&mut e, &records[res_i], &bab)?]);
    let load = |e: &mut Emit, off: u32, n: u8| -> xcl_dxil::Result<Val> {
        let a = Val::scalar(Sk::Uint, e.c_u32(off));
        let name = if n == 1 { "Load".to_string() } else { format!("Load{n}") };
        Ok(resources::method(e, &io, &name, &[a])?.ret)
    };
    let store = |e: &mut Emit, off: u32, v: &Val| -> xcl_dxil::Result<()> {
        let HTy::Num(sk, _, n) = v.ty else { panic!("store of {:?}", v.ty) };
        let comps: Vec<Value> = match sk {
            Sk::Float => v.comps().iter().map(|&x| e.cast(CastOp::BitCast, Ty::I32, x)).collect(),
            Sk::Int | Sk::Uint => v.comps().to_vec(),
            _ => {
                let x = v.one();
                let lo = e.cast(CastOp::Trunc, Ty::I32, x);
                let c32 = e.m.const_i64(32);
                let hi = e.bin(BinOp::LShr, x, c32);
                vec![lo, e.cast(CastOp::Trunc, Ty::I32, hi)]
            }
        };
        let a = Val::scalar(Sk::Uint, e.c_u32(off));
        let name = if comps.len() == 1 { "Store".to_string() } else { format!("Store{}", comps.len()) };
        let _ = n;
        resources::method(e, &res, &name, &[a, Val::new(HTy::vec(Sk::Uint, comps.len() as u8), &comps)])?;
        Ok(())
    };
    let mut objs = Vec::new();
    for d in &c.decls {
        if let Some(r) = &d.register {
            let (ty, rty) = match &d.ty {
                DeclTy::Res(t) => (HTy::Res(*t), Some(*t)),
                DeclTy::Sampler { comparison } => (HTy::Sampler { comparison: *comparison }, None),
            };
            let i = resources::declare(&mut records, &d.name, ty, Some(r))?;
            let h = match rty {
                Some(t) => resources::bound_handle(&mut e, &records[i], &t)?,
                None => resources::bound_sampler(&mut e, &records[i])?,
            };
            objs.push(Val::new(ty, &[h]));
        } else {
            objs.push(Val::void());
        }
    }
    for (j, d) in c.decls.iter().enumerate() {
        if d.register.is_some() {
            continue;
        }
        let index = load(&mut e, 4 * j as u32, 1)?.one();
        objs[j] = match &d.ty {
            DeclTy::Res(t) => Val::new(HTy::Res(*t), &[resources::heap_handle(&mut e, index, d.nonuniform, Some(t), false)?]),
            DeclTy::Sampler { comparison } => Val::new(HTy::Sampler { comparison: *comparison }, &[resources::heap_handle(&mut e, index, d.nonuniform, None, *comparison)?]),
        };
    }
    let mut k = 0usize;
    let mut outs: Vec<(usize, usize)> = Vec::new();
    let mut arg = |a: &Arg, e: &mut Emit, at: usize, outs: &mut Vec<(usize, usize)>| -> xcl_dxil::Result<Val> {
        let me = k;
        k += 1;
        Ok(match a {
            Arg::Obj(o) => objs[*o].clone(),
            Arg::Const(sk, vals, _) => {
                let comps: Vec<Value> = vals
                    .iter()
                    .map(|&v| match sk {
                        Sk::Float => e.c_f32(v as f32),
                        _ => e.c_u32(v as i64 as u32),
                    })
                    .collect();
                Val::new(HTy::vec(*sk, comps.len() as u8), &comps)
            }
            Arg::In(sk, n) => {
                let off = slot(me);
                match sk {
                    Sk::Float => {
                        let raw = load(e, off, *n)?;
                        let comps: Vec<Value> = raw.comps().iter().map(|&x| e.cast(CastOp::BitCast, Ty::F32, x)).collect();
                        Val::new(HTy::vec(Sk::Float, *n), &comps)
                    }
                    Sk::Int | Sk::Uint => {
                        let raw = load(e, off, *n)?;
                        Val::new(HTy::vec(*sk, *n), raw.comps())
                    }
                    _ => {
                        let lo = load(e, off, 1)?.one();
                        let hi = load(e, off + 4, 1)?.one();
                        let lo = e.cast(CastOp::ZExt, Ty::I64, lo);
                        let hi = e.cast(CastOp::ZExt, Ty::I64, hi);
                        let c32 = e.m.const_i64(32);
                        let hi = e.bin(BinOp::Shl, hi, c32);
                        Val::scalar(*sk, e.bin(BinOp::Or, hi, lo))
                    }
                }
            }
            Arg::Out(sk, n) => {
                outs.push((at, me));
                // the current value of an out argument is ignored: undef of its type
                let u = e.m.undef(sk.ir());
                Val::new(HTy::vec(*sk, *n), &vec![u; *n as usize])
            }
        })
    };
    let check_ret = |got: &Val, want: (Sk, u8)| {
        assert_eq!(got.ty, HTy::vec(want.0, want.1), "result type of {}", c.text);
    };
    let called = match &c.stmt {
        Stmt::Method { ret, obj, name, args } => {
            let mut a = Vec::new();
            for (i, x) in args.iter().enumerate() {
                a.push(arg(x, &mut e, i, &mut outs)?);
            }
            let called = resources::method(&mut e, &objs[*obj], name, &a)?;
            match ret {
                Some(want) => {
                    check_ret(&called.ret, *want);
                    store(&mut e, 0, &called.ret)?;
                }
                None => assert_eq!(called.ret.ty, HTy::Void, "result type of {}", c.text),
            }
            called.outs
        }
        Stmt::IndexLoad { ret, obj, coord } => {
            let a = arg(coord, &mut e, 0, &mut outs)?;
            let v = resources::index_load(&mut e, &objs[*obj], &a)?;
            check_ret(&v, *ret);
            store(&mut e, 0, &v)?;
            Vec::new()
        }
        Stmt::IndexStore { obj, coord, value } => {
            let a = arg(coord, &mut e, 0, &mut outs)?;
            let v = arg(value, &mut e, 1, &mut outs)?;
            resources::index_store(&mut e, &objs[*obj], &a, &v)?;
            Vec::new()
        }
        Stmt::IndexAtomic { obj, coord, name, args } => {
            let a = arg(coord, &mut e, 0, &mut outs)?;
            let mut rest = Vec::new();
            for (i, x) in args.iter().enumerate() {
                rest.push(arg(x, &mut e, i + 1, &mut outs)?);
            }
            resources::index_atomic(&mut e, &objs[*obj], &a, name, &rest)?.outs
        }
    };
    assert_eq!(called.len(), outs.len(), "out arguments of {}: {:?}", c.text, called.iter().map(|o| o.0).collect::<Vec<_>>());
    for (at, me) in outs {
        let v = &called.iter().find(|o| o.0 == at).unwrap_or_else(|| panic!("no out value for argument {at} of {}", c.text)).1;
        store(&mut e, slot(me), v)?;
    }
    let bb = e.bb;
    e.body.ret(bb);
    Ok(Built { e })
}

// ---------------------------------------------------------------------------------------
// canonical form of built IR

fn ty_name(m: &Module, t: Ty) -> String {
    match m.def(t) {
        TypeDef::Void => "void".into(),
        TypeDef::Int(b) => format!("i{b}"),
        TypeDef::Half => "half".into(),
        TypeDef::Float => "float".into(),
        TypeDef::Double => "double".into(),
        TypeDef::Struct { name: Some(n), .. } => format!("%{n}"),
        other => format!("{other:?}"),
    }
}

/// An integer constant in the canonical form: i32 is the plain number, other widths carry
/// their type (`15:i8`), i1 is `true` / `false`.
fn int_text(ty: &str, v: i64) -> String {
    match ty {
        "i32" => v.to_string(),
        "i1" => (v != 0).to_string(),
        _ => format!("{v}:{ty}"),
    }
}

fn const_text(m: &Module, v: Value) -> String {
    let c = m.as_const(v).expect("constant");
    match &c.c {
        Const::Int(x) => {
            let bits = m.bits(c.ty);
            let s = 64 - bits;
            int_text(&ty_name(m, c.ty), ((*x << s) as i64) >> s)
        }
        Const::Float(x) => match c.ty {
            Ty::F32 => format!("f:{:08X}", *x as u32),
            _ => format!("{}:{:016X}", ty_name(m, c.ty), x),
        },
        Const::Undef => format!("undef.{}", ty_name(m, c.ty)),
        Const::Zero => match m.def(c.ty) {
            TypeDef::Struct { fields, .. } => format!("{{{}}}", fields.iter().map(|&f| int_text(&ty_name(m, f), 0)).collect::<Vec<_>>().join(",")),
            _ => format!("zero.{}", ty_name(m, c.ty)),
        },
        Const::Aggregate(fields) => format!("{{{}}}", fields.iter().map(|&f| const_text(m, f)).collect::<Vec<_>>().join(",")),
    }
}

/// Calls that only make or annotate a handle are part of the tree of what uses the handle.
fn is_handle_fn(name: &str) -> bool {
    matches!(name, "dx.op.createHandleFromBinding" | "dx.op.createHandleFromHeap" | "dx.op.annotateHandle")
}

fn short(name: &str) -> &str {
    name.strip_prefix("dx.op.").unwrap_or(name)
}

/// Name every annotated handle by its expression (`$0 = annotateHandle(...)` before the first
/// line that uses it), so that lines stay short and a handle annotated once or once per use
/// reads the same.
fn name_handles(expanded: &str) -> String {
    const START: &str = "annotateHandle(";
    let mut seen: Vec<String> = Vec::new();
    let mut out = String::new();
    for line in expanded.lines() {
        let mut rest = line;
        let mut text = String::new();
        let mut defs = String::new();
        while let Some(at) = rest.find(START) {
            let mut depth = 0;
            let mut end = at;
            for (i, ch) in rest[at..].char_indices() {
                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            end = at + i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let h = &rest[at..end];
            let k = seen.iter().position(|s| s == h).unwrap_or_else(|| {
                seen.push(h.to_string());
                let _ = writeln!(defs, "${} = {h}", seen.len() - 1);
                seen.len() - 1
            });
            text.push_str(&rest[..at]);
            let _ = write!(text, "${k}");
            rest = &rest[end..];
        }
        text.push_str(rest);
        out.push_str(&defs);
        out.push_str(&text);
        out.push('\n');
    }
    out
}

fn bin_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::FAdd => "fadd",
        BinOp::Sub => "sub",
        BinOp::FSub => "fsub",
        BinOp::Mul => "mul",
        BinOp::FMul => "fmul",
        BinOp::UDiv => "udiv",
        BinOp::SDiv => "sdiv",
        BinOp::FDiv => "fdiv",
        BinOp::URem => "urem",
        BinOp::SRem => "srem",
        BinOp::FRem => "frem",
        BinOp::Shl => "shl",
        BinOp::LShr => "lshr",
        BinOp::AShr => "ashr",
        BinOp::And => "and",
        BinOp::Or => "or",
        BinOp::Xor => "xor",
    }
}

fn cast_name(op: CastOp) -> &'static str {
    match op {
        CastOp::Trunc => "trunc",
        CastOp::ZExt => "zext",
        CastOp::SExt => "sext",
        CastOp::FPToUI => "fptoui",
        CastOp::FPToSI => "fptosi",
        CastOp::UIToFP => "uitofp",
        CastOp::SIToFP => "sitofp",
        CastOp::FPTrunc => "fptrunc",
        CastOp::FPExt => "fpext",
        CastOp::BitCast => "bitcast",
    }
}

fn canon_built(b: &Built) -> String {
    let (m, body) = (&b.e.m, &b.e.body);
    assert_eq!(body.blocks.len(), 1, "fixtures are straight-line code");
    let order = &body.blocks[0].insts;
    // liveness: effects, and what they use
    let mut live = vec![false; body.insts.len()];
    let uses = |i: usize| -> Vec<Value> {
        let inst = &body.insts[i];
        let mut u = match inst.op {
            Op::Bin(_) | Op::ICmp(_) | Op::FCmp(_) | Op::Store | Op::AtomicRmw(_) => vec![inst.a, inst.b],
            Op::Cast(_) | Op::ExtractValue | Op::Load => vec![inst.a],
            Op::Select | Op::CmpXchg => vec![inst.a, inst.b, inst.c],
            Op::Call => body.operands_of(inst).to_vec(),
            Op::Ret => vec![],
            other => panic!("fixture instruction {other:?}"),
        };
        for v in &mut u {
            *v = body.resolve(*v);
        }
        u
    };
    for &i in order.iter().rev() {
        let i = i as usize;
        if body.insts[i].flags & flag::EFFECT != 0 {
            live[i] = true;
        }
        if live[i] {
            for v in uses(i) {
                if let ValueKind::Inst(k) = v.kind() {
                    live[k as usize] = true;
                }
            }
        }
    }
    let mut names: HashMap<u32, String> = HashMap::new();
    let mut out = String::new();
    let mut anchors = 0;
    fn tree(m: &Module, body: &Body, names: &HashMap<u32, String>, v: Value) -> String {
        let v = body.resolve(v);
        let ValueKind::Inst(i) = v.kind() else { return const_text(m, v) };
        if let Some(n) = names.get(&i) {
            return n.clone();
        }
        let inst = &body.insts[i as usize];
        let t = |x: Value| tree(m, body, names, x);
        match inst.op {
            Op::Bin(op) => format!("{}({}, {})", bin_name(op), t(inst.a), t(inst.b)),
            Op::Cast(op) => format!("{}({}) to {}", cast_name(op), t(inst.a), ty_name(m, inst.ty)),
            Op::ICmp(p) => format!("icmp {}({}, {})", format!("{p:?}").to_lowercase(), t(inst.a), t(inst.b)),
            Op::FCmp(p) => format!("fcmp {}({}, {})", format!("{p:?}").to_lowercase(), t(inst.a), t(inst.b)),
            Op::Select => format!("select({}, {}, {})", t(inst.a), t(inst.b), t(inst.c)),
            Op::ExtractValue => format!("ev({}, {})", t(inst.a), inst.b.0),
            Op::Call => {
                let name = short(&m.function(inst.a).name);
                let args: Vec<String> = body.operands_of(inst).iter().map(|&x| t(x)).collect();
                format!("{name}({}){}", args.join(", "), if inst.flags & flag::PRECISE != 0 { " !precise" } else { "" })
            }
            other => panic!("fixture instruction {other:?}"),
        }
    }
    for &i in order {
        let inst = &body.insts[i as usize];
        if !live[i as usize] {
            continue;
        }
        match inst.op {
            Op::Ret => out.push_str("ret\n"),
            Op::Call if !is_handle_fn(&m.function(inst.a).name) => {
                let text = tree(m, body, &names, Value::inst(i));
                if inst.ty == Ty::VOID {
                    let _ = writeln!(out, "{text}");
                } else {
                    let n = format!("@{anchors}");
                    anchors += 1;
                    let _ = writeln!(out, "{n} = {text}");
                    names.insert(i, n);
                }
            }
            _ => {}
        }
    }
    name_handles(&out)
}

// ---------------------------------------------------------------------------------------
// canonical form of DXC's disassembly

/// Split `type value` into its two parts (the type is everything up to the last top-level
/// space that is not inside braces).
fn split_typed(s: &str) -> (&str, &str) {
    let s = s.trim();
    if let Some(open) = s.find(" {") {
        if s.ends_with('}') {
            return (&s[..open], &s[open + 1..]);
        }
    }
    match s.rfind(' ') {
        Some(i) => (&s[..i], &s[i + 1..]),
        None => ("", s),
    }
}

fn canon_operand(names: &HashMap<String, String>, s: &str) -> String {
    let (ty, v) = split_typed(s);
    if let Some(n) = v.strip_prefix('%') {
        return names.get(n).unwrap_or_else(|| panic!("undefined value %{n}")).clone();
    }
    if v.starts_with('{') {
        let inner = &v[1..v.len() - 1];
        return format!("{{{}}}", split_top(inner, ',').iter().map(|f| canon_operand(names, f)).collect::<Vec<_>>().join(","));
    }
    if v == "zeroinitializer" {
        return match ty {
            "%dx.types.ResBind" => "{0,0,0,0:i8}".into(),
            "%dx.types.ResourceProperties" => "{0,0}".into(),
            other => format!("zero.{other}"),
        };
    }
    if v == "undef" {
        return format!("undef.{ty}");
    }
    match ty {
        "float" | "double" => {
            let bits = match v.strip_prefix("0x") {
                Some(h) => u64::from_str_radix(h, 16).expect("hex float"),
                None => v.parse::<f64>().expect("float").to_bits(),
            };
            if ty == "float" {
                format!("f:{:08X}", (f64::from_bits(bits) as f32).to_bits())
            } else {
                format!("double:{bits:016X}")
            }
        }
        "i1" => v.to_string(),
        _ => int_text(ty, v.parse().unwrap_or_else(|_| panic!("operand {s}"))),
    }
}

fn canon_dxc(ll: &str) -> String {
    let start = ll.find("define void @main()").expect("no main in DXC output");
    let body = &ll[start..];
    let body = &body[body.find('\n').unwrap() + 1..body.find("\n}").expect("unterminated function")];
    let mut names: HashMap<String, String> = HashMap::new();
    let mut out = String::new();
    let mut anchors = 0;
    for line in body.lines() {
        let line = match line.find("  ; ") {
            Some(i) => &line[..i],
            None => line,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "ret void" {
            out.push_str("ret\n");
            continue;
        }
        let (def, inst) = match line.strip_prefix('%').and_then(|l| l.split_once(" = ")) {
            Some((d, i)) => (Some(d.to_string()), i),
            None => (None, line),
        };
        let precise = inst.contains("!dx.precise");
        let inst = inst.split(", !").next().unwrap();
        let (op, rest) = inst.split_once(' ').unwrap_or_else(|| panic!("instruction {inst}"));
        let text = match op {
            "call" => {
                let at = rest.find(" @").unwrap_or_else(|| panic!("call {rest}"));
                let open = rest[at..].find('(').unwrap() + at;
                let name = &rest[at + 2..open];
                let args: Vec<String> = split_top(&rest[open + 1..rest.rfind(')').unwrap()], ',').iter().map(|a| canon_operand(&names, a)).collect();
                let text = format!("{}({}){}", short(name), args.join(", "), if precise { " !precise" } else { "" });
                if !is_handle_fn(name) {
                    if rest.starts_with("void ") {
                        let _ = writeln!(out, "{text}");
                        continue;
                    }
                    let n = format!("@{anchors}");
                    anchors += 1;
                    let _ = writeln!(out, "{n} = {text}");
                    names.insert(def.expect("call result"), n);
                    continue;
                }
                text
            }
            "extractvalue" => {
                let (agg, idx) = rest.rsplit_once(", ").unwrap();
                format!("ev({}, {idx})", canon_operand(&names, agg))
            }
            "icmp" | "fcmp" => {
                let rest = rest.trim_start_matches("fast ");
                let (pred, rest) = rest.split_once(' ').unwrap();
                let (a, b) = rest.rsplit_once(", ").unwrap();
                let (ty, _) = split_typed(a);
                format!("{op} {pred}({}, {})", canon_operand(&names, a), canon_operand(&names, &format!("{ty} {b}")))
            }
            "select" => {
                let p = split_top(rest, ',');
                format!("select({}, {}, {})", canon_operand(&names, &p[0]), canon_operand(&names, &p[1]), canon_operand(&names, &p[2]))
            }
            "trunc" | "zext" | "sext" | "fptoui" | "fptosi" | "uitofp" | "sitofp" | "fptrunc" | "fpext" | "bitcast" => {
                let (a, to) = rest.rsplit_once(" to ").unwrap();
                format!("{op}({}) to {to}", canon_operand(&names, a))
            }
            "add" | "fadd" | "sub" | "fsub" | "mul" | "fmul" | "udiv" | "sdiv" | "fdiv" | "urem" | "srem" | "frem" | "shl" | "lshr" | "ashr" | "and" | "or" | "xor" => {
                // wrap and exactness flags are inferred by DXC's optimiser, not part of the lowering
                let mut rest = rest;
                while let Some(r) = ["nuw ", "nsw ", "exact ", "fast "].iter().find_map(|f| rest.strip_prefix(f)) {
                    rest = r;
                }
                let (a, b) = rest.rsplit_once(", ").unwrap();
                let (ty, _) = split_typed(a);
                format!("{op}({}, {})", canon_operand(&names, a), canon_operand(&names, &format!("{ty} {b}")))
            }
            other => panic!("fixture instruction {other}: {line}"),
        };
        names.insert(def.unwrap_or_else(|| panic!("no result: {line}")), text);
    }
    name_handles(&out)
}

// ---------------------------------------------------------------------------------------
// the oracle and the golden file

fn dxc_dir() -> Option<PathBuf> {
    std::env::var_os("XCL_DXIL_DXC").map(PathBuf::from)
}

fn run_dxc(dxc: &std::path::Path, c: &Case, dir: &std::path::Path, id: usize) -> Result<String, String> {
    let src = dir.join(format!("c{id}.hlsl"));
    let ll = dir.join(format!("c{id}.ll"));
    let hlsl = hlsl_of(c);
    std::fs::write(&src, &hlsl).map_err(|e| e.to_string())?;
    let o = std::process::Command::new(dxc.join("bin/dxc"))
        .env("LD_LIBRARY_PATH", dxc.join("lib"))
        .args(["-E", "main", "-T", if c.pixel { "ps_6_6" } else { "cs_6_6" }, "-HV", "2021", "-Gis", "-O1", "-Fc"])
        .arg(&ll)
        .arg(&src)
        .output()
        .map_err(|e| format!("cannot run dxc: {e}"))?;
    if !o.status.success() {
        return Err(format!("dxc rejected the snippet of {}:\n{}\n{hlsl}", c.text, String::from_utf8_lossy(&o.stderr)));
    }
    std::fs::read_to_string(&ll).map_err(|e| e.to_string())
}

fn golden_map() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();
    for entry in GOLDEN.split("### ").skip(1) {
        let (head, body) = entry.split_once('\n').unwrap_or((entry, ""));
        map.insert(head.trim(), body);
    }
    map
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/hlsl/resources/golden.txt")
}

/// DXC's canonical output for every case, in case order (run in parallel).
fn oracle(cases: &[Case], dxc: &std::path::Path) -> Vec<Result<String, String>> {
    let dir = std::env::temp_dir().join(format!("xcl-dxil-resources-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<std::sync::Mutex<Option<Result<String, String>>>> = cases.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()).min(16) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(c) = cases.get(i) else { break };
                let r = run_dxc(dxc, c, &dir, i).map(|ll| canon_dxc(&ll));
                *results[i].lock().unwrap() = Some(r);
            });
        }
    });
    let _ = std::fs::remove_dir_all(&dir);
    results.into_iter().map(|r| r.into_inner().unwrap().unwrap()).collect()
}

/// The golden copy is DXC's output: checked against DXC when it is there, rewritten with
/// `XCL_DXIL_BLESS=1`.
#[test]
fn golden_matches_dxc() {
    let Some(dxc) = dxc_dir() else {
        eprintln!("resources_fixtures: XCL_DXIL_DXC is not set, the golden copy is not checked against DXC");
        return;
    };
    let cases: Vec<Case> = cases().iter().map(|t| parse_case(t)).collect();
    let got = oracle(&cases, &dxc);
    let mut file = String::from("# DXC 1.9 (-HV 2021 -Gis -O1) in the canonical form of tests/resources_fixtures.rs, one entry per\n# case there. Generated: XCL_DXIL_DXC=... XCL_DXIL_BLESS=1 cargo test --test resources_fixtures\n");
    let mut failed = Vec::new();
    let golden = golden_map();
    for (c, g) in cases.iter().zip(&got) {
        match g {
            Ok(text) => {
                let _ = write!(file, "### {}\n{text}", c.text);
                if golden.get(c.text.as_str()).copied() != Some(text.as_str()) {
                    failed.push(format!("{}: golden copy differs from DXC", c.text));
                }
            }
            Err(e) => failed.push(e.clone()),
        }
    }
    if std::env::var_os("XCL_DXIL_BLESS").is_some() {
        let errors: Vec<&String> = failed.iter().filter(|f| !f.ends_with("golden copy differs from DXC")).collect();
        assert!(errors.is_empty(), "{}", errors.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n"));
        std::fs::write(golden_path(), file).expect("write golden");
        eprintln!("resources_fixtures: wrote {} entries to {}", cases.len(), golden_path().display());
        return;
    }
    assert!(failed.is_empty(), "{} of {} cases:\n{}", failed.len(), cases.len(), failed.join("\n"));
}

/// Every access built through the resources API equals DXC's lowering.
#[test]
fn lowering_matches_golden() {
    let golden = golden_map();
    let mut failed = Vec::new();
    let all = cases();
    let mut bad = 0;
    for text in &all {
        let c = parse_case(text);
        let Some(want) = golden.get(text.as_str()) else {
            bad += 1;
            failed.push(format!("{text}: no golden entry (bless with DXC)"));
            continue;
        };
        match build(&c) {
            Ok(b) => {
                let got = canon_built(&b);
                if got != *want {
                    bad += 1;
                    failed.push(format!("{text}\n--- DXC\n{want}--- ours\n{got}"));
                }
            }
            Err(e) => {
                bad += 1;
                failed.push(format!("{text}: {e}"));
            }
        }
    }
    failed.truncate(std::env::var("XCL_DXIL_SHOW").ok().and_then(|n| n.parse().ok()).unwrap_or(8));
    assert!(bad == 0, "{bad} of {} cases differ; the first {}:\n{}", all.len(), failed.len(), failed.join("\n"));
}

/// The printed form of a built case goes through the IR printer without a hitch (it is what
/// a person debugging a mismatch reads).
#[test]
fn built_cases_print() {
    let c = parse_case(&cases()[0]);
    let b = build(&c).expect("build");
    let text = xcl_dxil::irprint::function(&b.e.m, &b.e.body);
    assert!(text.contains("dx.op."), "{text}");
}
