//! Direct DXIL backend for XCL's shader translator.
//!
//! [`compile`] takes the HLSL the translator emits (`gpu::recomp::emit`; the dialect is what
//! that module generates, not all of HLSL) and returns a signed DXIL container, as DXC with
//! `-HV 2021 -Gis` would, but in time linear in the program: no optimiser pipeline, SSA built
//! directly while lowering. Anything outside the dialect is [`Error::Unsupported`], and the
//! caller falls back to DXC for that program.
//!
//! Pipeline: [`hlsl::parse`] (text → AST) → [`hlsl::lower`] (AST → [`ir::Module`] with one
//! entry function, plus a [`shader::ShaderInfo`]) → [`flags`] → [`meta`] (dx.* metadata) →
//! [`bcwriter`] (LLVM 3.7 bitcode) → [`container`] (DXIL, SFI0, ISG1 / OSG1 / PSG1, PSV0,
//! HASH).
//!
//! The crate has no dependencies and no Windows code, so it builds and tests on Linux; the
//! validator (`dxv` / `dxil.dll`) is run from outside ([`bin/xdxc`], `tools/`).

pub mod bcreader;
pub mod bcwriter;
pub mod bitstream;
pub mod consts;
pub mod container;
pub mod dxop;
pub mod flags;
pub mod hlsl;
pub mod interp;
pub mod ir;
pub mod irprint;
pub mod meta;
pub mod shader;

/// A hash of the crate's sources (`build.rs`). The caller keys its DXIL cache by it, so a
/// cached container never comes from another build of the crate. Never written into a
/// container: the output does not depend on it.
pub const BUILD_ID: &str = env!("XCL_DXIL_BUILD_ID");

/// Part of the DXIL cache key: bump on every change that can alter the container produced
/// for some input.
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Valid input this backend does not handle (yet): fall back to DXC, log and count it.
    /// The text names the construct ("intrinsic frexp", "stage hs", ...), never the program.
    Unsupported(String),
    /// The input is not what the translator emits (a syntax or type error): DXC would reject
    /// it too. `line` is 1-based.
    Source { line: u32, msg: String },
    /// A bug in this crate.
    Internal(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
            Error::Source { line, msg } => write!(f, "line {line}: {msg}"),
            Error::Internal(s) => write!(f, "internal error: {s}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub fn unsupported<T>(what: impl Into<String>) -> Result<T> {
    Err(Error::Unsupported(what.into()))
}

/// Where the time of one [`compile_timed`] went (seconds) and what came out.
#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub parse: f64,
    pub lower: f64,
    pub write: f64,
    pub container: f64,
    /// freeing the syntax tree and the module: handing them to a thread that frees them, or
    /// (`XCL_DXIL_THREADS=1`) freeing them; not in the other times
    pub free: f64,
    pub instructions: usize,
    pub blocks: usize,
    pub dxil_bytes: usize,
}

/// Compile `src` for `profile` ("cs_6_6", "vs_6_6", ...), entry point `entry`.
pub fn compile(src: &str, entry: &str, profile: &str) -> Result<Vec<u8>> {
    compile_timed(src, entry, profile).map(|(c, _)| c)
}

pub fn compile_timed(src: &str, entry: &str, profile: &str) -> Result<(Vec<u8>, Stats)> {
    let mut stats = Stats::default();
    let t = std::time::Instant::now();
    let unit = hlsl::parse::parse(src)?;
    stats.parse = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let (module, mut info) = hlsl::lower::lower(&unit, entry, profile)?;
    let f = std::time::Instant::now();
    free_aside(unit);
    stats.free = f.elapsed().as_secs_f64();
    // the code is finished: its liveness is computed once, for the flags, the ViewID state and
    // the writer
    let mut module = ir::Sealed::new(module)?;
    // the flags, and what dx.resources / dx.viewIdState and PSV0 both state about the code
    meta::prepare_sealed(&module, &mut info);
    module.extend(|m| meta::emit(m, &info))?;
    stats.lower = t.elapsed().as_secs_f64() - stats.free;
    let t = std::time::Instant::now();
    // the bitcode is written into the container: nothing copies it
    let c = container::open(&info)?;
    let (c, counts) = bcwriter::write_sealed_into(&module, c)?;
    stats.instructions = counts.instructions;
    stats.blocks = counts.blocks;
    stats.write = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let c = container::close(c)?;
    stats.container = t.elapsed().as_secs_f64();
    stats.dxil_bytes = c.len();
    let f = std::time::Instant::now();
    free_aside(module);
    stats.free += f.elapsed().as_secs_f64();
    Ok((c, stats))
}

/// Whether the crate may start threads of its own: not with `XCL_DXIL_THREADS=1`. Its output
/// is the same either way.
pub(crate) fn threads_allowed() -> bool {
    !std::env::var_os("XCL_DXIL_THREADS").is_some_and(|n| n == "1")
}

/// Frees `v` on a thread of its own, so that the caller does not wait for it: the syntax tree
/// and the module of a large program are hundreds of MB in some million allocations, tens of
/// milliseconds to free. Inline with `XCL_DXIL_THREADS=1`, or when no thread can be started.
fn free_aside<T: Send + 'static>(v: T) {
    if !threads_allowed() {
        return drop(v);
    }
    // a failed spawn drops the closure, and `v` with it, here
    let _ = std::thread::Builder::new().name("xcl-dxil-free".into()).spawn(move || drop(v));
}
