//! Gives the crate a build id: FNV-1a 64 of its sources. Callers that cache the containers this
//! backend produces key them by it (`xcl_dxil::BUILD_ID`), so a change to the generated code can
//! never be served from a cache filled by an earlier build, and nobody has to remember to bump
//! a number.

use std::path::{Path, PathBuf};

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    let mut files = Vec::new();
    collect(&root.join("src"), &mut files);
    // the same id on every machine: names with '/', in byte order, text without carriage returns
    let mut named: Vec<(String, PathBuf)> = files.into_iter().map(|p| (p.strip_prefix(&root).unwrap_or(&p).to_string_lossy().replace('\\', "/"), p)).collect();
    named.sort();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    };
    for (name, path) in &named {
        name.bytes().chain([0]).for_each(&mut eat);
        std::fs::read(path).unwrap_or_default().into_iter().filter(|&b| b != b'\r').chain([0]).for_each(&mut eat);
    }
    println!("cargo:rustc-env=XCL_DXIL_BUILD_ID={h:016x}");
}
