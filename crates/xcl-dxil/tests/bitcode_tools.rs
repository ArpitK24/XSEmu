//! Tools for looking at bitcode by hand; they do nothing unless their variable is set.
//!
//! `XCL_DXIL_DUMP=<container or .bc> cargo test --test bitcode_tools dump -- --nocapture`
//! prints every block and record of the bitcode.
//!
//! `XCL_DXIL_RT=<container> XCL_DXIL_RT_OUT=<out container> cargo test --release --test
//! bitcode_tools roundtrip -- --nocapture` reads the container's bitcode, writes it again
//! and saves the container with the new bitcode (`XCL_DXIL_RT_KEEP_DEAD=1` keeps DXC's dead
//! code, `XCL_DXIL_RT_TEXT=<file>` also saves the IR text).

use xcl_dxil::{bcreader, bcwriter, bitstream, irprint};

#[test]
fn dump() {
    let Ok(path) = std::env::var("XCL_DXIL_DUMP") else { return };
    let bytes = std::fs::read(&path).expect("read");
    let bc = if bytes.starts_with(b"DXBC") { bcreader::container_bitcode(&bytes).expect("container") } else { &bytes[..] };
    let max = std::env::var("XCL_DXIL_DUMP_OPS").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    println!("{} bytes of bitcode", bc.len());
    match bitstream::dump(bc, max) {
        Ok(text) => println!("{text}"),
        Err(e) => panic!("{e}"),
    }
}

#[test]
fn roundtrip() {
    let Ok(path) = std::env::var("XCL_DXIL_RT") else { return };
    let bytes = std::fs::read(&path).expect("read");
    let bc = bcreader::container_bitcode(&bytes).expect("container");
    let keep_dead = std::env::var("XCL_DXIL_RT_KEEP_DEAD").is_ok();
    let t = std::time::Instant::now();
    let (module, notes) = bcreader::read_with(bc, bcreader::Options { keep_dead }).expect("read bitcode");
    let read = t.elapsed();
    let t = std::time::Instant::now();
    let (out, counts) = bcwriter::write(&module).expect("write bitcode");
    let write = t.elapsed();
    println!("{path}: dxc {} bytes, ours {} bytes ({:+.2}%); {} instructions in {} blocks; read {:.3} s, write {:.4} s; {notes:?}", bc.len(), out.len(), (out.len() as f64 / bc.len() as f64 - 1.0) * 100.0, counts.instructions, counts.blocks, read.as_secs_f64(), write.as_secs_f64());
    if let Ok(to) = std::env::var("XCL_DXIL_RT_OUT") {
        std::fs::write(to, bcreader::replace_bitcode(&bytes, &out).expect("container")).expect("write");
    }
    if let Ok(to) = std::env::var("XCL_DXIL_RT_TEXT") {
        std::fs::write(to, irprint::module(&module)).expect("write");
    }
}
