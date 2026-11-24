//! [`xcl_dxil::bcwriter`] on modules built by hand: what it drops, what it refuses, that the
//! reader gets back what was written, and how long a very large module takes.
//!
//! `cargo test --release --test bitcode_writer -- --ignored --nocapture` runs the timing of a
//! 2 million instruction module.

use xcl_dxil::consts::OpCode;
use xcl_dxil::ir::{flag, BinOp, Block, Body, CastOp, FPred, IPred, Inst, Module, Op, Ty, Value};
use xcl_dxil::{bcreader, bcwriter, dxop, irprint, Error};

fn finish(mut m: Module, b: Body) -> Module {
    m.define("main", b);
    m
}

/// The text of a module with its lines sorted, and the text of its function: the reader
/// lists the functions in the order of the bitcode (the defined one first), so the module
/// text of what it returns has the same lines in another order.
fn texts(m: &Module) -> (Vec<String>, String) {
    let text = irprint::module(m);
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    lines.sort();
    let body = m.funcs.iter().find_map(|f| f.body.as_ref()).expect("a defined function");
    (lines, irprint::function(m, body))
}

/// A body shaped like shader code: per round 21 instructions in 4 blocks (dx.op calls, float
/// and integer arithmetic, a store and a load through a getelementptr, a diamond and a phi).
fn synthetic(rounds: usize) -> Module {
    let mut m = Module::new();
    let mut b = Body::new();
    let (zero, undef) = (m.const_i32(0), m.undef(Ty::I32));
    let array = m.array(64, Ty::F32);
    let (array_ptr, f32_ptr) = (m.ptr(array), m.ptr(Ty::F32));
    let slot = b.alloca(array_ptr);
    let mut bb = Body::ENTRY;
    let col = m.const_i8(0);
    let mut acc = dxop::call(&mut m, &mut b, bb, OpCode::LoadInput, Ty::F32, &[zero, zero, col, undef], true);
    let mut n = dxop::call(&mut m, &mut b, bb, OpCode::LoadInput, Ty::I32, &[zero, zero, col, undef], true);
    for k in 0..rounds {
        let (id, col) = (m.const_i32(k as u32 % 8), m.const_i8(k as u8 % 4));
        let x = dxop::call(&mut m, &mut b, bb, OpCode::LoadInput, Ty::F32, &[id, zero, col, undef], true);
        let y = b.bin(bb, BinOp::FMul, Ty::F32, acc, x);
        let c = m.const_f32((k % 16) as f32 + 0.5);
        let z = b.bin(bb, BinOp::FAdd, Ty::F32, y, c);
        let s = dxop::call(&mut m, &mut b, bb, OpCode::Sqrt, Ty::F32, &[z], true);
        let mx = dxop::call(&mut m, &mut b, bb, OpCode::FMax, Ty::F32, &[s, x], true);
        let i = b.cast(bb, CastOp::FPToUI, Ty::I32, mx);
        let j = b.bin(bb, BinOp::Add, Ty::I32, i, n);
        let mask = m.const_i32(63);
        let at = b.bin(bb, BinOp::And, Ty::I32, j, mask);
        let one = m.const_i32(1);
        let sh = b.bin(bb, BinOp::Shl, Ty::I32, at, one);
        let p = b.gep(bb, f32_ptr, slot, &[zero, at]);
        b.store(bb, p, mx);
        let l = b.load(bb, Ty::F32, p);
        let lt = b.fcmp(bb, FPred::Olt, l, acc);
        let sel = b.select(bb, Ty::F32, lt, l, mx);
        let limit = m.const_i32(17 + k as u32 % 5);
        let small = b.icmp(bb, IPred::Ult, sh, limit);
        let (t, f, join) = (b.new_block(), b.new_block(), b.new_block());
        b.cond_br(bb, small, t, f);
        let a1 = b.bin(t, BinOp::FSub, Ty::F32, sel, x);
        b.br(t, join);
        let a2 = b.bin(f, BinOp::FDiv, Ty::F32, sel, x);
        b.br(f, join);
        acc = b.phi(join, Ty::F32);
        b.phi_add(acc, a1, t);
        b.phi_add(acc, a2, f);
        n = j;
        bb = join;
    }
    let col = m.const_i8(0);
    dxop::call(&mut m, &mut b, bb, OpCode::StoreOutput, Ty::F32, &[zero, zero, col, acc], false);
    b.ret(bb);
    finish(m, b)
}

#[test]
fn reader_gets_back_what_was_written() {
    let m = synthetic(40);
    let (bytes, counts) = bcwriter::write(&m).expect("write");
    assert_eq!(counts.instructions, 3 + 40 * 21 + 2);
    assert_eq!(counts.blocks, 1 + 40 * 3);
    let (back, notes) = bcreader::read(&bytes).expect("read");
    assert_eq!(notes, bcreader::Notes::default());
    assert_eq!(texts(&back), texts(&m));
    let (again, _) = bcwriter::write(&back).expect("write again");
    assert_eq!(again, bytes);
}

#[test]
fn dead_code_is_not_written() {
    let mut m = Module::new();
    let mut b = Body::new();
    let e = Body::ENTRY;
    let (zero, undef, col) = (m.const_i32(0), m.undef(Ty::I32), m.const_i8(0));
    let x = dxop::call(&mut m, &mut b, e, OpCode::LoadInput, Ty::F32, &[zero, zero, col, undef], true);
    // dead: nothing with an effect uses it, and neither its constant
    let unused = m.const_f32(1234.5);
    b.bin(e, BinOp::FAdd, Ty::F32, x, unused);
    // forwarded: users of 'twice' get 'x'
    let twice = b.bin(e, BinOp::FAdd, Ty::F32, x, x);
    b.replace(twice, x);
    let (live, dead, join) = (b.new_block(), b.new_block(), b.new_block());
    b.br(e, live);
    b.br(live, join);
    // unreachable: no branch leads here, so the phi loses its incoming value from it
    let lost = b.bin(dead, BinOp::FMul, Ty::F32, x, x);
    b.br(dead, join);
    let phi = b.phi(join, Ty::F32);
    b.phi_add(phi, twice, live);
    b.phi_add(phi, lost, dead);
    dxop::call(&mut m, &mut b, join, OpCode::StoreOutput, Ty::F32, &[zero, zero, col, phi], false);
    b.ret(join);
    let m = finish(m, b);
    let (bytes, counts) = bcwriter::write(&m).expect("write");
    // loadInput, br, br, phi, storeOutput, ret
    assert_eq!((counts.instructions, counts.blocks), (6, 3));
    let (back, _) = bcreader::read(&bytes).expect("read");
    assert_eq!(texts(&back), texts(&m));
    let text = irprint::module(&back);
    assert!(text.contains("phi float [ %1, %2 ]"), "{text}");
    assert!(!text.contains("1.234500e+03") && !text.contains("fmul") && !text.contains("fadd"), "{text}");
}

#[test]
fn control_flow_hints_and_constant_addresses_survive() {
    let mut m = Module::new();
    let mut b = Body::new();
    let e = Body::ENTRY;
    let array = m.array(8, Ty::I32);
    let shared = m.global("shared", array, 3, None, false);
    let i32_ptr = m.ptr_in(Ty::I32, 3);
    let (zero, seven) = (m.const_i32(0), m.const_i32(7));
    let last = m.const_gep(i32_ptr, shared, &[zero, seven]);
    let v = b.load(e, Ty::I32, last);
    let c = b.icmp(e, IPred::Eq, v, zero);
    let (t, f, out) = (b.new_block(), b.new_block(), b.new_block());
    b.cond_br(e, c, t, f);
    let Value(i) = Value(b.blocks[0].insts[b.blocks[0].insts.len() - 1]);
    b.insts[i as usize].flags |= flag::HINT_BRANCH;
    b.store(t, last, seven);
    b.br(t, out);
    b.switch(f, v, out, vec![(1, t), (2, out)]);
    let i = b.blocks[f.0 as usize].insts[0];
    b.insts[i as usize].flags |= flag::HINT_FLATTEN;
    b.ret(out);
    let m = finish(m, b);
    let (bytes, _) = bcwriter::write(&m).expect("write");
    let (back, notes) = bcreader::read(&bytes).expect("read");
    assert_eq!(notes, bcreader::Notes::default());
    assert_eq!(texts(&back), texts(&m));
    let text = irprint::module(&back);
    for part in [
        "load i32, i32 addrspace(3)* getelementptr inbounds ([8 x i32], [8 x i32] addrspace(3)* @shared, i32 0, i32 7), align 4",
        "label %4, !dx.controlflow.hints !0",
        "  ], !dx.controlflow.hints !1",
        "!0 = distinct !{!0, !\"dx.controlflow.hints\", i32 1}",
        "!1 = distinct !{!1, !\"dx.controlflow.hints\", i32 2}",
    ] {
        assert!(text.contains(part), "{part} is missing in\n{text}");
    }
}

fn internal(m: &Module, part: &str) {
    match bcwriter::write(m) {
        Err(Error::Internal(text)) => assert!(text.contains(part), "{text}"),
        Err(other) => panic!("{other:?}"),
        Ok(_) => panic!("written, but '{part}' was expected"),
    }
}

#[test]
fn a_malformed_body_is_an_error() {
    // a block that does not end
    let mut b = Body::new();
    let next = b.new_block();
    b.br(Body::ENTRY, next);
    internal(&finish(Module::new(), b), "block 1 has no terminator");

    // a use of something that is in no block the entry reaches
    let mut m = Module::new();
    let mut b = Body::new();
    let (zero, col) = (m.const_i32(0), m.const_i8(0));
    let island = b.new_block();
    let one = m.const_f32(1.0);
    let lost = b.bin(island, BinOp::FAdd, Ty::F32, one, one);
    b.ret(island);
    dxop::call(&mut m, &mut b, Body::ENTRY, OpCode::StoreOutput, Ty::F32, &[zero, zero, col, lost], false);
    b.ret(Body::ENTRY);
    internal(&finish(m, b), "uses a value that is not written");

    // an operand that names no instruction, a branch to no block
    let mut m = Module::new();
    let mut b = Body::new();
    let one = m.const_f32(1.0);
    let ptr = m.ptr(Ty::F32);
    let slot = b.alloca(ptr);
    b.push(Body::ENTRY, Inst { op: Op::Store, flags: flag::EFFECT, ty: Ty::VOID, a: slot, b: Value::inst(1000), c: Value::NONE });
    b.store(Body::ENTRY, slot, one);
    b.ret(Body::ENTRY);
    internal(&finish(m, b), "does not exist");
    let mut b = Body::new();
    b.br(Body::ENTRY, Block(9));
    internal(&finish(Module::new(), b), "");

    // forwarding that never ends
    let mut m = Module::new();
    let mut b = Body::new();
    let (zero, col, one) = (m.const_i32(0), m.const_i8(0), m.const_f32(1.0));
    let x = b.bin(Body::ENTRY, BinOp::FAdd, Ty::F32, one, one);
    let y = b.bin(Body::ENTRY, BinOp::FAdd, Ty::F32, x, one);
    dxop::call(&mut m, &mut b, Body::ENTRY, OpCode::StoreOutput, Ty::F32, &[zero, zero, col, y], false);
    b.ret(Body::ENTRY);
    b.replace(x, y);
    b.replace(y, x);
    internal(&finish(m, b), "forwarding cycle");

    // a dx.op the table has no signature for fails at the write, with its own error
    let mut m = Module::new();
    let mut b = Body::new();
    dxop::call(&mut m, &mut b, Body::ENTRY, OpCode::HitObject_MakeNop, Ty::VOID, &[], false);
    b.ret(Body::ENTRY);
    assert!(matches!(bcwriter::write(&finish(m, b)), Err(Error::Unsupported(_))));
}

#[test]
fn the_reader_refuses_damaged_bitcode() {
    let (bytes, _) = bcwriter::write(&synthetic(3)).expect("write");
    for cut in [0, 3, 4, 8, 20, bytes.len() / 2, bytes.len() - 4] {
        assert!(bcreader::read(&bytes[..cut]).is_err(), "cut at {cut}");
    }
    // every single flipped bit is either read or refused, never a panic
    let mut damaged = bytes.clone();
    for at in 0..bytes.len() {
        for bit in [0, 3, 7] {
            damaged[at] ^= 1 << bit;
            let _ = bcreader::read(&damaged);
            damaged[at] = bytes[at];
        }
    }
}

#[test]
#[ignore = "timing: run with --release -- --ignored --nocapture"]
fn two_million_instructions() {
    let t = std::time::Instant::now();
    let m = synthetic(95_240);
    let build = t.elapsed();
    let mut best = f64::MAX;
    let mut out = (Vec::new(), bcwriter::Counts::default());
    for _ in 0..5 {
        let t = std::time::Instant::now();
        out = bcwriter::write(&m).expect("write");
        best = best.min(t.elapsed().as_secs_f64());
    }
    let (bytes, counts) = out;
    println!("{} instructions in {} blocks: built in {:.3} s, written in {best:.4} s ({:.1} ns an instruction), {} bytes ({:.2} an instruction)", counts.instructions, counts.blocks, build.as_secs_f64(), best * 1e9 / counts.instructions as f64, bytes.len(), bytes.len() as f64 / counts.instructions as f64);
    assert!(counts.instructions >= 2_000_000);
    let t = std::time::Instant::now();
    let (back, _) = bcreader::read(&bytes).expect("read");
    println!("read back in {:.3} s", t.elapsed().as_secs_f64());
    assert_eq!(back.funcs.iter().filter_map(|f| f.body.as_ref()).map(|b| b.insts.len()).sum::<usize>(), counts.instructions);
    if !cfg!(debug_assertions) {
        assert!(best < 0.2, "{best} s");
    }
}
