//! How long the passes of [`xcl_dxil::flags`] and [`xcl_dxil::meta`] take on a module the size of
//! the game's largest programs (millions of instructions): they run on every compile, in the
//! second the whole backend has. Ignored by default; run with
//! `cargo test --release --test meta_perf -- --ignored --nocapture`.

use std::time::Instant;

use xcl_dxil::consts::*;
use xcl_dxil::dxop;
use xcl_dxil::ir::*;
use xcl_dxil::shader::*;
use xcl_dxil::{flags, meta};

/// `blocks` diamonds in a row, each loading from a raw buffer (or an input), mixing with
/// 64-bit arithmetic and storing (to the buffer, or an output) on one side.
fn module(blocks: usize, per_block: usize, graphics: bool) -> Module {
    let mut m = Module::new();
    let mut body = Body::new();
    let mut bb = Body::ENTRY;
    let (bind_ty, props_ty) = (dxop::res_bind_ty(&mut m), dxop::res_props_ty(&mut m));
    let fields = vec![m.const_i32(0), m.const_i32(0), m.const_i32(0), m.const_i8(ResourceClass::UAV as u8)];
    let bind = m.aggregate(bind_ty, fields);
    let (zero, no) = (m.const_i32(0), m.const_bool(false));
    let h = dxop::call(&mut m, &mut body, bb, OpCode::CreateHandleFromBinding, Ty::VOID, &[bind, zero, no], false);
    let fields = vec![m.const_i32(ResourceKind::RawBuffer as u32 | 1 << 12), zero];
    let props = m.aggregate(props_ty, fields);
    let io = dxop::call(&mut m, &mut body, bb, OpCode::AnnotateHandle, Ty::VOID, &[h, props], false);
    let (undef, mask, align) = (m.undef(Ty::I32), m.const_i8(1), m.const_i32(4));
    let mut acc = zero;
    for k in 0..blocks {
        let at = m.const_i32(4 * (k as u32 & 1023));
        let mut x = if graphics {
            let (id, col) = (m.const_i32(k as u32 % 8), m.const_i8(k as u8 % 4));
            dxop::call(&mut m, &mut body, bb, OpCode::LoadInput, Ty::I32, &[id, zero, col, undef], false)
        } else {
            let r = dxop::call(&mut m, &mut body, bb, OpCode::RawBufferLoad, Ty::I32, &[io, at, undef, mask, align], false);
            body.extract(bb, Ty::I32, r, 0)
        };
        for _ in 0..per_block / 4 {
            let wide = body.cast(bb, CastOp::ZExt, Ty::I64, x);
            let sum = body.bin(bb, BinOp::Add, Ty::I64, wide, wide);
            let low = body.cast(bb, CastOp::Trunc, Ty::I32, sum);
            x = body.bin(bb, BinOp::Xor, Ty::I32, low, acc);
        }
        let (then, merge) = (body.new_block(), body.new_block());
        let cond = body.icmp(bb, IPred::Ne, x, zero);
        body.cond_br(bb, cond, then, merge);
        if graphics {
            let (id, col) = (m.const_i32(k as u32 % 8), m.const_i8(k as u8 % 4));
            dxop::call(&mut m, &mut body, then, OpCode::StoreOutput, Ty::I32, &[id, zero, col, x], false);
        } else {
            dxop::call(&mut m, &mut body, then, OpCode::RawBufferStore, Ty::I32, &[io, at, undef, x, undef, undef, undef, mask, align], false);
        }
        body.br(then, merge);
        let phi = body.phi(merge, Ty::I32);
        body.phi_add(phi, x, then);
        body.phi_add(phi, zero, bb);
        acc = phi;
        bb = merge;
    }
    body.ret(bb);
    m.define("main", body);
    m
}

#[test]
#[ignore]
fn passes_are_linear_and_fast() {
    let mut cs = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [64, 1, 1], wave_size: None });
    cs.resources.push(Resource::new(ResourceClass::UAV, 0, "io", ResourceKind::RawBuffer, 0, 0));
    for blocks in [50_000, 200_000] {
        let m = module(blocks, 16, false);
        let n = m.funcs.iter().filter_map(|f| f.body.as_ref()).map(|b| b.insts.len()).sum::<usize>();
        let t = Instant::now();
        let mut info = cs.clone();
        meta::prepare(&m, &mut info);
        eprintln!("compute, {n} instructions: prepare {:?} (flags {:#x})", t.elapsed(), info.flags);
        assert_eq!(info.flags, flags::collect(&m, &cs));
    }

    let mut vs = ShaderInfo::new(ShaderKind::Vertex, "main", StageProps::Vertex);
    let element = |row: i32| SigElement {
        name: "TEXCOORD".into(),
        indices: vec![row as u32],
        kind: SemanticKind::Arbitrary,
        comp_type: ComponentType::U32,
        interp: InterpolationMode::Constant,
        rows: 1,
        cols: 4,
        start_row: row,
        start_col: 0,
        stream: 0,
        usage_mask: 15,
        dynamic_index_mask: 0,
    };
    vs.inputs = (0..8).map(element).collect();
    vs.outputs = (0..8).map(element).collect();
    for blocks in [10_000, 40_000, 160_000] {
        let m = module(blocks, 16, true);
        let n = m.funcs.iter().filter_map(|f| f.body.as_ref()).map(|b| b.insts.len()).sum::<usize>();
        let t = Instant::now();
        let state = meta::view_id_state(&m, &vs);
        eprintln!("vertex, {n} instructions: view id state {:?} ({} words)", t.elapsed(), state.len());
        assert_eq!(state[..2], [32, 32]);
    }
}
