//! The lowering, construct by construct: each test lowers a small compute shader and checks
//! the shape of the IR ([`irprint`] text) against what DXC -O1 produces for the same source:
//! no memory for scalars, vectors and small arrays, phis only where values really merge,
//! HLSL's shift, conversion and short-circuit rules.

use xcl_dxil::hlsl;
use xcl_dxil::Error;

const HEAD: &str = "RWByteAddressBuffer io : register(u0);\nRWByteAddressBuffer res : register(u1);\n";

fn lower(body: &str) -> Result<(String, xcl_dxil::shader::ShaderInfo), Error> {
    let src = format!("{HEAD}{body}");
    let unit = hlsl::parse::parse(&src)?;
    let (m, info) = hlsl::lower::lower(&unit, "main", "cs_6_6")?;
    let f = m.funcs.iter().find_map(|f| f.body.as_ref()).expect("the entry function has a body");
    // only what the writer keeps
    let live = xcl_dxil::bcwriter::liveness(f);
    let text = xcl_dxil::irprint::function(&m, f);
    assert!(live.blocks[0]);
    Ok((text, info))
}

/// The IR of `void main() { <stmts> }`.
fn ir(stmts: &str) -> String {
    ir_with("", stmts)
}

fn ir_with(globals: &str, stmts: &str) -> String {
    match lower(&format!("{globals}\n[numthreads(1, 1, 1)]\nvoid main() {{\n{stmts}\n}}\n")) {
        Ok((t, _)) => t,
        Err(e) => panic!("{e}"),
    }
}

fn count(text: &str, what: &str) -> usize {
    text.matches(what).count()
}

fn blocks(text: &str) -> usize {
    1 + count(text, "; <label>:")
}

#[test]
fn small_branches_are_selects() {
    let t = ir("uint a = io.Load(0u); uint x = 1u; if (a > 3u) x = a + 2u; else x = a ^ 5u; res.Store(0u, x);");
    assert_eq!(blocks(&t), 1, "{t}");
    assert_eq!(count(&t, " = select "), 1, "{t}");
    assert_eq!(count(&t, " = phi "), 0, "{t}");
    assert_eq!(count(&t, "alloca"), 0, "{t}");
}

#[test]
fn a_branch_around_a_select_call_is_selects() {
    let t = ir("uint a = io.Load(0u); uint x = io.Load(4u); bool m = a > 7u; if (a != 3u) x = select(m, a, 0u); res.Store(0u, x);");
    assert_eq!(blocks(&t), 1, "{t}");
    assert_eq!(count(&t, " = select "), 2, "{t}");
}

#[test]
fn constant_offsets_combine() {
    let t = ir("uint a = io.Load(0u); res.Store(0u, (a + 3u) - 1u); res.Store(4u, (a - 2u) - 5u); res.Store(8u, (a + 1u) - 1u); res.Store(12u, (a - 4u) + 6u);");
    assert_eq!(count(&t, " = sub "), 0, "{t}");
    // a + 2 (twice: the last store reuses it), a - 7; the third store is a itself
    assert_eq!(count(&t, " = add "), 2, "{t}");
}

#[test]
fn a_mask_moves_after_its_shift() {
    // DXC's (s << 3) & 24: the multiply is a shift, the mask moves after it, the second folds in
    let t = ir("uint a = io.Load(0u); res.Store(0u, ((a & 3u) * 8u) & 31u);");
    assert_eq!(count(&t, " = mul "), 0, "{t}");
    assert_eq!(count(&t, " = shl i32 "), 1, "{t}");
    assert_eq!(count(&t, " = and i32 "), 1, "{t}");
    assert!(t.contains(", 24"), "{t}");
}

#[test]
fn a_branch_with_an_effect_stays_a_branch() {
    let t = ir("uint a = io.Load(0u); uint x = 1u; if (a > 3u) { res.Store(4u, a); x = a; } res.Store(0u, x);");
    assert_eq!(count(&t, "br i1 "), 1, "{t}");
    assert_eq!(count(&t, " = phi "), 1, "{t}");
    // the store is under the branch, not before it
    let br = t.find("br i1 ").unwrap();
    assert!(t.find("i32 4, i32 undef").unwrap() > br, "{t}");
}

#[test]
fn a_loop_has_one_phi_per_variable_it_changes() {
    let t = ir("uint n = io.Load(0u); uint k = io.Load(4u); uint s = 0u; for (uint i = 0u; i < n; i++) { s += i * k; } res.Store(0u, s);");
    // i and s; k and n are not touched by the loop
    assert_eq!(count(&t, " = phi "), 2, "{t}");
    assert_eq!(count(&t, "alloca"), 0, "{t}");
    assert_eq!(count(&t, "br i1 "), 1, "{t}");
}

#[test]
fn breaks_meet_in_one_exit_block() {
    let t = ir("uint n = io.Load(0u); uint s = 0u; uint i = 0u; while (true) { if (i == n) break; s += io.Load(i * 4u); if (s > 100u) break; i++; } res.Store(0u, s);");
    // header: i, s. exit: s once (two breaks, one phi)
    assert_eq!(count(&t, " = phi "), 3, "{t}");
    assert_eq!(count(&t, "ret void"), 1, "{t}");
}

#[test]
fn a_small_array_indexed_at_run_time_is_selects() {
    let t = ir("uint a = io.Load(0u); uint v[4] = { 1u, 2u, 3u, 4u }; v[a & 3u] = a; res.Store(0u, v[(a >> 2) & 3u]);");
    assert_eq!(count(&t, "alloca"), 0, "{t}");
    assert_eq!(count(&t, "getelementptr"), 0, "{t}");
    // four for the write, three for the read
    assert_eq!(count(&t, " = select "), 7, "{t}");
}

#[test]
fn a_large_array_is_memory() {
    let t = ir("uint a = io.Load(0u); uint v[64]; v[a & 63u] = a; res.Store(0u, v[(a >> 6) & 63u]);");
    assert_eq!(count(&t, "alloca [64 x i32]"), 1, "{t}");
    assert_eq!(count(&t, "getelementptr inbounds"), 2, "{t}");
}

#[test]
fn shifts_mask_their_count() {
    let t = ir("uint a = io.Load(0u); uint b = io.Load(4u); int s = (int)a; uint64_t q = (uint64_t)a; res.Store(0u, (a << b) + (uint)(s >> b) + (uint)(q << b));");
    assert_eq!(count(&t, "and i32 %"), 1, "{t}");
    assert!(t.contains(", 31\n"), "{t}");
    assert!(t.contains("and i64") && t.contains(", 63\n"), "{t}");
    assert_eq!(count(&t, " = ashr i32 "), 1, "{t}");
    assert_eq!(count(&t, " = shl i64 "), 1, "{t}");
}

#[test]
fn division_keeps_its_operation() {
    let t = ir("uint a = io.Load(0u); int s = (int)a; res.Store(0u, a / 7u + a % 7u + (uint)(s / 7) + (uint)(s % 7) + a / 8u);");
    assert_eq!(count(&t, " = udiv i32 "), 1, "{t}");
    assert_eq!(count(&t, " = urem i32 "), 1, "{t}");
    assert_eq!(count(&t, " = sdiv i32 "), 1, "{t}");
    assert_eq!(count(&t, " = srem i32 "), 1, "{t}");
    // an unsigned division by a power of two is the shift, as in DXC
    assert_eq!(count(&t, " = lshr i32 "), 1, "{t}");
    assert_eq!(count(&t, " = mul "), 0, "{t}");
}

#[test]
fn equal_expressions_are_computed_once() {
    let t = ir("uint a = io.Load(0u); uint b = io.Load(4u); uint x = a * b + 1u; uint y = b * a + 1u; res.Store(0u, x ^ y ^ ((1u << 4) + 3u));");
    assert_eq!(count(&t, " = mul i32 "), 0, "{t}");
    // x ^ x is 0, so only the constant is left
    assert!(t.contains("i32 19, i32 undef"), "{t}");
}

#[test]
fn a_constant_table_element_is_read_once_where_its_read_dominates() {
    let table = "static const uint T[16] = { 3u, 1u, 4u, 1u, 5u, 9u, 2u, 6u, 5u, 3u, 5u, 8u, 9u, 7u, 9u, 3u };";
    let t = ir_with(table, "uint i = io.Load(0u) & 15u; uint a = T[i]; uint b = T[i] * 3u; if (a > 2u) res.Store(4u, T[i] + 1u); res.Store(0u, a + b + T[(i + 1u) & 15u]);");
    assert_eq!(count(&t, " = load "), 2, "{t}");
    // a read under a branch does not stand for the reads after it
    let t = ir_with(table, "uint i = io.Load(0u) & 15u; if (i > 2u) res.Store(4u, T[i]); res.Store(0u, T[i]);");
    assert_eq!(count(&t, " = load "), 2, "{t}");
}

#[test]
fn short_circuit_keeps_effects_under_the_condition() {
    let g = "uint bump(uint v) { res.Store(8u, v); return v; }";
    let t = ir_with(g, "uint a = io.Load(0u); bool c = a > 1u && bump(a) > 2u; res.Store(0u, c ? 1u : 0u);");
    assert_eq!(count(&t, "br i1 "), 1, "{t}");
    let t = ir("uint a = io.Load(0u); uint b = io.Load(4u); bool c = a > 1u && b > 2u; res.Store(0u, c ? 1u : 0u);");
    assert_eq!(count(&t, "br i1 "), 0, "{t}");
    assert_eq!(count(&t, " = and i1 "), 1, "{t}");
}

#[test]
fn calls_are_inlined_with_copy_in_copy_out() {
    let g = "uint f(uint a, out uint o, inout uint io2) { o = a * 3u; io2 += a; if (a > 5u) return a - 1u; return a + 1u; }";
    let t = ir_with(g, "uint a = io.Load(0u); uint o; uint z = 7u; uint r = f(a, o, z); res.Store(0u, r); res.Store(4u, o); res.Store(8u, z);");
    assert_eq!(count(&t, "alloca"), 0, "{t}");
    assert_eq!(count(&t, " = mul i32 "), 1, "{t}");
    // z + a with z = 7
    assert!(t.contains(", 7\n"), "{t}");
    assert_eq!(count(&t, "call void @dx.op.rawBufferStore"), 3, "{t}");
}

#[test]
fn a_return_inside_a_loop_of_a_callee_leaves_only_the_callee() {
    let g = "uint find(uint n) { for (uint i = 0u; i < n; i++) { if (io.Load(i * 4u) == 0u) return i; } return n; }";
    let t = ir_with(g, "uint n = io.Load(64u); uint r = find(n); res.Store(0u, r + 1u);");
    assert_eq!(count(&t, "ret void"), 1, "{t}");
    assert_eq!(count(&t, "call void @dx.op.rawBufferStore"), 1, "{t}");
    assert_eq!(count(&t, "alloca"), 0, "{t}");
}

#[test]
fn groupshared_atomics() {
    let g = "groupshared uint lds[128];\ngroupshared uint one;";
    let (t, info) = lower(&format!("{g}\n[numthreads(1, 1, 1)]\nvoid main() {{ uint a = io.Load(0u); uint old; InterlockedAdd(lds[a & 127u], 5u, old); InterlockedCompareExchange(lds[1], a, 9u, old); InterlockedOr(one, a); res.Store(0u, old); }}")).unwrap();
    assert_eq!(count(&t, " = atomicrmw add i32 addrspace(3)* "), 1, "{t}");
    assert_eq!(count(&t, " = atomicrmw or i32 addrspace(3)* "), 1, "{t}");
    assert_eq!(count(&t, " = cmpxchg i32 addrspace(3)* "), 1, "{t}");
    // DXC: no inbounds where a run-time index addresses groupshared memory
    assert_eq!(count(&t, "getelementptr [128 x i32]"), 1, "{t}");
    assert_eq!(info.groupshared_bytes, 128 * 4 + 4);
}

#[test]
fn unused_resources_and_globals_are_dropped() {
    let g = "Texture2D<float4> unused : register(t3);\ngroupshared uint never[16];\nstatic uint table[64];";
    let (t, info) = lower(&format!("{g}\n[numthreads(1, 1, 1)]\nvoid main() {{ res.Store(0u, io.Load(0u)); }}")).unwrap();
    assert_eq!(info.resources.len(), 2, "{t}");
    assert_eq!(info.groupshared_bytes, 0);
    assert_eq!(count(&t, "createHandleFromBinding"), 2, "{t}");
}

#[test]
fn vectors_and_structs_are_components() {
    let g = "struct P { uint3 v; float w; };";
    let t = ir_with(g, "uint a = io.Load(0u); P p; p.v = uint3(a, a + 1u, a + 2u); p.w = 1.5f; p.v.zx = p.v.xy; P q = p; res.Store(0u, q.v.x + q.v.y + q.v.z + (uint)q.w);");
    assert_eq!(count(&t, "alloca"), 0, "{t}");
    assert_eq!(count(&t, "insertelement"), 0, "{t}");
    assert_eq!(count(&t, " = phi "), 0, "{t}");
}

#[test]
fn what_is_not_lowered_says_so() {
    let e = lower("[numthreads(1, 1, 1)]\nvoid main() { float4x4 m; }").unwrap_err();
    assert_eq!(e, Error::Unsupported("matrix types".into()));
    let e = lower("uint f(uint a) { return a == 0u ? 0u : f(a - 1u); }\n[numthreads(1, 1, 1)]\nvoid main() { res.Store(0u, f(io.Load(0u))); }").unwrap_err();
    assert_eq!(e, Error::Unsupported("recursion".into()));
}

#[test]
fn wave_size_is_for_compute_only() {
    let src = "[WaveSize(32)]\n[numthreads(1, 1, 1)]\n[outputtopology(\"triangle\")]\nvoid main() { }\n";
    let unit = hlsl::parse::parse(src).unwrap();
    for profile in ["ms_6_6", "ms_6_8"] {
        assert!(matches!(hlsl::lower::lower(&unit, "main", profile), Err(Error::Source { .. })), "{profile}");
    }
}

/// A resource local is its heap index: the handle is fetched where the variable is assigned
/// and reused by every use that fetch dominates, the index merges like any integer, and no
/// handle goes through a phi or a select.
#[test]
fn a_heap_handle_local_is_fetched_once() {
    let t = ir("uint v = io.Load(0u); uint n = io.Load(4u); RWByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(v + 3u)]; uint s = 0u; for (uint i = 0u; i < n; i++) { if ((i & 1u) != 0u) s += b.Load(i * 4u); else b.Store(i * 4u, s); } res.Store(0u, s + b.Load(64u));");
    assert_eq!(count(&t, "@dx.op.createHandleFromHeap(i32 218, "), 1, "{t}");
    assert!(t.contains("i1 false, i1 true)"), "{t}");
    assert_eq!(count(&t, "phi %dx.types.Handle"), 0, "{t}");
    // the fetch is before the loop
    assert!(t.find("createHandleFromHeap").unwrap() < t.find(" = phi ").unwrap(), "{t}");
}

#[test]
fn a_heap_handle_local_assigned_in_both_arms_merges_its_index() {
    let t = ir("uint v = io.Load(0u); uint n = io.Load(4u); RWByteAddressBuffer b; if (n > 3u) { b = ResourceDescriptorHeap[NonUniformResourceIndex(v + 3u)]; res.Store(8u, b.Load(0u)); } else { b = ResourceDescriptorHeap[NonUniformResourceIndex(v + 5u)]; } res.Store(0u, b.Load(64u)); if (n > 7u) res.Store(4u, b.Load(68u));");
    assert_eq!(count(&t, "phi %dx.types.Handle"), 0, "{t}");
    assert_eq!(count(&t, " = phi i32 "), 1, "{t}");
    // one in the arm that uses it, one after the merge (which the later use shares)
    assert_eq!(count(&t, "@dx.op.createHandleFromHeap(i32 218, "), 2, "{t}");
}

#[test]
fn a_heap_handle_local_assigned_in_a_loop_is_a_phi_of_indices() {
    let t = ir("uint v = io.Load(0u); uint n = io.Load(4u); RWByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(v)]; uint s = 0u; for (uint i = 0u; i < n; i++) { s += b.Load(i * 4u); if (s > 100u) b = ResourceDescriptorHeap[NonUniformResourceIndex(v + i)]; } res.Store(0u, s + b.Load(64u));");
    assert_eq!(count(&t, "phi %dx.types.Handle"), 0, "{t}");
    // i, s and the index of b
    assert!(count(&t, " = phi i32 ") >= 3, "{t}");
}

#[test]
fn resource_locals_copy_and_pass_what_they_hold() {
    let g = "uint rd(RWByteAddressBuffer m, uint a) { return m.Load(a) + m.Load(a + 4u); }\nvoid wr(globallycoherent RWByteAddressBuffer m, uint a, uint x) { m.Store(a, x); }";
    let t = ir_with(g, "uint v = io.Load(0u); uint n = io.Load(4u); globallycoherent RWByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(v + 3u)]; Texture2D<float4> t = ResourceDescriptorHeap[v + 9u]; SamplerState sm = SamplerDescriptorHeap[n]; RWByteAddressBuffer e = ResourceDescriptorHeap[NonUniformResourceIndex(v + 2u)]; RWByteAddressBuffer c = e; RWByteAddressBuffer d = io; if (n > 3u) { c = ResourceDescriptorHeap[NonUniformResourceIndex(v + 4u)]; d = io; } float4 px = t.SampleLevel(sm, float2(0.5f, 0.5f), 0.0f); wr(b, 8u, rd(b, 0u) + rd(c, 16u) + rd(d, 32u) + (uint)px.x);");
    assert_eq!(count(&t, "phi %dx.types.Handle"), 0, "{t}");
    // the annotation b was fetched with (globallycoherent) stays on its handle
    assert!(t.contains("%dx.types.ResourceProperties { i32 20491, i32 0 }"), "{t}");
    // the sampler heap
    assert!(t.contains("i1 true, i1 false)"), "{t}");
}

#[test]
fn a_resource_local_of_two_kinds_is_not_lowered() {
    let e = lower("[numthreads(1, 1, 1)]\nvoid main() { uint v = io.Load(0u); RWByteAddressBuffer c = ResourceDescriptorHeap[v]; if (v > 3u) c = io; res.Store(0u, c.Load(0u)); }").unwrap_err();
    assert_eq!(e, Error::Unsupported("a resource variable assigned resources of different kinds".into()));
}

#[test]
fn a_switch_on_a_constant_is_its_case() {
    let t = ir("uint a = io.Load(0u); uint m = 0x00020003u; uint x = 0u; switch ((m >> 16) & 0x3FFu) { case 0u: x = a; break; case 1u: x = a + 1u; break; case 2u: x = a + 2u; case 3u: x += 5u; break; default: x = 9u; break; } res.Store(0u, x);");
    assert_eq!(blocks(&t), 1, "{t}");
    assert_eq!(count(&t, "switch"), 0, "{t}");
    // a + 2 + 5
    assert_eq!(count(&t, " = add i32 "), 1, "{t}");
    assert!(t.contains(", 7\n"), "{t}");
    let t = ir("uint a = io.Load(0u); uint x = 1u; switch (7u) { case 0u: x = a; break; } res.Store(0u, x);");
    assert!(t.contains("i32 0, i32 undef, i32 1, i32 undef"), "{t}");
}

#[test]
fn the_lane_count_is_asked_for_once() {
    let t = ir("uint a = WaveGetLaneCount(); uint b = io.Load(0u); if (b > 3u) { b += WaveGetLaneCount() + WaveGetLaneIndex(); } res.Store(0u, a + b + WaveGetLaneCount() + WaveGetLaneIndex());");
    assert_eq!(count(&t, "@dx.op.waveGetLaneCount("), 1, "{t}");
    // the index is first asked for under the branch: the use after it is not dominated
    assert_eq!(count(&t, "@dx.op.waveGetLaneIndex("), 2, "{t}");
}

/// A heap index is only known to be valid where the source uses it: an access the source
/// guards stays under its guard (DXC never runs a dx.op its branch skips).
#[test]
fn a_guarded_heap_access_stays_guarded() {
    let g = "uint ld(RWByteAddressBuffer m) { return m.Load(0u); }";
    let t = ir_with(g, "uint v = io.Load(0u); uint n = io.Load(4u); RWByteAddressBuffer b = ResourceDescriptorHeap[n]; uint x = v != 0u ? ld(ResourceDescriptorHeap[v]) : 7u; bool c = v > 4u && b.Load(4u) != 0u; res.Store(0u, x + (c ? 1u : 0u));");
    assert_eq!(count(&t, "br i1 "), 2, "{t}");
    // a register-bound resource can be read whether or not the guard holds
    let t = ir("uint v = io.Load(0u); uint x = v != 0u ? io.Load(8u) : 7u; res.Store(0u, x);");
    assert_eq!(count(&t, "br i1 "), 0, "{t}");
}

/// No instruction but a phi reads an undefined value and no UAV store writes one: the two
/// things the validator rejects ("Instructions should not read uninitialized value",
/// "Assignment of undefined values to UAV").
fn reads_no_undef(t: &str) {
    for line in t.lines() {
        let l = line.trim();
        if l.contains(" = phi ") {
            continue;
        }
        if l.contains("call ") {
            if l.contains("@dx.op.rawBufferStore") {
                let args: Vec<&str> = l[l.find('(').unwrap() + 1..].split(", ").collect();
                assert_ne!(args[2], "i32 undef", "{t}");
                assert_ne!(args[4], "i32 undef", "{t}");
            }
            continue;
        }
        assert!(!l.contains("undef"), "{l}\n{t}");
    }
}

/// The values the raw buffer stores write, in order.
fn stored(t: &str) -> Vec<String> {
    t.lines().filter(|l| l.contains("@dx.op.rawBufferStore")).map(|l| l[l.find('(').unwrap() + 1..].split(", ").nth(4).unwrap().to_string()).collect()
}

#[test]
fn an_undefined_operand_folds_as_in_dxc() {
    // what DXC -O1 stores for each (LLVM's folds of undef): 0, 0, -1, 0, 1, 0, 0, 0, 7
    let t = ir("uint a = io.Load(0u); uint x; res.Store(0u, x * a); res.Store(4u, x >> 2); res.Store(8u, x | a); res.Store(12u, x & a); \
                res.Store(16u, (uint)(x == a)); res.Store(20u, x << a); res.Store(24u, x / a); res.Store(28u, (uint)(float)x); res.Store(32u, x != 0u ? a : 7u);");
    assert_eq!(stored(&t), ["i32 0", "i32 0", "i32 -1", "i32 0", "i32 1", "i32 0", "i32 0", "i32 0", "i32 7"], "{t}");
    reads_no_undef(&t);
}

#[test]
fn an_undefined_value_that_is_stored_is_zero() {
    // DXC leaves these undefined and its output fails validation; here they are 0
    let t = ir("uint a = io.Load(0u); uint x; float f; uint z[4]; res.Store(0u, x); res.Store(4u, x + a); res.Store(8u, a - x); res.Store(12u, asuint(f * 2.0f)); \
                res.Store(16u, z[a & 3u]); res.Store(x, a); res.Store(20u, min(x, a)); res.Store(24u, countbits(x));");
    reads_no_undef(&t);
    assert_eq!(&stored(&t)[..3], ["i32 0", "i32 0", "i32 0"], "{t}");
}

#[test]
fn a_variable_assigned_on_one_path_is_that_value() {
    // DXC: phi [5], [undef] is 5
    let t = ir("uint a = io.Load(0u); uint x; if (a > 1u) x = 5u; res.Store(0u, x);");
    assert_eq!(stored(&t), ["i32 5"], "{t}");
    assert_eq!(blocks(&t), 1, "{t}");
    reads_no_undef(&t);
}

#[test]
fn an_undefined_condition_takes_the_else_side() {
    let t = ir("uint a = io.Load(0u); uint x; if (x > 3u) res.Store(0u, a); else res.Store(4u, a);");
    assert_eq!(count(&t, "@dx.op.rawBufferStore"), 1, "{t}");
    assert!(t.contains("i32 4, i32 undef, i32 %"), "{t}");
    assert_eq!(count(&t, "br i1 "), 0, "{t}");
    reads_no_undef(&t);
}

#[test]
fn an_out_parameter_the_callee_does_not_write() {
    let t = ir_with(
        "void f(uint a, out uint o) { if (a > 2u) o = a; }\nvoid g(out uint o) { }",
        "uint a = io.Load(0u); uint o; f(a, o); res.Store(0u, o); uint o2; g(o2); res.Store(4u, o2 & a); uint o3; g(o3); res.Store(8u, o3);",
    );
    let s = stored(&t);
    // as DXC: the first is a itself, the second 0; the third DXC rejects
    assert!(s[0].starts_with("i32 %"), "{t}");
    assert_eq!(&s[1..], ["i32 0", "i32 0"], "{t}");
    assert_eq!(count(&t, " = select "), 0, "{t}");
    reads_no_undef(&t);
}

#[test]
fn a_variable_defined_only_inside_a_loop() {
    let t = ir("uint a = io.Load(0u); uint x; for (uint i = 0u; i < a; i++) { x = i * 3u; } res.Store(0u, x); \
                uint y; for (uint j = 0u; j < a; j++) { if (y > 4u) break; y = j; } res.Store(4u, y);");
    // the loop headers keep a phi with an undefined entry value, as DXC's do
    assert!(count(&t, "[ undef, ") >= 2, "{t}");
    reads_no_undef(&t);
}

#[test]
fn a_variable_declared_in_a_loop_and_read_there() {
    // DXC: c - need is undefined, the signed compare with it true: the loop is left at once
    let t = ir("uint need = io.Load(0u); [loop] for (uint n = 0u; ; n++) { uint c; if ((int)(c - need) >= 0) break; } res.Store(0u, need);");
    assert_eq!(count(&t, "br i1 "), 0, "{t}");
    assert_eq!(count(&t, " = phi "), 0, "{t}");
    reads_no_undef(&t);
}

#[test]
fn partly_assigned_vectors_and_structs() {
    let t = ir_with(
        "struct P { uint a; float3 v; };\ngroupshared uint g[8];",
        "uint a = io.Load(0u); P p; p.a = a; res.Store(0u, p.a + (uint)p.v.y); float4 q; q.xy = float2(1.0f, 2.0f); res.Store(4u, asuint(q.x + q.w)); \
         res.Store3(8u, asuint(q.yzw)); uint k; g[k & 7u] = k; g[a & 7u] = p.v.x > 1.0f ? 1u : k; uint4 u; u.z = a; if (u.x == u.z) res.Store(20u, u.w);",
    );
    reads_no_undef(&t);
    assert!(!t.contains("store i32 undef"), "{t}");
}

#[test]
fn numthreads_outside_the_limits_is_an_error() {
    for attr in ["[numthreads(64, 0u, 1)]", "[numthreads(0xFFFFFFFFu, 1, 1)]", "[numthreads(64, 64, 1)]", "[numthreads(1, 1, 65)]"] {
        let r = lower(&format!("{attr}\nvoid main() {{ res.Store(0u, 1u); }}\n"));
        assert!(matches!(r, Err(Error::Source { .. })), "{attr}: {:?}", r.map(|x| x.0));
    }
    assert!(lower("[numthreads(32, 32, 1)]\nvoid main() { res.Store(0u, 1u); }\n").is_ok());
}

#[test]
fn literal_floats_and_ints_mix_as_doubles_and_fold() {
    // a literal float is a double until it meets a type, and a literal int next to one
    // becomes one: 0.5 + 1 is 1.5f, (1 + 0.5) meets the float as 1.5f
    let t = ir("uint a = io.Load(0u); float x = 0.5 + 1; float y = 2 * 0.25; float z = asfloat(a) * (1 + 0.5); \
                res.Store(0u, asuint(x)); res.Store(4u, asuint(y)); res.Store(8u, asuint(z));");
    assert_eq!(&stored(&t)[..2], ["i32 1069547520", "i32 1056964608"], "{t}");
    assert!(t.contains("fmul float %") && t.contains(", 1.5"), "{t}");
    assert!(!t.contains("double"), "{t}");
}

#[test]
fn a_literal_ternary_meets_its_type_without_a_double() {
    // DXC (instcombine's FoldOpIntoSelect): the conversion and an operation with a constant
    // go into the arms, so nothing of type double is left
    let t = ir("uint a = io.Load(0u); bool c = a > 3u; bool d = a > 7u; float f1 = c ? 0.1 : 0.2; float f2 = (c ? 1.5 : 2.5) * 3.0; \
                float f3 = c ? (d ? 0.1 : 0.7) : 0.2; float f4 = (c ? 2 : 3) * 0.5; int i = c ? 1.7 : 2.7; \
                res.Store(0u, asuint(f1)); res.Store(4u, asuint(f2)); res.Store(8u, asuint(f3)); res.Store(12u, asuint(f4)); res.Store(16u, (uint)i);");
    assert!(!t.contains("double"), "{t}");
    assert!(t.contains("i32 1036831949, i32 1045220557"), "{t}");
    assert!(t.contains("i32 1083179008, i32 1089470464"), "{t}");
    assert!(t.contains("i32 1065353216, i32 1069547520"), "{t}");
    assert!(t.contains("i32 1, i32 2"), "{t}");
    assert_eq!(count(&t, "fptrunc") + count(&t, "fmul") + count(&t, "sitofp") + count(&t, "fptosi"), 0, "{t}");
}

#[test]
fn a_literal_ternary_that_stays_a_double_is_one() {
    // as DXC: the destination is a double, and two literal ternaries added are doubles
    let t = ir("uint a = io.Load(0u); bool c = a > 3u; bool d = a > 7u; double x = c ? 0.1 : 0.2; uint lo, hi; asuint(x, lo, hi); \
                float f = (c ? 0.1 : 0.2) + (d ? 1 : 2); res.Store(0u, lo); res.Store(4u, hi); res.Store(8u, asuint(f));");
    assert_eq!(count(&t, "fadd double"), 1, "{t}");
    assert_eq!(count(&t, "fptrunc double"), 1, "{t}");
}

#[test]
fn a_division_by_a_constant_zero_is_undefined() {
    // LLVM's simplifier makes it undef (the validator rejects the instruction): what is
    // stored is 0, and a condition made of it is false
    let t = ir("uint a = io.Load(0u); int s = (int)a; res.Store(0u, a / 0u); res.Store(4u, a % 0u); res.Store(8u, (uint)(s / 0)); \
                if ((a & 2u) / 0u) res.Store(12u, a);");
    assert_eq!(count(&t, "div ") + count(&t, "rem "), 0, "{t}");
    assert_eq!(stored(&t), ["i32 0", "i32 0", "i32 0"], "{t}");
    reads_no_undef(&t);
}

#[test]
fn a_duplicate_case_is_an_error() {
    let r = lower("[numthreads(1, 1, 1)]\nvoid main() { uint a = io.Load(0u); switch (a) { case 1: res.Store(0u, a); break; case 2: break; case 1: res.Store(4u, a); break; } }\n");
    assert!(matches!(r, Err(Error::Source { .. })), "{:?}", r.map(|x| x.0));
}

#[test]
fn long_literals_are_32_bits_wide() {
    // as DXC: l is int, ul is uint
    let t = ir("uint a = io.Load(0u); res.Store(0u, a + 5l); res.Store(4u, (uint)(-1l >> 28)); res.Store(8u, (uint)(0x80000000UL >> 31)); res.Store(12u, (uint)((a + 0x7fffffffl + 0x7fffffffl) >> 31l));");
    assert_eq!(&stored(&t)[1..3], ["i32 -1", "i32 1"], "{t}");
    assert!(t.contains("add i32 %") && !t.contains("i64"), "{t}");
}

/// The module of a 32-lane compute shader: `n` is stored at the end, `rounds` differs by lane.
fn wave_ir(body: &str) -> String {
    let src = format!(
        "{HEAD}[WaveSize(32)]\n[numthreads(32, 1, 1)]\nvoid main(uint gi : SV_GroupIndex) {{\nuint n = 0u; uint m = 0u; uint rounds = gi & 3u;\n{body}\nres.Store(gi * 8u, n); res.Store(gi * 8u + 4u, m);\n}}\n"
    );
    let unit = hlsl::parse::parse(&src).unwrap_or_else(|e| panic!("{e}"));
    let (m, _) = hlsl::lower::lower(&unit, "main", "cs_6_6").unwrap_or_else(|e| panic!("{e}"));
    xcl_dxil::irprint::module(&m)
}

#[test]
fn a_break_that_computes_with_a_wave_result_stays_in_its_loop() {
    // DXC's dx.break: the block must still reach the back edge for the driver, or the wave
    // operation counts the lanes that left earlier too. The text is what drivers look for.
    for body in [
        "[loop] for (uint i = 0u; i < 4u; i++) { if (i == rounds) { n = WaveActiveCountBits(true) + 100u * i; break; } }",
        // the operation in a dominating block of the same iteration, its result used beside the break
        "[loop] for (uint i = 0u; i < 4u; i++) { uint w = WaveActiveCountBits((gi & 1u) == (i & 1u)); if (i == rounds) { n = w + 100u * i; break; } }",
    ] {
        let t = wave_ir(body);
        assert!(t.contains("@dx.break.cond = internal constant [1 x i32] zeroinitializer\n"), "{t}");
        assert!(t.contains("define void @main() {\n  %1 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.break.cond, i32 0, i32 0)\n  %2 = icmp eq i32 %1, 0\n"), "{t}");
        assert_eq!(count(&t, "br i1 %2, "), 1, "{t}");
    }
}

#[test]
fn a_break_no_wave_result_reaches_is_a_plain_branch() {
    // DXC takes the condition out again (CleanupDxBreak) where nothing depends on it
    for body in [
        // a copy is no instruction
        "[loop] for (uint i = 0u; i < 4u; i++) { uint w = WaveActiveCountBits((gi & 1u) == (i & 1u)) + 100u * i; if (i == rounds) { n = w; break; } }",
        // the operation is outside the loop
        "uint w = WaveActiveCountBits((gi & 1u) != 0u); [loop] for (uint i = 0u; i < 4u; i++) { if (i == rounds) { n = w + 100u * i; break; } }",
        // not sensitive to the lanes that are active
        "[loop] for (uint i = 0u; i < 4u; i++) { if (i == rounds) { n = WaveReadLaneAt(gi, 0u) + WaveGetLaneIndex() + 100u * i; break; } }",
    ] {
        let t = wave_ir(body);
        assert!(!t.contains("dx.break"), "{t}");
    }
    // and a program without wave operations never has one
    let t = ir("uint a = io.Load(0u); uint n = 0u; for (uint i = 0u; i < 4u; i++) { if (i == a) { n = i * 3u; break; } } res.Store(0u, n);");
    assert!(!t.contains("dx.break"), "{t}");
}

#[test]
fn a_flag_set_in_a_branch_takes_the_branch_where_the_flag_leads() {
    // DXC's jump threading: the edge on which the flag is known skips its test, so the wave
    // operation's block leaves the loop directly (and sees every lane that left)
    let t = wave_ir("[loop] for (uint i = 0u; i < 4u; i++) { bool skip = false; if (i == rounds) { n = WaveActiveCountBits(true) + 100u * i; skip = true; } if (skip) break; }");
    assert!(!t.contains("dx.break") && !t.contains("phi i1"), "{t}");
    assert_eq!(count(&t, "br i1 "), 2, "{t}");
    // through the join of an enclosing if, and through || with a flag that is not known
    let t = wave_ir(
        "[loop] for (uint i = 0u; i < 4u; i++) { bool a = false; bool b = (gi & 8u) != 0u; \
         if (i >= rounds) { m++; if ((gi & 4u) != 0u) { n = WaveActiveCountBits(true) + 100u * i; a = true; } } if (b || a) break; }",
    );
    let wave = t.find("= call i32 @dx.op.waveAllOp").unwrap();
    let end = wave + t[wave..].find("\n\n").unwrap();
    assert!(t[wave..end].lines().last().unwrap().trim_start().starts_with("br label "), "{t}");
    assert!(!t.contains("phi i1"), "{t}");
}

#[test]
fn a_flag_is_followed_only_where_a_lane_operation_can_tell() {
    // the same flag, but nothing in the branch depends on which lanes run it: one test on a phi
    let t = wave_ir("[loop] for (uint i = 0u; i < 4u; i++) { bool skip = false; if (i == rounds) { res.Store(64u, i); skip = true; } if (skip) break; n += WaveActiveCountBits(true); }");
    assert_eq!(count(&t, "phi i1"), 1, "{t}");
    // and outside a loop no block changes sides
    let t = wave_ir("bool f = false; if (rounds == 2u) { n = WaveActiveCountBits(true); f = true; } if (f) { m = 5u; }");
    assert_eq!(count(&t, "phi i1"), 1, "{t}");
}

#[test]
fn a_way_out_dxc_makes_by_copying_code_is_refused() {
    // DXC's jump threading copies "m += i * 3u" and the test into the branch; here that is not
    // done, and the program must not come out as if it were
    let src = format!(
        "{HEAD}[WaveSize(32)]\n[numthreads(32, 1, 1)]\nvoid main(uint gi : SV_GroupIndex) {{\nuint n = 0u; uint m = 0u; uint rounds = gi & 3u;\n\
         [loop] for (uint i = 0u; i < 4u; i++) {{ bool skip = false; if (i == rounds) {{ n = WaveActiveCountBits(true); skip = true; }} m += i * 3u; if (skip) break; }}\n\
         res.Store(gi * 8u, n + m);\n}}\n"
    );
    let unit = hlsl::parse::parse(&src).unwrap_or_else(|e| panic!("{e}"));
    let r = hlsl::lower::lower(&unit, "main", "cs_6_6");
    assert!(matches!(r, Err(Error::Unsupported(_))), "{:?}", r.map(|_| ()));
}

/// The message of the source error `src` gets, "ok" when it lowers.
fn verdict(src: &str, profile: &str) -> String {
    let unit = hlsl::parse::parse(src).unwrap();
    match hlsl::lower::lower(&unit, "main", profile) {
        Ok(_) => "ok".into(),
        Err(Error::Source { msg, .. }) => msg,
        Err(e) => panic!("{e}"),
    }
}

/// What the validator asks of a mesh shader beyond the front end: at most 128 threads, and
/// one SetMeshOutputCounts call whose block dominates every output store. DXC fails on each
/// of these sources with the same message (scratch/lower/i1).
#[test]
fn a_mesh_shader_keeps_the_validators_thread_group_and_output_count_rules() {
    let store = "XclVert o; o.pos = asfloat(res.Load4(16u)); verts[gi] = o; tris[0] = uint3(0u, 1u, 2u);";
    let set = "SetMeshOutputCounts(nv, 1u);";
    let cases = [
        ("8, 1, 1", format!("{set} {store}"), "ok"),
        ("8, 4, 4", format!("{set} {store}"), "ok"),
        ("1, 1, 128", format!("{set} {store}"), "ok"),
        ("128, 1, 32u", format!("{set} {store}"), "mesh entry point must have a valid numthreads attribute"),
        ("0, 1, 1", format!("{set} {store}"), "mesh entry point must have a valid numthreads attribute"),
        ("129, 1, 1", format!("{set} {store}"), "Declared Thread Group X size 129 outside valid range [1..128]."),
        ("64, 2, 2", format!("{set} {store}"), "Declared Thread Group Count 256 (X*Y*Z) is beyond the valid maximum of 128."),
        ("8, 1, 1", store.to_string(), "Missing SetMeshOutputCounts call."),
        ("8, 1, 1", format!("if (nv > 3u) {{ {set} }} {store}"), "Non-Dominating SetMeshOutputCounts call."),
        ("8, 1, 1", format!("if (nv > 3u) {{ {set} }} else {{ SetMeshOutputCounts(3u, 1u); }} {store}"), "SetMeshOutputCounts cannot be called multiple times."),
        // the call's block dominates: the same branch, the same block in either order, after an early return
        ("8, 1, 1", format!("if (nv > 3u) {{ {set} {store} }}"), "ok"),
        ("8, 1, 1", format!("{store} {set}"), "ok"),
        ("8, 1, 1", format!("if (nv == 0u) return; {set} {store}"), "ok"),
        ("8, 1, 1", format!("{set} for (uint i = gi; i < nv; i += 8u) {{ XclVert o; o.pos = asfloat(res.Load4(i * 16u)); verts[i] = o; }}"), "ok"),
    ];
    for (threads, body, want) in cases {
        let src = format!(
            "{HEAD}struct XclVert {{ float4 pos : SV_Position; }};\n[outputtopology(\"triangle\")]\n[numthreads({threads})]\nvoid main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID, out vertices XclVert verts[128], out indices uint3 tris[128])\n{{\n    uint nv = min(res.Load(gid.x * 4u), 128u);\n    {body}\n}}\n"
        );
        assert_eq!(verdict(&src, "ms_6_6"), want, "[numthreads({threads})] {body}");
    }
}

fn texture_case(threads: &str, texture: &str, stmts: &str) -> String {
    verdict(
        &format!("{HEAD}{texture} t : register(t0);\nSamplerState s : register(s0);\nSamplerComparisonState sc : register(s1);\n[numthreads({threads})]\nvoid main() {{\n    float2 uv = asfloat(io.Load2(12u));\n    {stmts}\n}}\n"),
        "cs_6_6",
    )
}

/// DXC samples a float texture only (integers from SM 6.7 on, never with a comparison); the
/// validator says the same of the instruction. Gather and Load take any element.
#[test]
fn only_a_float_texture_is_sampled() {
    let cases = [
        ("Texture2D<int>", "t.SampleLevel(s, uv, 0.0)", "cannot SampleLevel from resource containing int"),
        ("Texture2D<uint4>", "t.Sample(s, uv)", "cannot Sample from resource containing uint4"),
        ("Texture2D<int2>", "t.SampleGrad(s, uv, uv, uv)", "cannot SampleGrad from resource containing int2"),
        ("Texture2DArray<uint>", "t.SampleBias(s, float3(uv, 0.0), 1.0)", "cannot SampleBias from resource containing uint"),
        ("Texture2D<int>", "t.SampleCmp(sc, uv, 0.5)", "cannot SampleCmp from resource containing int"),
        ("Texture2D<uint>", "t.SampleCmpLevelZero(sc, uv, 0.5)", "cannot SampleCmpLevelZero from resource containing uint"),
        ("TextureCube<int4>", "t.SampleLevel(s, float3(uv, 1.0), 0.0)", "cannot SampleLevel from resource containing int4"),
        ("Texture2D<int>", "t.Gather(s, uv)", "ok"),
        ("Texture2D<uint4>", "t.GatherRed(s, uv)", "ok"),
        ("Texture2D<int>", "t.GatherCmp(sc, uv, 0.5)", "ok"),
        ("Texture2D<int>", "t.Load(int3(1, 2, 0))", "ok"),
        ("Texture2D<float4>", "t.SampleLevel(s, uv, 0.0)", "ok"),
        ("TextureCube<float4>", "t.Load(int4(1, 2, 0, 0))", "Load is not a member of a cube texture"),
    ];
    for (texture, call, want) in cases {
        assert_eq!(texture_case("4, 1, 1", texture, &format!("res.Store4(0u, asuint(({call}).xxxx));")), want, "{texture} {call}");
    }
}

/// Derivatives in a compute shader need threads that make quads: one dimension with X a
/// multiple of 4, or X and Y both even (the validator's CompatibilityChecker). Only the
/// operations that get written count, as in DXC, which removes an unused one first.
#[test]
fn derivatives_in_a_compute_shader_need_a_thread_group_of_quads() {
    let no = |t: &str| format!("Function uses derivatives in compute-model shader with NumThreads ({t}); derivatives require NumThreads to be 1D and a multiple of 4, or 2D/3D with X and Y both being a multiple of 2.");
    let store = |call: &str| format!("res.Store4(0u, asuint(({call}).xxxx));");
    for (threads, fits) in [("1, 1, 1", false), ("4, 1, 1", true), ("6, 1, 1", false), ("2, 2, 1", true), ("2, 3, 1", false), ("2, 1, 2", false), ("4, 1, 2", false), ("2, 2, 3", true)] {
        assert_eq!(texture_case(threads, "Texture2D<float4>", &store("t.Sample(s, uv)")), if fits { "ok".to_string() } else { no(threads) }, "{threads}");
    }
    for call in ["t.SampleBias(s, uv, 1.0)", "t.SampleCmp(sc, uv, 0.5)", "t.CalculateLevelOfDetail(s, uv)", "ddx(uv)", "ddy_fine(uv)", "fwidth(uv)"] {
        assert_eq!(texture_case("1, 1, 1", "Texture2D<float4>", &store(call)), no("1, 1, 1"), "{call}");
    }
    for call in ["t.SampleLevel(s, uv, 0.0)", "t.SampleGrad(s, uv, uv, uv)", "t.SampleCmpLevelZero(sc, uv, 0.5)", "t.Gather(s, uv)"] {
        assert_eq!(texture_case("1, 1, 1", "Texture2D<float4>", &store(call)), "ok", "{call}");
    }
    assert_eq!(texture_case("1, 1, 1", "Texture2D<float4>", "float4 unused = t.Sample(s, uv); res.Store(0u, 1u);"), "ok");
    assert_eq!(texture_case("1, 1, 1", "Texture2D<float4>", "if (false) res.Store4(0u, asuint(t.Sample(s, uv))); res.Store(0u, 1u);"), "ok");
}
