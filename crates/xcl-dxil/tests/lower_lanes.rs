//! The lane-uniformity check behind `XCL_DXIL_LANE_CHECK` (`super::lanes` in the lowering):
//! a loop around a wave operation is refused where DXC threads a way out of it that the
//! lowering does not, and lanes may leave it at different times. Every test here runs with
//! the switch on (`2`: the message says why), so they live in a file of their own.

use xcl_dxil::hlsl;
use xcl_dxil::Error;

const HEAD: &str = "RWByteAddressBuffer io : register(u0);\nRWByteAddressBuffer res : register(u1);\n";
const APART: &str = "a loop around a wave operation whose exit can differ between lanes";

/// The switch is the process's: one lowering at a time sets it and lowers.
static SWITCH: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// "ok", or the message of the refusal.
fn check(globals: &str, setup: &str, body: &str) -> String {
    check_mode("2", globals, setup, body)
}

fn check_mode(mode: &str, globals: &str, setup: &str, body: &str) -> String {
    let src = format!(
        "{HEAD}{globals}\n[WaveSize(32)]\n[numthreads(32, 1, 1)]\nvoid main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID) {{\nuint n = 0u; uint m = 0u;\n{setup}\n{body}\nres.Store(gi * 8u, n); res.Store(gi * 8u + 4u, m);\n}}\n"
    );
    let unit = hlsl::parse::parse(&src).unwrap_or_else(|e| panic!("{e}"));
    let _one = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("XCL_DXIL_LANE_CHECK", mode);
    match hlsl::lower::lower(&unit, "main", "cs_6_6") {
        Ok(_) => "ok".into(),
        Err(Error::Unsupported(m)) => m,
        Err(e) => panic!("{e}"),
    }
}

/// DXC threads the flag's edge to the loop's exit through the rotated condition.
const WHILE_FLAG: &str = "bool done = false; uint i = 0u; [loop] while (!done) { if (i == rounds) { n = WaveActiveCountBits(true) + 100u * i; done = true; } i++; }";

fn refused(r: &str) -> bool {
    r.starts_with(APART)
}

#[test]
fn a_loop_whose_lanes_leave_apart_is_refused() {
    let r = check("", "uint rounds = gi & 3u;", WHILE_FLAG);
    assert!(refused(&r), "{r}");
    // the message follows the variance back to where it comes from
    assert!(r.contains("gi") && r.contains("rounds") && r.contains("done"), "{r}");
}

#[test]
fn a_loop_whose_exit_is_the_same_for_every_lane_is_not() {
    for setup in [
        "uint rounds = res.Load(0u) & 3u;",
        "uint rounds = gid.x & 3u;",
        "uint rounds = WaveReadLaneFirst(gi) & 3u;",
        "uint rounds = WaveActiveMax(gi) & 3u;",
        "uint rounds = countbits(WaveActiveBallot(gi > 4u).x) & 3u;",
        "uint rounds = res.Load(res.Load(0u) * 4u) & 3u;",
        "uint rounds = WaveReadLaneAt(gi, 3u) & 3u;",
    ] {
        assert_eq!(check("", setup, WHILE_FLAG), "ok", "{setup}");
    }
    for setup in [
        "uint rounds = WaveReadLaneAt(3u, gi) & 3u;",
        "uint rounds = WavePrefixSum(1u) & 3u;",
        "uint rounds = (uint)WaveIsFirstLane();",
        "uint rounds = res.Load(gi * 4u) & 3u;",
        "uint rounds = 0u; res.InterlockedAdd(0u, 1u, rounds);",
    ] {
        assert!(refused(&check("", setup, WHILE_FLAG)), "{setup}");
    }
}

#[test]
fn what_a_varying_branch_assigns_varies() {
    assert!(refused(&check("", "uint rounds = 1u; if (gi > 3u) rounds = 2u;", WHILE_FLAG)));
    assert!(refused(&check("", "uint rounds = 1u; for (uint k = 0u; k < gi; k++) rounds++;", WHILE_FLAG)));
    assert_eq!(check("", "uint rounds = 1u; if (res.Load(0u) > 3u) rounds = 2u;", WHILE_FLAG), "ok");
    // after lanes return, the others run on: a local they assign is the same for all of them
    assert_eq!(check("", "if (gi == 31u) return; uint rounds = res.Load(0u) & 3u;", WHILE_FLAG), "ok");
}

#[test]
fn a_function_is_summarised_by_the_parameters_its_result_depends_on() {
    let f = "uint low(uint a, uint b) { return a & 3u; }";
    assert_eq!(check(f, "uint rounds = low(res.Load(0u), gi);", WHILE_FLAG), "ok");
    assert!(refused(&check(f, "uint rounds = low(gi, 0u);", WHILE_FLAG)));
    // another call with a per-lane argument does not make this one vary
    assert_eq!(check(f, "uint other = low(gi, gi); uint rounds = low(res.Load(0u), 0u); m = other;", WHILE_FLAG), "ok");
    let out = "void get(uint a, out uint r) { r = a & 3u; }";
    assert_eq!(check(out, "uint rounds; get(res.Load(0u), rounds);", WHILE_FLAG), "ok");
    assert!(refused(&check(out, "uint rounds; get(gi, rounds);", WHILE_FLAG)));
    // a global a function stores depends on nothing of the caller's: what reaches it varies
    let g = "static uint lane_g = 0u; uint lane() { return lane_g; }";
    assert!(refused(&check(g, "lane_g = WaveGetLaneIndex(); uint rounds = lane() & 3u;", WHILE_FLAG)));
    assert_eq!(check(g, "lane_g = res.Load(0u); uint rounds = lane() & 3u;", WHILE_FLAG), "ok");
}

#[test]
fn the_half_of_a_split_hardware_wave_counts_as_uniform() {
    // a wave32 console program on 64-lane hardware: each half is a console wave of its own,
    // and the precondition is control flow uniform within a console wave
    let g = "static bool xcl_w32 = false; static uint xcl_half = 0u; static uint my_half = 0u;";
    let set = "xcl_w32 = WaveGetLaneCount() > 32u; xcl_half = xcl_w32 ? WaveGetLaneIndex() >> 5 : 0u; my_half = xcl_half;";
    assert_eq!(check(g, &format!("{set} uint rounds = (res.Load(0u) + 32u * xcl_half) & 3u;"), WHILE_FLAG), "ok");
    // the rule is for those two globals only
    assert!(refused(&check(g, &format!("{set} my_half = WaveGetLaneIndex() >> 5; uint rounds = my_half & 3u;"), WHILE_FLAG)));
    // rla's lane read: uniform with a uniform lane once its fallback is not the lane's own
    let rla = "static bool xcl_w32 = false; static uint xcl_half = 0u;\nuint rla(uint v, uint l) { uint target = l + 32u * xcl_half; bool in_range = target < WaveGetLaneCount(); uint h = select(in_range, target, FALLBACK); uint w = WaveReadLaneAt(v, h); return select(in_range, w, 0u); }";
    let setup = "xcl_w32 = WaveGetLaneCount() > 32u; xcl_half = xcl_w32 ? WaveGetLaneIndex() >> 5 : 0u; uint rounds = rla(gi, res.Load(0u) & 31u) & 3u;";
    assert_eq!(check(&rla.replace("FALLBACK", "0u"), setup, WHILE_FLAG), "ok");
    assert!(refused(&check(&rla.replace("FALLBACK", "WaveGetLaneIndex()"), setup, WHILE_FLAG)));
}

#[test]
fn the_dispatcher_of_translated_programs_is_not_refused() {
    // pc only takes constants and choices on values that are the same for the wave
    let body = "[loop] while (pc >= 16u && pc < 64u) { if (pc == 16u) { n = WaveActiveCountBits(true); pc = 100u; } if (pc == 20u) { m = 1u; pc = (res.Load(4u) == 0u) ? 16u : 24u; } }";
    assert_eq!(check("", "uint pc = 16u + (res.Load(0u) & 1u) * 4u;", body), "ok");
    assert!(refused(&check("", "uint pc = 16u + (gi & 1u) * 4u;", body)));
}

/// The half check (`=3`): the two halves of a split console wave must reach every
/// `xcl_x_meet` together. A wave operation's result is its own half's; an exchange's combines
/// both halves.
#[test]
fn a_meeting_of_the_halves_under_a_condition_one_half_alone_takes_is_reported() {
    let prelude = "static uint xcl_half = 0u; static uint xcl_xn = 32u;\n\
        bool xcl_x_meet() { if (xcl_xn == 0u) return false; GroupMemoryBarrierWithGroupSync(); return true; }\n\
        uint2 xcl_x_ballot(bool b) { uint m = WaveActiveBallot(b).x; uint o = xcl_x_meet() ? m : 0u; return xcl_half != 0u ? uint2(o, m) : uint2(m, o); }";
    let half = "xcl_half = WaveReadLaneFirst(gi) >> 5;";
    let reported = |body: &str| check_mode("3", prelude, half, body);
    let one = "1 calls that reach xcl_x_meet under a condition that can differ between the halves of a wave";
    let r = reported("if (xcl_half == 0u) { uint2 e = xcl_x_ballot(true); n = e.x; }");
    assert!(r.starts_with(one) && r.contains("xcl_half"), "{r}");
    // a hardware wave's own operation differs between the halves
    assert!(reported("if (WaveActiveAnyTrue(gi > 3u)) { uint2 e = xcl_x_ballot(true); n = e.x; }").starts_with(one));
    // so does a lane's return before the meeting
    assert!(reported("if (gi == 3u) return; uint2 e = xcl_x_ballot(true); n = e.x;").starts_with(one));
    // an exchange's result is the same in both halves: what it decides is not reported
    assert_eq!(reported("uint2 e = xcl_x_ballot(gi > 3u); if (e.x != 0u) { uint2 f = xcl_x_ballot(true); n = f.y; }"), "ok");
    assert_eq!(reported("uint2 e = xcl_x_ballot(true); n = e.x + e.y;"), "ok");
    // mode 1 does not look at it
    assert_eq!(check_mode("1", prelude, half, "if (xcl_half == 0u) { uint2 e = xcl_x_ballot(true); n = e.x; }"), "ok");
}

/// The half check follows each local through its assignments in program order: what it held
/// before a whole assignment does not reach the reads after it, and what a loop leaves in it
/// reaches the next round and the code after the loop.
#[test]
fn the_half_check_reads_what_reaches_each_use() {
    let prelude = "static uint xcl_half = 0u; static uint xcl_xn = 32u;\n\
        bool xcl_x_meet() { if (xcl_xn == 0u) return false; GroupMemoryBarrierWithGroupSync(); return true; }\n\
        uint2 xcl_x_ballot(bool b) { uint m = WaveActiveBallot(b).x; uint o = xcl_x_meet() ? m : 0u; return xcl_half != 0u ? uint2(o, m) : uint2(m, o); }";
    let half = "xcl_half = WaveReadLaneFirst(gi) >> 5;";
    let reported = |body: &str| check_mode("3", prelude, half, body);
    let meet = "{ uint2 f = xcl_x_ballot(true); n += f.x; }";
    // a mask from the half's own ballot, replaced by an exchanged one before the loop reads it
    let r = reported(&format!("m = WaveActiveBallot(gi > 3u).x; uint2 e = xcl_x_ballot(gi > 5u); m = e.x; for (uint k = 0u; k < 4u; k++) {{ if (m != 0u) {meet} m = m >> 1; }}"));
    assert_eq!(r, "ok");
    // a flag rewritten in the loop before it is tested
    let r = reported(&format!("uint s = WaveActiveBallot(true).x; uint2 e = xcl_x_ballot(gi > 5u); for (uint k = 0u; k < 4u; k++) {{ s = e.x & k; if (s != 0u) {meet} }}"));
    assert_eq!(r, "ok");
    // what one way of a branch leaves meets what the other leaves
    let one = "1 calls that reach xcl_x_meet";
    let r = reported(&format!("uint2 e = xcl_x_ballot(true); m = e.x; if (gi > 7u) m = WaveActiveBallot(true).x; if (m != 0u) {meet}"));
    assert!(r.starts_with(one), "{r}");
    // the next round reads what the last one left
    let r = reported(&format!("for (uint k = 0u; k < 4u; k++) {{ if (m != 0u) {meet} m = WaveActiveBallot(gi > k).x; }}"));
    assert!(r.starts_with(one), "{r}");
    // and so does the code after a break
    let r = reported(&format!("for (uint k = 0u; k < 4u; k++) {{ m = WaveActiveCountBits(gi > k); if (k == 2u) break; m = 0u; }} if (m != 0u) {meet}"));
    assert!(r.starts_with(one), "{r}");
    // a part of a vector keeps the rest
    let r = reported(&format!("uint2 v = uint2(WaveActiveCountBits(gi > 3u), 0u); v.y = 1u; if (v.x != 0u) {meet}"));
    assert!(r.starts_with(one), "{r}");
}
