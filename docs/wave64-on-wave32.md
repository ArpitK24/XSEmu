# Wave64 console programs on 32-lane GPUs

Status: implemented for compute / kernel mode, the mesh (NGG) first pass and ES+GS
(`gpu/recomp/emit.rs`, `PRELUDE_X64`). The native path is untouched: the form below is only
generated when the adapter cannot run 64-lane waves (`D3D12_FEATURE_DATA_D3D12_OPTIONS1`
`WaveLaneCountMax < 64`, e.g. NVIDIA, whose D3D12 range is 32..32). On such an adapter a
`[WaveSize(64)]` compute shader is rejected at PSO creation, so until now wave64 compute
programs were refused there, and mesh / ES+GS programs ran as two independent halves only
when they had no cross-half operation (`geometry_wave32_hazards`) and were refused otherwise.

## Execution model it relies on

The recompiler runs every console lane through every instruction; EXEC only masks writes.
Branches are scalar (SCC, VCC/EXEC == 0, SGPR values), so control flow is uniform per
console wave as long as every scalar value is identical in all of the wave's threads.
Scalar values come from scalar code (identical when its inputs are), from scalar memory, and
from a handful of vector → scalar instructions: masks (`v_cmp*`, carries – all through
`ballot()`), `v_readlane` / `v_readfirstlane` (through `rla()`), and nothing else. Cross-lane
vector reads (DPP16, DPP8, `ds_swizzle`, `v_permlane(x)16`, `ds_bpermute`) also go through
`rla()`.

## The form

One console wave64 = two hardware wave32s of the same thread group ("halves"): the console
group keeps its `numthreads` and the shader gets `[WaveSize(32)]`. Which hardware threads
play which console thread is decided at the start (`xcl_x_thread`): the threads of a group
are interchangeable, but the console forms its waves from consecutive thread indices and
the PC GPU need not – on the RTX 5070 Ti, 1-D groups get consecutive `SV_GroupIndex` values
per wave, an 8 × 8 × 4 group does not. Each hardware wave takes the next 32 console threads
from a groupshared counter as it arrives (a partial wave the last ones), and the program
gets that console thread's ids (v0..v2 in compute, the vertex slot in mesh mode). Console
lane = console thread % 64, half = bit 5, console wave = console thread / 64.

Per-lane code is unchanged. The two halves meet ("exchange") exactly where a value has to be
whole-wave:

| operation | exchange |
|---|---|
| `ballot()` (every mask: `v_cmp*`, `v_cmpx*`, carries, EXEC at entry) | one word: each half gives its 32-bit ballot, both get the 64-bit mask – unless the mask is only read lane by lane (below) |
| `v_readlane`, `v_readfirstlane` (lane index uniform) | one word: each half gives the value of lane `l & 31`, the half that owns `l` wins |
| `ds_bpermute` (any lane) | 32 words: each half gives its lanes |
| DPP16, DPP8, `ds_swizzle`, `v_permlane(x)16` | none: the ISA keeps them inside a 32-lane half (`WaveReadLaneAt(v, l & 31)`) |
| `s_load*` / `s_buffer_load*` with GLC (or DLC) | up to 16 words: half 0's values for both (memory another wave writes may change between the two halves' loads; without GLC the console's scalar cache is not coherent either, so plain loads are read by both halves) |

So both halves hold the full 64-bit EXEC / VCC / masks and identical SGPRs; every branch is
taken the same way by both, and the existing scalar code (`s_bcnt1_i32_b64`, `s_ff1`,
`s_cbranch_execz`, `v_mbcnt_hi`, …) needs no change. The kernel-mode runner stores the
SGPRs of both halves and the tests fail if they ever differ.

### Synchronisation

* **One console wave per group** (≤ 64 threads: kernel mode, most compute, ES+GS):
  `GroupMemoryBarrierWithGroupSync` between writing and reading the slot. All hardware
  waves of the group are the two halves of the same console wave, which execute the same
  sequence of exchanges, so every barrier is reached by both. Portable (no forward-progress
  assumption) and cheap.
* **Several console waves per group** (mesh: 128 threads, compute > 64 threads): a group
  barrier would also wait for the *other* console waves, whose control flow may differ (wave
  index, per-wave data). Each pair instead meets through groupshared arrival counters:
  every lane stores its slot word(s) and fences (`GroupMemoryBarrier`); the wave's lane
  count (`WaveActiveCountBits` – a wave operation, so all its lanes have fenced) goes into
  its half's counter in one atomic; one lane waits until the partner's counter reaches
  `lanes × (k + 1)` (k = exchange number, identical in both halves) and the others take its
  verdict (`WaveReadLaneFirst`); a fence, then the read. Slots are double-buffered by the
  parity of k: the partner can be at most one exchange ahead, so it never overwrites the
  slot being read. The wait is bounded (2^20 polls): if it ever runs out (a desynchronised
  pair – should not happen, see hazards), the wave stops exchanging (`xcl_x_dead`) and
  finishes with half-local values instead of hanging the GPU until a TDR. This relies on
  co-resident waves of one group making progress, which holds on GPUs (all waves of a group
  are resident on one SM / CU) but not necessarily on WARP. `XCL_WAVE64_SPIN=1` forces this
  variant everywhere (the test suite passes with it too).

A console wave whose upper half does not exist (group of 32 or 96 threads) exchanges with
nobody: the missing lanes count as inactive (ballot bits 0), exactly as on the console. The
GS phase of ES+GS runs in the lower half only (≤ 21 primitives), likewise without a partner.

### Memory

The exchange needs groupshared space: per console wave 2 counters + 2 halves × 2 parities ×
slot (1 word, 16 with wide GLC scalar loads, 32 with `ds_bpermute`). Without console LDS it
has arrays of its own; with LDS (`xcl_lds`, already the full 32 KB D3D12 limit) it lives in
the top of the LDS block (as PR #33 did), and the program's LDS accesses are clamped below
it. A program that really uses the whole 32 KB of LDS therefore loses its top dwords on a
32-lane GPU (the console allows 64 KB; XCL already wraps at 32 KB).

## Hazards (and why they are handled)

1. **Divergent halves** – anything that lets the two halves compute different scalar values
   desynchronises the exchanges. Covered: masks and lane reads (exchanged), GLC scalar loads
   (half 0's value), `s_memtime` (constant 0). Not exchanged: non-GLC scalar loads of memory
   that changes during the dispatch (incoherent on the console as well).
2. **Barriers in divergent control flow** – the barrier variant is only used when the group
   is exactly one console wave, whose control flow is identical in both halves; console
   `s_barrier` stays a group barrier (all console waves reach it, as on the console). The GS
   phase (lower half only) has no exchanges and its `s_barrier` is dropped (one wave).
3. **Lane numbering** – handled by `xcl_x_thread` (see above); NVIDIA really does form waves
   of non-consecutive thread indices for 3-D groups (`gpu_split_wave64_waves_diverge` fails
   without the remap).
4. **`WaveReadLaneAt` with a per-lane index** – used inside a half as before.
5. **Forward progress** – only the multi-wave variant spins; see above.

### Masks read lane by lane

A mask whose only readers take the reading lane's own bit – `v_cndmask_b32`'s selector, the
carry-in of `v_add_co_ci` / `v_sub(rev)_co_ci` – needs just that lane's half, so it is not
exchanged: `xcl_x_ballot_local` gives the half's own word and 0 for the other half
(`x64_masks.rs`). Which masks those are is a backward liveness analysis over the SGPRs: a
register is *whole-live* after an instruction when some path may read its whole value (any
SALU / SMEM operand, a VALU scalar operand, `s_cbranch_vcc*`, `src_vccz`, …) before it is
written again; a mask written to registers that are not whole-live stays in its half.

The halves then hold different values in those registers, so the rules follow what the
emitter does, not only the ISA's operand tables:
- a write is an overwrite only when the emitter replaces the whole register whatever it held:
  not `s_cmov*` / `s_cmovk` (conditional), `s_addk` / `s_mulk` / `s_bitset*` (they read the
  old value), `s_movreld` / `s_movrelsd_2` (they write `s[dst + m0]`), nor an SDWA compare
  with SD = 1 (it writes its SDST, VCC keeps its value);
- `s_movrels*` read `s[src + m0]`: every register from `src` up to s123, VCC included;
- SDWA scalar sources (the S0 / S1 bits), SMEM register offsets, `src_vccz` / `src_execz` and
  M0 are reads;
- EXEC is always whole-live; an indirect jump (`s_setpc` / `s_swappc` other than a far jump,
  which is a branch) makes every SGPR whole-live, and so does the end of a kernel-mode program
  (the runner reads the SGPRs back – the halves still end with the same SGPRs);
- every emitted loop can be left where its iteration budget (`LOOP_GUARD`) is checked – at
  its top, and in the `pc` dispatcher form before each of its blocks, to the code after it or
  (dispatcher) the end of the program: what is live there is live before each instruction of
  the loop.

A mask with no destination (VOP3 compare into NULL) or one that is 0 in every lane (the
`false` mask of `v_div_scale`) is not exchanged either. Both halves make the same static
choice, so their exchange counts stay in step. Geometry mode (ES → GS through `s_setpc`)
has no liveness pass: its masks are exchanged as before, except those two kinds (no
destination, 0 in every lane), which are the same in both halves either way.
`XCL_X64_ALL_MASKS=1` exchanges every mask (for comparison).

## Cost

A wave64 program pays one exchange per mask write that is read whole (`v_cmp*` feeding
`s_and_saveexec`, `s_cbranch_vccz`, `s_bcnt1`, …) and per readlane / readfirstlane /
ds_bpermute / coherent scalar load, nothing per ALU, memory or DPP instruction, nor for
masks read lane by lane (selects, 64-bit carry chains). Wave32 programs, VS / PS and everything on GPUs with 64-lane waves are
unchanged (the HLSL of every native translation the test suite makes is byte-identical).

Measured on an RTX 5070 Ti (`bench_split_wave64`, an ignored test): a loop of 8
instructions with one exchange, 2^18 threads × 32 768 iterations, wave32 program (no
exchange) vs the same as wave64 split:

| | 1 exchange / 8 instructions | 2 exchanges / 10 instructions |
|---|---|---|
| one console wave per group (barrier) | 1.3× | 1.6–2.2× |
| four console waves per group (counters) | 4.6–5× | 8.4–9.7× |

This is the worst case – real shaders have far more ALU and memory work per mask. The
counter-paired variant is the expensive one: a waiting wave keeps issuing (no way to park
it in HLSL), taking issue slots from the waves it waits for. The next step would be to skip
the exchange for masks only used lane by lane (`v_cmp` + `v_cndmask` selects; each half's
own word is enough there), which needs a liveness pass over the mask registers.

That pass is now in (see "Masks read lane by lane"). The benchmark now counts the
exchanges a loop iteration makes and times each dispatch with GPU timestamps (median of 9
dispatches in one command list; `runner::dispatch_timed`). Before, it took the wall time of
a whole `runner::dispatch` – pipeline creation, buffers, uploads, submission and the CPU's
wait included – which shrank every ratio towards 1 (the 5070 Ti figures above were taken
that way). On an RTX 3090, three runs with all masks exchanged (`XCL_X64_ALL_MASKS=1`) and
five with the default:

| | mask only | mask + `v_readfirstlane` |
|---|---|---|
| exchanges per iteration (all masks / default) | 1 / 0 | 2 / 1 |
| one console wave per group (barrier) | 1.64–1.68× → 1.04–1.11× | 1.99–2.02× → 1.74–1.81× |
| four console waves per group (counters) | 3.90–4.14× → 1.07–1.16× | 6.45–6.64× → 4.09–4.34× |

(wave32 baseline 11.4–14.1 ms.) The 4–16 % left with no exchange at all is the rest of
the split form (64-bit masks, the threads' start), not measured further. The benchmark's
loop mask is exactly the kind the pass keeps local – it shows the exchange gone and what
one costs, not how often real shaders gain.

What is left is readlane / readfirstlane, `ds_bpermute`, coherent scalar loads and masks
read whole. Not measured on real compute shaders (none on that machine); Minecraft's 273
wave64 programs, translated as compute, have 38 mask writes, all only read lane by lane.

## Relation to earlier work

* **PR #33** (ElHombreAlex, draft): first wave64 → wave32 "rendezvous" through groupshared
  for compute, and the finding that NVIDIA rejects `[WaveSize(64)]`. Its exchange keeps one
  slot per half (the partner can overwrite it with the next exchange before it is read),
  publishes from lane 0 only and times out into stale data; the FSR4 pass it targeted hung
  the GPU. This design keeps its idea of placing the exchange in the LDS tail and fixes the
  protocol (double-buffered slots, counted arrivals, barriers where they are legal).
* **PR #35** (bmjask): a separate per-slot lowering of mesh programs, validation only and
  not connected to translation; its merged parts (`mesh_wave.rs` requirement report,
  `mesh_dispatch.rs` scratch / rendezvous model) are not used by this form.
* **PRs #14 / #39** (split ES+GS / mesh into independent halves and reject cross-half
  operations): replaced by the exchange form, which needs no rejection list.
