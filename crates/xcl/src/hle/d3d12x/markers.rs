//! `XCL_GPU_MARKERS=1`: names the command a queue is stuck on. Around each draw, dispatch, mesh
//! dispatch and ExecuteIndirect of console code, three WriteBufferImmediate words go to the
//! recording's slot in a persistently mapped readback buffer. Before the op (and before what XCL
//! records for it: its bindings, an indirect call's sanitiser, a mesh call's prepass) its number
//! once every command before it has started (MARKER_IN): the GPU has reached the op. After it
//! its number once every command before has started (MARKER_IN: the GPU is past the op's own
//! commands) and once they have all completed (MARKER_OUT). A word carries the slot's
//! generation, so a reused slot is never read as the new recording's. The CPU keeps each
//! recording's ops (kind and pipeline) and the submissions it went into; `dump` (the first GPU
//! wait that times out) reads every submitted recording that has not completed, on every queue:
//! the ops completed, the ops started and not completed, the op the GPU is inside (reached, the
//! commands after it not started) – or that it is between two ops, in commands that carry no
//! marker.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// recordings with a slot at once; a slot still in flight is never given to a new recording
const SLOTS: usize = 1024;
/// a slot's words: started, completed, reached (and one not used)
const SLOT_BYTES: usize = 16;
const OP_BITS: u32 = 20;
const OP_MASK: u32 = (1 << OP_BITS) - 1;

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_GPU_MARKERS").is_ok_and(|v| v.trim() == "1"))
}

#[derive(Clone, Copy)]
struct Op {
    title_pso: usize,
    kind: &'static str,
}

struct Recording {
    list: usize,
    generation: u32,
    ops: Vec<Op>,
    /// (queue, submission number, the queue's last sync-graph seq then: `list_x::queue_seq_now`)
    submissions: Vec<(usize, u64, u64)>,
}

struct Markers {
    va: u64,
    cpu: usize,
    slots: Vec<Option<Recording>>,
    /// the slot of each list's recording, until the list is reset
    current: HashMap<usize, usize>,
    next: usize,
    submissions: u64,
}

static MARKERS: Mutex<Option<Markers>> = Mutex::new(None);

/// A committed buffer of `bytes` in a readback heap, mapped for good: (resource, GPU address, CPU address).
pub(super) unsafe fn readback_buffer(bytes: u64) -> Option<(usize, u64, usize)> {
    let raw = super::real::device_raw();
    let dev = ID3D12Device::from_raw_borrowed(&raw)?;
    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    super::title_textures::own_work(|| dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r)).ok()?;
    let r = r?;
    let mut p: *mut c_void = std::ptr::null_mut();
    r.Map(0, None, Some(&mut p)).ok()?;
    std::ptr::write_bytes(p as *mut u8, 0, bytes as usize);
    let va = r.GetGPUVirtualAddress();
    Some((r.into_raw() as usize, va, p as usize))
}

fn markers() -> std::sync::MutexGuard<'static, Option<Markers>> {
    let mut g = MARKERS.lock().unwrap();
    if g.is_none() {
        if let Some((_, va, cpu)) = unsafe { readback_buffer((SLOTS * SLOT_BYTES) as u64) } {
            crate::xlog_info!("D3D12.X: GPU markers on (XCL_GPU_MARKERS): {SLOTS} recording slots");
            *g = Some(Markers { va, cpu, slots: (0..SLOTS).map(|_| None).collect(), current: HashMap::new(), next: 0, submissions: 0 });
        }
    }
    g
}

impl Markers {
    /// The op counts of `slot` as the GPU wrote them – ops started, ops completed, the op
    /// reached –, each None when written by another generation (nothing yet).
    fn words(&self, slot: usize, generation: u32) -> [Option<u32>; 3] {
        let read = |k: usize| unsafe { ((self.cpu + slot * SLOT_BYTES + 4 * k) as *const u32).read_volatile() };
        let ours = |w: u32| (w >> OP_BITS == generation).then_some(w & OP_MASK);
        [ours(read(0)), ours(read(1)), ours(read(2))]
    }

    fn complete(&self, slot: usize) -> bool {
        match &self.slots[slot] {
            None => true,
            Some(r) if self.current.get(&r.list) == Some(&slot) => false,
            // (no ops: an op was about to be recorded and was skipped – nothing to wait for)
            Some(r) => r.submissions.is_empty() || r.ops.is_empty() || self.words(slot, r.generation)[1] == Some((r.ops.len() as u32).min(OP_MASK)),
        }
    }

    fn slot_of(&mut self, list: usize) -> Option<usize> {
        if let Some(&s) = self.current.get(&list) {
            return Some(s);
        }
        let s = (0..SLOTS).map(|k| (self.next + k) % SLOTS).find(|&s| self.complete(s))?;
        self.next = (s + 1) % SLOTS;
        let generation = self.slots[s].as_ref().map_or(1, |r| r.generation % 0xfff + 1);
        self.slots[s] = Some(Recording { list, generation, ops: Vec::new(), submissions: Vec::new() });
        self.current.insert(list, s);
        Some(s)
    }
}

/// After a draw, dispatch or indirect call (`kind`) with the title's pipeline `title_pso` was
/// recorded on the PC list `list`.
pub(super) unsafe fn after_op(list: *mut c_void, kind: &'static str, title_pso: usize) {
    if enabled() {
        write(list, kind, title_pso);
    }
}

/// Before a draw, dispatch or indirect call is recorded on the PC list `list`, and before what
/// XCL records for it (its bindings, an indirect call's sanitiser, a mesh call's prepass): the
/// GPU writes the op's number once every command before it has started – it has reached the op.
/// An op skipped after this leaves its number for the next one.
pub(super) unsafe fn before_op(list: *mut c_void) {
    if enabled() {
        write_before(list);
    }
}

unsafe fn write_before(list: *mut c_void) {
    let Some(l) = ID3D12GraphicsCommandList2::from_raw_borrowed(&list) else { return };
    let (va, value) = {
        let mut g = markers();
        let Some(m) = g.as_mut() else { return };
        let Some(s) = m.slot_of(list as usize) else { return };
        let r = m.slots[s].as_ref().unwrap();
        (m.va + (s * SLOT_BYTES) as u64 + 8, r.generation << OP_BITS | (r.ops.len() as u32 + 1).min(OP_MASK))
    };
    let params = [D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: va, Value: value }];
    let modes = [D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN];
    l.WriteBufferImmediate(1, params.as_ptr(), Some(modes.as_ptr()));
}

unsafe fn write(list: *mut c_void, kind: &'static str, title_pso: usize) {
    let Some(l) = ID3D12GraphicsCommandList2::from_raw_borrowed(&list) else { return };
    let (va, value) = {
        let mut g = markers();
        let Some(m) = g.as_mut() else { return };
        let Some(s) = m.slot_of(list as usize) else { return };
        let r = m.slots[s].as_mut().unwrap();
        r.ops.push(Op { title_pso, kind });
        (m.va + (s * SLOT_BYTES) as u64, r.generation << OP_BITS | (r.ops.len() as u32).min(OP_MASK))
    };
    let params = [D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: va, Value: value }, D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: va + 4, Value: value }];
    let modes = [D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT];
    l.WriteBufferImmediate(2, params.as_ptr(), Some(modes.as_ptr()));
}

/// Where the next op of `list` goes: (slot, generation, ops so far), for naming its submission later.
pub(super) fn position(list: usize) -> Option<(usize, u32, usize)> {
    let g = MARKERS.lock().unwrap();
    let m = g.as_ref()?;
    let s = *m.current.get(&list)?;
    m.slots[s].as_ref().map(|r| (s, r.generation, r.ops.len()))
}

/// The submissions of the recording at `slot` of `generation`, as text.
pub(super) fn submissions_of(slot: usize, generation: u32) -> String {
    let g = MARKERS.lock().unwrap();
    match g.as_ref().and_then(|m| m.slots[slot].as_ref()).filter(|r| r.generation == generation) {
        Some(r) if !r.submissions.is_empty() => r.submissions.iter().map(|&(q, n, seq)| format!("q{} submission {n} (after its seq {seq})", super::list_x::queue_id(q))).collect::<Vec<_>>().join(", "),
        Some(_) => "not submitted yet".into(),
        None => "its recording slot was reused".into(),
    }
}

/// `list` was reset, split or released: its next op starts a new recording.
pub(super) fn recording_ended(list: usize) {
    if let Some(m) = MARKERS.lock().unwrap().as_mut() {
        m.current.remove(&list);
    }
}

/// The PC lists `lists` went to `queue` in one ExecuteCommandLists.
pub(super) fn submitted(queue: usize, lists: &[*mut c_void]) {
    if MARKERS.lock().unwrap().is_none() {
        return;
    }
    let seq = super::list_x::queue_seq_now(queue);
    let mut g = MARKERS.lock().unwrap();
    let Some(m) = g.as_mut() else { return };
    m.submissions += 1;
    let n = m.submissions;
    for &l in lists {
        if let Some(&s) = m.current.get(&(l as usize)) {
            if let Some(r) = m.slots[s].as_mut() {
                r.submissions.push((super::single::queue_now().unwrap_or(queue), n, seq));
            }
        }
    }
}

/// the latest wait or signal XCL itself put on each queue, by what it is for: (fence, value, when)
static QUEUE_FENCES: Mutex<Option<HashMap<(usize, &'static str), (usize, u64, std::time::Instant)>>> = Mutex::new(None);

/// XCL made `queue` wait on the GPU for `fence` to reach `value`, or signal it (`what` it is
/// for): the sync graph does not see these, so the snapshot lists them.
pub(super) fn queue_fence(queue: usize, what: &'static str, fence: &ID3D12Fence, value: u64) {
    if enabled() {
        QUEUE_FENCES.lock().unwrap().get_or_insert_with(HashMap::new).insert((queue, what), (fence.as_raw() as usize, value, std::time::Instant::now()));
    }
}

/// The state of the GPU at the first GPU wait that timed out or at the device's removal,
/// whichever comes first (`why`): the open recordings (`dump`), XCL's own fences on each queue
/// and the indirect captures.
pub(super) fn snapshot(why: &str) {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !enabled() || DONE.swap(true, std::sync::atomic::Ordering::AcqRel) {
        return;
    }
    crate::xlog_error!("GPU snapshot ({why}):");
    dump();
    let waits: Vec<String> = QUEUE_FENCES.lock().unwrap().as_ref().map_or_else(Vec::new, |m| {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by_key(|((q, what), _)| (super::list_x::queue_id(*q), *what));
        v.into_iter()
            .map(|(&(q, what), &(f, value, at))| {
                let p = f as *mut c_void;
                let done = unsafe { ID3D12Fence::from_raw_borrowed(&p).map(|f| f.GetCompletedValue()) };
                let state = match done {
                    Some(c) if c >= value => format!("met (completed {c:#x})"),
                    Some(c) => format!("NOT MET (completed {c:#x})"),
                    None => "unknown".into(),
                };
                format!("q{} {what}: fence {f:#x} >= {value:#x}, {state}, issued {} ms ago", super::list_x::queue_id(q), at.elapsed().as_millis())
            })
            .collect()
    });
    crate::xlog_error!("GPU snapshot: XCL's own fences on each queue ({}): {}", waits.len(), if waits.is_empty() { "none".into() } else { waits.join("; ") });
    super::indirect::log_captures(true);
    crate::xlog_error!("GPU snapshot: {}", super::bind::mesh_counts());
    for line in super::bind::trips(24) {
        crate::xlog_error!("GPU snapshot: {line}");
    }
}

/// the console shaders of translated graphics pipelines made on the title's own thread (the
/// background builds keep theirs: `pso_async::what`), by PC pipeline
static GRAPHICS: Mutex<Option<HashMap<usize, (u64, u64)>>> = Mutex::new(None);

/// The translated graphics pipeline `pso` was made of the console programs `vs` and `ps` (0:
/// none), by their hashes.
pub(super) fn note_graphics(pso: usize, vs: u64, ps: u64) {
    if enabled() {
        GRAPHICS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, (vs, ps));
    }
}

/// The pipeline state `pso` is gone: a later one at its address is not made of its shaders.
pub(super) fn pipeline_gone(pso: usize) {
    if let Some(m) = GRAPHICS.lock().unwrap().as_mut() {
        m.remove(&pso);
    }
}

/// A title pipeline as text: its PC pipeline, shaders and name.
pub(super) fn pipeline(title_pso: usize) -> String {
    let real = super::pso_async::resolve(title_pso);
    let noted = || GRAPHICS.lock().unwrap().as_ref().and_then(|m| m.get(&real).copied()).map(|(vs, ps)| format!("VS {vs:016x} / PS {ps:016x}"));
    let what = super::pso_async::what(title_pso).or_else(noted).unwrap_or_else(|| "shaders unknown".into());
    let name = super::real::pso_name(title_pso).map(|n| format!(", \"{n}\"")).unwrap_or_default();
    format!("pipeline {title_pso:#x} → {real:#x} ({what}{name})")
}

/// How far the GPU is in a recording, from its words (`Markers::words`): (ops completed, ops
/// started, the op reached), each at least the one before it.
fn progress(words: [Option<u32>; 3]) -> (u32, u32, u32) {
    let done = words[1].unwrap_or(0);
    let started = words[0].unwrap_or(0).max(done);
    (done, started, words[2].unwrap_or(0).max(started))
}

/// Where the GPU is in a recording of `total` ops (`progress`), in words.
fn state(done: u32, started: u32, reached: u32, total: u32) -> String {
    let mut s = Vec::new();
    if started > done {
        s.push(if started == done + 1 { format!("op {started} STARTED AND NOT COMPLETED") } else { format!("ops {}-{started} STARTED AND NOT COMPLETED", done + 1) });
    }
    if reached > started {
        s.push(format!("the GPU IS INSIDE op {reached} (reached, the commands after it not started)"));
    } else if reached == 0 {
        s.push("op 1 not reached (the GPU has not begun this recording)".into());
    } else if started < total {
        s.push(format!("op {} not reached{}", started + 1, if started == done { format!(" (the GPU is between op {done} and op {}, in commands that carry no marker, or its queue is not running)", done + 1) } else { String::new() }));
    }
    s.join("; ")
}

/// A recording the GPU has not completed, as `frontier` finds it.
struct Begun {
    queue: usize,
    submission: u64,
    slot: usize,
    generation: u32,
    /// ops completed, started, the op reached (`progress`)
    at: (u32, u32, u32),
    total: u32,
    /// the op the GPU is in, or would begin next
    op: String,
}

/// `frontier` in words: a line for each recording the GPU has begun – its ops completed, with
/// those completed since `was` (slot → generation, ops completed, ops in all), and where it is
/// now –, the recordings completed since, and how many the GPU has not begun.
fn frontier_lines(open: &[Begun], was: &HashMap<usize, (u32, u32, u32)>) -> Vec<String> {
    let mut lines: Vec<String> = open
        .iter()
        .filter(|b| b.at.2 > 0)
        .map(|b| {
            let (done, started, reached) = b.at;
            let since = match was.get(&b.slot) {
                Some(&(generation, d, _)) if generation == b.generation && done > d => format!(" (op{} completed since)", if done == d + 1 { format!(" {done}") } else { format!("s {}-{done}", d + 1) }),
                Some(&(generation, _, _)) if generation == b.generation => String::new(),
                _ => " (new)".into(),
            };
            format!("q{} submission {} slot {}: {done} of {} ops completed{since}; {}; {}", b.queue, b.submission, b.slot, b.total, state(done, started, reached, b.total), b.op)
        })
        .collect();
    let mut gone: Vec<(usize, String)> = was
        .iter()
        .filter(|(slot, (generation, _, _))| !open.iter().any(|b| b.slot == **slot && b.generation == *generation))
        .map(|(slot, (_, done, total))| (*slot, format!("slot {slot} (its ops {}-{total})", done + 1)))
        .collect();
    gone.sort();
    if !gone.is_empty() {
        lines.push(format!("recordings completed since: {}", gone.into_iter().map(|g| g.1).collect::<Vec<_>>().join(", ")));
    }
    lines.push(format!("{} submitted recordings not begun", open.iter().filter(|b| b.at.2 == 0).count()));
    lines
}

/// (`list_x::watch_gpu_waits`) Where the GPU is in the recordings it has not completed, and what
/// it completed since the call before (`frontier_lines`): called every second while a GPU wait
/// is watched, the log's times say when each op and recording was done.
pub(super) fn frontier() -> Vec<String> {
    static BEFORE: Mutex<Option<HashMap<usize, (u32, u32, u32)>>> = Mutex::new(None);
    let g = MARKERS.lock().unwrap();
    let Some(m) = g.as_ref() else { return vec!["no GPU markers (XCL_GPU_MARKERS=1)".into()] };
    let mut open = Vec::new();
    let mut selected = Vec::new();
    for (slot, r) in m.slots.iter().enumerate() {
        let Some(r) = r else { continue };
        let Some(&(q, submission, _)) = r.submissions.last() else { continue };
        let total = (r.ops.len() as u32).min(OP_MASK);
        let words = m.words(slot, r.generation);
        if words[1] == Some(total) || total == 0 {
            continue;
        }
        let at = progress(words.map(|word| word.map(|n| n.min(total))));
        let k = at.2.max(at.0 + 1);
        if at.2 > at.0 && r.ops.get(k as usize - 1).is_some_and(|o| {
            matches!(o.kind, "dispatch" | "ExecuteIndirect(dispatch)")
                && super::dispatch_timing::shader(super::pso_async::resolve(o.title_pso)).is_some()
        }) {
            selected.push((slot, r.generation, at.0, k));
        }
        let op = r.ops.get(k as usize - 1).map_or(format!("op {k} (none)"), |o| format!("op {k} {} with {}", o.kind, pipeline(o.title_pso)));
        open.push(Begun { queue: super::list_x::queue_id(q), submission, slot, generation: r.generation, at, total, op });
    }
    drop(g);
    open.sort_by_key(|b| (b.queue, b.submission, b.slot));
    let mut before = BEFORE.lock().unwrap();
    let lines = frontier_lines(&open, before.as_ref().unwrap_or(&HashMap::new()));
    let newly_active: Vec<_> = selected.into_iter().filter(|&(slot, generation, done, _)| {
        before.as_ref().is_some_and(|b| b.get(&slot).is_none_or(|&(previous_generation, previous_done, _)| {
            previous_generation != generation || previous_done != done
        }))
    }).collect();
    *before = Some(open.iter().map(|b| (b.slot, (b.generation, b.at.0, b.total))).collect());
    drop(before);
    for (slot, generation, _, op) in newly_active {
        for line in super::bind::describe_dispatch(slot, generation, op as usize - 1) {
            crate::xlog_info!("GPU dispatch became active: slot {slot}/{generation} op {op}: {line}");
        }
    }
    lines
}

/// The first GPU wait that timed out: every submitted recording that has not completed, by
/// queue and submission – what it completed, started and reached (`state`), then a line for
/// each queue and one for the ops in flight on all of them, then what the ops the GPU is at
/// were bound to and the guard tripwire's counters of every op of an open recording.
pub(super) fn dump() -> Vec<String> {
    let g = MARKERS.lock().unwrap();
    let Some(m) = g.as_ref() else {
        if enabled() {
            crate::xlog_error!("GPU markers: none recorded");
        }
        return Vec::new();
    };
    struct Open {
        queue: usize,
        submission: u64,
        slot: usize,
        generation: u32,
        at: (u32, u32, u32),
        line: String,
    }
    let mut open: Vec<Open> = Vec::new();
    for (s, r) in m.slots.iter().enumerate() {
        let Some(r) = r else { continue };
        let Some(&(q, n, seq)) = r.submissions.last() else { continue };
        let total = (r.ops.len() as u32).min(OP_MASK);
        let words = m.words(s, r.generation);
        if words[1] == Some(total) || total == 0 {
            continue;
        }
        let op = |k: u32| match k.checked_sub(1).and_then(|i| r.ops.get(i as usize)) {
            Some(o) => format!("op {k} {} with {}", o.kind, pipeline(o.title_pso)),
            None => format!("op {k} (none)"),
        };
        let (done, started, reached) = progress(words.map(|word| word.map(|n| n.min(total))));
        // the ops in flight, or the one the GPU would begin next
        let live: Vec<String> = (done + 1..=reached.max(done + 1)).take(4).map(op).collect();
        open.push(Open {
            queue: q,
            submission: n,
            slot: s,
            generation: r.generation,
            at: (done, started, reached),
            line: format!(
                "q{} submission {n} (after its seq {seq}), list {:#x} (slot {s}): {done} of {total} ops completed{}; {}; {}; next {}; last completed {}",
                super::list_x::queue_id(q),
                r.list,
                if words == [None; 3] { " (no marker written yet)" } else { "" },
                state(done, started, reached, total),
                live.join(", "),
                op(reached.max(done + 1) + 1),
                if done == 0 { "none".into() } else { op(done) }
            ),
        });
    }
    open.sort_by_key(|o| (super::list_x::queue_id(o.queue), o.submission, o.slot));
    crate::xlog_error!("GPU markers: {} submitted recordings not completed ({} submissions so far)", open.len(), m.submissions);
    for o in &open {
        crate::xlog_error!("GPU markers: {}", o.line);
    }
    // each queue: the recordings the GPU has been in, and the ops in flight on all queues
    let short = |o: &Open| {
        let r = m.slots[o.slot].as_ref().unwrap();
        let (done, started, reached) = o.at;
        let ops: Vec<String> = (done + 1..=reached).take(4).filter_map(|k| r.ops.get(k as usize - 1).map(|p| format!("op {k} {} with {}", p.kind, pipeline(p.title_pso)))).collect();
        format!("slot {} (submission {}): {}{}", o.slot, o.submission, state(done, started, reached, r.ops.len() as u32), if ops.is_empty() { String::new() } else { format!(": {}", ops.join(", ")) })
    };
    let mut queues: Vec<usize> = open.iter().map(|o| o.queue).collect();
    queues.dedup();
    for q in queues {
        let of: Vec<&Open> = open.iter().filter(|o| o.queue == q).collect();
        let begun: Vec<String> = of.iter().filter(|o| o.at.2 > 0).map(|o| short(o)).collect();
        crate::xlog_error!(
            "GPU markers: q{}: {} open recordings in submissions {}-{}; {}",
            super::list_x::queue_id(q),
            of.len(),
            of[0].submission,
            of[of.len() - 1].submission,
            if begun.is_empty() { "the GPU has begun none of them".into() } else { format!("begun: {}", begun.join(" | ")) }
        );
    }
    let flying: Vec<String> = open.iter().filter(|o| o.at.2 > o.at.0).map(|o| format!("q{} {}", super::list_x::queue_id(o.queue), short(o))).collect();
    crate::xlog_error!(
        "GPU markers: ops reached and not completed, all queues ({}): {}",
        flying.len(),
        if flying.is_empty() { "none – no queue is inside a marked op".into() } else { flying.join(" | ") }
    );
    // what the ops the GPU is at were bound to: from the first not completed to the one after
    // the last reached (the GPU may be at its bindings), in every recording it has begun
    let mut described: Vec<(usize, usize)> = Vec::new();
    for o in open.iter().filter(|o| o.at.2 > 0) {
        let r = m.slots[o.slot].as_ref().unwrap();
        for op in o.at.0 as usize..r.ops.len().min(o.at.2 as usize + 1) {
            let p = r.ops[op];
            let dispatch = p.kind == "dispatch" || p.kind == "ExecuteIndirect(dispatch)";
            if dispatch {
                for line in super::bind::describe_dispatch(o.slot, o.generation, op) {
                    crate::xlog_error!("GPU markers: slot {} op {}: {line}", o.slot, op + 1);
                }
            } else {
                let trips = super::bind::trips_of_draws(p.title_pso).unwrap_or_else(|| "not watched".into());
                crate::xlog_error!("GPU markers: slot {} op {}: {} with {}; guard tripwire now, in all draws of the pipeline so far: {trips}", o.slot, op + 1, p.kind, pipeline(p.title_pso));
            }
            // what the first of them were given, in full: every root argument and what is
            // behind it now (`hang`)
            if described.len() < super::hang::OPS {
                for line in super::hang::describe(o.slot, o.generation, op, dispatch) {
                    crate::xlog_error!("GPU markers: slot {} op {}: {line}", o.slot, op + 1);
                }
            }
            described.push((o.slot, op));
        }
    }
    // the guard tripwire in every op of the open recordings, begun or not
    let slots: Vec<(usize, u32)> = open.iter().map(|o| (o.slot, o.generation)).collect();
    let tripped = super::bind::trips_in(&slots);
    crate::xlog_error!("GPU markers: guard tripwire now: events in {} dispatches of the {} open recordings", tripped.len(), open.len());
    for (slot, op, lanes) in &tripped {
        let p = m.slots[*slot].as_ref().and_then(|r| r.ops.get(*op)).map_or("an op not recorded".into(), |p| format!("{} with {}", p.kind, pipeline(p.title_pso)));
        crate::xlog_error!("GPU markers: guard tripwire now: slot {slot} op {}: {lanes} in {p}", op + 1);
        if !described.contains(&(*slot, *op)) {
            let generation = slots.iter().find(|s| s.0 == *slot).map_or(0, |s| s.1);
            for line in super::bind::describe_dispatch(*slot, generation, *op) {
                crate::xlog_error!("GPU markers: slot {slot} op {}: {line}", op + 1);
            }
        }
    }
    open.into_iter().map(|o| o.line).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three words of a recording say where the GPU is: inside an op it reached whose
    /// following commands have not started, past ops that started and did not complete, between
    /// two ops (in commands that carry no marker), or not in the recording at all.
    #[test]
    fn the_words_tell_an_op_in_flight_from_the_gap_between_two_ops() {
        let at = |started: Option<u32>, completed: Option<u32>, reached: Option<u32>| {
            let (d, s, r) = progress([started, completed, reached]);
            state(d, s, r, 42)
        };
        // reached op 28, its own commands not passed: the GPU is in it (or in its bindings)
        let inside = at(Some(27), Some(27), Some(28));
        assert!(inside.contains("the GPU IS INSIDE op 28") && !inside.contains("STARTED AND NOT COMPLETED"), "{inside}");
        // past op 28's commands, which have not completed
        let running = at(Some(28), Some(27), Some(28));
        assert!(running.contains("op 28 STARTED AND NOT COMPLETED") && running.contains("op 29 not reached") && !running.contains("INSIDE"), "{running}");
        // several ops in the pipeline, and the next one reached
        let many = at(Some(30), Some(27), Some(31));
        assert!(many.contains("ops 28-30 STARTED AND NOT COMPLETED") && many.contains("the GPU IS INSIDE op 31"), "{many}");
        // op 27 complete and op 28 not reached: between the two
        let gap = at(Some(27), Some(27), Some(27));
        assert!(gap.contains("op 28 not reached (the GPU is between op 27 and op 28"), "{gap}");
        // nothing written by this generation
        assert!(at(None, None, None).contains("op 1 not reached (the GPU has not begun this recording)"));
        // the first op reached and nothing after it: inside op 1, not "0 done"
        let first = at(None, None, Some(1));
        assert!(first.contains("the GPU IS INSIDE op 1"), "{first}");
        // a site without a marker before its op: the later words carry the earlier ones
        assert_eq!(progress([Some(5), Some(3), Some(2)]), (3, 5, 5));
        // all done but the last marker's completion: nothing more to reach
        let last = at(Some(42), Some(41), Some(42));
        assert!(last.contains("op 42 STARTED AND NOT COMPLETED") && !last.contains("not reached"), "{last}");
    }

    /// The frontier of a watched GPU wait names the ops a begun recording completed since the
    /// line before, the recordings completed since, and counts those not begun.
    #[test]
    fn the_frontier_says_what_the_gpu_completed_since_the_line_before() {
        let begun = |slot: usize, generation: u32, at: (u32, u32, u32), total: u32| Begun { queue: 1, submission: 20860, slot, generation, at, total, op: format!("op {} dispatch", at.2.max(at.0 + 1)) };
        // the first line: the recording the GPU is in, and one it has not begun
        let open = [begun(677, 3, (6, 6, 7), 18), begun(679, 1, (0, 0, 0), 294)];
        let first = frontier_lines(&open, &HashMap::new());
        assert_eq!(first.len(), 2, "{first:?}");
        assert!(first[0].starts_with("q1 submission 20860 slot 677: 6 of 18 ops completed (new); the GPU IS INSIDE op 7") && first[0].ends_with("op 7 dispatch"), "{}", first[0]);
        assert_eq!(first[1], "1 submitted recordings not begun");
        let was: HashMap<usize, (u32, u32, u32)> = open.iter().map(|b| (b.slot, (b.generation, b.at.0, b.total))).collect();
        // nothing moved: the same place, nothing completed
        let still = frontier_lines(&open, &was);
        assert!(still[0].starts_with("q1 submission 20860 slot 677: 6 of 18 ops completed; the GPU IS INSIDE op 7"), "{}", still[0]);
        // ops 7 to 9 done; then the recording done, the next one begun and its slot's old tenant not taken for it
        let moved = frontier_lines(&[begun(677, 3, (9, 9, 10), 18), begun(679, 1, (0, 0, 0), 294)], &was);
        assert!(moved[0].contains("9 of 18 ops completed (ops 7-9 completed since)"), "{}", moved[0]);
        let one = frontier_lines(&[begun(677, 3, (7, 7, 7), 18)], &was);
        assert!(one[0].contains("7 of 18 ops completed (op 7 completed since)"), "{}", one[0]);
        let next = frontier_lines(&[begun(679, 1, (0, 0, 1), 294), begun(677, 4, (0, 0, 0), 5)], &was);
        assert!(next[0].starts_with("q1 submission 20860 slot 679: 0 of 294 ops completed; the GPU IS INSIDE op 1"), "{}", next[0]);
        assert_eq!(next[1], "recordings completed since: slot 677 (its ops 7-18)");
        assert_eq!(next[2], "1 submitted recordings not begun");
    }

    /// Markers after each op of a recording that ran show all of them started and completed, so
    /// the dump leaves it out; a recording submitted that the GPU never ran is named, with no
    /// marker written yet and its first op as the one the queue is on.
    #[test]
    fn the_dump_names_only_recordings_the_gpu_did_not_finish() {
        assert_eq!(super::super::backend(), super::super::Backend::Real);
        unsafe {
            let raw = super::super::real::device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw).expect("D3D12 device");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).expect("queue");
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("allocator");
            let ran: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            let alloc2: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("allocator");
            let never: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc2, None).expect("list");
            for k in 0..3 {
                write_before(ran.as_raw());
                write(ran.as_raw(), "dispatch", 0x7E57_8800 + k);
            }
            write_before(never.as_raw());
            write(never.as_raw(), "draw", 0x7E57_8810);
            ran.Close().expect("close");
            never.Close().expect("close");
            submitted(queue.as_raw() as usize, &[ran.as_raw(), never.as_raw()]);
            queue.ExecuteCommandLists(&[Some(ran.cast::<ID3D12CommandList>().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            queue.Signal(&fence, 1).expect("signal");
            let t0 = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 {
                assert!(t0.elapsed() < std::time::Duration::from_secs(10), "the list did not finish");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let (s, g) = position(ran.as_raw() as usize).map(|p| (p.0, p.1)).expect("a recording");
            assert_eq!(MARKERS.lock().unwrap().as_ref().unwrap().words(s, g), [Some(3), Some(3), Some(3)], "all three ops reached, started and completed");
            let open: Vec<String> = dump().into_iter().filter(|l| l.contains(&format!("{:#x}", never.as_raw() as usize)) || l.contains(&format!("{:#x}", ran.as_raw() as usize))).collect();
            assert_eq!(open.len(), 1, "{open:?}");
            assert!(open[0].contains("0 of 1 ops completed (no marker written yet)") && open[0].contains("op 1 not reached"), "{}", open[0]);
            recording_ended(ran.as_raw() as usize);
            recording_ended(never.as_raw() as usize);
        }
    }
}
