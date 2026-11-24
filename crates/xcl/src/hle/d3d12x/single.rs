//! `XCL_SINGLE_QUEUE=1` (a test of what a GPU hang needs): the work of every title queue runs on
//! one PC queue, in the order the title's threads hand it over.
//!
//! The first queue a title makes is the PC queue of them all (`host_of`); the others stay the
//! title's objects and nothing runs on them. Lists and allocators of every type are made
//! direct ones (`list_type`): a direct queue runs nothing else.
//!
//! A wait for a value another queue writes (`Wait64BitValueX` after a `Write64BitValue*`, a
//! queue's `Wait` for a fence) needs no GPU wait once that write or Signal has been handed to
//! the PC queue: it runs before whatever is handed over later. Until then the waiting title
//! queue's work is held here in its order (`Queue::held`) and goes on when the write or Signal
//! is handed over. Nothing is released by time: a hold that cannot end (`cycle`) or lasts
//! `HOLD_LIMIT`, and a PC queue that does not finish its work within `GPU_LIMIT`, stop the
//! title after a snapshot (`list_x::hard_stop`).

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

const HOLD_LIMIT: Duration = Duration::from_secs(10);
const GPU_LIMIT: Duration = Duration::from_secs(8);

/// What held work waits for: a value written at a console address by a list, or a fence's
/// value signalled by a queue (or on the CPU).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Need {
    Cell { va: u64, value: u64 },
    Fence { fence: usize, value: u64 },
}

/// What a title queue was given, in its order.
enum Op {
    /// PC lists to run
    Lists(Vec<usize>),
    /// the queue's Signal of a fence (`call` makes it on the PC queue)
    Signal { fence: usize, value: u64, call: Box<dyn FnOnce() + Send> },
    /// the queue's Wait for a fence
    Wait { fence: usize, value: u64 },
}

#[derive(Default)]
struct Queue {
    /// work not handed to the PC queue yet; with `need`, the first of it waits for that
    held: VecDeque<Op>,
    /// a thread hands this queue's work over now (the lists it is at): what comes meanwhile
    /// goes behind it
    busy: bool,
    running: Vec<usize>,
    need: Option<(Need, Instant)>,
}

#[derive(Default)]
struct State {
    host: usize,
    /// title queues in the order they were made
    order: Vec<usize>,
    queues: HashMap<usize, Queue>,
    /// the highest value a queue's Signal handed over for each fence
    signalled: HashMap<usize, u64>,
    /// the title queue each PC list was last given to
    list_queue: HashMap<usize, usize>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static CHANGED: Condvar = Condvar::new();
/// the PC queue of all title queues (0: not in this mode)
static HOST: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// the title queue whose work this thread hands to the PC queue now
    static RUNNING: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SINGLE_QUEUE").is_ok_and(|v| v.trim() == "1"))
}

/// The PC type of a list or allocator a title asks for with PC type `ty`.
pub(super) fn list_type(ty: usize) -> usize {
    if enabled() {
        D3D12_COMMAND_LIST_TYPE_DIRECT.0 as usize
    } else {
        ty
    }
}

/// The title made `queue` (of title type `ty`): the first is the PC queue of them all.
pub(super) fn adopt(queue: usize, ty: i32) {
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    if s.queues.contains_key(&queue) {
        return;
    }
    let first = s.host == 0;
    if first {
        s.host = queue;
        HOST.store(queue, Ordering::Release);
    }
    s.order.push(queue);
    s.queues.insert(queue, Queue::default());
    let host = s.host;
    drop(g);
    crate::xlog_info!(
        "single queue (XCL_SINGLE_QUEUE=1): title queue q{} (type {ty}) {}",
        super::list_x::queue_id(queue),
        if first { "is the PC queue all title queues' work runs on".to_string() } else { format!("runs its work on q{}'s PC queue", super::list_x::queue_id(host)) }
    );
    if first {
        watch(queue);
    }
}

/// The PC queue the work of title queue `queue` runs on, in this mode.
pub(super) fn host_of(queue: *mut c_void) -> Option<*mut c_void> {
    let host = HOST.load(Ordering::Acquire);
    if host == 0 {
        return None;
    }
    (host == queue as usize || STATE.lock().unwrap().as_ref().is_some_and(|s| s.queues.contains_key(&(queue as usize)))).then_some(host as *mut c_void)
}

/// This thread hands a title queue's work to the PC queue now: the queue methods it calls
/// are that work, not the title's calls.
pub(super) fn running() -> bool {
    RUNNING.with(Cell::get) != 0
}

/// The title queue whose work this thread hands over now.
pub(super) fn queue_now() -> Option<usize> {
    Some(RUNNING.with(Cell::get)).filter(|&q| q != 0)
}

/// The title gave `queue` these PC lists (`ExecuteCommandLists`).
pub(super) unsafe fn execute(queue: usize, lists: Vec<usize>) {
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        for &l in &lists {
            s.list_queue.insert(l, queue);
        }
    }
    give(queue, Op::Lists(lists));
}

/// The title's Signal of `fence` to `value` on `queue`; `call` makes it on the PC queue.
pub(super) unsafe fn signal(queue: usize, fence: usize, value: u64, call: impl FnOnce() + Send + 'static) {
    give(queue, Op::Signal { fence, value, call: Box::new(call) });
}

/// The title's Wait for `fence` to reach `value` on `queue`.
pub(super) unsafe fn wait(queue: usize, fence: usize, value: u64) {
    give(queue, Op::Wait { fence, value });
}

/// A fence was signalled on the CPU: held work may go on.
pub(super) unsafe fn poke() {
    if HOST.load(Ordering::Acquire) != 0 && !running() {
        pump();
    }
}

/// A call on title queue `queue` that cannot be held (`what`: a present, tile mappings) runs
/// on the PC queue after all the queue was given before; the PC queue is returned.
pub(super) unsafe fn host_after_held(queue: *mut c_void, what: &str) -> *mut c_void {
    let Some(host) = host_of(queue) else { return queue };
    if !running() {
        let q = queue as usize;
        wait_until(what, |s| s.queues.get(&q).is_none_or(|x| x.held.is_empty() && !x.busy));
    }
    host
}

/// KickoffX of `list` on `queue` (the queue it last ran on, or one of its type): after all
/// the list's title queue was given before, on the PC queue.
pub(super) unsafe fn kickoff_queue(list: usize, queue: *mut c_void) -> *mut c_void {
    if HOST.load(Ordering::Acquire) == 0 {
        return queue;
    }
    let of_list = STATE.lock().unwrap().as_ref().and_then(|s| s.list_queue.get(&list).copied());
    host_after_held(of_list.map_or(queue, |q| q as *mut c_void), "KickoffX")
}

/// The title resets these PC lists: not before the PC queue has been given them.
pub(super) unsafe fn before_list_reset(lists: &[usize]) {
    if HOST.load(Ordering::Acquire) == 0 || running() {
        return;
    }
    wait_until("a command list's Reset", |s| {
        !s.queues.values().any(|q| q.running.iter().any(|l| lists.contains(l)) || q.held.iter().any(|op| matches!(op, Op::Lists(held) if held.iter().any(|l| lists.contains(l)))))
    });
}

unsafe fn wait_until(what: &str, done: impl Fn(&State) -> bool) {
    pump();
    let t0 = Instant::now();
    let mut g = STATE.lock().unwrap();
    loop {
        let Some(s) = g.as_ref() else { return };
        if done(s) {
            return;
        }
        if t0.elapsed() > HOLD_LIMIT {
            let text = states(s);
            drop(g);
            stop(format!("{what} has to follow work that is still held after {} ms", t0.elapsed().as_millis()), text);
        }
        g = CHANGED.wait_timeout(g, Duration::from_millis(20)).unwrap().0;
    }
}

unsafe fn give(queue: usize, op: Op) {
    {
        let mut g = STATE.lock().unwrap();
        let Some(x) = g.as_mut().and_then(|s| s.queues.get_mut(&queue)) else { return };
        x.held.push_back(op);
    }
    pump();
}

fn completed(fence: usize) -> u64 {
    if !super::real::is_fence(fence) {
        return 0;
    }
    let p = fence as *mut c_void;
    unsafe { ID3D12Fence::from_raw_borrowed(&p).map_or(0, |f| f.GetCompletedValue()) }
}

impl State {
    fn met(&self, need: &Need) -> bool {
        match *need {
            Need::Cell { va, value } => unsafe { super::list_x::cell_written(va, value) },
            Need::Fence { fence, value } => self.signalled.get(&fence).is_some_and(|&v| v >= value) || completed(fence) >= value,
        }
    }

    /// The title queue whose held work would end `need`.
    fn provider(&self, need: &Need) -> Option<usize> {
        self.order.iter().copied().find(|q| {
            self.queues[q].held.iter().any(|op| match (op, need) {
                (Op::Lists(lists), Need::Cell { va, value }) => super::list_x::lists_write(lists, *va, *value),
                (Op::Signal { fence, value, .. }, Need::Fence { fence: f, value: v }) => fence == f && value >= v,
                _ => false,
            })
        })
    }
}

fn text(need: &Need) -> String {
    match *need {
        Need::Cell { va, value } => format!("a write of {value:#x} or more at {va:#x}"),
        Need::Fence { fence, value } => format!("fence {fence:#x} at {value:#x} or more"),
    }
}

fn states(s: &State) -> String {
    let all: Vec<String> = s
        .order
        .iter()
        .map(|q| {
            let x = &s.queues[q];
            let lists: usize = x.held.iter().map(|op| if let Op::Lists(l) = op { l.len() } else { 0 }).sum();
            format!(
                "q{}{}: {} calls held ({lists} lists){}{}",
                super::list_x::queue_id(*q),
                if *q == s.host { " (the PC queue)" } else { "" },
                x.held.len(),
                if x.busy { ", being handed over" } else { "" },
                x.need.as_ref().map_or(String::new(), |(n, at)| format!(", waiting {} ms for {}", at.elapsed().as_millis(), text(n)))
            )
        })
        .collect();
    all.join("; ")
}

/// Held queues that wait for each other's held work: none of them can ever go on.
fn cycle(s: &State) -> Option<String> {
    for &start in &s.order {
        let (mut at, mut path) = (start, Vec::new());
        while let Some((need, _)) = s.queues[&at].need.as_ref() {
            let Some(next) = s.provider(need) else { break };
            path.push(format!("q{} waits for {}, which q{}'s held work gives", super::list_x::queue_id(at), text(need), super::list_x::queue_id(next)));
            if next == start {
                return Some(path.join("; "));
            }
            if path.len() > s.order.len() {
                break;
            }
            at = next;
        }
    }
    None
}

unsafe fn stop(what: String, states: String) -> ! {
    crate::xlog_error!("single queue: {what}. Title queues: {states}");
    super::list_x::hard_stop(format!("single queue: {what}"))
}

/// Hand over all the held work that can go: each title queue's in its order, by one thread
/// at a time.
unsafe fn pump() {
    loop {
        let (queue, op, host) = {
            let mut g = STATE.lock().unwrap();
            let Some(s) = g.as_mut() else { return };
            let ready = s.order.iter().copied().find(|q| {
                let x = &s.queues[q];
                !x.busy && !x.held.is_empty() && x.need.as_ref().is_none_or(|(n, _)| s.met(n))
            });
            let Some(queue) = ready else { break };
            let host = s.host;
            let x = s.queues.get_mut(&queue).unwrap();
            x.busy = true;
            x.need = None;
            let op = x.held.pop_front().unwrap();
            if let Op::Lists(lists) = &op {
                x.running = lists.clone();
            }
            (queue, op, host)
        };
        let was = RUNNING.with(|r| r.replace(queue));
        let again = run(op, host);
        RUNNING.with(|r| r.set(was));
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        let x = s.queues.get_mut(&queue).unwrap();
        x.busy = false;
        x.running.clear();
        if let Some((need, op)) = again {
            x.held.push_front(op);
            x.need = Some((need, Instant::now()));
        }
        CHANGED.notify_all();
    }
    let found = STATE.lock().unwrap().as_ref().and_then(|s| cycle(s).map(|c| (c, states(s))));
    if let Some((c, states)) = found {
        stop(format!("held queues wait for each other's held work, which the console would never run either: {c}"), states);
    }
}

/// Hand `op` to the PC queue `host`, or as much of it as can go: the rest and what it waits for
/// are returned.
unsafe fn run(op: Op, host: usize) -> Option<(Need, Op)> {
    match op {
        Op::Lists(lists) => {
            let mut from = 0;
            loop {
                let hold = super::list_x::first_unwritten_wait(&lists[from..]);
                let upto = hold.map_or(lists.len(), |(i, ..)| from + i);
                if upto > from {
                    let pc: Vec<*mut c_void> = lists[from..upto].iter().map(|&l| l as *mut c_void).collect();
                    super::real::execute_mapped(host as *mut c_void, &pc);
                }
                match hold {
                    None => return None,
                    Some((0, va, value)) => return Some((Need::Cell { va, value }, Op::Lists(lists[from..].to_vec()))),
                    Some(_) => from = upto,
                }
            }
        }
        Op::Signal { fence, value, call } => {
            call();
            if let Some(s) = STATE.lock().unwrap().as_mut() {
                let v = s.signalled.entry(fence).or_insert(0);
                *v = (*v).max(value);
            }
            None
        }
        Op::Wait { fence, value } => {
            let need = Need::Fence { fence, value };
            let met = STATE.lock().unwrap().as_ref().is_none_or(|s| s.met(&need));
            (!met).then_some((need, Op::Wait { fence, value }))
        }
    }
}

/// The thread that stops the title when a hold lasts `HOLD_LIMIT` or the PC queue has not
/// finished what it was given `GPU_LIMIT` ago (no GPU wait is armed in this mode, so nothing
/// else sees a GPU that hangs).
fn watch(host: usize) {
    let _ = std::thread::Builder::new().name("xcl-single-queue".into()).spawn(move || unsafe {
        let (raw, hostp) = (super::real::device_raw(), host as *mut c_void);
        let fence: Option<ID3D12Fence> = ID3D12Device::from_raw_borrowed(&raw).and_then(|d| d.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok());
        let (mut sent, mut at) = (0u64, Instant::now());
        loop {
            std::thread::sleep(Duration::from_millis(500));
            let overdue = STATE.lock().unwrap().as_ref().and_then(|s| {
                s.order.iter().find_map(|q| s.queues[q].need.as_ref().filter(|(_, since)| since.elapsed() > HOLD_LIMIT).map(|(n, since)| {
                    (format!("q{}'s work has been held {} ms for {}, which no queue has been given", super::list_x::queue_id(*q), since.elapsed().as_millis(), text(n)), states(s))
                }))
            });
            if let Some((what, states)) = overdue {
                stop(what, states);
            }
            let (Some(fence), Some(q)) = (fence.as_ref(), ID3D12CommandQueue::from_raw_borrowed(&hostp)) else { continue };
            if fence.GetCompletedValue() >= sent {
                sent += 1;
                at = Instant::now();
                let _ = q.Signal(fence, sent);
            } else if at.elapsed() > GPU_LIMIT {
                let states = STATE.lock().unwrap().as_ref().map(states).unwrap_or_default();
                stop(format!("the PC queue has not finished the work it was given {} ms ago (the GPU hangs or is busy with it)", at.elapsed().as_millis()), states);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn fresh(queues: &[usize]) -> State {
        let mut s = State { host: queues[0], order: queues.to_vec(), ..Default::default() };
        for &q in queues {
            s.queues.insert(q, Queue::default());
        }
        s
    }

    /// A queue's Wait for a fence another queue signals later holds the waiting queue's later
    /// calls, in order, until that Signal is handed over; the other queue goes on meanwhile.
    #[test]
    fn a_wait_before_its_signal_holds_that_queues_work_until_the_signal_is_handed_over() {
        let (a, b, f, g) = (0x7E57_8001usize, 0x7E57_8002usize, 0x7E57_8101usize, 0x7E57_8102usize);
        let log = Arc::new(Mutex::new(Vec::new()));
        let note = |what: &'static str| {
            let log = log.clone();
            move || log.lock().unwrap().push(what)
        };
        unsafe {
            adopt_for_test(&[a, b]);
            wait(b, f, 1);
            signal(b, g, 1, note("b signals g"));
            assert!(log.lock().unwrap().is_empty(), "held behind the wait");
            signal(a, g, 7, note("a signals g"));
            assert_eq!(*log.lock().unwrap(), ["a signals g"], "the other queue is not held");
            signal(a, f, 1, note("a signals f"));
            assert_eq!(*log.lock().unwrap(), ["a signals g", "a signals f", "b signals g"]);
            wait(b, f, 1);
            signal(b, g, 2, note("b again"));
            assert_eq!(log.lock().unwrap().len(), 4, "a wait for a Signal handed over before holds nothing");
        }
        let g = STATE.lock().unwrap();
        assert!(g.as_ref().unwrap().queues.values().all(|q| q.held.is_empty() && q.need.is_none() && !q.busy));
    }

    /// Two held queues each waiting for a Signal held behind the other's wait can never go on.
    #[test]
    fn held_queues_waiting_for_each_others_held_signals_are_a_cycle() {
        let (a, b, f, g) = (0x7E57_8011usize, 0x7E57_8012usize, 0x7E57_8111usize, 0x7E57_8112usize);
        let mut s = fresh(&[a, b]);
        let hold = |s: &mut State, q: usize, waits: usize, signals: usize| {
            let x = s.queues.get_mut(&q).unwrap();
            x.held.push_back(Op::Wait { fence: waits, value: 1 });
            x.held.push_back(Op::Signal { fence: signals, value: 1, call: Box::new(|| ()) });
            x.need = Some((Need::Fence { fence: waits, value: 1 }, Instant::now()));
        };
        hold(&mut s, a, f, g);
        assert_eq!(cycle(&s), None, "the Signal may still come from a queue that is not held");
        hold(&mut s, b, g, f);
        let found = cycle(&s).expect("a cycle");
        assert!(found.contains("waits for fence") && found.matches("held work gives").count() == 2, "{found}");
        assert!(states(&s).contains("2 calls held"));
    }

    unsafe fn adopt_for_test(queues: &[usize]) {
        let mut g = STATE.lock().unwrap();
        let s = g.get_or_insert_with(State::default);
        if s.host == 0 {
            s.host = queues[0];
            HOST.store(queues[0], Ordering::Release);
        }
        for &q in queues {
            s.order.push(q);
            s.queues.insert(q, Queue::default());
        }
    }
}
