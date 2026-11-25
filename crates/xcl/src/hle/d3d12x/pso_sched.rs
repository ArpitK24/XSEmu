//! The policy of background pipeline builds, apart from the threads and the driver that carry it
//! out (`pso_async`, `real::DRIVER_COMPILES`): what a driver compile may start within the memory
//! left, and which build a background worker takes next. Nothing here asks the system anything,
//! so its tests run on any host.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

/// DXIL from which a program is a giant: at most `GIANTS_AT_ONCE` of them compile at once
pub const GIANT_DXIL: usize = 1 << 20;
pub const GIANTS_AT_ONCE: usize = 3;

/// A driver compile's expected peak of this process's private memory, from the bytes of its
/// DXIL. Measured on an NVIDIA driver (RTX 3080), one compile at a time, private bytes sampled
/// every 50 ms (`real::driver_measure`):
///
/// | DXIL      | peak    | per byte | time  |
/// |-----------|---------|----------|-------|
/// | 0.51 MB   | 0.30 GB | 575 B    | 11 s  |
/// | 0.63 MB   | 0.42 GB | 643 B    | 18 s  |
/// | 1.64 MB   | 1.04 GB | 621 B    | 38 s  |
/// | 1.86 MB   | 1.19 GB | 625 B    | 47 s  |
/// | 3.09 MB   | 2.83 GB | 894 B    | 173 s |
/// | 3.35 MB   | 4.36 GB | 1271 B   | 166 s |
///
/// All of it came back when the compile ended. A giant's cost grows faster than its size (by
/// its square, among these), and two of nearly one size differ by half. Under the giant line
/// the estimate is `SMALL_PEAK_PER_BYTE` for each byte of DXIL (`set_small_peak`), a fifth to a
/// third over the measured: a retail game makes some 1 100 compute programs of 0.5 to 1 MB,
/// and at 1 KB a byte (~700 MB each) with ~4 GB available hardly one of them started in the
/// background. From there it is one and a half times the most measured: 600 MB times the square
/// of its MB (2 GB at least); and from 4 MB, where nothing was measured, 3.3 GB for each MB as
/// before (a 6.69 MB program took a warm-up's process to a 21.9 GB working set, the driver's
/// mapped shader cache included). The estimates at first were 1.5 GB and 3.3 GB for each MB
/// throughout: 2.5 to 5 times what the four giants above took, so that none of them was
/// admitted beside a game that left 4.6 GB available.
pub fn peak(dxil_len: usize) -> u64 {
    peak_at(dxil_len, SMALL_PEAK.load(Ordering::Relaxed))
}

/// `peak` with `small` bytes of memory for each byte of DXIL under the giant line.
pub fn peak_at(dxil_len: usize, small: u64) -> u64 {
    let len = dxil_len as u64;
    if dxil_len >= 4 * GIANT_DXIL {
        len * 3380
    } else if dxil_len >= GIANT_DXIL {
        (len * len * 600 >> 20).max(2 << 30)
    } else {
        // (never over the least giant's: more DXIL is never expected to take less)
        (len * small).clamp(128 << 20, 2 << 30)
    }
}

/// Memory expected for each byte of DXIL of a driver compile under the giant line.
pub const SMALL_PEAK_PER_BYTE: u64 = 800;
static SMALL_PEAK: AtomicU64 = AtomicU64::new(SMALL_PEAK_PER_BYTE);

/// The memory expected for each byte of DXIL under the giant line is `per_byte` from now on
/// (256 to 4096: a PC whose driver takes more or less than the one measured).
pub fn set_small_peak(per_byte: u64) {
    SMALL_PEAK.store(per_byte.clamp(256, 4096), Ordering::Relaxed);
}

/// The driver compiles running or waiting now.
#[derive(Clone, Copy, Debug, Default)]
pub struct Load {
    pub running: usize,
    /// giants among `running`, the ones a bind waits for too
    pub giants: usize,
    /// compiles a bind waits for among `running`
    pub urgent: usize,
    /// other compiles a bind waits for, waiting for memory
    pub starving: usize,
    /// the expected peaks of those running
    pub reserved: u64,
}

/// A driver compile asking to start.
#[derive(Clone, Copy, Debug)]
pub struct Ask {
    pub dxil_len: usize,
    /// a bind waits for it
    pub wanted: bool,
    /// background compiles at once at most
    pub most: usize,
    /// the process's private memory to stay within, when given
    pub cap: Option<u64>,
}

/// The memory a compile starts in.
#[derive(Clone, Copy, Debug)]
pub struct Room {
    /// physical memory available now
    pub available: u64,
    /// what a background compile leaves available
    pub floor: u64,
    /// what a compile a bind waits for leaves available
    pub urgent_floor: u64,
    /// the process's private memory now (read with `Ask::cap`)
    pub commit: u64,
    /// `floor` is the line a run is stopped at, not a margin: the expected peaks of the compiles
    /// running count whole against a background compile, not half (one that has not grown yet
    /// would otherwise be counted for half of what it will take)
    pub firm: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// it starts as a compile a bind waits for, outside the background limits
    Urgent,
    /// it starts as a background compile
    Background,
    /// it does not start now
    Hold,
}

/// Whether a driver compile starts now. The memory available, less half the expected peaks of
/// the compiles running (they may still be growing), must hold its own expected peak and the
/// floor. A compile a bind waits for needs nothing else. A background one also stays within
/// `most` at once and the private-memory cap, holds back while a waited-for compile waits
/// for memory, and as a giant starts only as the second at once, or the third
/// (`GIANTS_AT_ONCE`) while the memory would still hold one more giant after it – counting the
/// giants binds wait for, so those never add to a full set of background ones.
pub fn admit(ask: &Ask, load: &Load, room: &Room) -> Verdict {
    let cost = peak(ask.dxil_len);
    let spare = room.available.saturating_sub(load.reserved / 2);
    if ask.wanted {
        return if spare >= cost + room.urgent_floor { Verdict::Urgent } else { Verdict::Hold };
    }
    let spare = if room.firm { room.available.saturating_sub(load.reserved) } else { spare };
    let giants = if spare >= room.floor + cost + peak(GIANT_DXIL.max(ask.dxil_len)) { GIANTS_AT_ONCE } else { GIANTS_AT_ONCE - 1 };
    let held = load.starving > 0
        || load.running.saturating_sub(load.urgent) >= ask.most
        || (ask.dxil_len >= GIANT_DXIL && load.giants >= giants)
        || spare < room.floor + cost
        || ask.cap.is_some_and(|c| room.commit + load.reserved / 2 + cost > c);
    if held {
        Verdict::Hold
    } else {
        Verdict::Background
    }
}

/// Where a pipeline made for later waits for a background worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// its big driver compile did not fit the memory when a worker came to it (`Backlog::park`)
    Parked,
    /// console shaders of `HEAVY_SHADER_BYTES` or more: its driver compile is likely a big one
    Heavy,
    /// shaders an earlier run used
    Known,
    Compute,
    Graphics,
}

/// Console shader bytes from which a pipeline is built ahead of the others (`Lane::Heavy`): a
/// retail game's compute programs came to ~16 B of DXIL for each byte of console code, so these
/// are the ones whose driver compile takes seconds to minutes and hundreds of MB to GB. Started
/// when the title makes them, they compile while it still loads and the memory is there; left
/// for their first bind, the bind waits that long, and a giant no longer fits beside the game.
pub const HEAVY_SHADER_BYTES: usize = 16 << 10;

/// Root signatures of the pipelines bound last that `Backlog::pop` looks at.
pub const RECENT_GROUPS: usize = 8;

/// The pipelines made for later, in the order background workers take them – by how soon a
/// bind is likely to ask for them, not by when they were made:
/// 1. a parked one whose compile fits now – first those a bind asked for and went without
///    (they fit as the compiles binds wait for do), then the others; of each first those of a
///    group bound lately (2.), then the biggest (what is biggest fits least often, and the
///    game's own memory only grows);
/// 2. those of a group (the title's root signature: a pass's pipelines share one) bound
///    lately, the last bound group first, the newest of it first;
/// 3. the heavy ones, the biggest first, on a share of the workers so the rest keep going
///    through the many small ones;
/// 4. those of shaders an earlier run used, then compute (a dispatch waits for its pipeline
///    however long it takes), then graphics – of each the newest first: a title makes a
///    pipeline when what draws with it comes in, and of the tens of thousands it makes while
///    it loads the first ones are no likelier than the last.
pub struct Backlog<T> {
    /// (DXIL bytes, (group, pipeline)) a bind asked for, parked for memory, smallest first
    asked: Vec<(usize, (u64, u64, T))>,
    /// (DXIL bytes, (group, pipeline)), smallest first
    parked: Vec<(usize, (u64, u64, T))>,
    /// the pipelines waiting for a worker, by the number each got when it was made; a number
    /// in the lists below that is no longer here was taken through another list
    waiting: HashMap<u64, (T, Lane)>,
    /// (console shader bytes, number), smallest first
    heavy: Vec<(usize, u64)>,
    /// numbers, oldest first
    known: Vec<u64>,
    compute: Vec<u64>,
    graphics: Vec<u64>,
    /// the numbers of each group, oldest first
    groups: HashMap<u64, Vec<u64>>,
    promoted: VecDeque<u64>,
    next: u64,
    /// heavy ones among `waiting`
    heavy_waiting: usize,
    /// workers building a heavy one now
    heavy_out: usize,
}

impl<T> Default for Backlog<T> {
    fn default() -> Self {
        Backlog {
            asked: Vec::new(),
            parked: Vec::new(),
            waiting: HashMap::new(),
            heavy: Vec::new(),
            known: Vec::new(),
            compute: Vec::new(),
            graphics: Vec::new(),
            groups: HashMap::new(),
            promoted: VecDeque::new(),
            next: 1,
            heavy_waiting: 0,
            heavy_out: 0,
        }
    }
}

fn insert_by_size<T>(list: &mut Vec<(usize, T)>, size: usize, item: T) {
    // (before its equals: of two the same size the one made first is taken first, from the end)
    let at = list.partition_point(|(s, _)| *s < size);
    list.insert(at, (size, item));
}

/// The last number of `list` still in `waiting`; the ones after it, taken through another list, go.
fn last_waiting<T>(list: &mut Vec<u64>, waiting: &HashMap<u64, (T, Lane)>) -> Option<u64> {
    while let Some(n) = list.pop() {
        if waiting.contains_key(&n) {
            return Some(n);
        }
    }
    None
}

impl<T> Backlog<T> {
    /// A pipeline the title made: `shader_bytes` of console shaders, `known` from an earlier
    /// run, of `group` (its root signature; 0: unknown).
    pub fn push(&mut self, shader_bytes: usize, known: bool, compute: bool, group: u64, item: T) -> u64 {
        let n = self.next;
        self.next += 1;
        let lane = if shader_bytes >= HEAVY_SHADER_BYTES {
            insert_by_size(&mut self.heavy, shader_bytes, n);
            self.heavy_waiting += 1;
            Lane::Heavy
        } else if known {
            self.known.push(n);
            Lane::Known
        } else if compute {
            self.compute.push(n);
            Lane::Compute
        } else {
            self.graphics.push(n);
            Lane::Graphics
        };
        if group != 0 {
            self.groups.entry(group).or_default().push(n);
        }
        self.waiting.insert(n, (item, lane));
        n
    }

    pub fn promote(&mut self, n: u64) {
        if n == 0 || self.promoted.back() == Some(&n) {
            return;
        }
        self.promoted.retain(|&id| id != n);
        if self.promoted.len() >= 64 {
            self.promoted.pop_front();
        }
        self.promoted.push_back(n);
    }

    /// A pipeline of `group` whose driver compile of `dxil_len` bytes did not fit: it waits for
    /// memory. `asked`: a bind asked for it and went without.
    pub fn park(&mut self, dxil_len: usize, asked: bool, group: u64, n: u64, item: T) {
        insert_by_size(if asked { &mut self.asked } else { &mut self.parked }, dxil_len, (group, n, item));
    }

    fn take(&mut self, n: u64) -> Option<(T, Lane)> {
        let (item, lane) = self.waiting.remove(&n)?;
        if lane == Lane::Heavy {
            self.heavy_waiting -= 1;
            self.heavy_out += 1;
        }
        Some((item, lane))
    }

    /// The next pipeline for one of `workers` background workers, and the lane it came from
    /// (`done` is told when it is built). `recent`: the groups of the pipelines bound last, the
    /// last one first; `live`: it still wants building (parked ones that do not are dropped);
    /// `fits`: a driver compile of so many bytes of DXIL would start now, as one a bind asked
    /// for (true) or as a background one.
    pub fn pop(&mut self, workers: usize, recent: &[u64], live: impl Fn(&T) -> bool, fits: impl Fn(usize, bool) -> bool) -> Option<(T, Lane)> {
        let recent = &recent[..recent.len().min(RECENT_GROUPS)];
        for asked in [true, false] {
            let list = if asked { &mut self.asked } else { &mut self.parked };
            list.retain(|(_, (_, _, p))| live(p));
            if !asked {
                let promoted = self.promoted.iter().enumerate().rev().find_map(|(pos, &id)| {
                    if self.waiting.contains_key(&id) {
                        Some((pos, id, None))
                    } else {
                        list.iter().position(|(len, (_, n, _))| *n == id && fits(*len, false)).map(|at| (pos, id, Some(at)))
                    }
                });
                if let Some((pos, id, parked)) = promoted {
                    self.promoted.remove(pos);
                    if let Some(at) = parked {
                        return Some((list.remove(at).1 .2, Lane::Parked));
                    }
                    return self.take(id);
                }
            }
            let lately = recent.iter().filter(|g| **g != 0).find_map(|g| list.iter().rposition(|(len, (group, _, _))| group == g && fits(*len, asked)));
            if let Some(at) = lately.or_else(|| list.iter().rposition(|(len, _)| fits(*len, asked))) {
                return Some((list.remove(at).1 .2, Lane::Parked));
            }
        }
        for group in recent {
            let Some(list) = self.groups.get_mut(group) else { continue };
            match last_waiting(list, &self.waiting) {
                Some(n) => return self.take(n),
                None => drop(self.groups.remove(group)),
            }
        }
        if self.heavy_waiting > 0 && (self.heavy_waiting == self.waiting.len() || self.heavy_out < heavy_share(workers)) {
            while let Some((_, n)) = self.heavy.pop() {
                if self.waiting.contains_key(&n) {
                    return self.take(n);
                }
            }
        }
        for lane in [Lane::Known, Lane::Compute, Lane::Graphics] {
            let list = match lane {
                Lane::Known => &mut self.known,
                Lane::Compute => &mut self.compute,
                _ => &mut self.graphics,
            };
            if let Some(n) = last_waiting(list, &self.waiting) {
                return self.take(n);
            }
        }
        None
    }

    /// A worker is done with what it took from `lane`.
    pub fn done(&mut self, lane: Lane) {
        if lane == Lane::Heavy {
            self.heavy_out = self.heavy_out.saturating_sub(1);
        }
    }

    /// (pipelines waiting for a worker, of them parked for memory)
    pub fn waiting(&self) -> (usize, usize) {
        (self.asked.len() + self.parked.len() + self.waiting.len(), self.asked.len() + self.parked.len())
    }

    pub fn has_parked(&self) -> bool {
        !self.parked.is_empty() || !self.asked.is_empty()
    }

    /// The smallest driver compile parked, in bytes of DXIL (0: none).
    pub fn least_parked(&self) -> usize {
        self.asked.first().into_iter().chain(self.parked.first()).map(|(len, _)| (*len).max(1)).min().unwrap_or(0)
    }
}

/// The workers that may build heavy pipelines at once while others wait: a quarter, one at least.
pub fn heavy_share(workers: usize) -> usize {
    workers.div_ceil(4).max(1)
}

/// Background workers for a PC of `threads` hardware threads on `cores` cores: one for each
/// core, and two threads left over at least; two at least, 32 at most. (One for each thread but
/// two – 22 on 12 cores, ~17 of them busy – made a retail game's loading 2.6 times as long: a
/// worker on a core's second thread takes from the title's thread on its first.) How many of
/// them build at a time follows what the title is doing (`Pace`).
pub fn background_workers(threads: usize, cores: usize) -> usize {
    cores.min(threads.saturating_sub(2)).clamp(2, 32)
}

/// The background workers that build while the title needs the PC most: a quarter, two at least.
pub fn least_workers(pool: usize) -> usize {
    (pool / 4).max(2).min(pool.max(1))
}

/// How the background workers go about it now (`Pace::step`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tempo {
    /// workers that may build at once
    pub workers: usize,
    /// they run at their lowest priority: behind title threads that run below normal themselves
    pub low: bool,
}

/// Steps (`Pace::step`) the title counts as loading for after its storage was last busy.
pub const LOADING_HOLD: u32 = 4;

/// Whether the title loads, from what its storage API did within `ms`: 16 requests or 8 MB a
/// second, or more. (A retail game ran 160 to 740 requests a second while it loaded, 55 with its
/// loader starved, and one request in half a minute once in its world.)
pub fn loading(requests: u64, bytes: u64, ms: u64) -> bool {
    requests * 1000 >= 16 * ms || bytes * 1000 >= (8 << 20) * ms
}

/// How many background workers build at once, from what the rest of the process is doing: the
/// cores its other threads (the title's, its loader's, the storage's) are not using – half of
/// those, at the lowest priority, while the title loads through its storage API – and never
/// fewer than `least_workers`. Fewer at once as soon as the title needs the cores, one more a
/// step when it does not. (The storage alone does not tell: a retail game's loader read
/// nothing for four minutes of a load that 22 workers had slowed down.)
pub struct Pace {
    pool: usize,
    cores: usize,
    workers: usize,
    calm: u32,
}

impl Pace {
    /// For `pool` workers on `cores` cores. It starts at the least: a title starts by loading.
    pub fn new(pool: usize, cores: usize) -> Self {
        Pace { pool: pool.max(1), cores, workers: least_workers(pool), calm: 0 }
    }

    /// One step (they come at even intervals). `others`: the CPU the process's other threads
    /// used since the last step, in thousandths of a thread; `loading`: the title's storage
    /// ran requests since then, or holds some.
    pub fn step(&mut self, others: u64, loading: bool) -> Tempo {
        self.calm = if loading { 0 } else { self.calm.saturating_add(1) };
        let loading = self.calm < LOADING_HOLD;
        let free = self.cores.saturating_sub(others.div_ceil(1000) as usize);
        let target = if loading { free / 2 } else { free }.clamp(least_workers(self.pool), self.pool);
        self.workers = if target > self.workers { self.workers + 1 } else { target };
        Tempo { workers: self.workers, low: loading }
    }
}

/// What a run's background compiles leave available at the least while the title loads, and
/// once it has not loaded for `RISE_KEPT_MS` (`Headroom::floor`).
pub const MIN_HEADROOM: u64 = 4 << 30;
pub const CALM_HEADROOM: u64 = 3 << 29;

/// Console bytes already backed, and the host's private commit at the same sample.
#[derive(Clone, Copy, Default)]
pub struct TitleMemory {
    pub backed: u64,
    pub commit: u64,
}

impl TitleMemory {
    /// Project the remaining console bytes at the observed host bytes per backed byte.
    /// Before a quarter of the budget is backed, use that quarter as the denominator
    /// so startup's fixed costs don't get multiplied by a nearly empty budget.
    pub fn host_reserve(self, budget: u64) -> u64 {
        let left = budget.saturating_sub(self.backed);
        let observed = self.backed.max(budget / 4).max(1) as u128;
        let host = observed.max(self.commit as u128);
        (u128::from(left) * host).div_ceil(observed).min(u128::from(u64::MAX)) as u64
    }
}
/// The span a rise of the process's memory is measured over, and how long a rise – and that
/// the title loaded – is kept in mind.
const RISE_SPAN_MS: u64 = 10_000;
const RISE_KEPT_MS: u64 = 60_000;

/// The memory a run's background compiles leave available, following what the title does:
/// the line a run is stopped at and twice the biggest rise of the process's private memory
/// within `RISE_SPAN_MS` of the last `RISE_KEPT_MS` – another rise like it still ends above
/// that line – and at the least one amount while the title loads (and for `RISE_KEPT_MS`
/// after, and from the start: a title starts by loading) and a smaller one once it is calm.
/// (With 1 GB left throughout, a retail game's world load committed 2.2 GB more within seconds
/// and took the PC to 688 MB available. With 4 GB left throughout, its ~1 100 compute programs
/// of 0.5 to 1 MB stayed parked in a world that left ~4 GB available, and each was compiled
/// when a bind already waited for it. Its rises within 10 s were 2.5 to 3 GB while it loaded
/// and 0.3 to 0.9 GB in its world; within 30 s over 5 GB – twice that would have kept every
/// background build out of its load.) The compiles' own memory is in the rise too: they hold
/// back from each other while it grows.
#[derive(Default)]
pub struct Headroom {
    /// (ms, private memory) within the last `RISE_SPAN_MS`, and the one before them
    seen: VecDeque<(u64, u64)>,
    /// (ms, the rise up to then) within the last `RISE_KEPT_MS`
    rises: VecDeque<(u64, u64)>,
    /// when it was last told (ms), and when it was last told the title loads
    now: u64,
    loaded: u64,
}

impl Headroom {
    pub const fn empty() -> Self {
        Self { seen: VecDeque::new(), rises: VecDeque::new(), now: 0, loaded: 0 }
    }

    /// The process's private memory is `commit` at `now_ms`, and the title loads or not (`loading`).
    pub fn note(&mut self, now_ms: u64, commit: u64, loading: bool) {
        // (the one before the span stays: after a pause in the samples the rise is measured from it)
        while self.seen.len() > 1 && self.seen[1].0 + RISE_SPAN_MS <= now_ms {
            self.seen.pop_front();
        }
        self.seen.push_back((now_ms, commit));
        let low = self.seen.iter().map(|s| s.1).min().unwrap_or(commit);
        while self.rises.front().is_some_and(|r| r.0 + RISE_KEPT_MS < now_ms) {
            self.rises.pop_front();
        }
        self.rises.push_back((now_ms, commit - low));
        self.now = now_ms;
        if loading {
            self.loaded = now_ms;
        }
    }

    /// Whether the title counts as loading now.
    pub fn loading(&self) -> bool {
        self.now < self.loaded + RISE_KEPT_MS
    }

    /// What background compiles leave available now, above `stop`, the line a run is stopped
    /// at: `calm` at the least, `loading` while the title loads. `bind_waits`: a bind waits for
    /// a pipeline now – the title is held anyway, and what is built meanwhile is one bind less
    /// to wait: no least amount then, only what follows the growth (`stop` itself, the floor of
    /// the compile the bind waits for, while the memory is flat). (With a least amount alone
    /// the memory available settled just over it once a retail game was in its world: nothing
    /// heavy started in the background, and the binds built 2.2 pipelines a minute, one at a
    /// time, 26 to 28 s each.)
    pub fn floor(&self, stop: u64, calm: u64, loading: u64, bind_waits: bool) -> u64 {
        let growth = stop + 2 * self.rises.iter().map(|r| r.1).max().unwrap_or(0);
        if bind_waits {
            return growth;
        }
        growth.max(if self.loading() { loading } else { calm.min(loading) })
    }

    pub fn title_floor(&self, stop: u64, calm: u64, loading: u64, bind_waits: bool, reserve: u64) -> u64 {
        let growth_floor = self.floor(stop, calm, loading, true);
        if bind_waits {
            growth_floor
        } else {
            let least = if self.loading() { loading } else { calm.min(loading) };
            least.max(growth_floor.saturating_add(reserve))
        }
    }
}

/// Which built pipelines nothing has bound yet to release, of `built` (id, when built): the ones
/// built `limit` or longer before `now`, and of the rest the oldest beyond the newest `keep`. A
/// run that builds every pipeline when the title makes it would otherwise hold thousands a
/// frame never binds (~400 KB each in an NVIDIA driver); one released is built again at its
/// bind from the driver's blob in ~0.1 s.
pub fn idle_to_release<I: Copy>(built: &mut Vec<(I, u64)>, now: u64, limit: u64, keep: usize) -> Vec<I> {
    built.sort_by_key(|&(_, at)| std::cmp::Reverse(at));
    let fresh = built.iter().take_while(|&&(_, at)| now.saturating_sub(at) < limit).count().min(keep);
    built[fresh..].iter().map(|&(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    fn room(available: u64) -> Room {
        Room { available, floor: GB, urgent_floor: GB, commit: 0, firm: false }
    }

    fn ask(dxil_len: usize, wanted: bool) -> Ask {
        Ask { dxil_len, wanted, most: 20, cap: None }
    }

    #[test]
    fn peaks_stay_above_the_measured_costs() {
        const MB: u64 = 1 << 20;
        // (bytes of DXIL, the private peak measured in MB)
        for (len, measured) in [(533_812, 307), (660_536, 425), (1_717_176, 1067), (1_945_712, 1217), (3_241_524, 2897), (3_515_708, 4468)] {
            let estimate = peak_at(len, SMALL_PEAK_PER_BYTE);
            // (under the giant line a fifth over at least, from it a half)
            let (least, most) = if len < GIANT_DXIL { (measured * MB * 115 / 100, measured * MB * 3 / 2) } else { (measured * MB * 3 / 2, measured * MB * 3) };
            assert!(estimate >= least, "{len} B: {} MB estimated, {measured} MB measured", estimate / MB);
            assert!(estimate <= most, "{len} B: {} MB estimated is no longer near the {measured} MB measured", estimate / MB);
        }
        let at = |len| peak_at(len, SMALL_PEAK_PER_BYTE);
        assert_eq!(at(1000), 128 << 20);
        assert_eq!(at(512 << 10), (512 << 10) * 800);
        assert_eq!(at(GIANT_DXIL), 2 << 30);
        assert_eq!(at(6 << 20), 6 * 3380 << 20);
        // never less for more DXIL, whatever a byte under the giant line is expected to take
        let sizes = [0, 1000, 256 << 10, GIANT_DXIL - 1, GIANT_DXIL, 2 << 20, (4 << 20) - 1, 4 << 20, 27 << 20];
        for small in [256, SMALL_PEAK_PER_BYTE, 1024, 4096] {
            assert!(sizes.windows(2).all(|w| peak_at(w[0], small) <= peak_at(w[1], small)), "{small} B a byte");
        }
        assert_eq!(peak_at(700 << 10, 1024), 700 << 20, "what it was before");
        assert_eq!(peak_at(900 << 10, 4096), 2 << 30, "never over the least giant's");
    }

    /// A compile is admitted when its expected peak fits the memory available less the floor,
    /// and held – not refused – when it does not.
    #[test]
    fn a_compile_starts_when_its_peak_fits_above_the_floor() {
        let len = 512 << 10;
        let cost = peak(len);
        assert_eq!(admit(&ask(len, false), &Load::default(), &room(cost + GB)), Verdict::Background);
        assert_eq!(admit(&ask(len, false), &Load::default(), &room(cost + GB - 1)), Verdict::Hold);
        assert_eq!(admit(&ask(len, true), &Load::default(), &room(cost + GB)), Verdict::Urgent);
        assert_eq!(admit(&ask(len, true), &Load::default(), &room(cost + GB - 1)), Verdict::Hold);
    }

    /// Half the expected peaks of the compiles running count against the memory available.
    #[test]
    fn running_compiles_count_against_the_memory() {
        let len = 512 << 10;
        let cost = peak(len);
        let load = Load { running: 2, reserved: 2 * cost, ..Load::default() };
        assert_eq!(admit(&ask(len, false), &load, &room(2 * cost + GB)), Verdict::Background);
        assert_eq!(admit(&ask(len, false), &load, &room(2 * cost + GB - 1)), Verdict::Hold);
    }

    /// The floor of a background compile is its own: a run keeps the 1 GB its watchdog stops
    /// at, a warm-up 6 GB – and a compile a bind waits for only ever its 1 GB.
    #[test]
    fn a_background_compile_keeps_its_own_floor() {
        let len = 512 << 10;
        let cost = peak(len);
        let warm = Room { available: cost + 2 * GB, floor: 6 * GB, urgent_floor: GB, commit: 0, firm: false };
        assert_eq!(admit(&ask(len, false), &Load::default(), &warm), Verdict::Hold);
        assert_eq!(admit(&ask(len, true), &Load::default(), &warm), Verdict::Urgent);
        assert_eq!(admit(&ask(len, false), &Load::default(), &Room { floor: GB, ..warm }), Verdict::Background);
    }

    /// Against a firm floor the peaks of the compiles running count whole for a background
    /// compile: three of them admitted at once on half counts would together pass the floor.
    #[test]
    fn a_firm_floor_counts_running_peaks_whole() {
        let len = 900 << 10;
        let cost = peak(len);
        let one = Load { running: 1, reserved: cost, ..Load::default() };
        let firm = |available| Room { firm: true, ..room(available) };
        assert_eq!(admit(&ask(len, false), &one, &room(cost + cost / 2 + GB)), Verdict::Background);
        assert_eq!(admit(&ask(len, false), &one, &firm(cost + cost / 2 + GB)), Verdict::Hold);
        assert_eq!(admit(&ask(len, false), &one, &firm(2 * cost + GB)), Verdict::Background);
        assert_eq!(admit(&ask(len, true), &one, &firm(cost + cost / 2 + GB)), Verdict::Urgent, "a bind's compile counts them half, as before");
    }

    /// Two giants at once, a third only while the memory would hold one more after it; the
    /// giants binds wait for count, and are never held by the count themselves.
    #[test]
    fn giants_are_capped_at_two_or_three_at_once() {
        let giant = 2 << 20;
        let cost = peak(giant);
        let plenty = room(64 * GB);
        let two = Load { running: 2, giants: 2, reserved: 2 * cost, ..Load::default() };
        assert_eq!(admit(&ask(giant, false), &Load { running: 1, giants: 1, reserved: cost, ..Load::default() }, &plenty), Verdict::Background);
        assert_eq!(admit(&ask(giant, false), &two, &plenty), Verdict::Background, "the third, with memory for a fourth");
        assert_eq!(admit(&ask(giant, false), &two, &room(3 * cost + GB)), Verdict::Background);
        assert_eq!(admit(&ask(giant, false), &two, &room(3 * cost + GB - 1)), Verdict::Hold, "no third without memory for one more after it");
        let three = Load { running: 3, giants: 3, reserved: 3 * cost, ..Load::default() };
        assert_eq!(admit(&ask(giant, false), &three, &plenty), Verdict::Hold);
        assert_eq!(admit(&ask(512 << 10, false), &three, &plenty), Verdict::Background, "the cap is on giants only");
        let waited = Load { running: 3, giants: 3, urgent: 1, reserved: 3 * cost, ..Load::default() };
        assert_eq!(admit(&ask(giant, false), &waited, &plenty), Verdict::Hold, "a giant a bind waits for takes a place of the three");
        assert_eq!(admit(&ask(giant, true), &three, &plenty), Verdict::Urgent, "a bind's giant is not held by the count");
    }

    /// Background compiles hold back while a compile a bind waits for waits for memory, and stay
    /// within `most` at once and the private-memory cap; the ones binds wait for do not count
    /// towards `most`.
    #[test]
    fn background_compiles_give_way_and_stay_within_their_limits() {
        let len = 300 << 10;
        let plenty = room(64 * GB);
        assert_eq!(admit(&ask(len, false), &Load { starving: 1, ..Load::default() }, &plenty), Verdict::Hold);
        let busy = Load { running: 4, urgent: 2, ..Load::default() };
        assert_eq!(admit(&Ask { most: 2, ..ask(len, false) }, &busy, &plenty), Verdict::Hold);
        assert_eq!(admit(&Ask { most: 3, ..ask(len, false) }, &busy, &plenty), Verdict::Background);
        assert_eq!(admit(&Ask { most: 2, ..ask(len, true) }, &busy, &plenty), Verdict::Urgent);
        let capped = Ask { cap: Some(4 * GB), ..ask(len, false) };
        assert_eq!(admit(&capped, &Load::default(), &Room { commit: 4 * GB, ..plenty }), Verdict::Hold);
        assert_eq!(admit(&capped, &Load::default(), &Room { commit: GB, ..plenty }), Verdict::Background);
    }

    fn drain(b: &mut Backlog<&'static str>, workers: usize) -> Vec<&'static str> {
        std::iter::from_fn(|| {
            let (p, lane) = b.pop(workers, &[], |_| true, |_, _| true)?;
            b.done(lane);
            Some(p)
        })
        .collect()
    }

    /// Heavy pipelines first, the biggest of them first; then the ones an earlier run used, then
    /// compute, then graphics, each in the order made.
    #[test]
    fn the_backlog_goes_heavy_known_compute_graphics() {
        let mut b = Backlog::default();
        b.push(100, false, false, 0, "g1");
        b.push(100, false, true, 0, "c1");
        b.push(HEAVY_SHADER_BYTES, false, true, 0, "heavy-small");
        b.push(100, true, false, 0, "k1");
        b.push(100, false, false, 0, "g2");
        b.push(4 * HEAVY_SHADER_BYTES, false, true, 0, "heavy-big");
        b.push(100, true, true, 0, "k2");
        b.push(100, false, true, 0, "c2");
        assert_eq!(b.waiting(), (8, 0));
        // (of each lane the newest first)
        assert_eq!(drain(&mut b, 8), ["heavy-big", "heavy-small", "k2", "k1", "c2", "c1", "g2", "g1"]);
        assert_eq!(b.waiting(), (0, 0));
    }

    /// Only a share of the workers builds heavy pipelines while small ones wait: the rest go on
    /// with those; once nothing else waits, every worker takes heavy ones.
    #[test]
    fn heavy_pipelines_take_a_share_of_the_workers() {
        let mut b = Backlog::default();
        for name in ["h1", "h2", "h3", "h4"] {
            b.push(HEAVY_SHADER_BYTES, false, true, 0, name);
        }
        b.push(100, false, true, 0, "c1");
        b.push(100, false, true, 0, "c2");
        let workers = 8;
        assert_eq!(heavy_share(workers), 2);
        let take = |b: &mut Backlog<&'static str>| b.pop(workers, &[], |_| true, |_, _| true).unwrap();
        assert_eq!(take(&mut b), ("h1", Lane::Heavy));
        assert_eq!(take(&mut b), ("h2", Lane::Heavy));
        assert_eq!(take(&mut b), ("c2", Lane::Compute), "two workers hold heavy ones: the third takes a small one");
        b.done(Lane::Heavy);
        assert_eq!(take(&mut b), ("h3", Lane::Heavy), "a heavy one done frees its place");
        assert_eq!(take(&mut b), ("c1", Lane::Compute));
        assert_eq!(take(&mut b), ("h4", Lane::Heavy), "nothing else waits: no share");
        assert_eq!(heavy_share(1), 1);
        assert_eq!(heavy_share(22), 6);
    }

    /// A compile that does not fit is parked and the worker goes on: nothing waits behind it.
    /// It comes back ahead of everything once it fits, the biggest that fits first.
    #[test]
    fn a_parked_build_holds_nothing_up_and_returns_when_it_fits() {
        let mut b = Backlog::default();
        b.push(100, false, true, 0, "c1");
        b.push(100, false, false, 0, "g1");
        b.park(3 << 20, false, 0, 0, "giant");
        b.park(300 << 10, false, 0, 0, "mid");
        b.park(600 << 10, false, 0, 0, "big");
        assert_eq!(b.waiting(), (5, 3));
        assert!(b.has_parked());
        assert_eq!(b.least_parked(), 300 << 10);
        let none_fit = |b: &mut Backlog<&'static str>| b.pop(4, &[], |_| true, |_, _| false).map(|(p, _)| p);
        assert_eq!(none_fit(&mut b), Some("c1"), "no memory for the parked ones: the others go on");
        assert_eq!(b.pop(4, &[], |_| true, |len, _| len < 1 << 20), Some(("big", Lane::Parked)), "the biggest that fits");
        assert_eq!(b.pop(4, &[], |_| true, |_, _| true), Some(("giant", Lane::Parked)));
        assert_eq!(b.pop(4, &[], |_| true, |len, _| len < 100 << 10), Some(("g1", Lane::Graphics)));
        assert_eq!(none_fit(&mut b), None, "only a parked one left, and no memory for it: the worker waits");
        assert_eq!(b.pop(4, &[], |_| true, |_, _| true), Some(("mid", Lane::Parked)));
        assert!(!b.has_parked());
        assert_eq!(b.least_parked(), 0);
    }

    /// A parked build a bind took over (or whose pipeline the title released) is dropped: it
    /// does not come back to a background worker.
    #[test]
    fn a_parked_build_something_else_took_is_dropped() {
        let mut b = Backlog::default();
        b.park(300 << 10, false, 0, 0, "taken by a bind");
        b.park(600 << 10, false, 0, 0, "still waiting");
        assert_eq!(b.pop(4, &[], |p| *p != "taken by a bind", |_, _| false), None);
        assert_eq!(b.waiting(), (1, 1));
        assert_eq!(b.pop(4, &[], |_| true, |_, _| true), Some(("still waiting", Lane::Parked)));
    }

    #[test]
    fn background_workers_are_one_for_each_core_and_leave_two_threads() {
        assert_eq!(background_workers(24, 12), 12);
        assert_eq!(background_workers(16, 8), 8);
        assert_eq!(background_workers(8, 8), 6);
        assert_eq!(background_workers(8, 4), 4);
        assert_eq!(background_workers(4, 4), 2);
        assert_eq!(background_workers(2, 1), 2);
        assert_eq!(background_workers(128, 64), 32);
        assert_eq!((least_workers(12), least_workers(8), least_workers(4), least_workers(1)), (3, 2, 2, 1));
    }

    /// Unbound pipelines are released after the idle limit, and before it the oldest beyond
    /// the number kept.
    #[test]
    fn idle_pipelines_go_by_age_and_by_number() {
        let mut built = vec![(1, 1_000), (2, 29_000), (3, 30_000), (4, 31_000), (5, 5_000)];
        let mut gone = idle_to_release(&mut built, 35_000, 30_000, 100);
        gone.sort();
        assert_eq!(gone, [1, 5], "30 s old or more");
        let mut gone = idle_to_release(&mut built, 35_000, 30_000, 2);
        gone.sort();
        assert_eq!(gone, [1, 2, 5], "and the oldest beyond the two newest");
        assert!(idle_to_release(&mut built, 30_500, 30_000, 5).is_empty());
        assert_eq!(idle_to_release(&mut built, 35_000, 30_000, 0).len(), 5);
    }

    #[test]
    fn a_parked_build_a_bind_asked_for_goes_first_and_fits_as_its_own_kind() {
        let mut b = Backlog::default();
        b.park(3 << 20, false, 0, 0, "giant made ahead");
        b.park(600 << 10, true, 0, 0, "asked");
        b.park(2 << 20, true, 0, 0, "asked giant");
        b.push(100, false, false, 0, "g1");
        assert_eq!(b.waiting(), (4, 3));
        assert_eq!(b.least_parked(), 600 << 10);
        // background compiles have no room: only what a bind asked for comes back, the biggest first
        assert_eq!(b.pop(4, &[], |_| true, |_, asked| asked), Some(("asked giant", Lane::Parked)));
        assert_eq!(b.pop(4, &[], |_| true, |len, asked| asked && len > 1 << 20), Some(("g1", Lane::Graphics)));
        // with room for both kinds, the asked one still goes before the bigger one made ahead
        assert_eq!(b.pop(4, &[], |_| true, |_, _| true), Some(("asked", Lane::Parked)));
        assert_eq!(b.pop(4, &[], |_| true, |_, _| true), Some(("giant made ahead", Lane::Parked)));
        assert!(!b.has_parked());
    }

    #[test]
    fn the_pace_starts_low_and_rises_a_worker_a_step_while_the_title_is_idle() {
        let mut p = Pace::new(12, 12);
        // (the first steps still count as loading: a title starts by loading)
        for _ in 0..LOADING_HOLD - 1 {
            let t = p.step(1500, false);
            assert!(t.low && t.workers <= 5, "{t:?}");
        }
        let seen: Vec<Tempo> = (0..10).map(|_| p.step(1500, false)).collect();
        assert!(seen.iter().all(|t| !t.low));
        assert_eq!(seen.iter().map(|t| t.workers).collect::<Vec<_>>(), [6, 7, 8, 9, 10, 10, 10, 10, 10, 10], "up to the cores the title leaves");
    }

    #[test]
    fn the_pace_drops_at_once_when_the_title_takes_the_cores() {
        let mut p = Pace::new(12, 12);
        for _ in 0..20 {
            p.step(1000, false);
        }
        assert_eq!(p.step(1000, false).workers, 11);
        // the title's threads take six cores: six workers, at once; their priority stays
        assert_eq!(p.step(5200, false), Tempo { workers: 6, low: false });
        // … and all of them: never under the least
        assert_eq!(p.step(14_000, false), Tempo { workers: 3, low: false });
        assert_eq!(p.step(0, false), Tempo { workers: 4, low: false }, "back a worker a step");
    }

    #[test]
    fn the_pace_halves_and_lowers_the_workers_while_the_title_loads() {
        let mut p = Pace::new(12, 12);
        for _ in 0..20 {
            p.step(2000, false);
        }
        assert_eq!(p.step(2000, false), Tempo { workers: 10, low: false });
        assert_eq!(p.step(2000, true), Tempo { workers: 5, low: true });
        assert_eq!(p.step(6000, true), Tempo { workers: 3, low: true });
        // it still counts as loading for a few steps after the storage went quiet
        for _ in 0..LOADING_HOLD - 1 {
            assert!(p.step(2000, false).low);
        }
        assert_eq!(p.step(2000, false), Tempo { workers: 6, low: false });
        // a small pool keeps two workers going, a pool of one its one
        let mut small = Pace::new(4, 4);
        assert_eq!(small.step(4000, true).workers, 2);
        assert_eq!(Pace::new(1, 4).step(4000, true).workers, 1);
        // … and no pace goes beyond its pool
        let mut wide = Pace::new(6, 16);
        for _ in 0..20 {
            wide.step(0, false);
        }
        assert_eq!(wide.step(0, false).workers, 6);
    }

    #[test]
    fn the_headroom_is_its_least_while_the_memory_is_flat_or_shrinks() {
        let mut h = Headroom::default();
        assert_eq!(h.floor(0, MIN_HEADROOM, MIN_HEADROOM, false), MIN_HEADROOM);
        for s in 0..120u64 {
            h.note(s * 500, (10 << 30) - s * (8 << 20), true);
        }
        assert_eq!(h.floor(0, MIN_HEADROOM, MIN_HEADROOM, false), MIN_HEADROOM);
        assert_eq!(h.floor(0, 6 << 30, 6 << 30, false), 6 << 30, "the least is the caller's");
    }

    #[test]
    fn the_headroom_is_twice_the_biggest_rise_within_ten_seconds() {
        let mut h = Headroom::default();
        let gb = 1u64 << 30;
        // 3 GB within 6 s, as a world load commits
        for s in 0..=12u64 {
            h.note(s * 500, 10 * gb + s * gb / 4, true);
        }
        assert_eq!(h.floor(0, MIN_HEADROOM, MIN_HEADROOM, false), 6 * gb);
        // a slow rise does not add up beyond its ten seconds: 100 MB a second is 1 GB, under the least
        let mut slow = Headroom::default();
        for s in 0..300u64 {
            slow.note(s * 500, 10 * gb + s * (50 << 20), true);
        }
        assert_eq!(slow.floor(0, MIN_HEADROOM, MIN_HEADROOM, false), MIN_HEADROOM);
        assert_eq!(slow.floor(0, gb, gb, false), 2 * (20 * (50u64 << 20)), "twice the rise of the 20 steps in the span");
        assert_eq!(slow.floor(gb, gb, gb, false), gb + 2 * (20 * (50u64 << 20)), "above the line a run is stopped at");
    }

    #[test]
    fn the_headroom_forgets_a_rise_a_minute_after_its_ten_seconds() {
        let mut h = Headroom::default();
        let gb = 1u64 << 30;
        h.note(0, 10 * gb, true);
        h.note(2_000, 14 * gb, true);
        assert_eq!(h.floor(0, MIN_HEADROOM, MIN_HEADROOM, false), 8 * gb);
        for s in 1..=160u64 {
            let now = 2_000 + s * 500;
            h.note(now, 14 * gb, true);
            // (the rise was within the span until 9.5 s after it)
            let kept = now <= 2_000 + RISE_SPAN_MS - 500 + RISE_KEPT_MS;
            assert_eq!(h.floor(0, MIN_HEADROOM, MIN_HEADROOM, false), if kept { 8 * gb } else { MIN_HEADROOM }, "at {now} ms");
        }
    }

    #[test]
    fn the_headroom_sees_a_rise_across_a_pause_in_the_samples() {
        let mut h = Headroom::default();
        let gb = 1u64 << 30;
        h.note(0, 10 * gb, true);
        h.note(500, 10 * gb, true);
        // nothing asked for 40 s, and the process is 5 GB bigger
        h.note(40_500, 15 * gb, true);
        assert_eq!(h.floor(0, MIN_HEADROOM, MIN_HEADROOM, false), 10 * gb);
    }

    #[test]
    fn the_pipelines_of_a_group_bound_lately_go_before_the_others() {
        let mut b = Backlog::default();
        let (shadow, sky, post) = (0x51, 0x52, 0x53);
        b.push(100, false, false, shadow, "shadow 1");
        b.push(100, true, false, sky, "sky 1, known");
        b.push(100, false, true, post, "post 1");
        b.push(HEAVY_SHADER_BYTES, false, true, post, "post heavy");
        b.push(100, false, false, shadow, "shadow 2");
        b.push(100, false, false, 0, "no group");
        assert_eq!(b.waiting(), (6, 0));
        let take = |b: &mut Backlog<&'static str>, recent: &[u64]| b.pop(1, recent, |_| true, |_, _| true).map(|(p, _)| p);
        // post was bound last, shadow before it: post's first (its newest first), then shadow's
        let recent = [post, shadow];
        assert_eq!(b.pop(1, &recent, |_| true, |_, _| true), Some(("post heavy", Lane::Heavy)));
        b.done(Lane::Heavy);
        assert_eq!(take(&mut b, &recent), Some("post 1"));
        assert_eq!(take(&mut b, &recent), Some("shadow 2"));
        // a bind of another group since: that one first; an unknown group changes nothing
        assert_eq!(take(&mut b, &[0x99, sky, post, shadow]), Some("sky 1, known"));
        assert_eq!(take(&mut b, &recent), Some("shadow 1"));
        assert_eq!(b.waiting(), (1, 0));
        // nothing of those groups is left: the rest in its own order, and nothing is given twice
        assert_eq!(take(&mut b, &recent), Some("no group"));
        assert_eq!(take(&mut b, &recent), None);
        assert_eq!(b.waiting(), (0, 0));
    }

    #[test]
    fn a_pipeline_taken_through_its_lane_is_not_given_again_through_its_group() {
        let mut b = Backlog::default();
        b.push(100, false, false, 0x51, "a");
        b.push(HEAVY_SHADER_BYTES, false, true, 0x51, "heavy");
        b.push(100, false, true, 0x51, "c");
        let take = |b: &mut Backlog<&'static str>, recent: &[u64]| b.pop(4, recent, |_| true, |_, _| true).map(|(p, _)| p);
        assert_eq!(take(&mut b, &[]), Some("heavy"));
        assert_eq!(take(&mut b, &[]), Some("c"));
        assert_eq!(b.waiting(), (1, 0));
        assert_eq!(take(&mut b, &[0x51]), Some("a"));
        assert_eq!(take(&mut b, &[0x51]), None);
        // … nor the other way round, and heavy ones taken through their group leave the count right
        b.push(HEAVY_SHADER_BYTES, false, true, 0x52, "h1");
        b.push(HEAVY_SHADER_BYTES * 2, false, true, 0x52, "h2");
        b.push(100, false, false, 0, "g");
        assert_eq!(take(&mut b, &[0x52]), Some("h2"));
        assert_eq!(take(&mut b, &[0x52]), Some("h1"));
        b.done(Lane::Heavy);
        b.done(Lane::Heavy);
        assert_eq!(take(&mut b, &[]), Some("g"));
        assert_eq!(take(&mut b, &[]), None);
        assert_eq!(b.waiting(), (0, 0));
    }

    #[test]
    fn while_a_bind_waits_the_headroom_is_the_binds_floor_and_the_growth() {
        let (mut h, gb) = (Headroom::default(), 1u64 << 30);
        // the title loads, its memory flat: a bind that waits lets background compiles go
        // down to the floor of the compile it waits for; none waits any more, and the least is back
        h.note(0, 12 * gb, true);
        assert_eq!(h.floor(gb, CALM_HEADROOM, MIN_HEADROOM, true), gb);
        assert_eq!(h.floor(gb, CALM_HEADROOM, MIN_HEADROOM, false), MIN_HEADROOM);
        // … but not into a load that takes 3 GB within seconds: twice that stays above the floor
        h.note(4_000, 15 * gb, true);
        assert_eq!(h.floor(gb, CALM_HEADROOM, MIN_HEADROOM, true), 7 * gb);
        assert_eq!(h.floor(gb, CALM_HEADROOM, MIN_HEADROOM, false), 7 * gb);
        // calm, in the world: the bind's floor again once the rise is forgotten
        for s in 1..=160u64 {
            h.note(4_000 + s * 500, 15 * gb, false);
        }
        assert!(!h.loading());
        assert_eq!(h.floor(gb, CALM_HEADROOM, MIN_HEADROOM, true), gb);
        assert_eq!(h.floor(gb, CALM_HEADROOM, MIN_HEADROOM, false), CALM_HEADROOM);
    }

    #[test]
    fn parked_builds_of_a_group_bound_lately_come_back_before_bigger_ones() {
        let mut b = Backlog::default();
        b.park(900 << 10, false, 0x61, 0, "big, another pass");
        b.park(600 << 10, false, 0x62, 0, "this pass");
        b.park(500 << 10, false, 0x62, 0, "this pass, smaller");
        b.park(700 << 10, true, 0x61, 0, "asked, another pass");
        b.park(300 << 10, true, 0x62, 0, "asked, this pass");
        let take = |b: &mut Backlog<&'static str>, recent: &[u64], most: usize| b.pop(4, recent, |_| true, |len, _| len <= most).map(|(p, _)| p);
        // what a bind asked for first, of the pass bound last first; then the same among the others
        assert_eq!(take(&mut b, &[0x62, 0], 1 << 20), Some("asked, this pass"));
        assert_eq!(take(&mut b, &[0x62, 0], 1 << 20), Some("asked, another pass"));
        assert_eq!(take(&mut b, &[0x62, 0], 1 << 20), Some("this pass"));
        // only what fits: the smaller one of the pass, not the big one of another
        assert_eq!(take(&mut b, &[0x62], 550 << 10), Some("this pass, smaller"));
        assert_eq!(take(&mut b, &[0x62], 550 << 10), None);
        assert_eq!(take(&mut b, &[0x62], 1 << 20), Some("big, another pass"));
        assert!(!b.has_parked());
    }

    #[test]
    fn background_reserves_the_host_cost_of_unbacked_title_memory() {
        const MIB: u64 = 1 << 20;
        let budget = 8192 * MIB;
        let backed = (4095 + 1627 + 715) * MIB;
        let mut h = Headroom::empty();
        for (commit, growth, low, high) in [(15637, 2577, 7800, 8000), (15937, 1963, 7300, 7500)] {
            let reserve = TitleMemory { backed, commit: commit * MIB }.host_reserve(budget);
            h.note(0, (commit - growth / 2) * MIB, false);
            h.note(500, commit * MIB, false);
            let floor = h.title_floor(1024 * MIB, CALM_HEADROOM, MIN_HEADROOM, false, reserve) / MIB;
            assert!((low..=high).contains(&floor), "floor {floor} MiB");
            assert!((4100..=4400).contains(&(reserve / MIB)));
            assert_eq!(h.title_floor(1024 * MIB, CALM_HEADROOM, MIN_HEADROOM, true, reserve), h.floor(1024 * MIB, CALM_HEADROOM, MIN_HEADROOM, true));
            h = Headroom::empty();
        }
    }

    #[test]
    fn the_title_reserve_reaches_the_calm_floor_as_the_budget_fills() {
        let budget = 8 << 30;
        assert_eq!(TitleMemory::default().host_reserve(budget), budget);
        let mut h = Headroom::empty();
        h.note(60_000, 20 << 30, false);
        for backed in [budget - (64 << 20), budget, budget + (1 << 30)] {
            let reserve = TitleMemory { backed, commit: 20 << 30 }.host_reserve(budget);
            assert_eq!(h.title_floor(1 << 30, CALM_HEADROOM, MIN_HEADROOM, false, reserve), CALM_HEADROOM);
        }
        assert_eq!(TitleMemory { backed: 6 << 30, commit: 15 << 30 }.host_reserve(8 << 30), 5 << 30);
        assert_eq!(TitleMemory { backed: 6 << 30, commit: 3 << 30 }.host_reserve(8 << 30), 2 << 30);
    }

    #[test]
    fn passive_promotion_prefers_the_exact_pipeline_without_bypassing_memory() {
        let mut b = Backlog::default();
        let promoted = b.push(100, false, false, 11, "promoted");
        b.push(100, true, true, 12, "hot");
        b.park(600 << 10, true, 12, 0, "actually waited");
        b.promote(promoted);
        assert_eq!(b.pop(4, &[12], |_| true, |_, _| true), Some(("actually waited", Lane::Parked)));
        assert_eq!(b.pop(4, &[12], |_| true, |_, _| true), Some(("promoted", Lane::Graphics)));

        b.park(700 << 10, false, 11, promoted, "promoted after parking");
        b.promote(promoted);
        assert_eq!(b.pop(4, &[12], |_| true, |_, asked| asked), Some(("hot", Lane::Known)));
        assert_eq!(b.pop(4, &[12], |_| true, |_, asked| asked), None);
        assert_eq!(b.pop(4, &[12], |_| true, |_, _| true), Some(("promoted after parking", Lane::Parked)));
        assert_eq!(b.pop(4, &[12], |_| true, |_, _| true), None);
    }

    #[test]
    fn the_title_loads_while_its_storage_runs_many_requests_or_bytes() {
        assert!(!loading(0, 0, 500));
        assert!(!loading(1, 1 << 20, 500), "a request now and then is not loading");
        assert!(loading(8, 0, 500));
        assert!(loading(1, 4 << 20, 500), "one big read is");
        assert!(!loading(8, 4 << 20, 1100));
    }

    #[test]
    fn the_headroom_is_smaller_once_the_title_has_not_loaded_for_a_minute() {
        let (mut h, gb) = (Headroom::default(), 1u64 << 30);
        let floor = |h: &Headroom| h.floor(gb, CALM_HEADROOM, MIN_HEADROOM, false);
        // from the start it counts as loading
        assert!(h.loading());
        assert_eq!(floor(&h), MIN_HEADROOM);
        for s in 0..=40u64 {
            h.note(s * 500, 12 * gb, s <= 20);
        }
        assert_eq!(floor(&h), MIN_HEADROOM, "10 s after its storage went quiet");
        for s in 41..=139u64 {
            h.note(s * 500, 12 * gb, false);
        }
        assert!(h.loading(), "59.5 s after");
        h.note(70_000, 12 * gb, false);
        assert!(!h.loading());
        assert_eq!(floor(&h), CALM_HEADROOM, "calm: background compiles go down to 1.5 GB available");
        // the world streams in 0.4 GB within seconds: twice that above the line a run is stopped at
        h.note(72_000, 12 * gb + 4 * gb / 10, false);
        assert_eq!(floor(&h), gb + 2 * (4 * gb / 10));
        // … and the title loads again: the bigger least at once
        h.note(72_500, 12 * gb + 4 * gb / 10, true);
        assert_eq!(floor(&h), MIN_HEADROOM);
        // a calm least over the loading one is the loading one
        assert_eq!(Headroom::default().floor(gb, 8 * gb, 2 * gb, false), 2 * gb);
    }
}
