//! The policy of background pipeline builds, apart from the threads and the driver that carry it
//! out (`pso_async`, `real::DRIVER_COMPILES`): what a driver compile may start within the memory
//! left, and which build a background worker takes next. Nothing here asks the system anything,
//! so its tests run on any host.

use std::collections::VecDeque;

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
/// its square, among these), and two of nearly one size differ by half. The estimate is one and
/// a half times the most measured: 1 GB for each MB of DXIL under the giant line (256 MB at
/// least); from there 600 MB times the square of its MB (2 GB at least); and from 4 MB, where
/// nothing was measured, 3.3 GB for each MB as before (a 6.69 MB program took a warm-up's
/// process to a 21.9 GB working set, the driver's mapped shader cache included). The estimates
/// before were 1.5 GB and 3.3 GB for each MB throughout: 2.5 to 5 times what the four giants
/// above took, so that none of them was admitted beside a game that left 4.6 GB available.
pub fn peak(dxil_len: usize) -> u64 {
    let len = dxil_len as u64;
    if dxil_len >= 4 * GIANT_DXIL {
        len * 3380
    } else if dxil_len >= GIANT_DXIL {
        (len * len * 600 >> 20).max(2 << 30)
    } else {
        (len * 1024).max(256 << 20)
    }
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

/// The pipelines made for later, in the order background workers take them: a parked one whose
/// compile fits now, the biggest first (what is biggest fits least often, and the game's own
/// memory only grows); then the heavy ones, the biggest first, on a share of the workers so the
/// rest keep going through the many small ones; then those of shaders an earlier run used, then
/// compute (a dispatch waits for its pipeline however long it takes), then graphics, each in the
/// order made.
pub struct Backlog<T> {
    /// (DXIL bytes, pipeline), smallest first
    parked: Vec<(usize, T)>,
    /// (console shader bytes, pipeline), smallest first
    heavy: Vec<(usize, T)>,
    known: VecDeque<T>,
    compute: VecDeque<T>,
    graphics: VecDeque<T>,
    /// workers building a heavy one now
    heavy_out: usize,
}

impl<T> Default for Backlog<T> {
    fn default() -> Self {
        Backlog { parked: Vec::new(), heavy: Vec::new(), known: VecDeque::new(), compute: VecDeque::new(), graphics: VecDeque::new(), heavy_out: 0 }
    }
}

fn insert_by_size<T>(list: &mut Vec<(usize, T)>, size: usize, item: T) {
    // (after its equals: of two the same size the one made first goes first)
    let at = list.partition_point(|(s, _)| *s < size);
    list.insert(at, (size, item));
}

impl<T> Backlog<T> {
    /// A pipeline the title made: `shader_bytes` of console shaders, `known` from an earlier run.
    pub fn push(&mut self, shader_bytes: usize, known: bool, compute: bool, item: T) {
        if shader_bytes >= HEAVY_SHADER_BYTES {
            insert_by_size(&mut self.heavy, shader_bytes, item);
        } else if known {
            self.known.push_back(item);
        } else if compute {
            self.compute.push_back(item);
        } else {
            self.graphics.push_back(item);
        }
    }

    /// A pipeline whose driver compile of `dxil_len` bytes did not fit: it waits for memory.
    pub fn park(&mut self, dxil_len: usize, item: T) {
        insert_by_size(&mut self.parked, dxil_len, item);
    }

    /// The next pipeline for one of `workers` background workers, and the lane it came from
    /// (`done` is told when it is built). `live`: it still wants building (parked ones that do not
    /// are dropped); `fits`: a driver compile of so many bytes of DXIL would start now.
    pub fn pop(&mut self, workers: usize, live: impl Fn(&T) -> bool, fits: impl Fn(usize) -> bool) -> Option<(T, Lane)> {
        self.parked.retain(|(_, p)| live(p));
        if let Some(at) = self.parked.iter().rposition(|(len, _)| fits(*len)) {
            return Some((self.parked.remove(at).1, Lane::Parked));
        }
        let rest = !self.known.is_empty() || !self.compute.is_empty() || !self.graphics.is_empty();
        if !self.heavy.is_empty() && (!rest || self.heavy_out < heavy_share(workers)) {
            self.heavy_out += 1;
            return self.heavy.pop().map(|(_, p)| (p, Lane::Heavy));
        }
        if let Some(p) = self.known.pop_front() {
            return Some((p, Lane::Known));
        }
        if let Some(p) = self.compute.pop_front() {
            return Some((p, Lane::Compute));
        }
        self.graphics.pop_front().map(|p| (p, Lane::Graphics))
    }

    /// A worker is done with what it took from `lane`.
    pub fn done(&mut self, lane: Lane) {
        if lane == Lane::Heavy {
            self.heavy_out = self.heavy_out.saturating_sub(1);
        }
    }

    /// (pipelines waiting for a worker, of them parked for memory)
    pub fn waiting(&self) -> (usize, usize) {
        (self.parked.len() + self.heavy.len() + self.known.len() + self.compute.len() + self.graphics.len(), self.parked.len())
    }

    pub fn has_parked(&self) -> bool {
        !self.parked.is_empty()
    }

    /// The smallest driver compile parked, in bytes of DXIL (0: none).
    pub fn least_parked(&self) -> usize {
        self.parked.first().map_or(0, |(len, _)| (*len).max(1))
    }
}

/// The workers that may build heavy pipelines at once while others wait: a quarter, one at least.
pub fn heavy_share(workers: usize) -> usize {
    workers.div_ceil(4).max(1)
}

/// Background workers for a PC of `cores` hardware threads: all but two, so the title's threads and
/// the ones a bind waits on always find a core; two at least, 32 at most.
pub fn background_workers(cores: usize) -> usize {
    cores.saturating_sub(2).clamp(2, 32)
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
            let estimate = peak(len);
            assert!(estimate >= measured * MB * 3 / 2, "{len} B: {} MB estimated, {measured} MB measured", estimate / MB);
            assert!(estimate <= measured * MB * 3, "{len} B: {} MB estimated is no longer near the {measured} MB measured", estimate / MB);
        }
        assert_eq!(peak(1000), 256 << 20);
        assert_eq!(peak(512 << 10), 512 << 20);
        assert_eq!(peak(GIANT_DXIL), 2 << 30);
        assert_eq!(peak(6 << 20), 6 * 3380 << 20);
        // never less for more DXIL
        let sizes = [0, 1000, 256 << 10, GIANT_DXIL - 1, GIANT_DXIL, 2 << 20, (4 << 20) - 1, 4 << 20, 27 << 20];
        assert!(sizes.windows(2).all(|w| peak(w[0]) <= peak(w[1])));
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
            let (p, lane) = b.pop(workers, |_| true, |_| true)?;
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
        b.push(100, false, false, "g1");
        b.push(100, false, true, "c1");
        b.push(HEAVY_SHADER_BYTES, false, true, "heavy-small");
        b.push(100, true, false, "k1");
        b.push(100, false, false, "g2");
        b.push(4 * HEAVY_SHADER_BYTES, false, true, "heavy-big");
        b.push(100, true, true, "k2");
        b.push(100, false, true, "c2");
        assert_eq!(b.waiting(), (8, 0));
        assert_eq!(drain(&mut b, 8), ["heavy-big", "heavy-small", "k1", "k2", "c1", "c2", "g1", "g2"]);
        assert_eq!(b.waiting(), (0, 0));
    }

    /// Only a share of the workers builds heavy pipelines while small ones wait: the rest go on
    /// with those; once nothing else waits, every worker takes heavy ones.
    #[test]
    fn heavy_pipelines_take_a_share_of_the_workers() {
        let mut b = Backlog::default();
        for name in ["h1", "h2", "h3", "h4"] {
            b.push(HEAVY_SHADER_BYTES, false, true, name);
        }
        b.push(100, false, true, "c1");
        b.push(100, false, true, "c2");
        let workers = 8;
        assert_eq!(heavy_share(workers), 2);
        let take = |b: &mut Backlog<&'static str>| b.pop(workers, |_| true, |_| true).unwrap();
        assert_eq!(take(&mut b), ("h1", Lane::Heavy));
        assert_eq!(take(&mut b), ("h2", Lane::Heavy));
        assert_eq!(take(&mut b), ("c1", Lane::Compute), "two workers hold heavy ones: the third takes a small one");
        b.done(Lane::Heavy);
        assert_eq!(take(&mut b), ("h3", Lane::Heavy), "a heavy one done frees its place");
        assert_eq!(take(&mut b), ("c2", Lane::Compute));
        assert_eq!(take(&mut b), ("h4", Lane::Heavy), "nothing else waits: no share");
        assert_eq!(heavy_share(1), 1);
        assert_eq!(heavy_share(22), 6);
    }

    /// A compile that does not fit is parked and the worker goes on: nothing waits behind it.
    /// It comes back ahead of everything once it fits, the biggest that fits first.
    #[test]
    fn a_parked_build_holds_nothing_up_and_returns_when_it_fits() {
        let mut b = Backlog::default();
        b.push(100, false, true, "c1");
        b.push(100, false, false, "g1");
        b.park(3 << 20, "giant");
        b.park(300 << 10, "mid");
        b.park(600 << 10, "big");
        assert_eq!(b.waiting(), (5, 3));
        assert!(b.has_parked());
        assert_eq!(b.least_parked(), 300 << 10);
        let none_fit = |b: &mut Backlog<&'static str>| b.pop(4, |_| true, |_| false).map(|(p, _)| p);
        assert_eq!(none_fit(&mut b), Some("c1"), "no memory for the parked ones: the others go on");
        assert_eq!(b.pop(4, |_| true, |len| len < 1 << 20), Some(("big", Lane::Parked)), "the biggest that fits");
        assert_eq!(b.pop(4, |_| true, |_| true), Some(("giant", Lane::Parked)));
        assert_eq!(b.pop(4, |_| true, |len| len < 100 << 10), Some(("g1", Lane::Graphics)));
        assert_eq!(none_fit(&mut b), None, "only a parked one left, and no memory for it: the worker waits");
        assert_eq!(b.pop(4, |_| true, |_| true), Some(("mid", Lane::Parked)));
        assert!(!b.has_parked());
        assert_eq!(b.least_parked(), 0);
    }

    /// A parked build a bind took over (or whose pipeline the title released) is dropped: it
    /// does not come back to a background worker.
    #[test]
    fn a_parked_build_something_else_took_is_dropped() {
        let mut b = Backlog::default();
        b.park(300 << 10, "taken by a bind");
        b.park(600 << 10, "still waiting");
        assert_eq!(b.pop(4, |p| *p != "taken by a bind", |_| false), None);
        assert_eq!(b.waiting(), (1, 1));
        assert_eq!(b.pop(4, |_| true, |_| true), Some(("still waiting", Lane::Parked)));
    }

    #[test]
    fn background_workers_leave_two_cores() {
        assert_eq!(background_workers(24), 22);
        assert_eq!(background_workers(8), 6);
        assert_eq!(background_workers(4), 2);
        assert_eq!(background_workers(2), 2);
        assert_eq!(background_workers(128), 32);
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
}
