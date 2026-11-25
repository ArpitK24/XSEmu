use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};

const LATE_EPOCHS: u64 = 8;
const MAX_EPOCH_SPANS: usize = 131_072;
const MAX_EPOCH_QUEUES: usize = 64;
const TOP: usize = 20;
const STRIDED_PARTITION: &str = " exclusive_partition=disabled(strided_sampling)";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Kind {
    List,
    Draw,
    Dispatch,
    Indirect,
    MeshCompute,
    MeshDraw,
    Transfer,
    Clear,
    Barrier,
}

impl Kind {
    const CLASSES: [Self; 8] = [
        Self::Draw,
        Self::Dispatch,
        Self::Indirect,
        Self::MeshCompute,
        Self::MeshDraw,
        Self::Transfer,
        Self::Clear,
        Self::Barrier,
    ];
}

const _: () = {
    assert!(Kind::List as usize == 0);
    let mut i = 0;
    while i < Kind::CLASSES.len() {
        assert!(Kind::CLASSES[i] as usize == i + 1);
        i += 1;
    }
};

pub(super) struct Span {
    pub kind: Kind,
    pub pso: usize,
    pub shaders: Option<(u64, u64)>,
    pub grid: Option<[u32; 3]>,
    pub sampled: bool,
    pub begin: u64,
    pub end: u64,
}

pub(super) struct Batch {
    pub frame: u64,
    pub queue: usize,
    pub ty: i32,
    pub hz: u64,
    pub spans: Vec<Span>,
    pub timestamps: u64,
    pub dropped: u64,
}

#[derive(Default)]
struct Queue {
    ty: Option<i32>,
    hz: u64,
    noncorrelated: bool,
    spans: Vec<Span>,
    timestamps: u64,
    dropped: u64,
    waits: u64,
}

impl Queue {
    fn note_type(&mut self, ty: i32) {
        if *self.ty.get_or_insert(ty) != ty {
            self.noncorrelated = true;
        }
    }

    fn accept(&mut self, batch: Batch, budget: &mut usize) {
        self.note_type(batch.ty);
        self.timestamps = self.timestamps.saturating_add(batch.timestamps);
        self.dropped = self.dropped.saturating_add(batch.dropped);
        if batch.spans.is_empty() {
            return;
        }
        if batch.hz == 0 || batch.queue == 0 {
            self.dropped = self.dropped.saturating_add(batch.spans.len() as u64);
            return;
        }
        if self.hz == 0 {
            self.hz = batch.hz;
        } else if self.hz != batch.hz {
            self.noncorrelated = true;
        }
        for span in batch.spans {
            if span.end < span.begin || *budget == MAX_EPOCH_SPANS {
                self.dropped = self.dropped.saturating_add(1);
            } else {
                *budget += 1;
                self.spans.push(span);
            }
        }
    }
}

#[derive(Default)]
struct Epoch {
    queues: BTreeMap<usize, Queue>,
    used_spans: usize,
}

#[derive(Default)]
struct Rejected {
    batches: u64,
    spans: u64,
    timestamps: u64,
    waits: u64,
}

impl Rejected {
    fn report(&self, reason: &str, frame: u64) {
        if self.batches != 0 || self.waits != 0 {
            crate::xlog_info!(
                "GPU timing: CPU epoch {frame} discarded {reason} results: batches={} spans={} timestamps={} explicit_waits={}",
                self.batches,
                self.spans,
                self.timestamps,
                self.waits
            );
        }
    }
}

#[derive(Default)]
struct Stats {
    current: u64,
    epochs: BTreeMap<u64, Epoch>,
    late: Rejected,
    ahead: Rejected,
    overflow: Rejected,
}

impl Stats {
    fn queue(
        &mut self,
        frame: u64,
        queue: usize,
    ) -> Result<(&mut Queue, &mut usize), &mut Rejected> {
        if frame < self.current.saturating_sub(LATE_EPOCHS) {
            return Err(&mut self.late);
        }
        if frame > self.current.saturating_add(1) {
            return Err(&mut self.ahead);
        }
        let epoch = self.epochs.entry(frame).or_default();
        if epoch.queues.len() == MAX_EPOCH_QUEUES && !epoch.queues.contains_key(&queue) {
            return Err(&mut self.overflow);
        }
        Ok((
            epoch.queues.entry(queue).or_default(),
            &mut epoch.used_spans,
        ))
    }

    fn record(&mut self, batch: Batch) {
        if batch.spans.is_empty() && batch.timestamps == 0 && batch.dropped == 0 {
            return;
        }
        match self.queue(batch.frame, batch.queue) {
            Ok((queue, budget)) => queue.accept(batch, budget),
            Err(rejected) => {
                rejected.batches = rejected.batches.saturating_add(1);
                rejected.timestamps = rejected.timestamps.saturating_add(batch.timestamps);
                rejected.spans = rejected
                    .spans
                    .saturating_add(batch.spans.len() as u64)
                    .saturating_add(batch.dropped);
            }
        }
    }

    fn wait(&mut self, frame: u64, queue: usize, ty: i32) {
        match self.queue(frame, queue) {
            Ok((queue, _)) => {
                queue.note_type(ty);
                queue.waits = queue.waits.saturating_add(1);
            }
            Err(rejected) => rejected.waits = rejected.waits.saturating_add(1),
        }
    }

    fn boundary(&mut self, frame: u64) -> BTreeMap<u64, Epoch> {
        self.current = self.current.max(frame);
        let cutoff = self.current.saturating_sub(LATE_EPOCHS);
        if self
            .epochs
            .first_key_value()
            .is_none_or(|(&first, _)| first >= cutoff)
        {
            return BTreeMap::new();
        }
        let retained = self.epochs.split_off(&cutoff);
        std::mem::replace(&mut self.epochs, retained)
    }
}

static STATS: LazyLock<Mutex<Stats>> = LazyLock::new(|| Mutex::new(Stats::default()));

pub(super) fn record(batch: Batch) {
    STATS.lock().unwrap().record(batch);
}

pub(super) fn wait(frame: u64, queue: usize, ty: i32) {
    STATS.lock().unwrap().wait(frame, queue, ty);
}

#[cfg(test)]
pub(super) fn waits_for_test(frame: u64, queue: usize) -> u64 {
    STATS
        .lock()
        .unwrap()
        .epochs
        .get(&frame)
        .and_then(|e| e.queues.get(&queue))
        .map_or(0, |q| q.waits)
}

#[cfg(test)]
pub(super) fn captures_for_test(queue: usize) -> (usize, u64, u64) {
    STATS
        .lock()
        .unwrap()
        .epochs
        .values()
        .filter_map(|epoch| epoch.queues.get(&queue))
        .fold((0, 0, 0), |(spans, timestamps, dropped), q| {
            (
                spans + q.spans.len(),
                timestamps + q.timestamps,
                dropped + q.dropped,
            )
        })
}

pub(super) fn boundary(frame: u64) {
    let (retired, late, ahead, overflow) = {
        let mut stats = STATS.lock().unwrap();
        let retired = stats.boundary(frame);
        (
            retired,
            std::mem::take(&mut stats.late),
            std::mem::take(&mut stats.ahead),
            std::mem::take(&mut stats.overflow),
        )
    };
    for (frame, epoch) in retired {
        report(frame, epoch);
    }
    late.report("late", frame);
    ahead.report("ahead-of-epoch", frame);
    overflow.report("queue-capacity", frame);
}

pub(super) fn shaders(pso: usize) -> Option<(u64, u64)> {
    let pso = super::pso_async::resolve_if_ready(pso);
    super::real::pso_shaders(pso)
        .or_else(|| super::dispatch_timing::profile_shader(pso).map(|cs| (cs, 0)))
}

#[derive(Debug, PartialEq, Eq)]
struct IntervalSummary {
    union: u64,
    envelope: u64,
    exclusive: [u64; Kind::CLASSES.len()],
    overlap: u64,
    residual: u64,
}

impl IntervalSummary {
    fn between_lists(&self) -> u64 {
        self.envelope - self.union
    }
}

fn partition(spans: &[Span], classes: bool) -> Option<IntervalSummary> {
    let mut events = Vec::with_capacity(2 * spans.len());
    let mut first = u64::MAX;
    let mut last = None;
    for span in spans {
        if span.kind == Kind::List {
            first = first.min(span.begin);
            last = Some(last.map_or(span.end, |end: u64| end.max(span.end)));
        } else if !classes {
            continue;
        }
        if span.begin < span.end {
            events.push((span.begin, span.kind, true));
            events.push((span.end, span.kind, false));
        }
    }
    let mut summary = IntervalSummary {
        union: 0,
        envelope: last? - first,
        exclusive: [0; Kind::CLASSES.len()],
        overlap: 0,
        residual: 0,
    };
    events.sort_unstable();
    let mut active = [0usize; Kind::CLASSES.len() + 1];
    let mut scopes = 0usize;
    let mut previous = events.first().map_or(first, |&(tick, ..)| tick);
    for (tick, kind, opening) in events {
        if active[Kind::List as usize] != 0 && tick > previous {
            let elapsed = tick - previous;
            summary.union += elapsed;
            match scopes {
                0 => summary.residual += elapsed,
                1 => {
                    let class = active[1..].iter().position(|&n| n != 0).unwrap();
                    summary.exclusive[class] += elapsed;
                }
                _ => summary.overlap += elapsed,
            }
        }
        let depth = &mut active[kind as usize];
        if opening {
            *depth += 1;
            scopes += usize::from(kind != Kind::List);
        } else {
            *depth -= 1;
            scopes -= usize::from(kind != Kind::List);
        }
        previous = tick;
    }
    Some(summary)
}

#[derive(Default)]
struct Weight {
    nanos: u128,
    count: u64,
    peak_known_grid: Option<(u128, [u32; 3])>,
}

type Pipeline = (Kind, usize, Option<(u64, u64)>);
type Weights = BTreeMap<Pipeline, Weight>;

fn add_sample(weights: &mut Weights, span: &Span, hz: u64) -> usize {
    if span.pso == 0 {
        return 1;
    }
    let weight = weights
        .entry((span.kind, span.pso, span.shaders))
        .or_default();
    let nanos = u128::from(span.end - span.begin) * 1_000_000_000 / u128::from(hz);
    weight.nanos = weight.nanos.saturating_add(nanos);
    weight.count = weight.count.saturating_add(1);
    if let Some(grid) = span.grid {
        weight.peak_known_grid = weight.peak_known_grid.max(Some((nanos, grid)));
    }
    0
}

fn sample_weights(spans: &[Span], hz: u64) -> (Weights, usize, usize) {
    let mut weights = Weights::new();
    let mut without_pso = 0;
    let mut samples: Vec<_> = spans
        .iter()
        .filter(|span| span.kind != Kind::List && span.sampled)
        .collect();
    samples.retain(|span| {
        if span.begin == span.end {
            without_pso += add_sample(&mut weights, span, hz);
            false
        } else {
            true
        }
    });
    samples.sort_unstable_by_key(|s| (s.begin, s.end, s.kind, s.pso));
    let mut dropped = 0;
    let mut start = 0;
    while start < samples.len() {
        let mut next = start + 1;
        let mut end = samples[start].end;
        while next < samples.len() && samples[next].begin < end {
            end = end.max(samples[next].end);
            next += 1;
        }
        if next == start + 1 {
            without_pso += add_sample(&mut weights, samples[start], hz);
        } else {
            dropped += next - start;
        }
        start = next;
    }
    (weights, dropped, without_pso)
}

fn ranked(weights: Weights) -> Vec<(Pipeline, Weight)> {
    let mut ranked = weights.into_iter().collect::<Vec<_>>();
    ranked.sort_unstable_by_key(|&(pipeline, ref w)| (Reverse(w.nanos), pipeline));
    ranked.truncate(TOP);
    ranked
}

fn list_timing(spans: &[Span], hz: u64, classes: bool) -> String {
    let mode = if classes { "" } else { STRIDED_PARTITION };
    let Some(s) = partition(spans, classes) else {
        return format!("list timing unavailable (no captured lists){mode}");
    };
    let ms = |ticks| ticks as f64 * 1000.0 / hz as f64;
    let timing = format!(
        "list_union_ms={:.3} elapsed_envelope_ms={:.3} between_list_gap_ms={:.3}",
        ms(s.union),
        ms(s.envelope),
        ms(s.between_lists()),
    );
    if !classes {
        return format!("{timing}{mode}");
    }
    let classes = Kind::CLASSES
        .into_iter()
        .zip(s.exclusive)
        .map(|(kind, ticks)| format!("{kind:?}:{:.3}", ms(ticks)))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{timing} exclusive_class_ms={{{classes}}} overlap_ms={:.3} residual_unclassified_ms={:.3}",
        ms(s.overlap),
        ms(s.residual)
    )
}

fn report(frame: u64, epoch: Epoch) {
    let classes = super::gpu_timing::classes_enabled();
    for (id, q) in epoch.queues {
        let ty = q.ty.unwrap_or(-1);
        if q.noncorrelated || q.hz == 0 || id == 0 {
            let mode = if classes { "" } else { STRIDED_PARTITION };
            crate::xlog_info!(
                "GPU timing: CPU epoch {frame} queue {id:#x} type={ty}: list timing unavailable (missing or non-correlated clock/identity){mode}, spans={} timestamps={} explicit_waits={} dropped_spans={}",
                q.spans.len(),
                q.timestamps,
                q.waits,
                q.dropped
            );
            continue;
        }
        let lists = q.spans.iter().filter(|s| s.kind == Kind::List).count();
        let (weights, overlap, samples_without_pso) = sample_weights(&q.spans, q.hz);
        let ranked_samples: u64 = weights.values().map(|w| w.count).sum();
        let timing = list_timing(&q.spans, q.hz, classes);
        crate::xlog_info!(
            "GPU timing: CPU epoch {frame} queue {id:#x} type={ty}: captured_lists={lists} captured_class_scopes={} {timing} timestamps={} explicit_waits={} dropped_spans={} ranked_samples={ranked_samples} overlapping_samples_discarded={overlap} samples_without_pso={samples_without_pso}",
            q.spans.len() - lists,
            q.timestamps,
            q.waits,
            q.dropped
        );
        if !weights.is_empty() {
            let top = ranked(weights)
                .into_iter()
                .map(|((kind, pso, shaders), w)| {
                    let shaders = shaders
                        .map_or_else(|| "unknown".into(), |(a, b)| format!("{a:016x}/{b:016x}"));
                    let grid = w.peak_known_grid.map_or_else(
                        || "unknown".into(),
                        |(_, [x, y, z])| format!("{x}x{y}x{z}"),
                    );
                    format!(
                        "{kind:?} pso={pso:#x} shaders={shaders} peak_known_grid={grid} sampled_gpu_ms={:.3} samples={}",
                        w.nanos as f64 / 1_000_000.0,
                        w.count
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            crate::xlog_info!(
                "GPU timing: CPU epoch {frame} queue {id:#x} top sampled GPU durations: {top}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(kind: Kind, begin: u64, end: u64) -> Span {
        Span {
            kind,
            pso: 0,
            shaders: None,
            grid: None,
            sampled: false,
            begin,
            end,
        }
    }

    #[test]
    fn partitions_captured_list_time_exactly_once() {
        let spans = [
            (Kind::List, 10, 20),
            (Kind::List, 18, 24),
            (Kind::List, 30, 38),
            (Kind::Draw, 8, 12),
            (Kind::Draw, 12, 16),
            (Kind::Dispatch, 14, 18),
            (Kind::Dispatch, 16, 19),
            (Kind::Indirect, 22, 32),
            (Kind::Draw, 36, 40),
        ]
        .map(|(kind, begin, end)| span(kind, begin, end));
        let summary = partition(&spans, true).unwrap();
        assert_eq!(summary.union, 22);
        assert_eq!(summary.between_lists(), 6);
        assert_eq!(summary.exclusive[Kind::Draw as usize - 1], 6);
        assert_eq!(summary.exclusive[Kind::Dispatch as usize - 1], 1);
        assert_eq!(summary.exclusive[Kind::Indirect as usize - 1], 4);
        assert_eq!(summary.overlap, 4);
        assert_eq!(summary.residual, 7);
        assert_eq!(
            summary.exclusive.iter().sum::<u64>() + summary.overlap + summary.residual,
            summary.union,
        );
    }

    #[test]
    fn queue_union_merges_overlap_and_touching_endpoints() {
        let spans = |intervals: &[(u64, u64)]| {
            intervals
                .iter()
                .map(|&(begin, end)| span(Kind::List, begin, end))
                .collect::<Vec<_>>()
        };
        let summary = |intervals: &[(u64, u64)]| {
            partition(&spans(intervals), true).map(|s| (s.union, s.envelope))
        };
        assert_eq!(
            summary(&[(19, 23), (3, 8), (7, 10), (1, 4), (10, 11), (4, 5)]),
            Some((14, 22)),
        );
        assert_eq!(summary(&[]), None);
        assert_eq!(summary(&[(u64::MAX, u64::MAX)]), Some((0, 0)),);
        assert_eq!(
            summary(&[(0, 5), (u64::MAX - 5, u64::MAX)]),
            Some((10, u64::MAX)),
        );
    }

    #[test]
    fn all_classes_are_distinct_even_without_ranked_samples() {
        let mut spans = vec![span(Kind::List, 0, 36)];
        spans.extend(Kind::CLASSES.into_iter().enumerate().map(|(n, kind)| {
            let begin = (n * (n + 1) / 2) as u64;
            span(kind, begin, begin + n as u64 + 1)
        }));
        let summary = partition(&spans, true).unwrap();
        assert_eq!(summary.exclusive, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            (summary.union, summary.overlap, summary.residual),
            (36, 0, 0)
        );
        assert!(sample_weights(&spans, 1_000).0.is_empty());
        assert!(partition(&spans[1..], true).is_none());
    }

    #[test]
    fn strided_reports_list_time_without_exclusive_class_claims() {
        let spans = [span(Kind::List, 0, 10), span(Kind::Draw, 2, 4)];
        let strided = list_timing(&spans, 1_000, false);
        assert!(strided.contains("list_union_ms=10.000"));
        assert!(strided.contains("exclusive_partition=disabled"));
        assert!(!strided.contains("exclusive_class_ms="));
        assert!(!strided.contains("residual_unclassified_ms="));

        let full = list_timing(&spans, 1_000, true);
        assert!(full.contains("list_union_ms=10.000"));
        assert!(full.contains("exclusive_class_ms={Draw:2.000"));
        assert!(full.contains("residual_unclassified_ms=8.000"));
        assert!(!full.contains("exclusive_partition=disabled"));

        let no_lists = list_timing(&spans[1..], 1_000, false);
        assert!(no_lists.contains("no captured lists"));
        assert!(no_lists.contains("exclusive_partition=disabled"));
    }

    #[test]
    fn keeps_late_results_in_their_own_epoch_until_the_window_closes() {
        let mut stats = Stats::default();
        assert!(stats.boundary(10).is_empty());
        stats.wait(2, 5, 2);
        stats.record(Batch {
            frame: 2,
            queue: 5,
            ty: 2,
            hz: 1_000,
            spans: vec![],
            timestamps: 0,
            dropped: 3,
        });
        stats.wait(1, 5, 2);
        stats.wait(11, 5, 2);
        stats.wait(12, 5, 2);
        assert_eq!(stats.late.waits, 1);
        assert_eq!(stats.ahead.waits, 1);
        assert!(stats.boundary(9).is_empty());
        let retired = stats.boundary(11);
        let queue = &retired[&2].queues[&5];
        assert_eq!((queue.waits, queue.dropped), (1, 3));
        assert_eq!(stats.epochs[&11].queues[&5].waits, 1);
        stats.record(Batch {
            frame: 2,
            queue: 5,
            ty: 2,
            hz: 1_000,
            spans: vec![],
            timestamps: 4,
            dropped: 2,
        });
        assert_eq!((stats.late.batches, stats.late.spans), (1, 2));
        assert_eq!(stats.late.timestamps, 4);
    }

    #[test]
    fn ranks_nonoverlapping_samples_by_duration_with_deterministic_ties() {
        let sample = |kind, pso, begin, end| Span {
            pso,
            sampled: true,
            ..span(kind, begin, end)
        };
        let samples = [
            sample(Kind::Indirect, 9, 10, 20),
            sample(Kind::Draw, 7, 0, 4),
            sample(Kind::Draw, 7, 4, 10),
            sample(Kind::Dispatch, 1, 30, 40),
            sample(Kind::Draw, 8, 40, 45),
            sample(Kind::Dispatch, 2, 43, 47),
            sample(Kind::Draw, 7, 45, 45),
            sample(Kind::MeshDraw, 0, 50, 100),
            sample(Kind::Draw, 8, 110, 120),
            sample(Kind::MeshDraw, 0, 115, 130),
            Span {
                pso: 7,
                ..span(Kind::Draw, 100, 1000)
            },
        ];
        let (weights, overlap, without_pso) = sample_weights(&samples, 1_000);
        assert_eq!((overlap, without_pso), (4, 1));
        let ranked = ranked(weights);
        assert_eq!(
            ranked
                .iter()
                .map(|(key, w)| (*key, w.nanos, w.count))
                .collect::<Vec<_>>(),
            [
                ((Kind::Draw, 7, None), 10_000_000, 3),
                ((Kind::Dispatch, 1, None), 10_000_000, 1),
                ((Kind::Indirect, 9, None), 10_000_000, 1),
            ]
        );
    }

    #[test]
    fn aggregates_the_pso_and_reports_the_longest_sample_with_a_known_grid() {
        let mut elapsed = 0;
        let samples =
            [(4, Some([2, 9, 1])), (6, Some([1, 2, 3])), (20, None)].map(|(ticks, grid)| {
                let begin = elapsed;
                elapsed += ticks;
                Span {
                    pso: 7,
                    shaders: Some((0x12, 0x34)),
                    grid,
                    sampled: true,
                    ..span(Kind::MeshCompute, begin, elapsed)
                }
            });
        let (weights, overlap, without_pso) = sample_weights(&samples, 1_000);
        assert_eq!((weights.len(), overlap, without_pso), (1, 0, 0));
        let weight = &weights[&(Kind::MeshCompute, 7, Some((0x12, 0x34)))];
        assert_eq!((weight.nanos, weight.count), (30_000_000, 3));
        assert_eq!(weight.peak_known_grid, Some((6_000_000, [1, 2, 3])));
    }

    #[test]
    fn bounds_an_epoch_across_queues_and_counts_every_drop() {
        let mut stats = Stats::default();
        stats.record(Batch {
            frame: 0,
            queue: 1,
            ty: 0,
            hz: 1_000,
            spans: (0..MAX_EPOCH_SPANS - 1)
                .map(|_| span(Kind::List, 0, 10))
                .collect(),
            timestamps: 2 * (MAX_EPOCH_SPANS - 1) as u64,
            dropped: 0,
        });
        stats.record(Batch {
            frame: 0,
            queue: 2,
            ty: 0,
            hz: 1_000,
            spans: vec![
                span(Kind::Clear, 10, 0),
                span(Kind::Barrier, 10, 11),
                span(Kind::Transfer, 10, 12),
            ],
            timestamps: 6,
            dropped: 3,
        });
        stats.record(Batch {
            frame: 0,
            queue: 2,
            ty: 0,
            hz: 0,
            spans: vec![],
            timestamps: 2,
            dropped: 0,
        });
        let epoch = &stats.epochs[&0];
        assert_eq!(epoch.used_spans, MAX_EPOCH_SPANS);
        assert_eq!(epoch.queues[&1].spans.len(), MAX_EPOCH_SPANS - 1);
        assert_eq!(epoch.queues[&2].spans.len(), 1);
        assert_eq!(epoch.queues[&2].dropped, 5);
        assert_eq!(
            epoch.queues[&1].timestamps,
            2 * (MAX_EPOCH_SPANS - 1) as u64
        );
        assert_eq!(epoch.queues[&2].timestamps, 8);
    }

    #[test]
    fn caps_ranked_entries_in_stable_order() {
        let weights = (0..25)
            .rev()
            .map(|pso| {
                (
                    (Kind::Draw, pso, None),
                    Weight {
                        nanos: 1,
                        count: 1,
                        peak_known_grid: None,
                    },
                )
            })
            .collect::<Weights>();
        let ranked = ranked(weights);
        assert_eq!(ranked.len(), TOP);
        assert_eq!(ranked.last().unwrap().0, (Kind::Draw, TOP - 1, None));
    }

    #[test]
    fn preserves_resolved_timestamp_count_when_epoch_queue_budget_is_full() {
        let mut stats = Stats::default();
        for id in 1..=MAX_EPOCH_QUEUES {
            stats.wait(0, id, 0);
        }
        stats.record(Batch {
            frame: 0,
            queue: MAX_EPOCH_QUEUES + 1,
            ty: 0,
            hz: 1_000,
            spans: vec![span(Kind::List, 0, 10)],
            timestamps: 2,
            dropped: 0,
        });
        assert_eq!(stats.epochs[&0].queues.len(), MAX_EPOCH_QUEUES);
        assert_eq!((stats.overflow.batches, stats.overflow.timestamps), (1, 2));
    }
}
