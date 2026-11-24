//! GPU-owned textures in title memory, aliased as on the console.
//!
//! A console title places render targets, depth buffers and UAV textures at addresses of its
//! own memory and reuses that memory: transient targets of a frame share it, and it creates
//! several resource objects over the same address expecting one memory (seen on XDefiant: 16
//! placements at an occupied address in its first seconds, also of other formats and sizes).
//! A committed PC resource per placement breaks that – what the GPU writes through one never
//! shows in the other. So these textures are placed in *arenas*: D3D12 default heaps (video
//! memory) standing for ranges of the title's address space (32 MiB+), at the offset of their
//! address in it. Resources at one address then share memory as on the console (the same
//! description: the same PC layout, so the same texels). When the title requests the address
//! of a supported color texture, `title_textures` also mirrors writes from its CPU storage.
//!
//! Arenas cost video memory for whole ranges, and a placement straddling two made a third over
//! both while the two lived on with their resources: a retail game's streaming targets, packed
//! back to back over 3.5 GB of its addresses, made 15 GB of arenas in two seconds – past the
//! card's memory and the system's commit, and the device was removed. So the first choice is
//! the console's own model (`Pages`): a 64 KiB page of video memory for each 64 KiB page of
//! the title's address a target covers, and each target a reserved (tiled) resource whose tiles
//! are mapped onto the pages of its addresses. Targets at one address share pages, a page
//! exists once however many targets cover it, and what no target covers takes nothing.
//! Arenas remain for what cannot be tiled (an address not 64 KiB aligned, a resource the
//! driver does not make reserved: D32S8, MSAA): heaps of the placement's size, shared by
//! placements inside them, let go 2 s after their last resource (256 MiB ones: 2.8 GB of D32S8).

use std::collections::{BTreeSet, HashSet};
use std::ffi::c_void;
use std::sync::{Condvar, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

const SMALL: usize = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT as usize;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Placement {
    va: usize,
    host_end: usize,
    title_end: usize,
    align: usize,
}

impl Placement {
    fn new(va: usize, bytes: usize, align: usize, console_bytes: Option<u64>) -> Option<Self> {
        Some(Self {
            va,
            host_end: va.checked_add(bytes)?,
            title_end: va.checked_add(console_bytes.map_or(bytes, |b| b as usize))?,
            align,
        })
    }

    fn small(self) -> bool {
        self.align == SMALL
    }

    fn host_overlaps(self, other: Self) -> bool {
        self.va < other.host_end && other.va < self.host_end
    }

    fn title_overlaps(self, va: usize, end: usize) -> bool {
        self.va < end && va < self.title_end
    }
}

/// A heap of video memory for placements at title addresses `[base, base + size)`.
struct Arena<H> {
    device: usize,
    base: usize,
    size: usize,
    /// the heap's alignment (a placement that needs more does not go in)
    align: usize,
    compact: bool,
    heap: H,
    placements: Vec<Placement>,
    title_end: usize,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
    id: u64,
}

/// The arenas, apart from D3D12 (`H`: the heap; numbers in the tests).
struct Arenas<H> {
    list: Vec<Arena<H>>,
    next_id: u64,
}

struct NewArena {
    base: usize,
    size: usize,
    align: usize,
}

impl<H: Clone> Arenas<H> {
    const fn new() -> Self {
        Arenas { list: Vec::new(), next_id: 0 }
    }

    /// Arenas no resource used for `IDLE` go (their heap with the last reference).
    fn sweep(&mut self, now: std::time::Instant) -> Vec<Arena<H>> {
        self.list.extract_if(.., |a| a.placements.is_empty()
            && a.idle_since.is_some_and(|t| now.saturating_duration_since(t) >= IDLE)).collect()
    }

    fn overlaps_small(&self, device: usize, va: usize, end: usize) -> bool {
        self.overlaps(device, va, end, true)
    }

    fn overlaps(&self, device: usize, va: usize, end: usize, compact: bool) -> bool {
        self.list.iter().filter(|a| a.device == device && a.compact == compact && va < a.title_end && a.base < end).any(|a| {
            a.placements.iter().any(|&p| p.title_overlaps(va, end))
        })
    }

    fn selection(&self, device: usize, p: Placement, require_alias: bool) -> Result<Option<usize>, ()> {
        let fits = |a: &Arena<H>| {
            a.device == device && a.align >= p.align && a.base <= p.va
                && p.host_end <= a.base + a.size && (p.va - a.base) % p.align == 0
                && (!a.compact || p.title_end <= a.base + a.size)
                && !a.placements.iter().any(|&q| {
                    (p.small() || a.compact) && p.host_overlaps(q) && !p.title_overlaps(q.va, q.title_end)
                })
        };
        let mut alias = None;
        if p.small() || self.overlaps_small(device, p.va, p.title_end) {
            for (i, a) in self.list.iter().enumerate() {
                if a.device != device || p.va >= a.title_end || a.base >= p.title_end {
                    continue;
                }
                for &q in &a.placements {
                    if p.title_overlaps(q.va, q.title_end) {
                        if !p.host_overlaps(q) || alias.is_some_and(|prev| prev != i) {
                            return Err(());
                        }
                        alias = Some(i);
                    }
                }
            }
        }
        if let Some(i) = alias {
            if !fits(&self.list[i]) || (p.small() && !self.list[i].compact) {
                return Err(());
            }
            return Ok(Some(i));
        }
        if require_alias {
            return Err(());
        }
        Ok(self.list.iter().rposition(|a| a.compact == p.small() && fits(a)))
    }

    fn new_arena(p: Placement) -> Option<NewArena> {
        let align = p.align.max(PAGE);
        let base = p.va & !(align - 1);
        let size = p.host_end.checked_next_multiple_of(align)? - base;
        Some(NewArena { base, size, align })
    }

    fn insert(&mut self, device: usize, p: Placement, new: NewArena, heap: H) -> (H, usize, u64, bool) {
        let NewArena { base, size, align } = new;
        self.next_id += 1;
        self.list.push(Arena { device, base, size, align, compact: p.small(), heap, placements: Vec::new(), title_end: base, idle_since: None, id: self.next_id });
        self.join(self.list.len() - 1, p)
    }

    fn join(&mut self, i: usize, p: Placement) -> (H, usize, u64, bool) {
        let a = &mut self.list[i];
        a.placements.push(p);
        a.title_end = a.title_end.max(p.title_end);
        a.idle_since = None;
        (a.heap.clone(), p.va - a.base, a.id, a.compact)
    }

    #[cfg(test)]
    fn take(&mut self, device: usize, p: Placement, now: std::time::Instant, make: impl FnOnce(usize, usize, usize) -> Option<H>) -> Result<Option<(H, usize, u64, bool)>, ()> {
        self.sweep(now);
        if let Some(i) = self.selection(device, p, false)? {
            return Ok(Some(self.join(i, p)));
        }
        let Some(new) = Self::new_arena(p) else { return Ok(None) };
        Ok(make(new.base, new.size, new.align).map(|heap| self.insert(device, p, new, heap)))
    }

    /// A resource in arena `id` released.
    fn release(&mut self, id: u64, p: Placement, now: std::time::Instant) {
        if let Some(a) = self.list.iter_mut().find(|a| a.id == id) {
            if let Some(i) = a.placements.iter().position(|&q| p == q) {
                a.placements.swap_remove(i);
            }
            if p.title_end == a.title_end {
                a.title_end = a.placements.iter().map(|p| p.title_end).max().unwrap_or(a.base);
            }
            if a.placements.is_empty() {
                a.idle_since = Some(now);
            }
        }
    }

    fn bytes(&self) -> usize { self.list.iter().map(|a| a.size).sum() }
}

// D3D12 heaps are free-threaded
unsafe impl Send for Arenas<ID3D12Heap> {}

static ARENAS: Mutex<Arenas<ID3D12Heap>> = Mutex::new(Arenas::new());
#[derive(Clone, Copy, PartialEq, Eq)]
struct PlacementRange {
    device: usize,
    first_page: usize,
    last_page: usize,
}

impl PlacementRange {
    fn overlaps(self, other: Self) -> bool {
        self.device == other.device && self.first_page <= other.last_page && other.first_page <= self.last_page
    }
}

static IN_PLACEMENT: Mutex<Vec<PlacementRange>> = Mutex::new(Vec::new());
static PLACEMENT_FINISHED: Condvar = Condvar::new();

struct PendingPlacement(PlacementRange);

impl PendingPlacement {
    fn enter(device: usize, va: usize, end: usize) -> Self {
        let range = PlacementRange { device, first_page: va / PAGE, last_page: end.saturating_sub(1).max(va) / PAGE };
        let pending = IN_PLACEMENT.lock().unwrap();
        let mut pending = PLACEMENT_FINISHED.wait_while(pending, |all| all.iter().any(|r| range.overlaps(*r))).unwrap();
        pending.push(range);
        Self(range)
    }
}

impl Drop for PendingPlacement {
    fn drop(&mut self) {
        let mut pending = IN_PLACEMENT.lock().unwrap();
        let i = pending.iter().position(|r| *r == self.0).unwrap();
        pending.swap_remove(i);
        PLACEMENT_FINISHED.notify_all();
    }
}
/// resource → its arena's id and placement
static ARENA_OF: Mutex<Option<std::collections::HashMap<usize, (u64, Placement)>>> = Mutex::new(None);
/// bytes of every arena made
static MADE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// (bytes of the arenas in use, bytes of all arenas made) – `XCL_ACTIVITY`
pub fn arena_bytes() -> (usize, usize) {
    (ARENAS.lock().unwrap().bytes(), MADE_BYTES.load(std::sync::atomic::Ordering::Relaxed))
}
static PLACED_TEXTURES: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

#[cfg(test)]
pub(super) fn arena_of(resource: usize) -> Option<u64> {
    ARENA_OF.lock().unwrap().as_ref()?.get(&resource).map(|&(id, _)| id)
}

/// A texture backed by one of the aliasable D3D12 heaps.
pub(super) fn is_placed_texture(resource: usize) -> bool {
    PLACED_TEXTURES.lock().unwrap().as_ref().is_some_and(|textures| textures.contains(&resource))
}

pub(super) fn forget(resource: usize) {
    let placed = PLACED_TEXTURES.lock().unwrap().as_mut().is_some_and(|textures| textures.remove(&resource));
    if placed {
        forget_pages(resource);
        if let Some((id, p)) = ARENA_OF.lock().unwrap().as_mut().and_then(|m| m.remove(&resource)) {
            ARENAS.lock().unwrap().release(id, p, std::time::Instant::now());
        }
    }
}

/// Whether a console texture with these D3D12_RESOURCE_FLAGS is the GPU's (render target,
/// depth buffer, UAV).
pub fn gpu_owned(flags: u32) -> bool {
    flags & 0x7 != 0
}

/// A new arena's heap: `size` bytes of video memory for title addresses from `va`.
unsafe fn arena_heap(dev: &ID3D12Device, va: usize, size: usize, align: usize, prefer_local: bool) -> Option<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        // 64 KiB, or 4 MiB for MSAA targets
        Alignment: align as u64,
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let heap = super::residency::create_heap(dev, &desc, prefer_local).map_err(|e| {
        crate::xlog_warn!("D3D12.X: video-memory arena for {va:#x}: CreateHeap: {e}");
    }).ok()?;
    let made = MADE_BYTES.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: video-memory arena #{n} for title addresses {va:#x}..{:#x} ({made} bytes of arenas made so far)", va + size);
    }
    Some(heap)
}

/// The adapter's local video memory in use against its budget, for the log.
unsafe fn local_memory(dev: &ID3D12Device) -> String {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory4, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO};
    let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else { return String::new() };
    let Ok(adapter) = factory.EnumAdapterByLuid::<IDXGIAdapter3>(dev.GetAdapterLuid()) else { return String::new() };
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    if adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info).is_err() {
        return String::new();
    }
    format!(" (video memory {} of {} MiB budget in use)", info.CurrentUsage >> 20, info.Budget >> 20)
}

/// bytes of a page – of the title's addresses and of a tile
const PAGE: usize = 0x1_0000;
/// pages of a video-memory pool heap (64 MiB)
const POOL_PAGES: u32 = 1024;
/// a page no target uses any more is kept this long for the next target at its address (with
/// what it holds – a title may make a target again over memory it rendered to)
const IDLE: std::time::Duration = std::time::Duration::from_secs(2);
/// a pool tile given back is not reused before this (the GPU may still be at work on what the
/// title released)
const COOL: std::time::Duration = std::time::Duration::from_secs(2);

/// A pool tile: (pool heap, tile in it).
type Tile = (u32, u32);

struct Page {
    tile: Tile,
    /// targets using the page
    users: u32,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
}

/// A page of `Pages::pages` (its title address) or of `Pages::extra_pages` (its key there).
#[derive(Clone, Copy)]
enum PageKey {
    Own(usize),
    Extra(usize, usize),
}

/// The pages of `idle` – those whose last user went, oldest first, each with when – that no
/// resource has used for `IDLE` go back to the pools: out of their map, their tile to `cooling`.
/// An entry whose page was used again since (in use now, or idle again from a later moment, with
/// a later entry) is dropped. Only the queue is looked at: every page of both maps, looked at at
/// every placement, grew with the square of what a title streams.
fn sweep_idle(
    idle: &mut std::collections::VecDeque<(PageKey, std::time::Instant)>,
    pages: &mut std::collections::HashMap<usize, Page>,
    extra_pages: &mut std::collections::HashMap<(usize, usize), Page>,
    cooling: &mut std::collections::VecDeque<(Tile, std::time::Instant)>,
    now: std::time::Instant,
) {
    while idle.front().is_some_and(|&(_, since)| now.duration_since(since) >= IDLE) {
        let Some((key, since)) = idle.pop_front() else { break };
        let still_idle = |p: &Page| p.idle_since == Some(since);
        let page = match key {
            PageKey::Own(a) if pages.get(&a).is_some_and(still_idle) => pages.remove(&a),
            PageKey::Extra(va, i) if extra_pages.get(&(va, i)).is_some_and(still_idle) => extra_pages.remove(&(va, i)),
            _ => None,
        };
        if let Some(p) = page {
            cooling.push_back((p.tile, now));
        }
    }
}

struct Target {
    va: usize,
    /// (format, width, height, depth or array size, mips): for `report_growth`
    shape: (i32, u64, u32, u16, u16),
    console_bytes: Option<u64>,
    total: u32,
    mapped: bool,
    used: Vec<usize>,
    used_extra: Vec<(usize, usize)>,
}

impl Target {
    fn end(&self) -> usize {
        self.va.saturating_add(self.console_bytes.map_or(self.total as usize * PAGE, |b| b as usize))
    }
}

/// The title's pages of video memory for tiled textures (see the module's comment).
struct Pages {
    queue: ID3D12CommandQueue,
    /// signalled after each mapping; title queues wait for it before their next work
    fence: ID3D12Fence,
    value: u64,
    pools: Vec<ID3D12Heap>,
    system_pools: HashSet<u32>,
    free: [Vec<Tile>; 2],
    /// tiles given back, reusable after `COOL`
    cooling: std::collections::VecDeque<(Tile, std::time::Instant)>,
    /// title page address → its video memory
    pages: std::collections::HashMap<usize, Page>,
    /// extra / sub-64 KiB tiles keyed by `(va, tile_index)` so repeated placements at the same
    /// address share them without aliasing unrelated addresses
    extra_pages: std::collections::HashMap<(usize, usize), Page>,
    /// pages of both maps whose last user went, oldest first, with when (`sweep_idle`)
    idle: std::collections::VecDeque<(PageKey, std::time::Instant)>,
    /// resource → its mapping metadata and currently held pages / extra keys
    targets: std::collections::HashMap<usize, Target>,
    target_starts: BTreeSet<(usize, usize)>,
    max_target_span: usize,
    /// queue → the fence value it last waited for
    waited: std::collections::HashMap<usize, u64>,
}

// D3D12 objects are free-threaded
unsafe impl Send for Pages {}

/// device raw pointer → its pages (None inside: tiled tier < 2 on this device)
static PAGES: Mutex<std::collections::BTreeMap<usize, Option<Pages>>> = Mutex::new(std::collections::BTreeMap::new());

impl Pages {
    unsafe fn new(dev: &ID3D12Device) -> Option<Pages> {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).ok()?;
        // tier 2: unmapped tiles read as zero, writes to them go nowhere
        if o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
            crate::xlog_info!("D3D12.X: tiled resources tier {}: GPU-owned textures in arenas", o.TiledResourcesTier.0);
            return None;
        }
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).ok()?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
        crate::xlog_info!("D3D12.X: textures in title memory: tiled resources on 64 KiB pages of video memory, one per page of the title's");
        Some(Pages {
            queue,
            fence,
            value: 0,
            pools: Vec::new(),
            system_pools: HashSet::new(),
            free: Default::default(),
            cooling: std::collections::VecDeque::new(),
            pages: std::collections::HashMap::new(),
            extra_pages: std::collections::HashMap::new(),
            idle: std::collections::VecDeque::new(),
            targets: std::collections::HashMap::new(),
            target_starts: BTreeSet::new(),
            max_target_span: 0,
            waited: std::collections::HashMap::new(),
        })
    }

    fn overlapping_targets(&self, va: usize, end: usize) -> impl Iterator<Item = &Target> {
        let start = va.saturating_sub(self.max_target_span);
        self.target_starts.range((start, 0)..(end, 0)).filter_map(move |&(_, resource)| {
            let t = &self.targets[&resource];
            (va < t.end()).then_some(t)
        })
    }

    /// Pages no resource used for `IDLE` go back to the pools.
    fn sweep(&mut self, now: std::time::Instant) {
        sweep_idle(&mut self.idle, &mut self.pages, &mut self.extra_pages, &mut self.cooling, now);
    }

    fn available_tile(&mut self, local_only: bool) -> Option<Tile> {
        self.free[0].pop()
            .or_else(|| (!local_only).then(|| self.free[1].pop()).flatten())
            .or_else(|| {
                let i = self.cooling.iter().take_while(|(_, at)| at.elapsed() >= COOL)
                    .position(|(tile, _)| !local_only || !self.system_pools.contains(&tile.0))?;
                self.cooling.remove(i).map(|(tile, _)| tile)
            })
    }

    unsafe fn take_tile(&mut self, dev: &ID3D12Device, prefer_local: bool, allow_system: &mut bool) -> Option<Tile> {
        if let Some(tile) = self.available_tile(!*allow_system) {
            return Some(tile);
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_PAGES as u64 * PAGE as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            // every kind of texture (resource heap tier 2: the only flags for both)
            Flags: D3D12_HEAP_FLAG_NONE,
        };
        let heap = match super::residency::create_heap(dev, &desc, prefer_local) {
            Ok(heap) => heap,
            Err(e) => {
                if !*allow_system {
                    *allow_system = true;
                    if let Some(tile) = self.available_tile(false) {
                        return Some(tile);
                    }
                }
                crate::xlog_warn!("D3D12.X: no memory for more pages of title textures ({e}){}", local_memory(dev));
                return None;
            }
        };
        super::residency::manage_heap(&heap, prefer_local);
        let props = heap.GetDesc().Properties;
        let system = props.Type == D3D12_HEAP_TYPE_CUSTOM && props.MemoryPoolPreference == D3D12_MEMORY_POOL_L0;
        self.pools.push(heap);
        let h = self.pools.len() as u32 - 1;
        if system {
            self.system_pools.insert(h);
            *allow_system = true;
        }
        self.free[usize::from(system)].extend((1..POOL_PAGES).rev().map(|t| (h, t)));
        Some((h, 0))
    }

    /// Map a linear range of tiles starting at the given subresource, one call per pool heap.
    unsafe fn map(&self, res: &ID3D12Resource, subresource: u32, tiles: &[Tile]) {
        let mut heaps: Vec<u32> = tiles.iter().map(|t| t.0).collect();
        heaps.sort_unstable();
        heaps.dedup();
        for h in heaps {
            let ranges = heap_ranges(h, tiles);
            let flags: Vec<D3D12_TILE_RANGE_FLAGS> = ranges.iter().map(|r| if r.0 { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_SKIP }).collect();
            let offsets: Vec<u32> = ranges.iter().map(|r| r.1).collect();
            let counts: Vec<u32> = ranges.iter().map(|r| r.2).collect();
            let coord = D3D12_TILED_RESOURCE_COORDINATE { Subresource: subresource, ..Default::default() };
            let region = D3D12_TILE_REGION_SIZE { NumTiles: tiles.len() as u32, UseBox: false.into(), ..Default::default() };
            self.queue.UpdateTileMappings(
                res,
                1,
                Some(&coord),
                Some(&region),
                &self.pools[h as usize],
                ranges.len() as u32,
                Some(flags.as_ptr()),
                Some(offsets.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
    }

    /// Unmap every tile of `res` (`D3D12_TILE_RANGE_FLAG_NULL`) so its unshared tiles can return
    /// to the pool even while the title keeps the COM object alive.
    unsafe fn unmap(&mut self, res: &ID3D12Resource, total: u32) {
        let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
        let region = D3D12_TILE_REGION_SIZE { NumTiles: total, UseBox: false.into(), ..Default::default() };
        let flag = D3D12_TILE_RANGE_FLAG_NULL;
        self.queue.UpdateTileMappings(
            res,
            1,
            Some(&coord),
            Some(&region),
            None,
            1,
            Some(&flag),
            None,
            Some(&total),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }

    fn release_tiles(&mut self, used: Vec<usize>, used_extra: Vec<(usize, usize)>, now: std::time::Instant, immediate_cool: bool) {
        for a in used {
            if let Some(p) = self.pages.get_mut(&a) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    if immediate_cool {
                        if let Some(p) = self.pages.remove(&a) {
                            self.cooling.push_back((p.tile, now));
                        }
                    } else {
                        p.idle_since = Some(now);
                        self.idle.push_back((PageKey::Own(a), now));
                    }
                }
            }
        }
        for k in used_extra {
            if let Some(p) = self.extra_pages.get_mut(&k) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    if immediate_cool {
                        if let Some(p) = self.extra_pages.remove(&k) {
                            self.cooling.push_back((p.tile, now));
                        }
                    } else {
                        p.idle_since = Some(now);
                        self.idle.push_back((PageKey::Extra(k.0, k.1), now));
                    }
                }
            }
        }
    }

    /// Acquire/share all `total` tiles for `res` at `va` and map them on the `Pages` queue.
    unsafe fn bind(&mut self, dev: &ID3D12Device, res: &ID3D12Resource, va: usize, console_bytes: Option<u64>, total: u32) -> Option<(Vec<usize>, Vec<(usize, usize)>)> {
        let (first, own) = (total > 0).then(|| owned_pages(va, console_bytes, total as usize)).flatten()?;
        let now = std::time::Instant::now();
        self.sweep(now);
        let mut tiles = Vec::with_capacity(total as usize);
        let mut used = Vec::with_capacity(own);
        let mut used_extra = Vec::new();
        let prefer_local = super::residency::protected_texture(&res.GetDesc());
        let mut allow_system = !prefer_local;
        for i in 0..total as usize {
            let tile = if i < own {
                let a = first + i * PAGE;
                let tile = match self.pages.get_mut(&a) {
                    Some(p) => {
                        p.users += 1;
                        p.idle_since = None;
                        Some(p.tile)
                    }
                    None => self.take_tile(dev, prefer_local, &mut allow_system).inspect(|&t| {
                        self.pages.insert(a, Page { tile: t, users: 1, idle_since: None });
                    }),
                };
                if tile.is_some() {
                    used.push(a);
                }
                tile
            } else {
                let k = (va, i);
                let tile = match self.extra_pages.get_mut(&k) {
                    Some(p) => {
                        p.users += 1;
                        p.idle_since = None;
                        Some(p.tile)
                    }
                    None => self.take_tile(dev, prefer_local, &mut allow_system).inspect(|&t| {
                        self.extra_pages.insert(k, Page { tile: t, users: 1, idle_since: None });
                    }),
                };
                if tile.is_some() {
                    used_extra.push(k);
                }
                tile
            };
            let Some(t) = tile else { break };
            tiles.push(t);
        }
        if tiles.len() < total as usize {
            self.release_tiles(used, used_extra, now, true);
            return None;
        }
        self.map_all(dev, res, &tiles);
        Some((used, used_extra))
    }

    /// Map every tile of `res` (`tiles[i]`: overall tile i): a single-slice texture's packed mip
    /// tail at its own subresource (the PC lays it out apart from the standard mips), the rest
    /// from subresource 0.
    unsafe fn map_all(&mut self, dev: &ID3D12Device, res: &ID3D12Resource, tiles: &[Tile]) {
        let total = tiles.len() as u32;
        let mut packed = D3D12_PACKED_MIP_INFO::default();
        dev.GetResourceTiling(res, None, Some(&mut packed), None, None, 0, std::ptr::null_mut());
        let desc = res.GetDesc();
        let packed_start = packed.StartTileIndexInOverallResource;
        let single_slice = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D || desc.DepthOrArraySize == 1;
        if single_slice
            && packed.NumStandardMips != 0
            && packed.NumPackedMips != 0
            && packed_start > 0
            && packed_start < total
            && packed.NumTilesForPackedMips == total - packed_start
        {
            let (standard, tail) = tiles.split_at(packed_start as usize);
            self.map(res, 0, standard);
            self.map(res, u32::from(packed.NumStandardMips), tail);
        } else {
            self.map(res, 0, tiles);
        }
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }
}

/// `XCL_PAGES_REPORT=1`: each time the pages in use pass another 512 MiB, which targets hold
/// them – grouped by shape, with the tiles past their console size (`extra_pages`) apart.
fn report_growth(p: &Pages) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    const STEP: usize = 512 << 20;
    static NEXT: AtomicUsize = AtomicUsize::new(STEP);
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let bytes = (p.pages.len() + p.extra_pages.len()) * PAGE;
    let next = NEXT.load(Ordering::Relaxed);
    if bytes < next || !*ON.get_or_init(|| std::env::var("XCL_PAGES_REPORT").is_ok_and(|v| v == "1")) || NEXT.compare_exchange(next, (bytes / STEP + 1) * STEP, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let mut groups: std::collections::HashMap<(i32, u64, u32, u16, u16), (usize, usize, usize)> = std::collections::HashMap::new();
    for t in p.targets.values().filter(|t| t.mapped) {
        let g = groups.entry(t.shape).or_default();
        g.0 += 1;
        g.1 += t.used.len();
        g.2 += t.used_extra.len();
    }
    let mut groups: Vec<_> = groups.into_iter().collect();
    groups.sort_by_key(|(_, (_, own, extra))| std::cmp::Reverse(own + extra));
    let idle = p.pages.values().chain(p.extra_pages.values()).filter(|pg| pg.users == 0).count();
    let mb = |tiles: usize| (tiles * PAGE) >> 20;
    let top: Vec<String> = groups
        .iter()
        .take(8)
        .map(|((f, w, h, d, m), (n, own, extra))| format!("fmt {f} {w}x{h}x{d} {m} mips ×{n}: {} MB ({} MB past console size)", mb(own + extra), mb(*extra)))
        .collect();
    crate::xlog_info!(
        "D3D12.X: target pages {} MB: {} MB at title addresses, {} MB past console sizes, {} MB idle; {} targets mapped in {} shapes; most: {}",
        bytes >> 20,
        mb(p.pages.len()),
        mb(p.extra_pages.len()),
        mb(idle),
        p.targets.values().filter(|t| t.mapped).count(),
        groups.len(),
        top.join("; ")
    );
}

/// The ranges of one UpdateTileMappings call for pool heap `heap` over a resource whose overall
/// tile i goes to `tiles[i]`: (mapped, tile offset in the heap, count) – runs of consecutive
/// tiles of the heap, and skipped runs for tiles of other heaps.
fn heap_ranges(heap: u32, tiles: &[Tile]) -> Vec<(bool, u32, u32)> {
    let mut out: Vec<(bool, u32, u32)> = Vec::new();
    for &(h, t) in tiles {
        let mine = h == heap;
        match out.last_mut() {
            Some(r) if mine && r.0 && r.1 + r.2 == t => r.2 += 1,
            Some(r) if !mine && !r.0 => r.2 += 1,
            _ => out.push((mine, if mine { t } else { 0 }, 1)),
        }
    }
    out
}

/// 64 KiB page boundaries `P` with `va <= P < va + bytes`, capped at `total` tiles:
/// `(first_page, count)`. Because every `P` lies strictly inside `[va, va + bytes)`, two
/// non-overlapping ranges never share a page here even when `va` is only 4 KiB aligned.
fn owned_pages(va: usize, console_bytes: Option<u64>, total: usize) -> Option<(usize, usize)> {
    match console_bytes {
        Some(b) => {
            let bytes = usize::try_from(b).ok()?;
            let first = va.checked_next_multiple_of(PAGE)?;
            let end = va.saturating_add(bytes);
            let n = if first < end { (end - first).div_ceil(PAGE) } else { 0 };
            Some((first, n.min(total)))
        }
        None => (va % PAGE == 0).then_some((va, total)),
    }
}

unsafe fn supports_reserved_texture(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> bool {
    let mut format = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: desc.Format, ..Default::default() };
    if dev.CheckFeatureSupport(
        D3D12_FEATURE_FORMAT_SUPPORT,
        (&mut format as *mut D3D12_FEATURE_DATA_FORMAT_SUPPORT).cast(),
        std::mem::size_of_val(&format) as u32,
    ).is_ok() && format.Support2.0 & D3D12_FORMAT_SUPPORT2_TILED.0 == 0 {
        return false;
    }
    if desc.SampleDesc.Count <= 1 {
        return true;
    }
    let mut msaa = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
        Format: desc.Format,
        SampleCount: desc.SampleDesc.Count,
        Flags: D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_TILED_RESOURCE,
        NumQualityLevels: 0,
    };
    dev.CheckFeatureSupport(
        D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
        (&mut msaa as *mut D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS).cast(),
        std::mem::size_of_val(&msaa) as u32,
    ).is_err() || msaa.NumQualityLevels != 0
}

/// A texture at `va` as a reserved resource on the title's shared 64 KiB pages
/// (`console_bytes`: its size on the console; tiles past the owned page boundaries are shared by
/// `(va, tile_index)` so repeated placements at the same address do not leak). None: it cannot be.
#[allow(clippy::too_many_arguments)]
unsafe fn place_on_pages(
    dev: &ID3D12Device,
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    require_aligned: bool,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    if desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN || desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D {
        return None;
    }
    if (require_aligned || console_bytes.is_none()) && va % PAGE != 0 {
        return None;
    }
    if !supports_reserved_texture(dev, desc) {
        return None;
    }
    let mut g = PAGES.lock().unwrap();
    let pages = g.entry(dev.as_raw() as usize).or_insert_with(|| Pages::new(dev)).as_mut()?;
    // reserved textures have the PC's 64 KiB tiles
    let tiled = D3D12_RESOURCE_DESC { Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, ..*desc };
    let hr = super::real::dev_create_reserved_resource(this, &tiled as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        if require_aligned {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_info!("D3D12.X: GPU-owned texture at {va:#x} ({}x{}, format {}, {} samples) not reserved ({hr:#x}): in an arena", desc.Width, desc.Height, desc.Format.0, desc.SampleDesc.Count);
            }
        }
        return None;
    }
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?;
    let mut total = 0u32;
    dev.GetResourceTiling(res, Some(&mut total), None, None, None, 0, std::ptr::null_mut());
    let Some((used, used_extra)) = pages.bind(dev, res, va, console_bytes, total) else {
        drop(g);
        drop(windows::core::IUnknown::from_raw(raw));
        *ppv = std::ptr::null_mut();
        return None;
    };
    let shape = (desc.Format.0, desc.Width, desc.Height, desc.DepthOrArraySize, desc.MipLevels);
    let target = Target { va, shape, console_bytes, total, mapped: true, used, used_extra };
    pages.max_target_span = pages.max_target_span.max(target.end() - va);
    pages.target_starts.insert((va, raw as usize));
    pages.targets.insert(raw as usize, target);
    report_growth(pages);
    drop(g);
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(raw as usize);
    Some(hr)
}

/// Unmap a placed texture superseded by a placement with a different address range, returning its
/// unshared tiles (`extra_pages` and unshared tail pages) to the cooling pool immediately.
pub(super) unsafe fn unmap_superseded(res: &ID3D12Resource) {
    let key = res.as_raw() as usize;
    let mut g = PAGES.lock().unwrap();
    let now = std::time::Instant::now();
    for pages in g.values_mut().flatten() {
        let Some(t) = pages.targets.get_mut(&key) else { continue };
        if !t.mapped {
            return;
        }
        let total = t.total;
        let used = std::mem::take(&mut t.used);
        let used_extra = std::mem::take(&mut t.used_extra);
        t.mapped = false;
        pages.unmap(res, total);
        pages.release_tiles(used, used_extra, now, true);
        return;
    }
}

/// Re-map a previously unmapped placed texture when the newer placement shadowing it is released.
pub(super) unsafe fn remap_if_unmapped(res: &ID3D12Resource) -> bool {
    let mut dev: Option<ID3D12Device> = None;
    if res.GetDevice(&mut dev).is_err() {
        return false;
    }
    let Some(dev) = dev else { return false };
    let key = res.as_raw() as usize;
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return false };
    let Some((va, console_bytes, total, mapped)) = pages.targets.get(&key).map(|t| (t.va, t.console_bytes, t.total, t.mapped)) else {
        return false;
    };
    if mapped {
        return true;
    }
    let Some((used, used_extra)) = pages.bind(&dev, res, va, console_bytes, total) else {
        return false;
    };
    if let Some(t) = pages.targets.get_mut(&key) {
        t.used = used;
        t.used_extra = used_extra;
        t.mapped = true;
    }
    true
}

/// A resource released: its pages are no longer its.
fn forget_pages(resource: usize) {
    let mut g = PAGES.lock().unwrap();
    let now = std::time::Instant::now();
    for pages in g.values_mut().flatten() {
        let Some(t) = pages.targets.remove(&resource) else { continue };
        pages.target_starts.remove(&(t.va, resource));
        if t.mapped {
            pages.release_tiles(t.used, t.used_extra, now, false);
        }
        return;
    }
}

/// Before GPU work on `queue`: it waits for the pages mapped since its last work on that device.
pub(super) unsafe fn before_execute(queue: &ID3D12CommandQueue) {
    let mut dev: Option<ID3D12Device> = None;
    if queue.GetDevice(&mut dev).is_err() {
        return;
    }
    let Some(dev) = dev else { return };
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return };
    let key = queue.as_raw() as usize;
    if pages.waited.get(&key).copied().unwrap_or(0) < pages.value {
        let _ = queue.Wait(&pages.fence, pages.value);
        super::markers::queue_fence(key, "VRAM page mapping", &pages.fence, pages.value);
        pages.waited.insert(key, pages.value);
    }
}

#[cfg(test)]
pub(super) static VRAM_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
pub(super) fn page_bytes_for(device: usize) -> (usize, usize) {
    let g = PAGES.lock().unwrap();
    let Some(Some(p)) = g.get(&device) else { return (0, 0) };
    ((p.pages.len() + p.extra_pages.len()) * PAGE, p.pools.len() * POOL_PAGES as usize * PAGE)
}

/// (bytes of pages in use by tiled textures, bytes of pool heaps) – `XCL_ACTIVITY`
pub fn page_bytes() -> (usize, usize) {
    let g = PAGES.lock().unwrap();
    g.values().flatten().fold((0, 0), |(used, pools), p| {
        (used + (p.pages.len() + p.extra_pages.len()) * PAGE, pools + p.pools.len() * POOL_PAGES as usize * PAGE)
    })
}

/// Heaps of every kind of resource in one (resource heap tier 2) – else no arenas.
fn tier2(dev: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.ResourceHeapTier.0 >= 2 }
}

#[derive(Clone, Copy)]
pub(super) enum TextureInitial {
    Legacy(u32),
    Undefined,
}

const RT_DS: i32 = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0;

unsafe fn small_desc(
    dev: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    va: usize,
    console_bytes: Option<u64>,
    default_info: impl FnOnce() -> D3D12_RESOURCE_ALLOCATION_INFO,
) -> Option<(D3D12_RESOURCE_DESC, Placement)> {
    if va % SMALL != 0 || desc.SampleDesc.Count != 1 || desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN
        || desc.Flags.0 & RT_DS != 0
        || !matches!(desc.Dimension, D3D12_RESOURCE_DIMENSION_TEXTURE1D
            | D3D12_RESOURCE_DIMENSION_TEXTURE2D | D3D12_RESOURCE_DIMENSION_TEXTURE3D)
        || default_info().SizeInBytes > PAGE as u64
    {
        return None;
    }
    let desc = D3D12_RESOURCE_DESC { Alignment: SMALL as u64, ..*desc };
    let info = dev.GetResourceAllocationInfo(0, &[desc]);
    if info.Alignment != SMALL as u64 || info.SizeInBytes == 0 || info.SizeInBytes >= PAGE as u64 {
        return None;
    }
    Some((desc, Placement::new(va, info.SizeInBytes as usize, SMALL, console_bytes)?))
}

fn compact_group(p: Placement, console_bytes: Option<u64>) -> Option<(usize, usize)> {
    console_bytes?;
    let uma = crate::gpu::uma::get()?;
    let (base, bytes) = if uma.is_sparse(p.va) {
        uma.sparse_reservation(p.va)?
    } else {
        uma.committed_reservation(p.va)?
    };
    bounded_compact_page(p, base, bytes)
}

fn bounded_compact_page(p: Placement, base: usize, bytes: usize) -> Option<(usize, usize)> {
    let end = base.checked_add(bytes)?;
    let page = base & !(PAGE - 1);
    let limit = page.checked_add(PAGE)?;
    (end <= limit && base <= p.va && p.title_end.max(p.host_end) <= end).then_some((page, limit))
}

/// A sampled texture of the title at `va`, backed by compact placement or shared tiles.
#[allow(clippy::too_many_arguments)]
pub unsafe fn place_pages(
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    place_impl(this, va, desc, TextureInitial::Legacy(state), clear, console_bytes, riid, ppv, false)
}

/// A GPU-owned texture of the title at `va` (`desc`: the PC description, `console_bytes` its
/// size on the console): on the title's pages there, else in the arena for that address (None:
/// neither – the caller makes a resource of its own).
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn place(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, initial: TextureInitial, clear: *const c_void, console_bytes: Option<u64>, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    place_impl(this, va, desc, initial, clear, console_bytes, riid, ppv, true)
}

#[allow(clippy::too_many_arguments)]
unsafe fn place_impl(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, initial: TextureInitial, clear: *const c_void, console_bytes: Option<u64>, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void, allow_arena: bool) -> Option<HRESULT> {
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) {
        return None;
    }
    let dev10 = if matches!(initial, TextureInitial::Undefined) {
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS12).cast(),
            std::mem::size_of_val(&options) as u32).ok()?;
        if !options.EnhancedBarriersSupported.as_bool() {
            return None;
        }
        Some(dev.cast::<ID3D12Device10>().ok()?)
    } else {
        None
    };
    let info = std::cell::OnceCell::new();
    let default_info = || *info.get_or_init(|| dev.GetResourceAllocationInfo(0, &[*desc]));
    let candidate = small_desc(dev, desc, va, console_bytes, default_info);
    let end = console_bytes.and_then(|bytes| va.checked_add(bytes as usize))
        .unwrap_or_else(|| {
            let bytes = default_info().SizeInBytes;
            va.saturating_add(if bytes == u64::MAX { PAGE } else { (bytes as usize).max(PAGE) })
        });
    let device = this as usize;
    let group = candidate.and_then(|(_, p)| compact_group(p, console_bytes));
    let _placement = PendingPlacement::enter(device, va, end.max(candidate.map_or(end, |(_, p)| p.host_end)));
    let (has_compact_alias, legacy_in_group) = {
        let arenas = ARENAS.lock().unwrap();
        (arenas.overlaps_small(device, va, end),
            group.is_some_and(|(base, limit)| arenas.overlaps(device, base, limit, false)))
    };
    let tiled_in = |start, limit| PAGES.lock().unwrap().get(&device)
        .and_then(Option::as_ref).is_some_and(|pages| pages.overlapping_targets(start, limit).next().is_some());
    if has_compact_alias && tiled_in(va, end) {
        alias_conflict(va);
        return None;
    }
    let compact = candidate.filter(|_| has_compact_alias
        || (!legacy_in_group && group.is_some_and(|(base, limit)| !tiled_in(base, limit))));
    if let Some((small, p)) = compact {
        if let Some(hr) = place_in_arena(dev, &small, p, initial, dev10.as_ref(), clear, riid, ppv, has_compact_alias) {
            return Some(hr);
        }
    }
    let arena_placement = || {
        let info = default_info();
        let align = (info.Alignment as usize).max(PAGE);
        (info.SizeInBytes != 0 && info.SizeInBytes != u64::MAX && va % align == 0)
            .then(|| Placement::new(va, info.SizeInBytes as usize, align, console_bytes)).flatten()
    };
    if has_compact_alias {
        let placed = arena_placement().and_then(|p| place_in_arena(dev, desc, p, initial, dev10.as_ref(), clear, riid, ppv, true));
        if placed.is_none() {
            alias_conflict(va);
        }
        return placed;
    }
    if std::env::var_os("XCL_VRAM_ARENAS").is_none() {
        if let TextureInitial::Legacy(state) = initial {
            if let Some(hr) = place_on_pages(dev, this, va, desc, state, clear, console_bytes, allow_arena, riid, ppv) {
                return Some(hr);
            }
        }
    }
    if allow_arena {
        return arena_placement().and_then(|p| place_in_arena(dev, desc, p, initial, dev10.as_ref(), clear, riid, ppv, false));
    }
    None
}

fn alias_conflict(va: usize) {
    static CONFLICTS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = CONFLICTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n <= 8 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: compact texture alias conflict #{n} at {va:#x}: the live resources cannot share this placement, using the fallback");
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn place_in_arena(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC, p: Placement, initial: TextureInitial, dev10: Option<&ID3D12Device10>, clear: *const c_void, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void, preserve_alias: bool) -> Option<HRESULT> {
    let now = std::time::Instant::now();
    let device = dev.as_raw() as usize;
    let (expired, selected) = {
        let mut arenas = ARENAS.lock().unwrap();
        let expired = arenas.sweep(now);
        let selected = arenas.selection(device, p, preserve_alias).map(|i| i.map(|i| arenas.join(i, p)));
        (expired, selected)
    };
    drop(expired);
    let prefer_local = super::residency::protected_texture(desc);
    let (heap, offset, id, compact) = if let Some(existing) = selected.ok()? {
        existing
    } else {
        let new = Arenas::<ID3D12Heap>::new_arena(p)?;
        let heap = arena_heap(dev, new.base, new.size, new.align, prefer_local)?;
        ARENAS.lock().unwrap().insert(device, p, new, heap)
    };
    let hr = match initial {
        TextureInitial::Legacy(state) => {
            let hr = super::real::dev_create_placed_resource(dev.as_raw(), heap.as_raw() as usize, offset, desc as *const _ as usize, state as usize, clear as usize, riid, ppv);
            if hr < 0 || (*ppv).is_null() {
                ARENAS.lock().unwrap().release(id, p, now);
                return None;
            }
            hr
        }
        TextureInitial::Undefined => {
            let dev10 = dev10.unwrap();
            let desc = D3D12_RESOURCE_DESC1 {
                Dimension: desc.Dimension, Alignment: desc.Alignment, Width: desc.Width, Height: desc.Height,
                DepthOrArraySize: desc.DepthOrArraySize, MipLevels: desc.MipLevels, Format: desc.Format,
                SampleDesc: desc.SampleDesc, Layout: desc.Layout, Flags: desc.Flags, ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            let clear = (!clear.is_null()).then_some(clear.cast::<D3D12_CLEAR_VALUE>());
            let mut placed = dev10.CreatePlacedResource2(&heap, offset as u64, &desc, D3D12_BARRIER_LAYOUT_UNDEFINED, clear, None, &mut out);
            // a clear value is only a hint for fast clears: one the PC refuses for this resource
            // (a format its views cannot have) is dropped rather than the resource
            if let (Err(e), Some(c)) = (&placed, clear) {
                if e.code() == windows::Win32::Foundation::E_INVALIDARG {
                    static DROPPED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                        crate::xlog_warn!("D3D12.X: texture format {} flags {:#x} refused clear format {} – placed without a clear value", desc.Format.0, desc.Flags.0, (*c).Format.0);
                    }
                    placed = dev10.CreatePlacedResource2(&heap, offset as u64, &desc, D3D12_BARRIER_LAYOUT_UNDEFINED, None, None, &mut out);
                }
            }
            if let Err(e) = placed {
                ARENAS.lock().unwrap().release(id, p, now);
                return (!p.small() && !preserve_alias).then_some(e.code().0);
            }
            let Some(out) = out else {
                ARENAS.lock().unwrap().release(id, p, now);
                return Some(crate::util::E_FAIL);
            };
            super::real::adopt(out.as_raw(), super::real::Kind::Resource);
            super::residency::link(out.as_raw() as usize, heap.as_raw() as usize);
            *ppv = out.into_raw();
            crate::util::S_OK
        }
    };
    if !compact {
        super::residency::manage_heap(&heap, prefer_local);
    }
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    ARENA_OF.lock().unwrap().get_or_insert_with(std::collections::HashMap::new).insert(*ppv as usize, (id, p));
    Some(hr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const MIB: usize = 1 << 20;

    /// One call per heap: its tiles in runs, the other heaps' skipped.
    #[test]
    fn tile_ranges_per_heap() {
        let tiles = [(0, 5), (0, 6), (1, 0), (1, 1), (0, 9), (0, 10), (0, 11)];
        assert_eq!(heap_ranges(0, &tiles), [(true, 5, 2), (false, 0, 2), (true, 9, 3)]);
        assert_eq!(heap_ranges(1, &tiles), [(false, 0, 2), (true, 0, 2), (false, 0, 3)]);
        // shared pages out of order: runs break where the tiles do
        assert_eq!(heap_ranges(0, &[(0, 3), (0, 1), (0, 2)]), [(true, 3, 1), (true, 1, 2)]);
    }

    /// Idle pages go back after `IDLE`, and a sweep looks at the queue of idle pages alone: one
    /// used again stays, one idle again later goes when that is `IDLE` ago, and the sweeps of
    /// 20 000 placements beside 200 000 pages in use take no time to speak of (a look at every
    /// page at each of them: 4 000 000 000 looks).
    #[test]
    fn idle_pages_are_swept_from_their_queue() {
        use std::collections::{HashMap, VecDeque};
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut idle: VecDeque<(PageKey, Instant)> = VecDeque::new();
        let mut pages: HashMap<usize, Page> = (0..200_000usize).map(|i| (i * PAGE, Page { tile: (0, i as u32), users: 1, idle_since: None })).collect();
        let mut extra: HashMap<(usize, usize), Page> = HashMap::from([((7 * PAGE, 3), Page { tile: (1, 0), users: 1, idle_since: None })]);
        let mut cooling: VecDeque<(Tile, Instant)> = VecDeque::new();
        // what `release_tiles` and `bind` do to a page
        fn release(p: &mut Page, idle: &mut VecDeque<(PageKey, Instant)>, key: PageKey, when: Instant) {
            (p.users, p.idle_since) = (0, Some(when));
            idle.push_back((key, when));
        }
        fn use_again(p: &mut Page) {
            (p.users, p.idle_since) = (1, None);
        }
        // pages 1, 2, 3 idle at 0 ms and an extra page at 100 ms; page 2 is used again at
        // 500 ms; page 3 too, and is idle again at 1500 ms
        for a in [PAGE, 2 * PAGE, 3 * PAGE] {
            release(pages.get_mut(&a).unwrap(), &mut idle, PageKey::Own(a), at(0));
        }
        release(extra.get_mut(&(7 * PAGE, 3)).unwrap(), &mut idle, PageKey::Extra(7 * PAGE, 3), at(100));
        use_again(pages.get_mut(&(2 * PAGE)).unwrap());
        use_again(pages.get_mut(&(3 * PAGE)).unwrap());
        release(pages.get_mut(&(3 * PAGE)).unwrap(), &mut idle, PageKey::Own(3 * PAGE), at(1500));

        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(1900));
        assert!(cooling.is_empty() && pages.len() == 200_000 && idle.len() == 5, "nothing is idle for IDLE yet");
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2050));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1)], "page 1 alone: 2 is in use, 3 idle since 1500 ms");
        assert!(!pages.contains_key(&PAGE) && pages.contains_key(&(2 * PAGE)) && pages.contains_key(&(3 * PAGE)) && extra.len() == 1);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2150));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1), (1, 0)], "the extra page, idle since 100 ms");
        assert!(extra.is_empty() && pages.len() == 199_999);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(3499));
        assert!(pages.contains_key(&(3 * PAGE)));
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(3500));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1), (1, 0), (0, 3)]);
        assert!(idle.is_empty() && pages.len() == 199_998 && pages[&(2 * PAGE)].users == 1);

        // every placement sweeps: with nothing idle that is nothing, whatever is in use
        let t = Instant::now();
        for k in 0..20_000u64 {
            sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(4000 + k));
        }
        assert!(t.elapsed() < Duration::from_secs(2), "{:?} for 20 000 sweeps", t.elapsed());
        assert_eq!(pages.len(), 199_998);
    }

    /// Any 64 KiB page boundary inside `[va, va + bytes)` belongs to that range alone, so
    /// non-overlapping ranges never share an owned page even when only 4 KiB aligned.
    #[test]
    fn owned_pages_never_alias_disjoint_ranges() {
        assert_eq!(owned_pages(0x1_0000, Some(0x1_0000), 4), Some((0x1_0000, 1)));
        assert_eq!(owned_pages(0x1_0000, Some(0x1_4000), 4), Some((0x1_0000, 2)));
        assert_eq!(owned_pages(0x1_4000, Some(0x4000), 4), Some((0x2_0000, 0)));
        assert_eq!(owned_pages(0x1_4000, Some(0x1_0000), 4), Some((0x2_0000, 1)));
        assert_eq!(owned_pages(0x1_4000, Some(0x4_0000), 4), Some((0x2_0000, 4)));
        assert_eq!(owned_pages(0x1_4000, None, 4), None);
        // Exhaustively check 4 KiB-aligned disjoint pairs over 256 KiB.
        const K4: usize = 0x1000;
        for a in (0..16).map(|i| 0x10_0000 + i * K4) {
            for sa in [K4, 4 * K4, 16 * K4, 20 * K4] {
                let (fa, na) = owned_pages(a, Some(sa as u64), 8).unwrap();
                let pa: Vec<usize> = (0..na).map(|i| fa + i * PAGE).collect();
                for b in (0..16).map(|i| 0x10_0000 + i * K4) {
                    for sb in [K4, 4 * K4, 16 * K4, 20 * K4] {
                        if a + sa <= b || b + sb <= a {
                            let (fb, nb) = owned_pages(b, Some(sb as u64), 8).unwrap();
                            for j in 0..nb {
                                assert!(!pa.contains(&(fb + j * PAGE)), "disjoint [{a:#x}..{:#x}) and [{b:#x}..{:#x}) shared page", a + sa, b + sb);
                            }
                        }
                    }
                }
            }
        }
    }

    /// A placement's arena id, with heaps numbered as they are made.
    fn take(a: &mut Arenas<u32>, va: usize, bytes: usize, align: usize, now: Instant) -> u64 {
        let n = a.list.len() as u32;
        let p = Placement::new(va, bytes, align, None).unwrap();
        a.take(0, p, now, |_, _, _| Some(n)).unwrap().unwrap().2
    }

    #[test]
    fn compact_neighbors_and_overlaps_use_one_64k_heap() {
        let mut arenas = Arenas::new();
        let now = Instant::now();
        let base = 0x8_0000;
        let first = Placement::new(base + SMALL, 2 * SMALL, SMALL, None).unwrap();
        let (_, offset, id, compact) = arenas.take(0, first, now, |addr, size, align| {
            assert_eq!((addr, size, align), (base, PAGE, PAGE));
            Some(0)
        }).unwrap().unwrap();
        assert_eq!((offset, compact), (SMALL, true));
        for offset in [SMALL, 2 * SMALL, 3 * SMALL, PAGE - SMALL] {
            let p = Placement::new(base + offset, SMALL, SMALL, None).unwrap();
            let (_, actual, other, _) = arenas.take(0, p, now, |_, _, _| panic!("unexpected heap")).unwrap().unwrap();
            assert_eq!((actual, other), (offset, id));
        }
        assert_eq!(arenas.bytes(), PAGE);
    }

    #[test]
    fn compact_alias_cannot_split_live_content() {
        let mut arenas = Arenas::new();
        let now = Instant::now();
        let base = 0x8_0000;
        let small = Placement::new(base + PAGE - SMALL, SMALL, SMALL, None).unwrap();
        let (_, _, id, _) = arenas.take(0, small, now, |_, _, _| Some(0)).unwrap().unwrap();
        let bigger = Placement::new(base, 2 * PAGE, PAGE, None).unwrap();
        let straddling = Placement::new(small.va, 2 * SMALL, SMALL, None).unwrap();
        let title_straddles = Placement::new(small.va, SMALL, SMALL, Some((2 * SMALL) as u64)).unwrap();
        for p in [bigger, straddling, title_straddles] {
            assert!(arenas.take(0, p, now, |_, _, _| panic!("split alias")).is_err());
        }
        let page = Placement::new(base, PAGE, PAGE, None).unwrap();
        assert_eq!(arenas.take(0, page, now, |_, _, _| panic!("split alias")).unwrap().unwrap().2, id);
        arenas.release(id, page, now);
        arenas.release(id, small, now);
        assert!(arenas.selection(0, bigger, true).is_err(), "the required live alias disappeared");
        assert!(arenas.take(0, bigger, now, |_, _, _| Some(1)).unwrap().is_some());
    }

    #[test]
    fn placement_reservations_only_conflict_on_shared_device_pages() {
        let same_page = PendingPlacement::enter(0, SMALL, 2 * SMALL);
        let next_page = PendingPlacement::enter(0, PAGE, PAGE + SMALL);
        let other_device = PendingPlacement::enter(1, SMALL, 2 * SMALL);
        let within_first = PlacementRange { device: 0, first_page: 0, last_page: 0 };
        assert!(same_page.0.overlaps(within_first));
        assert!(!next_page.0.overlaps(within_first));
        assert!(!other_device.0.overlaps(within_first));
    }

    #[test]
    fn known_suballocations_share_the_enclosing_page_only_if_their_extent_fits() {
        let page = 8 * PAGE;
        let a = Placement::new(page + SMALL, SMALL, SMALL, None).unwrap();
        let b = Placement::new(page + 2 * SMALL, SMALL, SMALL, None).unwrap();
        assert_eq!(bounded_compact_page(a, a.va, SMALL), Some((page, page + PAGE)));
        assert_eq!(bounded_compact_page(b, b.va, SMALL), Some((page, page + PAGE)));
        assert_eq!(bounded_compact_page(a, a.va, PAGE), None);
        assert_eq!(bounded_compact_page(b, a.va, SMALL), None);
    }

    #[test]
    fn compact_host_overlap_cannot_turn_disjoint_title_ranges_into_aliases() {
        let mut arenas = Arenas::new();
        let now = Instant::now();
        let first = Placement::new(0x8_0000, 2 * SMALL, SMALL, Some(SMALL as u64)).unwrap();
        let second = Placement::new(first.va + SMALL, SMALL, SMALL, None).unwrap();
        let (_, _, id, _) = arenas.take(0, first, now, |_, _, _| Some(0)).unwrap().unwrap();
        let (_, _, other, _) = arenas.take(0, second, now, |_, _, _| Some(1)).unwrap().unwrap();
        assert_ne!(id, other);
        let covering = Placement::new(first.va, 2 * SMALL, SMALL, None).unwrap();
        assert!(arenas.take(0, covering, now, |_, _, _| panic!("split alias")).is_err());
    }

    #[test]
    fn an_arena_is_shared_at_its_addresses_and_otherwise_its_placements_size() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let first = take(&mut a, a0, 70 * MIB, PAGE, t);
        assert_eq!(a.bytes(), 70 * MIB, "the placement's size, not 256 MiB");
        // the same address, and a smaller placement inside: the same memory
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t), first);
        assert_eq!(take(&mut a, a0 + 4 * MIB, 8 * MIB, PAGE, t), first);
        // one straddling its end gets an arena of its own size; so does an MSAA target that needs 4 MiB
        let straddle = take(&mut a, a0 + 64 * MIB, 70 * MIB, PAGE, t);
        let msaa = take(&mut a, a0 + 8 * MIB, 8 * MIB, 4 * MIB, t);
        assert!(straddle != first && msaa != first && msaa != straddle);
        assert_eq!(a.bytes(), (70 + 70 + 8) * MIB);
        // the newest arena holding an address is the one placements there share
        assert_eq!(take(&mut a, a0 + 64 * MIB, 4 * MIB, PAGE, t), straddle);
    }

    #[test]
    fn an_arena_no_resource_uses_goes_after_two_seconds() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let p = Placement::new(a0, 70 * MIB, PAGE, None).unwrap();
        let id = take(&mut a, a0, 70 * MIB, PAGE, t);
        a.release(id, p, t);
        // placed again within 2 s: the same memory (with what it holds)
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t + Duration::from_secs(1)), id);
        a.release(id, p, t + Duration::from_secs(1));
        take(&mut a, a0 + 512 * MIB, MIB, PAGE, t + Duration::from_secs(3) + Duration::from_millis(1));
        assert_eq!(a.bytes(), MIB, "the idle arena went");
    }

    /// The retail game's pattern (HANDOFF): targets packed back to back over 3.5 GB of its
    /// addresses and placed again and again, each round at the last round's addresses or a third
    /// of a target further on, so placements straddle earlier ones. Main's 256 MiB arenas merged
    /// straddled arenas into growing unions while the old ones lived on (15 GB in 2 s).
    #[test]
    fn streamed_targets_stay_bounded() {
        const S: usize = 64 * MIB; // a 3840x2160 D32S8, rounded
        const K: usize = 8; // targets a round
        const SPAN: usize = 3584 * MIB;
        let (t0, a0) = (Instant::now(), 0x1_0000_0000usize);
        let mut a = Arenas::new();
        let mut live: std::collections::VecDeque<Vec<(u64, Placement)>> = std::collections::VecDeque::new();
        let mut bytes = Vec::new();
        for r in 0..400usize {
            let now = t0 + Duration::from_millis(500 * r as u64);
            let start = ((r / 2) * S / 3) % (SPAN - K * S) & !(PAGE - 1);
            live.push_back((0..K).map(|k| {
                let va = a0 + start + k * S;
                (take(&mut a, va, S, PAGE, now), Placement::new(va, S, PAGE, None).unwrap())
            }).collect());
            // a target lives three rounds (1.5 s)
            if live.len() > 3 {
                for (id, p) in live.pop_front().unwrap() {
                    a.release(id, p, now);
                }
            }
            bytes.push(a.bytes());
        }
        // live: 3 rounds of targets; idle and kept (< 2 s): 4 more
        let most = *bytes.iter().max().unwrap();
        assert!(most <= 7 * K * S, "{most} bytes of arenas");
        let early = *bytes[20..120].iter().max().unwrap();
        let late = *bytes[300..].iter().max().unwrap();
        assert!(late <= early, "arenas grow over time: {early} then {late} bytes");
    }

    #[test]
    fn compact_gpu_textures_share_an_arena_without_neighbor_crosstalk() {
        use std::mem::ManuallyDrop;
        use crate::gpu::texture::{self, Subresource};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory4};

        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(tier2) else { return };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 32, Height: 32, DepthOrArraySize: 1, MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let Some((_, p)) = small_desc(&dev, &desc, SMALL, None, || dev.GetResourceAllocationInfo(0, &[desc]))
                else { return };
            let stride = p.host_end - p.va;
            if SMALL + 2 * stride > PAGE {
                return;
            }
            let Some(uma) = crate::gpu::uma::get() else { return };
            let base = uma.reserve(PAGE, PAGE).expect("small texture reservation") as usize;
            let va = base + SMALL;
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let state = D3D12_RESOURCE_STATE_COMMON;
            let place_at = |va| {
                let mut raw = std::ptr::null_mut();
                let hr = place_pages(dev.as_raw(), va, &desc, state.0 as u32, std::ptr::null(), Some(stride as u64), riid, &mut raw);
                assert!(hr.is_some_and(|hr| hr >= 0) && !raw.is_null(), "small texture at {va:#x}: {hr:?}");
                let res = ID3D12Resource::from_raw(raw);
                assert_eq!(res.GetDesc().Alignment, SMALL as u64);
                res
            };
            let a = place_at(va);
            let b = place_at(va + stride);
            let alias = place_at(va);
            let id = arena_of(a.as_raw() as usize).unwrap();
            assert_eq!(arena_of(b.as_raw() as usize), Some(id));
            assert_eq!(arena_of(alias.as_raw() as usize), Some(id));
            assert_eq!(ARENAS.lock().unwrap().list.iter().find(|arena| arena.id == id).unwrap().heap.GetDesc().Alignment, PAGE as u64);

            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let activate = |before: Option<&ID3D12Resource>, after: &ID3D12Resource| {
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 { Aliasing: ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                        pResourceBefore: before.map_or_else(|| ManuallyDrop::new(None), |r| std::mem::transmute_copy(r)),
                        pResourceAfter: std::mem::transmute_copy(after),
                    }) },
                    ..Default::default()
                }]);
                list.Close().unwrap();
                super::super::title_textures::own_work(|| {
                    queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                    texture::wait(&dev, &queue).unwrap();
                });
            };
            for (res, byte) in [(&a, 0x4D), (&b, 0x71)] {
                activate(None, res);
                let pixels = Subresource { rows: 32, row_bytes: 128, depth: 1, data: vec![byte; 4096] };
                super::super::title_textures::own_work(|| texture::upload(&dev, &queue, res, &[pixels], state, state)).unwrap();
            }
            activate(Some(&a), &alias);
            for (res, byte) in [(&alias, 0x4D), (&b, 0x71)] {
                let got = super::super::title_textures::read_gpu_subresource(&dev, res, 0, state).unwrap();
                assert_eq!(got, vec![byte; 4096]);
            }
            drop((a, b, alias));
            assert!(uma.free(base));
        }
    }

    /// On the GPU: D32S8 targets 256 MiB apart take their own size of video memory, not an arena
    /// of 256 MiB each; depth and stencil survive a clear, CopyResource and a read of both planes;
    /// arenas no target uses go. Needs a hardware adapter (skips on WARP).
    #[test]
    fn d32s8_targets_take_their_own_size_of_video_memory() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let memory: IDXGIAdapter3 = adapter.cast().unwrap();
            let local = || {
                let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let _ = memory.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut i);
                i.CurrentUsage
            };
            let sample = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
            let tex = |w: u64, h: u32| D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: w, Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R32G8X24_TYPELESS, SampleDesc: sample, Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, ..Default::default() };
            let ds = D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 };
            let clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: ds } };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize, d: &D3D12_RESOURCE_DESC| {
                let mut p = std::ptr::null_mut();
                let hr = place(dev.as_raw(), va, d, TextureInitial::Legacy(D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32), &clear as *const _ as *const c_void, None, riid, &mut p);
                assert!(hr.is_some_and(|hr| hr >= 0) && !p.is_null(), "D32S8 target at {va:#x}");
                ID3D12Resource::from_raw(p)
            };

            // 14 4K targets over 3.5 GB of addresses
            let big = tex(3840, 2160);
            let size = dev.GetResourceAllocationInfo(0, &[big]).SizeInBytes as usize;
            let (used0, arenas0) = (local(), arena_bytes().0);
            let targets: Vec<ID3D12Resource> = (0..14).map(|k| place_at(0x40_0000_0000 + k * 256 * MIB, &big)).collect();
            let (used, arenas) = (local().saturating_sub(used0) as usize, arena_bytes().0 - arenas0);
            println!("14 D32S8 3840x2160 targets ({size} bytes each): {arenas} bytes of arenas, {used} bytes more local video memory");
            assert!(arenas <= 14 * size.next_multiple_of(PAGE), "{arenas} bytes of arenas");
            assert!(used < 14 * 256 * MIB, "{used} bytes of video memory");
            drop(targets);

            // clear one, copy it to another, read both planes back
            let small = tex(64, 64);
            let (a, b) = (place_at(0x48_0000_0000, &small), place_at(0x48_1000_0000, &small));
            let dsv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() }).unwrap();
            let dsv = dsv_heap.GetCPUDescriptorHandleForHeapStart();
            let view = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, ..Default::default() };
            dev.CreateDepthStencilView(&a, Some(&view), dsv);
            let mut layouts = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
            let mut total = 0u64;
            dev.GetCopyableFootprints(&small, 0, 2, 0, Some(layouts.as_mut_ptr()), None, None, Some(&mut total));
            let buffer = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: sample, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
            let mut readback: Option<ID3D12Resource> = None;
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).unwrap();
            let readback = readback.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let barrier = |r: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES| {
                let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: from, StateAfter: to };
                D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
            };
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.25, 0x5A, None);
            list.ResourceBarrier(&[barrier(&a, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE), barrier(&b, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.CopyResource(&b, &a);
            list.ResourceBarrier(&[barrier(&b, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            for (plane, layout) in layouts.iter().enumerate() {
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *layout } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&b), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: plane as u32 } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            // no event: returns when the fence has the value
            fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).unwrap();
            let bytes = std::slice::from_raw_parts(p as *const u8, total as usize);
            let row = |plane: usize, y: usize| layouts[plane].Offset as usize + y * layouts[plane].Footprint.RowPitch as usize;
            for y in [0, 31, 63] {
                let depth = f32::from_le_bytes(bytes[row(0, y)..row(0, y) + 4].try_into().unwrap());
                assert_eq!((depth, bytes[row(1, y)]), (0.25, 0x5A), "row {y}: depth, stencil");
            }
            readback.Unmap(0, None);
            drop((a, b));

            // 2 s later the arenas no target uses are let go: one new target's is all that is left
            std::thread::sleep(IDLE + Duration::from_millis(100));
            let last = place_at(0x50_0000_0000, &small);
            let small_size = dev.GetResourceAllocationInfo(0, &[small]).SizeInBytes as usize;
            assert_eq!(arena_bytes().0 - arenas0, small_size.next_multiple_of(PAGE));
            drop(last);
        }
    }
}
