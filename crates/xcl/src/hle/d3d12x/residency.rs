//! D3D12 backing memory policy for the title's unified memory. Only fence-idle
//! allocations with complete use tracking can be evicted; D3D12 retains their
//! contents. New allocations can instead prefer non-local memory.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::{Interface, Result, PCWSTR};
use windows::Win32::Foundation::{CloseHandle, E_OUTOFMEMORY, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory4, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

const MIB: u64 = 1 << 20;
const POLL_EVERY: Duration = Duration::from_millis(200);
const LOG_EVERY: Duration = Duration::from_secs(30);

#[inline]
pub(super) fn eviction_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_VRAM_EVICT").is_some_and(|v| v == "1"))
}

#[derive(Clone, Copy, Debug)]
struct Heat {
    id: usize,
    bytes: u64,
    last_used: u64,
    in_flight: bool,
    resident: bool,
    high: bool,
}

fn percent(bytes: u64, percent: u64) -> u64 {
    bytes / 100 * percent + bytes % 100 * percent / 100
}

fn budget_mb(raw: &str) -> Option<u64> {
    raw.trim().parse::<u64>().ok().filter(|&v| v != 0).and_then(|v| v.checked_mul(MIB))
}

fn pick(heats: impl IntoIterator<Item = Heat>, usage: u64, budget: u64) -> Vec<usize> {
    if budget == 0 || usage <= percent(budget, 90) {
        return Vec::new();
    }
    let mut cold: Vec<_> = heats.into_iter().filter(|h| h.resident && !h.in_flight).collect();
    cold.sort_unstable_by_key(|h| (h.last_used, h.high, std::cmp::Reverse(h.bytes), h.id));
    let mut left = usage;
    cold.into_iter().take_while(|h| {
        if left <= percent(budget, 80) {
            return false;
        }
        left = left.saturating_sub(h.bytes);
        true
    }).map(|h| h.id).collect()
}

struct BudgetEvent {
    handle: HANDLE,
    cookie: u32,
    adapter: IDXGIAdapter3,
}

impl Drop for BudgetEvent {
    fn drop(&mut self) {
        unsafe {
            self.adapter.UnregisterVideoMemoryBudgetChangeNotification(self.cookie);
            let _ = CloseHandle(self.handle);
        }
    }
}

#[derive(Default)]
struct Budget {
    adapter: Option<IDXGIAdapter3>,
    event: Option<BudgetEvent>,
    local: DXGI_QUERY_VIDEO_MEMORY_INFO,
    non_local: DXGI_QUERY_VIDEO_MEMORY_INFO,
    usage: u64,
    polled: Option<Instant>,
}

impl Budget {
    unsafe fn new(dev: &ID3D12Device) -> Self {
        let mut budget = Self::default();
        budget.adapter = CreateDXGIFactory1::<IDXGIFactory4>().ok()
            .and_then(|factory| factory.EnumAdapterByLuid(dev.GetAdapterLuid()).ok());
        if let Some(adapter) = &budget.adapter {
            let Ok(handle) = CreateEventW(None, false, false, PCWSTR::null()) else { return budget };
            match adapter.RegisterVideoMemoryBudgetChangeNotificationEvent(handle) {
                Ok(cookie) => budget.event = Some(BudgetEvent { handle, cookie, adapter: adapter.clone() }),
                Err(_) => { let _ = CloseHandle(handle); }
            }
        }
        budget
    }

    unsafe fn poll(&mut self) {
        let Some(adapter) = &self.adapter else { return };
        let notified = self.event.as_ref().is_some_and(|e| WaitForSingleObject(e.handle, 0) == WAIT_OBJECT_0);
        let now = Instant::now();
        if !notified && self.polled.is_some_and(|last| now.duration_since(last) < POLL_EVERY) {
            return;
        }
        self.polled = Some(now);
        let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        if adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local).is_ok() {
            self.local = local;
            self.usage = local.CurrentUsage;
        }
        let _ = adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut self.non_local);
    }
}

type LastUse = Vec<(Arc<ID3D12Fence>, u64)>;

fn stamp(last: &mut LastUse, fence: &Arc<ID3D12Fence>, value: u64) {
    if let Some((_, previous)) = last.iter_mut().find(|(f, _)| Arc::ptr_eq(f, fence)) {
        *previous = value;
    } else {
        last.push((Arc::clone(fence), value));
    }
}

unsafe fn in_flight(last: &LastUse, completed: &mut HashMap<usize, u64>) -> bool {
    last.iter().any(|(f, value)| {
        let done = *completed.entry(f.as_raw() as usize).or_insert_with(|| f.GetCompletedValue());
        done == u64::MAX || done < *value
    })
}

struct Object {
    bytes: u64,
    local: bool,
    high: bool,
    evictable: bool,
    resident: bool,
    last_used: u64,
    last: LastUse,
    pins: u32,
    releasing: u32,
}

struct QueueFence {
    fence: Arc<ID3D12Fence>,
    value: u64,
}

struct State {
    budget: Budget,
    budget_override: Option<u64>,
    pending_local: u64,
    evicted_local: u64,
    objects: HashMap<usize, Object>,
    queues: HashMap<usize, QueueFence>,
    opaque_last: LastUse,
    opaque_used: u64,
    opaque_pending: u32,
    serial: u64,
    log: Tally,
}

struct Tally {
    at: Instant,
    evicted: u64,
    restored: u64,
    spilled: u64,
    spills: u64,
    protected_spills: u64,
    reported_protected_spills: u64,
    worst_restore: Duration,
    trim_checks: u64,
    trim_opaque: u64,
}

impl State {
    unsafe fn new(dev: &ID3D12Device) -> Self {
        static OVERRIDE: OnceLock<Option<u64>> = OnceLock::new();
        let budget_override = *OVERRIDE.get_or_init(|| std::env::var("XCL_VRAM_BUDGET_MB").ok().as_deref().and_then(budget_mb));
        Self {
            budget: Budget::new(dev),
            budget_override,
            pending_local: 0,
            evicted_local: 0,
            objects: HashMap::new(),
            queues: HashMap::new(),
            opaque_last: Vec::new(),
            opaque_used: 0,
            opaque_pending: 0,
            serial: 0,
            log: Tally {
                at: Instant::now(), evicted: 0, restored: 0, spilled: 0, spills: 0,
                protected_spills: 0, reported_protected_spills: 0,
                worst_restore: Duration::ZERO, trim_checks: 0, trim_opaque: 0,
            },
        }
    }

    fn limit(&self) -> u64 {
        self.budget_override.unwrap_or(self.budget.local.Budget)
    }

    unsafe fn track(&mut self, raw: usize, bytes: u64, local: bool, high: bool) {
        let previous = self.objects.insert(raw, Object {
            bytes, local, high, evictable: local, resident: true, last_used: self.serial, last: Vec::new(), pins: 0, releasing: 0,
        });
        if let Some(old) = previous.filter(|o| !o.resident) {
            self.evicted_local = self.evicted_local.saturating_sub(old.bytes);
        }
    }

    unsafe fn trim(&mut self, dev: &ID3D12Device, incoming: u64) -> bool {
        self.budget.poll();
        let budget = self.limit();
        let usage = self.budget.usage.saturating_add(self.pending_local).saturating_add(incoming);
        let soft = percent(budget, 90);
        if budget == 0 || usage <= soft {
            return true;
        }
        if incoming > soft {
            return false;
        }
        self.log.trim_checks += 1;
        let mut completed = HashMap::new();
        if self.opaque_pending != 0 || in_flight(&self.opaque_last, &mut completed) {
            self.log.trim_opaque += 1;
            return false;
        }
        let heats = self.objects.iter().filter(|(_, o)| o.evictable).map(|(&id, o)| Heat {
            id,
            bytes: o.bytes,
            last_used: o.last_used.max(self.opaque_used),
            in_flight: o.pins != 0 || o.releasing != 0 || in_flight(&o.last, &mut completed),
            resident: o.resident,
            high: o.high,
        });
        let ids = pick(heats, usage, budget);
        if !ids.is_empty() {
            let objects: Vec<_> = ids.iter().map(|&id| id as *mut c_void).collect();
            match (dev.vtable().Evict)(dev.as_raw(), objects.len() as u32, objects.as_ptr()).ok() {
                Ok(()) => {
                    for id in ids {
                        let o = self.objects.get_mut(&id).unwrap();
                        o.resident = false;
                        self.budget.usage = self.budget.usage.saturating_sub(o.bytes);
                        self.evicted_local += o.bytes;
                        self.log.evicted += 1;
                    }
                }
                Err(e) => crate::xlog_warn!("D3D12.X: Evict of {} allocations failed: {e}", ids.len()),
            }
        }
        self.budget.usage.saturating_add(self.pending_local).saturating_add(incoming) <= soft
    }

    unsafe fn restore(&mut self, dev: &ID3D12Device, ids: &[usize]) -> Result<()> {
        let evicted: Vec<_> = ids.iter().copied().filter(|id| self.objects.get(id).is_some_and(|o| !o.resident)).collect();
        if evicted.is_empty() {
            return Ok(());
        }
        let incoming = evicted.iter().map(|id| self.objects[id].bytes).sum::<u64>();
        self.trim(dev, incoming);
        let start = Instant::now();
        let objects: Vec<_> = evicted.iter().map(|&id| id as *mut c_void).collect();
        let make_resident = || (dev.vtable().MakeResident)(dev.as_raw(), objects.len() as u32, objects.as_ptr()).ok();
        let result = make_resident().or_else(|_| {
            self.budget.polled = None;
            self.trim(dev, incoming);
            make_resident()
        });
        self.log.worst_restore = self.log.worst_restore.max(start.elapsed());
        result?;
        for id in evicted {
            self.objects.get_mut(&id).unwrap().resident = true;
            self.log.restored += 1;
        }
        self.budget.usage = self.budget.usage.saturating_add(incoming);
        self.evicted_local = self.evicted_local.saturating_sub(incoming);
        Ok(())
    }

    fn report(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.log.at) < LOG_EVERY {
            return;
        }
        let row = format!(
            "D3D12.X: VRAM local {} / {} MiB (DXGI budget {} MiB), non-local {} / {} MiB; {} evicted, {} made resident, {} total spills ({} protected, {:.2} MiB), worst MakeResident {:.2} ms, opaque blocked {}/{} pressure trim checks",
            self.budget.local.CurrentUsage / MIB, self.limit() / MIB, self.budget.local.Budget / MIB,
            self.budget.non_local.CurrentUsage / MIB, self.budget.non_local.Budget / MIB,
            self.log.evicted, self.log.restored, self.log.spills, self.log.protected_spills, self.log.spilled as f64 / MIB as f64, self.log.worst_restore.as_secs_f64() * 1000.0,
            self.log.trim_opaque, self.log.trim_checks,
        );
        if self.log.protected_spills != self.log.reported_protected_spills {
            crate::xlog_warn!("{row}");
            self.log.reported_protected_spills = self.log.protected_spills;
        } else {
            crate::xlog_info!("{row}");
        }
        self.log.at = now;
        self.log.evicted = 0;
        self.log.restored = 0;
        self.log.worst_restore = Duration::ZERO;
        self.log.trim_checks = 0;
        self.log.trim_opaque = 0;
    }
}

#[derive(Default)]
struct ListUse {
    resources: HashSet<usize>,
    opaque: bool,
}

struct Link {
    heaps: Vec<usize>,
    generation: u64,
}

#[derive(Default)]
struct Registry {
    devices: HashMap<usize, State>,
    objects: HashMap<usize, (usize, u64)>,
    links: HashMap<usize, Link>,
    generation: u64,
    lists: HashMap<usize, ListUse>,
}

unsafe impl Send for Registry {}

static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
static RELEASED: Condvar = Condvar::new();

fn registry() -> &'static Mutex<Registry> {
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}

fn state<'a>(g: &'a mut Registry, dev: &ID3D12Device) -> &'a mut State {
    g.devices.entry(dev.as_raw() as usize).or_insert_with(|| unsafe { State::new(dev) })
}

pub(super) fn may_evict(flags: D3D12_HEAP_FLAGS) -> bool {
    const NO_SPILL: D3D12_HEAP_FLAGS = D3D12_HEAP_FLAGS(
        D3D12_HEAP_FLAG_ALLOW_DISPLAY.0 | D3D12_HEAP_FLAG_SHARED.0 | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER.0
        | D3D12_HEAP_FLAG_HARDWARE_PROTECTED.0 | D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH.0
        | D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT.0,
    );
    flags.0 & NO_SPILL.0 == 0
}

pub(super) fn could_spill(props: D3D12_HEAP_PROPERTIES, flags: D3D12_HEAP_FLAGS) -> bool {
    let gpu_only = props.Type == D3D12_HEAP_TYPE_DEFAULT
        || (props.Type == D3D12_HEAP_TYPE_CUSTOM && props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE);
    may_evict(flags) && gpu_only
}

pub(super) fn protected_texture(desc: &D3D12_RESOURCE_DESC) -> bool {
    desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
        && desc.Flags.0 & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) != 0
}

unsafe fn spill_properties(dev: &ID3D12Device, props: D3D12_HEAP_PROPERTIES, flags: D3D12_HEAP_FLAGS) -> Option<D3D12_HEAP_PROPERTIES> {
    if !could_spill(props, flags) || !local_memory(dev, props) {
        return None;
    }
    Some(D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
        ..props
    })
}

unsafe fn local_memory(dev: &ID3D12Device, props: D3D12_HEAP_PROPERTIES) -> bool {
    if props.CreationNodeMask > 1 {
        return false;
    }
    match props.Type {
        D3D12_HEAP_TYPE_CUSTOM => props.MemoryPoolPreference == D3D12_MEMORY_POOL_L1,
        D3D12_HEAP_TYPE_DEFAULT | D3D12_HEAP_TYPE_GPU_UPLOAD => {
            let node = if dev.GetNodeCount() == 1 { 0 } else { props.CreationNodeMask };
            dev.GetCustomHeapProperties(node, props.Type).MemoryPoolPreference == D3D12_MEMORY_POOL_L1
        }
        _ => false,
    }
}

fn allocate_with<T>(
    props: D3D12_HEAP_PROPERTIES, spill: D3D12_HEAP_PROPERTIES, fits_local: bool,
    mut create: impl FnMut(&D3D12_HEAP_PROPERTIES) -> Result<T>,
) -> Result<(T, bool)> {
    if !fits_local {
        if let Ok(v) = create(&spill) {
            return Ok((v, true));
        }
        return create(&props).map(|v| (v, false));
    }
    match create(&props) {
        Err(e) if e.code() == E_OUTOFMEMORY => create(&spill).map(|v| (v, true)).or(Err(e)),
        other => other.map(|v| (v, false)),
    }
}

pub(super) unsafe fn allocate<T>(
    dev: &ID3D12Device, props: D3D12_HEAP_PROPERTIES, flags: D3D12_HEAP_FLAGS, bytes: u64, prefer_local: bool,
    mut create: impl FnMut(&D3D12_HEAP_PROPERTIES) -> Result<T>,
) -> Result<(T, bool)> {
    let Some(spill) = (bytes != 0 && bytes != u64::MAX).then(|| spill_properties(dev, props, flags)).flatten() else {
        return create(&props).map(|v| (v, false));
    };
    let fits = {
        let mut g = registry().lock().unwrap();
        let s = state(&mut g, dev);
        if eviction_enabled() {
            s.trim(dev, bytes);
        } else {
            s.budget.poll();
        }
        let fits = prefer_local || s.limit() == 0
            || s.budget.usage.saturating_add(s.evicted_local).saturating_add(s.pending_local).saturating_add(bytes) <= percent(s.limit(), 90);
        s.pending_local = s.pending_local.saturating_add(bytes);
        fits
    };
    let made = allocate_with(props, spill, fits, create);
    let mut g = registry().lock().unwrap();
    let s = state(&mut g, dev);
    s.pending_local = s.pending_local.saturating_sub(bytes);
    if let Ok((_, spilled)) = &made {
        if *spilled {
            s.log.spilled = s.log.spilled.saturating_add(bytes);
            s.log.spills += 1;
            if prefer_local {
                s.log.protected_spills += 1;
                if s.log.spills <= 4 {
                    crate::xlog_warn!("D3D12.X: local allocation ran out of memory; spilled {bytes} protected bytes to non-local memory");
                }
            } else if s.log.spills <= 4 {
                crate::xlog_info!("D3D12.X: spilled {bytes} bytes to non-local memory");
            }
        } else {
            s.budget.usage = s.budget.usage.saturating_add(bytes);
        }
    }
    s.report();
    made
}

pub(super) unsafe fn track(dev: &ID3D12Device, raw: usize, props: D3D12_HEAP_PROPERTIES, bytes: u64, high: bool, evictable: bool, spilled: bool) {
    if !eviction_enabled() || bytes == 0 || bytes == u64::MAX {
        return;
    }
    let local = !spilled && local_memory(dev, props);
    let mut g = registry().lock().unwrap();
    g.generation += 1;
    let generation = g.generation;
    g.objects.insert(raw, (dev.as_raw() as usize, generation));
    let s = state(&mut g, dev);
    s.track(raw, bytes, local, high);
    let gpu_only = props.Type == D3D12_HEAP_TYPE_DEFAULT || props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE;
    s.objects.get_mut(&raw).unwrap().evictable = local && evictable && gpu_only;
}

pub(super) unsafe fn create_heap(dev: &ID3D12Device, desc: &D3D12_HEAP_DESC, prefer_local: bool) -> Result<ID3D12Heap> {
    let (heap, spilled) = allocate(dev, desc.Properties, desc.Flags, desc.SizeInBytes, prefer_local, |properties| {
        let d = D3D12_HEAP_DESC { Properties: *properties, ..*desc };
        let mut heap: Option<ID3D12Heap> = None;
        super::title_textures::own_work(|| dev.CreateHeap(&d, &mut heap))?;
        heap.ok_or_else(|| windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL))
    })?;
    if eviction_enabled() {
        super::real::adopt(heap.as_raw(), super::real::Kind::Heap);
        track(dev, heap.as_raw() as usize, desc.Properties, desc.SizeInBytes, prefer_local, false, spilled);
    }
    Ok(heap)
}

pub(super) fn manage_heap(heap: &ID3D12Heap, high: bool) {
    if !eviction_enabled() {
        return;
    }
    let raw = heap.as_raw() as usize;
    let mut g = registry().lock().unwrap();
    if let Some((device, _)) = g.objects.get(&raw).copied() {
        let o = g.devices.get_mut(&device).unwrap().objects.get_mut(&raw).unwrap();
        o.high |= high;
        o.evictable = o.local;
    }
}

pub(super) fn link(resource: usize, heap: usize) {
    if !eviction_enabled() {
        return;
    }
    let mut g = registry().lock().unwrap();
    if !g.objects.contains_key(&heap) {
        return;
    }
    if let Some(link) = g.links.get_mut(&resource) {
        if !link.heaps.contains(&heap) {
            link.heaps.push(heap);
        }
    } else {
        g.generation += 1;
        let generation = g.generation;
        g.links.insert(resource, Link { heaps: vec![heap], generation });
    }
}

pub(super) fn copy_links(dst: usize, src: usize) {
    let heaps = registry().lock().unwrap().links.get(&src).map(|l| l.heaps.clone());
    for heap in heaps.into_iter().flatten() {
        link(dst, heap);
    }
}

pub(super) fn release(raw: usize, f: impl FnOnce() -> u32) -> u32 {
    if !eviction_enabled() {
        return f();
    }
    let Some(lock) = REGISTRY.get() else { return f() };
    let (owner, link) = {
        let mut g = lock.lock().unwrap();
        let owner = g.objects.get(&raw).copied();
        if let Some((dev, _)) = owner {
            g.devices.get_mut(&dev).unwrap().objects.get_mut(&raw).unwrap().releasing += 1;
        }
        (owner, g.links.get(&raw).map(|l| l.generation))
    };
    let n = f();
    if owner.is_none() && link.is_none() {
        return n;
    }
    let mut g = lock.lock().unwrap();
    if let Some((dev, _)) = owner.filter(|o| Some(o) == g.objects.get(&raw)) {
        let s = g.devices.get_mut(&dev).unwrap();
        if n == 0 {
            let o = s.objects.remove(&raw).unwrap();
            if o.resident && o.local {
                s.budget.usage = s.budget.usage.saturating_sub(o.bytes);
            } else if !o.resident {
                s.evicted_local = s.evicted_local.saturating_sub(o.bytes);
            }
            g.objects.remove(&raw);
        } else {
            s.objects.get_mut(&raw).unwrap().releasing -= 1;
        }
    }
    if n == 0 && link.is_some() && link == g.links.get(&raw).map(|l| l.generation) {
        g.links.remove(&raw);
    }
    RELEASED.notify_all();
    n
}

pub(super) fn note_list_resources(list: usize, resources: impl IntoIterator<Item = usize>) {
    if eviction_enabled() {
        registry().lock().unwrap().lists.entry(list).or_default().resources.extend(resources.into_iter().filter(|&r| r != 0));
    }
}

pub(super) fn note_list_untracked(list: usize) {
    if eviction_enabled() {
        registry().lock().unwrap().lists.entry(list).or_default().opaque = true;
    }
}

pub(super) fn list_reset(list: usize) {
    if !eviction_enabled() {
        return;
    }
    if let Some(lock) = REGISTRY.get() {
        lock.lock().unwrap().lists.remove(&list);
    }
}

pub(super) fn queue_released(queue: usize) {
    if !eviction_enabled() {
        return;
    }
    if let Some(lock) = REGISTRY.get() {
        for s in lock.lock().unwrap().devices.values_mut() {
            s.queues.remove(&queue);
        }
    }
}

pub(super) struct Submission {
    device: ID3D12Device,
    ids: Vec<usize>,
    opaque: bool,
}

pub(super) unsafe fn before_submit(queue: *mut c_void, lists: &[*mut c_void]) -> Option<Submission> {
    if !eviction_enabled() {
        return None;
    }
    before_use(queue, |g| {
        let mut resources = HashSet::new();
        for list in lists {
            match g.lists.get(&(*list as usize)) {
                Some(use_) if !use_.opaque => {
                    if use_.resources.iter().any(|r| !g.links.contains_key(r) && !g.objects.contains_key(r)) {
                        return None;
                    }
                    resources.extend(use_.resources.iter().copied());
                }
                _ => return None,
            }
        }
        Some(resources)
    }).unwrap_or_else(|e| {
        crate::xlog_warn!("D3D12.X: unable to prepare GPU residency; submitting anyway: {e}");
        None
    })
}

unsafe fn before_use<R: IntoIterator<Item = usize>>(
    queue: *mut c_void, resources: impl FnOnce(&Registry) -> Option<R>,
) -> Result<Option<Submission>> {
    let Some(lock) = REGISTRY.get() else { return Ok(None) };
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return Ok(None) };
    let mut g = lock.lock().unwrap();
    if g.objects.is_empty() {
        return Ok(None);
    }
    let mut device = None;
    q.GetDevice(&mut device)?;
    let Some(device): Option<ID3D12Device> = device else { return Ok(None) };
    let dev = device.as_raw() as usize;
    let Some(s) = g.devices.get(&dev) else { return Ok(None) };
    let (mut ids, opaque): (Vec<_>, bool) = match resources(&g) {
        None => {
            let ids = if s.evicted_local == 0 { Vec::new() } else {
                s.objects.iter().filter_map(|(&id, o)| (!o.resident).then_some(id)).collect()
            };
            (ids, true)
        }
        Some(resources) => (resources.into_iter().flat_map(|r| {
            std::iter::once(r).chain(g.links.get(&r).into_iter().flat_map(|l| l.heaps.iter().copied()))
        }).filter(|id| g.objects.get(id).is_some_and(|&(owner, _)| owner == dev)).collect(), false),
    };
    ids.sort_unstable();
    ids.dedup();
    if !opaque && ids.is_empty() {
        return Ok(None);
    }
    let s = g.devices.get_mut(&dev).unwrap();
    if opaque {
        s.opaque_pending += 1;
    } else {
        for id in &ids {
            if let Some(o) = s.objects.get_mut(id) {
                o.pins += 1;
            }
        }
    }
    while ids.iter().any(|id| g.devices[&dev].objects.get(id).is_some_and(|o| !o.resident && o.releasing != 0)) {
        g = RELEASED.wait(g).unwrap();
    }
    let s = g.devices.get_mut(&dev).unwrap();
    if let Err(e) = s.restore(&device, &ids) {
        crate::xlog_warn!("D3D12.X: MakeResident failed after retry; submitting anyway: {e}");
    }
    Ok(Some(Submission { device, ids, opaque }))
}

pub(super) unsafe fn after_submit(queue: *mut c_void, submission: Option<Submission>) {
    let Some(Submission { device, ids, opaque }) = submission else { return };
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let mut g = registry().lock().unwrap();
    let s = state(&mut g, &device);
    let key = queue as usize;
    let fence = match s.queues.entry(key) {
        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
        std::collections::hash_map::Entry::Vacant(e) => match super::real::create_residency_fence(&device) {
            Ok(fence) => e.insert(QueueFence { fence: Arc::new(fence), value: 0 }),
            Err(e) => {
                crate::xlog_warn!("D3D12.X: cannot track submission completion: {e}");
                return;
            }
        },
    };
    fence.value += 1;
    if let Err(e) = q.Signal(&*fence.fence, fence.value) {
        crate::xlog_warn!("D3D12.X: cannot signal residency fence: {e}");
        return;
    }
    super::markers::queue_fence(key, "the latest submission (residency signal)", &fence.fence, fence.value);
    s.serial += 1;
    if opaque {
        stamp(&mut s.opaque_last, &fence.fence, fence.value);
        s.opaque_used = s.serial;
        s.opaque_pending -= 1;
    } else {
        for id in ids {
            if let Some(o) = s.objects.get_mut(&id) {
                stamp(&mut o.last, &fence.fence, fence.value);
                o.last_used = s.serial;
                o.pins = o.pins.saturating_sub(1);
            }
        }
    }
    s.trim(&device, 0);
    s.report();
}

#[cfg(test)]
#[path = "residency_tests.rs"]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    fn heat(id: usize, mib: u64, last_used: u64) -> Heat {
        Heat { id, bytes: mib * MIB, last_used, in_flight: false, resident: true, high: false }
    }

    #[test]
    fn nothing_goes_at_or_below_ninety_percent() {
        let heats = [heat(1, 512, 0), heat(2, 512, 0)];
        assert!(pick(heats, 9 * GIB, 10 * GIB).is_empty());
        assert!(pick(heats, 0, 0).is_empty(), "no budget known");
    }

    #[test]
    fn vram_budget_override_is_positive_and_checked() {
        assert_eq!(budget_mb("1"), Some(1 << 20));
        assert_eq!(budget_mb("8192"), Some(8 * GIB));
        for val in ["0", "-1", "mistyped", "18446744073709551615"] {
            assert_eq!(budget_mb(val), None);
        }
    }

    #[test]
    fn protected_allocation_attempts_local_and_only_spills_on_oom() {
        let local = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let l0 = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, ..Default::default() };
        let mut attempts = Vec::new();
        let made = allocate_with(local, l0, true, |props| {
            attempts.push(props.Type);
            if props.Type == local.Type {
                Err(windows::core::Error::from_hresult(E_OUTOFMEMORY))
            } else {
                Ok(())
            }
        }).unwrap();
        assert_eq!(attempts, [local.Type, l0.Type]);
        assert!(made.1);

        attempts.clear();
        let invalid = windows::Win32::Foundation::E_INVALIDARG;
        let e = allocate_with(local, l0, true, |props| {
            attempts.push(props.Type);
            Err::<(), _>(windows::core::Error::from_hresult(invalid))
        }).unwrap_err();
        assert_eq!(e.code(), invalid);
        assert_eq!(attempts, [local.Type]);
    }

    #[test]
    fn the_coldest_go_first_until_eighty_percent() {
        // 9.5 of 10 GiB: 1.5 GiB must go
        let heats = [heat(1, 512, 50), heat(2, 1024, 10), heat(3, 512, 20), heat(4, 1024, 30)];
        assert_eq!(pick(heats, 9 * GIB + 512 * MIB, 10 * GIB), [2, 3]);
    }

    #[test]
    fn in_flight_and_evicted_allocations_are_not_candidates() {
        let mut busy = heat(1, 4096, 0);
        busy.in_flight = true;
        let mut gone = heat(2, 4096, 0);
        gone.resident = false;
        let warm = heat(3, 4096, 98);
        let cold = heat(4, 256, 97);
        assert_eq!(pick([busy, gone, warm, cold], 10 * GIB, 10 * GIB), [4, 3]);
    }

    #[test]
    fn at_equal_age_other_memory_before_targets_and_the_bigger_first() {
        let mut target = heat(1, 2048, 10);
        target.high = true;
        let heats = [target, heat(2, 256, 10), heat(3, 2048, 10)];
        assert_eq!(pick(heats, 10 * GIB + 256 * MIB, 10 * GIB), [3, 2]);
    }
}
