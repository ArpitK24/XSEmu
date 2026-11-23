//! GPU-owned textures in title memory, aliased as on the console.
//!
//! A console title places render targets, depth buffers and UAV textures at addresses of its
//! own memory and reuses that memory: transient targets of a frame share it, and it creates
//! several resource objects over the same address expecting one memory (seen on XDefiant: 16
//! placements at an occupied address in its first seconds, also of other formats and sizes).
//! A committed PC resource per placement breaks that – what the GPU writes through one never
//! shows in the other. So these textures are placed in *arenas*: D3D12 default heaps (video
//! memory) standing for ranges of the title's address space (128 MiB+), at the offset of their
//! address in it. Resources at one address then share memory as on the console (the same
//! description: the same PC layout, so the same texels). The CPU never sees these textures
//! (on the console it could, but GPU-owned targets are not CPU-written).

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

/// bytes of title address space an arena stands for at least (and of video memory it takes)
const ARENA: usize = 128 << 20;
/// arenas start and end at multiples of this
const ARENA_ALIGN: usize = 32 << 20;

struct Arena {
    base: usize,
    size: usize,
    heap: ID3D12Heap,
}

// D3D12 heaps are free-threaded
unsafe impl Send for Arena {}

static ARENAS: Mutex<Vec<Arena>> = Mutex::new(Vec::new());
static PLACED_TEXTURES: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// A texture backed by one of the aliasable D3D12 heaps.
pub(super) fn is_placed_texture(resource: usize) -> bool {
    PLACED_TEXTURES.lock().unwrap().as_ref().is_some_and(|textures| textures.contains(&resource))
}

pub(super) fn forget(resource: usize) {
    if let Some(textures) = PLACED_TEXTURES.lock().unwrap().as_mut() {
        textures.remove(&resource);
    }
}

/// Whether a console texture with these D3D12_RESOURCE_FLAGS is the GPU's (render target,
/// depth buffer, UAV).
pub fn gpu_owned(flags: u32) -> bool {
    flags & 0x7 != 0
}

/// The arena holding `[va, va + bytes)` – an existing one, or a new one from the 32 MiB below
/// `va`, 128 MiB or what the resource needs (None: no video memory).
unsafe fn arena_for(dev: &ID3D12Device, va: usize, bytes: usize) -> Option<(ID3D12Heap, u64)> {
    let mut g = ARENAS.lock().unwrap();
    if let Some(a) = g.iter().find(|a| a.base <= va && va + bytes <= a.base + a.size) {
        return Some((a.heap.clone(), (va - a.base) as u64));
    }
    let base = va & !(ARENA_ALIGN - 1);
    let size = (va + bytes - base).div_ceil(ARENA_ALIGN).max(ARENA / ARENA_ALIGN) * ARENA_ALIGN;
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        // MSAA targets need 4 MiB placement
        Alignment: D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT as u64,
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = dev.CreateHeap(&desc, &mut heap) {
        crate::xlog_warn!("D3D12.X: video-memory arena for {base:#x}: CreateHeap: {e}");
        return None;
    }
    let heap = heap?;
    crate::xlog_info!("D3D12.X: video-memory arena {} for title addresses {base:#x}..{:#x}", g.len(), base + size);
    g.push(Arena { base, size, heap: heap.clone() });
    Some((heap, (va - base) as u64))
}

/// Heaps of every kind of resource in one (resource heap tier 2) – else no arenas.
fn tier2(dev: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.ResourceHeapTier.0 >= 2 }
}

/// A GPU-owned texture of the title at `va` (`desc`: the PC description): placed in the arena
/// for that address (None: it cannot be – the caller makes a resource of its own).
pub unsafe fn place(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, clear: *const c_void, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) {
        return None;
    }
    let info = dev.GetResourceAllocationInfo(0, &[*desc]);
    if info.SizeInBytes == u64::MAX || info.SizeInBytes == 0 || va as u64 % info.Alignment.max(1) != 0 {
        return None;
    }
    let (heap, offset) = arena_for(dev, va, info.SizeInBytes as usize)?;
    let hr = super::real::dev_create_placed_resource(this, heap.as_raw() as usize, offset as usize, desc as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        return None;
    }
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    Some(hr)
}
