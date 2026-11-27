//! Resizable BAR: XCL's own upload buffers in video memory.
//!
//! An upload heap is system memory: the CPU writes it and the GPU reads it over the bus. When
//! the adapter's video memory is all behind a resizable BAR, D3D12 has a second kind,
//! `D3D12_HEAP_TYPE_GPU_UPLOAD` (`D3D12_FEATURE_D3D12_OPTIONS16::GPUUploadHeapSupported`): video
//! memory the CPU maps. XCL's buffers that the CPU only writes and the GPU reads go there
//! (`upload_buffer`) while the adapter's local budget has room, and stay upload heaps otherwise.
//!
//! What does not go there, and why:
//! - the title's own memory (the coherent window, its physical pages, the CPU side of the sparse
//!   windows): it sits at addresses the title and its translated shaders hold, and D3D12 maps a
//!   GPU upload heap where it likes; the title also reads it back;
//! - anything the CPU reads (readback buffers, CPU-cached heaps): the mapping is write-combined,
//!   and a read of it is uncached;
//! - buffers of one call (a texture upload recorded on a title's list, fill patterns): they are
//!   gone before they would count.
//!
//! `XCL_REBAR=0` turns it off; `XCL_REBAR=sparse,textures` names the classes that may go
//! (`Class`). One line at start says what the device supports; `summary` has the bytes placed.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory4, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO};

/// Local video memory left under the adapter's budget after a buffer is placed: the title's
/// textures and targets grow into it, and they spill to system memory when it is gone.
const RESERVE: u64 = 1 << 30;
/// The most of XCL's upload buffers in video memory at once.
const CAP: u64 = 512 << 20;

/// What a buffer is for: each can be kept in system memory on its own (`XCL_REBAR`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    /// the staging ring and the zero buffer of the sparse windows' uploads (`sparse`)
    Sparse,
    /// a queue's texture upload buffer (`texture::upload_batch`)
    Textures,
    /// the roots and leaves of the physical page directory translated shaders read
    Directory,
}

impl Class {
    const ALL: [Class; 3] = [Class::Sparse, Class::Textures, Class::Directory];

    fn name(self) -> &'static str {
        match self {
            Class::Sparse => "sparse",
            Class::Textures => "textures",
            Class::Directory => "directory",
        }
    }
}

/// The classes `XCL_REBAR` allows, as a bit per class, and the names in it that are none.
fn classes(raw: Option<&str>) -> (u8, Vec<String>) {
    let all = (1 << Class::ALL.len()) - 1;
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return (all, Vec::new());
    };
    if ["0", "off", "none", "false"].contains(&raw.to_ascii_lowercase().as_str()) {
        return (0, Vec::new());
    }
    if ["1", "on", "all", "true"].contains(&raw.to_ascii_lowercase().as_str()) {
        return (all, Vec::new());
    }
    let (mut mask, mut unknown) = (0, Vec::new());
    for name in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match Class::ALL.iter().position(|c| c.name().eq_ignore_ascii_case(name)) {
            Some(i) => mask |= 1 << i,
            None => unknown.push(name.to_string()),
        }
    }
    (mask, unknown)
}

/// Whether `bytes` more may go to video memory: XCL's buffers there stay under `CAP`, and local
/// use stays `RESERVE` under the budget.
fn fits(bytes: u64, placed: u64, usage: u64, budget: u64) -> bool {
    placed.saturating_add(bytes) <= CAP && usage.saturating_add(bytes).saturating_add(RESERVE) <= budget
}

/// The device whose buffers may go to video memory.
struct Target {
    device: usize,
    adapter: IDXGIAdapter3,
    classes: u8,
}

// the adapter is a free-threaded DXGI object; the device is compared, never called
unsafe impl Send for Target {}
unsafe impl Sync for Target {}

static TARGET: OnceLock<Option<Target>> = OnceLock::new();
/// bytes in video memory now, by class
static PLACED: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
/// buffers placed in video memory since the start, by class
static PLACED_EVER: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
/// bytes of buffers that stayed in system memory for the budget or the cap, since the start
static KEPT: AtomicU64 = AtomicU64::new(0);
/// buffers the device refused in video memory
static REFUSED: AtomicU64 = AtomicU64::new(0);

fn placed() -> u64 {
    PLACED.iter().map(|p| p.load(Relaxed)).sum()
}

/// A buffer's bytes in video memory, counted while this lives beside the buffer.
#[derive(Default)]
pub struct Placed(Option<(Class, u64)>);

impl Drop for Placed {
    fn drop(&mut self) {
        if let Some((class, bytes)) = self.0 {
            PLACED[class as usize].fetch_sub(bytes, Relaxed);
        }
    }
}

unsafe fn local_memory(adapter: &IDXGIAdapter3) -> Option<DXGI_QUERY_VIDEO_MEMORY_INFO> {
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info).ok()?;
    Some(info)
}

/// Look at `dev` once, when it is made: whether its buffers may go to video memory, and the
/// line that says so.
pub fn init(dev: &ID3D12Device) {
    TARGET.get_or_init(|| unsafe { probe(dev, std::env::var("XCL_REBAR").ok().as_deref()) });
}

unsafe fn probe(dev: &ID3D12Device, raw: Option<&str>) -> Option<Target> {
    let (mask, unknown) = classes(raw);
    if !unknown.is_empty() {
        crate::xlog_warn!("ReBAR: XCL_REBAR names no class {} (the classes: sparse, textures, directory)", unknown.join(", "));
    }
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    let supported = dev
        .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &mut options as *mut _ as *mut c_void, std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS16>() as u32)
        .is_ok()
        && options.GPUUploadHeapSupported.as_bool();
    let adapter = CreateDXGIFactory1::<IDXGIFactory4>().ok().and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter3>(dev.GetAdapterLuid()).ok());
    let desc = adapter.as_ref().and_then(|a| a.GetDesc().ok());
    let bar = match desc.as_ref().and_then(|d| bar_bytes(d.VendorId, d.DeviceId)) {
        Some(b) => format!("BAR {} MiB", b >> 20),
        None => "BAR size unknown".to_string(),
    };
    let video = desc.as_ref().map_or(0, |d| d.DedicatedVideoMemory as u64 >> 20);
    let local = adapter.as_ref().and_then(|a| local_memory(a));
    let names = || Class::ALL.iter().enumerate().filter(|(i, _)| mask & 1 << i != 0).map(|(_, c)| c.name()).collect::<Vec<_>>().join(", ");
    let state = match (supported, mask, &adapter, &local) {
        (false, ..) => "GPU upload heaps not supported (resizable BAR off in the firmware, or the driver or Windows older than it); XCL's upload buffers stay in system memory".to_string(),
        (true, 0, ..) => format!("GPU upload heaps supported; off (XCL_REBAR={})", raw.unwrap_or("").trim()),
        (true, _, Some(_), Some(l)) => format!(
            "GPU upload heaps supported; XCL's upload buffers ({}) go to video memory, at most {} MiB and while local use stays {} MiB under its budget (now {} of {} MiB); XCL_REBAR=0 turns it off",
            names(),
            CAP >> 20,
            RESERVE >> 20,
            l.CurrentUsage >> 20,
            l.Budget >> 20
        ),
        (true, ..) => "GPU upload heaps supported; the adapter does not tell its local budget, so XCL's upload buffers stay in system memory".to_string(),
    };
    crate::xlog_info!("ReBAR: {state}; {bar} for {video} MiB of video memory; 0 MiB placed");
    let adapter = adapter.filter(|_| supported && mask != 0 && local.is_some())?;
    Some(Target { device: dev.as_raw() as usize, adapter, classes: mask })
}

/// A committed buffer (`desc`) for bytes the CPU only writes and the GPU reads, in the state an
/// upload heap's buffer is in (`D3D12_RESOURCE_STATE_GENERIC_READ`): in video memory when the
/// device has GPU upload heaps, `class` is allowed and the local budget has room, else in an
/// upload heap. The CPU must not read what it maps of it.
pub unsafe fn upload_buffer(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC, class: Class) -> windows::core::Result<(ID3D12Resource, Placed)> {
    let create = |heap: D3D12_HEAP_TYPE| {
        let mut r: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut r)?;
        r.ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_POINTER))
    };
    let bytes = desc.Width;
    if wants(dev, class, bytes) {
        match create(D3D12_HEAP_TYPE_GPU_UPLOAD) {
            Ok(r) => {
                if PLACED_EVER[class as usize].fetch_add(1, Relaxed) == 0 {
                    crate::xlog_info!("ReBAR: the first {} buffer ({} KiB) is in video memory", class.name(), bytes >> 10);
                }
                PLACED[class as usize].fetch_add(bytes, Relaxed);
                return Ok((r, Placed(Some((class, bytes)))));
            }
            Err(e) => {
                if REFUSED.fetch_add(1, Relaxed) == 0 {
                    crate::xlog_warn!("ReBAR: the device refused a {} buffer of {} KiB in video memory ({e}); it is in system memory", class.name(), bytes >> 10);
                }
            }
        }
    }
    Ok((create(D3D12_HEAP_TYPE_UPLOAD)?, Placed(None)))
}

/// Whether a `class` buffer of `bytes` on `dev` goes to video memory now.
unsafe fn wants(dev: &ID3D12Device, class: Class, bytes: u64) -> bool {
    // (tests) XCL_REBAR_TESTS=1: the first device a test makes such a buffer on is looked at as
    // the game's is, so the suites that upload run with their buffers in video memory
    #[cfg(test)]
    if std::env::var_os("XCL_REBAR_TESTS").is_some() {
        init(dev);
    }
    let Some(t) = TARGET.get().and_then(Option::as_ref) else { return false };
    if t.device != dev.as_raw() as usize || t.classes & 1 << class as usize == 0 {
        return false;
    }
    if local_memory(&t.adapter).is_some_and(|l| fits(bytes, placed(), l.CurrentUsage, l.Budget)) {
        return true;
    }
    if KEPT.fetch_add(bytes, Relaxed) == 0 {
        crate::xlog_info!("ReBAR: a {} buffer of {} KiB stays in system memory: local video memory has no room under its budget ({})", class.name(), bytes >> 10, summary());
    }
    false
}

/// XCL's upload buffers in video memory now, for the memory lines of the log.
pub fn summary() -> String {
    let Some(t) = TARGET.get() else { return "ReBAR: no device yet".into() };
    if t.is_none() {
        return "ReBAR: XCL's upload buffers are in system memory (off or not supported)".into();
    }
    let by = |c: Class| PLACED[c as usize].load(Relaxed);
    format!(
        "ReBAR: {} MiB of XCL's upload buffers in video memory (sparse staging {} MiB, texture staging {} MiB, page directory {} KiB; {} buffers placed since the start); {} MiB kept in system memory for the local budget since the start, {} refused by the device",
        placed() >> 20,
        by(Class::Sparse) >> 20,
        by(Class::Textures) >> 20,
        by(Class::Directory) >> 10,
        PLACED_EVER.iter().map(|n| n.load(Relaxed)).sum::<u64>(),
        KEPT.load(Relaxed) >> 20,
        REFUSED.load(Relaxed)
    )
}

/// The bytes of a memory range in a resource descriptor of the configuration manager
/// (`MEM_DES`, `MEM_LARGE_DES`: count, type, first address, last address).
fn range_bytes(kind: u32, data: &[u8]) -> Option<u64> {
    const MEM: u32 = 1;
    const MEM_LARGE: u32 = 7;
    if kind != MEM && kind != MEM_LARGE {
        return None;
    }
    let word = |at: usize| data.get(at..at + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()));
    let (first, last) = (word(8)?, word(16)?);
    last.checked_sub(first)?.checked_add(1)
}

/// The largest memory range Windows gave the PCI device `vendor`:`device` – for a display
/// adapter, the BAR its video memory is behind (256 MiB without resizable BAR). None: not
/// found, or the configuration manager does not tell.
fn bar_bytes(vendor: u32, device: u32) -> Option<u64> {
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    type ListSize = unsafe extern "system" fn(*mut u32, *const u16, u32) -> u32;
    type List = unsafe extern "system" fn(*const u16, *mut u16, u32, u32) -> u32;
    type Locate = unsafe extern "system" fn(*mut u32, *const u16, u32) -> u32;
    type FirstConf = unsafe extern "system" fn(*mut usize, u32, u32) -> u32;
    type NextRes = unsafe extern "system" fn(*mut usize, usize, u32, *mut u32, u32) -> u32;
    type DataSize = unsafe extern "system" fn(*mut u32, usize, u32) -> u32;
    type Data = unsafe extern "system" fn(usize, *mut c_void, u32, u32) -> u32;
    type Free = unsafe extern "system" fn(usize) -> u32;
    const FILTER_ENUMERATOR_PRESENT: u32 = 0x101;
    const ALLOC_LOG_CONF: u32 = 2;
    let wide = |s: &str| s.encode_utf16().chain([0]).collect::<Vec<u16>>();
    unsafe {
        let dll = LoadLibraryW(wide("cfgmgr32.dll").as_ptr());
        if dll.is_null() {
            return None;
        }
        macro_rules! f {
            ($name:literal, $ty:ty) => {
                std::mem::transmute::<unsafe extern "system" fn() -> isize, $ty>(GetProcAddress(dll, concat!($name, "\0").as_ptr())?)
            };
        }
        let list_size: ListSize = f!("CM_Get_Device_ID_List_SizeW", ListSize);
        let list: List = f!("CM_Get_Device_ID_ListW", List);
        let locate: Locate = f!("CM_Locate_DevNodeW", Locate);
        let first_conf: FirstConf = f!("CM_Get_First_Log_Conf", FirstConf);
        let next_res: NextRes = f!("CM_Get_Next_Res_Des", NextRes);
        let data_size: DataSize = f!("CM_Get_Res_Des_Data_Size", DataSize);
        let data: Data = f!("CM_Get_Res_Des_Data", Data);
        let free_res: Free = f!("CM_Free_Res_Des_Handle", Free);
        let free_conf: Free = f!("CM_Free_Log_Conf_Handle", Free);

        let pci = wide("PCI");
        let mut len = 0u32;
        if list_size(&mut len, pci.as_ptr(), FILTER_ENUMERATOR_PRESENT) != 0 || len == 0 {
            return None;
        }
        let mut ids = vec![0u16; len as usize];
        if list(pci.as_ptr(), ids.as_mut_ptr(), len, FILTER_ENUMERATOR_PRESENT) != 0 {
            return None;
        }
        let prefix = format!("PCI\\VEN_{vendor:04X}&DEV_{device:04X}&");
        let id = ids.split(|&c| c == 0).find(|id| String::from_utf16_lossy(id).to_ascii_uppercase().starts_with(&prefix))?;
        let id: Vec<u16> = id.iter().copied().chain([0]).collect();
        let mut node = 0u32;
        if locate(&mut node, id.as_ptr(), 0) != 0 {
            return None;
        }
        let mut conf = 0usize;
        if first_conf(&mut conf, node, ALLOC_LOG_CONF) != 0 {
            return None;
        }
        let mut largest = None;
        let mut at = conf;
        loop {
            let (mut next, mut kind) = (0usize, 0u32);
            let more = next_res(&mut next, at, 0, &mut kind, 0) == 0;
            if at != conf {
                free_res(at);
            }
            if !more {
                break;
            }
            at = next;
            let mut size = 0u32;
            if data_size(&mut size, at, 0) != 0 || size == 0 {
                continue;
            }
            let mut bytes = vec![0u8; size as usize];
            if data(at, bytes.as_mut_ptr().cast(), size, 0) == 0 {
                largest = largest.max(range_bytes(kind, &bytes));
            }
        }
        free_conf(conf);
        largest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// XCL_REBAR: unset or on is every class, 0 is none, names are those classes, and a name
    /// that is no class is told.
    #[test]
    fn the_switch_names_the_classes_that_go_to_video_memory() {
        assert_eq!(classes(None), (0b111, vec![]));
        assert_eq!(classes(Some("")), (0b111, vec![]));
        assert_eq!(classes(Some("1")), (0b111, vec![]));
        for off in ["0", "off", "OFF", " none ", "false"] {
            assert_eq!(classes(Some(off)), (0, vec![]), "{off}");
        }
        assert_eq!(classes(Some("sparse")), (0b001, vec![]));
        assert_eq!(classes(Some("Textures, directory")), (0b110, vec![]));
        assert_eq!(classes(Some("sparse,windows")), (0b001, vec!["windows".to_string()]));
    }

    /// A buffer goes to video memory only while local use stays the reserve under the budget
    /// and XCL's own buffers there stay under the cap.
    #[test]
    fn a_buffer_goes_to_video_memory_only_under_the_budget_and_the_cap() {
        const MIB: u64 = 1 << 20;
        // play-94: 5475 of 9283 MiB local
        assert!(fits(64 * MIB, 0, 5475 * MIB, 9283 * MIB));
        // the last buffer that leaves the reserve whole, and the first that does not
        assert!(fits(64 * MIB, 0, 9283 * MIB - RESERVE - 64 * MIB, 9283 * MIB));
        assert!(!fits(64 * MIB + 1, 0, 9283 * MIB - RESERVE - 64 * MIB, 9283 * MIB));
        // over the budget already (play-56: 7886 of 8526 MiB), or no budget told
        assert!(!fits(MIB, 0, 7886 * MIB, 8526 * MIB));
        assert!(!fits(MIB, 0, 0, 0));
        assert!(!fits(u64::MAX, 0, u64::MAX, u64::MAX));
        // the cap on XCL's own
        assert!(fits(64 * MIB, CAP - 64 * MIB, 0, 9283 * MIB));
        assert!(!fits(64 * MIB, CAP - 64 * MIB + 1, 0, 9283 * MIB));
    }

    /// The configuration manager's memory descriptors give a range's bytes; other resources and
    /// short or inverted data give none.
    #[test]
    fn a_memory_descriptor_gives_the_bytes_of_its_range() {
        let des = |first: u64, last: u64| {
            let mut d = vec![0u8; 32];
            d[8..16].copy_from_slice(&first.to_le_bytes());
            d[16..24].copy_from_slice(&last.to_le_bytes());
            d
        };
        // an RTX 3080 with resizable BAR: 16 MiB of registers, 16 GiB of video memory
        assert_eq!(range_bytes(1, &des(0xfb00_0000, 0xfbff_ffff)), Some(16 << 20));
        assert_eq!(range_bytes(7, &des(0x78_0000_0000, 0x7b_ffff_ffff)), Some(16 << 30));
        assert_eq!(range_bytes(2, &des(0xe000, 0xe07f)), None);
        assert_eq!(range_bytes(7, &des(0x1000, 0xfff)), None);
        assert_eq!(range_bytes(7, &[0; 23]), None);
        assert_eq!(range_bytes(7, &des(0, u64::MAX)), None);
    }

    /// A buffer's bytes count as placed while its `Placed` lives.
    #[test]
    fn placed_bytes_are_counted_while_the_buffer_lives() {
        let before = PLACED[Class::Directory as usize].load(Relaxed);
        PLACED[Class::Directory as usize].fetch_add(4096, Relaxed);
        let p = Placed(Some((Class::Directory, 4096)));
        assert!(p.0.is_some() && !Placed::default().0.is_some());
        assert_eq!(PLACED[Class::Directory as usize].load(Relaxed), before + 4096);
        drop(p);
        drop(Placed::default());
        assert_eq!(PLACED[Class::Directory as usize].load(Relaxed), before);
    }

    /// On a device: a buffer from `upload_buffer` is in an upload heap before the device is
    /// looked at, and in video memory after when the device has GPU upload heaps and room; the
    /// GPU reads what the CPU wrote to it the same in either memory, and its bytes are counted
    /// while it lives.
    #[test]
    fn gpu_reads_what_the_cpu_wrote_to_an_upload_buffer_in_either_memory() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            const BYTES: u64 = 1 << 20;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: BYTES,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let heap_of = |r: &ID3D12Resource| {
                let mut props = D3D12_HEAP_PROPERTIES::default();
                r.GetHeapProperties(Some(&mut props), None).unwrap();
                props.Type
            };
            let ours = |t: &Option<Target>| t.as_ref().is_some_and(|t| t.device == dev.as_raw() as usize);
            let mut buffers = Vec::new();
            if std::env::var_os("XCL_REBAR_TESTS").is_none() && !TARGET.get().is_some_and(ours) {
                let (r, placed) = upload_buffer(&dev, &desc, Class::Sparse).unwrap();
                assert!(!placed.0.is_some());
                assert_eq!(heap_of(&r), D3D12_HEAP_TYPE_UPLOAD);
                buffers.push((r, placed));
            }
            let on = ours(TARGET.get_or_init(|| probe(&dev, None)));
            let before = PLACED[Class::Textures as usize].load(Relaxed);
            let (r, placed) = upload_buffer(&dev, &desc, Class::Textures).unwrap();
            assert_eq!(heap_of(&r), if placed.0.is_some() { D3D12_HEAP_TYPE_GPU_UPLOAD } else { D3D12_HEAP_TYPE_UPLOAD });
            assert!(on || !placed.0.is_some());
            assert_eq!(PLACED[Class::Textures as usize].load(Relaxed) - before, if placed.0.is_some() { BYTES } else { 0 });
            println!("GPU upload heaps: {}; {}", if placed.0.is_some() { "the buffer is in video memory" } else { "the buffer is in system memory" }, summary());
            buffers.push((r, placed));

            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let word = |i: usize, k: usize| (i as u32).wrapping_mul(0x9e37_79b9) ^ k as u32;
            let mut readbacks = Vec::new();
            for (k, (source, _)) in buffers.iter().enumerate() {
                let mut p = std::ptr::null_mut();
                source.Map(0, None, Some(&mut p)).unwrap();
                for i in 0..BYTES as usize / 4 {
                    (p as *mut u32).add(i).write(word(i, k));
                }
                let mut readback: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).unwrap();
                let readback = readback.unwrap();
                list.CopyBufferRegion(&readback, 0, source, 0, BYTES);
                readbacks.push(readback);
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            crate::gpu::texture::wait(&dev, &queue).unwrap();
            for (k, readback) in readbacks.iter().enumerate() {
                let mut p = std::ptr::null_mut();
                readback.Map(0, None, Some(&mut p)).unwrap();
                let words = std::slice::from_raw_parts(p as *const u32, BYTES as usize / 4);
                assert!(words.iter().enumerate().all(|(i, &w)| w == word(i, k)), "buffer {k}: the GPU read other bytes than the CPU wrote");
                readback.Unmap(0, None);
            }
            drop(buffers);
            assert_eq!(PLACED[Class::Textures as usize].load(Relaxed), before);
        }
    }
}
