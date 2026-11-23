//! Process-wide table of modules XCL knows about:
//! * images mapped by the XCL loader (game exe + game-shipped DLLs),
//! * pseudo handles for HLE modules (`kernelx.dll` & co. have no real image).
//!
//! Address → module lookups used from exception dispatch (`RtlPcToFileHeader` hook) go
//! through a lock-free fixed array so they can never deadlock.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::pe::OwnedExports;

pub struct MappedModule {
    /// Lower-case file name, e.g. "game.exe".
    pub name: String,
    pub path: PathBuf,
    pub base: usize,
    pub size: usize,
    pub entry_rva: u32,
    pub is_dll: bool,
    pub exports: OwnedExports,
    pub tls_callbacks: Mutex<Vec<usize>>,
    /// DisableThreadLibraryCalls was called.
    pub no_thread_calls: AtomicBool,
    /// DllMain(PROCESS_ATTACH) has completed.
    pub initialized: AtomicBool,
    pub stack_reserve: u64,
}

impl MappedModule {
    pub fn contains(&self, addr: usize) -> bool {
        addr >= self.base && addr < self.base + self.size
    }

    fn available(&self) -> bool {
        !self.is_dll || self.initialized.load(Ordering::Acquire) || crate::loader::on_loading_thread()
    }
}

static MAPPED: RwLock<Vec<Arc<MappedModule>>> = RwLock::new(Vec::new());
static MAIN: RwLock<Option<Arc<MappedModule>>> = RwLock::new(None);

const MAX_RANGES: usize = 512;
static RANGE_BASE: [AtomicUsize; MAX_RANGES] = [const { AtomicUsize::new(0) }; MAX_RANGES];
static RANGE_SIZE: [AtomicUsize; MAX_RANGES] = [const { AtomicUsize::new(0) }; MAX_RANGES];
static RANGE_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn register_mapped(m: Arc<MappedModule>) {
    let i = RANGE_COUNT.fetch_add(1, Ordering::AcqRel);
    if i < MAX_RANGES {
        RANGE_SIZE[i].store(m.size, Ordering::Release);
        RANGE_BASE[i].store(m.base, Ordering::Release);
    }
    MAPPED.write().unwrap().push(m);
}

pub fn unregister_mapped(m: &Arc<MappedModule>) {
    MAPPED.write().unwrap().retain(|loaded| !Arc::ptr_eq(loaded, m));
    let n = RANGE_COUNT.load(Ordering::Acquire).min(MAX_RANGES);
    for slot in &RANGE_BASE[..n] {
        if slot.compare_exchange(m.base, 0, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            break;
        }
    }
    let mut main = MAIN.write().unwrap();
    if main.as_ref().is_some_and(|loaded| Arc::ptr_eq(loaded, m)) {
        *main = None;
    }
}

/// Lock-free: base of the mapped image containing `addr`.
pub fn image_base_for_addr(addr: usize) -> Option<usize> {
    let n = RANGE_COUNT.load(Ordering::Acquire).min(MAX_RANGES);
    for i in 0..n {
        let b = RANGE_BASE[i].load(Ordering::Acquire);
        let s = RANGE_SIZE[i].load(Ordering::Acquire);
        if b != 0 && addr >= b && addr < b + s {
            return Some(b);
        }
    }
    None
}

pub fn mapped_modules() -> Vec<Arc<MappedModule>> {
    MAPPED.read().map(|v| v.iter().filter(|m| m.available()).cloned().collect()).unwrap_or_default()
}

pub fn find_mapped_by_name(name: &str) -> Option<Arc<MappedModule>> {
    let key = crate::util::dll_key(name);
    MAPPED.read().ok()?.iter().find(|m| m.name == key && m.available()).cloned()
}

pub fn find_mapped_by_base(base: usize) -> Option<Arc<MappedModule>> {
    MAPPED.read().ok()?.iter().find(|m| m.base == base && m.available()).cloned()
}

pub fn find_mapped_by_addr(addr: usize) -> Option<Arc<MappedModule>> {
    MAPPED.read().ok()?.iter().find(|m| m.contains(addr)).cloned()
}

pub fn set_main(m: Arc<MappedModule>) {
    *MAIN.write().unwrap() = Some(m);
}

pub fn main_module() -> Option<Arc<MappedModule>> {
    MAIN.read().ok()?.clone()
}

// ---------------------------------------------------------------------------------------
// HLE pseudo handles
// ---------------------------------------------------------------------------------------

static HLE_HANDLES: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

/// Stable fake HMODULE for an HLE module: a committed page starting with "MZ".
pub fn hle_handle(module_name: &str) -> usize {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE};
    let key = crate::util::dll_key(module_name);
    let mut g = HLE_HANDLES.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    if let Some(&h) = map.get(&key) {
        return h;
    }
    let p = unsafe { VirtualAlloc(std::ptr::null(), 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as *mut u8;
    unsafe {
        *p = b'M';
        *p.add(1) = b'Z';
    }
    map.insert(key, p as usize);
    p as usize
}

pub fn find_hle_handle(module_name: &str) -> Option<usize> {
    let key = crate::util::dll_key(module_name);
    let g = HLE_HANDLES.lock().ok()?;
    g.as_ref()?.get(&key).copied()
}

pub fn hle_name_by_handle(h: usize) -> Option<String> {
    let g = HLE_HANDLES.lock().ok()?;
    g.as_ref()?.iter().find(|(_, &v)| v == h).map(|(k, _)| k.clone())
}

/// "game.exe+0x1234", "KERNELBASE.dll+0x5678" or "0x...".
pub fn describe_address(addr: usize) -> String {
    if let Some(b) = image_base_for_addr(addr) {
        if let Ok(g) = MAPPED.try_read() {
            if let Some(m) = g.iter().find(|m| m.base == b) {
                return format!("{}+{:#x}", m.name, addr - b);
            }
        }
        return format!("{b:#x}+{:#x}", addr - b);
    }
    unsafe {
        use windows_sys::Win32::System::LibraryLoader::{
            GetModuleFileNameW, GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        };
        let mut h = std::ptr::null_mut();
        if GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            addr as *const u16,
            &mut h,
        ) != 0
        {
            let mut buf = [0u16; 260];
            let n = GetModuleFileNameW(h, buf.as_mut_ptr(), 260) as usize;
            let full = String::from_utf16_lossy(&buf[..n]);
            let name = full.rsplit('\\').next().unwrap_or(&full).to_string();
            return format!("{name}+{:#x}", addr - h as usize);
        }
    }
    format!("{addr:#x}")
}
