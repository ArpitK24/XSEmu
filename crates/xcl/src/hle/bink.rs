//! Bink 2 (RAD's video codec, shipped by many titles as `bink2*.dll`) decodes into frame buffers
//! the title allocates and registers (`BinkRegisterFrameBuffers`); the title then samples the
//! planes as textures. On the console the CPU's writes are simply in memory the GPU reads; here
//! a texture is uploaded again only when its memory is seen written, and the decoder's writes to
//! pool pages that GPU views show are not always seen (a frame of zero planes is a green image).
//! So the title's imports of the decoding family go through these wrappers: a registered frame
//! buffer set's planes are recorded, and each decode that finishes (`BinkDoFrame`,
//! `BinkDoFramePlane`, `BinkDoFrameAsyncWait` reporting done, `BinkNextFrame`, `BinkGoto`) marks
//! them written for the next submit (`title_textures::mark_cpu_written`).
//!
//! The console Bink 2 ABI has a 24-byte `BINKFRAMEBUFFERS` header followed by up to three frame
//! sets. Each contains five 16-byte `BINKPLANE`s in Y, cR, cB, A, auxiliary order. The header
//! supplies the aligned luma and chroma heights used to allocate the planes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// HBINK → the planes (address, bytes) of its registered frame buffers
static SETS: Mutex<Option<HashMap<usize, Vec<(usize, usize)>>>> = Mutex::new(None);

#[repr(C)]
struct FrameBuffersHeader {
    total_frames: u32,
    luma_width: u32,
    luma_height: u32,
    _chroma_width: u32,
    chroma_height: u32,
    frame_index: u32,
}

#[repr(C)]
struct BinkPlane {
    buffer: usize,
    _allocate: u32,
    pitch: u32,
}

type FramePlanes = [BinkPlane; 5];
const MAX_FRAMES: usize = 3;

type Fn4 = unsafe extern "system" fn(usize, usize, usize, usize) -> usize;

macro_rules! wrappers {
    ($($name:ident => $after:expr),* $(,)?) => {
        $(
            #[allow(non_snake_case)]
            mod $name {
                pub(super) static REAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                pub(super) static CALLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            }
            #[allow(non_snake_case)]
            unsafe extern "system" fn $name(a: usize, b: usize, c: usize, d: usize) -> usize {
                if !$name::CALLED.swap(true, Ordering::Relaxed) {
                    crate::xlog_info!("Bink: first call of {} ({a:#x}, {b:#x})", stringify!($name));
                }
                let f: Fn4 = std::mem::transmute($name::REAL.load(Ordering::Acquire));
                let r = f(a, b, c, d);
                let after: fn(&'static str, usize, usize, usize) = $after;
                after(stringify!($name), a, b, r);
                r
            }
        )*
        /// The wrapper for `function`, given the real export.
        fn wrapper(function: &str, real: usize) -> Option<usize> {
            match function {
                $(stringify!($name) => {
                    $name::REAL.store(real, Ordering::Release);
                    Some($name as usize)
                })*
                _ => None,
            }
        }
    };
}

wrappers! {
    BinkRegisterFrameBuffers => |_, bink, set, _| unsafe { register(bink, set) },
    BinkAllocateFrameBuffers => |_, bink, set, ok| if ok as u32 != 0 { unsafe { register(bink, set) } },
    BinkDoFrame => |what, bink, _, _| decoded(what, bink),
    BinkDoFramePlane => |what, bink, _, _| decoded(what, bink),
    BinkNextFrame => |what, bink, _, _| decoded(what, bink),
    BinkGoto => |what, bink, _, _| decoded(what, bink),
    BinkDoFrameAsyncWait => |what, bink, _, done| if done as u32 != 0 { decoded(what, bink) },
    BinkClose => |_, bink, _, _| {
        if let Some(m) = SETS.lock().unwrap().as_mut() {
            m.remove(&bink);
        }
    },
}

/// Route a title's import of `dll!function` (resolved to `real`) through its wrapper when it is
/// one of a Bink 2 DLL's decoding family.
pub fn wrap(dll: &str, function: &str, real: usize) -> usize {
    if real == 0
        || !dll.to_ascii_lowercase().starts_with("bink2")
        || std::env::var("XCL_BINK_HOOK").is_ok_and(|v| v == "0")
    {
        return real;
    }
    let w = wrapper(function, real);
    if w.is_some() {
        crate::xlog_info!("Bink: {dll}!{function} goes through XCL's wrapper");
    }
    w.unwrap_or(real)
}

unsafe fn register(bink: usize, set: usize) {
    use crate::gpu::shader::readable;
    use std::mem::size_of;

    let header_len = size_of::<FrameBuffersHeader>();
    if bink == 0 || !readable(set, header_len) {
        return;
    }
    let header = (set as *const FrameBuffersHeader).read_unaligned();
    let frames = header.total_frames as usize;
    if !(1..=MAX_FRAMES).contains(&frames)
        || !readable(set, header_len + frames * size_of::<FramePlanes>())
    {
        return;
    }
    let y = header.luma_height as usize;
    let c = header.chroma_height as usize;
    let mut planes = Vec::with_capacity(frames * 5);
    for i in 0..frames {
        let frame = (set + header_len + i * size_of::<FramePlanes>()) as *const FramePlanes;
        for (plane, height) in frame.read_unaligned().into_iter().zip([y, c, c, y, y]) {
            let Some(len) = (plane.pitch as usize).checked_mul(height) else {
                continue;
            };
            if readable(plane.buffer, len) {
                planes.push((plane.buffer, len));
            }
        }
    }
    planes.sort_unstable();
    planes.dedup();
    crate::xlog_info!(
        "Bink: frame buffers of {bink:#x} registered: {frames} frames of {}x{y}, planes (address, bytes) {planes:x?}, frame index {}",
        header.luma_width,
        header.frame_index
    );
    SETS.lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(bink, planes);
}

fn decoded(what: &'static str, bink: usize) {
    let planes = match SETS.lock().unwrap().as_ref().and_then(|m| m.get(&bink)) {
        Some(p) => p.clone(),
        None => return,
    };
    static FIRST: AtomicUsize = AtomicUsize::new(0);
    if FIRST.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_info!(
            "Bink: first decode marked written ({what} of {bink:#x}: {} planes, {} bytes)",
            planes.len(),
            planes.iter().map(|p| p.1).sum::<usize>()
        );
    }
    for (at, len) in planes {
        crate::hle::d3d12x::title_textures::mark_cpu_written(at, len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_all_bink2_frame_planes() {
        let mut set = [0u8; 0x108];
        let buffers = vec![[0u8; 1024]; 15];
        for (i, field) in [3u32, 24, 12, 16, 8, 2].into_iter().enumerate() {
            set[i * 4..i * 4 + 4].copy_from_slice(&field.to_le_bytes());
        }

        let mut expected = Vec::new();
        for (frame, pitches) in [
            [32u32, 16, 32, 32, 0],
            [64, 32, 16, 0, 0],
            [32, 16, 16, 64, 32],
        ]
        .into_iter()
        .enumerate()
        {
            for (plane, pitch) in pitches.into_iter().enumerate() {
                let i = frame * 5 + plane;
                let at = 0x18 + frame * 0x50 + plane * 0x10;
                let addr = buffers[i].as_ptr() as usize;
                set[at + 8..at + 12].copy_from_slice(&1u32.to_le_bytes());
                set[at + 12..at + 16].copy_from_slice(&pitch.max(16).to_le_bytes());
                if pitch != 0 {
                    set[at..at + 8].copy_from_slice(&(addr as u64).to_le_bytes());
                    let height = if plane == 1 || plane == 2 { 8 } else { 12 };
                    expected.push((addr, pitch as usize * height));
                }
            }
        }

        let set_addr = set.as_ptr() as usize;
        unsafe { register(set_addr, set_addr) };
        expected.sort_unstable();
        assert_eq!(SETS.lock().unwrap().as_ref().unwrap()[&set_addr], expected);

        set[..4].copy_from_slice(&1u32.to_le_bytes());
        set[0x14..0x18].fill(0);
        set[0x18..0x20].fill(0);
        unsafe { register(set_addr, set_addr) };
        let actual = SETS
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .remove(&set_addr)
            .unwrap();
        assert_eq!(actual, expected[1..4]);
    }
}
