#![allow(non_snake_case)]

use std::ffi::c_void;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::real::{orig_fn, slot_of};
use super::residency::{note_list_resources, note_list_untracked};

pub(super) const OPAQUE_METHODS: &[&str] = &[
    "DrawInstanced", "DrawIndexedInstanced", "Dispatch", "OMSetRenderTargets", "ExecuteBundle",
    "ClearDepthStencilView", "ClearRenderTargetView", "ClearUnorderedAccessViewUint",
    "ClearUnorderedAccessViewFloat", "ExecuteIndirect", "AtomicCopyBufferUINT",
    "AtomicCopyBufferUINT64", "WriteBufferImmediate", "BeginRenderPass", "InitializeMetaCommand",
    "ExecuteMetaCommand", "BuildRaytracingAccelerationStructure",
    "EmitRaytracingAccelerationStructurePostbuildInfo", "CopyRaytracingAccelerationStructure",
    "DispatchRays", "DispatchMesh", "Barrier", "SetProgram", "DispatchGraph", "WaitBarrier",
    "SignalBarrier", "CopyBufferRegions", "CopyTextureRegions", "CopyResources", "CopyTilesAsync",
    "ResolveSubresourceRegionAsync", "ResolveQueryDataAsync", "ClearBoundDepthStencilView",
    "ClearBoundRenderTargetViews", "ClearTextureSubresources", "FillBuffers",
];

type Obj = *mut c_void;

macro_rules! direct_uses {
    ($($method:ident($this:ident $(, $arg:ident: $ty:ty)*) => $record:expr;)*) => {
        pub(super) fn overrides(v: &mut Vec<(&'static str, usize)>) {
            v.extend_from_slice(&[$((stringify!($method), $method as usize)),*]);
        }

        $(unsafe extern "system" fn $method($this: Obj $(, $arg: $ty)*) {
            let f: unsafe extern "system" fn(Obj $(, $ty)*) =
                std::mem::transmute(orig_fn($this, slot_of($this, stringify!($method))));
            $record;
            f($this $(, $arg)*);
        })*
    };
}

direct_uses! {
    CopyBufferRegion(list, dst: Obj, dst_off: u64, src: Obj, src_off: u64, bytes: u64)
        => note_list_resources(list as usize, [dst as usize, src as usize]);
    CopyTextureRegion(list, dst: *const D3D12_TEXTURE_COPY_LOCATION, x: u32, y: u32, z: u32,
                      src: *const D3D12_TEXTURE_COPY_LOCATION, box_: *const D3D12_BOX)
        => note_list_resources(list as usize, [dst, src].into_iter().filter_map(|p|
            p.as_ref()?.pResource.as_ref().map(|r| r.as_raw() as usize)));
    CopyResource(list, dst: Obj, src: Obj)
        => note_list_resources(list as usize, [dst as usize, src as usize]);
    CopyTiles(list, tiled: Obj, coord: *const D3D12_TILED_RESOURCE_COORDINATE,
              region: *const D3D12_TILE_REGION_SIZE, buf: Obj, offset: u64, flags: D3D12_TILE_COPY_FLAGS)
        => note_list_resources(list as usize, [tiled as usize, buf as usize]);
    ResolveSubresource(list, dst: Obj, dst_sub: u32, src: Obj, src_sub: u32, fmt: i32)
        => note_list_resources(list as usize, [dst as usize, src as usize]);
    ResourceBarrier(list, n: u32, barriers: *const D3D12_RESOURCE_BARRIER)
        => note_barriers(list as usize, n, barriers);
    DiscardResource(list, res: Obj, desc: *const D3D12_DISCARD_REGION)
        => note_list_resources(list as usize, [res as usize]);
    ResolveQueryData(list, heap: Obj, ty: D3D12_QUERY_TYPE, first: u32, n: u32, dst: Obj, offset: u64)
        => note_list_resources(list as usize, [dst as usize]);
    SetPredication(list, buf: Obj, offset: u64, op: D3D12_PREDICATION_OP)
        => note_list_resources(list as usize, [buf as usize]);
    ResolveSubresourceRegion(list, dst: Obj, dst_sub: u32, x: u32, y: u32, src: Obj, src_sub: u32,
                             rect: *const windows::Win32::Foundation::RECT, fmt: i32, mode: D3D12_RESOLVE_MODE)
        => note_list_resources(list as usize, [dst as usize, src as usize]);
    RSSetShadingRateImage(list, img: Obj) => note_list_resources(list as usize, [img as usize]);
}

unsafe fn note_barriers(list: usize, n: u32, barriers: *const D3D12_RESOURCE_BARRIER) {
    if n == 0 || barriers.is_null() {
        return;
    }
    let mut opaque = false;
    let resources = std::slice::from_raw_parts(barriers, n as usize).iter().flat_map(|b| {
        let raw = |r: Option<&ID3D12Resource>| r.map_or(0, |r| r.as_raw() as usize);
        match b.Type {
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION => [raw(b.Anonymous.Transition.pResource.as_ref()), 0],
            D3D12_RESOURCE_BARRIER_TYPE_ALIASING => {
                let a = &b.Anonymous.Aliasing;
                let before = raw(a.pResourceBefore.as_ref());
                let after = raw(a.pResourceAfter.as_ref());
                opaque |= before == 0 || after == 0;
                [before, after]
            }
            D3D12_RESOURCE_BARRIER_TYPE_UAV => [raw(b.Anonymous.UAV.pResource.as_ref()), 0],
            _ => [0, 0],
        }
    });
    note_list_resources(list, resources);
    if opaque {
        note_list_untracked(list);
    }
}
