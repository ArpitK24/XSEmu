//! [`xcl_dxil::meta`] and [`xcl_dxil::flags`] against DXC: every case is a small shader of one
//! stage with the `ShaderInfo` a front end would build for it. With the oracle (the
//! `XCL_DXIL_DXC` environment variable names DXC's folder, as in the brief) each is compiled
//! with `dxc -validator-version 1.8` and the metadata this crate emits for the hand-built
//! `ShaderInfo` is compared node for node with DXC's. Without the oracle the tests pass with a
//! note.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use xcl_dxil::consts::*;
use xcl_dxil::ir::{Body, Module};
use xcl_dxil::shader::*;
use xcl_dxil::{bcreader, bcwriter, flags, irprint, meta};

struct Case {
    name: &'static str,
    stage: &'static str,
    source: &'static str,
    info: fn() -> ShaderInfo,
}

/// A signature element: name, semantic indices, kind, component type, interpolation, rows,
/// columns, start row, start column, stream, usage mask, dynamic index mask.
#[allow(clippy::too_many_arguments)]
fn el(name: &str, indices: &[u32], kind: SemanticKind, comp_type: ComponentType, interp: InterpolationMode, rows: u32, cols: u32, start_row: i32, start_col: i32, stream: u32, usage_mask: u8, dynamic_index_mask: u8) -> SigElement {
    SigElement { name: name.into(), indices: indices.to_vec(), kind, comp_type, interp, rows, cols, start_row, start_col, stream, usage_mask, dynamic_index_mask }
}

fn res(class: ResourceClass, id: u32, name: &str, kind: ResourceKind, lower_bound: u32, space: u32, set: impl Fn(&mut Resource)) -> Resource {
    let mut r = Resource::new(class, id, name, kind, lower_bound, space);
    set(&mut r);
    r
}

#[rustfmt::skip]
const CASES: &[Case] = &[
    Case {
        name: "as_basic",
        stage: "as",
        source: r#"RWByteAddressBuffer io : register(u0);
struct Payload { float4 a; uint b; };
[numthreads(4, 1, 1)]
void main(uint gi : SV_GroupIndex)
{
    Payload p;
    p.a = float4(1, 2, 3, 4);
    p.b = io.Load(0);
    DispatchMesh(1, p.b, 1, p);
}
"#,
        // DXC: flags 0x10, view id state []
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Amplification, "main", StageProps::Amplification { numthreads: [4, 1, 1], payload_bytes: 20 });
            i.resources = vec![
                res(ResourceClass::UAV, 0, "r0", ResourceKind::RawBuffer, 0, 0, |_r| {}),
            ];
            i.inputs = vec![];
            i.outputs = vec![];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "cs_atomic64",
        stage: "cs",
        source: r#"RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
RWTexture2D<uint64_t> tex64 : register(u2);
groupshared uint64_t gs64;
[numthreads(1, 1, 1)]
void main(uint gi : SV_GroupIndex)
{
    uint64_t v = (uint64_t)io.Load(0), old = 0, o2 = 0, o3 = 0, o4 = 0;
    res.InterlockedAdd64(8u, v, old);
    res.InterlockedCompareExchange64(16u, old, v, o2);
    RWByteAddressBuffer mb = ResourceDescriptorHeap[gi];
    mb.InterlockedMax64(8u, v, o3);
    InterlockedAdd(tex64[uint2(0, 0)], v, o4);
    InterlockedAdd(gs64, v);
    io.Store(0u, (uint)(old + o2 + o3 + o4));
}
"#,
        // DXC: flags 0x158100010, view id state []
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [1, 1, 1], wave_size: None });
            i.resources = vec![
                res(ResourceClass::UAV, 0, "r0", ResourceKind::RawBuffer, 0, 0, |_r| {}),
                res(ResourceClass::UAV, 1, "r1", ResourceKind::RawBuffer, 1, 0, |_r| {}),
                res(ResourceClass::UAV, 2, "r2", ResourceKind::Texture2D, 2, 0, |r| {r.comp_type = ComponentType::U32; r.comp_count = 1;}),
            ];
            i.inputs = vec![];
            i.outputs = vec![];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "cs_basic",
        stage: "cs",
        source: r#"RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
[WaveSize(32)]
[numthreads(32, 1, 1)]
void main(uint gi : SV_GroupIndex)
{
    uint lane = WaveGetLaneIndex();
    uint a = io.Load(4u * gi);
    uint64_t w = (uint64_t)a * (uint64_t)lane;
    uint s = WaveActiveSum(a);
    res.Store(4u * gi, (uint)(w >> 7) + s);
}
"#,
        // DXC: flags 0x180010, view id state []
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [32, 1, 1], wave_size: Some(32) });
            i.resources = vec![
                res(ResourceClass::UAV, 0, "r0", ResourceKind::RawBuffer, 0, 0, |_r| {}),
                res(ResourceClass::UAV, 1, "r1", ResourceKind::RawBuffer, 1, 0, |_r| {}),
            ];
            i.inputs = vec![];
            i.outputs = vec![];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "cs_dbl_noext",
        stage: "cs",
        source: r#"RWByteAddressBuffer io : register(u0);
[numthreads(1, 1, 1)]
void main(uint gi : SV_GroupIndex)
{
    uint lo = io.Load(0), hi = io.Load(4);
    double d = asdouble(lo, hi);
    double e = d * 3.0 + d;
    uint l, h;
    asuint(e, l, h);
    io.Store(8u, l + h);
}
"#,
        // DXC: flags 0x14, view id state []
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [1, 1, 1], wave_size: None });
            i.resources = vec![
                res(ResourceClass::UAV, 0, "r0", ResourceKind::RawBuffer, 0, 0, |_r| {}),
            ];
            i.inputs = vec![];
            i.outputs = vec![];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "cs_deriv",
        stage: "cs",
        source: r#"RWByteAddressBuffer io : register(u0);
Texture2D<float4> tex : register(t0);
SamplerState samp : register(s0);
[numthreads(2, 2, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    float2 uv = float2(id.xy) * 0.5;
    float4 c = tex.Sample(samp, uv) + tex.SampleGrad(samp, uv, ddx(uv), ddy(uv), int2(0, 0), 2.0);
    io.Store(0u, asuint(c.x + ddx_fine(uv.x)));
}
"#,
        // DXC: flags 0x1010, view id state []
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [2, 2, 1], wave_size: None });
            i.resources = vec![
                res(ResourceClass::SRV, 0, "r0", ResourceKind::Texture2D, 0, 0, |r| {r.comp_type = ComponentType::F32; r.comp_count = 4;}),
                res(ResourceClass::UAV, 0, "r1", ResourceKind::RawBuffer, 0, 0, |_r| {}),
                res(ResourceClass::Sampler, 0, "r2", ResourceKind::Sampler, 0, 0, |_r| {}),
            ];
            i.inputs = vec![];
            i.outputs = vec![];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "cs_double",
        stage: "cs",
        source: r#"RWByteAddressBuffer io : register(u0);
[numthreads(1, 1, 1)]
void main(uint gi : SV_GroupIndex)
{
    uint lo = io.Load(0), hi = io.Load(4);
    double d = asdouble(lo, hi);
    double e = d / 3.0 + (double)(int)lo + fma(d, d, d);
    uint l, h;
    asuint(e, l, h);
    io.Store(8u, l + h + (uint)e);
}
"#,
        // DXC: flags 0x54, view id state []
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [1, 1, 1], wave_size: None });
            i.resources = vec![
                res(ResourceClass::UAV, 0, "r0", ResourceKind::RawBuffer, 0, 0, |_r| {}),
            ];
            i.inputs = vec![];
            i.outputs = vec![];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "cs_heap",
        stage: "cs",
        source: r#"RWByteAddressBuffer io : register(u0);
SamplerState xcl_ss0 : register(s0, space1);
SamplerComparisonState xcl_ss1 : register(s1, space1);
groupshared uint xcl_lds[64];
[numthreads(64, 1, 1)]
void main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID)
{
    uint i = io.Load(0);
    Texture2D<float4> t = ResourceDescriptorHeap[NonUniformResourceIndex(i)];
    SamplerState s = SamplerDescriptorHeap[NonUniformResourceIndex(i & 0xFFFu)];
    float4 c = t.SampleLevel(s, float2(0.5, 0.25), 0.0);
    c += t.SampleLevel(xcl_ss0, float2(0.5, 0.25), 1.0);
    c.x += t.SampleCmpLevelZero(xcl_ss1, float2(0.5, 0.25), 0.5);
    RWTexture2D<float4> u = ResourceDescriptorHeap[NonUniformResourceIndex(i + 1u)];
    c += u[uint2(1, 2)];
    u[uint2(3, 4)] = c;
    globallycoherent RWByteAddressBuffer mb = ResourceDescriptorHeap[i + 2u];
    uint old;
    mb.InterlockedAdd(16u, 1u, old);
    xcl_lds[gi] = old;
    GroupMemoryBarrierWithGroupSync();
    io.Store(8u, xcl_lds[(gi + 1u) & 63u] + asuint(c.x));
}
"#,
        // DXC: flags 0xc0002010, view id state []
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [64, 1, 1], wave_size: None });
            i.resources = vec![
                res(ResourceClass::UAV, 0, "r0", ResourceKind::RawBuffer, 0, 0, |_r| {}),
                res(ResourceClass::Sampler, 0, "r1", ResourceKind::Sampler, 0, 1, |_r| {}),
                res(ResourceClass::Sampler, 1, "r2", ResourceKind::Sampler, 1, 1, |r| {r.sampler_kind = SamplerKind::Comparison;}),
            ];
            i.inputs = vec![];
            i.outputs = vec![];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "cs_min16",
        stage: "cs",
        source: r#"RWByteAddressBuffer io : register(u0);
[numthreads(1, 1, 1)]
void main(uint gi : SV_GroupIndex)
{
    min16float a = (min16float)asfloat(io.Load(0));
    min16uint b = (min16uint)io.Load(4);
    a = a * a + (min16float)b;
    io.Store(8u, asuint((float)a));
}
"#,
        // DXC: flags 0x30, view id state []
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [1, 1, 1], wave_size: None });
            i.resources = vec![
                res(ResourceClass::UAV, 0, "r0", ResourceKind::RawBuffer, 0, 0, |_r| {}),
            ];
            i.inputs = vec![];
            i.outputs = vec![];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "cs_plain",
        stage: "cs",
        source: r#"RWByteAddressBuffer io : register(u0);
[numthreads(8, 4, 2)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID)
{
    io.Store(4u * dtid.x, gid.y + gtid.z);
}
"#,
        // DXC: flags 0x10, view id state []
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [8, 4, 2], wave_size: None });
            i.resources = vec![
                res(ResourceClass::UAV, 0, "r0", ResourceKind::RawBuffer, 0, 0, |_r| {}),
            ];
            i.inputs = vec![];
            i.outputs = vec![];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "cs_resources",
        stage: "cs",
        source: r#"cbuffer Params : register(b2, space3) { float4 scale; uint count; float3 bias; };
struct Elem { float3 p; uint id; };
Texture2D<float4> tex : register(t0);
Texture2DMS<float4, 4> texms : register(t1);
Texture3D<uint2> tex3 : register(t2);
StructuredBuffer<Elem> sbuf : register(t3);
ByteAddressBuffer raw : register(t4);
Texture2D<float4> texarr[4] : register(t10);
Texture2D<float> texunb[] : register(t0, space2);
TextureCubeArray<float4> cube : register(t5);
Buffer<int4> tbuf : register(t6);
RWStructuredBuffer<Elem> usb : register(u0);
RWTexture2D<float4> utex : register(u1);
RWTexture3D<uint> utex3 : register(u2);
RWBuffer<float2> ubuf : register(u3);
RWByteAddressBuffer uraw[8] : register(u4);
RWTexture2DArray<int> uarr : register(u12);
RWTexture1D<float> u1d : register(u13);
SamplerState samp : register(s0);
SamplerComparisonState cmp : register(s1);
[numthreads(4, 4, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    float4 c = tex.SampleLevel(samp, float2(0.5, 0.5), 0) * scale + float4(bias, 0);
    c += texms.Load(int2(id.xy), 1);
    c.xy += (float2)tex3.Load(int4(id, 0));
    Elem e = sbuf[id.x];
    c.xyz += e.p;
    c.x += (float)raw.Load(4 * id.y);
    c += texarr[id.x & 3].Load(int3(id.xy, 0));
    c.x += texunb[id.z].Load(int3(id.xy, 0));
    c += cube.SampleLevel(samp, float4(1, 0, 0, 0), 0);
    c.x += cube.SampleCmpLevelZero(cmp, float4(1, 0, 0, 0), 0.5);
    c += (float4)tbuf[id.x];
    uint n = usb.IncrementCounter();
    e.id = n + count;
    usb[n] = e;
    c += utex[id.xy];
    utex[id.xy] = c;
    utex3[id] = n;
    ubuf[id.x] = c.xy;
    uraw[id.x & 7].Store(0, n);
    uarr[id] = (int)n;
    u1d[id.x] = c.x;
}
"#,
        // DXC: flags 0xa010, view id state []
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [4, 4, 1], wave_size: None });
            i.resources = vec![
                res(ResourceClass::SRV, 0, "r0", ResourceKind::Texture2D, 0, 0, |r| {r.comp_type = ComponentType::F32; r.comp_count = 4;}),
                res(ResourceClass::SRV, 1, "r1", ResourceKind::Texture2DMS, 1, 0, |r| {r.comp_type = ComponentType::F32; r.comp_count = 4; r.sample_count = 4;}),
                res(ResourceClass::SRV, 2, "r2", ResourceKind::Texture3D, 2, 0, |r| {r.comp_type = ComponentType::U32; r.comp_count = 2;}),
                res(ResourceClass::SRV, 3, "r3", ResourceKind::StructuredBuffer, 3, 0, |r| {r.struct_stride = 16;}),
                res(ResourceClass::SRV, 4, "r4", ResourceKind::RawBuffer, 4, 0, |_r| {}),
                res(ResourceClass::SRV, 5, "r5", ResourceKind::Texture2D, 10, 0, |r| {r.range_size = 4; r.comp_type = ComponentType::F32; r.comp_count = 4;}),
                res(ResourceClass::SRV, 6, "r6", ResourceKind::Texture2D, 0, 2, |r| {r.range_size = u32::MAX; r.comp_type = ComponentType::F32; r.comp_count = 1;}),
                res(ResourceClass::SRV, 7, "r7", ResourceKind::TextureCubeArray, 5, 0, |r| {r.comp_type = ComponentType::F32; r.comp_count = 4;}),
                res(ResourceClass::SRV, 8, "r8", ResourceKind::TypedBuffer, 6, 0, |r| {r.comp_type = ComponentType::I32; r.comp_count = 4;}),
                res(ResourceClass::UAV, 0, "r9", ResourceKind::StructuredBuffer, 0, 0, |r| {r.struct_stride = 16; r.has_counter = true;}),
                res(ResourceClass::UAV, 1, "r10", ResourceKind::Texture2D, 1, 0, |r| {r.comp_type = ComponentType::F32; r.comp_count = 4;}),
                res(ResourceClass::UAV, 2, "r11", ResourceKind::Texture3D, 2, 0, |r| {r.comp_type = ComponentType::U32; r.comp_count = 1;}),
                res(ResourceClass::UAV, 3, "r12", ResourceKind::TypedBuffer, 3, 0, |r| {r.comp_type = ComponentType::F32; r.comp_count = 2;}),
                res(ResourceClass::UAV, 4, "r13", ResourceKind::RawBuffer, 4, 0, |r| {r.range_size = 8;}),
                res(ResourceClass::UAV, 5, "r14", ResourceKind::Texture2DArray, 12, 0, |r| {r.comp_type = ComponentType::I32; r.comp_count = 1;}),
                res(ResourceClass::UAV, 6, "r15", ResourceKind::Texture1D, 13, 0, |r| {r.comp_type = ComponentType::F32; r.comp_count = 1;}),
                res(ResourceClass::CBuffer, 0, "Params", ResourceKind::CBuffer, 2, 3, |r| {r.cbuffer_bytes = 32;}),
                res(ResourceClass::Sampler, 0, "r17", ResourceKind::Sampler, 0, 0, |_r| {}),
                res(ResourceClass::Sampler, 1, "r18", ResourceKind::Sampler, 1, 0, |r| {r.sampler_kind = SamplerKind::Comparison;}),
            ];
            i.inputs = vec![];
            i.outputs = vec![];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "ds_flow",
        stage: "ds",
        source: r#"struct CP { float4 pos : POSITION; float2 uv : TEXCOORD0; uint m : MODE0; };
struct PC { float edges[4] : SV_TessFactor; float inside[2] : SV_InsideTessFactor; float3 extra : EXTRA; uint which : WHICH; };
struct Out { float4 pos : SV_Position; float2 uv : TEXCOORD0; float k : COLOR0; uint v : COLOR1; };
[domain("quad")]
Out main(PC pc, float2 loc : SV_DomainLocation, const OutputPatch<CP, 4> op, uint view : SV_ViewID)
{
    Out o;
    float4 p = 0.0;
    for (uint n = 0; n < 4; ++n) {
        if (op[n].m != 0u) { p += op[n].pos * loc.x; }
    }
    o.pos = p;
    o.uv = pc.which > 1u ? float2(1.0, 1.0) : op[pc.which & 3u].uv;
    float k = 0.0;
    if (pc.edges[2] > 1.0) { k = 1.0; } else if (pc.inside[1] > 1.0) { k = 2.0; }
    o.k = k + pc.extra.y;
    o.v = view == 1 ? 4u : 5u;
    return o;
}
"#,
        // DXC: flags 0x200000, view id state [9, 9, 256, 1, 2, 4, 8, 16, 32, 0, 0, 15, 24, 0, 64, 0, 0, 48, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Domain, "main", StageProps::Domain { domain: TessellatorDomain::Quad, input_control_points: 4 });
            i.resources = vec![];
            i.inputs = vec![
            el("POSITION", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 1, 0, 0, 3, 0),
            el("MODE", &[0], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 0, 0, 1, 0),
        ];
            i.outputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 1, 0, 0, 3, 0),
            el("COLOR", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 1, 1, 2, 0, 1, 0),
            el("COLOR", &[1], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 0, 0, 1, 0),
        ];
            i.patch_or_prim = vec![
            el("SV_TessFactor", &[0, 1, 2, 3], SemanticKind::TessFactor, ComponentType::F32, InterpolationMode::Undefined, 4, 1, 0, 3, 0, 1, 0),
            el("SV_InsideTessFactor", &[0, 1], SemanticKind::InsideTessFactor, ComponentType::F32, InterpolationMode::Undefined, 2, 1, 4, 3, 0, 1, 0),
            el("EXTRA", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Undefined, 1, 3, 0, 0, 0, 2, 0),
            el("WHICH", &[0], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Undefined, 1, 1, 1, 0, 0, 1, 0),
        ];
            i
        },
    },
    Case {
        name: "ds_tri",
        stage: "ds",
        source: r#"struct CP { float4 pos : POSITION; float2 uv : TEXCOORD0; };
struct PC { float edges[3] : SV_TessFactor; float inside : SV_InsideTessFactor; float3 extra : EXTRA; };
struct Out { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
[domain("tri")]
Out main(PC pc, float3 bary : SV_DomainLocation, const OutputPatch<CP, 3> op)
{
    Out o;
    o.pos = op[0].pos * bary.x + op[1].pos * bary.y + op[2].pos * bary.z + float4(pc.extra, pc.inside);
    o.uv = op[0].uv + pc.edges[1];
    return o;
}
"#,
        // DXC: flags 0x0, view id state [6, 6, 1, 2, 4, 8, 16, 32, 13, 1, 2, 4, 0, 0, 0, 0, 48, 0, 0, 0, 0, 8]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Domain, "main", StageProps::Domain { domain: TessellatorDomain::Tri, input_control_points: 3 });
            i.resources = vec![];
            i.inputs = vec![
            el("POSITION", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 1, 0, 0, 3, 0),
        ];
            i.outputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 1, 0, 0, 3, 0),
        ];
            i.patch_or_prim = vec![
            el("SV_TessFactor", &[0, 1, 2], SemanticKind::TessFactor, ComponentType::F32, InterpolationMode::Undefined, 3, 1, 0, 3, 0, 1, 0),
            el("SV_InsideTessFactor", &[0], SemanticKind::InsideTessFactor, ComponentType::F32, InterpolationMode::Undefined, 1, 1, 3, 0, 0, 1, 0),
            el("EXTRA", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Undefined, 1, 3, 0, 0, 0, 7, 0),
        ];
            i
        },
    },
    Case {
        name: "gs_adj",
        stage: "gs",
        source: r#"struct V { float4 pos : SV_Position; };
[maxvertexcount(2)]
void main(lineadj V v[4], inout LineStream<V> s)
{
    V a; a.pos = v[0].pos + v[3].pos; s.Append(a);
    V b; b.pos = v[1].pos + v[2].pos; s.Append(b);
}
"#,
        // DXC: flags 0x0, view id state [4, 4, 1, 2, 4, 8, 0, 0, 0]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Geometry, "main", StageProps::Geometry { input_primitive: InputPrimitive::LineWithAdjacency, max_vertex_count: 2, active_streams: 1, stream_topology: PrimitiveTopology::LineStrip, instance_count: 1 });
            i.resources = vec![];
            i.inputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
        ];
            i.outputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
        ];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "gs_flow",
        stage: "gs",
        source: r#"struct In { float4 pos : SV_Position; float2 uv : TEXCOORD0; uint sel : SEL0; };
struct Out0 { float4 pos : SV_Position; float2 uv : TEXCOORD0; uint rt : SV_RenderTargetArrayIndex; };
struct Out1 { float3 n : NORMAL0; uint pid : SV_PrimitiveID; };
[maxvertexcount(9)]
void main(triangle In v[3], uint pid : SV_PrimitiveID, uint view : SV_ViewID, inout PointStream<Out0> s0, inout PointStream<Out1> s1)
{
    for (uint k = 0; k < 3; ++k) {
        Out0 a;
        a.pos = v[k].pos;
        a.uv = v[k].sel > 2u ? float2(1.0, 0.0) : v[(k + 1) % 3].uv;
        a.rt = view;
        s0.Append(a);
        if (v[k].uv.x > 0.5) {
            Out1 b;
            b.n = float3(0.0, 1.0, 0.0);
            b.pid = pid;
            s1.Append(b);
        }
    }
}
"#,
        // DXC: flags 0x200000, view id state [9, 9, 256, 1, 2, 4, 8, 16, 32, 0, 0, 48, 5, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Geometry, "main", StageProps::Geometry { input_primitive: InputPrimitive::Triangle, max_vertex_count: 9, active_streams: 3, stream_topology: PrimitiveTopology::PointList, instance_count: 1 });
            i.resources = vec![];
            i.inputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 1, 0, 0, 3, 0),
            el("SEL", &[0], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 0, 0, 1, 0),
            el("SV_PrimitiveID", &[0], SemanticKind::PrimitiveID, ComponentType::U32, InterpolationMode::Undefined, 1, 1, -1, -1, 0, 0, 0),
        ];
            i.outputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 1, 0, 0, 3, 0),
            el("SV_RenderTargetArrayIndex", &[0], SemanticKind::RenderTargetArrayIndex, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 0, 0, 1, 0),
            el("NORMAL", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 3, 0, 0, 1, 7, 0),
            el("SV_PrimitiveID", &[0], SemanticKind::PrimitiveID, ComponentType::U32, InterpolationMode::Constant, 1, 1, 1, 0, 1, 1, 0),
        ];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "gs_rect",
        stage: "gs",
        source: r#"struct XclOut {
    float4 pos : SV_Position;
    float4 p0 : TEXCOORD0;
    uint p1 : COLOR1;
};
[maxvertexcount(6)]
void main(triangle XclOut v[3], inout TriangleStream<XclOut> s)
{
    uint c = 3u;
    if (v[0].pos.w > 0.0 && v[1].pos.w == v[0].pos.w) c = 0u;
    if (c == 3u) {
        s.Append(v[0]);
        s.Append(v[1]);
        s.Append(v[2]);
        return;
    }
    XclOut d = v[0];
    d.pos = v[1].pos + v[2].pos - v[0].pos;
    d.p0 = v[1].p0 + v[2].p0 - v[0].p0;
    s.Append(v[0]);
    s.Append(v[1]);
    s.Append(d);
    s.RestartStrip();
    s.Append(v[0]);
    s.Append(d);
    s.Append(v[2]);
}
"#,
        // DXC: flags 0x0, view id state [9, 9, 1, 2, 4, 511, 16, 32, 64, 128, 256, 0, 0, 0]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Geometry, "main", StageProps::Geometry { input_primitive: InputPrimitive::Triangle, max_vertex_count: 6, active_streams: 1, stream_topology: PrimitiveTopology::TriangleStrip, instance_count: 1 });
            i.resources = vec![];
            i.inputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 1, 0, 0, 15, 0),
            el("COLOR", &[1], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 0, 0, 1, 0),
        ];
            i.outputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 1, 0, 0, 15, 0),
            el("COLOR", &[1], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 0, 0, 1, 0),
        ];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "gs_streams",
        stage: "gs",
        source: r#"struct In { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct Out0 { float4 pos : SV_Position; uint rt : SV_RenderTargetArrayIndex; };
struct Out1 { float2 uv : TEXCOORD0; uint pid : SV_PrimitiveID; };
[instance(3)]
[maxvertexcount(4)]
void main(point In v[1], uint pid : SV_PrimitiveID, uint inst : SV_GSInstanceID, inout PointStream<Out0> s0, inout PointStream<Out1> s1)
{
    Out0 a; a.pos = v[0].pos; a.rt = inst;
    s0.Append(a);
    Out1 b; b.uv = v[0].uv; b.pid = pid;
    s1.Append(b);
}
"#,
        // DXC: flags 0x0, view id state [6, 5, 1, 2, 4, 8, 0, 0, 5, 0, 0, 0, 0, 1, 2, 0, 0]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Geometry, "main", StageProps::Geometry { input_primitive: InputPrimitive::Point, max_vertex_count: 4, active_streams: 3, stream_topology: PrimitiveTopology::PointList, instance_count: 3 });
            i.resources = vec![];
            i.inputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 1, 0, 0, 3, 0),
            el("SV_PrimitiveID", &[0], SemanticKind::PrimitiveID, ComponentType::U32, InterpolationMode::Undefined, 1, 1, -1, -1, 0, 0, 0),
        ];
            i.outputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("SV_RenderTargetArrayIndex", &[0], SemanticKind::RenderTargetArrayIndex, ComponentType::U32, InterpolationMode::Constant, 1, 1, 1, 0, 0, 1, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 0, 0, 1, 3, 0),
            el("SV_PrimitiveID", &[0], SemanticKind::PrimitiveID, ComponentType::U32, InterpolationMode::Constant, 1, 1, 1, 0, 1, 1, 0),
        ];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "hs_flow",
        stage: "hs",
        source: r#"struct VIn { float4 pos : POSITION; float2 uv : TEXCOORD0; uint m : MODE0; };
struct CP { float4 pos : POSITION; float2 uv : TEXCOORD0; uint m : MODE0; };
struct PC { float edges[4] : SV_TessFactor; float inside[2] : SV_InsideTessFactor; float3 extra : EXTRA; uint which : WHICH; };
PC pcf(InputPatch<VIn, 4> ip, const OutputPatch<CP, 4> op, uint pid : SV_PrimitiveID, uint view : SV_ViewID)
{
    PC pc;
    float f = 1.0;
    for (uint n = 0; n < 4; ++n) {
        if (ip[n].m > 1u) { f = 2.0; }
        pc.edges[n] = op[n].pos.x > 0.0 ? 3.0 : f;
    }
    pc.inside[0] = view == 0 ? 1.0 : 2.0;
    pc.inside[1] = ip[pid & 3u].uv.y;
    pc.extra = float3(op[1].uv, 0.0);
    pc.which = pid > 7u ? 1u : 0u;
    return pc;
}
[domain("quad")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(4)]
[patchconstantfunc("pcf")]
CP main(InputPatch<VIn, 4> ip, uint id : SV_OutputControlPointID, uint view : SV_ViewID)
{
    CP o;
    o.pos = ip[id].pos;
    o.uv = ip[id].m != 0u ? float2(1.0, 0.0) : ip[(id + 1) & 3u].uv;
    o.m = view != 0 ? 1u : ip[0].m;
    return o;
}
"#,
        // DXC: flags 0x200000, view id state [9, 9, 256, 1, 2, 4, 8, 16, 32, 0, 0, 304, 24, 524288, 34952, 0, 0, 0, 1, 8388610, 0, 0, 34955]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Hull, "main", StageProps::Hull { patch_constant_fn: "\u{1}?pcf@@YA?AUPC@@V?$InputPatch@UVIn@@$03@@V?$OutputPatch@UCP@@$03@@II@Z".into(), input_control_points: 4, output_control_points: 4, domain: TessellatorDomain::Quad, partitioning: TessellatorPartitioning::Integer, output_primitive: TessellatorOutputPrimitive::TriangleCW, max_tess_factor: 64.0 });
            i.resources = vec![];
            i.inputs = vec![
            el("POSITION", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 1, 0, 0, 3, 0),
            el("MODE", &[0], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 0, 0, 1, 0),
        ];
            i.outputs = vec![
            el("POSITION", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 1, 0, 0, 3, 0),
            el("MODE", &[0], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 0, 0, 1, 0),
        ];
            i.patch_or_prim = vec![
            el("SV_TessFactor", &[0, 1, 2, 3], SemanticKind::TessFactor, ComponentType::F32, InterpolationMode::Undefined, 4, 1, 0, 3, 0, 1, 0),
            el("SV_InsideTessFactor", &[0, 1], SemanticKind::InsideTessFactor, ComponentType::F32, InterpolationMode::Undefined, 2, 1, 4, 3, 0, 1, 0),
            el("EXTRA", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Undefined, 1, 3, 0, 0, 0, 7, 0),
            el("WHICH", &[0], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Undefined, 1, 1, 1, 0, 0, 1, 0),
        ];
            i
        },
    },
    Case {
        name: "hs_quad",
        stage: "hs",
        source: r#"struct CP { float4 pos : POSITION; };
struct PC { float edges[4] : SV_TessFactor; float inside[2] : SV_InsideTessFactor; };
PC pcf(InputPatch<CP, 16> ip)
{
    PC o;
    o.edges[0] = o.edges[1] = o.edges[2] = o.edges[3] = ip[0].pos.x;
    o.inside[0] = o.inside[1] = 2.0;
    return o;
}
[domain("quad")]
[partitioning("integer")]
[outputtopology("triangle_ccw")]
[outputcontrolpoints(4)]
[patchconstantfunc("pcf")]
CP main(InputPatch<CP, 16> ip, uint i : SV_OutputControlPointID)
{
    return ip[i];
}
"#,
        // DXC: flags 0x0, view id state [4, 4, 1, 2, 4, 8, 24, 34952, 0, 0, 0]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Hull, "main", StageProps::Hull { patch_constant_fn: "\u{1}?pcf@@YA?AUPC@@V?$InputPatch@UCP@@$0BA@@@@Z".into(), input_control_points: 16, output_control_points: 4, domain: TessellatorDomain::Quad, partitioning: TessellatorPartitioning::Integer, output_primitive: TessellatorOutputPrimitive::TriangleCCW, max_tess_factor: 64.0 });
            i.resources = vec![];
            i.inputs = vec![
            el("POSITION", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 0, 0, 0, 15, 0),
        ];
            i.outputs = vec![
            el("POSITION", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 0, 0, 0, 15, 0),
        ];
            i.patch_or_prim = vec![
            el("SV_TessFactor", &[0, 1, 2, 3], SemanticKind::TessFactor, ComponentType::F32, InterpolationMode::Undefined, 4, 1, 0, 3, 0, 1, 0),
            el("SV_InsideTessFactor", &[0, 1], SemanticKind::InsideTessFactor, ComponentType::F32, InterpolationMode::Undefined, 2, 1, 4, 3, 0, 1, 0),
        ];
            i
        },
    },
    Case {
        name: "hs_tri",
        stage: "hs",
        source: r#"struct CP { float4 pos : POSITION; float2 uv : TEXCOORD0; };
struct PC { float edges[3] : SV_TessFactor; float inside : SV_InsideTessFactor; float3 extra : EXTRA; };
PC pcf(InputPatch<CP, 3> ip, uint pid : SV_PrimitiveID)
{
    PC o;
    o.edges[0] = ip[0].pos.x; o.edges[1] = ip[1].pos.y; o.edges[2] = ip[2].pos.z;
    o.inside = (float)pid;
    o.extra = ip[0].pos.xyz;
    return o;
}
[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("pcf")]
[maxtessfactor(64.0)]
CP main(InputPatch<CP, 3> ip, uint i : SV_OutputControlPointID)
{
    CP o;
    o.pos = ip[i].pos;
    o.uv = ip[i].uv * 2.0;
    return o;
}
"#,
        // DXC: flags 0x0, view id state [6, 6, 1, 2, 4, 8, 16, 32, 13, 9, 130, 2052, 0, 0, 0]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Hull, "main", StageProps::Hull { patch_constant_fn: "\u{1}?pcf@@YA?AUPC@@V?$InputPatch@UCP@@$02@@I@Z".into(), input_control_points: 3, output_control_points: 3, domain: TessellatorDomain::Tri, partitioning: TessellatorPartitioning::FractionalOdd, output_primitive: TessellatorOutputPrimitive::TriangleCW, max_tess_factor: 64.0 });
            i.resources = vec![];
            i.inputs = vec![
            el("POSITION", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 1, 0, 0, 3, 0),
        ];
            i.outputs = vec![
            el("POSITION", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 1, 0, 0, 3, 0),
        ];
            i.patch_or_prim = vec![
            el("SV_TessFactor", &[0, 1, 2], SemanticKind::TessFactor, ComponentType::F32, InterpolationMode::Undefined, 3, 1, 0, 3, 0, 1, 0),
            el("SV_InsideTessFactor", &[0], SemanticKind::InsideTessFactor, ComponentType::F32, InterpolationMode::Undefined, 1, 1, 3, 0, 0, 1, 0),
            el("EXTRA", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Undefined, 1, 3, 0, 0, 0, 7, 0),
        ];
            i
        },
    },
    Case {
        name: "ms_flow",
        stage: "ms",
        source: r#"struct Payload { float4 a; uint b; };
struct Vert { float4 pos : SV_Position; float2 uv : TEXCOORD0; uint v : COLOR0; };
struct Prim { uint id : SV_PrimitiveID; float3 n : NORMAL; bool cull : SV_CullPrimitive; };
[outputtopology("triangle")]
[numthreads(4, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint view : SV_ViewID, in payload Payload pl, out vertices Vert verts[6], out indices uint3 tris[2], out primitives Prim prims[2])
{
    SetMeshOutputCounts(6, 2);
    for (uint k = 0; k < 6; ++k) {
        Vert v;
        v.pos = float4(k, gtid.x, 0.0, 1.0);
        v.uv = view == 0 ? float2(0.0, 0.0) : float2(1.0, 1.0);
        v.v = pl.b > k ? 1u : 0u;
        verts[k] = v;
    }
    for (uint t = 0; t < 2; ++t) {
        tris[t] = uint3(3 * t, 3 * t + 1, 3 * t + 2);
        Prim p;
        p.id = t;
        p.n = float3(0.0, 0.0, view);
        p.cull = view > 1;
        prims[t] = p;
    }
}
"#,
        // DXC: flags 0x200000, view id state [0, 9, 48, 7, 64]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Mesh, "main", StageProps::Mesh { numthreads: [4, 1, 1], max_vertices: 6, max_primitives: 2, topology: MeshOutputTopology::Triangle, payload_bytes: 20 });
            i.resources = vec![];
            i.inputs = vec![];
            i.outputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 1, 0, 0, 3, 0),
            el("COLOR", &[0], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 0, 0, 1, 0),
        ];
            i.patch_or_prim = vec![
            el("SV_PrimitiveID", &[0], SemanticKind::PrimitiveID, ComponentType::U32, InterpolationMode::Constant, 1, 1, 0, 0, 0, 1, 0),
            el("NORMAL", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Constant, 1, 3, 1, 0, 0, 7, 0),
            el("SV_CullPrimitive", &[0], SemanticKind::CullPrimitive, ComponentType::I1, InterpolationMode::Constant, 1, 1, -1, -1, 0, 1, 0),
        ];
            i
        },
    },
    Case {
        name: "ms_pass",
        stage: "ms",
        source: r#"RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
struct XclVert {
    float4 pos : SV_Position;
    float4 p0 : TEXCOORD0;
    uint2 p1 : COLOR1;
    float p2 : SV_ClipDistance0;
};
[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID, out vertices XclVert verts[64], out indices uint3 tris[42])
{
    uint rec = gid.x * 100u;
    uint nv = min(res.Load(rec * 4u), 64u);
    uint np = min(res.Load(rec * 4u + 4u), 42u);
    SetMeshOutputCounts(nv, np);
    for (uint i = gi; i < nv; i += 32u) {
        uint b = (rec + 2u + i * 3u) * 4u;
        XclVert o;
        o.pos = asfloat(res.Load4(b));
        o.p0 = asfloat(res.Load4(b + 16u));
        o.p1 = res.Load4(b + 32u).xy;
        o.p2 = asfloat(res.Load4(b + 48u).x);
        verts[i] = o;
    }
    for (uint j = gi; j < np; j += 32u) {
        uint p = res.Load((rec + 300u + j) * 4u);
        uint3 x = uint3(p & 0x3FFu, (p >> 10) & 0x3FFu, (p >> 20) & 0x3FFu);
        bool bad = or((p >> 31) != 0u, any(x >= nv));
        tris[j] = select(bad, uint3(0u, 0u, 0u), x);
    }
}
"#,
        // DXC: flags 0x10, view id state [0, 13, 0]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Mesh, "main", StageProps::Mesh { numthreads: [32, 1, 1], max_vertices: 64, max_primitives: 42, topology: MeshOutputTopology::Triangle, payload_bytes: 0 });
            i.resources = vec![
                res(ResourceClass::UAV, 0, "r0", ResourceKind::RawBuffer, 1, 0, |_r| {}),
            ];
            i.inputs = vec![];
            i.outputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 1, 0, 0, 15, 0),
            el("COLOR", &[1], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Constant, 1, 2, 2, 0, 0, 3, 0),
            el("SV_ClipDistance", &[0], SemanticKind::ClipDistance, ComponentType::F32, InterpolationMode::Linear, 1, 1, 3, 0, 0, 1, 0),
        ];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "ms_prim",
        stage: "ms",
        source: r#"Texture2D<float4> tex : register(t0);
SamplerState samp : register(s0);
struct Payload { float4 a; uint b; };
struct Vert { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct Prim { uint id : SV_PrimitiveID; bool cull : SV_CullPrimitive; float3 n : NORMAL; uint rt : SV_RenderTargetArrayIndex; };
[outputtopology("line")]
[numthreads(2, 2, 2)]
void main(uint3 gtid : SV_GroupThreadID, in payload Payload pl, out vertices Vert verts[8], out indices uint2 lines[4], out primitives Prim prims[4])
{
    SetMeshOutputCounts(8, 4);
    Vert v; v.pos = pl.a + tex.Sample(samp, pl.a.xy); v.uv = pl.a.xy;
    verts[gtid.x] = v;
    lines[gtid.y] = uint2(0, 1);
    Prim p; p.id = pl.b; p.cull = false; p.n = pl.a.xyz; p.rt = 0;
    prims[gtid.y] = p;
}
"#,
        // DXC: flags 0x20000000, view id state [0, 6, 7]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Mesh, "main", StageProps::Mesh { numthreads: [2, 2, 2], max_vertices: 8, max_primitives: 4, topology: MeshOutputTopology::Line, payload_bytes: 20 });
            i.resources = vec![
                res(ResourceClass::SRV, 0, "r0", ResourceKind::Texture2D, 0, 0, |r| {r.comp_type = ComponentType::F32; r.comp_count = 4;}),
                res(ResourceClass::Sampler, 0, "r1", ResourceKind::Sampler, 0, 0, |_r| {}),
            ];
            i.inputs = vec![];
            i.outputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 1, 0, 0, 3, 0),
        ];
            i.patch_or_prim = vec![
            el("SV_PrimitiveID", &[0], SemanticKind::PrimitiveID, ComponentType::U32, InterpolationMode::Constant, 1, 1, 0, 0, 0, 1, 0),
            el("SV_CullPrimitive", &[0], SemanticKind::CullPrimitive, ComponentType::I1, InterpolationMode::Constant, 1, 1, -1, -1, 0, 1, 0),
            el("NORMAL", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Constant, 1, 3, 1, 0, 0, 7, 0),
            el("SV_RenderTargetArrayIndex", &[0], SemanticKind::RenderTargetArrayIndex, ComponentType::U32, InterpolationMode::Constant, 1, 1, 0, 1, 0, 1, 0),
        ];
            i
        },
    },
    Case {
        name: "ps_basic",
        stage: "ps",
        source: r#"RWByteAddressBuffer io : register(u0);
Texture2D<float4> tex : register(t0);
SamplerState samp : register(s0);
struct XclIn {
    float4 xcl_pos : SV_Position;
    float4 p0 : TEXCOORD0;
    nointerpolation uint p1 : COLOR3;
    centroid float2 p2 : TEXCOORD1;
    noperspective float3 p3 : TEXCOORD2;
    sample float p4 : TEXCOORD3;
    noperspective centroid float p5 : TEXCOORD4;
    noperspective sample float p6 : TEXCOORD5;
    bool xcl_ff : SV_IsFrontFace;
    uint xcl_cov : SV_Coverage;
};
struct XclOut {
    float4 t0 : SV_Target0;
    uint4 t1 : SV_Target1;
    int4 t3 : SV_Target3;
    float xcl_depth : SV_Depth;
};
XclOut main(XclIn i)
{
    XclOut o;
    float4 c = tex.Sample(samp, i.p0.xy) + ddx(i.p0);
    if (c.w < 0.5) discard;
    o.t0 = c * i.p3.xyzz + i.p4 + i.p5 + i.p6 + i.xcl_pos;
    o.t1 = uint4(i.p1, i.xcl_cov, i.xcl_ff ? 1u : 0u, io.Load(0));
    o.t3 = int4(i.p2.xyxy);
    o.xcl_depth = c.x;
    return o;
}
"#,
        // DXC: flags 0x10, view id state [29, 16, 1, 2, 4, 8, 15, 15, 4, 8, 16, 64, 0, 0, 20480, 40960, 0, 0, 1, 2, 12, 0, 15, 0, 0, 0, 15, 0, 0, 0, 15]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Pixel, "main", StageProps::Pixel { early_depth_stencil: false });
            i.resources = vec![
                res(ResourceClass::SRV, 0, "r0", ResourceKind::Texture2D, 0, 0, |r| {r.comp_type = ComponentType::F32; r.comp_count = 4;}),
                res(ResourceClass::UAV, 0, "r1", ResourceKind::RawBuffer, 0, 0, |_r| {}),
                res(ResourceClass::Sampler, 0, "r2", ResourceKind::Sampler, 0, 0, |_r| {}),
            ];
            i.inputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 1, 0, 0, 15, 0),
            el("COLOR", &[3], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 0, 0, 1, 0),
            el("TEXCOORD", &[1], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::LinearCentroid, 1, 2, 3, 0, 0, 3, 0),
            el("TEXCOORD", &[2], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 3, 4, 0, 0, 7, 0),
            el("TEXCOORD", &[3], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::LinearSample, 1, 1, 5, 0, 0, 1, 0),
            el("TEXCOORD", &[4], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::LinearNoperspectiveCentroid, 1, 1, 6, 0, 0, 1, 0),
            el("TEXCOORD", &[5], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::LinearNoperspectiveSample, 1, 1, 7, 0, 0, 1, 0),
            el("SV_IsFrontFace", &[0], SemanticKind::IsFrontFace, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 1, 0, 1, 0),
        ];
            i.outputs = vec![
            el("SV_Target", &[0], SemanticKind::Target, ComponentType::F32, InterpolationMode::Undefined, 1, 4, 0, 0, 0, 15, 0),
            el("SV_Target", &[1], SemanticKind::Target, ComponentType::U32, InterpolationMode::Undefined, 1, 4, 1, 0, 0, 15, 0),
            el("SV_Target", &[3], SemanticKind::Target, ComponentType::I32, InterpolationMode::Undefined, 1, 4, 3, 0, 0, 15, 0),
            el("SV_Depth", &[0], SemanticKind::Depth, ComponentType::F32, InterpolationMode::Undefined, 1, 1, -1, -1, 0, 1, 0),
        ];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "ps_depth_ge",
        stage: "ps",
        source: r#"struct XclIn {
    noperspective centroid float4 xcl_pos : SV_Position;
    float4 p0 : TEXCOORD0;
};
struct XclOut {
    float4 t0 : SV_Target0;
    float xcl_depth : SV_DepthGreaterEqual;
};
[earlydepthstencil]
XclOut main(XclIn i)
{
    XclOut o;
    o.t0 = i.p0;
    o.xcl_depth = i.xcl_pos.z;
    return o;
}
"#,
        // DXC: flags 0x8, view id state [8, 4, 0, 0, 0, 0, 1, 2, 4, 8]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Pixel, "main", StageProps::Pixel { early_depth_stencil: true });
            i.resources = vec![];
            i.inputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspectiveCentroid, 1, 4, 0, 0, 0, 4, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 1, 0, 0, 15, 0),
        ];
            i.outputs = vec![
            el("SV_Target", &[0], SemanticKind::Target, ComponentType::F32, InterpolationMode::Undefined, 1, 4, 0, 0, 0, 15, 0),
            el("SV_DepthGreaterEqual", &[0], SemanticKind::DepthGreaterEqual, ComponentType::F32, InterpolationMode::Undefined, 1, 1, -1, -1, 0, 1, 0),
        ];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "ps_flow",
        stage: "ps",
        source: r#"Texture2D<float4> tex : register(t0);
SamplerState samp : register(s0);
struct Out { float4 t0 : SV_Target0; float4 t1 : SV_Target1; float depth : SV_Depth; };
Out main(float4 pos : SV_Position, float4 p0 : TEXCOORD0, nointerpolation uint3 p1 : TEXCOORD1, float2 uv[2] : TEXCOORD2, uint view : SV_ViewID, bool front : SV_IsFrontFace)
{
    Out o;
    float4 c = float4(0.0, 0.0, 0.0, 1.0);
    if (front) {
        c.x = 1.0;
        if (p0.x > 0.5) { c.y = 0.5; }
    } else {
        for (uint k = 0; k < (p1.x & 3u); ++k) { c.z += 0.25; }
    }
    o.t0 = c;
    float w[3] = { 1.0, 2.0, 3.0 };
    if (p1.y != 0u) { w[p1.z % 3u] = p0.w; }
    float4 s = tex.Sample(samp, uv[view & 1u]);
    o.t1 = float4(w[0], w[2], s.x, view != 0 ? 1.0 : 0.0);
    if (p0.z < 0.0) discard;
    o.depth = p0.y > 0.0 ? 0.25 : 0.75;
    return o;
}
"#,
        // DXC: flags 0x200000, view id state [18, 8, 192, 0, 0, 0, 0, 2, 0, 0, 48, 4, 48, 48, 7, 64, 64, 0, 0, 64, 64]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Pixel, "main", StageProps::Pixel { early_depth_stencil: false });
            i.resources = vec![
                res(ResourceClass::SRV, 0, "r0", ResourceKind::Texture2D, 0, 0, |r| {r.comp_type = ComponentType::F32; r.comp_count = 4;}),
                res(ResourceClass::Sampler, 0, "r1", ResourceKind::Sampler, 0, 0, |_r| {}),
            ];
            i.inputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 0, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 1, 0, 0, 15, 0),
            el("TEXCOORD", &[1], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Constant, 1, 3, 2, 0, 0, 7, 0),
            el("TEXCOORD", &[2, 3], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 2, 2, 3, 0, 0, 3, 3),
            el("SV_IsFrontFace", &[0], SemanticKind::IsFrontFace, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 3, 0, 1, 0),
        ];
            i.outputs = vec![
            el("SV_Target", &[0], SemanticKind::Target, ComponentType::F32, InterpolationMode::Undefined, 1, 4, 0, 0, 0, 15, 0),
            el("SV_Target", &[1], SemanticKind::Target, ComponentType::F32, InterpolationMode::Undefined, 1, 4, 1, 0, 0, 15, 0),
            el("SV_Depth", &[0], SemanticKind::Depth, ComponentType::F32, InterpolationMode::Undefined, 1, 1, -1, -1, 0, 1, 0),
        ];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "ps_misc",
        stage: "ps",
        source: r#"struct XclOut {
    float4 t0 : SV_Target0;
    float4 t7 : SV_Target7;
    float xcl_depth : SV_DepthLessEqual;
    uint cov : SV_Coverage;
    uint sref : SV_StencilRef;
};
XclOut main(noperspective sample float4 pos : SV_Position, float4 p0 : TEXCOORD0, uint si : SV_SampleIndex, uint pid : SV_PrimitiveID,
            uint ic : SV_InnerCoverage, uint vid : SV_ViewID, uint rt : SV_RenderTargetArrayIndex, uint vp : SV_ViewportArrayIndex,
            float clip : SV_ClipDistance0, float cull : SV_CullDistance0)
{
    XclOut o;
    o.t0 = EvaluateAttributeAtSample(p0, si) + EvaluateAttributeCentroid(p0) + EvaluateAttributeSnapped(p0, int2(1, 2));
    o.t7 = float4(pid, ic, vid, rt + vp) + clip + cull;
    o.xcl_depth = pos.z;
    o.cov = si;
    o.sref = pid;
    return o;
}
"#,
        // DXC: flags 0x200c00, view id state [18, 32, 1073741824, 0, 0, 0, 0, 0, 0, 0, 0, 268435456, 0, 0, 0, 2147483648, 2147483648, 0, 0, 4026531840, 4026531840]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Pixel, "main", StageProps::Pixel { early_depth_stencil: false });
            i.resources = vec![];
            i.inputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspectiveSample, 1, 4, 0, 0, 0, 4, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 1, 0, 0, 0, 0),
            el("SV_SampleIndex", &[0], SemanticKind::SampleIndex, ComponentType::U32, InterpolationMode::Constant, 1, 1, -1, -1, 0, 0, 0),
            el("SV_PrimitiveID", &[0], SemanticKind::PrimitiveID, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 0, 0, 1, 0),
            el("SV_RenderTargetArrayIndex", &[0], SemanticKind::RenderTargetArrayIndex, ComponentType::U32, InterpolationMode::Constant, 1, 1, 3, 0, 0, 1, 0),
            el("SV_ViewportArrayIndex", &[0], SemanticKind::ViewPortArrayIndex, ComponentType::U32, InterpolationMode::Constant, 1, 1, 3, 1, 0, 1, 0),
            el("SV_ClipDistance", &[0], SemanticKind::ClipDistance, ComponentType::F32, InterpolationMode::Linear, 1, 1, 4, 0, 0, 1, 0),
            el("SV_CullDistance", &[0], SemanticKind::CullDistance, ComponentType::F32, InterpolationMode::Linear, 1, 1, 4, 1, 0, 1, 0),
        ];
            i.outputs = vec![
            el("SV_Target", &[0], SemanticKind::Target, ComponentType::F32, InterpolationMode::Undefined, 1, 4, 0, 0, 0, 15, 0),
            el("SV_Target", &[7], SemanticKind::Target, ComponentType::F32, InterpolationMode::Undefined, 1, 4, 7, 0, 0, 15, 0),
            el("SV_DepthLessEqual", &[0], SemanticKind::DepthLessEqual, ComponentType::F32, InterpolationMode::Undefined, 1, 1, -1, -1, 0, 1, 0),
            el("SV_Coverage", &[0], SemanticKind::Coverage, ComponentType::U32, InterpolationMode::Undefined, 1, 1, -1, -1, 0, 1, 0),
            el("SV_StencilRef", &[0], SemanticKind::StencilRef, ComponentType::U32, InterpolationMode::Undefined, 1, 1, -1, -1, 0, 1, 0),
        ];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "vs_basic",
        stage: "vs",
        source: r#"RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
struct XclIn {
    float4 a0 : POSITION;
    float3 a1 : NORMAL;
    uint4 a2 : BLENDINDICES;
    int2 a3 : TEXCOORD3;
    float a4 : TEXCOORD1;
    uint xcl_vid : SV_VertexID;
    uint xcl_iid : SV_InstanceID;
};
struct XclOut {
    float4 pos : SV_Position;
    float4 p0 : TEXCOORD0;
    float2 p1 : TEXCOORD1;
    uint p2 : COLOR3;
    int3 p3 : FOO;
    float2 clip : SV_ClipDistance0;
    float cull : SV_CullDistance0;
};
XclOut main(XclIn i)
{
    XclOut o;
    uint k = io.Load(4u * i.xcl_vid) + i.xcl_iid;
    o.pos = i.a0 + float4(i.a1, i.a4);
    o.p0 = asfloat(i.a2 + k);
    o.p1 = float2(i.a3);
    o.p2 = k;
    o.p3 = int3(i.a3, 1);
    o.clip = i.a0.xy;
    o.cull = i.a0.z;
    return o;
}
"#,
        // DXC: flags 0x10010, view id state [25, 19, 65537, 131074, 262148, 8, 1, 2, 4, 0, 16, 32, 64, 128, 8448, 16896, 0, 0, 8, 0, 0, 0, 4336, 0, 0, 0, 4336]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Vertex, "main", StageProps::Vertex);
            i.resources = vec![
                res(ResourceClass::UAV, 0, "r0", ResourceKind::RawBuffer, 0, 0, |_r| {}),
            ];
            i.inputs = vec![
            el("POSITION", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Undefined, 1, 4, 0, 0, 0, 15, 0),
            el("NORMAL", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Undefined, 1, 3, 1, 0, 0, 7, 0),
            el("BLENDINDICES", &[0], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Undefined, 1, 4, 2, 0, 0, 15, 0),
            el("TEXCOORD", &[3], SemanticKind::Arbitrary, ComponentType::I32, InterpolationMode::Undefined, 1, 2, 3, 0, 0, 3, 0),
            el("TEXCOORD", &[1], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Undefined, 1, 1, 4, 0, 0, 1, 0),
            el("SV_VertexID", &[0], SemanticKind::VertexID, ComponentType::U32, InterpolationMode::Undefined, 1, 1, 5, 0, 0, 1, 0),
            el("SV_InstanceID", &[0], SemanticKind::InstanceID, ComponentType::U32, InterpolationMode::Undefined, 1, 1, 6, 0, 0, 1, 0),
        ];
            i.outputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 1, 0, 0, 15, 0),
            el("TEXCOORD", &[1], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 2, 2, 0, 0, 3, 0),
            el("COLOR", &[3], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Constant, 1, 1, 3, 0, 0, 1, 0),
            el("FOO", &[0], SemanticKind::Arbitrary, ComponentType::I32, InterpolationMode::Constant, 1, 3, 3, 1, 0, 7, 0),
            el("SV_ClipDistance", &[0], SemanticKind::ClipDistance, ComponentType::F32, InterpolationMode::Linear, 1, 2, 4, 0, 0, 3, 0),
            el("SV_CullDistance", &[0], SemanticKind::CullDistance, ComponentType::F32, InterpolationMode::Linear, 1, 1, 4, 2, 0, 1, 0),
        ];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "vs_flow",
        stage: "vs",
        source: r#"RWByteAddressBuffer io : register(u0);
struct In {
    float4 a : POSITION;
    float3 b : NORMAL;
    uint4 c : BLENDINDICES;
    float2 tc[3] : TEXCOORD0;
    uint vid : SV_VertexID;
    uint view : SV_ViewID;
};
struct Out {
    float4 pos : SV_Position;
    float4 p0 : TEXCOORD0;
    float2 p1[2] : TEXCOORD1;
    uint p2 : COLOR0;
    float p3 : COLOR1;
    float4 p4 : COLOR2;
};
Out main(In i)
{
    Out o;
    // a constant chosen by an input: only control dependence
    float sel = 0.0;
    if (i.a.x > 0.5) { sel = 1.0; } else if (i.b.y > 0.25) { sel = 2.0; }
    o.pos = float4(sel, i.a.yz, 1.0);
    // memory: a local array written under control and read with a dynamic index
    float tab[4] = { 0.0, 0.0, 0.0, 0.0 };
    for (uint k = 0; k < 4; ++k) {
        if ((i.c.x >> k) & 1u) { tab[k] = i.b.x; }
    }
    o.p0 = float4(tab[i.c.y & 3u], tab[1], i.tc[i.c.z % 3u].x, 0.0);
    // a loop whose trip count comes from an input, and the view
    uint acc = 0;
    for (uint n = 0; n < (i.c.w & 7u); ++n) { acc += io.Load(4u * n); if (acc > 100u) break; }
    o.p1[0] = float2(acc, 0.0);
    o.p1[1] = i.view == 0 ? float2(1.0, 2.0) : float2(3.0, 4.0);
    switch (i.vid & 3u) {
    case 0: o.p2 = 7u; break;
    case 1: o.p2 = i.c.x; break;
    case 2: o.p2 = 9u; break;
    default: o.p2 = i.view; break;
    }
    o.p3 = 5.0;
    bool both = i.a.w > 0.0 && i.b.z > 0.0;
    o.p4 = both ? float4(1.0, 1.0, 1.0, 1.0) : float4(0.0, i.tc[1].y, 0.0, 0.0);
    return o;
}
"#,
        // DXC: flags 0x210010, view id state [25, 24, 77824, 1, 2, 4, 15728640, 48, 1, 15728640, 0, 65584, 16, 64, 256, 64, 0, 0, 0, 64, 2097152, 0, 0, 64, 0, 0, 0, 65536]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Vertex, "main", StageProps::Vertex);
            i.resources = vec![
                res(ResourceClass::UAV, 0, "r0", ResourceKind::RawBuffer, 0, 0, |_r| {}),
            ];
            i.inputs = vec![
            el("POSITION", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Undefined, 1, 4, 0, 0, 0, 15, 0),
            el("NORMAL", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Undefined, 1, 3, 1, 0, 0, 7, 0),
            el("BLENDINDICES", &[0], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Undefined, 1, 4, 2, 0, 0, 15, 0),
            el("TEXCOORD", &[0, 1, 2], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Undefined, 3, 2, 3, 0, 0, 3, 1),
            el("SV_VertexID", &[0], SemanticKind::VertexID, ComponentType::U32, InterpolationMode::Undefined, 1, 1, 6, 0, 0, 1, 0),
        ];
            i.outputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 1, 0, 0, 15, 0),
            el("TEXCOORD", &[1, 2], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 2, 2, 2, 0, 0, 3, 0),
            el("COLOR", &[0], SemanticKind::Arbitrary, ComponentType::U32, InterpolationMode::Constant, 1, 1, 4, 0, 0, 1, 0),
            el("COLOR", &[1], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 1, 2, 2, 0, 1, 0),
            el("COLOR", &[2], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 5, 0, 0, 15, 0),
        ];
            i.patch_or_prim = vec![];
            i
        },
    },
    Case {
        name: "vs_rtidx",
        stage: "vs",
        source: r#"struct XclOut {
    float4 pos : SV_Position;
    float4 p0 : TEXCOORD0;
    uint rt : SV_RenderTargetArrayIndex;
    uint vp : SV_ViewportArrayIndex;
};
XclOut main(float4 a : POSITION, uint v : SV_ViewID, uint vid : SV_VertexID)
{
    XclOut o;
    o.pos = a;
    o.p0 = a * (float)v;
    o.rt = vid;
    o.vp = v;
    return o;
}
"#,
        // DXC: flags 0x200200, view id state [5, 10, 752, 17, 34, 68, 136, 256]
        info: || {
            let mut i = ShaderInfo::new(ShaderKind::Vertex, "main", StageProps::Vertex);
            i.resources = vec![];
            i.inputs = vec![
            el("POSITION", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Undefined, 1, 4, 0, 0, 0, 15, 0),
            el("SV_VertexID", &[0], SemanticKind::VertexID, ComponentType::U32, InterpolationMode::Undefined, 1, 1, 1, 0, 0, 1, 0),
        ];
            i.outputs = vec![
            el("SV_Position", &[0], SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 1, 4, 0, 0, 0, 15, 0),
            el("TEXCOORD", &[0], SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Linear, 1, 4, 1, 0, 0, 15, 0),
            el("SV_RenderTargetArrayIndex", &[0], SemanticKind::RenderTargetArrayIndex, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 0, 0, 1, 0),
            el("SV_ViewportArrayIndex", &[0], SemanticKind::ViewPortArrayIndex, ComponentType::U32, InterpolationMode::Constant, 1, 1, 2, 1, 0, 1, 0),
        ];
            i.patch_or_prim = vec![];
            i
        },
    },
];

fn oracle() -> Option<PathBuf> {
    match std::env::var_os("XCL_DXIL_DXC") {
        Some(dir) => Some(PathBuf::from(dir)),
        None => {
            eprintln!("XCL_DXIL_DXC is not set: skipping the comparison with DXC");
            None
        }
    }
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xcl-dxil-meta-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn tool(dxc: &std::path::Path, name: &str) -> Command {
    let mut c = Command::new(dxc.join("bin").join(name));
    c.env("LD_LIBRARY_PATH", dxc.join("lib"));
    c
}

/// Compile a case as XCL does, for the validator version this crate writes; returns the
/// disassembly and the container.
fn compile(dxc: &std::path::Path, dir: &std::path::Path, case: &Case) -> (String, PathBuf) {
    let src = dir.join(format!("{}.hlsl", case.name));
    let (ll, dxil) = (dir.join(format!("{}.ll", case.name)), dir.join(format!("{}.dxil", case.name)));
    std::fs::write(&src, case.source).unwrap();
    let valver = format!("{}.{}", VALIDATOR_VERSION.0, VALIDATOR_VERSION.1);
    let out = tool(dxc, "dxc")
        .args(["-E", "main", "-T", &format!("{}_6_6", case.stage), "-HV", "2021", "-Gis", "-O1", "-validator-version", &valver, "-Fo"])
        .arg(&dxil)
        .arg("-Fc")
        .arg(&ll)
        .arg(&src)
        .output()
        .unwrap();
    assert!(out.status.success(), "dxc {}: {}", case.name, String::from_utf8_lossy(&out.stderr));
    (std::fs::read_to_string(&ll).unwrap(), dxil)
}

/// The metadata of an LLVM assembly text: numbered tuples as operand lists, and the named
/// metadata.
#[derive(Default)]
struct Metadata {
    nodes: HashMap<u32, Vec<String>>,
    named: Vec<(String, Vec<String>)>,
}

/// Split at the commas outside brackets and strings.
fn operands(s: &str) -> Vec<String> {
    let (mut out, mut cur, mut depth, mut quoted) = (Vec::new(), String::new(), 0, false);
    for c in s.chars() {
        match c {
            '"' => quoted = !quoted,
            '{' | '[' | '<' | '(' if !quoted => depth += 1,
            '}' | ']' | '>' | ')' if !quoted => depth -= 1,
            ',' if !quoted && depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

impl Metadata {
    fn parse(text: &str) -> Metadata {
        let mut md = Metadata::default();
        for line in text.lines() {
            let Some((name, rest)) = line.strip_prefix('!').and_then(|l| l.split_once(" = ")) else { continue };
            let Some(body) = rest.trim().strip_prefix("!{").and_then(|b| b.strip_suffix('}')) else { continue };
            match name.parse::<u32>() {
                Ok(n) => {
                    md.nodes.insert(n, operands(body));
                }
                Err(_) => md.named.push((name.to_string(), operands(body))),
            }
        }
        md
    }

    fn node(&self, operand: &str) -> Option<&Vec<String>> {
        self.nodes.get(&operand.strip_prefix('!')?.parse().ok()?)
    }

    /// An operand with every node reference replaced by the node's text, so that two modules
    /// compare whatever their numbering. The element type of a structured buffer is not in
    /// `ShaderInfo`, so its name is left out of the comparison, and so is the
    /// width of a 64-bit texture element.
    fn expand(&self, operand: &str) -> String {
        match self.node(operand) {
            Some(ops) => format!("!{{{}}}", ops.iter().map(|o| self.expand(o)).collect::<Vec<_>>().join(", ")),
            None => match (operand.find("StructuredBuffer<"), operand.rfind('>')) {
                (Some(a), Some(b)) => format!("{}StructuredBuffer<T{}", &operand[..a], &operand[b..]),
                // a 64-bit texture element has a 32-bit component type in the record
                _ => operand.replace("unsigned long long", "unsigned int").replace("long long", "int"),
            },
        }
    }

    fn named(&self, name: &str) -> Option<String> {
        let (_, ops) = self.named.iter().find(|(n, _)| n == name)?;
        Some(ops.iter().map(|o| self.expand(o)).collect::<Vec<_>>().join(", "))
    }

    /// The tag-value list of the entry point as (tag, operand) pairs.
    fn entry_properties(&self) -> Vec<(u32, String)> {
        let entry = self.named.iter().find(|(n, _)| n == "dx.entryPoints").and_then(|(_, ops)| self.node(&ops[0])).expect("dx.entryPoints");
        let Some(props) = self.node(&entry[4]) else { return Vec::new() };
        props.chunks(2).map(|p| (int(&p[0]) as u32, p[1].clone())).collect()
    }

    /// The shader flags in dx.entryPoints.
    fn flags(&self) -> u64 {
        self.entry_properties().iter().find(|(t, _)| *t == 0).map_or(0, |(_, v)| int(v))
    }

    /// The words of dx.viewIdState.
    fn view_id_state(&self) -> Vec<u32> {
        let Some((_, ops)) = self.named.iter().find(|(n, _)| n == "dx.viewIdState") else { return Vec::new() };
        let array = &self.node(&ops[0]).expect("dx.viewIdState")[0];
        let list = &array[array.find("] [").expect("array") + 3..array.len() - 1];
        list.split(',').map(|w| int(w.trim()) as u32).collect()
    }

    /// Per UAV id: the record carries the "used by a 64-bit atomic" tag.
    fn atomic64_uavs(&self) -> Vec<bool> {
        let entry = self.named.iter().find(|(n, _)| n == "dx.entryPoints").and_then(|(_, ops)| self.node(&ops[0])).expect("dx.entryPoints");
        let Some(uavs) = self.node(&entry[3]).and_then(|r| self.node(&r[1])) else { return Vec::new() };
        uavs.iter().map(|u| self.node(u).and_then(|r| self.node(&r[10])).is_some_and(|x| x.chunks(2).any(|p| int(&p[0]) == 3 && int(&p[1]) == 1))).collect()
    }
}

/// The integer of a typed operand (`i32 7`, `i64 -1`).
fn int(operand: &str) -> u64 {
    operand.rsplit(' ').next().and_then(|v| v.parse::<i64>().ok()).unwrap_or_else(|| panic!("not an integer: {operand}")) as u64
}

const NAMED: [&str; 6] = ["dx.version", "dx.valver", "dx.shaderModel", "dx.resources", "dx.viewIdState", "dx.entryPoints"];

/// The case's `ShaderInfo` with what [`meta::prepare`] computes from the code taken from DXC's
/// metadata instead.
fn info_with(case: &Case, dxc: &Metadata) -> ShaderInfo {
    let mut info = (case.info)();
    // the early depth-stencil bit is metadata of the entry point, not a collected flag
    info.flags = dxc.flags() & !flags::bit::FORCE_EARLY_DEPTH_STENCIL;
    info.view_id_state = dxc.view_id_state();
    let atomic64 = dxc.atomic64_uavs();
    for r in info.resources.iter_mut().filter(|r| r.class == ResourceClass::UAV) {
        r.atomic64_use = atomic64[r.id as usize];
    }
    info
}

/// A module that defines the functions the metadata names, empty.
fn skeleton(info: &ShaderInfo) -> Module {
    let mut m = Module::new();
    let mut define = |name: &str| {
        let mut body = Body::new();
        body.ret(Body::ENTRY);
        m.define(name, body);
    };
    define(&info.entry);
    if let StageProps::Hull { patch_constant_fn, .. } = &info.props {
        define(patch_constant_fn);
    }
    m
}

fn assert_same_metadata(case: &Case, ours: &Metadata, dxc: &Metadata) {
    for name in NAMED {
        assert_eq!(ours.named(name), dxc.named(name), "{}: !{name}", case.name);
    }
    let ident = ours.named("llvm.ident").expect("llvm.ident");
    assert!(ident.starts_with("!{!\"xcl-dxil "), "{ident}");
}

/// The metadata emitted for each hand-built `ShaderInfo` is DXC's, node for node.
#[test]
fn metadata_matches_dxc() {
    let Some(dxc) = oracle() else { return };
    let dir = scratch("metadata");
    for case in CASES {
        let (ll, _) = compile(&dxc, &dir, case);
        let theirs = Metadata::parse(&ll);
        let info = info_with(case, &theirs);
        let mut m = skeleton(&info);
        meta::emit(&mut m, &info).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let ours = Metadata::parse(&irprint::module(&m));
        assert_same_metadata(case, &ours, &theirs);
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// Runs the validator on a container; its complaint when it does not accept it.
fn validate(dxc: &std::path::Path, path: &std::path::Path, container: &[u8]) -> Result<(), String> {
    std::fs::write(path, container).unwrap();
    let out = tool(dxc, "dxv").arg(path).output().unwrap();
    let said = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if out.status.success() && said.contains("Validation succeeded.") {
        Ok(())
    } else {
        Err(said.trim().to_string())
    }
}

/// What [`meta::prepare`] computes from DXC's own code, read back from its bitcode, is what
/// DXC wrote in its metadata; and DXC's module with this crate's metadata in place of its own
/// passes the validator, both in DXC's container (whose PSV0 and SFI0 parts the validator
/// compares with the metadata) and in the container this crate builds from the same
/// `ShaderInfo`.
#[test]
fn dxc_code_with_our_metadata_validates() {
    let Some(dxc) = oracle() else { return };
    let dir = scratch("validate");
    let mut failures = Vec::new();
    for case in CASES {
        let (ll, dxil) = compile(&dxc, &dir, case);
        let theirs = Metadata::parse(&ll);
        let container = std::fs::read(&dxil).unwrap();
        let (mut m, _) = match bcreader::read(bcreader::container_bitcode(&container).unwrap()) {
            Ok(read) => read,
            // what the reader does not cover says nothing about the metadata
            Err(xcl_dxil::Error::Unsupported(what)) => {
                eprintln!("{}: not checked on DXC's code: {what}", case.name);
                continue;
            }
            Err(e) => {
                failures.push(format!("{}: reading DXC's module: {e}", case.name));
                continue;
            }
        };
        m.named_md.clear();
        let mut info = (case.info)();
        meta::prepare(&m, &mut info);
        let expected = info_with(case, &theirs);
        if info.flags != expected.flags {
            failures.push(format!("{}: flags {:#x}, DXC has {:#x}", case.name, info.flags, expected.flags));
        }
        // DXC writes no dx.viewIdState when every word is zero
        let same_state = if expected.view_id_state.is_empty() { info.view_id_state.iter().all(|&w| w == 0) } else { info.view_id_state == expected.view_id_state };
        if !same_state {
            failures.push(format!("{}: view id state {:?}, DXC has {:?}", case.name, info.view_id_state, expected.view_id_state));
        }
        let atomic = |i: &ShaderInfo| i.resources.iter().map(|r| r.atomic64_use).collect::<Vec<_>>();
        if atomic(&info) != atomic(&expected) {
            failures.push(format!("{}: 64-bit atomic use {:?}, DXC has {:?}", case.name, atomic(&info), atomic(&expected)));
        }
        let written = meta::emit(&mut m, &info).and_then(|()| bcwriter::write(&m));
        let bitcode = match written {
            Ok((bitcode, _)) => bitcode,
            Err(e) => {
                failures.push(format!("{}: {e}", case.name));
                continue;
            }
        };
        let in_theirs = bcreader::replace_bitcode(&container, &bitcode).unwrap();
        if let Err(said) = validate(&dxc, &dir.join(format!("{}.theirs.dxil", case.name)), &in_theirs) {
            failures.push(format!("{}: dxv, in DXC's container: {said}", case.name));
        }
        match xcl_dxil::container::build(&info, &bitcode) {
            Ok(ours) => {
                if let Err(said) = validate(&dxc, &dir.join(format!("{}.ours.dxil", case.name)), &ours) {
                    failures.push(format!("{}: dxv, in our container: {said}", case.name));
                }
            }
            Err(e) => failures.push(format!("{}: container: {e}", case.name)),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    let _ = std::fs::remove_dir_all(dir);
}
