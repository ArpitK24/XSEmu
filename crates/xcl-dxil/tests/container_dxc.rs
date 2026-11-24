//! The container against its oracle: DXC's own containers for a set of shaders covering every
//! stage and signature shape the translator emits (and the rest of what PSV0 and the signature
//! parts encode). For each shader the test reads DXC's dx.* metadata from the disassembly
//! (what the validator rebuilds the parts from, and what the front end has to put into
//! `ShaderInfo`), serialises the parts from it and compares them with DXC's byte for byte;
//! then it lets dxv validate and sign what `container::build` assembles around DXC's bitcode.
//!
//! Needs DXC: set XCL_DXIL_DXC to a folder with bin/dxc, bin/dxv and lib/. Without it every
//! test here passes with a note.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use xcl_dxil::consts::*;
use xcl_dxil::container::{self, psv, sig, SigKind};
use xcl_dxil::shader::{parse_profile, Resource, ShaderInfo, SigElement, StageProps};

/// (name, profile and any further DXC arguments, source)
const SHADERS: &[(&str, &str, &str)] = &[
    (
        "as_basic",
        "as_6_6",
        r#"struct Payload { float4 base; uint count; };
RWByteAddressBuffer io : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID)
{
    Payload pl;
    pl.base = asfloat(io.Load4(gid.x * 16u));
    pl.count = gid.y;
    DispatchMesh(2u, 1u, 1u, pl);
}
"#,
    ),
    (
        "cs_basic",
        "cs_6_6",
        r#"RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
[numthreads(64, 1, 1)]
void main(uint gi : SV_GroupIndex)
{
    uint a = io.Load(gi * 4u);
    res.Store(gi * 4u, a * 3u + 1u);
}
"#,
    ),
    (
        "cs_heap",
        "cs_6_6",
        r#"RWByteAddressBuffer io : register(u0);
SamplerState xcl_ss0 : register(s0, space1);
SamplerComparisonState xcl_ss1 : register(s1, space1);
[numthreads(32, 1, 1)]
void main(uint gi : SV_GroupIndex)
{
    uint w = io.Load(gi * 4u);
    Texture2D<float4> t = ResourceDescriptorHeap[NonUniformResourceIndex(w)];
    RWTexture2D<uint> u = ResourceDescriptorHeap[NonUniformResourceIndex(w + 1u)];
    SamplerState s = SamplerDescriptorHeap[NonUniformResourceIndex(w & 0xFFFu)];
    RWByteAddressBuffer mb = ResourceDescriptorHeap[w + 2u];
    float4 c = t.SampleLevel(s, float2(0.5, 0.5), 0.0) + t.SampleLevel(xcl_ss0, float2(0.25, 0.5), 1.0);
    float d = t.SampleCmpLevelZero(xcl_ss1, float2(0.5, 0.5), 0.5);
    u[uint2(gi, 0u)] = asuint(c.x + d);
    mb.Store(0u, asuint(c.y));
}
"#,
    ),
    (
        "cs_res",
        "cs_6_6",
        r#"struct Rec { float4 a; uint b; };
cbuffer Params : register(b2, space3) { float4 scale; uint count; double dbl; };
ByteAddressBuffer raw : register(t0);
Texture2D<float4> tex : register(t1);
Texture3D<uint4> vol : register(t2);
Texture2DArray<int4> arr : register(t3);
TextureCube<float4> cube : register(t4);
StructuredBuffer<Rec> recs : register(t5);
Texture2D<float4> many[4] : register(t6);
Texture2D<float4> unbounded[] : register(t0, space2);
Buffer<float2> typed : register(t10);
Texture2DMS<float4> ms : register(t11);
RWByteAddressBuffer io : register(u0);
RWStructuredBuffer<Rec> out_recs : register(u1);
RWTexture2D<float4> out_tex : register(u2);
RWTexture3D<uint> out_vol : register(u3);
RWBuffer<int> out_buf : register(u4, space5);
SamplerState lin : register(s0, space1);
SamplerComparisonState cmp : register(s3, space1);
[numthreads(4, 4, 1)]
void main(uint3 dt : SV_DispatchThreadID)
{
    uint w = raw.Load(dt.x * 4u);
    float4 c = tex.SampleLevel(lin, float2(dt.xy) / 64.0, 0.0) * scale;
    c += float4(vol.Load(int4(dt, 0)));
    c += float4(arr.Load(int4(dt, 0)));
    c += cube.SampleLevel(lin, float3(1.0, 0.0, 0.0), 0.0);
    c += recs[w & 3u].a;
    c += many[w & 3u].Load(int3(0, 0, 0));
    c += unbounded[NonUniformResourceIndex(w)].Load(int3(0, 0, 0));
    c.xy += typed[w];
    c += ms.Load(int2(dt.xy), 0);
    c.x += tex.SampleCmpLevelZero(cmp, float2(0.5, 0.5), 0.5);
    c.y += (float)(dbl * 2.0);
    Rec r; r.a = c; r.b = out_recs.IncrementCounter() + count;
    out_recs[dt.x] = r;
    out_tex[dt.xy] = c + out_tex[dt.xy + 1u];
    out_vol[dt] = w;
    out_buf[dt.x] = (int)w;
    io.Store(0u, w);
}
"#,
    ),
    (
        "cs_wave",
        "cs_6_6",
        r#"globallycoherent RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
groupshared uint xcl_lds[8192];
[WaveSize(32)]
[numthreads(8, 4, 2)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint gi : SV_GroupIndex, uint3 dt : SV_DispatchThreadID)
{
    uint lane = WaveGetLaneIndex();
    xcl_lds[gi] = io.Load(gid.x * 4u) + lane;
    GroupMemoryBarrierWithGroupSync();
    uint64_t wide = (uint64_t)xcl_lds[(gi + 1u) & 8191u] * (uint64_t)gtid.y;
    res.Store2(dt.z * 8u, uint2((uint)wide, (uint)(wide >> 32)));
}
"#,
    ),
    (
        "ds_isoline",
        "ds_6_6",
        r#"struct Cp { float3 pos : POSITION; };
struct Pc { float edges[2] : SV_TessFactor; };
[domain("isoline")]
float4 main(Pc pc, float2 loc : SV_DomainLocation, const OutputPatch<Cp, 2> p) : SV_Position
{
    return float4(lerp(p[0].pos, p[1].pos, loc.x), pc.edges[0]);
}
"#,
    ),
    (
        "ds_quad",
        "ds_6_6",
        r#"struct Cp { float3 pos : POSITION; };
struct Pc { float edges[4] : SV_TessFactor; float inside[2] : SV_InsideTessFactor; };
[domain("quad")]
float4 main(Pc pc, float2 loc : SV_DomainLocation, const OutputPatch<Cp, 4> p) : SV_Position
{
    float3 a = lerp(p[0].pos, p[1].pos, loc.x);
    float3 b = lerp(p[3].pos, p[2].pos, loc.x);
    return float4(lerp(a, b, loc.y), pc.inside[1]);
}
"#,
    ),
    (
        "ds_tri",
        "ds_6_6",
        r#"struct Cp { float4 pos : POSITION; float2 uv : TEXCOORD0; };
struct Pc { float edges[3] : SV_TessFactor; float inside : SV_InsideTessFactor; float3 extra : EXTRA0; };
struct Out { float4 pos : SV_Position; float2 uv : TEXCOORD0; float3 e : TEXCOORD1; };
[domain("tri")]
Out main(Pc pc, float3 loc : SV_DomainLocation, const OutputPatch<Cp, 3> p, uint pid : SV_PrimitiveID)
{
    Out o;
    o.pos = p[0].pos * loc.x + p[1].pos * loc.y + p[2].pos * loc.z;
    o.uv = p[0].uv * pc.inside + float(pid);
    o.e = pc.extra * pc.edges[1];
    return o;
}
"#,
    ),
    (
        "gs_line",
        "gs_6_6",
        r#"struct V { float4 pos : SV_Position; float4 c : COLOR0; };
[maxvertexcount(2)]
void main(lineadj V v[4], inout LineStream<V> s)
{
    s.Append(v[1]);
    V o = v[2]; o.c = v[0].c + v[3].c;
    s.Append(o);
}
"#,
    ),
    (
        "gs_rect",
        "gs_6_6",
        r#"struct XclOut {
    float4 pos : SV_Position;
    float4 p0 : TEXCOORD0;
    float2 p1 : TEXCOORD1;
    uint p2 : TEXCOORD2;
    float3 clip : SV_ClipDistance0;
};
[maxvertexcount(6)]
void main(triangle XclOut v[3], inout TriangleStream<XclOut> s)
{
    uint c = 3u;
    if (v[0].pos.w > 0.0 && v[1].pos.w == v[0].pos.w) c = 1u;
    s.Append(v[0]); s.Append(v[1]); s.Append(v[2]);
    s.RestartStrip();
    if (c < 3u) {
        XclOut o = v[c];
        o.pos.xy = v[0].pos.xy + v[2].pos.xy - v[1].pos.xy;
        o.p0 = v[0].p0 + v[2].p0 - v[1].p0;
        s.Append(v[0]); s.Append(o); s.Append(v[2]);
    }
}
"#,
    ),
    (
        "gs_streams",
        "gs_6_6",
        r#"struct In { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct A { float4 pos : SV_Position; float2 uv : TEXCOORD0; uint layer : SV_RenderTargetArrayIndex; };
struct B { float3 n : NORMAL0; uint id : IDX; uint vp : SV_ViewportArrayIndex; };
[instance(2)]
[maxvertexcount(4)]
void main(point In v[1], uint prim : SV_PrimitiveID, uint inst : SV_GSInstanceID, inout PointStream<A> s0, inout PointStream<B> s1)
{
    A a; a.pos = v[0].pos; a.uv = v[0].uv; a.layer = inst;
    s0.Append(a);
    B b; b.n = v[0].pos.xyz; b.id = prim; b.vp = inst;
    s1.Append(b);
}
"#,
    ),
    (
        "hs_isoline",
        "hs_6_6",
        r#"struct Cp { float3 pos : POSITION; };
struct Pc { float edges[2] : SV_TessFactor; };
Pc pcf(InputPatch<Cp, 2> p)
{
    Pc o;
    o.edges[0] = 1.0; o.edges[1] = p[1].pos.y;
    return o;
}
[domain("isoline")]
[partitioning("pow2")]
[outputtopology("line")]
[outputcontrolpoints(2)]
[patchconstantfunc("pcf")]
Cp main(InputPatch<Cp, 2> p, uint i : SV_OutputControlPointID)
{
    return p[i];
}
"#,
    ),
    (
        "hs_quad",
        "hs_6_6",
        r#"struct Cp { float3 pos : POSITION; };
struct Pc { float edges[4] : SV_TessFactor; float inside[2] : SV_InsideTessFactor; };
Pc pcf(InputPatch<Cp, 4> p)
{
    Pc o;
    o.edges[0] = o.edges[1] = o.edges[2] = o.edges[3] = p[0].pos.x;
    o.inside[0] = o.inside[1] = 2.0;
    return o;
}
[domain("quad")]
[partitioning("integer")]
[outputtopology("triangle_ccw")]
[outputcontrolpoints(4)]
[patchconstantfunc("pcf")]
Cp main(InputPatch<Cp, 4> p, uint i : SV_OutputControlPointID)
{
    return p[i];
}
"#,
    ),
    (
        "hs_tri",
        "hs_6_6",
        r#"struct Cp { float4 pos : POSITION; float2 uv : TEXCOORD0; };
struct Pc { float edges[3] : SV_TessFactor; float inside : SV_InsideTessFactor; float3 extra : EXTRA0; };
Pc pcf(InputPatch<Cp, 3> p, uint pid : SV_PrimitiveID)
{
    Pc o;
    o.edges[0] = p[0].pos.x; o.edges[1] = p[1].pos.y; o.edges[2] = 3.0;
    o.inside = p[2].uv.x + float(pid);
    o.extra = p[0].pos.xyz;
    return o;
}
[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("pcf")]
[maxtessfactor(16.0)]
Cp main(InputPatch<Cp, 3> p, uint i : SV_OutputControlPointID, uint pid : SV_PrimitiveID)
{
    Cp o = p[i];
    o.uv.y += float(pid);
    return o;
}
"#,
    ),
    (
        "ms_pass",
        "ms_6_6",
        r#"RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
struct XclVert {
    float4 pos : SV_Position;
    float4 p0 : TEXCOORD0;
    float2 p1 : TEXCOORD1;
    uint p2 : TEXCOORD2;
    float3 p3 : SV_ClipDistance0;
    float p4 : SV_CullDistance0;
};
[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID, out vertices XclVert verts[64], out indices uint3 tris[126])
{
    uint rec = gid.x * 1024u;
    uint nv = min(res.Load(rec * 4u), 64u);
    uint np = min(res.Load(rec * 4u + 4u), 126u);
    SetMeshOutputCounts(nv, np);
    for (uint i = gi; i < nv; i += 32u) {
        uint b = (rec + 2u + i * 16u) * 4u;
        XclVert o;
        o.pos = asfloat(res.Load4(b));
        o.p0 = asfloat(res.Load4(b + 16u));
        o.p1 = asfloat(res.Load4(b + 32u).xy);
        o.p2 = res.Load4(b + 48u).x;
        o.p3 = asfloat(res.Load4(b + 64u).xyz);
        o.p4 = asfloat(res.Load4(b + 80u).x);
        verts[i] = o;
    }
    for (uint j = gi; j < np; j += 32u) {
        uint p = res.Load((rec + 1026u + j) * 4u);
        uint3 x = uint3(p & 0x3FFu, (p >> 10) & 0x3FFu, (p >> 20) & 0x3FFu);
        bool bad = or((p >> 31) != 0u, any(x >= nv));
        tris[j] = select(bad, uint3(0u, 0u, 0u), x);
    }
}
"#,
    ),
    (
        "ms_prim",
        "ms_6_6",
        r#"struct Payload { float4 base; uint count; };
struct Vert { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct Prim { uint id : SV_PrimitiveID; bool cull : SV_CullPrimitive; float4 color : COLOR0; uint layer : SV_RenderTargetArrayIndex; };
groupshared float4 shared_v[16];
[outputtopology("line")]
[numthreads(4, 2, 1)]
void main(uint gi : SV_GroupIndex, in payload Payload pl, out vertices Vert verts[8], out indices uint2 lines[4], out primitives Prim prims[4])
{
    SetMeshOutputCounts(8u, 4u);
    shared_v[gi] = pl.base;
    GroupMemoryBarrierWithGroupSync();
    verts[gi].pos = shared_v[(gi + 1u) & 7u];
    verts[gi].uv = pl.base.xy;
    if (gi < 4u) {
        lines[gi] = uint2(gi, gi + 4u);
        prims[gi].id = gi + pl.count;
        prims[gi].cull = false;
        prims[gi].color = pl.base;
        prims[gi].layer = 0u;
    }
}
"#,
    ),
    (
        "ps_basic",
        "ps_6_6",
        r#"struct XclIn {
    float4 xcl_pos : SV_Position;
     float4 a0 : TEXCOORD0;
    nointerpolation float4 a1 : TEXCOORD1;
    noperspective float4 a2 : TEXCOORD2;
    centroid float4 a3 : TEXCOORD3;
    sample float4 a4 : TEXCOORD4;
    noperspective centroid float4 a5 : TEXCOORD5;
    noperspective sample float4 a6 : TEXCOORD6;
    nointerpolation uint4 a7 : TEXCOORD7;
    nointerpolation int4 a8 : COLOR0;
    bool xcl_ff : SV_IsFrontFace;
    uint xcl_cov : SV_Coverage;
};
struct XclOut {
    float4 t0 : SV_Target0;
};
XclOut main(XclIn xi)
{
    XclOut xo = (XclOut)0;
    xo.t0 = xi.a0 + xi.a1 + xi.a2 + xi.a3 + xi.a4 + xi.a5 + xi.a6 + float4(xi.a7) + float4(xi.a8);
    xo.t0.x += xi.xcl_ff ? 1.0f : 0.0f;
    xo.t0.y += float(xi.xcl_cov) + xi.xcl_pos.z;
    return xo;
}
"#,
    ),
    (
        "ps_depth_ge",
        "ps_6_6",
        r#"struct XclIn {
    noperspective centroid float4 xcl_pos : SV_Position;
     float4 a0 : TEXCOORD0;
};
struct XclOut {
    float4 t1 : SV_Target1;
    float xcl_depth : SV_DepthGreaterEqual;
};
XclOut main(XclIn xi)
{
    XclOut xo = (XclOut)0;
    xo.t1 = xi.a0;
    xo.xcl_depth = xi.xcl_pos.z + 0.25;
    return xo;
}
"#,
    ),
    (
        "ps_depth_le",
        "ps_6_6",
        r#"struct XclIn {
    noperspective sample float4 xcl_pos : SV_Position;
    float2 a0 : TEXCOORD0;
    uint prim : SV_PrimitiveID;
    uint samp : SV_SampleIndex;
    uint rt : SV_RenderTargetArrayIndex;
    uint vp : SV_ViewportArrayIndex;
    float3 clip : SV_ClipDistance0;
    float cull : SV_CullDistance0;
};
struct XclOut {
    float4 t0 : SV_Target0;
    float xcl_depth : SV_DepthLessEqual;
    uint cov : SV_Coverage;
    uint sref : SV_StencilRef;
};
[earlydepthstencil]
XclOut main(XclIn xi)
{
    XclOut xo = (XclOut)0;
    xo.t0 = float4(xi.a0, float(xi.prim + xi.rt + xi.vp), xi.clip.y + xi.cull);
    xo.xcl_depth = xi.xcl_pos.z - 0.25;
    xo.cov = 1u << xi.samp;
    xo.sref = xi.prim;
    return xo;
}
"#,
    ),
    (
        "ps_mrt",
        "ps_6_6",
        r#"#define XCL_PIXEL 1
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
struct XclIn {
    float4 xcl_pos : SV_Position;
     float4 a0 : TEXCOORD0;
    nointerpolation uint4 a1 : TEXCOORD1;
};
struct XclOut {
    float4 t0 : SV_Target0;
    uint4 t1 : SV_Target1;
    int4 t2 : SV_Target2;
    float4 t3 : SV_Target3;
    float4 t4 : SV_Target4;
    float4 t5 : SV_Target5;
    float4 t6 : SV_Target6;
    float4 t7 : SV_Target7;
    float xcl_depth : SV_Depth;
};
XclOut main(XclIn xi)
{
    XclOut xo = (XclOut)0;
    bool xcl_live = xi.a0.x > 0.5;
    xo.t0 = xi.a0;
    xo.t1 = xi.a1;
    xo.t2 = int4(xi.a1);
    xo.t3.xy = ddx(xi.a0.xy);
    xo.t5 = xi.a0.yyyy;
    xo.t7.w = 1.0;
    xo.xcl_depth = xi.xcl_pos.z * 0.5;
    if (!xcl_live) discard;
    return xo;
}
"#,
    ),
    (
        "ps_unused",
        "ps_6_6",
        r#"struct XclIn {
    float4 xcl_pos : SV_Position;
     float4 a0 : TEXCOORD0;
     float4 a1 : TEXCOORD1;
};
struct XclOut {
    float4 xcl_unused : SV_Target7;
};
XclOut main(XclIn xi)
{
    XclOut xo = (XclOut)0;
    return xo;
}
"#,
    ),
    (
        "vs_basic",
        "vs_6_6",
        r#"struct XclIn {
    float4 a0 : TEXCOORD0;
    float2 a1 : TEXCOORD1;
    uint3 a2 : COLOR0;
    int a3 : BLENDINDICES3;
    float3 a4 : POSITION;
    uint xcl_vid : SV_VertexID;
    uint xcl_iid : SV_InstanceID;
};
struct XclOut {
    float4 pos : SV_Position;
    float4 p0 : TEXCOORD0;
    float2 p1 : TEXCOORD1;
    float p2 : TEXCOORD2;
    uint3 p3 : TEXCOORD3;
    int p4 : COLOR1;
};
XclOut main(XclIn xi)
{
    XclOut xo = (XclOut)0;
    xo.pos = float4(xi.a4, 1.0) + xi.a0;
    xo.p0 = xi.a0.wzyx;
    xo.p1 = xi.a1 * float(xi.xcl_vid);
    xo.p2 = float(xi.xcl_iid);
    xo.p3 = xi.a2;
    xo.p4 = xi.a3;
    return xo;
}
"#,
    ),
    (
        "vs_clip",
        "vs_6_6",
        r#"struct XclOut {
    float4 pos : SV_Position;
    float p0 : TEXCOORD0;
    float2 p1 : TEXCOORD1;
    float p2 : TEXCOORD2;
    float3 p3 : TEXCOORD3;
    float2 arr[3] : TEXCOORD4;
    float4 clip0 : SV_ClipDistance0;
    float2 clip1 : SV_ClipDistance1;
    float2 cull : SV_CullDistance0;
};
XclOut main(float4 a0 : ATTR0, float4x4 m : MAT, uint vid : SV_VertexID)
{
    XclOut xo = (XclOut)0;
    xo.pos = mul(m, a0);
    xo.p0 = a0.x;
    xo.p1 = a0.yz;
    xo.p3 = a0.xyz;
    xo.arr[vid % 3u] = a0.xy;
    xo.clip0 = a0;
    xo.clip1 = a0.xy;
    xo.cull = a0.zw;
    return xo;
}
"#,
    ),
    (
        "vs_min",
        "vs_6_6",
        r#"float4 main() : SV_Position
{
    return float4(0.0, 0.0, 0.0, 1.0);
}
"#,
    ),
    (
        "vs_unused",
        "vs_6_6",
        r#"struct XclIn {
    float4 a0 : TEXCOORD0;
    float4 a1 : TEXCOORD1;
    uint4 a2 : TEXCOORD2;
    uint xcl_vid : SV_VertexID;
    uint xcl_iid : SV_InstanceID;
};
struct XclOut {
    float4 pos : SV_Position;
    float4 p0 : TEXCOORD0;
    float3 p1 : TEXCOORD1;
    uint2 p2 : TEXCOORD2;
};
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
XclOut main(XclIn xi)
{
    XclOut xo = (XclOut)0;
    uint lane = WaveGetLaneIndex();
    xo.pos.xy = xi.a0.zw;
    xo.pos.w = 1.0;
    xo.p0.y = xi.a1.x + asfloat(io.Load(lane * 4u));
    xo.p2.x = xi.a2.w;
    return xo;
}
"#,
    ),
    (
        "as_heap",
        "as_6_6",
        r#"struct Payload { uint4 ids[8]; };
groupshared Payload pl;
[numthreads(8, 1, 1)]
void main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID)
{
    RWByteAddressBuffer mb = ResourceDescriptorHeap[gid.x];
    pl.ids[gi] = mb.Load4(gi * 16u);
    DispatchMesh(1u, 1u, 1u, pl);
}
"#,
    ),
    (
        "cs_res2",
        "cs_6_6",
        r#"Texture1D<float4> t1d : register(t0);
Texture1DArray<float2> t1da : register(t1);
Texture2DMSArray<float4> tmsa : register(t2);
TextureCubeArray<float4> tca : register(t3);
RaytracingAccelerationStructure tlas : register(t4);
StructuredBuffer<uint> sb[2] : register(t5, space7);
cbuffer A : register(b0) { float4 a; };
cbuffer B : register(b1) { uint4 b; };
struct Cb { float4 v; };
ConstantBuffer<Cb> cbs[3] : register(b4, space1);
SamplerState ss[3] : register(s4);
RWTexture2D<float4> u0 : register(u0);
RWTexture2D<float4> u1 : register(u1);
RWTexture2D<float4> u2 : register(u2);
RWTexture2D<float4> u3 : register(u3);
RWTexture2D<float4> u4 : register(u4);
RWTexture2D<float4> u5 : register(u5);
RWTexture2D<float4> u6 : register(u6);
RWTexture2D<float4> u7 : register(u7);
RWTexture2D<float4> u8 : register(u8);
RWTexture1D<uint> u9 : register(u9);
RWTexture2DArray<int> u10 : register(u10);
RWStructuredBuffer<int64_t> u11 : register(u11);
RWTexture2D<int64_t> u12[] : register(u0, space9);
[numthreads(1, 1, 1)]
void main(uint3 dt : SV_DispatchThreadID)
{
    float4 c = t1d.Load(int2(dt.x, 0)) + a + float4(b) + cbs[dt.x % 3u].v;
    c.xy += t1da.Load(int3(dt.xy, 0));
    c += tmsa.Load(int3(dt), 1);
    c += tca.SampleLevel(ss[dt.x % 3u], float4(1.0, 0.0, 0.0, 2.0), 0.0);
    c.x += float(sb[1][dt.x] + msad4(dt.x, uint2(dt.yz), uint4(0u, 0u, 0u, 0u)).x);
    RayQuery<RAY_FLAG_NONE> q;
    RayDesc ray; ray.Origin = c.xyz; ray.Direction = float3(0.0, 0.0, 1.0); ray.TMin = 0.0; ray.TMax = 1.0;
    q.TraceRayInline(tlas, 0u, 0xFFu, ray);
    q.Proceed();
    c.w += q.CommittedRayT();
    u0[dt.xy] = c; u1[dt.xy] = c; u2[dt.xy] = c; u3[dt.xy] = c; u4[dt.xy] = c; u5[dt.xy] = c; u6[dt.xy] = c; u7[dt.xy] = c; u8[dt.xy] = c;
    u9[dt.x] = dt.y;
    u10[dt] = (int)dt.z;
    InterlockedAdd(u11[dt.x], (int64_t)dt.y);
    InterlockedAdd(u12[NonUniformResourceIndex(dt.z)][dt.xy], (int64_t)1);
}
"#,
    ),
    (
        "ds_viewid",
        "ds_6_6",
        r#"struct Cp { float4 pos : POSITION; float2 uv : TEXCOORD0; };
struct Pc { float edges[3] : SV_TessFactor; float inside : SV_InsideTessFactor; float3 extra : EXTRA0; };
struct Out { float4 pos : SV_Position; float2 uv : TEXCOORD0; float3 e : TEXCOORD1; uint vp : SV_ViewportArrayIndex; };
[domain("tri")]
Out main(Pc pc, float3 loc : SV_DomainLocation, const OutputPatch<Cp, 5> p, uint view : SV_ViewID)
{
    Out o;
    o.pos = p[0].pos * loc.x + p[1].pos * loc.y + p[4].pos * loc.z;
    o.uv = p[0].uv * pc.inside + float(view);
    o.e = pc.extra * pc.edges[1];
    o.vp = view;
    return o;
}
"#,
    ),
    (
        "gs_viewid",
        "gs_6_6",
        r#"struct In { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct A { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct B { float3 n : NORMAL0; uint id : IDX; };
[maxvertexcount(4)]
void main(triangle In v[3], uint view : SV_ViewID, inout PointStream<A> s0, inout PointStream<B> s1)
{
    A a; a.pos = v[0].pos + v[view % 3u].pos; a.uv = v[1].uv;
    s0.Append(a);
    B b; b.n = v[2].pos.xyz; b.id = view;
    s1.Append(b);
}
"#,
    ),
    (
        "hs_viewid",
        "hs_6_6",
        r#"struct Cp { float4 pos : POSITION; float2 uv : TEXCOORD0; };
struct Pc { float edges[3] : SV_TessFactor; float inside : SV_InsideTessFactor; float3 extra : EXTRA0; };
Pc pcf(InputPatch<Cp, 3> p, uint view : SV_ViewID)
{
    Pc o;
    o.edges[0] = p[0].pos.x; o.edges[1] = p[1].pos.y; o.edges[2] = float(view);
    o.inside = p[2].uv.x;
    o.extra = p[0].pos.xyz * float(view);
    return o;
}
[domain("tri")]
[partitioning("fractional_even")]
[outputtopology("point")]
[outputcontrolpoints(5)]
[patchconstantfunc("pcf")]
Cp main(InputPatch<Cp, 3> p, uint i : SV_OutputControlPointID, uint view : SV_ViewID)
{
    Cp o = p[i % 3u];
    o.uv.y += float(view);
    return o;
}
"#,
    ),
    (
        "ms_viewid",
        "ms_6_6",
        r#"struct Vert { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct Prim { float4 color : COLOR0; uint id : SV_PrimitiveID; };
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void main(uint view : SV_ViewID, out vertices Vert verts[3], out indices uint3 tris[1], out primitives Prim prims[1])
{
    SetMeshOutputCounts(3u, 1u);
    for (uint i = 0u; i < 3u; i++) {
        verts[i].pos = float4(float(i), float(view), 0.0, 1.0);
        verts[i].uv = float2(0.0, 1.0);
    }
    tris[0] = uint3(0u, 1u, 2u);
    prims[0].color = float4(1.0, 0.0, float(view), 1.0);
    prims[0].id = 7u;
}
"#,
    ),
    (
        "ps_minprec",
        "ps_6_6",
        r#"struct XclIn {
    float4 pos : SV_Position;
    min16float4 a0 : TEXCOORD0;
    nointerpolation min16int2 a1 : TEXCOORD1;
    nointerpolation min16uint a2 : TEXCOORD2;
    nointerpolation bool flag : FLAG;
};
struct XclOut {
    min16float4 t0 : SV_Target0;
    min16uint4 t1 : SV_Target1;
};
XclOut main(XclIn xi)
{
    XclOut xo;
    xo.t0 = xi.a0 * (xi.flag ? (min16float)2.0 : (min16float)1.0);
    xo.t1 = min16uint4(xi.a1, xi.a2, xi.a2);
    return xo;
}
"#,
    ),
    (
        "ps_native16",
        "ps_6_6 -enable-16bit-types",
        r#"struct XclIn {
    float4 pos : SV_Position;
    float16_t4 a0 : TEXCOORD0;
    nointerpolation int16_t2 a1 : TEXCOORD1;
    nointerpolation uint16_t a2 : TEXCOORD2;
};
struct XclOut {
    float16_t4 t0 : SV_Target0;
    uint16_t4 t1 : SV_Target1;
};
XclOut main(XclIn xi)
{
    XclOut xo;
    xo.t0 = xi.a0;
    xo.t1 = uint16_t4(xi.a1, xi.a2, xi.a2);
    return xo;
}
"#,
    ),
    (
        "ps_special",
        "ps_6_6",
        r#"RasterizerOrderedTexture2D<float4> rov : register(u3);
struct XclIn {
    float4 pos : SV_Position;
    float3 bary : SV_Barycentrics;
    uint inner : SV_InnerCoverage;
    uint view : SV_ViewID;
    nointerpolation uint rate : SV_ShadingRate;
    float4 a0 : TEXCOORD0;
    float4 a1 : TEXCOORD1;
};
struct XclOut {
    float4 t0 : SV_Target0;
    float4 t3 : SV_Target3;
};
XclOut main(XclIn xi)
{
    XclOut xo;
    xo.t0 = float4(xi.bary, float(xi.inner + xi.rate)) + rov[uint2(xi.pos.xy)];
    xo.t3 = xi.a0 * float(xi.view);
    xo.t3.w = xi.a1.x;
    return xo;
}
"#,
    ),
    (
        "vs_mixed",
        "vs_6_6",
        r#"struct XclOut {
    float4 pos : SV_Position;
    float2 p0 : TEXCOORD0;
    float2 p1 : TEXCOORD1;
    float p2 : TEXCOORD2;
    float3 p3 : TEXCOORD3;
    float p4 : COLOR0;
    float2 p5 : TEXCOORD5;
    float p6 : COLOR1;
    float p7 : TEXCOORD7;
    uint p8 : TEXCOORD8;
    uint2 p9 : TEXCOORD9;
    int p10 : TEXCOORD10;
    float2 p11 : COLOR2;
    float2 p12 : TEXCOORD12;
    float p13 : TEXCOORD13;
    float p14 : TEXCOORD14;
    float p15 : TEXCOORD15;
    float p16 : TEXCOORD16;
    float2 p17 : TEXCOORD17;
    float3 p18 : TEXCOORD18;
    float p19 : COLOR3;
    uint layer : SV_RenderTargetArrayIndex;
    uint iid : SV_InstanceID;
};
XclOut main(float4 a : SV_Position, float4 arr[3] : ARR, bool flag : FLAG, uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    XclOut xo = (XclOut)0;
    xo.pos = a + arr[vid % 3u];
    xo.p0 = a.xy; xo.p1 = a.zw; xo.p2 = a.x; xo.p3 = a.xyz; xo.p4 = a.w; xo.p5 = a.yx; xo.p6 = flag ? 1.0 : 0.0;
    xo.p9 = uint2(vid, iid); xo.p12 = a.xx; xo.p14 = a.y; xo.p17 = a.zz; xo.p18.xz = a.xy; xo.p19 = a.z;
    xo.layer = iid; xo.iid = iid;
    return xo;
}
"#,
    ),
    (
        "vs_packed",
        "vs_6_6",
        r#"struct XclOut {
    float4 pos : SV_Position;
    float q0 : TEXCOORD0;
    float q1 : TEXCOORD1;
    float q2 : TEXCOORD2;
    float q3 : TEXCOORD3;
    float q4 : TEXCOORD4;
    float q5 : TEXCOORD5;
    float q6 : TEXCOORD6;
    float q7 : TEXCOORD7;
    float q8 : TEXCOORD8;
    float q9 : TEXCOORD9;
    float q10 : TEXCOORD10;
    float q11 : TEXCOORD11;
    float q12 : TEXCOORD12;
    float q13 : TEXCOORD13;
    float q14 : TEXCOORD14;
    float q15 : TEXCOORD15;
    float q16 : TEXCOORD16;
    float q17 : TEXCOORD17;
    float q18 : TEXCOORD18;
    float q19 : TEXCOORD19;
    float q20 : TEXCOORD20;
    float q21 : TEXCOORD21;
    float q22 : TEXCOORD22;
    float q23 : TEXCOORD23;
    float q24 : TEXCOORD24;
    float q25 : TEXCOORD25;
};
XclOut main(float4 a : ATTR0)
{
    XclOut xo = (XclOut)0;
    xo.pos = a;
    xo.q0 = a.x * 0.0;
    xo.q1 = a.x * 1.0;
    xo.q2 = a.x * 2.0;
    xo.q3 = a.x * 3.0;
    xo.q4 = a.x * 4.0;
    xo.q5 = a.x * 5.0;
    xo.q6 = a.x * 6.0;
    xo.q7 = a.x * 7.0;
    xo.q8 = a.x * 8.0;
    xo.q9 = a.x * 9.0;
    xo.q10 = a.x * 10.0;
    xo.q11 = a.x * 11.0;
    xo.q12 = a.x * 12.0;
    xo.q13 = a.x * 13.0;
    xo.q14 = a.x * 14.0;
    xo.q15 = a.x * 15.0;
    xo.q16 = a.x * 16.0;
    xo.q17 = a.x * 17.0;
    xo.q18 = a.x * 18.0;
    xo.q19 = a.x * 19.0;
    xo.q20 = a.x * 20.0;
    xo.q21 = a.x * 21.0;
    xo.q22 = a.x * 22.0;
    xo.q23 = a.x * 23.0;
    xo.q24 = a.x * 24.0;
    xo.q25 = a.x * 25.0;
    return xo;
}
"#,
    ),
    (
        "vs_viewid",
        "vs_6_6",
        r#"struct XclOut {
    float4 pos : SV_Position;
    float4 p0 : TEXCOORD0;
    float2 p1 : TEXCOORD1;
};
XclOut main(float4 a : ATTR0, float2 b : ATTR1, uint view : SV_ViewID)
{
    XclOut xo;
    xo.pos = a;
    xo.pos.x += float(view);
    xo.p0 = a.wzyx;
    xo.p1 = b * float(view);
    return xo;
}
"#,
    ),
];

/// What xcl passes to DXC (`gpu::recomp::dxc`).
const XCL_ARGS: &[&str] = &["-HV", "2021", "-Gis", "-O1"];
/// The same without the reflection part and with the validator version this crate writes for:
/// DXC then emits exactly the parts `container::build` does.
const BARE_ARGS: &[&str] = &["-HV", "2021", "-Gis", "-O1", "-Qstrip_reflect", "-validator-version", "1.8"];

fn dxc_dir() -> Option<PathBuf> {
    match std::env::var_os("XCL_DXIL_DXC") {
        Some(d) => Some(PathBuf::from(d)),
        None => {
            eprintln!("XCL_DXIL_DXC is not set: skipping the DXC oracle test");
            None
        }
    }
}

fn tool(dxc: &Path, name: &str) -> Command {
    let mut c = Command::new(dxc.join("bin").join(name));
    c.env("LD_LIBRARY_PATH", dxc.join("lib"));
    c
}

/// A scratch folder of this test's own.
fn work_dir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("xcl-dxil-container-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Compiled {
    container: Vec<u8>,
    ll: String,
}

fn compile(dxc: &Path, dir: &Path, name: &str, profile: &str, src: &str, args: &[&str]) -> Compiled {
    let (hlsl, out, ll) = (dir.join(format!("{name}.hlsl")), dir.join(format!("{name}.dxil")), dir.join(format!("{name}.ll")));
    std::fs::write(&hlsl, src).unwrap();
    let o = tool(dxc, "dxc").args(["-E", "main", "-T"]).args(profile.split(' ')).args(args).arg("-Fo").arg(&out).arg("-Fc").arg(&ll).arg(&hlsl).output().unwrap();
    assert!(o.status.success(), "dxc {name}: {}", String::from_utf8_lossy(&o.stderr));
    Compiled { container: std::fs::read(&out).unwrap(), ll: std::fs::read_to_string(&ll).unwrap() }
}

/// Run dxv; its message on failure.
fn dxv(dxc: &Path, args: &[&std::ffi::OsStr]) -> Result<String, String> {
    let o = tool(dxc, "dxv").args(args).output().unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    if o.status.success() && text.contains("Validation succeeded") {
        Ok(text)
    } else {
        Err(text)
    }
}

// ---- DXC's metadata, from the disassembly ----

#[derive(Debug, Clone, PartialEq)]
enum Md {
    Ref(u32),
    Int(i64),
    Str(String),
    Float(f32),
    Ints(Vec<u32>),
    Null,
    /// a function, a global, undef: nothing the container depends on
    Other,
}

struct Meta {
    nodes: HashMap<u32, Vec<Md>>,
    named: HashMap<String, Vec<Md>>,
}

/// Split at the commas outside brackets and strings.
fn split_operands(s: &str) -> Vec<&str> {
    let (mut out, mut depth, mut quoted, mut start) = (Vec::new(), 0i32, false, 0usize);
    for (i, c) in s.char_indices() {
        match c {
            '"' => quoted = !quoted,
            '[' | '{' | '(' if !quoted => depth += 1,
            ']' | '}' | ')' if !quoted => depth -= 1,
            ',' if !quoted && depth == 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    if !s[start..].trim().is_empty() {
        out.push(s[start..].trim());
    }
    out
}

fn operand(s: &str) -> Md {
    if let Some(q) = s.strip_prefix("!\"") {
        return Md::Str(q.strip_suffix('"').unwrap().to_string());
    }
    if let Some(n) = s.strip_prefix('!') {
        return Md::Ref(n.parse().unwrap());
    }
    if s == "null" {
        return Md::Null;
    }
    if let Some((ty, v)) = s.split_once(' ') {
        if ty == "float" {
            return Md::Float(match v.strip_prefix("0x") {
                Some(h) => f64::from_bits(u64::from_str_radix(h, 16).unwrap()) as f32,
                None => v.parse().unwrap(),
            });
        }
        if ty.starts_with('i') && ty[1..].chars().all(|c| c.is_ascii_digit()) {
            return match v {
                "true" => Md::Int(1),
                "false" => Md::Int(0),
                _ => Md::Int(v.parse().unwrap()),
            };
        }
    }
    if let Some(rest) = s.strip_prefix('[') {
        // [N x i32] [i32 a, i32 b, ...] or [N x i32] zeroinitializer
        let Some((ty, value)) = rest.split_once("] ") else { return Md::Other };
        let Some((n, elem)) = ty.split_once(" x ") else { return Md::Other };
        if elem == "i32" {
            if value == "zeroinitializer" {
                return Md::Ints(vec![0; n.parse().unwrap()]);
            }
            let list = value.strip_prefix('[').unwrap().strip_suffix(']').unwrap();
            return Md::Ints(list.split(',').map(|e| e.trim().strip_prefix("i32 ").unwrap().parse::<i64>().unwrap() as u32).collect());
        }
    }
    Md::Other
}

impl Meta {
    fn parse(ll: &str) -> Meta {
        let mut m = Meta { nodes: HashMap::new(), named: HashMap::new() };
        for line in ll.lines() {
            let Some(rest) = line.strip_prefix('!') else { continue };
            let Some((name, value)) = rest.split_once(" = ") else { continue };
            let value = value.strip_prefix("distinct ").unwrap_or(value);
            let Some(body) = value.strip_prefix("!{").and_then(|v| v.strip_suffix('}')) else { continue };
            let ops = split_operands(body).into_iter().map(operand).collect();
            match name.parse::<u32>() {
                Ok(n) => m.nodes.insert(n, ops).map(|_| ()),
                Err(_) => m.named.insert(name.to_string(), ops).map(|_| ()),
            };
        }
        m
    }
    fn node(&self, md: &Md) -> &[Md] {
        match md {
            Md::Ref(n) => &self.nodes[n],
            Md::Null => &[],
            other => panic!("not a node: {other:?}"),
        }
    }
}

fn int(md: &Md) -> i64 {
    match md {
        Md::Int(v) => *v,
        other => panic!("not an integer: {other:?}"),
    }
}

fn uint(md: &Md) -> u32 {
    int(md) as u32
}

/// The variants of an enum that counts from 0, to index with a metadata value.
macro_rules! variants {
    ($name:ident: $ty:ident = $($v:ident),* $(,)?) => {
        fn $name(v: &Md) -> $ty {
            const ALL: &[$ty] = &[$($ty::$v),*];
            let e = ALL[int(v) as usize];
            assert_eq!(e as i64, int(v), "the variant list is out of order");
            e
        }
    };
}

variants!(semantic: SemanticKind = Arbitrary, VertexID, InstanceID, Position, RenderTargetArrayIndex, ViewPortArrayIndex, ClipDistance, CullDistance,
    OutputControlPointID, DomainLocation, PrimitiveID, GSInstanceID, SampleIndex, IsFrontFace, Coverage, InnerCoverage, Target, Depth, DepthLessEqual,
    DepthGreaterEqual, StencilRef, DispatchThreadID, GroupID, GroupIndex, GroupThreadID, TessFactor, InsideTessFactor, ViewID, Barycentrics, ShadingRate,
    CullPrimitive, StartVertexLocation, StartInstanceLocation, Invalid);
variants!(comp_type: ComponentType = Invalid, I1, I16, U16, I32, U32, I64, U64, F16, F32, F64, SNormF16, UNormF16, SNormF32, UNormF32, SNormF64, UNormF64);
variants!(interpolation: InterpolationMode = Undefined, Constant, Linear, LinearCentroid, LinearNoperspective, LinearNoperspectiveCentroid, LinearSample,
    LinearNoperspectiveSample, Invalid);
variants!(resource_kind: ResourceKind = Invalid, Texture1D, Texture2D, Texture2DMS, Texture3D, TextureCube, Texture1DArray, Texture2DArray, Texture2DMSArray,
    TextureCubeArray, TypedBuffer, RawBuffer, StructuredBuffer, CBuffer, Sampler, TBuffer, RTAccelerationStructure, FeedbackTexture2D, FeedbackTexture2DArray);
variants!(sampler_kind: SamplerKind = Default, Comparison, Mono, Invalid);
variants!(input_primitive: InputPrimitive = Undefined, Point, Line, Triangle, Reserved4, Reserved5, LineWithAdjacency, TriangleWithAdjacency);
variants!(topology: PrimitiveTopology = Undefined, PointList, LineList, LineStrip, TriangleList, TriangleStrip);
variants!(domain: TessellatorDomain = Undefined, IsoLine, Tri, Quad);
variants!(partitioning: TessellatorPartitioning = Undefined, Integer, Pow2, FractionalOdd, FractionalEven);
variants!(output_primitive: TessellatorOutputPrimitive = Undefined, Point, Line, TriangleCW, TriangleCCW);
variants!(mesh_topology: MeshOutputTopology = Undefined, Line, Triangle);

/// Tag / value pairs of an extended property list.
fn tags<'a>(meta: &'a Meta, md: &Md) -> Vec<(u32, &'a Md)> {
    meta.node(md).chunks(2).map(|p| (uint(&p[0]), &p[1])).collect()
}

fn signature(meta: &Meta, md: &Md) -> Vec<SigElement> {
    let list = meta.node(md).iter();
    list.map(|e| {
        let n = meta.node(e);
        let Md::Str(name) = &n[1] else { panic!("element name") };
        let mut el = SigElement {
            name: name.clone(),
            indices: meta.node(&n[4]).iter().map(uint).collect(),
            kind: semantic(&n[3]),
            comp_type: comp_type(&n[2]),
            interp: interpolation(&n[5]),
            rows: uint(&n[6]),
            cols: uint(&n[7]),
            start_row: int(&n[8]) as i32,
            start_col: int(&n[9]) as i32,
            stream: 0,
            usage_mask: 0,
            dynamic_index_mask: 0,
        };
        for (tag, v) in tags(meta, &n[10]) {
            match tag {
                0 => el.stream = uint(v),
                2 => el.dynamic_index_mask = uint(v) as u8,
                3 => el.usage_mask = uint(v) as u8,
                _ => panic!("signature element tag {tag}"),
            }
        }
        el
    })
    .collect()
}

fn resources(meta: &Meta, md: &Md) -> Vec<Resource> {
    let mut out = Vec::new();
    let classes = [ResourceClass::SRV, ResourceClass::UAV, ResourceClass::CBuffer, ResourceClass::Sampler];
    for (class, list) in classes.into_iter().zip(meta.node(md)) {
        for r in meta.node(list) {
            let n = meta.node(r);
            let kind = match class {
                ResourceClass::CBuffer => ResourceKind::CBuffer,
                ResourceClass::Sampler => ResourceKind::Sampler,
                _ => resource_kind(&n[6]),
            };
            let Md::Str(name) = &n[2] else { panic!("resource name") };
            let mut res = Resource::new(class, uint(&n[0]), name, kind, uint(&n[4]), uint(&n[3]));
            res.range_size = uint(&n[5]);
            match class {
                ResourceClass::SRV => res.sample_count = uint(&n[7]),
                ResourceClass::UAV => {
                    res.globally_coherent = int(&n[7]) != 0;
                    res.has_counter = int(&n[8]) != 0;
                    res.rov = int(&n[9]) != 0;
                }
                ResourceClass::CBuffer => res.cbuffer_bytes = uint(&n[6]),
                _ => res.sampler_kind = sampler_kind(&n[6]),
            }
            for (tag, v) in tags(meta, n.last().unwrap()) {
                match (class, tag) {
                    (ResourceClass::SRV | ResourceClass::UAV, 0) => res.comp_type = comp_type(v),
                    (ResourceClass::SRV | ResourceClass::UAV, 1) => res.struct_stride = uint(v),
                    (ResourceClass::UAV, 3) => res.atomic64_use = int(v) != 0,
                    _ => panic!("resource tag {tag}"),
                }
            }
            out.push(res);
        }
    }
    out
}

/// Bytes of an LLVM type made of arrays, vectors and scalars.
fn type_size(t: &str) -> u32 {
    let t = t.trim();
    if let Some(inner) = t.strip_prefix('[').and_then(|x| x.strip_suffix(']')).or_else(|| t.strip_prefix('<').and_then(|x| x.strip_suffix('>'))) {
        let (n, elem) = inner.split_once(" x ").unwrap();
        return n.parse::<u32>().unwrap() * type_size(elem);
    }
    match t {
        "i8" | "i1" => 1,
        "i16" | "half" => 2,
        "i32" | "float" => 4,
        "i64" | "double" => 8,
        _ => panic!("groupshared type {t}"),
    }
}

/// The `ShaderInfo` of a module DXC disassembled.
fn info_from_ll(ll: &str) -> ShaderInfo {
    let meta = Meta::parse(ll);
    let model = meta.node(&meta.named["dx.shaderModel"][0]);
    let Md::Str(stage) = &model[0] else { panic!("shader model") };
    let (kind, major, minor) = parse_profile(&format!("{stage}_{}_{}", int(&model[1]), int(&model[2]))).unwrap();
    let entry = meta.node(&meta.named["dx.entryPoints"][0]);
    let Md::Str(name) = &entry[1] else { panic!("entry name") };

    let (mut flags, mut threads, mut wave_size) = (0u64, [0u32; 3], None);
    let mut props = match kind {
        ShaderKind::Vertex => Some(StageProps::Vertex),
        ShaderKind::Pixel => Some(StageProps::Pixel { early_depth_stencil: false }),
        _ => None,
    };
    let three = |md: &Md| -> [u32; 3] { meta.node(md).iter().map(uint).collect::<Vec<_>>().try_into().unwrap() };
    for (tag, v) in tags(&meta, &entry[4]) {
        match tag {
            0 => flags = int(v) as u64,
            1 => {
                let n = meta.node(v);
                props = Some(StageProps::Geometry {
                    input_primitive: input_primitive(&n[0]),
                    max_vertex_count: uint(&n[1]),
                    active_streams: uint(&n[2]),
                    stream_topology: topology(&n[3]),
                    instance_count: uint(&n[4]),
                });
            }
            2 => {
                let n = meta.node(v);
                props = Some(StageProps::Domain { domain: domain(&n[0]), input_control_points: uint(&n[1]) });
            }
            3 => {
                let n = meta.node(v);
                let Md::Float(max_tess_factor) = n[6] else { panic!("max tess factor") };
                props = Some(StageProps::Hull {
                    patch_constant_fn: String::new(),
                    input_control_points: uint(&n[1]),
                    output_control_points: uint(&n[2]),
                    domain: domain(&n[3]),
                    partitioning: partitioning(&n[4]),
                    output_primitive: output_primitive(&n[5]),
                    max_tess_factor,
                });
            }
            4 => threads = three(v),
            9 => {
                let n = meta.node(v);
                props = Some(StageProps::Mesh {
                    numthreads: three(&n[0]),
                    max_vertices: uint(&n[1]),
                    max_primitives: uint(&n[2]),
                    topology: mesh_topology(&n[3]),
                    payload_bytes: uint(&n[4]),
                });
            }
            10 => {
                let n = meta.node(v);
                props = Some(StageProps::Amplification { numthreads: three(&n[0]), payload_bytes: uint(&n[1]) });
            }
            11 => wave_size = Some(uint(&meta.node(v)[0])),
            _ => panic!("entry property tag {tag}"),
        }
    }
    if kind == ShaderKind::Compute {
        props = Some(StageProps::Compute { numthreads: threads, wave_size });
    }
    if let Some(StageProps::Pixel { early_depth_stencil }) = &mut props {
        *early_depth_stencil = flags & 8 != 0;
    }

    let mut info = ShaderInfo::new(kind, name, props.unwrap());
    info.sm = (major, minor);
    info.flags = flags;
    let sigs = meta.node(&entry[2]);
    if !sigs.is_empty() {
        info.inputs = signature(&meta, &sigs[0]);
        info.outputs = signature(&meta, &sigs[1]);
        info.patch_or_prim = signature(&meta, &sigs[2]);
    }
    info.resources = resources(&meta, &entry[3]);
    if let Some(state) = meta.named.get("dx.viewIdState") {
        let Md::Ints(words) = &meta.node(&state[0])[0] else { panic!("dx.viewIdState") };
        info.view_id_state = words.clone();
    }
    // only a mesh shader's PSV0 records the groupshared size
    for line in ll.lines().filter(|l| kind == ShaderKind::Mesh && l.starts_with('@')) {
        if let Some((_, rest)) = line.split_once("addrspace(3) global ") {
            // the type: a bracketed aggregate, or a word
            let mut depth = 0i32;
            let end = rest.char_indices().find_map(|(i, c)| {
                match c {
                    '[' | '<' => depth += 1,
                    ']' | '>' => depth -= 1,
                    _ => {}
                }
                (depth == 0 && (c == ']' || c == '>' || c == ' ' || c == ',')).then_some(if c == ' ' || c == ',' { i } else { i + 1 })
            });
            info.groupshared_bytes += type_size(&rest[..end.unwrap_or(rest.len())]);
        }
    }
    info
}

// ---- comparisons ----

fn part<'a>(c: &'a [u8], name: &[u8; 4]) -> &'a [u8] {
    container::part(c, name).unwrap_or_else(|| panic!("no {} part", String::from_utf8_lossy(name)))
}

fn hex(b: &[u8]) -> String {
    b.chunks(4).map(|w| w.iter().map(|x| format!("{x:02x}")).collect::<String>()).collect::<Vec<_>>().join(" ")
}

fn expect_bytes(what: &str, ours: &[u8], dxc: &[u8]) {
    if ours != dxc {
        let at = ours.iter().zip(dxc).position(|(a, b)| a != b).unwrap_or(ours.len().min(dxc.len()));
        // the 32 bytes before the first difference and the 64 from it on
        let window = |b: &[u8]| hex(&b[(at & !3).saturating_sub(32).min(b.len())..((at & !3) + 64).min(b.len())]);
        panic!("{what}: differs at byte {at} (ours {} bytes, DXC's {})\n ours: {}\n  dxc: {}", ours.len(), dxc.len(), window(ours), window(dxc));
    }
}

/// Every part this crate writes, serialised from DXC's metadata, against DXC's part.
fn compare_parts(name: &str, info: &ShaderInfo, dxc: &[u8]) {
    expect_bytes(&format!("{name} SFI0"), &container::feature_info(info.flags).to_le_bytes(), part(dxc, b"SFI0"));
    expect_bytes(&format!("{name} ISG1"), &sig::write(info, SigKind::Input).unwrap(), part(dxc, b"ISG1"));
    expect_bytes(&format!("{name} OSG1"), &sig::write(info, SigKind::Output).unwrap(), part(dxc, b"OSG1"));
    match container::part(dxc, b"PSG1") {
        Some(p) => expect_bytes(&format!("{name} PSG1"), &sig::write(info, SigKind::PatchConstOrPrim).unwrap(), p),
        None => assert!(info.patch_or_prim.is_empty(), "{name}: DXC wrote no PSG1"),
    }
    expect_bytes(&format!("{name} PSV0"), &psv::write(info).unwrap(), part(dxc, b"PSV0"));
    let bitcode = container::bitcode(dxc).unwrap();
    let mut shader_hash = vec![0u8; 4];
    shader_hash.extend_from_slice(&container::hash::md5(bitcode));
    expect_bytes(&format!("{name} HASH"), &shader_hash, part(dxc, b"HASH"));
}

/// The parts of what xcl gets from DXC today (validator 1.9, with reflection), one by one, and
/// DXC's digest recomputed.
#[test]
fn parts_match_dxc() {
    let Some(dxc) = dxc_dir() else { return };
    let dir = work_dir("parts");
    for &(name, profile, src) in SHADERS {
        let c = compile(&dxc, &dir, name, profile, src, XCL_ARGS);
        assert!(container::is_signed(&c.container), "{name}: DXC's digest is not what hash::digest computes");
        compare_parts(name, &info_from_ll(&c.ll), &c.container);
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// Without the reflection part DXC's container has the parts of ours and nothing else: the
/// whole container, digest included, is the same bytes.
#[test]
fn container_matches_dxc() {
    let Some(dxc) = dxc_dir() else { return };
    let dir = work_dir("whole");
    for &(name, profile, src) in SHADERS {
        let c = compile(&dxc, &dir, name, profile, src, BARE_ARGS);
        let info = info_from_ll(&c.ll);
        let ours = container::build(&info, container::bitcode(&c.container).unwrap()).unwrap();
        let names = |c: &[u8]| container::parts(c).unwrap().iter().map(|(n, _)| String::from_utf8_lossy(n).into_owned()).collect::<Vec<_>>();
        assert_eq!(names(&ours), names(&c.container), "{name}: parts");
        expect_bytes(&format!("{name} container"), &ours, &c.container);
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// dxv accepts what `build` assembles around DXC's bitcode, and signing leaves it as it is: the
/// digest is already the validator's.
#[test]
fn dxv_accepts_and_keeps_the_digest() {
    let Some(dxc) = dxc_dir() else { return };
    let dir = work_dir("dxv");
    for &(name, profile, src) in SHADERS {
        let c = compile(&dxc, &dir, name, profile, src, BARE_ARGS);
        let ours = container::build(&info_from_ll(&c.ll), container::bitcode(&c.container).unwrap()).unwrap();
        let (path, signed) = (dir.join(format!("{name}.ours.dxil")), dir.join(format!("{name}.signed.dxil")));
        std::fs::write(&path, &ours).unwrap();
        let mut o = std::ffi::OsString::from("-o=");
        o.push(&signed);
        let text = dxv(&dxc, &[&o, path.as_os_str()]).unwrap_or_else(|e| panic!("{name}: dxv: {e}"));
        assert!(text.contains("Unchanged DxilContainer"), "{name}: dxv signed it differently: {text}");
        expect_bytes(&format!("{name} signed"), &std::fs::read(&signed).unwrap(), &ours);
        // unsigned, dxv computes the digest build wrote
        let mut unsigned = ours.clone();
        unsigned[4..20].fill(0);
        std::fs::write(&path, &unsigned).unwrap();
        dxv(&dxc, &[&o, path.as_os_str()]).unwrap_or_else(|e| panic!("{name}: dxv (unsigned): {e}"));
        expect_bytes(&format!("{name} signed by dxv"), &std::fs::read(&signed).unwrap(), &ours);
        // and it does compare the parts: one flipped PSV0 byte is an error
        let mut broken = ours.clone();
        let at = broken.windows(4).position(|w| w == b"PSV0").unwrap() + 8 + 4 + 20;
        broken[at] ^= 1;
        std::fs::write(&path, &broken).unwrap();
        assert!(dxv(&dxc, &[path.as_os_str()]).is_err(), "{name}: dxv accepted a changed PSV0");
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// The ViewID tables are not the container's to choose: dxv reads the module's dx.viewIdState
/// and wants PSV0 to hold the same words. A table that says more than the metadata does is
/// refused like any other difference. (The other half was checked by hand with
/// IDxcAssembler: a module whose dx.viewIdState itself says more than the code does, or that
/// has none, validates as long as PSV0 agrees with it; the validator does not rerun the
/// dependency analysis.)
#[test]
fn dxv_wants_the_view_id_tables_of_the_metadata() {
    let Some(dxc) = dxc_dir() else { return };
    let dir = work_dir("viewid");
    for name in ["ps_depth_le", "vs_basic", "gs_streams", "hs_tri"] {
        let &(_, profile, src) = SHADERS.iter().find(|s| s.0 == name).unwrap();
        let c = compile(&dxc, &dir, name, profile, src, BARE_ARGS);
        let mut info = info_from_ll(&c.ll);
        let path = dir.join(format!("{name}.ours.dxil"));
        std::fs::write(&path, container::build(&info, container::bitcode(&c.container).unwrap()).unwrap()).unwrap();
        dxv(&dxc, &[path.as_os_str()]).unwrap_or_else(|e| panic!("{name}: dxv: {e}"));
        // one more dependency: the first input scalar reaches one more output scalar
        let (inputs, outputs) = (info.view_id_state[0], info.view_id_state[1]);
        assert!(inputs > 0 && outputs > 0 && outputs <= 32, "{name}: {:?}", info.view_id_state);
        let first_row = if info.flags & xcl_dxil::flags::bit::VIEW_ID != 0 { 3 } else { 2 };
        let free = (!info.view_id_state[first_row]).trailing_zeros();
        assert!(free < outputs, "{name}: the first input already reaches every output");
        info.view_id_state[first_row] |= 1 << free;
        std::fs::write(&path, container::build(&info, container::bitcode(&c.container).unwrap()).unwrap()).unwrap();
        let e = dxv(&dxc, &[path.as_os_str()]).expect_err("dxv accepted ViewID tables the metadata does not have");
        assert!(e.contains("PSV0"), "{name}: {e}");
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// The same comparison over a directory of sources (XCL_DXIL_CORPUS, e.g. the HLSL dumped from
/// xcl's gpu::recomp tests): every *.hlsl whose first line is the dump's
/// `// xdxc: -E <entry> -T <profile> <extra args>`, except those with a *.dxc-error.txt beside
/// them and profiles other than 6.6. The whole container has to be DXC's.
#[test]
fn corpus_containers_match_dxc() {
    let Some(dxc) = dxc_dir() else { return };
    let Some(corpus) = std::env::var_os("XCL_DXIL_CORPUS").map(PathBuf::from) else {
        eprintln!("XCL_DXIL_CORPUS is not set: skipping the corpus test");
        return;
    };
    let dir = work_dir("corpus");
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&corpus).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|e| e == "hlsl")).collect();
    sources.sort();
    let (mut compared, mut failed) = (0, Vec::new());
    for source in &sources {
        let name = source.file_name().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(source).unwrap();
        let Some(args) = text.lines().next().and_then(|l| l.strip_prefix("// xdxc: ")) else { continue };
        let args: Vec<&str> = args.split_whitespace().collect();
        let profile = args.iter().position(|&a| a == "-T").and_then(|i| args.get(i + 1)).copied().unwrap_or("");
        if !profile.ends_with("_6_6") || source.with_extension("dxc-error.txt").exists() {
            continue;
        }
        let (out, ll) = (dir.join("out.dxil"), dir.join("out.ll"));
        let o = tool(&dxc, "dxc").args(&args).args(BARE_ARGS).arg("-Fo").arg(&out).arg("-Fc").arg(&ll).arg(source).output().unwrap();
        assert!(o.status.success(), "dxc {name}: {}", String::from_utf8_lossy(&o.stderr));
        let theirs = std::fs::read(&out).unwrap();
        let info = info_from_ll(&std::fs::read_to_string(&ll).unwrap());
        compared += 1;
        match container::build(&info, container::bitcode(&theirs).unwrap()) {
            Ok(ours) if ours == theirs => {}
            Ok(ours) => {
                let part = container::parts(&ours).unwrap().into_iter().zip(container::parts(&theirs).unwrap()).find(|(a, b)| a != b).map(|(a, _)| String::from_utf8_lossy(&a.0).into_owned());
                failed.push(format!("{name}: differs in {part:?}"));
            }
            Err(e) => failed.push(format!("{name}: {e}")),
        }
    }
    let _ = std::fs::remove_dir_all(dir);
    eprintln!("{compared} corpus containers compared");
    assert!(failed.is_empty(), "{} of {compared} differ:\n{}", failed.len(), failed.join("\n"));
    assert!(compared > 0, "no sources in {}", corpus.display());
}
