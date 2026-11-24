//! The container's parts for four shaders whose `ShaderInfo` is written out by hand, against the
//! bytes DXC 1.9.2609 wrote for them (the sources are in tests/container_dxc.rs under the same
//! names, where the same comparison runs against a live DXC for many more). These show what a
//! front end has to put into `ShaderInfo`, field by field as DXC's dx.entryPoints metadata has
//! it, and they run without DXC.

use xcl_dxil::consts::*;
use xcl_dxil::container::{self, psv, sig, SigKind};
use xcl_dxil::shader::{Resource, ShaderInfo, SigElement, StageProps};

fn bytes(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect()
}

/// One row, in register `at` (row, column) or unallocated.
#[allow(clippy::too_many_arguments)]
fn element(name: &str, kind: SemanticKind, ty: ComponentType, interp: InterpolationMode, cols: u32, at: Option<(i32, i32)>, usage_mask: u8) -> SigElement {
    let (start_row, start_col) = at.unwrap_or((-1, -1));
    SigElement { name: name.to_string(), indices: vec![0], kind, comp_type: ty, interp, rows: 1, cols, start_row, start_col, stream: 0, usage_mask, dynamic_index_mask: 0 }
}

fn check(info: &ShaderInfo, sfi0: &str, isg1: &str, osg1: &str, psg1: Option<&str>, psv0: &str) {
    assert_eq!(container::feature_info(info.flags).to_le_bytes().to_vec(), bytes(sfi0), "SFI0");
    assert_eq!(sig::write(info, SigKind::Input).unwrap(), bytes(isg1), "ISG1");
    assert_eq!(sig::write(info, SigKind::Output).unwrap(), bytes(osg1), "OSG1");
    if let Some(psg1) = psg1 {
        assert_eq!(sig::write(info, SigKind::PatchConstOrPrim).unwrap(), bytes(psg1), "PSG1");
    }
    assert_eq!(psv::write(info).unwrap(), bytes(psv0), "PSV0");
    // and the same parts in the container
    let c = container::build(info, &[0x42, 0x43, 0xC0, 0xDE]).unwrap();
    assert_eq!(container::part(&c, b"PSV0").unwrap(), bytes(psv0));
    assert_eq!(container::part(&c, b"PSG1").map(|p| p.to_vec()), psg1.map(bytes));
}

/// cs_wave: no signatures (compute inputs are dx.op calls), two raw UAVs, a wave size.
#[test]
fn compute() {
    let mut info = ShaderInfo::new(ShaderKind::Compute, "main", StageProps::Compute { numthreads: [8, 4, 2], wave_size: Some(32) });
    // EnableRawAndStructuredBuffers, WaveOps, Int64Ops
    info.flags = 1 << 4 | 1 << 19 | 1 << 20;
    let mut io = Resource::new(ResourceClass::UAV, 0, "io", ResourceKind::RawBuffer, 0, 0);
    io.globally_coherent = true;
    info.resources = vec![io, Resource::new(ResourceClass::UAV, 1, "res", ResourceKind::RawBuffer, 1, 0)];
    check(&info, CS_WAVE_SFI0, CS_WAVE_ISG1, CS_WAVE_OSG1, None, CS_WAVE_PSV0);
}

/// ps_depth_le: packed system values, an input without a register (SV_SampleIndex) and
/// outputs without one (depth, coverage, stencil), usage masks, the ViewID tables.
#[test]
fn pixel() {
    use ComponentType::{F32, U32};
    use InterpolationMode::*;
    use SemanticKind as S;
    let mut info = ShaderInfo::new(ShaderKind::Pixel, "main", StageProps::Pixel { early_depth_stencil: true });
    // ForceEarlyDepthStencil, StencilRef
    info.flags = 1 << 3 | 1 << 11;
    info.inputs = vec![
        element("SV_Position", S::Position, F32, LinearNoperspectiveSample, 4, Some((0, 0)), 0b0100),
        element("TEXCOORD", S::Arbitrary, F32, Linear, 2, Some((1, 0)), 0b11),
        element("SV_PrimitiveID", S::PrimitiveID, U32, Constant, 1, Some((2, 0)), 1),
        element("SV_SampleIndex", S::SampleIndex, U32, Constant, 1, None, 0),
        element("SV_RenderTargetArrayIndex", S::RenderTargetArrayIndex, U32, Constant, 1, Some((3, 0)), 1),
        element("SV_ViewportArrayIndex", S::ViewPortArrayIndex, U32, Constant, 1, Some((3, 1)), 1),
        element("SV_ClipDistance", S::ClipDistance, F32, Linear, 3, Some((4, 0)), 0b010),
        element("SV_CullDistance", S::CullDistance, F32, Linear, 1, Some((4, 3)), 1),
    ];
    info.outputs = vec![
        element("SV_Target", S::Target, F32, Undefined, 4, Some((0, 0)), 0xF),
        element("SV_DepthLessEqual", S::DepthLessEqual, F32, Undefined, 1, None, 1),
        element("SV_Coverage", S::Coverage, U32, Undefined, 1, None, 1),
        element("SV_StencilRef", S::StencilRef, U32, Undefined, 1, None, 1),
    ];
    // 20 input components, 4 output components, then for each input component the outputs it
    // reaches: TEXCOORD.xy → SV_Target.xy, the primitive id, both array indices → .z, the clip
    // distance's y and the cull distance → .w
    info.view_id_state = vec![20, 4, 0, 0, 0, 0, 1, 2, 0, 0, 4, 0, 0, 0, 4, 4, 0, 0, 0, 8, 0, 8];
    check(&info, PS_DEPTH_LE_SFI0, PS_DEPTH_LE_ISG1, PS_DEPTH_LE_OSG1, None, PS_DEPTH_LE_PSV0);
}

/// hs_tri: control points in and out, patch constants with a three-row tessellation factor
/// packed beside an ordinary element.
#[test]
fn hull() {
    use ComponentType::F32;
    use InterpolationMode::*;
    use SemanticKind as S;
    let props = StageProps::Hull {
        patch_constant_fn: "pcf".to_string(),
        input_control_points: 3,
        output_control_points: 3,
        domain: TessellatorDomain::Tri,
        partitioning: TessellatorPartitioning::FractionalOdd,
        output_primitive: TessellatorOutputPrimitive::TriangleCW,
        max_tess_factor: 16.0,
    };
    let mut info = ShaderInfo::new(ShaderKind::Hull, "main", props);
    info.inputs = vec![element("POSITION", S::Arbitrary, F32, Linear, 4, Some((0, 0)), 0xF), element("TEXCOORD", S::Arbitrary, F32, Linear, 2, Some((1, 0)), 0b11)];
    info.outputs = info.inputs.clone();
    let mut edges = element("SV_TessFactor", S::TessFactor, F32, Undefined, 1, Some((0, 3)), 1);
    edges.rows = 3;
    edges.indices = vec![0, 1, 2];
    info.patch_or_prim = vec![edges, element("SV_InsideTessFactor", S::InsideTessFactor, F32, Undefined, 1, Some((3, 0)), 1), element("EXTRA", S::Arbitrary, F32, Undefined, 3, Some((0, 0)), 0b111)];
    info.view_id_state = vec![6, 6, 1, 2, 4, 8, 16, 32, 13, 9, 130, 4, 0, 4096, 0];
    check(&info, HS_TRI_SFI0, HS_TRI_ISG1, HS_TRI_OSG1, Some(HS_TRI_PSG1), HS_TRI_PSV0);
}

/// gs_streams: two output streams, the primitive id as an input without a register.
#[test]
fn geometry() {
    use ComponentType::{F32, U32};
    use InterpolationMode::*;
    use SemanticKind as S;
    let props = StageProps::Geometry { input_primitive: InputPrimitive::Point, max_vertex_count: 4, active_streams: 0b11, stream_topology: PrimitiveTopology::PointList, instance_count: 2 };
    let mut info = ShaderInfo::new(ShaderKind::Geometry, "main", props);
    let position = element("SV_Position", S::Position, F32, LinearNoperspective, 4, Some((0, 0)), 0xF);
    let uv = element("TEXCOORD", S::Arbitrary, F32, Linear, 2, Some((1, 0)), 0b11);
    info.inputs = vec![position.clone(), uv.clone(), element("SV_PrimitiveID", S::PrimitiveID, U32, Undefined, 1, None, 0)];
    let second = |mut e: SigElement| {
        e.stream = 1;
        e
    };
    info.outputs = vec![
        position,
        uv,
        element("SV_RenderTargetArrayIndex", S::RenderTargetArrayIndex, U32, Constant, 1, Some((2, 0)), 1),
        second(element("NORMAL", S::Arbitrary, F32, Linear, 3, Some((0, 0)), 0b111)),
        second(element("IDX", S::Arbitrary, U32, Constant, 1, Some((1, 0)), 1)),
        second(element("SV_ViewportArrayIndex", S::ViewPortArrayIndex, U32, Constant, 1, Some((1, 1)), 1)),
    ];
    info.view_id_state = vec![6, 9, 1, 2, 4, 8, 16, 32, 6, 1, 2, 4, 0, 0, 0, 0, 0];
    check(&info, GS_STREAMS_SFI0, GS_STREAMS_ISG1, GS_STREAMS_OSG1, None, GS_STREAMS_PSV0);
}

/// What `view_id_state` must agree with: the signatures' sizes.
#[test]
fn view_id_state_of_other_signatures_is_refused() {
    let mut info = ShaderInfo::new(ShaderKind::Vertex, "main", StageProps::Vertex);
    info.inputs = vec![element("ATTR", SemanticKind::Arbitrary, ComponentType::F32, InterpolationMode::Undefined, 4, Some((0, 0)), 0xF)];
    info.outputs = vec![element("SV_Position", SemanticKind::Position, ComponentType::F32, InterpolationMode::LinearNoperspective, 4, Some((0, 0)), 0xF)];
    assert!(psv::write(&info).is_ok(), "no state: empty tables");
    info.view_id_state = vec![4, 4, 1, 2, 4, 8];
    assert!(psv::write(&info).is_ok());
    for bad in [vec![4, 4, 1, 2, 4], vec![4, 4, 1, 2, 4, 8, 0], vec![8, 4, 1, 2, 4, 8, 0, 0, 0, 0], vec![4, 9, 1, 2, 4, 8]] {
        info.view_id_state = bad;
        assert!(psv::write(&info).is_err());
    }
}

// SFI0, ISG1, OSG1, PSG1 and PSV0 of DXC's containers (dxc -E main -T <stage>_6_6 -HV 2021 -Gis -O1)

const PS_DEPTH_LE_SFI0: &str = "\
    0002000000000000\
";
const PS_DEPTH_LE_ISG1: &str = "\
    08000000080000000000000008010000000000000100000003000000000000000f040000000000000000000014010000000000000000000003000000\
    010000000303000000000000000000001d010000000000000700000001000000020000000101000000000000000000003b0100000000000004000000\
    010000000300000001010000000000000000000055010000000000000500000001000000030000000202000000000000000000006b01000000000000\
    0200000003000000040000000702000000000000000000007b010000000000000300000003000000040000000808000000000000000000002c010000\
    000000000a00000001000000ffffffff010000000000000053565f506f736974696f6e00544558434f4f52440053565f5072696d6974697665494400\
    53565f53616d706c65496e6465780053565f52656e6465725461726765744172726179496e6465780053565f56696577706f72744172726179496e64\
    65780053565f436c697044697374616e63650053565f43756c6c44697374616e63650000\
";
const PS_DEPTH_LE_OSG1: &str = "\
    04000000080000000000000088000000000000004000000003000000000000000f000000000000000000000092000000000000004400000003000000\
    ffffffff010e00000000000000000000a4000000000000004200000001000000ffffffff010e00000000000000000000b00000000000000045000000\
    01000000ffffffff010e00000000000053565f5461726765740053565f44657074684c657373457175616c0053565f436f7665726167650053565f53\
    74656e63696c526566000000\
";
const PS_DEPTH_LE_PSV0: &str = "\
    340000000101000000000000000000000000000000000000ffffffff0000000008040005010000000000000000000000000000000a00000000000000\
    1000000000544558434f4f5244006d61696e000001000000000000001000000000000000000000000100440303070000010000000000000001014200\
    0302000000000000000000000102410a0101000000000000000000000100010c01010000000000000000000001034104010100000000000000000000\
    010351050101000000000000000000000104430603020000000000000000000001047107030200000000000000000000010044100300000000000000\
    00000000010001120300000000000000000000000100010e010000000000000000000000010001140100000000000000000000000000000000000000\
    010000000200000000000000000000000400000000000000000000000000000004000000040000000000000000000000000000000800000000000000\
    08000000\
";
const HS_TRI_SFI0: &str = "\
    0000000000000000\
";
const HS_TRI_ISG1: &str = "\
    02000000080000000000000048000000000000000000000003000000000000000f0f0000000000000000000051000000000000000000000003000000\
    010000000303000000000000504f534954494f4e00544558434f4f5244000000\
";
const HS_TRI_OSG1: &str = "\
    02000000080000000000000048000000000000000000000003000000000000000f000000000000000000000051000000000000000000000003000000\
    01000000030c000000000000504f534954494f4e00544558434f4f5244000000\
";
const HS_TRI_PSG1: &str = "\
    050000000800000000000000a8000000000000000d0000000300000000000000080700000000000000000000ca000000000000000000000003000000\
    00000000070800000000000000000000a8000000010000000d0000000300000001000000080700000000000000000000a8000000020000000d000000\
    0300000002000000080700000000000000000000b6000000000000000e0000000300000003000000010e00000000000053565f54657373466163746f\
    720053565f496e7369646554657373466163746f7200455854524100\
";
const HS_TRI_PSV0: &str = "\
    340000000300000003000000020000000300000000000000ffffffff0300040002020302020000000000000000000000000000002b00000000000000\
    3000000000504f534954494f4e00544558434f4f524400504f534954494f4e00544558434f4f5244004558545241006d61696e000400000000000000\
    00000000010000000200000010000000010000000000000001004400030200000a000000000000000101420003020000130000000000000001004400\
    030200001c0000000000000001014200030200000000000001000000030071190300000000000000000000000103411a030000002500000000000000\
    010043000300000001000000020000000400000008000000100000002000000000000000000000000900000082000000040000000000000000100000\
    000000000000000000000000\
";
const GS_STREAMS_SFI0: &str = "\
    0000000000000000\
";
const GS_STREAMS_ISG1: &str = "\
    03000000080000000000000068000000000000000100000003000000000000000f0f0000000000000000000074000000000000000000000003000000\
    010000000303000000000000000000007d000000000000000700000001000000ffffffff010000000000000053565f506f736974696f6e0054455843\
    4f4f52440053565f5072696d6974697665494400\
";
const GS_STREAMS_OSG1: &str = "\
    060000000800000000000000c8000000000000000100000003000000000000000f0000000000000000000000d4000000000000000000000003000000\
    01000000030c00000000000000000000dd00000000000000040000000100000002000000010e00000000000001000000f70000000000000000000000\
    0300000000000000070800000000000001000000fe00000000000000000000000100000001000000010e000000000000010000000201000000000000\
    050000000100000001000000020d00000000000053565f506f736974696f6e00544558434f4f52440053565f52656e64657254617267657441727261\
    79496e646578004e4f524d414c004944580053565f56696577706f72744172726179496e64657800\
";
const GS_STREAMS_PSV0: &str = "\
    340000000100000001000000030000000100000000000000ffffffff0200040003060002030200000000000000000000000000001e00000000000000\
    2400000000544558434f4f524400544558434f4f5244004e4f524d414c00494458006d61696e00000100000000000000100000000000000000000000\
    01004403030400000100000000000000010142000302000000000000000000000100010a01000000000000000000000001004403030400000a000000\
    00000000010142000302000000000000000000000102410401010000130000000000000001004300030210001a000000000000000101410001011000\
    000000000000000001015105010110000100000002000000040000000800000010000000200000000000000000000000010000000200000004000000\
    0000000000000000000000000000000000000000\
";
const CS_WAVE_SFI0: &str = "\
    00c0000000000000\
";
const CS_WAVE_ISG1: &str = "\
    0000000008000000\
";
const CS_WAVE_OSG1: &str = "\
    0000000008000000\
";
const CS_WAVE_PSV0: &str = "\
    340000000000000000000000000000000000000020000000200000000500000000000000000000000800000004000000020000000100000002000000\
    18000000070000000000000000000000000000000b00000000000000070000000000000001000000010000000b0000000000000008000000006d6169\
    6e00000000000000\
";
