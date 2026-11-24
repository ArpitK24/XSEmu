//! DXIL enumerations, generated from DXC's DxilConstants.h (v1.9.2609) by ~/.capy/work/dxil/tools/gen_consts.py.
//! Do not edit by hand: the names and numbers are DXC's.
#![allow(non_camel_case_types, non_upper_case_globals, clippy::upper_case_acronyms, dead_code)]

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum ShaderKind {
    Pixel = 0,
    Vertex = 1,
    Geometry = 2,
    Hull = 3,
    Domain = 4,
    Compute = 5,
    Library = 6,
    RayGeneration = 7,
    Intersection = 8,
    AnyHit = 9,
    ClosestHit = 10,
    Miss = 11,
    Callable = 12,
    Mesh = 13,
    Amplification = 14,
    Node = 15,
    Invalid = 16,
}
impl ShaderKind {
    pub const Last_1_2: ShaderKind = ShaderKind::Compute;
    pub const Last_1_4: ShaderKind = ShaderKind::Callable;
    pub const Last_1_7: ShaderKind = ShaderKind::Amplification;
    pub const Last_1_8: ShaderKind = ShaderKind::Node;
    pub const LastValid: ShaderKind = ShaderKind::Node;
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum SemanticKind {
    Arbitrary = 0,
    VertexID = 1,
    InstanceID = 2,
    Position = 3,
    RenderTargetArrayIndex = 4,
    ViewPortArrayIndex = 5,
    ClipDistance = 6,
    CullDistance = 7,
    OutputControlPointID = 8,
    DomainLocation = 9,
    PrimitiveID = 10,
    GSInstanceID = 11,
    SampleIndex = 12,
    IsFrontFace = 13,
    Coverage = 14,
    InnerCoverage = 15,
    Target = 16,
    Depth = 17,
    DepthLessEqual = 18,
    DepthGreaterEqual = 19,
    StencilRef = 20,
    DispatchThreadID = 21,
    GroupID = 22,
    GroupIndex = 23,
    GroupThreadID = 24,
    TessFactor = 25,
    InsideTessFactor = 26,
    ViewID = 27,
    Barycentrics = 28,
    ShadingRate = 29,
    CullPrimitive = 30,
    StartVertexLocation = 31,
    StartInstanceLocation = 32,
    Invalid = 33,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum ComponentType {
    Invalid = 0,
    I1 = 1,
    I16 = 2,
    U16 = 3,
    I32 = 4,
    U32 = 5,
    I64 = 6,
    U64 = 7,
    F16 = 8,
    F32 = 9,
    F64 = 10,
    SNormF16 = 11,
    UNormF16 = 12,
    SNormF32 = 13,
    UNormF32 = 14,
    SNormF64 = 15,
    UNormF64 = 16,
    PackedS8x32 = 17,
    PackedU8x32 = 18,
    I8 = 19,
    U8 = 20,
    F8_E4M3FN = 21,
    F8_E5M2 = 22,
    BFloat16 = 23,
    LastEntry = 24,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum InterpolationMode {
    Undefined = 0,
    Constant = 1,
    Linear = 2,
    LinearCentroid = 3,
    LinearNoperspective = 4,
    LinearNoperspectiveCentroid = 5,
    LinearSample = 6,
    LinearNoperspectiveSample = 7,
    Invalid = 8,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum ResourceClass {
    SRV = 0,
    UAV = 1,
    CBuffer = 2,
    Sampler = 3,
    Invalid = 4,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum ResourceKind {
    Invalid = 0,
    Texture1D = 1,
    Texture2D = 2,
    Texture2DMS = 3,
    Texture3D = 4,
    TextureCube = 5,
    Texture1DArray = 6,
    Texture2DArray = 7,
    Texture2DMSArray = 8,
    TextureCubeArray = 9,
    TypedBuffer = 10,
    RawBuffer = 11,
    StructuredBuffer = 12,
    CBuffer = 13,
    Sampler = 14,
    TBuffer = 15,
    RTAccelerationStructure = 16,
    FeedbackTexture2D = 17,
    FeedbackTexture2DArray = 18,
    NumEntries = 19,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum SamplerKind {
    Default = 0,
    Comparison = 1,
    Mono = 2,
    Invalid = 3,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum InputPrimitive {
    Undefined = 0,
    Point = 1,
    Line = 2,
    Triangle = 3,
    Reserved4 = 4,
    Reserved5 = 5,
    LineWithAdjacency = 6,
    TriangleWithAdjacency = 7,
    ControlPointPatch1 = 8,
    ControlPointPatch2 = 9,
    ControlPointPatch3 = 10,
    ControlPointPatch4 = 11,
    ControlPointPatch5 = 12,
    ControlPointPatch6 = 13,
    ControlPointPatch7 = 14,
    ControlPointPatch8 = 15,
    ControlPointPatch9 = 16,
    ControlPointPatch10 = 17,
    ControlPointPatch11 = 18,
    ControlPointPatch12 = 19,
    ControlPointPatch13 = 20,
    ControlPointPatch14 = 21,
    ControlPointPatch15 = 22,
    ControlPointPatch16 = 23,
    ControlPointPatch17 = 24,
    ControlPointPatch18 = 25,
    ControlPointPatch19 = 26,
    ControlPointPatch20 = 27,
    ControlPointPatch21 = 28,
    ControlPointPatch22 = 29,
    ControlPointPatch23 = 30,
    ControlPointPatch24 = 31,
    ControlPointPatch25 = 32,
    ControlPointPatch26 = 33,
    ControlPointPatch27 = 34,
    ControlPointPatch28 = 35,
    ControlPointPatch29 = 36,
    ControlPointPatch30 = 37,
    ControlPointPatch31 = 38,
    ControlPointPatch32 = 39,
    LastEntry = 40,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum PrimitiveTopology {
    Undefined = 0,
    PointList = 1,
    LineList = 2,
    LineStrip = 3,
    TriangleList = 4,
    TriangleStrip = 5,
    LastEntry = 6,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum TessellatorDomain {
    Undefined = 0,
    IsoLine = 1,
    Tri = 2,
    Quad = 3,
    LastEntry = 4,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum TessellatorPartitioning {
    Undefined = 0,
    Integer = 1,
    Pow2 = 2,
    FractionalOdd = 3,
    FractionalEven = 4,
    LastEntry = 5,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum TessellatorOutputPrimitive {
    Undefined = 0,
    Point = 1,
    Line = 2,
    TriangleCW = 3,
    TriangleCCW = 4,
    LastEntry = 5,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum MeshOutputTopology {
    Undefined = 0,
    Line = 1,
    Triangle = 2,
    LastEntry = 3,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum SigPointKind {
    /// Ordinary Vertex Shader input from Input Assembler
    VSIn = 0,
    /// Ordinary Vertex Shader output that may feed Rasterizer
    VSOut = 1,
    /// Patch Constant function non-patch inputs
    PCIn = 2,
    /// Hull Shader function non-patch inputs
    HSIn = 3,
    /// Hull Shader patch inputs - Control Points
    HSCPIn = 4,
    /// Hull Shader function output - Control Point
    HSCPOut = 5,
    /// Patch Constant function output - Patch Constant data passed to
    PCOut = 6,
    /// Domain Shader regular input - Patch Constant data plus system values
    DSIn = 7,
    /// Domain Shader patch input - Control Points
    DSCPIn = 8,
    /// Domain Shader output - vertex data that may feed Rasterizer
    DSOut = 9,
    /// Geometry Shader vertex input - qualified with primitive type
    GSVIn = 10,
    /// Geometry Shader non-vertex inputs (system values)
    GSIn = 11,
    /// Geometry Shader output - vertex data that may feed Rasterizer
    GSOut = 12,
    /// Pixel Shader input
    PSIn = 13,
    /// Pixel Shader output
    PSOut = 14,
    /// Compute Shader input
    CSIn = 15,
    /// Mesh Shader input
    MSIn = 16,
    /// Mesh Shader vertices output
    MSOut = 17,
    /// Mesh Shader primitives output
    MSPOut = 18,
    /// Amplification Shader input
    ASIn = 19,
    Invalid = 20,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum OpCode {
    /// reserved
    Reserved0 = 226,
    /// reserved
    Reserved1 = 227,
    /// reserved
    Reserved10 = 236,
    /// reserved
    Reserved11 = 237,
    /// reserved
    Reserved2 = 228,
    /// reserved
    Reserved3 = 229,
    /// reserved
    Reserved4 = 230,
    /// reserved
    Reserved5 = 231,
    /// reserved
    Reserved6 = 232,
    /// reserved
    Reserved7 = 233,
    /// reserved
    Reserved8 = 234,
    /// reserved
    Reserved9 = 235,
    /// reserved
    ReservedA0 = 259,
    /// reserved
    ReservedA1 = 260,
    /// reserved
    ReservedA2 = 261,
    /// reserved
    ReservedB28 = 290,
    /// reserved
    ReservedB29 = 291,
    /// reserved
    ReservedB30 = 292,
    /// reserved
    ReservedC0 = 293,
    /// reserved
    ReservedC1 = 294,
    /// reserved
    ReservedC2 = 295,
    /// reserved
    ReservedC3 = 296,
    /// reserved
    ReservedC4 = 297,
    /// reserved
    ReservedC5 = 298,
    /// reserved
    ReservedC6 = 299,
    /// reserved
    ReservedC7 = 300,
    /// reserved
    ReservedC8 = 301,
    /// reserved
    ReservedC9 = 302,
    /// reserved
    ReservedD0 = 305,
    /// reserved
    ReservedD1 = 306,
    /// reserved
    ReservedD2 = 307,
    /// reserved
    ReservedD3 = 308,
    /// Amplification shader intrinsic DispatchMesh
    DispatchMesh = 173,
    /// Used in an any hit shader to abort the ray query and the
    AcceptHitAndEndSearch = 156,
    /// Used in an any hit shader to reject an intersection and
    IgnoreHit = 155,
    /// returns a if a >= b, else b
    FMax = 35,
    /// returns a if a \< b, else b
    FMin = 36,
    /// multiply of 32-bit operands to produce the correct full 64-bit
    IMul = 41,
    /// IMax(a,b) returns a if a > b, else b
    IMax = 37,
    /// IMin(a,b) returns a if a \< b, else b
    IMin = 38,
    /// unsigned add of 32-bit operand with the carry
    UAddc = 44,
    /// unsigned subtract of 32-bit operands with the borrow
    USubb = 45,
    /// unsigned divide of the 32-bit operand src0 by the 32-bit operand
    UDiv = 43,
    /// multiply of 32-bit operands to produce the correct full 64-bit
    UMul = 42,
    /// unsigned integer maximum. UMax(a,b) = a > b ? a : b
    UMax = 39,
    /// unsigned integer minimum. UMin(a,b) = a \< b ? a : b
    UMin = 40,
    /// bitcast between different sizes
    BitcastF16toI16 = 125,
    /// bitcast between different sizes
    BitcastF32toI32 = 127,
    /// bitcast between different sizes
    BitcastF64toI64 = 129,
    /// bitcast between different sizes
    BitcastI16toF16 = 124,
    /// bitcast between different sizes
    BitcastI32toF32 = 126,
    /// bitcast between different sizes
    BitcastI64toF64 = 128,
    /// samples a texture after applying the input bias to the
    SampleCmpBias = 255,
    /// samples a texture using a gradient and compares a single component
    SampleCmpGrad = 254,
    /// provides a flattened index for a given
    FlattenedThreadIdInGroup = 96,
    /// reads the group ID (SV_GroupID)
    GroupId = 94,
    /// reads the thread ID
    ThreadId = 93,
    /// reads the thread ID within the group (SV_GroupThreadID)
    ThreadIdInGroup = 95,
    /// returns a handle for the output records
    AllocateNodeOutputRecords = 238,
    /// annotate handle with node properties
    AnnotateNodeHandle = 249,
    /// annotate handle with node record properties
    AnnotateNodeRecordHandle = 251,
    /// create a handle for an InputRecord
    CreateNodeInputRecordHandle = 250,
    /// Creates a handle to a NodeOutput
    CreateNodeOutputHandle = 247,
    /// returns the handle for the location in the output
    IndexNodeHandle = 248,
    /// calculates the level of detail
    CalculateLOD = 81,
    /// computes the rate of change per stamp in x direction.
    DerivCoarseX = 83,
    /// computes the rate of change per stamp in y direction.
    DerivCoarseY = 84,
    /// computes the rate of change per pixel in x direction.
    DerivFineX = 85,
    /// computes the rate of change per pixel in y direction.
    DerivFineY = 86,
    /// LoadOutputControlPoint
    LoadOutputControlPoint = 103,
    /// LoadPatchConstant
    LoadPatchConstant = 104,
    /// DomainLocation
    DomainLocation = 105,
    /// 2D half dot product with accumulate to float
    Dot2AddHalf = 162,
    /// signed dot product of 4 x i8 vectors packed into
    Dot4AddI8Packed = 163,
    /// unsigned dot product of 4 x u8 vectors packed into
    Dot4AddU8Packed = 164,
    /// Two-dimensional vector dot-product
    Dot2 = 54,
    /// Three-dimensional vector dot-product
    Dot3 = 55,
    /// Four-dimensional vector dot-product
    Dot4 = 56,
    /// computes the n-dimensional vector dot-product
    FDot = 311,
    /// legacy fuction to convert double to float
    LegacyDoubleToFloat = 132,
    /// legacy fuction to convert double to int32
    LegacyDoubleToSInt32 = 133,
    /// legacy fuction to convert double to uint32
    LegacyDoubleToUInt32 = 134,
    /// creates a double value
    MakeDouble = 101,
    /// splits a double into low and high parts
    SplitDouble = 102,
    /// returns the StartInstanceLocation from Draw*Instanced
    StartInstanceLocation = 257,
    /// returns the BaseVertexLocation from DrawIndexedInstanced or
    StartVertexLocation = 256,
    /// completes the current primitive topology at the specified stream
    CutStream = 98,
    /// emits a vertex to a given stream
    EmitStream = 97,
    /// equivalent to an EmitStream followed by a CutStream
    EmitThenCutStream = 99,
    /// GSInstanceID
    GSInstanceID = 100,
    /// retrieve node input/output record pointer in address space 6
    GetNodeRecordPtr = 239,
    /// annotate handle with resource properties
    AnnotateHandle = 216,
    /// create resource handle from binding
    CreateHandleFromBinding = 217,
    /// create resource handle from heap
    CreateHandleFromHeap = 218,
    /// returns the view index
    ViewID = 138,
    /// returns true on helper lanes in pixel shaders
    IsHelperLane = 221,
    /// OutputControlPointID
    OutputControlPointID = 107,
    /// StorePatchConstant
    StorePatchConstant = 106,
    /// PrimitiveID
    PrimitiveID = 108,
    /// Call a shader in the callable shader table supplied
    CallShader = 159,
    /// returns true if hit was accepted
    ReportHit = 158,
    /// initiates raytrace
    TraceRay = 157,
    /// allocates space for RayQuery and return handle
    AllocateRayQuery = 178,
    /// allocates space for RayQuery and return handle
    AllocateRayQuery2 = 258,
    /// aborts a ray query
    RayQuery_Abort = 181,
    /// returns candidate hit geometry index
    RayQuery_CandidateGeometryIndex = 203,
    /// returns candidate hit InstanceContributionToHitGroupIndex
    RayQuery_CandidateInstanceContributionToHitGroupIndex = 214,
    /// returns candidate hit instance ID
    RayQuery_CandidateInstanceID = 202,
    /// returns candidate hit instance index
    RayQuery_CandidateInstanceIndex = 201,
    /// returns candidate object ray direction
    RayQuery_CandidateObjectRayDirection = 206,
    /// returns candidate hit object ray origin
    RayQuery_CandidateObjectRayOrigin = 205,
    /// returns matrix for transforming from object-space to world-space
    RayQuery_CandidateObjectToWorld3x4 = 186,
    /// returns candidate hit geometry index
    RayQuery_CandidatePrimitiveIndex = 204,
    /// returns if current candidate procedural primitive is non opaque
    RayQuery_CandidateProceduralPrimitiveNonOpaque = 190,
    /// returns candidate triangle hit barycentrics
    RayQuery_CandidateTriangleBarycentrics = 193,
    /// returns if current candidate triangle is front facing
    RayQuery_CandidateTriangleFrontFace = 191,
    /// returns float representing the parametric point on the ray for the
    RayQuery_CandidateTriangleRayT = 199,
    /// returns uint candidate type (CANDIDATE_TYPE) of the current hit
    RayQuery_CandidateType = 185,
    /// returns matrix for transforming from world-space to object-space
    RayQuery_CandidateWorldToObject3x4 = 187,
    /// commits a non opaque triangle hit
    RayQuery_CommitNonOpaqueTriangleHit = 182,
    /// commits a procedural primitive hit
    RayQuery_CommitProceduralPrimitiveHit = 183,
    /// returns committed hit geometry index
    RayQuery_CommittedGeometryIndex = 209,
    /// returns committed hit InstanceContributionToHitGroupIndex
    RayQuery_CommittedInstanceContributionToHitGroupIndex = 215,
    /// returns committed hit instance ID
    RayQuery_CommittedInstanceID = 208,
    /// returns committed hit instance index
    RayQuery_CommittedInstanceIndex = 207,
    /// returns committed object ray direction
    RayQuery_CommittedObjectRayDirection = 212,
    /// returns committed hit object ray origin
    RayQuery_CommittedObjectRayOrigin = 211,
    /// returns matrix for transforming from object-space to world-space
    RayQuery_CommittedObjectToWorld3x4 = 188,
    /// returns committed hit geometry index
    RayQuery_CommittedPrimitiveIndex = 210,
    /// returns float representing the parametric point on the ray for the
    RayQuery_CommittedRayT = 200,
    /// returns uint status (COMMITTED_STATUS) of
    RayQuery_CommittedStatus = 184,
    /// returns committed triangle hit barycentrics
    RayQuery_CommittedTriangleBarycentrics = 194,
    /// returns if current committed triangle is front facing
    RayQuery_CommittedTriangleFrontFace = 192,
    /// returns matrix for transforming from world-space to object-space
    RayQuery_CommittedWorldToObject3x4 = 189,
    /// advances a ray query
    RayQuery_Proceed = 180,
    /// returns ray flags
    RayQuery_RayFlags = 195,
    /// returns float representing the parametric starting
    RayQuery_RayTMin = 198,
    /// initializes RayQuery for raytrace
    RayQuery_TraceRayInline = 179,
    /// returns world ray direction
    RayQuery_WorldRayDirection = 197,
    /// returns world ray origin
    RayQuery_WorldRayOrigin = 196,
    /// legacy fuction to convert half (f16) to float (f32)
    LegacyF16ToF32 = 131,
    /// legacy fuction to convert float (f32) to half (f16)
    LegacyF32ToF16 = 130,
    /// create resource handle from resource struct for library
    CreateHandleForLib = 160,
    /// emit a primitive's vertex indices in a mesh shader
    EmitIndices = 169,
    /// get the mesh payload which is from amplification shader
    GetMeshPayload = 170,
    /// Mesh shader intrinsic SetMeshOutputCounts
    SetMeshOutputCounts = 168,
    /// stores the value to mesh shader primitive output
    StorePrimitiveOutput = 172,
    /// stores the value to mesh shader vertex output
    StoreVertexOutput = 171,
    /// CycleCounterLegacy
    CycleCounterLegacy = 109,
    /// packs vector of 4 signed or unsigned values into a packed
    Pack4x8 = 220,
    /// returns the values of the attributes at the vertex.
    AttributeAtVertex = 137,
    /// returns the coverage mask input in a pixel shader
    Coverage = 91,
    /// discard the current pixel
    Discard = 82,
    /// evaluates an input attribute at pixel center
    EvalCentroid = 89,
    /// evaluates an input attribute at a sample location
    EvalSampleIndex = 88,
    /// evaluates an input attribute at pixel center with an offset
    EvalSnapped = 87,
    /// returns underestimated coverage input from conservative
    InnerCoverage = 92,
    /// returns the sample index in a sample-frequency pixel shader
    SampleIndex = 90,
    /// returns the result of a quad-level operation
    QuadOp = 123,
    /// reads from a lane in the quad
    QuadReadLaneAt = 122,
    /// compares boolean accross a quad
    QuadVote = 222,
    /// Given a bit range from the LSB of a number, places that number of
    Bfi = 53,
    /// The Width and Height values from the D3D12_DISPATCH_RAYS_DESC
    DispatchRaysDimensions = 146,
    /// The current x and y location within the Width and Height
    DispatchRaysIndex = 145,
    /// Matrix for transforming from object-space to world-space.
    ObjectToWorld = 151,
    /// Matrix for transforming from world-space to object-space.
    WorldToObject = 152,
    /// The world-space direction for the current ray.
    WorldRayDirection = 148,
    /// The world-space origin for the current ray.
    WorldRayOrigin = 147,
    /// Object-space direction for the current ray.
    ObjectRayDirection = 150,
    /// Object-space origin for the current ray.
    ObjectRayOrigin = 149,
    /// float representing the current parametric ending point for the ray
    RayTCurrent = 154,
    /// float representing the parametric starting point for the ray.
    RayTMin = 153,
    /// Returns the value passed as HitKind in ReportIntersection().
    HitKind = 143,
    /// The autogenerated index of the current geometry in the
    GeometryIndex = 213,
    /// The user-provided InstanceID on the bottom-level acceleration
    InstanceID = 141,
    /// The autogenerated index of the current instance in the
    InstanceIndex = 142,
    /// PrimitiveIndex for raytracing shaders
    PrimitiveIndex = 161,
    /// uint containing the current ray flags.
    RayFlags = 144,
    /// gathers the four texels that would be used in a
    TextureGather = 73,
    /// same as TextureGather, except this instrution
    TextureGatherCmp = 74,
    /// Gather raw elements from 4 texels with no type
    TextureGatherRaw = 223,
    /// gets the number of samples for a render target
    RenderTargetGetSampleCount = 77,
    /// gets the position of the specified sample
    RenderTargetGetSamplePosition = 76,
    /// samples a texture
    Sample = 60,
    /// samples a texture after applying the input bias to the mipmap level
    SampleBias = 61,
    /// samples a texture and compares a single component against
    SampleCmp = 64,
    /// samples a texture and compares a single component
    SampleCmpLevel = 224,
    /// samples a texture and compares a single component
    SampleCmpLevelZero = 65,
    /// samples a texture using a gradient to influence the way
    SampleGrad = 63,
    /// samples a texture using a mipmap-level offset
    SampleLevel = 62,
    /// gets the position of the specified sample
    Texture2DMSGetSamplePosition = 75,
    /// reads from a TypedBuffer
    BufferLoad = 68,
    /// writes to a RWTypedBuffer
    BufferStore = 69,
    /// atomically increments/decrements the hidden
    BufferUpdateCounter = 70,
    /// loads a value from a constant buffer resource
    CBufferLoad = 58,
    /// loads a value from a constant buffer resource
    CBufferLoadLegacy = 59,
    /// determines whether all values from a Sample, Gather, or Load
    CheckAccessFullyMapped = 71,
    /// creates the handle to a resource
    CreateHandle = 57,
    /// gets texture size information
    GetDimensions = 72,
    /// reads from a raw buffer and structured buffer
    RawBufferLoad = 139,
    /// writes to a RWByteAddressBuffer or RWStructuredBuffer
    RawBufferStore = 140,
    /// reads from a raw buffer and structured buffer
    RawBufferVectorLoad = 303,
    /// writes to a RWByteAddressBuffer or RWStructuredBuffer
    RawBufferVectorStore = 304,
    /// reads texel data without any filtering or sampling
    TextureLoad = 66,
    /// reads texel data without any filtering or sampling
    TextureStore = 67,
    /// stores texel data at specified sample index
    TextureStoreSample = 225,
    /// updates a feedback texture for a sampling operation
    WriteSamplerFeedback = 174,
    /// updates a feedback texture for a sampling
    WriteSamplerFeedbackBias = 175,
    /// updates a feedback texture for a sampling
    WriteSamplerFeedbackGrad = 177,
    /// updates a feedback texture for a sampling
    WriteSamplerFeedbackLevel = 176,
    /// Returns the attributes set for this HitObject
    HitObject_Attributes = 289,
    /// Creates a new HitObject representing a
    HitObject_FromRayQuery = 263,
    /// Creates a new HitObject representing a committed hit from a
    HitObject_FromRayQueryWithAttrs = 264,
    /// Returns the geometry index committed on hit
    HitObject_GeometryIndex = 281,
    /// Returns the HitKind of the hit
    HitObject_HitKind = 285,
    /// Returns the instance id committed on hit
    HitObject_InstanceID = 283,
    /// Returns the instance index committed on hit
    HitObject_InstanceIndex = 282,
    /// Represents the invocation of the CH/MS shader
    HitObject_Invoke = 267,
    /// Returns `true` if the HitObject is a NOP-HitObject
    HitObject_IsHit = 270,
    /// Returns `true` if the HitObject represents a miss
    HitObject_IsMiss = 269,
    /// Returns `true` if the HitObject represents a nop
    HitObject_IsNop = 271,
    /// Returns the root table constant for this HitObject and offset
    HitObject_LoadLocalRootTableConstant = 288,
    /// Creates a new HitObject representing a miss
    HitObject_MakeMiss = 265,
    /// Creates an empty nop HitObject
    HitObject_MakeNop = 266,
    /// Returns the ray direction in object space
    HitObject_ObjectRayDirection = 278,
    /// Returns the ray origin in object space
    HitObject_ObjectRayOrigin = 277,
    /// Returns the object to world space
    HitObject_ObjectToWorld3x4 = 279,
    /// Returns the primitive index committed on hit
    HitObject_PrimitiveIndex = 284,
    /// Returns the ray flags set in the HitObject
    HitObject_RayFlags = 272,
    /// Returns the current T value set in the HitObject
    HitObject_RayTCurrent = 274,
    /// Returns the TMin value set in the HitObject
    HitObject_RayTMin = 273,
    /// Returns a HitObject with updated shader table index
    HitObject_SetShaderTableIndex = 287,
    /// Returns the shader table index set for this HitObject
    HitObject_ShaderTableIndex = 286,
    /// Analogous to TraceRay but without invoking CH/MS
    HitObject_TraceRay = 262,
    /// Returns the ray direction in world space
    HitObject_WorldRayDirection = 276,
    /// Returns the ray origin in world space
    HitObject_WorldRayOrigin = 275,
    /// Returns the world to object space
    HitObject_WorldToObject3x4 = 280,
    /// Reorders the current thread
    MaybeReorderThread = 268,
    /// performs an atomic operation on two operands
    AtomicBinOp = 78,
    /// atomic compare and exchange to memory
    AtomicCompareExchange = 79,
    /// inserts a memory barrier in the shader
    Barrier = 80,
    /// Request a barrier for just the memory used by the specified object
    BarrierByMemoryHandle = 245,
    /// Request a barrier for a set of memory types
    BarrierByMemoryType = 244,
    /// Request a barrier for just the memory used by the node record
    BarrierByNodeRecordHandle = 246,
    /// Loads the value from shader input
    LoadInput = 4,
    /// Helper load operation for minprecision
    MinPrecXRegLoad = 2,
    /// Helper store operation for minprecision
    MinPrecXRegStore = 3,
    /// Stores the value to shader output
    StoreOutput = 5,
    /// Helper load operation
    TempRegLoad = 0,
    /// Helper store operation
    TempRegStore = 1,
    /// floating point multiply & add
    FMad = 46,
    /// fused multiply-add
    Fma = 47,
    /// Signed integer multiply & add
    IMad = 48,
    /// Integer bitfield extract
    Ibfe = 51,
    /// masked Sum of Absolute Differences.
    Msad = 50,
    /// Unsigned integer multiply & add
    UMad = 49,
    /// Unsigned integer bitfield extract
    Ubfe = 52,
    /// floating-point round to integral float.
    Round_ne = 26,
    /// floating-point round to integral float.
    Round_ni = 27,
    /// floating-point round to integral float.
    Round_pi = 28,
    /// floating-point round to integral float.
    Round_z = 29,
    /// Returns the arccosine of the specified value. Input should be a
    Acos = 15,
    /// Returns the arccosine of the specified value. Input should be a
    Asin = 16,
    /// Returns the arctangent of the specified value. The return value
    Atan = 17,
    /// returns cosine(theta) for theta in radians.
    Cos = 12,
    /// returns 2^exponent
    Exp = 21,
    /// returns the absolute value of the input value.
    FAbs = 6,
    /// extract fracitonal component.
    Frc = 22,
    /// returns the hyperbolic cosine of the specified value.
    Hcos = 18,
    /// returns the hyperbolic sine of the specified value.
    Hsin = 19,
    /// returns the hyperbolic tangent of the specified value.
    Htan = 20,
    /// Returns true if x is finite, false otherwise.
    IsFinite = 10,
    /// Returns true if x is +INF or -INF, false otherwise.
    IsInf = 9,
    /// Returns true if x is NAN or QNAN, false otherwise.
    IsNaN = 8,
    /// returns IsNormal
    IsNormal = 11,
    /// returns log base 2.
    Log = 23,
    /// returns reciprocal square root (1 / sqrt(src)
    Rsqrt = 25,
    /// clamps the result of a single or double precision floating
    Saturate = 7,
    /// returns sine(theta) for theta in radians.
    Sin = 13,
    /// returns square root
    Sqrt = 24,
    /// returns tan(theta) for theta in radians.
    Tan = 14,
    /// Reverses the order of the bits.
    Bfrev = 30,
    /// Counts the number of bits in the input integer.
    Countbits = 31,
    /// Returns the location of the first set bit starting from
    FirstbitLo = 32,
    /// Returns the location of the first set bit from the
    FirstbitSHi = 34,
    /// Returns the location of the first set bit starting from
    FirstbitHi = 33,
    /// unpacks 4 8-bit signed or unsigned values into int32 or
    Unpack4x8 = 219,
    /// Bitwise AND reduction of the vector returning a scalar
    VectorReduceAnd = 309,
    /// Bitwise OR reduction of the vector returning a scalar
    VectorReduceOr = 310,
    /// returns 1 if all the lanes have the same value
    WaveActiveAllEqual = 115,
    /// returns a struct with a bit set for each lane where
    WaveActiveBallot = 116,
    /// returns the result of the operation across all lanes
    WaveActiveBit = 120,
    /// returns the result the operation across waves
    WaveActiveOp = 119,
    /// returns the count of bits set to 1 across the wave
    WaveAllBitCount = 135,
    /// returns 1 if all the lanes evaluate the value to true
    WaveAllTrue = 114,
    /// returns 1 if any of the lane evaluates the value to true
    WaveAnyTrue = 113,
    /// returns the number of lanes in the wave
    WaveGetLaneCount = 112,
    /// returns the index of the current lane in the wave
    WaveGetLaneIndex = 111,
    /// returns 1 for the first lane in the wave
    WaveIsFirstLane = 110,
    /// returns the bitmask of active lanes that have the same value
    WaveMatch = 165,
    /// returns the count of bits set to 1 on groups
    WaveMultiPrefixBitCount = 167,
    /// returns the result of the operation on groups of
    WaveMultiPrefixOp = 166,
    /// returns the count of bits set to 1 on prior lanes
    WavePrefixBitCount = 136,
    /// returns the result of the operation on prior lanes
    WavePrefixOp = 121,
    /// returns the value from the specified lane
    WaveReadLaneAt = 117,
    /// returns the value from the first lane
    WaveReadLaneFirst = 118,
    /// returns true if the current thread group
    FinishedCrossGroupSharing = 243,
    /// returns the number of records that have been
    GetInputRecordCount = 242,
    /// returns how many levels of recursion remain
    GetRemainingRecursionLevels = 253,
    /// Select the next logical output count for an EmptyNodeOutput for
    IncrementOutputCount = 240,
    /// returns true if the specified output node is
    NodeOutputIsValid = 252,
    /// indicates all outputs for a given records are complete
    OutputComplete = 241,
    NumOpCodes_Dxil_1_9 = 312,
}
impl OpCode {
    pub const NumOpCodes_Dxil_1_0: OpCode = OpCode::AttributeAtVertex;
    pub const NumOpCodes_Dxil_1_1: OpCode = OpCode::RawBufferLoad;
    pub const NumOpCodes_Dxil_1_2: OpCode = OpCode::InstanceID;
    pub const NumOpCodes_Dxil_1_3: OpCode = OpCode::Dot2AddHalf;
    pub const NumOpCodes_Dxil_1_4: OpCode = OpCode::WaveMatch;
    pub const NumOpCodes_Dxil_1_5: OpCode = OpCode::AnnotateHandle;
    pub const NumOpCodes_Dxil_1_6: OpCode = OpCode::QuadVote;
    pub const NumOpCodes_Dxil_1_7: OpCode = OpCode::Reserved0;
    pub const NumOpCodes_Dxil_1_8: OpCode = OpCode::AllocateRayQuery2;
    pub const NumOpCodes: OpCode = OpCode::NumOpCodes_Dxil_1_9;
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum OpCodeClass {
    Reserved = 0,
    DispatchMesh = 1,
    AcceptHitAndEndSearch = 2,
    IgnoreHit = 3,
    BinaryWithCarryOrBorrow = 4,
    BinaryWithTwoOuts = 5,
    Binary = 6,
    BitcastF16toI16 = 7,
    BitcastF32toI32 = 8,
    BitcastF64toI64 = 9,
    BitcastI16toF16 = 10,
    BitcastI32toF32 = 11,
    BitcastI64toF64 = 12,
    SampleCmpBias = 13,
    SampleCmpGrad = 14,
    FlattenedThreadIdInGroup = 15,
    GroupId = 16,
    ThreadId = 17,
    ThreadIdInGroup = 18,
    AllocateNodeOutputRecords = 19,
    AnnotateNodeHandle = 20,
    AnnotateNodeRecordHandle = 21,
    CreateNodeInputRecordHandle = 22,
    IndexNodeHandle = 23,
    createNodeOutputHandle = 24,
    DebugBreak = 25,
    IsDebuggingEnabled = 26,
    CalculateLOD = 27,
    Unary = 28,
    LoadOutputControlPoint = 29,
    LoadPatchConstant = 30,
    DomainLocation = 31,
    Dot2AddHalf = 32,
    Dot4AddPacked = 33,
    Dot = 34,
    Dot2 = 35,
    Dot3 = 36,
    Dot4 = 37,
    LegacyDoubleToFloat = 38,
    LegacyDoubleToSInt32 = 39,
    LegacyDoubleToUInt32 = 40,
    MakeDouble = 41,
    SplitDouble = 42,
    StartInstanceLocation = 43,
    StartVertexLocation = 44,
    CutStream = 45,
    EmitStream = 46,
    EmitThenCutStream = 47,
    GSInstanceID = 48,
    GetNodeRecordPtr = 49,
    AnnotateHandle = 50,
    CreateHandleFromBinding = 51,
    CreateHandleFromHeap = 52,
    ViewID = 53,
    GetGroupWaveCount = 54,
    GetGroupWaveIndex = 55,
    IsHelperLane = 56,
    OutputControlPointID = 57,
    StorePatchConstant = 58,
    PrimitiveID = 59,
    CallShader = 60,
    ReportHit = 61,
    TraceRay = 62,
    AllocateRayQuery = 63,
    AllocateRayQuery2 = 64,
    RayQuery_Abort = 65,
    RayQuery_CandidateTriangleObjectPosition = 66,
    RayQuery_CommitNonOpaqueTriangleHit = 67,
    RayQuery_CommitProceduralPrimitiveHit = 68,
    RayQuery_CommittedTriangleObjectPosition = 69,
    RayQuery_Proceed = 70,
    RayQuery_StateMatrix = 71,
    RayQuery_StateScalar = 72,
    RayQuery_StateVector = 73,
    RayQuery_TraceRayInline = 74,
    LlvmInst = 75,
    LegacyF16ToF32 = 76,
    LegacyF32ToF16 = 77,
    CreateHandleForLib = 78,
    LinAlgConvert = 79,
    LinAlgCopyConvertMatrix = 80,
    LinAlgFillMatrix = 81,
    LinAlgMatVecMul = 82,
    LinAlgMatVecMulAdd = 83,
    LinAlgMatrixAccumulate = 84,
    LinAlgMatrixAccumulateToDescriptor = 85,
    LinAlgMatrixAccumulateToMemory = 86,
    LinAlgMatrixGetCoordinate = 87,
    LinAlgMatrixGetElement = 88,
    LinAlgMatrixLength = 89,
    LinAlgMatrixLoadFromDescriptor = 90,
    LinAlgMatrixLoadFromMemory = 91,
    LinAlgMatrixMultiply = 92,
    LinAlgMatrixMultiplyAccumulate = 93,
    LinAlgMatrixOuterProduct = 94,
    LinAlgMatrixQueryAccumulatorLayout = 95,
    LinAlgMatrixSetElement = 96,
    LinAlgMatrixStoreToDescriptor = 97,
    LinAlgMatrixStoreToMemory = 98,
    LinAlgVectorAccumulateToDescriptor = 99,
    EmitIndices = 100,
    GetMeshPayload = 101,
    SetMeshOutputCounts = 102,
    StorePrimitiveOutput = 103,
    StoreVertexOutput = 104,
    Nop = 105,
    CycleCounterLegacy = 106,
    Pack4x8 = 107,
    AttributeAtVertex = 108,
    Coverage = 109,
    Discard = 110,
    EvalCentroid = 111,
    EvalSampleIndex = 112,
    EvalSnapped = 113,
    InnerCoverage = 114,
    SampleIndex = 115,
    QuadOp = 116,
    QuadReadLaneAt = 117,
    QuadVote = 118,
    Quaternary = 119,
    DispatchRaysDimensions = 120,
    DispatchRaysIndex = 121,
    ObjectToWorld = 122,
    WorldToObject = 123,
    WorldRayDirection = 124,
    WorldRayOrigin = 125,
    ObjectRayDirection = 126,
    ObjectRayOrigin = 127,
    RayTCurrent = 128,
    RayTMin = 129,
    TriangleObjectPosition = 130,
    HitKind = 131,
    GeometryIndex = 132,
    InstanceID = 133,
    InstanceIndex = 134,
    PrimitiveIndex = 135,
    ClusterID = 136,
    RayFlags = 137,
    TextureGather = 138,
    TextureGatherCmp = 139,
    TextureGatherRaw = 140,
    RenderTargetGetSampleCount = 141,
    RenderTargetGetSamplePosition = 142,
    Sample = 143,
    SampleBias = 144,
    SampleCmp = 145,
    SampleCmpLevel = 146,
    SampleCmpLevelZero = 147,
    SampleGrad = 148,
    SampleLevel = 149,
    Texture2DMSGetSamplePosition = 150,
    BufferLoad = 151,
    BufferStore = 152,
    BufferUpdateCounter = 153,
    CBufferLoad = 154,
    CBufferLoadLegacy = 155,
    CheckAccessFullyMapped = 156,
    CreateHandle = 157,
    GetDimensions = 158,
    RawBufferLoad = 159,
    RawBufferStore = 160,
    RawBufferVectorLoad = 161,
    RawBufferVectorStore = 162,
    TextureLoad = 163,
    TextureStore = 164,
    TextureStoreSample = 165,
    WriteSamplerFeedback = 166,
    WriteSamplerFeedbackBias = 167,
    WriteSamplerFeedbackGrad = 168,
    WriteSamplerFeedbackLevel = 169,
    HitObject_Attributes = 170,
    HitObject_FromRayQuery = 171,
    HitObject_FromRayQueryWithAttrs = 172,
    HitObject_Invoke = 173,
    HitObject_LoadLocalRootTableConstant = 174,
    HitObject_MakeMiss = 175,
    HitObject_MakeNop = 176,
    HitObject_SetShaderTableIndex = 177,
    HitObject_StateMatrix = 178,
    HitObject_StateScalar = 179,
    HitObject_StateVector = 180,
    HitObject_TraceRay = 181,
    HitObject_TriangleObjectPosition = 182,
    MaybeReorderThread = 183,
    AtomicBinOp = 184,
    AtomicCompareExchange = 185,
    Barrier = 186,
    BarrierByMemoryHandle = 187,
    BarrierByMemoryType = 188,
    BarrierByNodeRecordHandle = 189,
    LoadInput = 190,
    MinPrecXRegLoad = 191,
    MinPrecXRegStore = 192,
    StoreOutput = 193,
    TempRegLoad = 194,
    TempRegStore = 195,
    Tertiary = 196,
    IsSpecialFloat = 197,
    UnaryBits = 198,
    Unpack4x8 = 199,
    VectorReduce = 200,
    WaveActiveAllEqual = 201,
    WaveActiveBallot = 202,
    WaveActiveBit = 203,
    WaveActiveOp = 204,
    WaveAllOp = 205,
    WaveAllTrue = 206,
    WaveAnyTrue = 207,
    WaveGetLaneCount = 208,
    WaveGetLaneIndex = 209,
    WaveIsFirstLane = 210,
    WaveMatch = 211,
    WaveMultiPrefixBitCount = 212,
    WaveMultiPrefixOp = 213,
    WavePrefixOp = 214,
    WaveReadLaneAt = 215,
    WaveReadLaneFirst = 216,
    FinishedCrossGroupSharing = 217,
    GetInputRecordCount = 218,
    GetRemainingRecursionLevels = 219,
    IncrementOutputCount = 220,
    NodeOutputIsValid = 221,
    OutputComplete = 222,
    /// exclusive last value of enumeration
    NumOpClasses = 223,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum AtomicBinOpCode {
    Add = 0,
    And = 1,
    Or = 2,
    Xor = 3,
    IMin = 4,
    IMax = 5,
    UMin = 6,
    UMax = 7,
    Exchange = 8,
    /// Must be last.
    Invalid = 9,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum BarrierMode {
    Invalid = 0,
    SyncThreadGroup = 1,
    UAVFenceGlobal = 2,
    UAVFenceThreadGroup = 4,
    TGSMFence = 8,
}
