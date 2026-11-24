//! DXC's signature packing: which register (row) and column each element of a signature
//! gets. A port of `DxilSignatureAllocator::PackPrefixStable` (include/dxc/HLSL/
//! DxilSignatureAllocator.inl), the strategy every shader model defaults to
//! (`ShaderModel::GetDefaultPackingStrategy`), and of `PackDxilSignature`
//! (lib/HLSL/DxilPackSignatureElement.cpp), which picks the scheme by signature point.
//!
//! The result has to equal DXC's, not merely be valid: D3D12 links the stages of a pipeline
//! by register, and a pipeline can mix a stage compiled here with one DXC compiled.
//!
//! Elements are 32 bits wide here (64-bit and 16-bit signature types are rejected before
//! packing), so DXC's data-width conflict never arises and is left out.

use crate::consts::InterpolationMode;

/// DXC's `SemanticInterpretationKind`: what a semantic is at a signature point.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Interpretation {
    /// not available at this point
    NA,
    /// a system value in the signature
    SV,
    /// a system-generated value: packed after everything else in its register
    SGV,
    /// an ordinary element, whatever its name
    Arb,
    /// read through an operation of its own, no element
    NotInSig,
    /// an element without a register (SV_Depth, SV_Coverage out, ...)
    NotPacked,
    /// SV_Target: the register is the semantic index
    Target,
    TessFactor,
    /// read through an operation of its own, but an element (without a register) exists
    Shadow,
    ClipCull,
}

impl Interpretation {
    /// DXC's `DxilSignature::ShouldBeAllocated`.
    pub fn allocated(self) -> bool {
        !matches!(self, Interpretation::NA | Interpretation::NotInSig | Interpretation::NotPacked | Interpretation::Shadow)
    }
}

/// How a whole signature is packed (DXC's `PackingKind` of its signature point).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Packing {
    /// vertex shader inputs: every element starts the next free row
    InputAssembler,
    /// everything between stages (also what DXC calls PatchConstant: the same allocator)
    Vertex,
    /// pixel shader outputs
    Target,
}

/// One element to pack. `at` is the result: (row, column), None when it has no register.
#[derive(Copy, Clone, Debug)]
pub struct Elem {
    pub interpretation: Interpretation,
    pub mode: InterpolationMode,
    pub rows: u32,
    pub cols: u32,
    /// first semantic index (the row of an SV_Target)
    pub index: u32,
    /// geometry shader output stream (each stream is packed on its own)
    pub stream: u32,
    pub at: Option<(u32, u32)>,
}

const OCCUPIED: u8 = 1 << 0;
const ARBITRARY: u8 = 1 << 1;
const SGV: u8 = 1 << 2;
const SV: u8 = 1 << 3;
const TESS_FACTOR: u8 = 1 << 4;
const CLIP_CULL: u8 = 1 << 5;
const CONFLICTS_WITH_INDEXED: u8 = SGV | SV;

const INDEXED_UP: u8 = 1 << 0;
const INDEXED_DOWN: u8 = 1 << 1;

fn element_flags(i: Interpretation) -> u8 {
    match i {
        Interpretation::Arb => ARBITRARY,
        Interpretation::SV => SV,
        Interpretation::SGV => SGV,
        Interpretation::TessFactor => TESS_FACTOR,
        Interpretation::ClipCull => CLIP_CULL,
        _ => 0,
    }
}

/// Element kinds that may not sit to the left of an element with `flags` in one register.
fn conflicts_left(flags: u8) -> u8 {
    let mut c = 0;
    if flags & ARBITRARY != 0 {
        c |= SGV | SV | TESS_FACTOR | CLIP_CULL;
    }
    if flags & (SV | TESS_FACTOR | CLIP_CULL) != 0 {
        c |= SGV;
    }
    c
}

/// Element kinds that may not sit to the right of it.
fn conflicts_right(flags: u8) -> u8 {
    let mut c = 0;
    if flags & SGV != 0 {
        c |= ARBITRARY | SV | TESS_FACTOR | CLIP_CULL;
    }
    if flags & (SV | TESS_FACTOR | CLIP_CULL) != 0 {
        c |= ARBITRARY;
    }
    c
}

/// Why an element may not go somewhere (DXC's `ConflictType`).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Conflict {
    /// outside the registers or the four columns
    Fit,
    /// an indexed element in a register with one that may not be indexed along with it
    Indexed,
    /// a register has one interpolation mode
    Mode,
    /// not enough free components in the register
    Full,
    Overlap,
    /// arbitrary, then system values, then system-generated values in one register
    Order,
}

fn index_flags(row: u32, rows: u32) -> u8 {
    (if row > 0 { INDEXED_UP } else { 0 }) | (if row + 1 < rows { INDEXED_DOWN } else { 0 })
}

/// DXC's `PackedRegister`: per component the element flags (occupied) or the kinds that may
/// no longer go there (free).
#[derive(Copy, Clone)]
struct Register {
    flags: [u8; 4],
    mode: InterpolationMode,
    index_flags: u8,
    indexing_fixed: bool,
}

impl Register {
    const EMPTY: Register = Register { flags: [0; 4], mode: InterpolationMode::Undefined, index_flags: 0, indexing_fixed: false };

    fn row_conflict(&self, flags: u8, index: u8, mode: InterpolationMode, width: u32) -> Option<Conflict> {
        if self.index_flags != 0 && flags & CONFLICTS_WITH_INDEXED != 0 {
            return Some(Conflict::Indexed);
        }
        if self.indexing_fixed && (index | self.index_flags) != self.index_flags {
            return Some(Conflict::Indexed);
        }
        if flags & TESS_FACTOR != 0 && (index | self.index_flags) != index {
            return Some(Conflict::Indexed);
        }
        if self.mode != InterpolationMode::Undefined && self.mode != mode {
            return Some(Conflict::Mode);
        }
        let mut free = 0;
        for f in self.flags {
            if f & OCCUPIED != 0 || f & flags != 0 {
                free = 0;
            } else {
                free += 1;
            }
            if width <= free {
                break;
            }
        }
        (width > free).then_some(Conflict::Full)
    }

    fn col_conflict(&self, flags: u8, col: u32, width: u32) -> Option<Conflict> {
        if col + width > 4 {
            return Some(Conflict::Fit);
        }
        let f = (col..col + width).map(|i| self.flags[i as usize]).find(|f| f & (flags | OCCUPIED) != 0)?;
        Some(if f & OCCUPIED != 0 { Conflict::Overlap } else { Conflict::Order })
    }

    fn place(&mut self, flags: u8, index: u8, mode: InterpolationMode, col: u32, width: u32) {
        self.mode = mode;
        self.index_flags |= index;
        if flags & (CONFLICTS_WITH_INDEXED | TESS_FACTOR) != 0 {
            self.indexing_fixed = true;
        }
        let (left, right) = (conflicts_left(flags), conflicts_right(flags));
        for i in 0..4u32 {
            let f = &mut self.flags[i as usize];
            if *f & OCCUPIED == 0 {
                if i < col {
                    *f |= left;
                } else if i < col + width {
                    *f = OCCUPIED | flags;
                } else {
                    *f |= right;
                }
            }
        }
    }
}

/// What the allocator needs of an element.
#[derive(Copy, Clone)]
struct Shape {
    flags: u8,
    mode: InterpolationMode,
    rows: u32,
    cols: u32,
}

struct Allocator {
    registers: Vec<Register>,
}

impl Allocator {
    fn new(registers: usize) -> Allocator {
        Allocator { registers: vec![Register::EMPTY; registers] }
    }

    fn row_conflict(&self, s: Shape, row: u32) -> Option<Conflict> {
        if (s.rows + row) as usize > self.registers.len() {
            return Some(Conflict::Fit);
        }
        (0..s.rows).find_map(|i| self.registers[(row + i) as usize].row_conflict(s.flags, index_flags(i, s.rows), s.mode, s.cols))
    }

    fn col_conflict(&self, s: Shape, row: u32, col: u32) -> Option<Conflict> {
        (0..s.rows).find_map(|i| self.registers[(row + i) as usize].col_conflict(s.flags, col, s.cols))
    }

    fn place(&mut self, s: Shape, row: u32, col: u32) {
        for i in 0..s.rows {
            self.registers[(row + i) as usize].place(s.flags, index_flags(i, s.rows), s.mode, col, s.cols);
        }
    }

    /// DXC's `FindNext`: the first row from `start_row` and, in it, the first column from
    /// `start_col` where the element fits.
    fn find_next(&self, s: Shape, start_row: u32, num_rows: u32, start_col: u32) -> Option<(u32, u32)> {
        if s.rows > num_rows || s.cols > 4 {
            return None;
        }
        for row in start_row..=(start_row + num_rows - s.rows) {
            if self.row_conflict(s, row).is_some() {
                continue;
            }
            for col in start_col..=(4 - s.cols) {
                if self.col_conflict(s, row, col).is_none() {
                    return Some((row, col));
                }
            }
        }
        None
    }

    fn pack_next(&mut self, s: Shape, start_row: u32, num_rows: u32, start_col: u32) -> Option<(u32, u32)> {
        let at = self.find_next(s, start_row, num_rows, start_col)?;
        self.place(s, at.0, at.1);
        Some(at)
    }
}

/// Registers of a signature (`DXIL::kMaxClipOrCullDistanceElementCount` of them may hold clip
/// and cull distances).
const REGISTERS: u32 = 32;
const CLIP_CULL_REGISTERS: u32 = 2;

/// DXC's `PackPrefixStable`: elements in declaration order, each at the first place it fits, so
/// appending an element never moves an earlier one. Clip and cull distances share at most two
/// registers that hold nothing else: the first one that needs a register reserves it whole.
fn pack_prefix_stable(elements: &mut [&mut Elem]) {
    let mut alloc = Allocator::new(REGISTERS as usize);
    let mut clip_alloc = Allocator::new(CLIP_CULL_REGISTERS as usize);
    // rows reserved in the signature for the two clip / cull registers
    let mut reserved: [Option<u32>; 2] = [None, None];
    let mut clip_used = 0u32;
    let mut clip_indexed = false;
    for se in elements.iter_mut() {
        se.at = None;
        let s = Shape { flags: element_flags(se.interpretation), mode: se.mode, rows: se.rows, cols: se.cols };
        match se.interpretation {
            Interpretation::ClipCull => {
                let Some((row, col)) = clip_alloc.find_next(s, 0, CLIP_CULL_REGISTERS, 0) else { continue };
                let used = row + s.rows;
                if s.rows > 1 && !clip_indexed {
                    // two rows reserved already must be adjacent to hold an indexed element
                    if clip_used == CLIP_CULL_REGISTERS && reserved[0].map(|r| r + 1) != reserved[1] {
                        continue;
                    }
                    clip_indexed = true;
                }
                if used > clip_used {
                    let mut placeholder = Shape { flags: s.flags, mode: s.mode, rows: 1, cols: 4 };
                    if clip_indexed {
                        if clip_used < 1 {
                            placeholder.rows = CLIP_CULL_REGISTERS;
                            let Some((r, _)) = alloc.pack_next(placeholder, 0, REGISTERS, 0) else { continue };
                            reserved = [Some(r), Some(r + 1)];
                        } else {
                            let Some(first) = reserved[0] else { continue };
                            // DXC passes "first + 2" as the row count here; kept as it is
                            let Some((r, _)) = alloc.pack_next(placeholder, first + 1, first + 2, 0) else { continue };
                            reserved[1] = Some(r);
                        }
                        clip_used = CLIP_CULL_REGISTERS;
                    } else {
                        let Some((r, _)) = alloc.pack_next(placeholder, 0, REGISTERS, 0) else { continue };
                        reserved[clip_used as usize] = Some(r);
                        clip_used = used;
                    }
                }
                clip_alloc.place(s, row, col);
                se.at = reserved[row as usize].map(|r| (r, col));
            }
            // indexed tessellation factors go to the last column
            Interpretation::TessFactor if s.rows > 1 => se.at = alloc.pack_next(s, 0, REGISTERS, 3),
            _ => se.at = alloc.pack_next(s, 0, REGISTERS, 0),
        }
    }
}

/// Assign rows and columns to the elements of one signature (DXC's `PackDxilSignature`).
/// Returns false when an element that needs a register did not get one (DXC stops with
/// "Failed to allocate all ... signature elements in available space").
pub fn pack(elements: &mut [Elem], packing: Packing, streams: bool) -> bool {
    for e in elements.iter_mut() {
        e.at = None;
    }
    match packing {
        Packing::InputAssembler => {
            let mut row = 0;
            for e in elements.iter_mut().filter(|e| e.interpretation.allocated()) {
                e.at = Some((row, 0));
                row += e.rows;
            }
        }
        Packing::Target => {
            for e in elements.iter_mut().filter(|e| e.interpretation == Interpretation::Target) {
                e.at = Some((e.index, 0));
            }
        }
        Packing::Vertex => {
            for stream in 0..if streams { 4 } else { 1 } {
                let mut list: Vec<&mut Elem> = elements.iter_mut().filter(|e| e.interpretation.allocated() && (!streams || e.stream == stream)).collect();
                pack_prefix_stable(&mut list);
            }
        }
    }
    elements.iter().all(|e| e.at.is_some() || !e.interpretation.allocated() || (packing == Packing::Target && e.interpretation != Interpretation::Target))
}

/// The validator's check of a packed signature (`ValidateSignatureOverlap` in
/// lib/DxilValidation/DxilValidation.cpp): put every element where the packing left it, in
/// declaration order, and return the first that conflicts with those before it.
/// `PackPrefixStable` can leave such a layout (an indexed clip / cull element whose second
/// reserved register is not the one after the first), and DXC then fails the compile.
pub fn conflict(elements: &[Elem], streams: bool) -> Option<(usize, Conflict)> {
    let mut allocators: Vec<Allocator> = (0..4).map(|_| Allocator::new(REGISTERS as usize)).collect();
    for (i, e) in elements.iter().enumerate() {
        let Some((row, col)) = e.at else { continue };
        if !e.interpretation.allocated() || e.interpretation == Interpretation::Target {
            continue;
        }
        let s = Shape { flags: element_flags(e.interpretation), mode: e.mode, rows: e.rows, cols: e.cols };
        let a = &mut allocators[if streams { e.stream as usize } else { 0 }];
        let found = match a.row_conflict(s, row) {
            None | Some(Conflict::Full) => a.col_conflict(s, row, col),
            c => c,
        };
        if let Some(c) = found {
            return Some((i, c));
        }
        a.place(s, row, col);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{pack, Elem, Interpretation, InterpolationMode, Packing};
    use Interpretation::*;
    use InterpolationMode::{Constant, Linear, LinearCentroid, LinearNoperspective};

    fn el(interpretation: Interpretation, mode: InterpolationMode, rows: u32, cols: u32) -> Elem {
        Elem { interpretation, mode, rows, cols, index: 0, stream: 0, at: None }
    }
    fn places(elements: &mut [Elem]) -> Vec<(u32, u32)> {
        assert!(pack(elements, Packing::Vertex, false));
        elements.iter().map(|e| e.at.unwrap()).collect()
    }

    /// The output signature of scratch snippet vs_pack as DXC 1.9 packs it: elements share a
    /// register only with the same interpolation mode, in declaration order.
    #[test]
    fn mixed_widths_and_modes() {
        let mut v = [
            el(SV, LinearNoperspective, 1, 4),
            el(Arb, Linear, 1, 2),
            el(Arb, Linear, 1, 3),
            el(Arb, Linear, 1, 1),
            el(Arb, Constant, 1, 1),
            el(Arb, Linear, 1, 2),
            el(Arb, Linear, 1, 1),
            el(Arb, LinearNoperspective, 1, 1),
            el(Arb, Constant, 1, 1),
            el(Arb, LinearCentroid, 1, 3),
            el(Arb, Linear, 1, 4),
        ];
        assert_eq!(places(&mut v), [(0, 0), (1, 0), (2, 0), (1, 2), (3, 0), (4, 0), (1, 3), (5, 0), (3, 1), (6, 0), (7, 0)]);
    }

    /// vs_clip: clip and cull distances share two reserved registers, and an ordinary element
    /// does not go into a register after a system value.
    #[test]
    fn clip_and_cull() {
        let mut v = [
            el(SV, LinearNoperspective, 1, 4),
            el(Arb, Linear, 1, 2),
            el(ClipCull, Linear, 1, 3),
            el(ClipCull, Linear, 1, 2),
            el(ClipCull, Linear, 1, 1),
            el(Arb, Linear, 1, 3),
            el(SV, Constant, 1, 1),
            el(SV, Constant, 1, 1),
            el(Arb, Linear, 1, 1),
        ];
        assert_eq!(places(&mut v), [(0, 0), (1, 0), (2, 0), (3, 0), (2, 3), (4, 0), (5, 0), (5, 1), (1, 2)]);
    }

    /// vs_arr: an indexed element (three rows) and an indexed clip distance.
    #[test]
    fn arrays() {
        let mut v = [el(SV, LinearNoperspective, 1, 4), el(Arb, Linear, 3, 2), el(Arb, Linear, 1, 1), el(ClipCull, Linear, 2, 1), el(Arb, Linear, 1, 3)];
        assert_eq!(places(&mut v), [(0, 0), (1, 0), (1, 2), (4, 0), (6, 0)]);
    }

    /// ps_sv: system-generated values after an ordinary element in its register, a system
    /// value not after them.
    #[test]
    fn generated_values() {
        let mut v = [el(SV, LinearNoperspective, 1, 4), el(Arb, Linear, 1, 4), el(Arb, Constant, 1, 1), el(SGV, Constant, 1, 1), el(SGV, Constant, 1, 1), el(Arb, Linear, 1, 2), el(SV, Constant, 1, 1), el(ClipCull, Linear, 1, 1), el(Arb, Constant, 1, 1)];
        assert_eq!(places(&mut v), [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 0), (4, 0), (5, 0), (6, 0)]);
    }

    /// ms_prim: per-primitive outputs; an ordinary element may not follow a system value.
    #[test]
    fn per_primitive() {
        let mut v = [el(SV, Constant, 1, 1), el(Arb, Constant, 1, 2), el(SV, Constant, 1, 1)];
        assert_eq!(places(&mut v), [(0, 0), (1, 0), (0, 1)]);
    }

    #[test]
    fn streams_targets_and_vertex_inputs() {
        let mut v = [el(SV, LinearNoperspective, 1, 4), el(Arb, Linear, 1, 2), el(Arb, Linear, 1, 3), el(Arb, Constant, 1, 1)];
        v[2].stream = 1;
        v[3].stream = 1;
        assert!(pack(&mut v, Packing::Vertex, true));
        assert_eq!(v.iter().map(|e| e.at.unwrap()).collect::<Vec<_>>(), [(0, 0), (1, 0), (0, 0), (1, 0)]);

        let mut t = [el(Target, InterpolationMode::Undefined, 1, 4), el(NotPacked, InterpolationMode::Undefined, 1, 1), el(Target, InterpolationMode::Undefined, 1, 2)];
        t[0].index = 2;
        t[2].index = 1;
        assert!(pack(&mut t, Packing::Target, false));
        assert_eq!(t.iter().map(|e| e.at).collect::<Vec<_>>(), [Some((2, 0)), None, Some((1, 0))]);

        let mut i = [el(Arb, InterpolationMode::Undefined, 1, 3), el(Arb, InterpolationMode::Undefined, 2, 4), el(SV, InterpolationMode::Undefined, 1, 1)];
        assert!(pack(&mut i, Packing::InputAssembler, false));
        assert_eq!(i.iter().map(|e| e.at.unwrap()).collect::<Vec<_>>(), [(0, 0), (1, 0), (3, 0)]);
    }

    #[test]
    fn overflow_is_reported() {
        let mut v: Vec<Elem> = (0..33).map(|_| el(Arb, Linear, 1, 4)).collect();
        assert!(!pack(&mut v, Packing::Vertex, false));
    }
}
