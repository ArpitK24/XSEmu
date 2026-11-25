use super::{rdna2, recomp, xbox};

const BIG_CODE: usize = 16 << 10;

pub(crate) fn big_compute_writes_only_textures(blob: &[u8]) -> bool {
    if blob.len() < BIG_CODE || recomp::loop_profile_enabled() {
        return false;
    }
    if let Some(f) = recomp::pso::fspx(blob) {
        return f.code.len() >= BIG_CODE && writes_only_textures(f.code);
    }
    let Some(shader) = xbox::parse(blob) else { return false };
    let Some(f) = shader.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == 6)) else { return false };
    f.programs.first().and_then(|p| f.data.get(p.offset..p.offset.checked_add(p.len)?))
        .is_some_and(|code| code.len() >= BIG_CODE && writes_only_textures(code))
}

fn writes_only_textures(mut code: &[u8]) -> bool {
    let (mut pc, mut texture) = (0, false);
    while !code.is_empty() {
        let Ok(d) = rdna2::decode(code, pc) else { return false };
        let name = d.inst.name;
        match d.enc.name {
            "ENC_MIMG" if name == "IMAGE_STORE" || name.starts_with("IMAGE_ATOMIC_") => texture = true,
            "ENC_MIMG" if name.starts_with("IMAGE_STORE") => return false,
            "ENC_MUBUF" | "ENC_MTBUF" | "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH" | "ENC_SMEM"
                if !(name.contains("_LOAD") || matches!(name, "BUFFER_GL0_INV" | "BUFFER_GL1_INV" | "S_DCACHE_INV" | "S_DCACHE_WB" | "S_MEMTIME" | "S_MEMREALTIME")) => return false,
            "ENC_DS" if rdna2::field_value(d.enc, &d.words, "GDS") != Some(0)
                || matches!(name, "DS_APPEND" | "DS_CONSUME" | "DS_ORDERED_COUNT") || name.starts_with("DS_GWS_") => return false,
            _ => {}
        }
        pc += d.len as u64;
        code = &code[d.len..];
    }
    texture
}

#[cfg(test)]
mod tests {
    use super::*;

    type Inst = (&'static str, &'static str, &'static [(&'static str, u64)]);
    const END: Inst = ("S_ENDPGM", "ENC_SOPP", &[]);
    const IMAGE: Inst = ("IMAGE_STORE", "ENC_MIMG", &[("DMASK", 15)]);
    const IMAGE_ATOMIC: Inst = ("IMAGE_ATOMIC_ADD", "ENC_MIMG", &[("DMASK", 1)]);
    const LOAD: Inst = ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[]);
    const SCALAR_LOAD: Inst = ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[]);
    const LDS: Inst = ("DS_WRITE_B32", "ENC_DS", &[]);

    fn asm(insts: &[Inst]) -> Vec<u8> {
        insts.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap_or_else(|err| panic!("{n}: {err}"))).collect()
    }

    #[test]
    fn texture_uavs_and_group_local_writes_do_not_write_guest_buffers() {
        assert!(writes_only_textures(&asm(&[LOAD, SCALAR_LOAD, LDS, IMAGE, END])));
        assert!(writes_only_textures(&asm(&[IMAGE_ATOMIC, END])));
        assert!(!writes_only_textures(&asm(&[LOAD, LDS, END])));
    }

    #[test]
    fn all_memory_writes_and_unknown_code_keep_the_wait() {
        let memory: &[Inst] = &[
            ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[]),
            ("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[]),
            ("TBUFFER_STORE_FORMAT_XYZW", "ENC_MTBUF", &[]),
            ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[]),
            ("FLAT_ATOMIC_ADD", "ENC_FLAT", &[]),
            ("SCRATCH_STORE_DWORD", "ENC_FLAT_SCRATCH", &[]),
            ("DS_ADD_U32", "ENC_DS", &[("GDS", 1)]),
            ("DS_APPEND", "ENC_DS", &[]),
        ];
        for &write in memory {
            assert!(!writes_only_textures(&asm(&[IMAGE, END, write, END])), "{}", write.0);
        }
        let mut unknown = asm(&[IMAGE, END]);
        unknown.push(0xff);
        assert!(!writes_only_textures(&unknown));
    }
}
