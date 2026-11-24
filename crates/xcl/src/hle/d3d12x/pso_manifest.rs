//! Every pipeline a title made, kept between runs (`<DXIL cache>\pipelines\`), so `xcl warm-up`
//! builds all of them before a run, not only the ones some run happened to bind: a pipeline the
//! warm-up never saw costs its first bind a driver compile (a retail game's big compute programs:
//! 15 – 180 s each), and binds waiting that long tripped the game's own GPU watchdog.
//!
//! A pipeline is kept as the title described it (`Record`): the description with its pointers
//! taken out, the console shaders it names (each with where the description points at it; the
//! bytes kept once, under their hash, in `shaders\`) and its input layout. Its key is a hash of
//! all that, and a key kept already is not written again. The translation is not kept: the
//! warm-up translates the shaders with the XCL it runs, so after a translator change it builds
//! every pipeline from the new translation. `XCL_PSO_MANIFEST=0`: nothing is kept.
//!
//! Each game run also notes every kept pipeline it made, once, with the run and the translation
//! era it was made in (`created.bin`; the runs counted in `runs`): a pipeline no run made in the
//! last `STALE_AFTER_RUNS` is left out of the budgeted warm-up (`xcl warm-up --full` still builds
//! it). `XCL_PSO_RUN_COUNTS=0`: this run's pipelines are noted, but the run is not counted (a
//! short check would otherwise age everything it never reached).

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex, OnceLock};

/// bytes of a `D3D12_INPUT_ELEMENT_DESC`
pub const ELEMENT: usize = 32;
const VERSION: u8 = 1;
/// runs after which a pipeline none of them made is stale
pub const STALE_AFTER_RUNS: u32 = 3;
/// bytes of a `created.bin` entry: the key, the translation era, the run
const CREATED: usize = 20;

/// How the pipeline was made (which of XCL's builds makes it again).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Compute = 0,
    Graphics = 1,
    /// a pipeline state stream with a mesh shader
    MeshStream = 2,
    /// a pipeline state stream with a vertex shader
    VertexStream = 3,
}

impl Kind {
    fn from_u8(v: u8) -> Option<Kind> {
        [Kind::Compute, Kind::Graphics, Kind::MeshStream, Kind::VertexStream].into_iter().find(|k| *k as u8 == v)
    }
}

/// One pipeline as the title described it.
#[derive(Clone, PartialEq, Debug)]
pub struct Record {
    pub kind: Kind,
    /// the description (a desc structure or a stream's bytes), its pointers 0
    pub desc: Vec<u8>,
    /// each console shader, with the offset in `desc` of the pointer to it
    pub shaders: Vec<(u32, Arc<Vec<u8>>)>,
    /// the input layout: the offset in `desc` of the pointer to its elements, and each element
    /// (its name pointer 0) with its semantic name
    pub layout: Option<(u32, Vec<(Vec<u8>, [u8; ELEMENT])>)>,
}

impl Record {
    /// What is kept of it: its shaders as their hash and length.
    fn encode(&self) -> Vec<u8> {
        let mut out = vec![VERSION, self.kind as u8];
        out.extend_from_slice(&(self.desc.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.desc);
        out.extend_from_slice(&(self.shaders.len() as u32).to_le_bytes());
        for (at, bytes) in &self.shaders {
            out.extend_from_slice(&at.to_le_bytes());
            out.extend_from_slice(&crate::gpu::shader::fnv64(bytes).to_le_bytes());
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        }
        match &self.layout {
            None => out.push(0),
            Some((at, elems)) => {
                out.push(1);
                out.extend_from_slice(&at.to_le_bytes());
                out.extend_from_slice(&(elems.len() as u32).to_le_bytes());
                for (name, e) in elems {
                    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
                    out.extend_from_slice(name);
                    out.extend_from_slice(e);
                }
            }
        }
        out
    }

    /// Its console shaders' bytes together (the warm-up builds the largest first).
    pub fn size(&self) -> usize {
        self.shaders.iter().map(|(_, s)| s.len()).sum()
    }

    /// What it is kept by.
    pub fn key(&self) -> u64 {
        key_of(&self.encode())
    }
}

/// The stage of the shader record `r` points at from `at` (for the log).
pub fn stage_at(r: &Record, at: u32) -> &'static str {
    use std::mem::offset_of;
    use windows::Win32::Graphics::Direct3D12::D3D12_GRAPHICS_PIPELINE_STATE_DESC as G;
    match r.kind {
        Kind::Compute => "CS",
        Kind::Graphics => [(offset_of!(G, VS), "VS"), (offset_of!(G, PS), "PS"), (offset_of!(G, DS), "DS"), (offset_of!(G, HS), "HS"), (offset_of!(G, GS), "GS")]
            .into_iter()
            .find(|o| o.0 as u32 == at)
            .map_or("?", |o| o.1),
        // a stream's subobject type is 8 bytes before its payload
        Kind::MeshStream | Kind::VertexStream => match (at as usize).checked_sub(8).and_then(|i| r.desc.get(i..i + 4)).map(|b| u32::from_le_bytes(b.try_into().unwrap())) {
            Some(1) => "VS",
            Some(2) => "PS",
            Some(3) => "DS",
            Some(4) => "HS",
            Some(5) => "GS",
            Some(6) => "CS",
            Some(24) => "AS",
            Some(25) => "MS",
            _ => "?",
        },
    }
}

fn key_of(encoded: &[u8]) -> u64 {
    crate::gpu::shader::fnv64(encoded) | 1
}

pub(super) fn shaders_key(hashes: impl Iterator<Item = u64>) -> u64 {
    hashes.fold(0xcbf2_9ce4_8422_2325u64, |h, hash| (h ^ hash).wrapping_mul(0x100_0000_01b3)) | 1
}

fn encoded_shaders_key(e: &[u8]) -> Option<u64> {
    if e.first() != Some(&VERSION) {
        return None;
    }
    let word = |p: usize| Some(u32::from_le_bytes(e.get(p..p + 4)?.try_into().ok()?) as usize);
    let p = 6usize.checked_add(word(2)?)?;
    let bytes = word(p)?.checked_mul(16)?;
    let n = p.checked_add(4)?;
    let hashes = e.get(n..n.checked_add(bytes)?)?;
    Some(shaders_key(hashes.chunks_exact(16).map(|s| u64::from_le_bytes(s[4..12].try_into().unwrap()))))
}

/// A kept record (None: not one this XCL reads, or a shader it names is missing).
fn decode(b: &[u8], shader: &mut impl FnMut(u64, u32) -> Option<Arc<Vec<u8>>>) -> Option<Record> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let s = b.get(at..at.checked_add(n)?)?;
        at += n;
        Some(s)
    };
    let u32_ = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap());
    let head = take(2)?;
    if head[0] != VERSION {
        return None;
    }
    let kind = Kind::from_u8(head[1])?;
    let n = u32_(take(4)?) as usize;
    let desc = take(n)?.to_vec();
    let n = u32_(take(4)?) as usize;
    let mut shaders = Vec::with_capacity(n.min(16));
    for _ in 0..n {
        let off = u32_(take(4)?);
        let hash = u64::from_le_bytes(take(8)?.try_into().unwrap());
        let len = u32_(take(4)?);
        shaders.push((off, shader(hash, len)?));
    }
    let layout = match take(1)?[0] {
        0 => None,
        _ => {
            let off = u32_(take(4)?);
            let n = u32_(take(4)?) as usize;
            let mut elems = Vec::with_capacity(n.min(64));
            for _ in 0..n {
                let len = u16::from_le_bytes(take(2)?.try_into().unwrap()) as usize;
                let name = take(len)?.to_vec();
                elems.push((name, take(ELEMENT)?.try_into().unwrap()));
            }
            Some((off, elems))
        }
    };
    (at == b.len()).then_some(Record { kind, desc, shaders, layout })
}

/// The kept entries' bytes: each a length and an encoded record (a torn last entry is left out).
fn entries(file: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while let Some(len) = file.get(at..at + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()) as usize) {
        let Some(e) = file.get(at + 4..at + 4 + len) else { break };
        out.push(e);
        at += 4 + len;
    }
    out
}

fn shader_path(dir: &Path, hash: u64, len: u32) -> PathBuf {
    dir.join("shaders").join(format!("{hash:016x}-{len:x}.bin"))
}

/// The list in folder `dir`.
pub struct Store {
    dir: PathBuf,
    /// the keys kept (in the file or written by this run)
    seen: Mutex<HashSet<u64>>,
}

/// When the game last made each kept pipeline (`Store::creations`).
pub struct Creations {
    /// runs counted
    pub runs: u32,
    /// each key's last run and translation era
    last: HashMap<u64, (u32, u64)>,
}

impl Creations {
    /// No run of the last `STALE_AFTER_RUNS` made pipeline `key` (one never noted counts as made
    /// before the first run counted).
    pub fn stale(&self, key: u64) -> bool {
        self.runs >= self.last.get(&key).map_or(0, |l| l.0) + STALE_AFTER_RUNS
    }

    /// The translation era pipeline `key` was last made in.
    pub fn era(&self, key: u64) -> Option<u64> {
        self.last.get(&key).map(|l| l.1)
    }
}

/// `xcl warm-up --full`'s record of the pipelines it went through in one translation era
/// (`Store::progress`): a pass stopped part way goes on from there.
pub struct Progress {
    file: Mutex<std::fs::File>,
    /// keys gone through by an earlier pass
    pub done: HashSet<u64>,
}

impl Progress {
    /// Pipeline `key` was gone through.
    pub fn note(&self, key: u64) {
        let _ = self.file.lock().unwrap().write_all(&key.to_le_bytes());
    }
}

/// What `Store::load` found.
pub struct Loaded {
    pub records: Vec<Record>,
    /// entries this XCL does not read or whose shaders are missing
    pub broken: usize,
    /// entries kept twice (two runs that wrote at once)
    pub repeated: usize,
}

impl Store {
    pub fn open(dir: PathBuf) -> Store {
        let file = std::fs::read(dir.join("pipelines.bin")).unwrap_or_default();
        let seen = entries(&file).into_iter().map(key_of).collect();
        Store { dir, seen: Mutex::new(seen) }
    }

    /// Pipelines kept.
    pub fn len(&self) -> usize {
        self.seen.lock().unwrap().len()
    }

    /// `r`'s key, and `r` encoded if it is new (then it is to be written: `write`).
    fn note(&self, r: &Record) -> (u64, Option<Vec<u8>>) {
        let encoded = r.encode();
        let key = key_of(&encoded);
        (key, self.seen.lock().unwrap().insert(key).then_some(encoded))
    }

    /// Keep `r` (encoded: `note`): its shaders not kept yet, then its entry.
    fn write(&self, r: &Record, encoded: &[u8]) -> std::io::Result<()> {
        std::fs::create_dir_all(self.dir.join("shaders"))?;
        for (_, bytes) in &r.shaders {
            let path = shader_path(&self.dir, crate::gpu::shader::fnv64(bytes), bytes.len() as u32);
            if !path.is_file() {
                let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
                std::fs::write(&tmp, bytes.as_slice())?;
                std::fs::rename(&tmp, &path)?;
            }
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("pipelines.bin"))?;
        let mut entry = (encoded.len() as u32).to_le_bytes().to_vec();
        entry.extend_from_slice(encoded);
        f.write_all(&entry)
    }

    /// Keep `r` now, unless it is kept already (true: written).
    pub fn add(&self, r: &Record) -> std::io::Result<bool> {
        match self.note(r).1 {
            Some(encoded) => self.write(r, &encoded).map(|_| true),
            None => Ok(false),
        }
    }

    /// Every pipeline kept, each once.
    pub fn load(&self) -> Loaded {
        Self::load_filtered(&self.dir, None, |_, _| true)
    }

    pub fn load_kind(dir: &Path, kind: Kind, mut include: impl FnMut(u64, Option<u64>) -> bool) -> Loaded {
        Self::load_filtered(dir, Some(kind), |key, e| include(key, encoded_shaders_key(e)))
    }

    fn load_filtered(dir: &Path, kind: Option<Kind>, mut include: impl FnMut(u64, &[u8]) -> bool) -> Loaded {
        let file = std::fs::read(dir.join("pipelines.bin")).unwrap_or_default();
        let mut shaders: HashMap<(u64, u32), Option<Arc<Vec<u8>>>> = HashMap::new();
        let mut read = |hash: u64, len: u32| {
            shaders
                .entry((hash, len))
                .or_insert_with(|| std::fs::read(shader_path(dir, hash, len)).ok().filter(|b| b.len() == len as usize && crate::gpu::shader::fnv64(b) == hash).map(Arc::new))
                .clone()
        };
        let (mut seen, mut out) = (HashSet::new(), Loaded { records: Vec::new(), broken: 0, repeated: 0 });
        for e in entries(&file) {
            if kind.is_some_and(|kind| e.get(1) != Some(&(kind as u8))) {
                continue;
            }
            let key = key_of(e);
            if !seen.insert(key) {
                out.repeated += 1;
            } else if include(key, e) {
                if let Some(r) = decode(e, &mut read) {
                    out.records.push(r);
                } else {
                    out.broken += 1;
                }
            }
        }
        out
    }

    /// The file again with only `records`, each once (broken and repeated entries left out).
    pub fn rewrite(&self, records: &[Record]) -> std::io::Result<()> {
        let mut file = Vec::new();
        for r in records {
            let encoded = r.encode();
            file.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            file.extend_from_slice(&encoded);
        }
        let path = self.dir.join("pipelines.bin");
        let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&tmp, file)?;
        std::fs::rename(&tmp, &path)?;
        *self.seen.lock().unwrap() = records.iter().map(|r| key_of(&r.encode())).collect();
        Ok(())
    }

    fn runs(&self) -> u32 {
        Self::runs_in(&self.dir)
    }

    fn runs_in(dir: &Path) -> u32 {
        std::fs::read_to_string(dir.join("runs")).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0)
    }

    pub(super) fn recently_created(dir: &Path, count: std::num::NonZeroU32) -> std::io::Result<(std::ops::RangeInclusive<u32>, HashSet<u64>)> {
        let last = Self::runs_in(dir);
        let runs = last.saturating_sub(count.get() - 1).max(1)..=last;
        let file = std::fs::read(dir.join("created.bin"))?;
        let keys = file.chunks_exact(CREATED)
            .filter(|e| runs.contains(&u32::from_le_bytes(e[16..].try_into().unwrap())))
            .map(|e| u64::from_le_bytes(e[..8].try_into().unwrap()))
            .collect();
        Ok((runs, keys))
    }

    /// A game run begins: the number its pipelines are noted with (`made`), a new one when it
    /// `counts`, else the last counted.
    pub fn begin_run(&self, counts: bool) -> std::io::Result<u32> {
        let run = self.runs() + counts as u32;
        if counts {
            std::fs::create_dir_all(&self.dir)?;
            let path = self.dir.join("runs");
            let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
            std::fs::write(&tmp, run.to_string())?;
            std::fs::rename(&tmp, &path)?;
        }
        Ok(run)
    }

    /// Run `run` made pipeline `key` in translation era `era` (`file`: `created.bin`, opened by
    /// the first).
    pub fn made(&self, file: &mut Option<std::fs::File>, key: u64, era: u64, run: u32) -> std::io::Result<()> {
        if file.is_none() {
            std::fs::create_dir_all(&self.dir)?;
            *file = Some(std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("created.bin"))?);
        }
        file.as_mut().unwrap().write_all(&created_entry(key, era, run))
    }

    /// When the game last made each pipeline (a torn last entry is left out).
    pub fn creations(&self) -> Creations {
        let file = std::fs::read(self.dir.join("created.bin")).unwrap_or_default();
        let mut last: HashMap<u64, (u32, u64)> = HashMap::new();
        for e in file.chunks_exact(CREATED) {
            let key = u64::from_le_bytes(e[..8].try_into().unwrap());
            let era = u64::from_le_bytes(e[8..16].try_into().unwrap());
            let run = u32::from_le_bytes(e[16..].try_into().unwrap());
            let l = last.entry(key).or_insert((run, era));
            if run >= l.0 {
                *l = (run, era);
            }
        }
        Creations { runs: self.runs(), last }
    }

    /// `created.bin` again with only the last of each of `keys` (the pipelines kept).
    pub fn compact_creations(&self, c: &Creations, keys: &HashSet<u64>) -> std::io::Result<()> {
        let mut file = Vec::new();
        for (key, (run, era)) in &c.last {
            if keys.contains(key) {
                file.extend_from_slice(&created_entry(*key, *era, *run));
            }
        }
        let path = self.dir.join("created.bin");
        let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&tmp, file)?;
        std::fs::rename(&tmp, &path)
    }

    /// `xcl warm-up --full`'s progress in translation era `era`: what a pass of this era went
    /// through already; a record of another era is started again.
    pub fn progress(&self, era: u64) -> std::io::Result<Progress> {
        std::fs::create_dir_all(&self.dir)?;
        let path = self.dir.join("full-progress.bin");
        let mut bytes = std::fs::read(&path).unwrap_or_default();
        if bytes.get(..8) != Some(&era.to_le_bytes()[..]) {
            bytes = era.to_le_bytes().to_vec();
        }
        bytes.truncate(8 + (bytes.len() - 8) / 8 * 8);
        std::fs::write(&path, &bytes)?;
        let done = bytes[8..].chunks_exact(8).map(|k| u64::from_le_bytes(k.try_into().unwrap())).collect();
        let file = std::fs::OpenOptions::new().append(true).open(&path)?;
        Ok(Progress { file: Mutex::new(file), done })
    }

    /// A `--full` pass went through everything: the next starts from the beginning.
    pub fn progress_complete(&self) {
        let _ = std::fs::remove_file(self.dir.join("full-progress.bin"));
    }
}

fn created_entry(key: u64, era: u64, run: u32) -> [u8; CREATED] {
    let mut e = [0u8; CREATED];
    e[..8].copy_from_slice(&key.to_le_bytes());
    e[8..16].copy_from_slice(&era.to_le_bytes());
    e[16..].copy_from_slice(&run.to_le_bytes());
    e
}

/// The folder of the list: beside the DXIL cache, not in a translator's folder of it.
pub fn dir() -> PathBuf {
    crate::gpu::recomp::dxc::cache_dir().join("pipelines")
}

enum ToWrite {
    /// a pipeline to keep, encoded
    Keep(Record, Vec<u8>),
    /// the key of a pipeline this run made for the first time
    Made(u64),
}

struct Global {
    store: Store,
    tx: Mutex<mpsc::Sender<ToWrite>>,
    /// the keys this run made
    made: Mutex<HashSet<u64>>,
}

/// This process's list and the thread that writes to it (None: `XCL_PSO_MANIFEST=0`, or a test).
fn global() -> Option<&'static Global> {
    static G: OnceLock<Option<Global>> = OnceLock::new();
    G.get_or_init(|| {
        if cfg!(test) || std::env::var("XCL_PSO_MANIFEST").is_ok_and(|v| v.trim() == "0") {
            return None;
        }
        let store = Store::open(dir());
        let counts = !std::env::var("XCL_PSO_RUN_COUNTS").is_ok_and(|v| v.trim() == "0");
        let run = store.begin_run(counts).unwrap_or(0);
        let era = crate::gpu::recomp::pso::translation_era();
        crate::xlog_info!(
            "D3D12.X: {} pipelines earlier runs made are kept for the warm-up; new ones are added (run {run}{}, translation {era:016x})",
            store.len(),
            if counts { "" } else { ", not counted" }
        );
        let (tx, rx) = mpsc::channel::<ToWrite>();
        let _ = std::thread::Builder::new().name("xcl-pso-manifest".into()).spawn(move || {
            let Some(g) = global() else { return };
            let (mut failed, mut created) = (false, None);
            for w in rx {
                let r = match w {
                    ToWrite::Keep(r, encoded) => g.store.write(&r, &encoded),
                    ToWrite::Made(key) => g.store.made(&mut created, key, era, run),
                };
                if let Err(e) = r {
                    if !std::mem::replace(&mut failed, true) {
                        crate::xlog_warn!("D3D12.X: a pipeline could not be kept for the warm-up ({e})");
                    }
                }
            }
        });
        Some(Global { store, tx: Mutex::new(tx), made: Mutex::new(HashSet::new()) })
    })
    .as_ref()
}

/// The title made pipeline `r`: kept for the warm-up unless it is already, and noted as made by
/// this run (written on a thread of its own; the title's thread only hashes it).
pub fn record(r: Record) {
    let Some(g) = global() else { return };
    let (key, encoded) = g.store.note(&r);
    let made = g.made.lock().unwrap().insert(key);
    let tx = g.tx.lock().unwrap();
    if let Some(encoded) = encoded {
        let _ = tx.send(ToWrite::Keep(r, encoded));
    }
    if made {
        let _ = tx.send(ToWrite::Made(key));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xcl-pso-manifest-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn sample(kind: Kind, shader: &[u8], name: &[u8]) -> Record {
        let mut e = [0u8; ELEMENT];
        e[8] = 3;
        Record {
            kind,
            desc: (0u8..64).collect(),
            shaders: vec![(8, Arc::new(shader.to_vec())), (24, Arc::new(b"shared shader".to_vec()))],
            layout: Some((40, vec![(name.to_vec(), e)])),
        }
    }

    /// Pipelines are kept once each, with their shaders, and read back as they were made.
    #[test]
    fn pipelines_are_kept_once_and_read_back_whole() {
        let dir = temp_dir("roundtrip");
        let (a, b) = (sample(Kind::Graphics, b"vertex shader a", b"POSITION"), sample(Kind::VertexStream, b"vertex shader b", b"TEXCOORD"));
        let compute = Record { kind: Kind::Compute, desc: vec![7; 40], shaders: vec![(8, Arc::new(vec![1, 2, 3]))], layout: None };
        let store = Store::open(dir.clone());
        assert!(store.add(&a).unwrap());
        assert!(store.add(&b).unwrap());
        assert!(!store.add(&a.clone()).unwrap(), "kept already");
        assert!(store.add(&compute).unwrap());
        let again = Store::open(dir.clone());
        assert_eq!(again.len(), 3);
        assert!(!again.add(&b).unwrap(), "kept by an earlier run");
        let loaded = again.load();
        assert_eq!(loaded.records, vec![a.clone(), b, compute]);
        assert_eq!((loaded.broken, loaded.repeated), (0, 0));
        assert_eq!(std::fs::read_dir(dir.join("shaders")).unwrap().count(), 4, "a shader two pipelines name is kept once");
        assert_eq!(a.size(), b"vertex shader a".len() + b"shared shader".len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn kind_and_key_filters_run_before_shader_reads() {
        let dir = temp_dir("kind-filter");
        let store = Store::open(dir.clone());
        let cold = sample(Kind::MeshStream, b"cold mesh", b"POSITION");
        let kept = sample(Kind::MeshStream, b"missing mesh", b"POSITION");
        let other = sample(Kind::Graphics, b"missing vertex", b"TEXCOORD");
        for r in [&cold, &kept, &other] {
            assert!(store.add(r).unwrap());
        }
        for bytes in [b"missing mesh".as_slice(), b"missing vertex".as_slice()] {
            std::fs::remove_file(shader_path(&dir, crate::gpu::shader::fnv64(bytes), bytes.len() as u32)).unwrap();
        }
        let mut inspected = Vec::new();
        let loaded = Store::load_kind(&dir, Kind::MeshStream, |key, shaders| {
            let r = [&cold, &kept].into_iter().find(|r| r.key() == key).unwrap();
            assert_eq!(shaders, Some(shaders_key(r.shaders.iter().map(|(_, s)| crate::gpu::shader::fnv64(s)))));
            inspected.push(key);
            key != kept.key()
        });
        assert_eq!(inspected, [cold.key(), kept.key()]);
        assert_eq!(loaded.records, [cold]);
        assert_eq!(loaded.broken, 0);
        assert_eq!(store.load().broken, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recent_creations_include_only_counted_runs_in_range() {
        let dir = temp_dir("recent-creations");
        let store = Store::open(dir.clone());
        let mut file = None;
        for key in [1, 2, 3] {
            let run = store.begin_run(true).unwrap();
            store.made(&mut file, key, 0x1234, run).unwrap();
        }
        drop(file);
        let (runs, keys) = Store::recently_created(&dir, std::num::NonZeroU32::new(2).unwrap()).unwrap();
        assert_eq!(runs, 2..=3);
        assert_eq!(keys, HashSet::from([2, 3]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A torn last entry, an entry kept twice and one whose shader is gone are left out, and a
    /// rewrite keeps only what loads.
    #[test]
    fn torn_repeated_and_broken_entries_are_left_out() {
        let dir = temp_dir("damage");
        let (a, b) = (sample(Kind::Graphics, b"shader a", b"A"), sample(Kind::Graphics, b"shader b", b"B"));
        let store = Store::open(dir.clone());
        store.add(&a).unwrap();
        store.add(&b).unwrap();
        let file = dir.join("pipelines.bin");
        let mut bytes = std::fs::read(&file).unwrap();
        let first = entries(&bytes)[0].len() + 4;
        let copy = bytes[..first].to_vec();
        bytes.extend_from_slice(&copy);
        bytes.extend_from_slice(&[9, 0, 0, 0, 1, 2]);
        std::fs::write(&file, &bytes).unwrap();
        std::fs::remove_file(shader_path(&dir, crate::gpu::shader::fnv64(b"shader b"), 8)).unwrap();
        let store = Store::open(dir.clone());
        let loaded = store.load();
        assert_eq!(loaded.records, vec![a.clone()]);
        assert_eq!((loaded.broken, loaded.repeated), (1, 1));
        store.rewrite(&loaded.records).unwrap();
        let loaded = Store::open(dir.clone()).load();
        assert_eq!((loaded.records, loaded.broken, loaded.repeated), (vec![a], 0, 0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A pipeline no counted run of the last `STALE_AFTER_RUNS` made is stale; a run not counted
    /// keeps what it made fresh without aging the rest, and the era of the last making is kept.
    #[test]
    fn a_pipeline_no_recent_run_made_is_stale() {
        let dir = temp_dir("stale");
        let store = Store::open(dir.clone());
        let (a, b, never) = (sample(Kind::Graphics, b"a", b"A").key(), sample(Kind::Graphics, b"b", b"B").key(), 7u64);
        let mut file = None;
        let run = store.begin_run(true).unwrap();
        assert_eq!(run, 1);
        store.made(&mut file, a, 10, run).unwrap();
        store.made(&mut file, b, 10, run).unwrap();
        let c = store.creations();
        assert!(!c.stale(a) && !c.stale(never), "noted before the runs began to count");
        for _ in 0..2 {
            let run = store.begin_run(true).unwrap();
            store.made(&mut file, b, 11, run).unwrap();
        }
        let c = store.creations();
        assert_eq!(c.runs, 3);
        assert!(!c.stale(a), "made in the oldest of the last three");
        assert!(c.stale(never), "never made in three runs");
        assert_eq!((c.era(a), c.era(b), c.era(never)), (Some(10), Some(11), None));
        let check = store.begin_run(false).unwrap();
        assert_eq!(check, 3, "a run not counted notes with the last counted");
        store.made(&mut file, b, 12, check).unwrap();
        let run = store.begin_run(true).unwrap();
        store.made(&mut file, b, 12, run).unwrap();
        let c = store.creations();
        assert_eq!(c.runs, 4);
        assert!(c.stale(a) && !c.stale(b));
        assert_eq!(c.era(b), Some(12));
        drop(file);
        std::fs::OpenOptions::new().append(true).open(dir.join("created.bin")).unwrap().write_all(&[1, 2, 3]).unwrap();
        store.compact_creations(&store.creations(), &[a].into_iter().collect()).unwrap();
        assert_eq!(std::fs::metadata(dir.join("created.bin")).unwrap().len(), CREATED as u64, "the last of each kept key, a torn entry gone");
        let c = store.creations();
        assert!(c.stale(a) && c.stale(b), "b is no longer kept");
        assert_eq!(c.era(a), Some(10));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A `--full` pass stopped part way goes on where it left off in the same translation era, and
    /// starts again in another or once it went through everything.
    #[test]
    fn a_full_pass_goes_on_where_it_stopped_in_its_era() {
        let dir = temp_dir("progress");
        let store = Store::open(dir.clone());
        let p = store.progress(5).unwrap();
        assert!(p.done.is_empty());
        p.note(1);
        p.note(2);
        drop(p);
        std::fs::OpenOptions::new().append(true).open(dir.join("full-progress.bin")).unwrap().write_all(&[9, 9, 9]).unwrap();
        let p = store.progress(5).unwrap();
        assert_eq!(p.done, [1, 2].into_iter().collect());
        p.note(3);
        drop(p);
        assert_eq!(store.progress(5).unwrap().done, [1, 2, 3].into_iter().collect(), "a torn key is dropped, not misread");
        assert!(store.progress(6).unwrap().done.is_empty(), "another era starts again");
        assert!(store.progress(5).unwrap().done.is_empty(), "the record of era 5 was replaced");
        store.progress(5).unwrap().note(4);
        store.progress_complete();
        assert!(store.progress(5).unwrap().done.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
