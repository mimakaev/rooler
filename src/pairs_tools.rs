//! Pairs tools: text .pairs -> sorted pairs file (`sort`), full decode with order check and size
//! breakdown (`scan`), and back to text (`view`). Files are Parquet (see `pairs_pq`).
//!
//! How `sort` works:
//!   phase 1  N threads parse text into fixed 24-byte records, each fills its share of the RAM
//!            budget, sorts it, and spills it as a run: a pairs Parquet file (internal variant)
//!            with small row groups, whose first keys (chrom1,pos1) are kept in memory.
//!   phase 2  the kept first keys give exact quantiles of the whole data set, so the key space is
//!            cut into units of bounded size. Each unit is built on its own thread: read the row
//!            groups of every run that overlap it, trim to the unit's key range, merge in memory,
//!            encode output row groups. Units are appended to the output in order.
//! There is no k-way streaming merge and no single merging thread; RAM is bounded by
//! threads x unit size.
use crate::pairs::{Header, Pair, HAS_MAPQ, HAS_PTYPE, SORTED_CANONICAL};
use crate::cload::parse_pos;
use crate::pairs_pq::{PqCodec, PqCtx, PqFile, PqWriter};
use anyhow::{anyhow, bail, Result};
use crossbeam_channel::{bounded, Receiver};
use memchr::{memchr, memrchr};
use parquet::arrow::arrow_writer::ArrowColumnChunk;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Pairs per row group in a spill run. Small, because a unit of phase 2 reads whole
/// blocks and the two blocks straddling its key range are decoded for nothing but their edge.
const RUN_BLOCK: usize = 1 << 16;
/// Pairs per row group of a Parquet output.
const OUT_ROW_GROUP: usize = 1 << 20;

// ---------------------------------------------------------------------------------------------
// text input

/// ".gz" -> parallel bgzip decode; "-" -> stdin; anything else -> plain file.
fn open_text(path: &str, threads: usize) -> Result<(Box<dyn Read>, Option<Child>)> {
    if path.ends_with(".gz") {
        let mut c = Command::new("bgzip").args(["-dc", "-@", &threads.to_string(), path])
            .stdout(Stdio::piped()).spawn()?;
        let out = c.stdout.take().ok_or_else(|| anyhow!("no bgzip stdout"))?;
        Ok((Box::new(out), Some(c)))
    } else if path == "-" {
        Ok((Box::new(std::io::stdin()), None))
    } else {
        Ok((Box::new(std::fs::File::open(path)?), None))
    }
}

/// Where each field we keep sits in a text line (tab-separated field index).
#[derive(Clone, Debug)]
struct TextCols { c1: usize, p1: usize, c2: usize, p2: usize, s1: usize, s2: usize,
                  ptype: Option<usize>, mapq1: Option<usize>, mapq2: Option<usize>, max: usize }

struct TextHeader { chroms: Vec<(String, u64)>, assembly: String, cols: TextCols }

/// Consume the `#` header of a text .pairs stream; returns it and the body bytes already read.
fn read_text_header(rd: &mut dyn Read, what: &str) -> Result<(TextHeader, Vec<u8>)> {
    let mut chroms = Vec::new();
    let mut assembly = String::new();
    let mut columns: Option<Vec<String>> = None;
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; 1 << 20];
    let mut start = 0usize;
    let tail = 'hdr: loop {
        while let Some(rel) = memchr(b'\n', &buf[start..]) {
            let line = &buf[start..start + rel];
            if line.first() != Some(&b'#') { break 'hdr buf[start..].to_vec(); }
            let l = String::from_utf8_lossy(line);
            if let Some(r) = l.strip_prefix("#chromsize:") {
                let mut it = r.split_whitespace();
                match (it.next(), it.next().and_then(|x| x.parse::<u64>().ok())) {
                    (Some(n), Some(len)) => chroms.push((n.to_string(), len)),
                    _ => bail!("bad #chromsize header line: {:?}", l),
                }
            } else if let Some(r) = l.strip_prefix("#genome_assembly:") {
                assembly = r.trim().to_string();
            } else if let Some(r) = l.strip_prefix("#columns:") {
                columns = Some(r.split_whitespace().map(|s| s.to_string()).collect());
            }
            start += rel + 1;
        }
        buf.drain(..start);
        start = 0;
        let n = rd.read(&mut chunk)?;
        if n == 0 {
            // EOF inside the header; an unterminated last line may still be a body line
            break if buf.first().is_some_and(|&c| c != b'#') { buf } else { Vec::new() };
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    if chroms.is_empty() {
        bail!("{}: no `#chromsize:` header lines found (empty or unreadable input, or a .pairs \
               file without the 4DN header)", what);
    }
    // no #columns line -> the 4DN minimum: readID chr1 pos1 chr2 pos2 strand1 strand2
    let find = |names: &[&str]| columns.as_ref().and_then(|c| c.iter().position(|x| names.contains(&x.as_str())));
    let need = |names: &[&str], dflt: usize| -> Result<usize> {
        match &columns {
            None => Ok(dflt),
            Some(_) => find(names).ok_or_else(|| anyhow!("{}: #columns has no {} column", what, names[0])),
        }
    };
    let mut cols = TextCols {
        c1: need(&["chr1", "chrom1"], 1)?, p1: need(&["pos1"], 2)?,
        c2: need(&["chr2", "chrom2"], 3)?, p2: need(&["pos2"], 4)?,
        s1: need(&["strand1"], 5)?, s2: need(&["strand2"], 6)?,
        ptype: find(&["pair_type"]), mapq1: find(&["mapq1"]), mapq2: find(&["mapq2"]), max: 0,
    };
    if cols.mapq1.is_none() || cols.mapq2.is_none() { cols.mapq1 = None; cols.mapq2 = None; }
    cols.max = [cols.c1, cols.p1, cols.c2, cols.p2, cols.s1, cols.s2].into_iter()
        .chain(cols.ptype).chain(cols.mapq1).chain(cols.mapq2).max().unwrap();
    Ok((TextHeader { chroms, assembly, cols }, tail))
}

struct ParseCtx { cmap: HashMap<Vec<u8>, u32>, lengths: Vec<u64>, cols: TextCols }

/// Last chromosome name -> id lookups (pairs files are chromosome-clustered).
#[derive(Default)]
struct ChromCache { n1: Vec<u8>, i1: u32, n2: Vec<u8>, i2: u32 }

#[inline]
fn chrom_id(name: &[u8], cached: &mut Vec<u8>, id: &mut u32, cx: &ParseCtx) -> Result<u32> {
    if !name.is_empty() && name == &cached[..] { return Ok(*id); }
    let i = *cx.cmap.get(name).ok_or_else(|| anyhow!(
        "pairs line names chromosome {:?}, which is not in the #chromsize header", String::from_utf8_lossy(name)))?;
    cached.clear(); cached.extend_from_slice(name); *id = i;
    Ok(i)
}

#[inline]
fn strand_minus(f: &[u8]) -> Result<bool> {
    match f { b"+" => Ok(false), b"-" => Ok(true), _ => bail!("bad strand {:?}", String::from_utf8_lossy(f)) }
}

/// One text body line -> a flipped Pair. Ok(None) for blank/comment lines.
fn parse_text_line(line: &[u8], cx: &ParseCtx, cache: &mut ChromCache) -> Result<Option<Pair>> {
    if line.is_empty() || line[0] == b'#' { return Ok(None); }
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    let mut f: [&[u8]; 32] = [&[]; 32];
    let (mut nf, mut start) = (0usize, 0usize);
    let want = cx.cols.max + 1;
    if want > f.len() { bail!("pairs columns beyond field {} are not supported", f.len()); }
    while nf < want {
        match memchr(b'\t', &line[start..]) {
            Some(rel) => { f[nf] = &line[start..start + rel]; start += rel + 1; nf += 1; }
            None => { f[nf] = &line[start..]; nf += 1; break; }
        }
    }
    if nf < want { bail!("pairs line has {} fields, expected at least {}: {:?}", nf, want, String::from_utf8_lossy(line)); }
    let c = &cx.cols;
    let c1 = chrom_id(f[c.c1], &mut cache.n1, &mut cache.i1, cx)?;
    let c2 = chrom_id(f[c.c2], &mut cache.n2, &mut cache.i2, cx)?;
    let pos = |b: &[u8], chrom: u32| -> Result<u32> {
        let p = parse_pos(b).ok_or_else(|| anyhow!("bad position {:?} in pairs line", String::from_utf8_lossy(b)))?;
        // a position past the chromosome end would bin into the next chromosome downstream
        if p as u64 > cx.lengths[chrom as usize] || p > u32::MAX as i64 {
            bail!("pairs line has position {} beyond chromosome id {} (or beyond 2^32)", p, chrom);
        }
        Ok(p as u32)
    };
    let (p1, p2) = (pos(f[c.p1], c1)?, pos(f[c.p2], c2)?);
    let ptype = match c.ptype {
        Some(i) if f[i].len() == 2 => [f[i][0], f[i][1]],
        Some(i) => bail!("pair_type {:?} is not two characters", String::from_utf8_lossy(f[i])),
        None => [0, 0],
    };
    let mapq = |i: Option<usize>| -> Result<u8> {
        match i {
            None => Ok(0),
            Some(i) => parse_pos(f[i]).filter(|&v| v <= 255).map(|v| v as u8)
                .ok_or_else(|| anyhow!("bad mapq {:?}", String::from_utf8_lossy(f[i]))),
        }
    };
    Ok(Some(Pair::flipped(c1, p1, c2, p2, strand_minus(f[c.s1])?, strand_minus(f[c.s2])?,
                          ptype, mapq(c.mapq1)?, mapq(c.mapq2)?)))
}

// ---------------------------------------------------------------------------------------------
// spill runs

/// A sorted spill run plus what phase 2 needs to address it: the first (chrom1,pos1) key and row
/// count of every row group. Kept in memory — we wrote the run, so nothing has to be re-read.
struct Run {
    path: String,
    first: Vec<u64>,
    rows: Vec<u32>,
    file: PqFile,
}

struct RunSink { ctx: PqCtx, tmpdir: String }

impl RunSink {
    /// Sort `buf` and spill it. Returns the run and the seconds spent sorting / writing.
    fn spill(&self, buf: &mut Vec<Pair>, wid: usize, seq: usize) -> Result<(Run, f64, f64)> {
        let t0 = Instant::now();
        buf.sort_unstable();
        let ts = t0.elapsed().as_secs_f64();
        let path = format!("{}/r{}_{}.parquet", self.tmpdir, wid, seq);
        let (mut first, mut rows) = (Vec::new(), Vec::new());
        let mut w = PqWriter::create(&path, &self.ctx)?;
        for ch in buf.chunks(RUN_BLOCK) {
            first.push(ch[0].k1);
            rows.push(ch.len() as u32);
            w.append(self.ctx.encode_row_group(ch)?, ch[0].k1)?;
        }
        w.close(None)?;
        buf.clear();
        let file = PqFile::open(&path)?;
        Ok((Run { path, first, rows, file }, ts, t0.elapsed().as_secs_f64() - ts))
    }
}

/// Turns one block of input text into pairs (appended to the vector). One per worker thread.
pub type BlockParser = Box<dyn FnMut(&[u8], &mut Vec<Pair>) -> Result<()> + Send>;

/// (runs, pairs, [parse, sort, write] thread-seconds)
type WorkerOut = (Vec<Run>, u64, [f64; 3]);

fn sort_worker(rx: Receiver<Vec<u8>>, mut parse: BlockParser, sink: Arc<RunSink>, cap: usize, wid: usize,
               failed: Arc<AtomicBool>) -> Result<WorkerOut> {
    let r = (|| {
        let mut buf: Vec<Pair> = Vec::with_capacity(cap);
        let mut tmp: Vec<Pair> = Vec::new();
        let (mut runs, mut np, mut t) = (Vec::new(), 0u64, [0f64; 3]);
        for blk in rx.iter() {
            let t0 = Instant::now();
            let mut spilled = 0f64;
            tmp.clear();
            parse(&blk, &mut tmp)?;
            np += tmp.len() as u64;
            let mut rest = &tmp[..];
            while !rest.is_empty() {
                let take = rest.len().min(cap - buf.len());
                buf.extend_from_slice(&rest[..take]);
                rest = &rest[take..];
                if buf.len() >= cap {
                    let (run, ts, tw) = sink.spill(&mut buf, wid, runs.len())?;
                    runs.push(run); t[1] += ts; t[2] += tw; spilled += ts + tw;
                }
            }
            t[0] += t0.elapsed().as_secs_f64() - spilled;
        }
        if !buf.is_empty() {
            let (run, ts, tw) = sink.spill(&mut buf, wid, runs.len())?;
            runs.push(run); t[1] += ts; t[2] += tw;
        }
        Ok((runs, np, t))
    })();
    if r.is_err() { failed.store(true, Ordering::Relaxed); }
    r
}

// ---------------------------------------------------------------------------------------------
// sort

pub struct SortOpts {
    pub mem_gb: f64,
    pub threads: usize,
    pub tmpdir: String,
    /// codec of the spill runs
    pub run_codec: PqCodec,
    /// codec of the output
    pub out_codec: PqCodec,
}

/// Pairs of every run with lo <= k1 < hi, appended to `out` (unsorted across runs).
fn load_unit(runs: &[Run], lo: u64, hi: u64, out: &mut Vec<Pair>, scratch: &mut Vec<Pair>) -> Result<()> {
    for r in runs {
        // row group b holds keys in [first[b], first[b+1]]: the one before the first starting
        // at or after `lo` may still reach into the range
        let b0 = r.first.partition_point(|&f| f < lo).saturating_sub(1);
        let b1 = r.first.partition_point(|&f| f < hi);
        for b in b0..b1 {
            scratch.clear();
            r.file.read_row_group(b, scratch)?;
            let a = scratch.partition_point(|p| p.k1 < lo);
            let e = scratch.partition_point(|p| p.k1 < hi);
            out.extend_from_slice(&scratch[a..e]);
        }
    }
    Ok(())
}

/// Block boundary for line-oriented text: just after the last newline.
fn cut_lines(buf: &[u8]) -> Option<usize> { memrchr(b'\n', buf).map(|p| p + 1) }

/// The input side of a sort: a byte stream, how to cut it into self-contained blocks, and a
/// parser per worker.
pub struct SortInput<'a> {
    pub rd: Box<dyn Read>,
    /// body bytes already consumed while reading the header
    pub body_tail: Vec<u8>,
    pub child: Option<Child>,
    pub name: String,
    pub header: Header,
    /// end (exclusive) of the last self-contained block in the buffer, if any
    pub cut: fn(&[u8]) -> Option<usize>,
    pub make_parser: &'a dyn Fn() -> BlockParser,
}

/// Phase 1 of a sort: parse on N threads into sorted spill runs.
fn make_runs(mut inp: SortInput, o: &SortOpts, log: bool) -> Result<(Vec<Run>, u64)> {
    let t0 = Instant::now();
    std::fs::create_dir_all(&o.tmpdir)?;
    let sink = Arc::new(RunSink { ctx: PqCtx::new(&inp.header, o.run_codec, true)?, tmpdir: o.tmpdir.clone() });
    let mem_pairs = o.mem_gb * 1e9 / std::mem::size_of::<Pair>() as f64;
    let cap = ((mem_pairs / o.threads as f64) as usize).max(RUN_BLOCK);
    let (tx, rx) = bounded::<Vec<u8>>(o.threads * 4);
    let failed = Arc::new(AtomicBool::new(false));
    let handles: Vec<_> = (0..o.threads).map(|wid| {
        let (rx, sink, fl) = (rx.clone(), sink.clone(), failed.clone());
        let parser = (inp.make_parser)();
        std::thread::spawn(move || sort_worker(rx, parser, sink, cap, wid, fl))
    }).collect();
    drop(rx);

    let mut chunk = vec![0u8; 1 << 24];
    let mut tail = std::mem::take(&mut inp.body_tail);
    let mut err: Option<anyhow::Error> = None;
    loop {
        // keep blocks to a few MB so the workers stay evenly fed
        if tail.len() >= 1 << 22 {
            if let Some(end) = (inp.cut)(&tail) {
                let blk = tail[..end].to_vec();
                tail.drain(..end);
                if tx.send(blk).is_err() { break; }
            }
        }
        if failed.load(Ordering::Relaxed) { break; }
        match inp.rd.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => tail.extend_from_slice(&chunk[..n]),
            Err(e) => { err = Some(e.into()); break; }
        }
    }
    if !tail.is_empty() && err.is_none() {
        if tail.last() != Some(&b'\n') { tail.push(b'\n'); }
        tx.send(tail).ok();
    }
    drop(tx);
    drop(inp.rd);
    if let Some(mut c) = inp.child {
        let st = c.wait()?;
        if !st.success() && !failed.load(Ordering::Relaxed) {
            err = Some(anyhow!("bgzip failed on {} ({}): the input is truncated or corrupt", inp.name, st));
        }
    }
    let (mut runs, mut npairs, mut t1) = (Vec::new(), 0u64, [0f64; 3]);
    for h in handles {
        match h.join() {
            Ok(Ok((r, np, t))) => { runs.extend(r); npairs += np; for k in 0..3 { t1[k] += t[k]; } }
            Ok(Err(e)) => err = Some(e),
            Err(_) => err = Some(anyhow!("sort worker thread panicked")),
        }
    }
    if let Some(e) = err { let _ = std::fs::remove_dir_all(&o.tmpdir); return Err(e); }
    if log {
        let wall1 = t0.elapsed().as_secs_f64();
        let run_bytes: u64 = runs.iter().filter_map(|r| std::fs::metadata(&r.path).ok()).map(|m| m.len()).sum();
        eprintln!("  phase 1: {} pairs -> {} runs ({:.2} B/pair), {:.1}s wall ({:.1} Mpairs/s); thread-s: parse {:.0}, sort {:.0}, write runs {:.0}",
                  npairs, runs.len(), run_bytes as f64 / npairs.max(1) as f64, wall1, npairs as f64 / wall1 / 1e6, t1[0], t1[1], t1[2]);
    }
    Ok((runs, npairs))
}

/// Duplicate removal over a sorted stream, fed one sorted slab at a time.
///
/// Rule ("anchored greedy"): going through the pairs in file order, a pair is a duplicate if a
/// pair that was KEPT earlier has the same chromosomes and strands and both positions within
/// `mm` bp. Deterministic and independent of how the stream is cut into slabs. (pairtools'
/// default backend instead links duplicates transitively inside 10,000-line chunks.)
pub struct Deduper {
    mm: u32,
    /// kept pairs of earlier slabs that a later pair can still match
    carry: Vec<Pair>,
    work: Vec<Pair>,
    kept: Vec<bool>,
}

impl Deduper {
    pub fn new(mm: u32) -> Deduper { Deduper { mm, carry: Vec::new(), work: Vec::new(), kept: Vec::new() } }

    /// Remove duplicates from `v` (sorted, following every earlier slab); returns how many.
    pub fn run(&mut self, v: &mut Vec<Pair>) -> u64 {
        if v.is_empty() { return 0; }
        let mm = self.mm;
        let c = self.carry.len();
        self.work.clear();
        self.work.extend_from_slice(&self.carry);
        self.work.extend_from_slice(v);
        let w = &self.work;
        self.kept.clear();
        self.kept.resize(w.len(), true);
        // starts of the most recent groups of equal k1 (a pair can only match within the last
        // mm+1 distinct positions)
        let mut groups: std::collections::VecDeque<(u64, usize)> = std::collections::VecDeque::new();
        let mut ndup = 0u64;
        for i in 0..w.len() {
            let p = w[i];
            if groups.back().is_none_or(|g| g.0 != p.k1) {
                groups.push_back((p.k1, i));
                if groups.len() > mm as usize + 2 { groups.pop_front(); }
            }
            if i < c { continue; }
            let lo1 = (p.k1 & !0xffff_ffff) | (p.pos1().saturating_sub(mm)) as u64;
            let lo2 = (p.k2 & !0xffff_ffff) | (p.pos2().saturating_sub(mm)) as u64;
            let hi2 = p.k2 + mm as u64;
            let strands = p.aux >> 32;
            let mut dup = false;
            'g: for gi in (0..groups.len()).rev() {
                let (k, start) = groups[gi];
                if k < lo1 { break; }
                let end = if gi + 1 < groups.len() { groups[gi + 1].1 } else { i };
                let g = &w[start..end];
                let mut j = if g.len() > 16 { g.partition_point(|q| q.k2 < lo2) } else { 0 };
                while j < g.len() && g[j].k2 <= hi2 {
                    if g[j].k2 >= lo2 && self.kept[start + j] && g[j].aux >> 32 == strands { dup = true; break 'g; }
                    j += 1;
                }
            }
            if dup { self.kept[i] = false; ndup += 1; }
        }
        let last = w[w.len() - 1];
        let keep_from = (last.k1 & !0xffff_ffff) | (last.pos1().saturating_sub(mm)) as u64;
        let a = w.partition_point(|q| q.k1 < keep_from);
        let carry: Vec<Pair> = (a..w.len()).filter(|&j| self.kept[j]).map(|j| w[j]).collect();
        let mut j = c;
        v.retain(|_| { j += 1; self.kept[j - 1] });
        self.carry = carry;
        ndup
    }
}

/// Counters in the vocabulary of `pairtools stats`.
#[derive(Default, Clone)]
pub struct Stats {
    /// counts by name, e.g. "pair_types/MU"; summed across inputs
    pub counts: std::collections::BTreeMap<String, u64>,
    /// set when this run computed the per-pair numbers below
    pub have_pairs: bool,
    pub dups: u64,
    pub nodups: u64,
    pub cis: u64,
    pub trans: u64,
    pub cis_ge: [u64; 6],
    pub chrom_freq: Vec<u64>,
    nchrom: usize,
}

const CIS_KB: [u32; 6] = [1, 2, 4, 10, 20, 40];

impl Stats {
    pub fn new(nchrom: usize) -> Stats { Stats { nchrom, chrom_freq: vec![0; nchrom * nchrom], ..Default::default() } }
    /// Add the `name\tvalue` lines of another stats block.
    pub fn add_text(&mut self, text: &str) {
        for l in text.lines() {
            if let Some((k, v)) = l.split_once('\t') {
                if let Ok(v) = v.trim().parse::<u64>() { *self.counts.entry(k.to_string()).or_insert(0) += v; }
            }
        }
    }
    fn add_kept(&mut self, v: &[Pair]) {
        self.have_pairs = true;
        self.nodups += v.len() as u64;
        for p in v {
            let (c1, c2) = (p.chrom1() as usize, p.chrom2() as usize);
            self.chrom_freq[c1 * self.nchrom + c2] += 1;
            if c1 == c2 {
                self.cis += 1;
                let d = p.pos2() - p.pos1();
                for (k, kb) in CIS_KB.iter().enumerate() { if d >= kb * 1000 { self.cis_ge[k] += 1; } }
            } else { self.trans += 1; }
        }
    }
    /// pairtools-stats style text (the subset computed here; no dist_freq).
    pub fn to_text(&self, chroms: &[(String, u64)]) -> String {
        let g = |k: &str| self.counts.get(k).copied().unwrap_or(0);
        let mut pt: std::collections::BTreeMap<String, u64> = self.counts.iter()
            .filter_map(|(k, v)| k.strip_prefix("pair_types/").map(|t| (t.to_string(), *v))).collect();
        let mut s = String::new();
        if self.have_pairs {
            pt.insert("UU".into(), self.nodups);
            if self.dups > 0 || pt.contains_key("DD") { pt.insert("DD".into(), self.dups); }
            let sum = |ts: &[&str]| ts.iter().map(|t| pt.get(*t).copied().unwrap_or(0)).sum::<u64>();
            let mapped = self.nodups + self.dups;
            let unmapped = sum(&["NN", "NM", "MM", "XX"]);
            let single = sum(&["NU", "MU"]);
            s += &format!("total\t{}\ntotal_unmapped\t{}\ntotal_single_sided_mapped\t{}\ntotal_mapped\t{}\ntotal_dups\t{}\ntotal_nodups\t{}\ncis\t{}\ntrans\t{}\n",
                          mapped + unmapped + single, unmapped, single, mapped, self.dups, self.nodups, self.cis, self.trans);
            for (t, v) in &pt { s += &format!("pair_types/{}\t{}\n", t, v); }
            for (k, kb) in CIS_KB.iter().enumerate() { s += &format!("cis_{}kb+\t{}\n", kb, self.cis_ge[k]); }
            let fr = |a: u64, b: u64| if b > 0 { a as f64 / b as f64 } else { 0.0 };
            s += &format!("summary/frac_cis\t{}\n", fr(self.cis, self.nodups));
            for (k, kb) in CIS_KB.iter().enumerate() { s += &format!("summary/frac_cis_{}kb+\t{}\n", kb, fr(self.cis_ge[k], self.nodups)); }
            s += &format!("summary/frac_dups\t{}\n", fr(self.dups, mapped));
            for c1 in 0..self.nchrom { for c2 in 0..self.nchrom {
                let v = self.chrom_freq[c1 * self.nchrom + c2];
                if v > 0 { s += &format!("chrom_freq/{}/{}\t{}\n", chroms[c1].0, chroms[c2].0, v); }
            } }
        } else {
            s += &format!("total\t{}\n", g("total"));
            for (t, v) in &pt { s += &format!("pair_types/{}\t{}\n", t, v); }
            if g("read_pairs") > 0 { s += &format!("read_pairs\t{}\n", g("read_pairs")); }
        }
        s
    }
}

/// Phase 2: merge sorted runs into one sorted output, optionally removing duplicates.
/// `stats` carries counters in (from the inputs) and out (per-pair numbers, when computed).
#[allow(clippy::too_many_arguments)]
fn merge_runs(runs: &[Run], header: &Header, out: &str, o: &SortOpts, dedup: Option<u32>,
              stats: &mut Stats, compute_stats: bool, log: bool) -> Result<u64> {
    let t0 = Instant::now();
    let npairs: u64 = runs.iter().map(|r| r.rows.iter().map(|&n| n as u64).sum::<u64>()).sum();
    let out_ctx = PqCtx::new(header, o.out_codec, false)?;
    let mem_pairs = o.mem_gb * 1e9 / std::mem::size_of::<Pair>() as f64;
    // cut the key space into units from the runs' row-group first keys
    let unit = ((mem_pairs / o.threads as f64 / 3.0) as u64).max(1 << 20);
    let mut marks: Vec<(u64, u32)> = runs.iter().flat_map(|r| r.first.iter().copied().zip(r.rows.iter().copied())).collect();
    marks.sort_unstable();
    let mut bounds = vec![0u64];
    let mut acc = 0u64;
    for (k, n) in marks {
        if acc >= unit && k > *bounds.last().unwrap() { bounds.push(k); acc = 0; }
        acc += n as u64;
    }
    bounds.push(u64::MAX);
    let units: Vec<(u64, u64)> = bounds.windows(2).map(|w| (w[0], w[1])).collect();

    let pool = rayon::ThreadPoolBuilder::new().num_threads(o.threads).build()?;
    let ns = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)]; // read, merge, encode
    let mut t_seq = 0f64;
    let mut dd = dedup.map(Deduper::new);
    let mut guard = crate::cooler::PartialFileGuard::new(out);
    let (mut read, mut written) = (0u64, 0u64);
    let res = (|| -> Result<()> {
        let mut w = PqWriter::create(out, &out_ctx)?;
        guard.arm();
        // a window of units is built in parallel, deduplicated in key order, encoded in parallel
        for win in units.chunks(o.threads) {
            let built: Vec<Result<Vec<Pair>>> = pool.install(|| win.par_iter().map(|&(lo, hi)| {
                let ta = Instant::now();
                let (mut v, mut scratch) = (Vec::new(), Vec::new());
                load_unit(runs, lo, hi, &mut v, &mut scratch)?;
                drop(scratch);
                let tb = Instant::now();
                // stable sort = merge of the runs' already-sorted stretches
                v.sort();
                ns[0].fetch_add((tb - ta).as_nanos() as u64, Ordering::Relaxed);
                ns[1].fetch_add(tb.elapsed().as_nanos() as u64, Ordering::Relaxed);
                Ok(v)
            }).collect());
            let mut slabs: Vec<Vec<Pair>> = built.into_iter().collect::<Result<_>>()?;
            let ts = Instant::now();
            for v in slabs.iter_mut() {
                read += v.len() as u64;
                if let Some(d) = dd.as_mut() { stats.dups += d.run(v); }
                if compute_stats { stats.add_kept(v); }
                written += v.len() as u64;
            }
            t_seq += ts.elapsed().as_secs_f64();
            let enc: Vec<Result<Vec<(Vec<ArrowColumnChunk>, u64)>>> = pool.install(|| slabs.par_iter().map(|v| {
                let tc = Instant::now();
                let r = v.chunks(OUT_ROW_GROUP).map(|c| Ok((out_ctx.encode_row_group(c)?, c[0].k1))).collect::<Result<Vec<_>>>();
                ns[2].fetch_add(tc.elapsed().as_nanos() as u64, Ordering::Relaxed);
                r
            }).collect());
            for e in enc { for (chunks, first) in e? { w.append(chunks, first)?; } }
        }
        if read != npairs { bail!("internal error: merged {} pairs, expected {}", read, npairs); }
        let text = stats.to_text(&header.chroms);
        w.close(Some(&text))?;
        Ok(())
    })();
    res?;
    guard.defuse();
    if log {
        let wall2 = t0.elapsed().as_secs_f64();
        let sz = std::fs::metadata(out)?.len();
        let s = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e9;
        eprintln!("  merge: {} units, {:.1}s wall ({:.1} Mpairs/s); thread-s: read {:.0}, merge {:.0}, encode {:.0}; serial dedup/stats {:.1}s",
                  units.len(), wall2, npairs as f64 / wall2 / 1e6, s(&ns[0]), s(&ns[1]), s(&ns[2]), t_seq);
        eprintln!("  wrote {} pairs ({} removed as duplicates); {} bytes, {:.3} B/pair", written, read - written, sz, sz as f64 / written.max(1) as f64);
    }
    Ok(written)
}

/// Any block-parsed input -> sorted pairs Parquet. `stats` ends up in the output footer.
pub fn sort_input(inp: SortInput, out: &str, o: &SortOpts, stats: &dyn Fn() -> Stats, log: bool) -> Result<u64> {
    let t0 = Instant::now();
    let header = inp.header.clone();
    let (runs, npairs) = make_runs(inp, o, log)?;
    let mut st = stats();
    let res = merge_runs(&runs, &header, out, o, None, &mut st, false, log);
    for r in &runs { let _ = std::fs::remove_file(&r.path); }
    let _ = std::fs::remove_dir(&o.tmpdir);
    let n = res?;
    if n != npairs { bail!("internal error: wrote {} pairs, parsed {}", n, npairs); }
    if log { eprintln!("  sort DONE: {:.1}s", t0.elapsed().as_secs_f64()); }
    Ok(n)
}

/// Text .pairs(.gz) -> pairs Parquet file in canonical order.
pub fn sort(input: &str, out: &str, o: &SortOpts, log: bool) -> Result<u64> {
    let (mut rd, child) = open_text(input, o.threads)?;
    let (th, body_tail) = read_text_header(&mut *rd, input)?;
    let mut columns = 0u8;
    if th.cols.ptype.is_some() { columns |= HAS_PTYPE; }
    if th.cols.mapq1.is_some() { columns |= HAS_MAPQ; }
    let header = Header {
        chroms: th.chroms.clone(), sorted: SORTED_CANONICAL.into(),
        assembly: th.assembly.clone(), columns,
    };
    if th.chroms.len() > u32::MAX as usize { bail!("too many chromosomes"); }
    let cx = Arc::new(ParseCtx {
        cmap: th.chroms.iter().enumerate().map(|(i, (n, _))| (n.as_bytes().to_vec(), i as u32)).collect(),
        lengths: th.chroms.iter().map(|c| c.1).collect(),
        cols: th.cols,
    });
    let make_parser = || -> BlockParser {
        let cx = cx.clone();
        let mut cache = ChromCache::default();
        Box::new(move |blk: &[u8], out: &mut Vec<Pair>| {
            let mut start = 0usize;
            while let Some(rel) = memchr(b'\n', &blk[start..]) {
                if let Some(p) = parse_text_line(&blk[start..start + rel], &cx, &mut cache)? { out.push(p); }
                start += rel + 1;
            }
            Ok(())
        })
    };
    let nchrom = header.chroms.len();
    sort_input(SortInput { rd, body_tail, child, name: input.to_string(), header, cut: cut_lines, make_parser: &make_parser },
               out, o, &|| Stats::new(nchrom), log)
}

/// Merge sorted pairs files (same chromosomes, canonical order) into one; with `dedup`,
/// remove duplicates within that many bp. Returns the stats of the result.
pub fn merge(inputs: &[String], out: &str, o: &SortOpts, dedup: Option<u32>, log: bool) -> Result<Stats> {
    let mut runs = Vec::new();
    let mut header: Option<Header> = None;
    let mut stats: Option<Stats> = None;
    for p in inputs {
        let file = PqFile::open(p)?;
        if file.header.sorted != SORTED_CANONICAL { bail!("{}: not sorted {} — run `pairs sort` first", p, SORTED_CANONICAL); }
        match &header {
            None => { stats = Some(Stats::new(file.header.chroms.len())); header = Some(file.header.clone()); }
            Some(h) => {
                if h.chroms != file.header.chroms { bail!("{}: chromosomes differ from {}", p, inputs[0]); }
                if h.columns != file.header.columns { bail!("{}: optional columns differ from {}", p, inputs[0]); }
            }
        }
        stats.as_mut().unwrap().add_text(&file.stats);
        let first = file.first_keys.clone().ok_or_else(|| anyhow!(
            "{}: no row-group keys in the footer (file from an older version) — re-create it", p))?;
        let rows = (0..file.num_row_groups()).map(|g| file.rows(g) as u32).collect();
        runs.push(Run { path: p.clone(), first, rows, file });
    }
    let header = header.ok_or_else(|| anyhow!("no inputs"))?;
    let mut stats = stats.unwrap();
    if dedup.is_none() { stats.dups = stats.counts.get("pair_types/DD").copied().unwrap_or(0); }
    merge_runs(&runs, &header, out, o, dedup, &mut stats, true, log)?;
    Ok(stats)
}

// ---------------------------------------------------------------------------------------------
// scan / view

pub struct ScanReport {
    pub npairs: u64,
    pub nblocks: u64,
    /// compressed bytes per column
    pub col_bytes: Vec<(String, u64)>,
    pub sorted: bool,
    /// order-independent digest of every pair (sum of a per-pair hash)
    pub digest: u64,
    pub seconds: f64,
}

#[inline]
fn pair_hash(p: &Pair) -> u64 {
    let mut h = p.k1.wrapping_mul(0x9E3779B97F4A7C15);
    h = (h ^ p.k2).wrapping_mul(0xC2B2AE3D27D4EB4F);
    h = (h ^ p.aux).wrapping_mul(0x165667B19E3779F9);
    h ^ (h >> 29)
}

/// (first, last, sorted within, digest, n) of one decoded block
type BlockCheck = (Pair, Pair, bool, u64, u64);

fn check_block(mut pairs: impl Iterator<Item = Pair>) -> Option<BlockCheck> {
    let first = pairs.next()?;
    let (mut prev, mut ok, mut dg, mut n) = (first, true, pair_hash(&first), 1u64);
    for p in pairs {
        ok &= prev <= p;
        dg = dg.wrapping_add(pair_hash(&p));
        n += 1;
        prev = p;
    }
    Some((first, prev, ok, dg, n))
}

/// Decode a whole file: count, check the canonical order, per-column size breakdown.
pub fn scan(path: &str, threads: usize) -> Result<ScanReport> {
    let t0 = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
    let mut rep = ScanReport { npairs: 0, nblocks: 0, col_bytes: Vec::new(), sorted: true, digest: 0, seconds: 0.0 };
    let mut last: Option<Pair> = None;
    let mut fold = |rep: &mut ScanReport, res: Vec<Result<Option<BlockCheck>>>| -> Result<()> {
        for r in res {
            if let Some((first, lastp, ok, dg, n)) = r? {
                rep.sorted &= ok && last.is_none_or(|l| l <= first);
                last = Some(lastp);
                rep.digest = rep.digest.wrapping_add(dg);
                rep.npairs += n;
                rep.nblocks += 1;
            }
        }
        Ok(())
    };
    let f = PqFile::open(path)?;
    let groups: Vec<usize> = (0..f.num_row_groups()).collect();
    for win in groups.chunks(threads * 2) {
        let res = pool.install(|| win.par_iter().map(|&g| {
            let mut v = Vec::new();
            f.read_row_group(g, &mut v)?;
            Ok(check_block(v.into_iter()))
        }).collect());
        fold(&mut rep, res)?;
    }
    rep.col_bytes = f.column_bytes();
    rep.seconds = t0.elapsed().as_secs_f64();
    Ok(rep)
}

#[inline]
fn push_u32(out: &mut Vec<u8>, mut v: u32) {
    let mut b = [0u8; 10];
    let mut i = b.len();
    loop { i -= 1; b[i] = b'0' + (v % 10) as u8; v /= 10; if v == 0 { break; } }
    out.extend_from_slice(&b[i..]);
}

/// Pairs file -> 4DN text on `w` (readID is always "."). A closed pipe downstream is not an error.
///
/// `min_mapq`: keep pairs with both MAPQs at least this. `chroms`: a chromsizes file; keep only
/// pairs with both sides on its chromosomes, and list only those in the header.
pub fn view(path: &str, w: &mut dyn Write, with_header: bool, min_mapq: u8, chroms: Option<&str>) -> Result<()> {
    let f = PqFile::open(path)?;
    let h = f.header.clone();
    if min_mapq > 0 && h.columns & HAS_MAPQ == 0 { bail!("{}: --min-mapq needs mapq1/mapq2 columns", path); }
    let keep: Vec<bool> = match chroms {
        None => vec![true; h.chroms.len()],
        Some(p) => {
            let names: std::collections::HashSet<String> = std::fs::read_to_string(p)?.lines()
                .filter_map(|l| l.split_whitespace().next().map(|s| s.to_string())).collect();
            h.chroms.iter().map(|c| names.contains(&c.0)).collect()
        }
    };
    let all = min_mapq == 0 && keep.iter().all(|&k| k);
    let mut g = 0usize;
    let mut next = || -> Result<Option<Vec<Pair>>> {
        if g >= f.num_row_groups() { return Ok(None); }
        let mut v = Vec::new();
        f.read_row_group(g, &mut v)?;
        g += 1;
        Ok(Some(v))
    };
    let res = (|| -> std::io::Result<Result<()>> {
        if with_header {
            let mut t = String::from("## pairs format v1.0\n");
            t += &format!("#sorted: {}\n#shape: upper triangle\n", h.sorted);
            if !h.assembly.is_empty() { t += &format!("#genome_assembly: {}\n", h.assembly); }
            for (i, (n, l)) in h.chroms.iter().enumerate() { if keep[i] { t += &format!("#chromsize: {} {}\n", n, l); } }
            t += "#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2";
            if h.columns & HAS_PTYPE != 0 { t += " pair_type"; }
            if h.columns & HAS_MAPQ != 0 { t += " mapq1 mapq2"; }
            t.push('\n');
            w.write_all(t.as_bytes())?;
        }
        let mut out: Vec<u8> = Vec::with_capacity(1 << 24);
        loop {
            let blk = match next() { Ok(Some(b)) => b, Ok(None) => break, Err(e) => return Ok(Err(e)) };
            out.clear();
            for p in &blk {
                if !all && (p.mapq1() < min_mapq || p.mapq2() < min_mapq
                            || !keep[p.chrom1() as usize] || !keep[p.chrom2() as usize]) { continue; }
                out.extend_from_slice(b".\t");
                out.extend_from_slice(h.chroms[p.chrom1() as usize].0.as_bytes()); out.push(b'\t');
                push_u32(&mut out, p.pos1()); out.push(b'\t');
                out.extend_from_slice(h.chroms[p.chrom2() as usize].0.as_bytes()); out.push(b'\t');
                push_u32(&mut out, p.pos2()); out.push(b'\t');
                out.push(if p.strands() & 1 != 0 { b'-' } else { b'+' }); out.push(b'\t');
                out.push(if p.strands() & 2 != 0 { b'-' } else { b'+' });
                if h.columns & HAS_PTYPE != 0 { out.push(b'\t'); out.extend_from_slice(&p.ptype()); }
                if h.columns & HAS_MAPQ != 0 {
                    out.push(b'\t'); push_u32(&mut out, p.mapq1() as u32);
                    out.push(b'\t'); push_u32(&mut out, p.mapq2() as u32);
                }
                out.push(b'\n');
            }
            w.write_all(&out)?;
        }
        w.flush()?;
        Ok(Ok(()))
    })();
    match res {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        Err(e) => Err(e.into()),
    }
}
