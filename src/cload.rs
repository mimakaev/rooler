//! cload: 4DN .pairs.gz -> .cool, out-of-core, parallel parse.
//! Decompress via `bgzip -@` (parallel), a producer splits the stream into line-aligned blocks,
//! N worker threads each parse+bin+sort+spill their own runs. Phase B: k-way drain-and-count
//! merge over all runs (count=1) -> writer. RAM bounded by --mem (N worker buffers of mem/N).
use crate::cooler::{CoolWriter, Comp};
use crate::merge::{merge_sources, merge_sources_to_writer, BlockSource, Counts};
use anyhow::{anyhow, bail, Result};
use crossbeam_channel::{bounded, Receiver};
use lz4_flex::block::{compress, decompress};
use memchr::{memchr, memrchr};
use std::collections::HashMap;
use std::io::{Read, Write, BufWriter, BufReader};
use std::process::{Command, Stdio};
use std::sync::Arc;

/// Keys per compressed spill block. Smaller blocks mean smaller decode buffers, which keeps
/// the ranged-parallel phase B parallel when there are many runs (resident RAM there is
/// ranges x runs x one decoded block). Measured cost of 128K vs 1M: 1.866 vs 1.860 B/key,
/// i.e. 0.3% — nothing.
const SPILL_BLK: usize = 131_072;

pub fn shuffle8(d: &[i64]) -> Vec<u8> {
    let n = d.len(); let mut o = vec![0u8; 8 * n];
    for i in 0..n { let b = d[i].to_le_bytes(); for p in 0..8 { o[p * n + i] = b[p]; } }
    o
}
pub fn unshuffle8(buf: &[u8], n: usize) -> Vec<i64> {
    (0..n).map(|i| { let mut b = [0u8; 8]; for p in 0..8 { b[p] = buf[p * n + i]; } i64::from_le_bytes(b) }).collect()
}

pub struct Bins { pub cmap: HashMap<Vec<u8>, i64>, pub off_lo: Vec<i64>, pub nbins: i64, pub binsize: i64 }

#[inline]
pub fn fast_atoi(b: &[u8]) -> i64 { let mut x = 0i64; for &c in b { x = x * 10 + (c - b'0') as i64; } x }

/// Per-thread cache of the last chrom name -> id lookup (pairs files are chrom-clustered).
/// id -1 = empty cache; an empty name never hits (it can't be a valid cached lookup).
pub struct ChromCache { c1: Vec<u8>, i1: i64, c2: Vec<u8>, i2: i64 }
impl Default for ChromCache {
    fn default() -> Self { ChromCache { c1: Vec::new(), i1: -1, c2: Vec::new(), i2: -1 } }
}

/// Parse one 4DN .pairs body line -> the sorted-pair key (bin1*nbins + bin2).
/// Fields (tab-separated): 0=readID 1=chr1 2=pos1 3=chr2 4=pos2 ...
/// Returns Ok(None) for comment/blank/short lines; Err for a chromosome missing from the header.
#[inline]
pub fn parse_line(line: &[u8], bins: &Bins, cache: &mut ChromCache) -> Result<Option<i64>> {
    if line.is_empty() || line[0] == b'#' { return Ok(None); }
    // t[1..=4] = positions of tabs 1..4 (after readID, chr1, pos1, chr2)
    let mut t = [0usize; 5]; let mut nt = 0;
    for (i, &c) in line.iter().enumerate() {
        if c == b'\t' { nt += 1; t[nt] = i; if nt == 4 { break; } }
    }
    if nt < 4 { return Ok(None); }
    let c1 = &line[t[1] + 1..t[2]];
    let p1 = fast_atoi(&line[t[2] + 1..t[3]]);
    let c2 = &line[t[3] + 1..t[4]];
    let p2rest = &line[t[4] + 1..];
    let p2 = fast_atoi(match memchr(b'\t', p2rest) { Some(x) => &p2rest[..x], None => p2rest });
    let lookup = |c: &[u8]| -> Result<i64> {
        bins.cmap.get(c).copied().ok_or_else(|| anyhow!(
            "pairs line references chromosome {:?} not present in the #chromsize header",
            String::from_utf8_lossy(c)))
    };
    let i1 = if cache.i1 >= 0 && c1 == cache.c1.as_slice() { cache.i1 }
             else { let v = lookup(c1)?; cache.c1 = c1.to_vec(); cache.i1 = v; v };
    let i2 = if cache.i2 >= 0 && c2 == cache.c2.as_slice() { cache.i2 }
             else { let v = lookup(c2)?; cache.c2 = c2.to_vec(); cache.i2 = v; v };
    let b1 = bins.off_lo[i1 as usize] + p1 / bins.binsize;
    let b2 = bins.off_lo[i2 as usize] + p2 / bins.binsize;
    let (lo, hi) = if b1 <= b2 { (b1, b2) } else { (b2, b1) };
    Ok(Some(lo * bins.nbins + hi))
}

/// Spill run file: [magic 8B] then blocks of [n u32][clen u32][first_key i64][lz4 bytes].
/// Keys are delta-coded, byte-shuffled and LZ4'd (~1.8 B/key vs 8 raw -> 4.5x less spill IO).
/// `first_key` lets a reader seek to a key range by scanning headers only, no decompression,
/// which is what makes the ranged-parallel phase-B merge possible.
/// Runs are transient within one cload call, so the format needs no backward compatibility.
const SPILL_MAGIC: &[u8; 8] = b"RKZ2\0\0\0\0";
const SPILL_HDR: usize = 16;

pub fn write_spill(w: &mut impl Write, sorted_keys: &[i64]) -> Result<()> {
    w.write_all(SPILL_MAGIC)?;
    for chunk in sorted_keys.chunks(SPILL_BLK) {
        let n = chunk.len();
        let mut d = vec![0i64; n];
        d[0] = chunk[0];
        for i in 1..n { d[i] = chunk[i] - chunk[i - 1]; }
        let comp = compress(&shuffle8(&d));
        w.write_all(&(n as u32).to_le_bytes())?;
        w.write_all(&(comp.len() as u32).to_le_bytes())?;
        w.write_all(&chunk[0].to_le_bytes())?;
        w.write_all(&comp)?;
    }
    Ok(())
}

/// (first_key, byte offset of the block header) for every block in a run — header scan only.
pub fn scan_spill_index(path: &str) -> Result<Vec<(i64, u64)>> {
    use std::io::{Seek, SeekFrom};
    let mut f = BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != SPILL_MAGIC { bail!("{}: not a rooler spill run (bad magic)", path); }
    let mut out = Vec::new();
    let mut pos = 8u64;
    loop {
        let mut hdr = [0u8; SPILL_HDR];
        if f.read_exact(&mut hdr).is_err() { break; } // EOF
        let clen = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as u64;
        out.push((i64::from_le_bytes(hdr[8..16].try_into().unwrap()), pos));
        pos += SPILL_HDR as u64 + clen;
        f.seek(SeekFrom::Start(pos))?;
    }
    Ok(out)
}

fn worker(rx: Receiver<Vec<u8>>, bins: Arc<Bins>, cap: usize, tmpdir: Arc<String>, wid: usize)
    -> Result<(Vec<String>, u64)> {
    let mut buf: Vec<i64> = Vec::with_capacity(cap);
    let mut paths = Vec::new();
    let mut np = 0u64;
    let mut seq = 0usize;
    let mut cache = ChromCache::default();
    let spill = |buf: &mut Vec<i64>, seq: &mut usize, paths: &mut Vec<String>| -> Result<()> {
        if buf.is_empty() { return Ok(()); }
        buf.sort_unstable();
        let p = format!("{}/r{}_{}.kz", tmpdir, wid, seq);
        let mut w = BufWriter::new(std::fs::File::create(&p)?);
        write_spill(&mut w, buf)?;
        w.flush()?; paths.push(p); *seq += 1; buf.clear();
        Ok(())
    };
    for block in rx.iter() {
        let mut start = 0usize;
        while let Some(rel) = memchr(b'\n', &block[start..]) {
            let line = &block[start..start + rel];
            start += rel + 1;
            if let Some(key) = parse_line(line, &bins, &mut cache)? {
                buf.push(key);
                np += 1;
                if buf.len() >= cap { spill(&mut buf, &mut seq, &mut paths)?; }
            }
        }
    }
    let mut buf = buf; let mut seq = seq; let mut paths2 = paths;
    spill(&mut buf, &mut seq, &mut paths2)?;
    Ok((paths2, np))
}

pub fn cload(pairs: &str, binsize: i64, out: &str, mem_gb: f64, nthreads: usize,
             comp: Comp, tmpdir: &str, assembly: Option<&str>, log: bool) -> Result<u64> {
    let t0 = std::time::Instant::now();
    // ".gz" -> parallel bgzip decode; "-" -> stdin; anything else -> plain text on disk
    let mut child = if pairs.ends_with(".gz") {
        Some(Command::new("bgzip").args(["-dc", "-@", &nthreads.to_string(), pairs])
            .stdout(Stdio::piped()).spawn()?)
    } else { None };
    let mut rd: Box<dyn Read> = match child.as_mut() {
        Some(c) => Box::new(c.stdout.take().ok_or_else(|| anyhow!("no bgzip stdout"))?),
        None if pairs == "-" => Box::new(std::io::stdin()),
        None => Box::new(std::fs::File::open(pairs)?),
    };

    // --- parse header (chromsizes) ---
    let mut names: Vec<String> = Vec::new();
    let mut lengths: Vec<i64> = Vec::new();
    let mut cmap: HashMap<Vec<u8>, i64> = HashMap::new();
    let mut leftover: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; 1 << 24];
    let mut body_tail: Vec<u8> = Vec::new();
    'hdr: loop {
        let n = rd.read(&mut chunk)?;
        if n == 0 { break; }
        leftover.extend_from_slice(&chunk[..n]);
        let mut start = 0;
        while let Some(rel) = memchr(b'\n', &leftover[start..]) {
            let line = &leftover[start..start + rel];
            if !line.is_empty() && line[0] == b'#' {
                if line.starts_with(b"#chromsize:") {
                    let rest = &line[b"#chromsize:".len()..];
                    let parts: Vec<&[u8]> = rest.split(|&c| c == b' ' || c == b'\t').filter(|s| !s.is_empty()).collect();
                    if parts.len() == 2 {
                        cmap.insert(parts[0].to_vec(), names.len() as i64);
                        names.push(String::from_utf8_lossy(parts[0]).into_owned());
                        lengths.push(fast_atoi(parts[1]));
                    }
                }
                start += rel + 1;
            } else {
                body_tail = leftover[start..].to_vec();
                break 'hdr;
            }
        }
        leftover.drain(..start);
    }
    let mut off_lo = vec![0i64];
    for &l in &lengths { off_lo.push(off_lo.last().unwrap() + (l + binsize - 1) / binsize); }
    let nbins = *off_lo.last().unwrap();
    off_lo.pop();
    // no mystery coolers: require a genome assembly (explicit or fingerprinted) or refuse
    let chromsizes: Vec<(String, i64)> = names.iter().cloned().zip(lengths.iter().cloned()).collect();
    let asm = crate::view::resolve_assembly(assembly, &chromsizes).ok_or_else(|| anyhow!(
        "refusing to create a cooler without a genome assembly: could not detect it from the chromsizes; pass --assembly <name>"))?;
    if log { eprintln!("  genome-assembly: {}", asm); }
    let bins = Arc::new(Bins { cmap, off_lo, nbins, binsize });

    // --- spawn workers ---
    std::fs::create_dir_all(tmpdir)?;
    let cap = (((mem_gb * 1e9 / 8.0) as usize) / nthreads).max(1 << 20);
    let (tx, rx) = bounded::<Vec<u8>>(nthreads * 4);
    let td = Arc::new(tmpdir.to_string());
    let mut handles = Vec::new();
    for wid in 0..nthreads {
        let (rx, bins, td) = (rx.clone(), bins.clone(), td.clone());
        handles.push(std::thread::spawn(move || worker(rx, bins, cap, td, wid)));
    }
    drop(rx);

    // --- producer: line-aligned blocks ---
    let mut tail = body_tail;
    if let Some(pos) = memrchr(b'\n', &tail) { let blk = tail[..pos + 1].to_vec(); tx.send(blk).ok(); tail.drain(..pos + 1); }
    loop {
        let n = rd.read(&mut chunk)?;
        if n == 0 { break; }
        tail.extend_from_slice(&chunk[..n]);
        if let Some(pos) = memrchr(b'\n', &tail) { let blk = tail[..pos + 1].to_vec(); tx.send(blk).ok(); tail.drain(..pos + 1); }
    }
    if !tail.is_empty() { tail.push(b'\n'); tx.send(tail).ok(); }
    drop(tx);
    drop(rd);
    if let Some(mut c) = child { c.wait()?; }

    let mut run_paths = Vec::new(); let mut npairs = 0u64;
    for h in handles { let (p, np) = h.join().unwrap()?; run_paths.extend(p); npairs += np; }
    if log { eprintln!("  phase1: {} pairs -> {} runs, {:.0}s ({:.0} Mpairs/s)",
        npairs, run_paths.len(), t0.elapsed().as_secs_f64(), npairs as f64 / t0.elapsed().as_secs_f64() / 1e6); }

    // --- phase B: ranged-parallel k-way drain-and-count merge over the spilled runs ---
    let mut off = vec![0i64];
    for &l in &lengths { off.push(off.last().unwrap() + (l + binsize - 1) / binsize); }
    let mut w = CoolWriter::create(out, &names, &lengths, binsize, nbins as usize, &off, comp, &asm)?;
    let nnz = merge_runs_parallel(&run_paths, nbins, nthreads, mem_gb, &mut w, log)?;
    w.close()?;
    for p in &run_paths { let _ = std::fs::remove_file(p); }
    let _ = std::fs::remove_dir(tmpdir);
    if log { eprintln!("  cload DONE: {} pairs -> {} pixels, {:.0}s", npairs, nnz, t0.elapsed().as_secs_f64()); }
    Ok(nnz)
}

/// Phase B: split the key space into P ranges (from the runs' per-block first_keys), merge each
/// range on its own thread, and stream the range outputs to the single writer IN ORDER over
/// bounded channels. Backpressure bounds RAM; no temp files, and HDF5 stays on one thread.
fn merge_runs_parallel(
    run_paths: &[String], nbins: i64, nthreads: usize, mem_gb: f64, w: &mut CoolWriter, log: bool,
) -> Result<u64> {
    // resident during phase B ~ ranges x runs x (SPILL_BLK keys decoded + scratch) ~ 1.1 MB each
    let per_reader_bytes = SPILL_BLK as f64 * 8.0 * 1.4;
    let nruns = run_paths.len().max(1);
    let budget = ((mem_gb * 0.5 * 1e9) / (nruns as f64 * per_reader_bytes)) as usize;
    let p = nthreads.min(budget).max(1);
    if p <= 1 || nruns == 0 {
        if log { eprintln!("  phase2: serial merge ({} runs)", nruns); }
        return merge_sources_to_writer(
            run_paths.iter().map(|s| RunReader::open(s)).collect::<Result<Vec<_>>>()?, nbins, w);
    }
    // one header-only scan per run gives both the range pivots and the per-run block index
    let indexes: Vec<Vec<(i64, u64)>> = run_paths.iter().map(|s| scan_spill_index(s)).collect::<Result<_>>()?;
    let mut firsts: Vec<i64> = indexes.iter().flat_map(|v| v.iter().map(|&(k, _)| k)).collect();
    firsts.sort_unstable();
    let mut bounds = vec![i64::MIN];
    if !firsts.is_empty() {
        for i in 1..p {
            let k = firsts[i * firsts.len() / p];
            if k > *bounds.last().unwrap() { bounds.push(k); }
        }
    }
    bounds.push(i64::MAX);
    let nranges = bounds.len() - 1;
    if log { eprintln!("  phase2: {} runs, {} ranges x {} threads", nruns, nranges, p); }

    let paths = Arc::new(run_paths.to_vec());
    let idx = Arc::new(indexes);
    let nclamped = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut rxs = Vec::new();
    let mut handles = Vec::new();
    for r in 0..nranges {
        let (lo, hi) = (bounds[r], bounds[r + 1]);
        let (tx, rx) = bounded::<(Vec<i64>, Vec<i64>, Vec<i32>)>(2);
        rxs.push(rx);
        let (pa, ia, nc) = (paths.clone(), idx.clone(), nclamped.clone());
        handles.push(std::thread::spawn(move || -> Result<u64> {
            let srcs: Vec<RunReader> = pa.iter().enumerate()
                .map(|(k, s)| RunReader::open_range(s, &ia[k], lo, hi))
                .collect::<Result<_>>()?;
            let mut local = 0u64;
            let n = merge_sources(srcs, 1 << 22, |keys, cnts| {
                let b1 = keys.iter().map(|&x| x / nbins).collect();
                let b2 = keys.iter().map(|&x| x % nbins).collect();
                let c = cnts.iter().map(|&x| crate::cooler::clamp_count(x, &mut local)).collect();
                tx.send((b1, b2, c)).map_err(|_| anyhow!("phase-B channel closed")).map(|_| ())
            });
            nc.fetch_add(local, std::sync::atomic::Ordering::Relaxed);
            n
        }));
    }
    // writer drains ranges in order; later ranges keep merging concurrently behind their channels
    let mut nnz = 0u64;
    for rx in &rxs {
        while let Ok((b1, b2, c)) = rx.recv() { w.append(&b1, &b2, &c)?; nnz += b1.len() as u64; }
    }
    for h in handles { h.join().unwrap()?; }
    let nc = nclamped.load(std::sync::atomic::Ordering::Relaxed);
    if nc > 0 { eprintln!("  WARNING: {} pixels clamped at i32::MAX (counts are stored as i32)", nc); }
    Ok(nnz)
}

/// Reads i64 keys from a spilled run in blocks; yields (keys, counts=1).
/// Optionally confined to a half-open key range [lo, hi) for the ranged-parallel phase B.
pub struct RunReader {
    f: BufReader<std::fs::File>,
    lo: i64,
    hi: Option<i64>,
    done: bool,
}
impl RunReader {
    pub fn open(p: &str) -> Result<RunReader> {
        let mut f = BufReader::new(std::fs::File::open(p)?);
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != SPILL_MAGIC { bail!("{}: not a rooler spill run (bad magic)", p); }
        Ok(RunReader { f, lo: i64::MIN, hi: None, done: false })
    }

    /// Reader restricted to keys in [lo, hi). `index` comes from `scan_spill_index` (headers
    /// only), so seeking costs no decompression. Blocks are key-sorted, so clipping within a
    /// block is a pair of binary searches; ranges partition key space and every duplicate of a
    /// key lands in exactly one range, so no pixel is split across ranges.
    ///
    /// Seek rule: start at the LAST block whose first_key is *strictly* less than lo. A key
    /// equal to lo may have duplicates trailing at the end of that earlier block (a run of
    /// equal keys can straddle a block boundary), and `<=` here would skip them.
    /// Everything before that block is bounded above by its first_key, hence entirely < lo.
    pub fn open_range(p: &str, index: &[(i64, u64)], lo: i64, hi: i64) -> Result<RunReader> {
        use std::io::{Seek, SeekFrom};
        let mut r = Self::open(p)?;
        r.lo = lo;
        r.hi = Some(hi);
        let start = index.partition_point(|&(fk, _)| fk < lo).saturating_sub(1);
        match index.get(start) {
            Some(&(_, off)) => { r.f.seek(SeekFrom::Start(off))?; }
            None => { r.done = true; }  // empty run
        }
        Ok(r)
    }
}
impl BlockSource for RunReader {
    fn next(&mut self) -> Result<Option<(Vec<i64>, Counts)>> {
        loop {
            if self.done { return Ok(None); }
            let mut hdr = [0u8; SPILL_HDR];
            if self.f.read_exact(&mut hdr).is_err() { return Ok(None); } // EOF
            let n = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
            let clen = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
            let first = i64::from_le_bytes(hdr[8..16].try_into().unwrap());
            // this block and all later ones start at/after hi -> nothing more for this range
            if let Some(hi) = self.hi {
                if first >= hi { self.done = true; return Ok(None); }
            }
            let mut cbuf = vec![0u8; clen];
            self.f.read_exact(&mut cbuf)?;
            let raw = decompress(&cbuf, 8 * n).map_err(|e| anyhow!("spill decompress: {}", e))?;
            let d = unshuffle8(&raw, n);
            let mut keys = vec![0i64; n];
            let mut acc = 0i64;
            for i in 0..n { acc += d[i]; keys[i] = acc; }
            if self.hi.is_none() { return Ok(Some((keys, Counts::Ones))); }
            // clip to [lo, hi) -- only the first and last block of a range are ever partial
            let s = keys.partition_point(|&k| k < self.lo);
            let e = keys.partition_point(|&k| k < self.hi.unwrap());
            if s >= e { continue; }  // wholly outside the range; try the next block
            return Ok(Some((keys[s..e].to_vec(), Counts::Ones)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bins3() -> Bins {
        // 2 chroms of 1000bp at binsize 100 -> 10 bins each, nbins 20
        let mut cmap = HashMap::new();
        cmap.insert(b"chr1".to_vec(), 0i64);
        cmap.insert(b"chr2".to_vec(), 1i64);
        Bins { cmap, off_lo: vec![0, 10], nbins: 20, binsize: 100 }
    }

    #[test]
    fn shuffle8_roundtrips() {
        for v in [vec![0i64], vec![-1, 0, 1, i64::MAX, i64::MIN], (0..5000).map(|i| i * 7 - 99).collect()] {
            let sh = shuffle8(&v);
            assert_eq!(sh.len(), 8 * v.len());
            assert_eq!(unshuffle8(&sh, v.len()), v);
        }
    }

    #[test]
    fn parse_line_basic() {
        let b = bins3();
        let mut c = ChromCache::default();
        // chr1:150 (bin 1) x chr1:350 (bin 3) -> key 1*20+3
        assert_eq!(parse_line(b"r1\tchr1\t150\tchr1\t350\t+\t+", &b, &mut c).unwrap(), Some(23));
        // reversed order must be normalized to (lo,hi)
        assert_eq!(parse_line(b"r2\tchr1\t350\tchr1\t150\t+\t+", &b, &mut c).unwrap(), Some(23));
        // trans: chr1:50 (bin 0) x chr2:250 (bin 10+2=12)
        assert_eq!(parse_line(b"r3\tchr1\t50\tchr2\t250\t+\t+", &b, &mut c).unwrap(), Some(12));
        // no trailing fields after pos2 (pos2 runs to end of line)
        assert_eq!(parse_line(b"r4\tchr1\t150\tchr1\t350", &b, &mut c).unwrap(), Some(23));
        // comments/blank/short lines are skipped
        assert_eq!(parse_line(b"#comment", &b, &mut c).unwrap(), None);
        assert_eq!(parse_line(b"", &b, &mut c).unwrap(), None);
        assert_eq!(parse_line(b"r5\tchr1\t150", &b, &mut c).unwrap(), None);
    }

    #[test]
    fn parse_line_unknown_chrom_errors() {
        let b = bins3();
        let mut c = ChromCache::default();
        let e = parse_line(b"r1\tchr1\t150\tchrBOGUS\t350\t+\t+", &b, &mut c).unwrap_err();
        assert!(e.to_string().contains("chrBOGUS"), "error should name the chromosome: {}", e);
        // an empty chrom name must not be silently accepted via the empty cache
        assert!(parse_line(b"r2\t\t150\tchr1\t350\t+\t+", &b, &mut c).is_err());
    }

    #[test]
    fn chrom_cache_survives_alternating_chroms() {
        let b = bins3();
        let mut c = ChromCache::default();
        // alternate so the cache is repeatedly invalidated; results must stay correct
        for _ in 0..4 {
            assert_eq!(parse_line(b"r\tchr1\t50\tchr1\t50\t+\t+", &b, &mut c).unwrap(), Some(0));
            assert_eq!(parse_line(b"r\tchr2\t50\tchr2\t50\t+\t+", &b, &mut c).unwrap(), Some(10 * 20 + 10));
        }
    }

    /// Ranged readers must partition the run exactly: concatenating the ranges in order must
    /// reproduce the whole key stream, with no key lost, duplicated or reordered.
    #[test]
    fn ranged_readers_partition_the_run_exactly() {
        let n = 3 * SPILL_BLK + 517;
        let mut keys: Vec<i64> = Vec::with_capacity(n);
        let mut acc = 0i64;
        let mut s = 42u64;
        for i in 0..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            // deliberately include duplicate keys: they must never straddle a range boundary
            acc += if i % 3 == 0 { 0 } else { (s >> 40) as i64 % 50 + 1 };
            keys.push(acc);
        }
        let dir = std::env::temp_dir().join(format!("rooler_range_test_{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.kz");
        {
            let mut w = BufWriter::new(std::fs::File::create(&path).unwrap());
            write_spill(&mut w, &keys).unwrap();
            w.flush().unwrap();
        }
        let ps = path.to_str().unwrap();
        let index = scan_spill_index(ps).unwrap();
        assert_eq!(index.len(), 4, "expected 4 spill blocks");
        assert!(index.windows(2).all(|w| w[0].0 <= w[1].0), "block first_keys must be sorted");

        for bounds in [
            vec![i64::MIN, i64::MAX],                                    // single range
            vec![i64::MIN, keys[n / 2], i64::MAX],                        // split on a real key
            vec![i64::MIN, index[1].0, index[2].0, index[3].0, i64::MAX],  // split on block starts
            vec![i64::MIN, keys[0], keys[0] + 1, keys[n - 1], i64::MAX],   // degenerate edges
            vec![i64::MIN, -5, i64::MAX],                                 // range before all keys
            vec![i64::MIN, keys[n - 1] + 1000, i64::MAX],                 // range after all keys
        ] {
            let mut got = Vec::with_capacity(n);
            for r in 0..bounds.len() - 1 {
                let mut rr = RunReader::open_range(ps, &index, bounds[r], bounds[r + 1]).unwrap();
                while let Some((k, _)) = rr.next().unwrap() {
                    assert!(k.iter().all(|&x| x >= bounds[r] && x < bounds[r + 1]),
                        "reader emitted a key outside [{}, {})", bounds[r], bounds[r + 1]);
                    got.extend_from_slice(&k);
                }
            }
            assert_eq!(got, keys, "ranges {:?} did not reproduce the run", bounds);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn spill_roundtrips_across_block_boundaries() {
        // more than 2 full SPILL_BLK blocks + a partial one
        let n = 2 * SPILL_BLK + 12_345;
        let mut keys: Vec<i64> = Vec::with_capacity(n);
        let mut acc = 0i64;
        let mut s = 7u64;
        for _ in 0..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            acc += (s >> 40) as i64 % 1000 + 1; // strictly increasing, varied deltas
            keys.push(acc);
        }
        let dir = std::env::temp_dir().join(format!("rooler_spill_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.kz");
        {
            let mut w = BufWriter::new(std::fs::File::create(&path).unwrap());
            write_spill(&mut w, &keys).unwrap();
            w.flush().unwrap();
        }
        // spill must actually be compact (that is its whole purpose)
        let sz = std::fs::metadata(&path).unwrap().len() as f64 / n as f64;
        assert!(sz < 4.0, "spill should be well under 8 B/key, got {:.2}", sz);

        let mut rr = RunReader::open(path.to_str().unwrap()).unwrap();
        let mut got = Vec::with_capacity(n);
        while let Some((k, c)) = rr.next().unwrap() {
            // spill runs are one pair per key -> Counts::Ones (no materialized vec of 1s)
            assert!(matches!(c, Counts::Ones), "spill runs must yield Counts::Ones");
            assert!((0..k.len()).all(|i| c.at(i) == 1));
            got.extend_from_slice(&k);
        }
        assert_eq!(got, keys);
        std::fs::remove_dir_all(&dir).ok();
    }
}
