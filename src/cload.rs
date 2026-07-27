//! cload: 4DN .pairs.gz -> .cool, out-of-core, parallel parse.
//! Decompress via `bgzip -@` (parallel), a producer splits the stream into line-aligned blocks,
//! N worker threads each parse+bin+sort+spill their own runs. Phase B: k-way drain-and-count
//! merge over all runs (count=1) -> writer. RAM bounded by --mem (N worker buffers of mem/N).
use crate::cooler::{CoolWriter, Comp};
use crate::merge::{merge_sources_parallel, BlockSource};
use anyhow::{anyhow, Result};
use crossbeam_channel::{bounded, Receiver};
use lz4_flex::block::{compress, decompress};
use memchr::{memchr, memrchr};
use std::collections::HashMap;
use std::io::{Read, Write, BufWriter, BufReader};
use std::process::{Command, Stdio};
use std::sync::Arc;

const SPILL_BLK: usize = 1_000_000; // keys per compressed spill block

fn shuffle8(d: &[i64]) -> Vec<u8> {
    let n = d.len(); let mut o = vec![0u8; 8 * n];
    for i in 0..n { let b = d[i].to_le_bytes(); for p in 0..8 { o[p * n + i] = b[p]; } }
    o
}
fn unshuffle8(buf: &[u8], n: usize) -> Vec<i64> {
    (0..n).map(|i| { let mut b = [0u8; 8]; for p in 0..8 { b[p] = buf[p * n + i]; } i64::from_le_bytes(b) }).collect()
}

struct Bins { cmap: HashMap<Vec<u8>, i64>, off_lo: Vec<i64>, nbins: i64, binsize: i64 }

#[inline]
fn fast_atoi(b: &[u8]) -> i64 { let mut x = 0i64; for &c in b { x = x * 10 + (c - b'0') as i64; } x }

fn worker(rx: Receiver<Vec<u8>>, bins: Arc<Bins>, cap: usize, tmpdir: Arc<String>, wid: usize)
    -> Result<(Vec<String>, u64)> {
    let mut buf: Vec<i64> = Vec::with_capacity(cap);
    let mut paths = Vec::new();
    let mut np = 0u64;
    let mut seq = 0usize;
    let (mut lc1, mut li1, mut lc2, mut li2): (Vec<u8>, i64, Vec<u8>, i64) = (Vec::new(), -1, Vec::new(), -1);
    let nb = bins.nbins;
    // compressed sorted-key runs: per block, delta + byte-shuffle + LZ4 (less spill IO)
    let spill = |buf: &mut Vec<i64>, seq: &mut usize, paths: &mut Vec<String>| -> Result<()> {
        if buf.is_empty() { return Ok(()); }
        buf.sort_unstable();
        let p = format!("{}/r{}_{}.kz", tmpdir, wid, seq);
        let mut w = BufWriter::new(std::fs::File::create(&p)?);
        for chunk in buf.chunks(SPILL_BLK) {
            let n = chunk.len();
            let mut d = vec![0i64; n];
            d[0] = chunk[0];
            for i in 1..n { d[i] = chunk[i] - chunk[i - 1]; }
            let comp = compress(&shuffle8(&d));
            w.write_all(&(n as u32).to_le_bytes())?;
            w.write_all(&(comp.len() as u32).to_le_bytes())?;
            w.write_all(&comp)?;
        }
        w.flush()?; paths.push(p); *seq += 1; buf.clear();
        Ok(())
    };
    for block in rx.iter() {
        let mut start = 0usize;
        while let Some(rel) = memchr(b'\n', &block[start..]) {
            let line = &block[start..start + rel];
            start += rel + 1;
            if line.is_empty() || line[0] == b'#' { continue; }
            // fields (tab-separated): 0=readID 1=chr1 2=pos1 3=chr2 4=pos2 ...
            // t[1..=4] = positions of tabs 1..4 (after readID, chr1, pos1, chr2)
            let mut t = [0usize; 5]; let mut nt = 0;
            for (i, &c) in line.iter().enumerate() {
                if c == b'\t' { nt += 1; t[nt] = i; if nt == 4 { break; } }
            }
            if nt < 4 { continue; }
            let c1 = &line[t[1] + 1..t[2]];
            let p1 = fast_atoi(&line[t[2] + 1..t[3]]);
            let c2 = &line[t[3] + 1..t[4]];
            let p2rest = &line[t[4] + 1..];
            let p2 = fast_atoi(match memchr(b'\t', p2rest) { Some(x) => &p2rest[..x], None => p2rest });
            let i1 = if c1 == lc1.as_slice() { li1 } else { let v = bins.cmap[c1]; lc1 = c1.to_vec(); li1 = v; v };
            let i2 = if c2 == lc2.as_slice() { li2 } else { let v = bins.cmap[c2]; lc2 = c2.to_vec(); li2 = v; v };
            let b1 = bins.off_lo[i1 as usize] + p1 / bins.binsize;
            let b2 = bins.off_lo[i2 as usize] + p2 / bins.binsize;
            let (lo, hi) = if b1 <= b2 { (b1, b2) } else { (b2, b1) };
            buf.push(lo * nb + hi);
            np += 1;
            if buf.len() >= cap { spill(&mut buf, &mut seq, &mut paths)?; }
        }
    }
    let mut buf = buf; let mut seq = seq; let mut paths2 = paths;
    spill(&mut buf, &mut seq, &mut paths2)?;
    Ok((paths2, np))
}

pub fn cload(pairs: &str, binsize: i64, out: &str, mem_gb: f64, nthreads: usize,
             comp: Comp, tmpdir: &str, assembly: Option<&str>, log: bool) -> Result<u64> {
    let t0 = std::time::Instant::now();
    let mut child = Command::new("bgzip").args(["-dc", "-@", &nthreads.to_string(), pairs])
        .stdout(Stdio::piped()).spawn()?;
    let mut rd = child.stdout.take().ok_or_else(|| anyhow!("no bgzip stdout"))?;

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
    child.wait()?;

    let mut run_paths = Vec::new(); let mut npairs = 0u64;
    for h in handles { let (p, np) = h.join().unwrap()?; run_paths.extend(p); npairs += np; }
    if log { eprintln!("  phase1: {} pairs -> {} runs, {:.0}s ({:.0} Mpairs/s)",
        npairs, run_paths.len(), t0.elapsed().as_secs_f64(), npairs as f64 / t0.elapsed().as_secs_f64() / 1e6); }

    // --- phase B: ranged-parallel drain-and-count merge ---
    let mut off = vec![0i64];
    for &l in &lengths { off.push(off.last().unwrap() + (l + binsize - 1) / binsize); }
    // cap phase-B per-run block so total merge RAM stays well under --mem (phase A already used it)
    let block = ((mem_gb * 0.4 * 1e9 / (8.0 * run_paths.len().max(1) as f64)) as usize).min(8_000_000).max(1 << 16);
    let mut w = CoolWriter::create(out, &names, &lengths, binsize, nbins as usize, &off, comp, &asm)?;
    let nnz = merge_sources_parallel(
        run_paths.iter().map(|p| RunReader::open(p, block)).collect::<Result<Vec<_>>>()?,
        nbins, nthreads, &mut w)?;
    w.close()?;
    for p in &run_paths { let _ = std::fs::remove_file(p); }
    let _ = std::fs::remove_dir(tmpdir);
    if log { eprintln!("  cload DONE: {} pairs -> {} pixels, {:.0}s", npairs, nnz, t0.elapsed().as_secs_f64()); }
    Ok(nnz)
}

/// Reads i64 keys from a spilled run in blocks; yields (keys, counts=1).
pub struct RunReader { f: BufReader<std::fs::File> }
impl RunReader {
    pub fn open(p: &str, _block: usize) -> Result<RunReader> {
        Ok(RunReader { f: BufReader::new(std::fs::File::open(p)?) })
    }
}
impl BlockSource for RunReader {
    fn next(&mut self) -> Result<Option<(Vec<i64>, Vec<i64>)>> {
        let mut hdr = [0u8; 8];
        if self.f.read_exact(&mut hdr).is_err() { return Ok(None); } // EOF
        let n = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
        let clen = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
        let mut cbuf = vec![0u8; clen];
        self.f.read_exact(&mut cbuf)?;
        let raw = decompress(&cbuf, 8 * n).map_err(|e| anyhow!("spill decompress: {}", e))?;
        let d = unshuffle8(&raw, n);
        let mut keys = vec![0i64; n];
        let mut acc = 0i64;
        for i in 0..n { acc += d[i]; keys[i] = acc; }
        Ok(Some((keys, vec![1i64; n])))
    }
}
