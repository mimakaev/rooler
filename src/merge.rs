//! K-way streaming drain-and-count merge of sorted (key,count) block sources.
//! Heap-based (O(N log K)); draining all equal keys (across streams and within a stream)
//! yields an aggregated sorted (key,count) stream. Bounded RAM = K * block + emit buffer.
use crate::cooler::{CoolWriter, CoolerPix, Comp, read_meta};
use anyhow::Result;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub trait BlockSource {
    /// Next sorted block as (keys, counts); None when exhausted. Empty blocks are skipped.
    fn next(&mut self) -> Result<Option<(Vec<i64>, Vec<i64>)>>;
}

struct Cursor<S: BlockSource> {
    src: S,
    k: Vec<i64>,
    c: Vec<i64>,
    pos: usize,
}
impl<S: BlockSource> Cursor<S> {
    fn new(src: S) -> Result<Cursor<S>> {
        let mut cur = Cursor { src, k: Vec::new(), c: Vec::new(), pos: 0 };
        cur.fill()?;
        Ok(cur)
    }
    /// Ensure pos points at a valid entry, pulling blocks as needed. false = exhausted.
    fn fill(&mut self) -> Result<bool> {
        while self.pos >= self.k.len() {
            match self.src.next()? {
                Some((k, c)) => { self.k = k; self.c = c; self.pos = 0; }
                None => return Ok(false),
            }
        }
        Ok(true)
    }
    fn head(&self) -> Option<i64> { self.k.get(self.pos).copied() }
}

/// Merge sources into an aggregated sorted (key,count) stream, feeding a callback per output block.
pub fn merge_sources<S: BlockSource>(
    srcs: Vec<S>, emit_block: usize, mut emit: impl FnMut(&[i64], &[i64]) -> Result<()>,
) -> Result<u64> {
    let mut cur: Vec<Cursor<S>> = srcs.into_iter().map(Cursor::new).collect::<Result<_>>()?;
    let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
    for (i, c) in cur.iter().enumerate() {
        if let Some(k) = c.head() { heap.push(Reverse((k, i))); }
    }
    let mut okey = Vec::with_capacity(emit_block);
    let mut ocnt = Vec::with_capacity(emit_block);
    let mut nout: u64 = 0;
    while let Some(&Reverse((k, _))) = heap.peek() {
        // pop all cursors currently at key k
        let mut idxs = Vec::new();
        while let Some(&Reverse((kk, i))) = heap.peek() {
            if kk == k { heap.pop(); idxs.push(i); } else { break; }
        }
        let mut sum: i64 = 0;
        for &i in &idxs {
            // drain this cursor's full run of k (may span blocks)
            loop {
                match cur[i].head() {
                    Some(h) if h == k => { sum += cur[i].c[cur[i].pos]; cur[i].pos += 1;
                        if cur[i].pos >= cur[i].k.len() && !cur[i].fill()? { break; } }
                    _ => break,
                }
            }
            if let Some(nk) = cur[i].head() { heap.push(Reverse((nk, i))); }
        }
        okey.push(k); ocnt.push(sum); nout += 1;
        if okey.len() >= emit_block { emit(&okey, &ocnt)?; okey.clear(); ocnt.clear(); }
    }
    if !okey.is_empty() { emit(&okey, &ocnt)?; }
    Ok(nout)
}

/// Merge sorted key/count sources directly into a cooler writer (bin1,bin2,count).
/// nthreads reserved for the ranged-parallel path (currently single-thread heap merge).
pub fn merge_sources_parallel<S: BlockSource>(
    srcs: Vec<S>, nbins: i64, _nthreads: usize, w: &mut CoolWriter,
) -> Result<u64> {
    let emit_block = 1 << 22;
    merge_sources(srcs, emit_block, |keys, cnts| {
        let b1: Vec<i64> = keys.iter().map(|&x| x / nbins).collect();
        let b2: Vec<i64> = keys.iter().map(|&x| x % nbins).collect();
        let c: Vec<i32> = cnts.iter().map(|&x| x as i32).collect();
        w.append(&b1, &b2, &c)
    })
}

/// Ranged-parallel merge: partition bin1 into P count-balanced ranges (sliced from each input via
/// bin1_offset), merge each range on its own thread, and stream range outputs to the writer IN ORDER
/// via bounded channels (backpressure bounds RAM; no temp files). Falls back to serial for P<=1.
pub fn merge_coolers_parallel(
    paths: &[String], res: Option<&str>, out: &str, mem_gb: f64, nthreads: usize, comp: Comp,
    assembly: Option<&str>, log: bool,
) -> Result<u64> {
    use std::sync::Arc;
    if nthreads <= 1 { return merge_coolers(paths, res, out, mem_gb, comp, assembly, log); }
    let t0 = std::time::Instant::now();
    let meta = read_meta(&paths[0], res)?;
    let nbins = meta.nbins as i64;
    let chromsizes: Vec<(String, i64)> = meta.names.iter().cloned().zip(meta.lengths.iter().cloned()).collect();
    let asm = match assembly {
        Some(a) if !a.trim().is_empty() => a.trim().to_string(),
        _ if !meta.assembly.is_empty() && meta.assembly != "unknown" => meta.assembly.clone(),
        _ => crate::view::detect("", &chromsizes).map(|s| s.to_string()).ok_or_else(|| anyhow::anyhow!(
            "refusing to merge without a genome assembly; pass --assembly"))?,
    };
    let offs: Vec<Vec<i64>> = paths.iter().map(|p| CoolerPix::bin1_offset(p, res)).collect::<Result<_>>()?;
    // count-balanced bin1 partition into P ranges
    let mut per = vec![0i64; meta.nbins];
    for o in &offs { for b in 0..meta.nbins { per[b] += o[b + 1] - o[b]; } }
    let total: i64 = per.iter().sum();
    let p = nthreads;
    let target = (total / p as i64).max(1);
    let mut bounds = vec![0usize]; let mut acc = 0i64;
    for b in 0..meta.nbins { acc += per[b]; if acc >= target && bounds.len() < p { bounds.push(b + 1); acc = 0; } }
    bounds.push(meta.nbins);
    let nranges = bounds.len() - 1;
    let block = ((mem_gb * 0.3 * 1e9 / (24.0 * nranges as f64)) as usize).min(4_000_000).max(1 << 16);
    if log { eprintln!("  merge(parallel): {} inputs, {} ranges, block={} pix, assembly={}", paths.len(), nranges, block, asm); }

    let paths_a = Arc::new(paths.to_vec());
    let offs_a = Arc::new(offs);
    let res_a: Option<String> = res.map(|s| s.to_string());
    let mut rxs = Vec::new();
    let mut handles = Vec::new();
    for r in 0..nranges {
        let (lo, hi) = (bounds[r], bounds[r + 1]);
        let (tx, rx) = crossbeam_channel::bounded::<(Vec<i64>, Vec<i64>, Vec<i32>)>(2);
        rxs.push(rx);
        let (pa, oa, ra) = (paths_a.clone(), offs_a.clone(), res_a.clone());
        let nb = meta.nbins;
        handles.push(std::thread::spawn(move || -> Result<u64> {
            let srcs: Vec<CoolerPix> = pa.iter().enumerate().map(|(k, pth)| {
                let (p0, p1) = (oa[k][lo] as usize, oa[k][hi] as usize);
                CoolerPix::open_slice(pth, ra.as_deref(), nb, block, p0, p1)
            }).collect::<Result<_>>()?;
            merge_sources(srcs, block, |keys, cnts| {
                let b1 = keys.iter().map(|&x| x / nbins).collect();
                let b2 = keys.iter().map(|&x| x % nbins).collect();
                let c = cnts.iter().map(|&x| x as i32).collect();
                tx.send((b1, b2, c)).map_err(|_| anyhow::anyhow!("merge channel closed")).map(|_| ())
            })
        }));
    }
    // writer drains ranges in order (range r blocks until produced; later ranges merge concurrently)
    let mut w = CoolWriter::create(out, &meta.names, &meta.lengths, meta.binsize, meta.nbins, &meta.chrom_offset, comp, &asm)?;
    let mut nnz = 0u64;
    for r in 0..nranges {
        while let Ok((b1, b2, c)) = rxs[r].recv() { w.append(&b1, &b2, &c)?; nnz += b1.len() as u64; }
    }
    w.close()?;
    for h in handles { h.join().unwrap()?; }
    if log { eprintln!("  merge(parallel) DONE: {} pixels in {:.0}s", nnz, t0.elapsed().as_secs_f64()); }
    Ok(nnz)
}

pub fn merge_coolers(
    paths: &[String], res: Option<&str>, out: &str, mem_gb: f64, comp: Comp, assembly: Option<&str>, log: bool,
) -> Result<u64> {
    let t0 = std::time::Instant::now();
    let meta = read_meta(&paths[0], res)?;
    let nbins = meta.nbins as i64;
    // no mystery coolers: assembly from override, else inputs, else fingerprint, else refuse
    let chromsizes: Vec<(String, i64)> = meta.names.iter().cloned().zip(meta.lengths.iter().cloned()).collect();
    let asm = match assembly {
        Some(a) if !a.trim().is_empty() => a.trim().to_string(),
        _ if !meta.assembly.is_empty() && meta.assembly != "unknown" => meta.assembly.clone(),
        _ => crate::view::detect("", &chromsizes).map(|s| s.to_string()).ok_or_else(|| anyhow::anyhow!(
            "refusing to merge into a cooler without a genome assembly (inputs lack one); pass --assembly <name>"))?,
    };
    let k = paths.len().max(1);
    let block = ((mem_gb * 1e9 / (16.0 * k as f64)) as usize).max(1 << 18); // 16 B/entry buffered
    let srcs: Vec<CoolerPix> = paths.iter()
        .map(|p| CoolerPix::open(p, res, meta.nbins, block))
        .collect::<Result<_>>()?;
    if log { eprintln!("  merge {} coolers, {} nbins, assembly={}, block={} pix (mem {:.1}G)", k, meta.nbins, asm, block, mem_gb); }
    let mut w = CoolWriter::create(out, &meta.names, &meta.lengths, meta.binsize, meta.nbins,
        &meta.chrom_offset, comp, &asm)?;
    let emit_block = 1 << 22;
    let nnz = merge_sources(srcs, emit_block, |keys, cnts| {
        let b1: Vec<i64> = keys.iter().map(|&x| x / nbins).collect();
        let b2: Vec<i64> = keys.iter().map(|&x| x % nbins).collect();
        let c: Vec<i32> = cnts.iter().map(|&x| x as i32).collect();
        w.append(&b1, &b2, &c)
    })?;
    w.close()?;
    if log { eprintln!("  merge DONE: {} pixels in {:.1}s", nnz, t0.elapsed().as_secs_f64()); }
    Ok(nnz)
}
