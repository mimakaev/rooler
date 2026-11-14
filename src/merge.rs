//! K-way streaming drain-and-count merge of sorted (key,count) block sources.
//! Heap-based (O(N log K)); draining all equal keys (across streams and within a stream)
//! yields an aggregated sorted (key,count) stream. Bounded RAM = K * block + emit buffer.
use crate::cooler::{clamp_count, CoolWriter, CoolerPix, Comp, read_meta};
use anyhow::{anyhow, Result};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Counts for a block. Spilled cload runs are one-pair-per-key, so `Ones` lets them skip
/// materializing a vec of 1s — that halves phase-B resident memory per source.
pub enum Counts {
    Ones,
    PerKey(Vec<i64>),
}
impl Counts {
    #[inline]
    pub fn at(&self, i: usize) -> i64 {
        match self { Counts::Ones => 1, Counts::PerKey(v) => v[i] }
    }
}

pub trait BlockSource {
    /// Next sorted block as (keys, counts); None when exhausted. Empty blocks are skipped.
    fn next(&mut self) -> Result<Option<(Vec<i64>, Counts)>>;
}

struct Cursor<S: BlockSource> {
    src: S,
    k: Vec<i64>,
    c: Counts,
    pos: usize,
}
impl<S: BlockSource> Cursor<S> {
    fn new(src: S) -> Result<Cursor<S>> {
        let mut cur = Cursor { src, k: Vec::new(), c: Counts::Ones, pos: 0 };
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
                    Some(h) if h == k => { sum += cur[i].c.at(cur[i].pos); cur[i].pos += 1;
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

/// All inputs must share the same bin layout, else the merged pixel table is meaningless.
/// `meta` is the reference (read from paths[0]).
fn check_inputs_match(meta: &crate::cooler::Meta, paths: &[String], grp: &str) -> Result<()> {
    for p in &paths[1..] {
        let m = read_meta(p, grp)?;
        if m.binsize != meta.binsize {
            anyhow::bail!("merge: {} has binsize {} but {} has {}", p, m.binsize, paths[0], meta.binsize);
        }
        if m.nbins != meta.nbins {
            anyhow::bail!("merge: {} has {} bins but {} has {}", p, m.nbins, paths[0], meta.nbins);
        }
        if m.names.len() != meta.names.len() {
            anyhow::bail!("merge: {} has {} chromosomes but {} has {}",
                p, m.names.len(), paths[0], meta.names.len());
        }
        if m.names != meta.names {
            let d = m.names.iter().zip(meta.names.iter()).position(|(a, b)| a != b);
            anyhow::bail!("merge: {} has different chromosomes than {} (first difference at index {:?}: {:?} vs {:?})",
                p, paths[0], d, d.and_then(|i| m.names.get(i)), d.and_then(|i| meta.names.get(i)));
        }
        if m.lengths != meta.lengths {
            anyhow::bail!("merge: {} has different chromosome lengths than {}", p, paths[0]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory BlockSource: a list of (keys, counts) blocks, yielded in order.
    struct VecSource(std::vec::IntoIter<(Vec<i64>, Vec<i64>)>);
    impl VecSource {
        fn new(blocks: Vec<(Vec<i64>, Vec<i64>)>) -> VecSource { VecSource(blocks.into_iter()) }
    }
    impl BlockSource for VecSource {
        fn next(&mut self) -> Result<Option<(Vec<i64>, Counts)>> {
            Ok(self.0.next().map(|(k, c)| (k, Counts::PerKey(c))))
        }
    }

    /// Same blocks but declared as count=1 (the cload spill-run shape).
    struct OnesSource(std::vec::IntoIter<Vec<i64>>);
    impl BlockSource for OnesSource {
        fn next(&mut self) -> Result<Option<(Vec<i64>, Counts)>> {
            Ok(self.0.next().map(|k| (k, Counts::Ones)))
        }
    }

    #[test]
    fn ones_counts_aggregate_like_explicit_ones() {
        // Counts::Ones must behave exactly like PerKey(vec![1; n]) -- it is the cload path
        let blocks = vec![vec![1i64, 2, 2], vec![2, 5], vec![5, 5, 9]];
        let ones = OnesSource(blocks.clone().into_iter());
        let explicit = VecSource::new(blocks.into_iter().map(|k| { let n = k.len(); (k, vec![1i64; n]) }).collect());
        let (mut ka, mut ca) = (Vec::new(), Vec::new());
        merge_sources(vec![ones], 2, |k, c| { ka.extend_from_slice(k); ca.extend_from_slice(c); Ok(()) }).unwrap();
        let (mut kb, mut cb) = (Vec::new(), Vec::new());
        merge_sources(vec![explicit], 2, |k, c| { kb.extend_from_slice(k); cb.extend_from_slice(c); Ok(()) }).unwrap();
        assert_eq!((ka, ca), (kb, cb));
    }

    /// run merge_sources with a tiny emit block and collect the whole output
    fn run(srcs: Vec<VecSource>, emit: usize) -> (Vec<i64>, Vec<i64>) {
        let (mut k, mut c) = (Vec::new(), Vec::new());
        let n = merge_sources(srcs, emit, |kk, cc| { k.extend_from_slice(kk); c.extend_from_slice(cc); Ok(()) }).unwrap();
        assert_eq!(n as usize, k.len(), "returned count must equal emitted keys");
        assert!(k.windows(2).all(|w| w[0] < w[1]), "output keys must be strictly increasing");
        (k, c)
    }

    #[test]
    fn merges_and_sums_overlapping_keys() {
        let a = VecSource::new(vec![(vec![1, 3, 5], vec![1, 1, 1])]);
        let b = VecSource::new(vec![(vec![3, 4], vec![10, 10])]);
        let c = VecSource::new(vec![(vec![1, 5, 9], vec![2, 2, 2])]);
        assert_eq!(run(vec![a, b, c], 2), (vec![1, 3, 4, 5, 9], vec![3, 11, 10, 3, 2]));
    }

    #[test]
    fn drains_duplicate_key_runs_spanning_blocks() {
        // the same key repeated *within* a source and *across* its block boundary
        let a = VecSource::new(vec![
            (vec![7, 7, 7], vec![1, 1, 1]),
            (vec![7, 7], vec![1, 1]),
            (vec![8], vec![5]),
        ]);
        let b = VecSource::new(vec![(vec![7], vec![100])]);
        assert_eq!(run(vec![a, b], 4), (vec![7, 8], vec![105, 5]));
    }

    #[test]
    fn handles_empty_and_degenerate_inputs() {
        assert_eq!(run(Vec::<VecSource>::new(), 4), (vec![], vec![]));
        assert_eq!(run(vec![VecSource::new(vec![])], 4), (vec![], vec![]));
        // empty blocks interleaved with real ones must be skipped, not terminate the source
        let a = VecSource::new(vec![(vec![], vec![]), (vec![2], vec![3]), (vec![], vec![]), (vec![4], vec![5])]);
        assert_eq!(run(vec![a], 1), (vec![2, 4], vec![3, 5]));
        let s = VecSource::new(vec![(vec![1, 2, 3], vec![1, 1, 1])]);
        assert_eq!(run(vec![s], 100), (vec![1, 2, 3], vec![1, 1, 1]));
    }

    #[test]
    fn emit_block_size_does_not_change_result() {
        let mk = || vec![
            VecSource::new(vec![(vec![1, 2, 2, 9], vec![1, 1, 1, 1])]),
            VecSource::new(vec![(vec![2, 3], vec![5, 5]), (vec![9, 10], vec![1, 1])]),
        ];
        let want = (vec![1, 2, 3, 9, 10], vec![1, 7, 5, 2, 1]);
        for emit in [1, 2, 3, 1000] {
            assert_eq!(run(mk(), emit), want, "emit block {}", emit);
        }
    }

    #[test]
    fn clamps_counts_at_i32_max() {
        let mut n = 0u64;
        assert_eq!(clamp_count(5, &mut n), 5);
        assert_eq!(clamp_count(i32::MAX as i64, &mut n), i32::MAX);
        assert_eq!(n, 0, "exactly i32::MAX must not count as clamped");
        assert_eq!(clamp_count(i32::MAX as i64 + 1, &mut n), i32::MAX);
        assert_eq!(clamp_count(9_000_000_000, &mut n), i32::MAX);
        assert_eq!(n, 2);
    }
}

/// Merge `nranges` disjoint, ascending slices of key space, each on its own thread, and stream
/// their outputs into the single writer IN ORDER over bounded channels: backpressure bounds RAM,
/// there are no temp files, and all HDF5 writes stay on the calling thread. `open(r)` builds the
/// sources for range r (called on that range's thread). Shared by merge and cload's phase B.
pub(crate) fn merge_ranges_to_writer<S: BlockSource>(
    nranges: usize, emit_block: usize, nbins: i64, w: &mut CoolWriter,
    open: impl Fn(usize) -> Result<Vec<S>> + Sync,
) -> Result<u64> {
    let nclamped = AtomicU64::new(0);
    let nnz = std::thread::scope(|s| -> Result<u64> {
        let mut rxs = Vec::new();
        let mut handles = Vec::new();
        for r in 0..nranges {
            let (tx, rx) = crossbeam_channel::bounded::<(Vec<i64>, Vec<i64>, Vec<i32>)>(2);
            rxs.push(rx);
            let (open, nclamped) = (&open, &nclamped);
            handles.push(s.spawn(move || -> Result<()> {
                let mut local = 0u64;
                let res = merge_sources(open(r)?, emit_block, |keys, cnts| {
                    let b1 = keys.iter().map(|&x| x / nbins).collect();
                    let b2 = keys.iter().map(|&x| x % nbins).collect();
                    let c = cnts.iter().map(|&x| clamp_count(x, &mut local)).collect();
                    tx.send((b1, b2, c)).map_err(|_| anyhow!("merge output channel closed"))
                });
                nclamped.fetch_add(local, Ordering::Relaxed);
                res.map(|_| ())
            }));
        }
        // drain ranges in order; later ranges keep merging behind their channels meanwhile.
        // (An early return drops the remaining receivers, which unblocks and ends the workers.)
        let mut nnz = 0u64;
        for rx in rxs {
            for (b1, b2, c) in rx { w.append(&b1, &b2, &c)?; nnz += b1.len() as u64; }
        }
        // a failed worker also just closes its channel, so the drain above cannot tell a
        // finished range from a failed one — check every worker BEFORE the caller finalizes
        for h in handles {
            h.join().map_err(|_| anyhow!("merge worker thread panicked"))??;
        }
        Ok(nnz)
    })?;
    let n = nclamped.load(Ordering::Relaxed);
    if n > 0 { eprintln!("  WARNING: {} pixels clamped at i32::MAX (counts are stored as i32)", n); }
    Ok(nnz)
}

/// Merge coolers (group `grp` of each path) by summing matching pixels. The bin1 axis is cut
/// into up to `nthreads` count-balanced ranges, sliced out of every input via its bin1_offset
/// index, and each range is k-way merged on its own thread.
#[allow(clippy::too_many_arguments)]
pub fn merge_coolers(
    paths: &[String], grp: &str, out: &str, mem_gb: f64, nthreads: usize, comp: Comp,
    assembly: Option<&str>, log: bool,
) -> Result<u64> {
    let t0 = std::time::Instant::now();
    let meta = read_meta(&paths[0], grp)?;
    check_inputs_match(&meta, paths, grp)?;
    let nbins = meta.nbins as i64;
    crate::cooler::check_key_space(nbins)?;
    let asm = crate::view::resolve_assembly(assembly, &meta.assembly, &meta.chromsizes())?;
    let offs: Vec<Vec<i64>> = paths.iter().map(|p| CoolerPix::bin1_offset(p, grp)).collect::<Result<_>>()?;
    // count-balanced bin1 partition
    let mut per = vec![0i64; meta.nbins];
    for o in &offs { for b in 0..meta.nbins { per[b] += o[b + 1] - o[b]; } }
    let p = nthreads.max(1);
    let target = (per.iter().sum::<i64>() / p as i64).max(1);
    let mut bounds = vec![0usize]; let mut acc = 0i64;
    for b in 0..meta.nbins { acc += per[b]; if acc >= target && bounds.len() < p { bounds.push(b + 1); acc = 0; } }
    bounds.push(meta.nbins);
    let nranges = bounds.len() - 1;
    // each range holds one decoded block per input cursor (~24 B/pixel incl. read transients),
    // so resident memory scales with ranges x inputs x block
    let block = ((mem_gb * 0.3 * 1e9 / (24.0 * nranges as f64 * paths.len() as f64)) as usize)
        .clamp(1 << 16, 4_000_000);
    if log { eprintln!("  merge: {} inputs, {} ranges, block={} pix, assembly={}", paths.len(), nranges, block, asm); }

    let mut guard = crate::cooler::PartialFileGuard::new(out);
    let mut w = CoolWriter::create(out, &meta.names, &meta.lengths, meta.binsize, comp, &asm)?;
    guard.arm();
    let nnz = merge_ranges_to_writer(nranges, block, nbins, &mut w, |r| {
        let (lo, hi) = (bounds[r], bounds[r + 1]);
        paths.iter().zip(&offs).map(|(pth, o)| {
            CoolerPix::open_slice(pth, grp, meta.nbins, block, o[lo] as usize, o[hi] as usize)
        }).collect()
    })?;
    w.close()?;
    guard.defuse();
    if log { eprintln!("  merge DONE: {} pixels in {:.0}s", nnz, t0.elapsed().as_secs_f64()); }
    Ok(nnz)
}
