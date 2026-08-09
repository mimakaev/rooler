//! Uniform 2D-tiled compressed scratch for cache-blocked SpMV (large/fine coolers). Bins split into
//! blocks of size B; tile (I,J) [I<=J] holds pixels with bin1 in block I, bin2 in block J. Pixels
//! store block-LOCAL bin1/bin2 (u32, shuffle+lz4) + u8-exc count. In the SpMV all v/y access is
//! confined to the I- and J-block slices (<=B elems each -> L2-resident) -> kills the cache-miss wall
//! at 12M bins. Same result as the row-chunk scratch. Build: counting-sort each I-block band by J,
//! encode tiles in parallel.
use crate::scratch::{dec_count, enc_count, shuffle4, unshuffle4, BlobSink, BlobStore, SpMV};
use anyhow::Result;
use lz4_flex::block::{compress, decompress};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct TiledScratch {
    pub nbins: usize,
    pub nnz: usize,
    pub chrom_offset: Vec<i64>,
    b: i64,
    tile_i: Vec<i32>,
    tile_j: Vec<i32>,
    tile_np: Vec<i64>,
    b1_blob: BlobStore, b1_off: Vec<i64>,
    b2_blob: BlobStore, b2_off: Vec<i64>,
    cn_blob: BlobStore, cn_off: Vec<i64>,
    max_tile: usize,
    nthreads: usize,
}

impl TiledScratch {
    /// `spill`: None -> blobs in RAM; Some(prefix) -> blobs in unlinked mmap'd files "{prefix}.*".
    pub fn build(g: &hdf5::Group, block: i64, nthreads: usize, spill: Option<&str>) -> Result<TiledScratch> {
        let nbins = g.dataset("bins/start")?.shape()[0];
        let nnz = g.dataset("pixels/count")?.shape()[0];
        let mut rowptr: Vec<i64> = g.dataset("indexes/bin1_offset")?.read_1d::<i64>()?.to_vec();
        if rowptr.len() == nbins { rowptr.push(nnz as i64); }
        let chrom_offset: Vec<i64> = g.dataset("indexes/chrom_offset")?.read_1d::<i64>()?.to_vec();
        let bsz = block as usize;
        let nblocks = (nbins + bsz - 1) / bsz;
        let b2d = g.dataset("pixels/bin2_id")?;
        let cnd = g.dataset("pixels/count")?;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(nthreads).build().unwrap();

        let (mut tile_i, mut tile_j, mut tile_np) = (Vec::new(), Vec::new(), Vec::new());
        let mut b1_blob = BlobSink::new(spill, "b1")?;
        let mut b2_blob = BlobSink::new(spill, "b2")?;
        let mut cn_blob = BlobSink::new(spill, "cn")?;
        let (mut b1_off, mut b2_off, mut cn_off) = (vec![0i64], vec![0i64], vec![0i64]);
        let mut max_tile = 1usize;

        // Batched-read + parallel-band build (mirrors the row-chunk scratch): read a batch of
        // consecutive I-blocks once on this thread (HDF5 Dataset handle never shared across threads),
        // then do ALL per-band CPU work (local coords, counting-sort, scatter, encode) in parallel.
        let band_pix = |ib: usize| -> i64 {
            let r1 = ((ib + 1) * bsz).min(nbins);
            rowptr[r1] - rowptr[ib * bsz]
        };
        // spilling implies a tight budget -> smaller read batches (a batch is ~12 B/pixel of heap)
        let read_block = if spill.is_some() { 20_000_000i64 } else { 50_000_000i64 };
        let mut ib = 0usize;
        while ib < nblocks {
            let bstart = ib;
            let mut pix = 0i64;
            while ib < nblocks && pix < read_block { pix += band_pix(ib); ib += 1; }
            let bend = ib;
            let (gp0, gp1) = (rowptr[bstart * bsz], rowptr[(bend * bsz).min(nbins)]);
            if gp1 <= gp0 { continue; }
            let bin2 = b2d.read_slice_1d::<i64, _>(gp0 as usize..gp1 as usize)?;
            let cn = cnd.read_slice_1d::<i32, _>(gp0 as usize..gp1 as usize)?;
            let (bin2, cn) = (bin2.as_slice().unwrap(), cn.as_slice().unwrap());
            let rp = &rowptr;
            // process each whole band in the batch in parallel; result[k] holds band bstart+k's tiles
            let bands: Vec<Vec<(usize, usize, Vec<u8>, Vec<u8>, Vec<u8>)>> = pool.install(|| {
                (bstart..bend).into_par_iter().map(|ibb| {
                    let r0 = ibb * bsz;
                    let r1 = ((ibb + 1) * bsz).min(nbins);
                    let (p0, p1) = (rp[r0], rp[r1]);
                    let npix = (p1 - p0) as usize;
                    if npix == 0 { return Vec::new(); }
                    let base = (p0 - gp0) as usize; // offset into this batch's read buffers
                    // per-pixel bin1-local, J block, bin2-local
                    let mut b1l = vec![0u32; npix];
                    let mut jj = vec![0u32; npix];
                    let mut b2l = vec![0u32; npix];
                    let mut idx = 0usize;
                    for r in r0..r1 {
                        let c = (rp[r + 1] - rp[r]) as usize;
                        let rl = (r - r0) as u32;
                        for _ in 0..c {
                            let bv = bin2[base + idx];
                            let j = (bv / block) as u32;
                            b1l[idx] = rl; jj[idx] = j; b2l[idx] = (bv - (j as i64) * block) as u32;
                            idx += 1;
                        }
                    }
                    // counting-sort by J (J in [ibb, nblocks)) -> contiguous per-tile buckets
                    let mut jcount = vec![0usize; nblocks + 1];
                    for &j in &jj { jcount[j as usize] += 1; }
                    let mut joff = vec![0usize; nblocks + 1];
                    { let mut a = 0usize; for j in 0..=nblocks { joff[j] = a; a += jcount[j]; } }
                    let (mut rb1, mut rb2, mut rcn) = (vec![0u32; npix], vec![0u32; npix], vec![0i32; npix]);
                    let mut pos = joff.clone();
                    for p in 0..npix {
                        let j = jj[p] as usize;
                        let d = pos[j]; rb1[d] = b1l[p]; rb2[d] = b2l[p]; rcn[d] = cn[base + p]; pos[j] += 1;
                    }
                    // encode non-empty tiles (bands already run in parallel -> encode serially here)
                    let mut out = Vec::new();
                    for j in ibb..nblocks {
                        if jcount[j] == 0 { continue; }
                        let (s, e) = (joff[j], joff[j] + jcount[j]);
                        out.push((j, e - s, compress(&shuffle4(&rb1[s..e])), compress(&shuffle4(&rb2[s..e])), enc_count(&rcn[s..e])));
                    }
                    out
                }).collect()
            });
            for (k, band) in bands.into_iter().enumerate() {
                let ibb = bstart + k;
                for (j, np, b1c, b2c, cnc) in band {
                    tile_i.push(ibb as i32); tile_j.push(j as i32); tile_np.push(np as i64);
                    max_tile = max_tile.max(np);
                    b1_blob.append(&b1c)?; b1_off.push(b1_off.last().unwrap() + b1c.len() as i64);
                    b2_blob.append(&b2c)?; b2_off.push(b2_off.last().unwrap() + b2c.len() as i64);
                    cn_blob.append(&cnc)?; cn_off.push(cn_off.last().unwrap() + cnc.len() as i64);
                }
            }
        }
        let (b1_blob, b2_blob, cn_blob) = (b1_blob.finish()?, b2_blob.finish()?, cn_blob.finish()?);
        Ok(TiledScratch {
            nbins, nnz, chrom_offset, b: block, tile_i, tile_j, tile_np,
            b1_blob, b1_off, b2_blob, b2_off, cn_blob, cn_off, max_tile, nthreads,
        })
    }

    fn decode(&self, t: usize, sh: &mut Vec<u8>, b1: &mut [u32], b2: &mut [u32], cn: &mut [i32]) -> usize {
        let np = self.tile_np[t] as usize;
        let d1 = decompress(&self.b1_blob.bytes()[self.b1_off[t] as usize..self.b1_off[t + 1] as usize], 4 * np).unwrap();
        sh.clear(); sh.extend_from_slice(&d1); unshuffle4(sh, np, &mut b1[..np]);
        let d2 = decompress(&self.b2_blob.bytes()[self.b2_off[t] as usize..self.b2_off[t + 1] as usize], 4 * np).unwrap();
        sh.clear(); sh.extend_from_slice(&d2); unshuffle4(sh, np, &mut b2[..np]);
        dec_count(&self.cn_blob.bytes()[self.cn_off[t] as usize..self.cn_off[t + 1] as usize], np, &mut cn[..np]);
        np
    }
}

impl SpMV for TiledScratch {
    fn nbins(&self) -> usize { self.nbins }
    fn nnz(&self) -> usize { self.nnz }
    fn chrom_offset(&self) -> &[i64] { &self.chrom_offset }
    fn comp_bytes(&self) -> usize { self.b1_blob.len() + self.b2_blob.len() + self.cn_blob.len() }

    fn spmv(&self, v: &[f64], ndiag: i64) -> Vec<f64> {
        let nbins = self.nbins; let ntiles = self.tile_i.len(); let mt = self.max_tile;
        let counter = AtomicUsize::new(0);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(self.nthreads).build().unwrap();
        let parts: Vec<Vec<f64>> = pool.install(|| {
            (0..self.nthreads).into_par_iter().map(|_| {
                let mut ly = vec![0f64; nbins];
                let (mut sh, mut b1, mut b2, mut cn) = (Vec::<u8>::new(), vec![0u32; mt], vec![0u32; mt], vec![0i32; mt]);
                loop {
                    let t = counter.fetch_add(1, Ordering::Relaxed);
                    if t >= ntiles { break; }
                    let np = self.decode(t, &mut sh, &mut b1, &mut b2, &mut cn);
                    let ib = self.tile_i[t] as i64 * self.b;
                    let jb = self.tile_j[t] as i64 * self.b;
                    for p in 0..np { unsafe {
                        let i = (ib + *b1.get_unchecked(p) as i64) as usize;
                        let k = (jb + *b2.get_unchecked(p) as i64) as usize;
                        if (k as i64) - (i as i64) < ndiag { continue; }
                        let c = *cn.get_unchecked(p) as f64;
                        *ly.get_unchecked_mut(i) += c * *v.get_unchecked(k);
                        if k != i { *ly.get_unchecked_mut(k) += c * *v.get_unchecked(i); }
                    }}
                }
                ly
            }).collect()
        });
        let mut out = vec![0f64; nbins];
        for part in &parts { for i in 0..nbins { out[i] += part[i]; } }
        out
    }

    fn marginals(&self, ndiag: i64) -> (Vec<f64>, Vec<f64>) {
        let nbins = self.nbins; let ntiles = self.tile_i.len(); let mt = self.max_tile;
        let counter = AtomicUsize::new(0);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(self.nthreads).build().unwrap();
        let parts: Vec<(Vec<f64>, Vec<f64>)> = pool.install(|| {
            (0..self.nthreads).into_par_iter().map(|_| {
                let (mut mr, mut mn) = (vec![0f64; nbins], vec![0f64; nbins]);
                let (mut sh, mut b1, mut b2, mut cn) = (Vec::<u8>::new(), vec![0u32; mt], vec![0u32; mt], vec![0i32; mt]);
                loop {
                    let t = counter.fetch_add(1, Ordering::Relaxed);
                    if t >= ntiles { break; }
                    let np = self.decode(t, &mut sh, &mut b1, &mut b2, &mut cn);
                    let ib = self.tile_i[t] as i64 * self.b;
                    let jb = self.tile_j[t] as i64 * self.b;
                    for p in 0..np { unsafe {
                        let i = (ib + *b1.get_unchecked(p) as i64) as usize;
                        let k = (jb + *b2.get_unchecked(p) as i64) as usize;
                        if (k as i64) - (i as i64) < ndiag { continue; }
                        let c = *cn.get_unchecked(p) as f64;
                        *mr.get_unchecked_mut(i) += c; *mn.get_unchecked_mut(i) += 1.0;
                        if k != i { *mr.get_unchecked_mut(k) += c; *mn.get_unchecked_mut(k) += 1.0; }
                    }}
                }
                (mr, mn)
            }).collect()
        });
        let (mut mr, mut mn) = (vec![0f64; nbins], vec![0f64; nbins]);
        for (a, b) in &parts { for i in 0..nbins { mr[i] += a[i]; mn[i] += b[i]; } }
        (mr, mn)
    }
}
