//! Uniform 2D-tiled compressed scratch for cache-blocked SpMV (large/fine coolers). Bins split into
//! blocks of size B; tile (I,J) [I<=J] holds pixels with bin1 in block I, bin2 in block J. Pixels
//! store block-LOCAL bin1/bin2 (u32, shuffle+lz4) + u8-exc count. In the SpMV all v/y access is
//! confined to the I- and J-block slices (<=B elems each -> L2-resident) -> kills the cache-miss wall
//! at 12M bins. Same result as the row-chunk scratch. Build: counting-sort each I-block band by J,
//! encode tiles in parallel.
use crate::scratch::{dec_count, enc_count, shuffle4, unshuffle4, SpMV};
use anyhow::Result;
use lz4_flex::block::{compress, decompress};
use rayon::prelude::*;

pub struct TiledScratch {
    pub nbins: usize,
    pub nnz: usize,
    pub chrom_offset: Vec<i64>,
    b: i64,
    tile_i: Vec<i32>,
    tile_j: Vec<i32>,
    tile_np: Vec<i64>,
    b1_blob: Vec<u8>, b1_off: Vec<i64>,
    b2_blob: Vec<u8>, b2_off: Vec<i64>,
    cn_blob: Vec<u8>, cn_off: Vec<i64>,
    max_tile: usize,
    nthreads: usize,
}

impl TiledScratch {
    pub fn build(g: &hdf5::Group, block: i64, nthreads: usize) -> Result<TiledScratch> {
        let nbins = g.dataset("bins/start")?.shape()[0];
        let nnz = g.dataset("pixels/count")?.shape()[0];
        let mut rowptr: Vec<i64> = g.dataset("indexes/bin1_offset")?.read_1d::<i64>()?.to_vec();
        if rowptr.len() == nbins { rowptr.push(nnz as i64); }
        let chrom_offset: Vec<i64> = g.dataset("indexes/chrom_offset")?.read_1d::<i64>()?.to_vec();
        let nblocks = ((nbins as i64 + block - 1) / block) as usize;
        let b2d = g.dataset("pixels/bin2_id")?;
        let cnd = g.dataset("pixels/count")?;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(nthreads).build().unwrap();

        let (mut tile_i, mut tile_j, mut tile_np) = (Vec::new(), Vec::new(), Vec::new());
        let (mut b1_blob, mut b2_blob, mut cn_blob) = (Vec::new(), Vec::new(), Vec::new());
        let (mut b1_off, mut b2_off, mut cn_off) = (vec![0i64], vec![0i64], vec![0i64]);
        let mut max_tile = 1usize;

        for ib in 0..nblocks {
            let r0 = ib * (block as usize);
            let r1 = ((ib + 1) * block as usize).min(nbins);
            let (p0, p1) = (rowptr[r0], rowptr[r1]);
            let npix = (p1 - p0) as usize;
            if npix == 0 { continue; }
            let bin2 = b2d.read_slice_1d::<i64, _>(p0 as usize..p1 as usize)?;
            let cn = cnd.read_slice_1d::<i32, _>(p0 as usize..p1 as usize)?;
            let (bin2, cn) = (bin2.as_slice().unwrap(), cn.as_slice().unwrap());
            // per-pixel bin1-local, J block, bin2-local
            let mut b1l = vec![0u32; npix];
            let mut jj = vec![0u32; npix];
            let mut b2l = vec![0u32; npix];
            {
                let mut idx = 0usize;
                for r in r0..r1 {
                    let c = (rowptr[r + 1] - rowptr[r]) as usize;
                    let rl = (r - r0) as u32;
                    for _ in 0..c {
                        let bv = bin2[idx];
                        let j = (bv / block) as u32;
                        b1l[idx] = rl; jj[idx] = j; b2l[idx] = (bv - (j as i64) * block) as u32;
                        idx += 1;
                    }
                }
            }
            // counting-sort by J (J in [ib, nblocks)) -> contiguous per-tile buckets
            let mut jcount = vec![0usize; nblocks + 1];
            for &j in &jj { jcount[j as usize] += 1; }
            let mut joff = vec![0usize; nblocks + 1];
            { let mut a = 0usize; for j in 0..=nblocks { joff[j] = a; a += jcount[j]; } }
            let (mut rb1, mut rb2, mut rcn) = (vec![0u32; npix], vec![0u32; npix], vec![0i32; npix]);
            let mut pos = joff.clone();
            for p in 0..npix {
                let j = jj[p] as usize;
                let d = pos[j]; rb1[d] = b1l[p]; rb2[d] = b2l[p]; rcn[d] = cn[p]; pos[j] += 1;
            }
            // encode non-empty tiles (I, j) in parallel
            let jlist: Vec<usize> = (ib..nblocks).filter(|&j| jcount[j] > 0).collect();
            let tiles: Vec<(usize, usize, Vec<u8>, Vec<u8>, Vec<u8>)> = pool.install(|| {
                jlist.par_iter().map(|&j| {
                    let (s, e) = (joff[j], joff[j] + jcount[j]);
                    (j, e - s, compress(&shuffle4(&rb1[s..e])), compress(&shuffle4(&rb2[s..e])), enc_count(&rcn[s..e]))
                }).collect()
            });
            for (j, np, b1c, b2c, cnc) in tiles {
                tile_i.push(ib as i32); tile_j.push(j as i32); tile_np.push(np as i64);
                max_tile = max_tile.max(np);
                b1_blob.extend_from_slice(&b1c); b1_off.push(b1_off.last().unwrap() + b1c.len() as i64);
                b2_blob.extend_from_slice(&b2c); b2_off.push(b2_off.last().unwrap() + b2c.len() as i64);
                cn_blob.extend_from_slice(&cnc); cn_off.push(cn_off.last().unwrap() + cnc.len() as i64);
            }
        }
        Ok(TiledScratch {
            nbins, nnz, chrom_offset, b: block, tile_i, tile_j, tile_np,
            b1_blob, b1_off, b2_blob, b2_off, cn_blob, cn_off, max_tile, nthreads,
        })
    }

    fn decode(&self, t: usize, sh: &mut Vec<u8>, b1: &mut [u32], b2: &mut [u32], cn: &mut [i32]) -> usize {
        let np = self.tile_np[t] as usize;
        let d1 = decompress(&self.b1_blob[self.b1_off[t] as usize..self.b1_off[t + 1] as usize], 4 * np).unwrap();
        sh.clear(); sh.extend_from_slice(&d1); unshuffle4(sh, np, &mut b1[..np]);
        let d2 = decompress(&self.b2_blob[self.b2_off[t] as usize..self.b2_off[t + 1] as usize], 4 * np).unwrap();
        sh.clear(); sh.extend_from_slice(&d2); unshuffle4(sh, np, &mut b2[..np]);
        dec_count(&self.cn_blob[self.cn_off[t] as usize..self.cn_off[t + 1] as usize], np, &mut cn[..np]);
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
        let pool = rayon::ThreadPoolBuilder::new().num_threads(self.nthreads).build().unwrap();
        pool.install(|| {
            (0..ntiles).into_par_iter()
                .fold(|| (vec![0f64; nbins], Vec::<u8>::new(), vec![0u32; mt], vec![0u32; mt], vec![0i32; mt]),
                    |(mut ly, mut sh, mut b1, mut b2, mut cn), t| {
                        let np = self.decode(t, &mut sh, &mut b1, &mut b2, &mut cn);
                        let ib = self.tile_i[t] as i64 * self.b;
                        let jb = self.tile_j[t] as i64 * self.b;
                        for p in 0..np {
                            let i = (ib + b1[p] as i64) as usize;
                            let k = (jb + b2[p] as i64) as usize;
                            if (k as i64) - (i as i64) < ndiag { continue; }
                            let c = cn[p] as f64;
                            ly[i] += c * v[k];
                            if k != i { ly[k] += c * v[i]; }
                        }
                        (ly, sh, b1, b2, cn)
                    })
                .map(|(ly, ..)| ly)
                .reduce(|| vec![0f64; nbins], |mut a, b| { for i in 0..nbins { a[i] += b[i]; } a })
        })
    }

    fn marginals(&self, ndiag: i64) -> (Vec<f64>, Vec<f64>) {
        let nbins = self.nbins; let ntiles = self.tile_i.len(); let mt = self.max_tile;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(self.nthreads).build().unwrap();
        pool.install(|| {
            (0..ntiles).into_par_iter()
                .fold(|| (vec![0f64; nbins], vec![0f64; nbins], Vec::<u8>::new(), vec![0u32; mt], vec![0u32; mt], vec![0i32; mt]),
                    |(mut mr, mut mn, mut sh, mut b1, mut b2, mut cn), t| {
                        let np = self.decode(t, &mut sh, &mut b1, &mut b2, &mut cn);
                        let ib = self.tile_i[t] as i64 * self.b;
                        let jb = self.tile_j[t] as i64 * self.b;
                        for p in 0..np {
                            let i = (ib + b1[p] as i64) as usize;
                            let k = (jb + b2[p] as i64) as usize;
                            if (k as i64) - (i as i64) < ndiag { continue; }
                            let c = cn[p] as f64;
                            mr[i] += c; mn[i] += 1.0;
                            if k != i { mr[k] += c; mn[k] += 1.0; }
                        }
                        (mr, mn, sh, b1, b2, cn)
                    })
                .map(|(mr, mn, ..)| (mr, mn))
                .reduce(|| (vec![0f64; nbins], vec![0f64; nbins]),
                    |mut a, b| { for i in 0..nbins { a.0[i] += b.0[i]; a.1[i] += b.1[i]; } a })
        })
    }
}
