//! Compressed row-chunk CSR scratch for fast repeated SpMV (balance). Built once from a cooler;
//! bin2 = within-row-delta u32 + byte-shuffle + LZ4; count = u8 base + u32 exceptions + LZ4.
//! Held in RAM (compressed ~2 B/pixel). Parallel (rayon) marginals/SpMV with per-thread reduction.
use anyhow::Result;
use lz4_flex::block::{compress, decompress};
use rayon::prelude::*;

fn shuffle4(src: &[u32]) -> Vec<u8> {
    let n = src.len();
    let mut o = vec![0u8; 4 * n];
    for i in 0..n {
        let b = src[i].to_le_bytes();
        o[i] = b[0]; o[n + i] = b[1]; o[2 * n + i] = b[2]; o[3 * n + i] = b[3];
    }
    o
}
fn unshuffle4(buf: &[u8], n: usize, out: &mut [u32]) {
    for i in 0..n {
        out[i] = u32::from_le_bytes([buf[i], buf[n + i], buf[2 * n + i], buf[3 * n + i]]);
    }
}
// count codec: [u32 nexc][u32 idx*nexc][u32 val*nexc][lz4(u8 base)]  (base_len = npix known by caller)
fn enc_count(cn: &[i32]) -> Vec<u8> {
    let n = cn.len();
    let mut idx = Vec::new(); let mut val = Vec::new();
    let mut base = vec![0u8; n];
    for i in 0..n {
        let v = cn[i] as u32;
        if v > 255 { idx.push(i as u32); val.push(v); } else { base[i] = v as u8; }
    }
    let comp = compress(&base);
    let mut out = Vec::with_capacity(12 + 8 * idx.len() + comp.len());
    out.extend_from_slice(&(idx.len() as u32).to_le_bytes());
    for &x in &idx { out.extend_from_slice(&x.to_le_bytes()); }
    for &x in &val { out.extend_from_slice(&x.to_le_bytes()); }
    out.extend_from_slice(&comp);
    out
}
fn dec_count(buf: &[u8], npix: usize, out: &mut [i32]) {
    let nexc = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let mut p = 4;
    let idx_end = p + 4 * nexc; let val_end = idx_end + 4 * nexc;
    let base = decompress(&buf[val_end..], npix).unwrap();
    for i in 0..npix { out[i] = base[i] as i32; }
    for k in 0..nexc {
        let i = u32::from_le_bytes(buf[p + 4 * k..p + 4 * k + 4].try_into().unwrap()) as usize;
        let v = u32::from_le_bytes(buf[idx_end + 4 * k..idx_end + 4 * k + 4].try_into().unwrap()) as i32;
        out[i] = v;
    }
    let _ = p; p = val_end; let _ = p;
}

pub struct Scratch {
    pub nbins: usize,
    pub nnz: usize,
    pub chrom_offset: Vec<i64>,
    rowptr: Vec<i64>,
    chunk_row: Vec<i64>,
    b2_blob: Vec<u8>, b2_off: Vec<i64>,
    cn_blob: Vec<u8>, cn_off: Vec<i64>,
    max_chunk_pix: usize,
    pub nthreads: usize,
}

impl Scratch {
    pub fn build(g: &hdf5::Group, chunk_target: usize, nthreads: usize) -> Result<Scratch> {
        let nbins = g.dataset("bins/start")?.shape()[0];
        let nnz = g.dataset("pixels/count")?.shape()[0];
        let mut rowptr: Vec<i64> = g.dataset("indexes/bin1_offset")?.read_1d::<i64>()?.to_vec();
        if rowptr.len() == nbins { rowptr.push(nnz as i64); }
        let chrom_offset: Vec<i64> = g.dataset("indexes/chrom_offset")?.read_1d::<i64>()?.to_vec();
        // row-aligned chunk plan
        let mut chunk_row = vec![0i64];
        let mut acc = 0i64;
        for r in 0..nbins {
            acc += rowptr[r + 1] - rowptr[r];
            if acc >= chunk_target as i64 { chunk_row.push((r + 1) as i64); acc = 0; }
        }
        if *chunk_row.last().unwrap() != nbins as i64 { chunk_row.push(nbins as i64); }
        let nchunks = chunk_row.len() - 1;
        let b2d = g.dataset("pixels/bin2_id")?;
        let cnd = g.dataset("pixels/count")?;
        let (mut b2_blob, mut cn_blob) = (Vec::new(), Vec::new());
        let (mut b2_off, mut cn_off) = (vec![0i64], vec![0i64]);
        let mut max_chunk_pix = 1usize;
        for c in 0..nchunks {
            let (r0, r1) = (chunk_row[c] as usize, chunk_row[c + 1] as usize);
            let (p0, p1) = (rowptr[r0], rowptr[r1]);
            let npix = (p1 - p0) as usize;
            if npix == 0 { b2_off.push(*b2_off.last().unwrap()); cn_off.push(*cn_off.last().unwrap()); continue; }
            max_chunk_pix = max_chunk_pix.max(npix);
            let b2 = b2d.read_slice_1d::<i64, _>(p0 as usize..p1 as usize)?;
            let cn = cnd.read_slice_1d::<i32, _>(p0 as usize..p1 as usize)?;
            // within-row delta of bin2 (first-of-row = bin2 - rowindex), u32
            let mut d = vec![0u32; npix];
            for r in r0..r1 {
                let (s, e) = ((rowptr[r] - p0) as usize, (rowptr[r + 1] - p0) as usize);
                if s >= e { continue; }
                let mut prev = r as i64;
                for j in s..e { d[j] = (b2[j] - prev) as u32; prev = b2[j]; }
            }
            let comp = compress(&shuffle4(&d));
            b2_blob.extend_from_slice(&comp); b2_off.push(b2_off.last().unwrap() + comp.len() as i64);
            let cc = enc_count(cn.as_slice().unwrap());
            cn_blob.extend_from_slice(&cc); cn_off.push(cn_off.last().unwrap() + cc.len() as i64);
        }
        Ok(Scratch { nbins, nnz, chrom_offset, rowptr, chunk_row, b2_blob, b2_off, cn_blob, cn_off, max_chunk_pix, nthreads })
    }

    pub fn comp_bytes(&self) -> usize { self.b2_blob.len() + self.cn_blob.len() }

    fn decode_chunk(&self, c: usize, sh: &mut Vec<u8>, b2: &mut [u32], cn: &mut [i32]) -> (usize, usize, usize) {
        let (r0, r1) = (self.chunk_row[c] as usize, self.chunk_row[c + 1] as usize);
        let p0 = self.rowptr[r0];
        let npix = (self.rowptr[r1] - p0) as usize;
        if npix == 0 { return (r0, r1, 0); }
        let src = &self.b2_blob[self.b2_off[c] as usize..self.b2_off[c + 1] as usize];
        let raw = decompress(src, 4 * npix).unwrap();
        sh.clear(); sh.extend_from_slice(&raw);
        unshuffle4(sh, npix, &mut b2[..npix]);
        // prefix-sum per row -> absolute bin2
        for r in r0..r1 {
            let (s, e) = ((self.rowptr[r] - p0) as usize, (self.rowptr[r + 1] - p0) as usize);
            let mut accv = r as u64;
            for j in s..e { accv += b2[j] as u64; b2[j] = accv as u32; }
        }
        dec_count(&self.cn_blob[self.cn_off[c] as usize..self.cn_off[c + 1] as usize], npix, &mut cn[..npix]);
        (r0, r1, npix)
    }

    /// symmetric SpMV: y[i] = sum_j A_ij v[j], diag-zeroed by ndiag. Parallel + per-thread reduction.
    pub fn spmv(&self, v: &[f64], ndiag: i64) -> Vec<f64> {
        let nbins = self.nbins;
        let nchunks = self.chunk_row.len() - 1;
        let mcp = self.max_chunk_pix;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(self.nthreads).build().unwrap();
        pool.install(|| {
            (0..nchunks).into_par_iter()
                .fold(|| (vec![0f64; nbins], Vec::<u8>::new(), vec![0u32; mcp], vec![0i32; mcp]),
                    |(mut ly, mut sh, mut b2, mut cn), c| {
                        let (r0, r1, npix) = self.decode_chunk(c, &mut sh, &mut b2, &mut cn);
                        if npix > 0 {
                            let p0 = self.rowptr[r0];
                            for r in r0..r1 {
                                let (s, e) = ((self.rowptr[r] - p0) as usize, (self.rowptr[r + 1] - p0) as usize);
                                let vi = v[r]; let mut acc = 0.0;
                                for j in s..e {
                                    let k = b2[j] as usize;
                                    if (k as i64) - (r as i64) < ndiag { continue; }
                                    let cc = cn[j] as f64;
                                    acc += cc * v[k];
                                    if k != r { ly[k] += cc * vi; }
                                }
                                ly[r] += acc;
                            }
                        }
                        (ly, sh, b2, cn)
                    })
                .map(|(ly, ..)| ly)
                .reduce(|| vec![0f64; nbins], |mut a, b| { for i in 0..nbins { a[i] += b[i]; } a })
        })
    }

    /// raw marginal (sum of counts) and nnz marginal, both symmetric + diag-zeroed.
    pub fn marginals(&self, ndiag: i64) -> (Vec<f64>, Vec<f64>) {
        let nbins = self.nbins;
        let nchunks = self.chunk_row.len() - 1;
        let mcp = self.max_chunk_pix;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(self.nthreads).build().unwrap();
        pool.install(|| {
            (0..nchunks).into_par_iter()
                .fold(|| (vec![0f64; nbins], vec![0f64; nbins], Vec::<u8>::new(), vec![0u32; mcp], vec![0i32; mcp]),
                    |(mut mr, mut mn, mut sh, mut b2, mut cn), c| {
                        let (r0, r1, npix) = self.decode_chunk(c, &mut sh, &mut b2, &mut cn);
                        if npix > 0 {
                            let p0 = self.rowptr[r0];
                            for r in r0..r1 {
                                let (s, e) = ((self.rowptr[r] - p0) as usize, (self.rowptr[r + 1] - p0) as usize);
                                for j in s..e {
                                    let k = b2[j] as usize;
                                    if (k as i64) - (r as i64) < ndiag { continue; }
                                    let cc = cn[j] as f64;
                                    mr[r] += cc; mn[r] += 1.0;
                                    if k != r { mr[k] += cc; mn[k] += 1.0; }
                                }
                            }
                        }
                        (mr, mn, sh, b2, cn)
                    })
                .map(|(mr, mn, ..)| (mr, mn))
                .reduce(|| (vec![0f64; nbins], vec![0f64; nbins]),
                    |mut a, b| { for i in 0..nbins { a.0[i] += b.0[i]; a.1[i] += b.1[i]; } a })
        })
    }
}
