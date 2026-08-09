//! Compressed row-chunk CSR scratch for fast repeated SpMV (balance). Built once from a cooler;
//! bin2 = within-row-delta u32 + byte-shuffle + LZ4; count = u8 base + u32 exceptions + LZ4.
//! Held in RAM (compressed ~2 B/pixel). Parallel (rayon) marginals/SpMV with per-thread reduction.
use anyhow::Result;
use lz4_flex::block::{compress, decompress};
use rayon::prelude::*;

pub fn shuffle4(src: &[u32]) -> Vec<u8> {
    let n = src.len();
    let mut o = vec![0u8; 4 * n];
    for i in 0..n {
        let b = src[i].to_le_bytes();
        o[i] = b[0]; o[n + i] = b[1]; o[2 * n + i] = b[2]; o[3 * n + i] = b[3];
    }
    o
}
pub fn unshuffle4(buf: &[u8], n: usize, out: &mut [u32]) {
    // safety: caller guarantees buf.len() >= 4*n and out.len() >= n
    for i in 0..n { unsafe {
        *out.get_unchecked_mut(i) = u32::from_le_bytes([
            *buf.get_unchecked(i), *buf.get_unchecked(n + i),
            *buf.get_unchecked(2 * n + i), *buf.get_unchecked(3 * n + i)]);
    }}
}
// count codec: [u32 nexc][u32 idx*nexc][u32 val*nexc][lz4(u8 base)]  (base_len = npix known by caller)
pub fn enc_count(cn: &[i32]) -> Vec<u8> {
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
pub fn dec_count(buf: &[u8], npix: usize, out: &mut [i32]) {
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
        let max_chunk_pix = (0..nchunks).map(|c| (rowptr[chunk_row[c + 1] as usize] - rowptr[chunk_row[c] as usize]) as usize).max().unwrap_or(1).max(1);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(nthreads).build().unwrap();
        let read_block = 50_000_000usize; // pixels per HDF5 read batch
        let mut c = 0usize;
        while c < nchunks {
            // gather a batch of whole chunks (~read_block pixels), read once, encode in parallel
            let cstart = c;
            let mut pix = 0i64;
            while c < nchunks && (pix as usize) < read_block { pix += rowptr[chunk_row[c + 1] as usize] - rowptr[chunk_row[c] as usize]; c += 1; }
            let cend = c;
            let (bp0, bp1) = (rowptr[chunk_row[cstart] as usize], rowptr[chunk_row[cend] as usize]);
            if bp1 <= bp0 { for _ in cstart..cend { b2_off.push(*b2_off.last().unwrap()); cn_off.push(*cn_off.last().unwrap()); } continue; }
            let b2blk = b2d.read_slice_1d::<i64, _>(bp0 as usize..bp1 as usize)?;
            let cnblk = cnd.read_slice_1d::<i32, _>(bp0 as usize..bp1 as usize)?;
            let (b2s, cns) = (b2blk.as_slice().unwrap(), cnblk.as_slice().unwrap());
            let rp = &rowptr;
            let cr = &chunk_row;
            let encoded: Vec<(Vec<u8>, Vec<u8>)> = pool.install(|| {
                (cstart..cend).into_par_iter().map(|cc| {
                    let (r0, r1) = (cr[cc] as usize, cr[cc + 1] as usize);
                    let (p0, p1) = (rp[r0], rp[r1]);
                    let npix = (p1 - p0) as usize;
                    if npix == 0 { return (Vec::new(), Vec::new()); }
                    let off = bp0;
                    let mut d = vec![0u32; npix];
                    for r in r0..r1 {
                        let (s, e) = ((rp[r] - p0) as usize, (rp[r + 1] - p0) as usize);
                        if s >= e { continue; }
                        let mut prev = r as i64;
                        for j in s..e { let bv = b2s[(p0 - off) as usize + j]; d[j] = (bv - prev) as u32; prev = bv; }
                    }
                    let cslice = &cns[(p0 - off) as usize..(p1 - off) as usize];
                    (compress(&shuffle4(&d)), enc_count(cslice))
                }).collect()
            });
            for (b2c, cnc) in encoded {
                b2_blob.extend_from_slice(&b2c); b2_off.push(b2_off.last().unwrap() + b2c.len() as i64);
                cn_blob.extend_from_slice(&cnc); cn_off.push(cn_off.last().unwrap() + cnc.len() as i64);
            }
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
        let counter = std::sync::atomic::AtomicUsize::new(0);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(self.nthreads).build().unwrap();
        let parts: Vec<Vec<f64>> = pool.install(|| {
            (0..self.nthreads).into_par_iter().map(|_| {
                let mut ly = vec![0f64; nbins];
                let (mut sh, mut b2, mut cn) = (Vec::<u8>::new(), vec![0u32; mcp], vec![0i32; mcp]);
                loop {
                    let c = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if c >= nchunks { break; }
                    let (r0, r1, npix) = self.decode_chunk(c, &mut sh, &mut b2, &mut cn);
                    if npix == 0 { continue; }
                    let p0 = self.rowptr[r0];
                    for r in r0..r1 {
                        let (s, e) = ((self.rowptr[r] - p0) as usize, (self.rowptr[r + 1] - p0) as usize);
                        let vi = unsafe { *v.get_unchecked(r) }; let mut acc = 0.0;
                        for j in s..e {
                            let k = unsafe { *b2.get_unchecked(j) } as usize; // k=bin2 < nbins
                            if (k as i64) - (r as i64) < ndiag { continue; }
                            let cc = unsafe { *cn.get_unchecked(j) } as f64;
                            acc += cc * unsafe { *v.get_unchecked(k) };
                            if k != r { unsafe { *ly.get_unchecked_mut(k) += cc * vi; } }
                        }
                        unsafe { *ly.get_unchecked_mut(r) += acc; }
                    }
                }
                ly
            }).collect()
        });
        let mut out = vec![0f64; nbins];
        for part in &parts { for i in 0..nbins { out[i] += part[i]; } }
        out
    }

    /// raw marginal (sum of counts) and nnz marginal, both symmetric + diag-zeroed.
    pub fn marginals(&self, ndiag: i64) -> (Vec<f64>, Vec<f64>) {
        let nbins = self.nbins;
        let nchunks = self.chunk_row.len() - 1;
        let mcp = self.max_chunk_pix;
        let counter = std::sync::atomic::AtomicUsize::new(0);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(self.nthreads).build().unwrap();
        let parts: Vec<(Vec<f64>, Vec<f64>)> = pool.install(|| {
            (0..self.nthreads).into_par_iter().map(|_| {
                let (mut mr, mut mn) = (vec![0f64; nbins], vec![0f64; nbins]);
                let (mut sh, mut b2, mut cn) = (Vec::<u8>::new(), vec![0u32; mcp], vec![0i32; mcp]);
                loop {
                    let c = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if c >= nchunks { break; }
                    let (r0, r1, npix) = self.decode_chunk(c, &mut sh, &mut b2, &mut cn);
                    if npix == 0 { continue; }
                    let p0 = self.rowptr[r0];
                    for r in r0..r1 {
                        let (s, e) = ((self.rowptr[r] - p0) as usize, (self.rowptr[r + 1] - p0) as usize);
                        for j in s..e {
                            let k = unsafe { *b2.get_unchecked(j) } as usize;
                            if (k as i64) - (r as i64) < ndiag { continue; }
                            let cc = unsafe { *cn.get_unchecked(j) } as f64;
                            unsafe { *mr.get_unchecked_mut(r) += cc; *mn.get_unchecked_mut(r) += 1.0;
                                if k != r { *mr.get_unchecked_mut(k) += cc; *mn.get_unchecked_mut(k) += 1.0; } }
                        }
                    }
                }
                (mr, mn)
            }).collect()
        });
        let (mut mr, mut mn) = (vec![0f64; nbins], vec![0f64; nbins]);
        for (a, b) in &parts { for i in 0..nbins { mr[i] += a[i]; mn[i] += b[i]; } }
        (mr, mn)
    }
}

/// Common interface so balance can run over either the row-chunk or the tiled scratch.
pub trait SpMV: Sync {
    fn nbins(&self) -> usize;
    fn nnz(&self) -> usize;
    fn chrom_offset(&self) -> &[i64];
    fn comp_bytes(&self) -> usize;
    fn marginals(&self, ndiag: i64) -> (Vec<f64>, Vec<f64>);
    fn spmv(&self, v: &[f64], ndiag: i64) -> Vec<f64>;
}
impl SpMV for Scratch {
    fn nbins(&self) -> usize { self.nbins }
    fn nnz(&self) -> usize { self.nnz }
    fn chrom_offset(&self) -> &[i64] { &self.chrom_offset }
    fn comp_bytes(&self) -> usize { Scratch::comp_bytes(self) }
    fn marginals(&self, ndiag: i64) -> (Vec<f64>, Vec<f64>) { Scratch::marginals(self, ndiag) }
    fn spmv(&self, v: &[f64], ndiag: i64) -> Vec<f64> { Scratch::spmv(self, v, ndiag) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// deterministic pseudo-random source (no rand dependency; reproducible failures)
    pub struct Lcg(pub u64);
    impl Lcg {
        pub fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        pub fn below(&mut self, n: u64) -> u64 { self.next() % n }
    }

    fn roundtrip4(v: &[u32]) {
        let sh = shuffle4(v);
        assert_eq!(sh.len(), 4 * v.len());
        let mut out = vec![0u32; v.len()];
        unshuffle4(&sh, v.len(), &mut out);
        assert_eq!(out, v, "shuffle4 round-trip");
        // and through LZ4, the way the scratch actually stores it
        let comp = lz4_flex::block::compress(&sh);
        let raw = lz4_flex::block::decompress(&comp, 4 * v.len()).unwrap();
        let mut out2 = vec![0u32; v.len()];
        unshuffle4(&raw, v.len(), &mut out2);
        assert_eq!(out2, v, "shuffle4+lz4 round-trip");
    }

    #[test]
    fn shuffle4_roundtrips() {
        roundtrip4(&[7]);
        roundtrip4(&[0, 1, 255, 256, 65535, 65536, u32::MAX]);
        let mut g = Lcg(12345);
        let v: Vec<u32> = (0..10_000).map(|_| g.next() as u32).collect();
        roundtrip4(&v);
        // small deltas (the realistic within-row-delta case) must compress
        let d: Vec<u32> = (0..10_000).map(|_| g.below(64) as u32).collect();
        roundtrip4(&d);
        assert!(lz4_flex::block::compress(&shuffle4(&d)).len() < 4 * d.len() / 2,
            "byte-shuffled small deltas should compress >2x");
    }

    fn roundtrip_count(v: &[i32]) {
        let enc = enc_count(v);
        let mut out = vec![0i32; v.len()];
        dec_count(&enc, v.len(), &mut out);
        assert_eq!(out, v, "count codec round-trip");
    }

    #[test]
    fn count_codec_roundtrips() {
        roundtrip_count(&[1]);
        // the u8-base / u32-exception boundary
        roundtrip_count(&[0, 1, 254, 255, 256, 257, 1000, i32::MAX]);
        roundtrip_count(&vec![3i32; 5000]);                       // all small
        roundtrip_count(&vec![100_000i32; 500]);                  // all exceptions
        let mut g = Lcg(999);
        // realistic mix: mostly small, ~1% exceptions
        let v: Vec<i32> = (0..20_000)
            .map(|_| if g.below(100) == 0 { 256 + g.below(1_000_000) as i32 } else { g.below(256) as i32 })
            .collect();
        roundtrip_count(&v);
    }

    #[test]
    fn count_codec_is_compact_for_small_counts() {
        let v = vec![1i32; 100_000];
        assert!(enc_count(&v).len() < 100_000 / 10, "all-ones counts should compress >10x");
    }
}
