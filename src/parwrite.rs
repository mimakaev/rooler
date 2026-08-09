//! Parallel chunked writer for the gzip (`--compat`) preset.
//!
//! HDF5's gzip filter compresses each chunk as an *independent* zlib stream, so we can do the
//! shuffle+deflate ourselves on rayon threads and hand HDF5 the finished bytes via
//! `H5Dwrite_chunk` (the sanctioned direct-chunk API, = h5py's `write_direct_chunk`). The
//! resulting file is ordinary SHUFFLE+DEFLATE HDF5 that vanilla cooler/h5py read with no
//! plugins — we only move deflate off HDF5's single thread. Measured ~8.6x on the prototype.
//!
//! HDF5 calls all happen on the caller's thread; only compression is parallel.
use anyhow::{bail, Result};
use hdf5_metno_sys::h5d::H5Dwrite_chunk;
use hdf5_metno_sys::h5i::hid_t;
use libdeflater::{CompressionLvl, Compressor};
use rayon::prelude::*;

/// How many full chunks to accumulate before compressing a batch in parallel.
const BATCH_CHUNKS: usize = 16;

/// Byte-transpose exactly like HDF5's SHUFFLE filter: element e's byte b moves to `b*n + e`.
/// `src.len()` must be `n * esize`.
fn shuffle(src: &[u8], esize: usize) -> Vec<u8> {
    let n = src.len() / esize;
    let mut out = vec![0u8; src.len()];
    for b in 0..esize {
        let dst = &mut out[b * n..(b + 1) * n];
        for e in 0..n {
            dst[e] = src[e * esize + b];
        }
    }
    out
}

/// zlib-wrapped deflate (windowBits 15, 0x78 header) — what the HDF5 gzip filter emits/expects.
/// Byte-identity with HDF5's own output is NOT required: any valid zlib stream inflates fine.
fn zlib(src: &[u8], level: i32) -> Vec<u8> {
    let mut c = Compressor::new(CompressionLvl::new(level).unwrap_or(CompressionLvl::default()));
    let mut out = vec![0u8; c.zlib_compress_bound(src.len())];
    let n = c.zlib_compress(src, &mut out).expect("zlib_compress into a bound-sized buffer");
    out.truncate(n);
    out
}

/// One extendible 1-D chunked column, written chunk-at-a-time with parallel compression.
pub struct ParColumn {
    dset: hdf5::Dataset,
    esize: usize,   // bytes per element
    chunk: usize,   // elements per chunk
    level: i32,
    staged: Vec<u8>,        // raw little-endian bytes not yet written
    chunks_written: usize,
    elems_written: usize,   // elements already in the dataset (= chunks_written * chunk)
    finished: bool,
}

impl ParColumn {
    /// `dset` must be 1-D, extendible, chunked with exactly `chunk` elements and a
    /// SHUFFLE+DEFLATE filter pipeline (that is what we reproduce byte-wise).
    pub fn new(dset: hdf5::Dataset, esize: usize, chunk: usize, level: u8) -> ParColumn {
        ParColumn { dset, esize, chunk, level: level as i32, staged: Vec::new(),
                    chunks_written: 0, elems_written: 0, finished: false }
    }

    pub fn push_i64(&mut self, v: &[i64]) -> Result<()> {
        self.staged.reserve(v.len() * 8);
        for &x in v { self.staged.extend_from_slice(&x.to_le_bytes()); }
        self.maybe_flush()
    }

    pub fn push_i32(&mut self, v: &[i32]) -> Result<()> {
        self.staged.reserve(v.len() * 4);
        for &x in v { self.staged.extend_from_slice(&x.to_le_bytes()); }
        self.maybe_flush()
    }

    fn maybe_flush(&mut self) -> Result<()> {
        let cbytes = self.chunk * self.esize;
        if self.staged.len() >= BATCH_CHUNKS * cbytes { self.flush_full_chunks()?; }
        Ok(())
    }

    /// Compress every *complete* chunk currently staged (in parallel) and write them in order.
    fn flush_full_chunks(&mut self) -> Result<()> {
        let cbytes = self.chunk * self.esize;
        let nfull = self.staged.len() / cbytes;
        if nfull == 0 { return Ok(()); }
        let (esize, level) = (self.esize, self.level);
        let blobs: Vec<Vec<u8>> = self.staged[..nfull * cbytes]
            .par_chunks(cbytes)
            .map(|raw| zlib(&shuffle(raw, esize), level))
            .collect();
        self.write_blobs(&blobs, nfull * self.chunk)?;
        self.staged.drain(..nfull * cbytes);
        Ok(())
    }

    /// Extend the dataset then write the pre-compressed chunks. HDF5 calls, this thread only.
    fn write_blobs(&mut self, blobs: &[Vec<u8>], new_elems: usize) -> Result<()> {
        self.dset.resize([self.elems_written + new_elems])?;
        self.elems_written += new_elems;
        for blob in blobs {
            let offset = [(self.chunks_written * self.chunk) as u64];
            // filters = 0: no filter in the pipeline was skipped for this chunk
            let rc = unsafe {
                H5Dwrite_chunk(self.dset.id() as hid_t, 0, 0, offset.as_ptr(),
                               blob.len(), blob.as_ptr().cast())
            };
            if rc < 0 { bail!("H5Dwrite_chunk failed at chunk {}", self.chunks_written); }
            self.chunks_written += 1;
        }
        Ok(())
    }

    /// Flush the tail: set the exact extent, then write the final partial chunk zero-padded to
    /// full chunk size (HDF5 stores whole chunks; the pad lies past the extent and is never read).
    pub fn finish(&mut self) -> Result<()> {
        if self.finished { return Ok(()); }
        self.finished = true;
        self.flush_full_chunks()?;
        if self.staged.is_empty() {
            self.dset.resize([self.elems_written])?;
            return Ok(());
        }
        let cbytes = self.chunk * self.esize;
        let tail_elems = self.staged.len() / self.esize;
        let mut raw = std::mem::take(&mut self.staged);
        raw.resize(cbytes, 0);
        let blob = zlib(&shuffle(&raw, self.esize), self.level);
        // exact extent first, then write the full-size edge chunk (h5py's write_direct_chunk pattern)
        self.dset.resize([self.elems_written + tail_elems])?;
        self.elems_written += tail_elems;
        let offset = [(self.chunks_written * self.chunk) as u64];
        let rc = unsafe {
            H5Dwrite_chunk(self.dset.id() as hid_t, 0, 0, offset.as_ptr(),
                           blob.len(), blob.as_ptr().cast())
        };
        if rc < 0 { bail!("H5Dwrite_chunk failed on the final partial chunk"); }
        self.chunks_written += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("rooler_parwrite_{}_{}", tag, std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn shuffle_matches_hdf5_byte_transpose() {
        // 2 elements of 4 bytes: [00 01 02 03][10 11 12 13] -> byte planes
        let src: Vec<u8> = vec![0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13];
        assert_eq!(shuffle(&src, 4), vec![0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13]);
    }

    /// The load-bearing test: chunks we compress ourselves must be readable through the
    /// ordinary HDF5 API, which proves the direct-chunk write reproduces the filter pipeline.
    #[test]
    fn direct_chunk_write_roundtrips_through_normal_api() {
        let d = tmpdir("smoke");
        let p = d.join("t.h5");
        let n = 3000usize;
        let chunk = 1024usize;
        let data: Vec<i32> = (0..n as i32).map(|i| i * 7 - 13).collect();
        {
            let f = hdf5::File::create(p.to_str().unwrap()).unwrap();
            let ds = f.new_dataset::<i32>().shape((0..,)).chunk([chunk])
                .shuffle().deflate(4).create("x").unwrap();
            let mut col = ParColumn::new(ds, 4, chunk, 4);
            // push in awkward slices so chunk boundaries fall mid-push
            col.push_i32(&data[..100]).unwrap();
            col.push_i32(&data[100..2500]).unwrap();
            col.push_i32(&data[2500..]).unwrap();
            col.finish().unwrap();
        }
        let f = hdf5::File::open(p.to_str().unwrap()).unwrap();
        let ds = f.dataset("x").unwrap();
        assert_eq!(ds.shape(), vec![n], "extent must be the exact element count, not padded");
        assert_eq!(ds.read_1d::<i32>().unwrap().to_vec(), data);
        drop(f);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn direct_chunk_matches_the_normal_write_path() {
        let d = tmpdir("cmp");
        let n = 5000usize;
        let chunk = 512usize;
        let data: Vec<i64> = (0..n as i64).map(|i| (i * 1_000_003) % 65_536).collect();
        // reference: let HDF5 do the filtering
        let pa = d.join("a.h5");
        {
            let f = hdf5::File::create(pa.to_str().unwrap()).unwrap();
            f.new_dataset::<i64>().shape([n]).chunk([chunk]).shuffle().deflate(4)
                .create("x").unwrap().write(&arr1(&data)).unwrap();
        }
        // ours: same pipeline, chunks compressed on rayon threads
        let pb = d.join("b.h5");
        {
            let f = hdf5::File::create(pb.to_str().unwrap()).unwrap();
            let ds = f.new_dataset::<i64>().shape((0..,)).chunk([chunk]).shuffle().deflate(4)
                .create("x").unwrap();
            let mut col = ParColumn::new(ds, 8, chunk, 4);
            col.push_i64(&data).unwrap();
            col.finish().unwrap();
        }
        let fa = hdf5::File::open(pa.to_str().unwrap()).unwrap();
        let fb = hdf5::File::open(pb.to_str().unwrap()).unwrap();
        let (da, db) = (fa.dataset("x").unwrap(), fb.dataset("x").unwrap());
        assert_eq!(da.read_1d::<i64>().unwrap().to_vec(), db.read_1d::<i64>().unwrap().to_vec());
        assert_eq!(da.shape(), db.shape());
        // both must advertise the same filter pipeline, else cooler/h5py would need a plugin
        assert_eq!(format!("{:?}", da.filters()), format!("{:?}", db.filters()));
        drop(fa); drop(fb);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn handles_exact_multiples_and_tiny_and_empty_columns() {
        let d = tmpdir("edge");
        let chunk = 256usize;
        for (tag, n) in [("empty", 0usize), ("tiny", 1), ("exact", 512), ("exact_plus1", 513)] {
            let p = d.join(format!("{}.h5", tag));
            let data: Vec<i32> = (0..n as i32).collect();
            {
                let f = hdf5::File::create(p.to_str().unwrap()).unwrap();
                let ds = f.new_dataset::<i32>().shape((0..,)).chunk([chunk]).shuffle().deflate(1)
                    .create("x").unwrap();
                let mut col = ParColumn::new(ds, 4, chunk, 1);
                col.push_i32(&data).unwrap();
                col.finish().unwrap();
                col.finish().unwrap(); // idempotent
            }
            let f = hdf5::File::open(p.to_str().unwrap()).unwrap();
            let got = f.dataset("x").unwrap().read_1d::<i32>().unwrap().to_vec();
            assert_eq!(got, data, "case {}", tag);
            drop(f);
        }
        std::fs::remove_dir_all(&d).ok();
    }
}
