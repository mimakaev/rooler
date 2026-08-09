//! Parallel direct-chunk reader — the mirror of `parwrite.rs`. HDF5's gzip filter stores each
//! chunk as an independent zlib stream, but the ordinary read path inflates them one at a time
//! on one thread, which is what makes streaming ops read-bound. Here the raw compressed chunks
//! are fetched with `H5Dread_chunk` (cheap: bytes off disk / page cache) and inflated +
//! unshuffled on rayon threads. Read-side twin rule of parwrite: every HDF5 call happens on the
//! caller's thread AND under the crate's global lock; only decompression is parallel.
//!
//! `stream_pixels` is the shared entry point for every op that scans a pixel table
//! (zoomify/coarsen, expected, repack): it takes the fast path when all three columns are
//! SHUFFLE+DEFLATE (rooler's default preset, and what cooler itself writes) and falls back to
//! the ordinary serial read for anything else (blosc, uncompressed, exotic layouts).
use anyhow::{bail, Result};
use hdf5::filters::Filter;
use hdf5_metno_sys::h5d::{H5Dget_chunk_info_by_coord, H5Dread_chunk};
use libdeflater::Decompressor;
use rayon::prelude::*;

/// How the stored bytes of a chunk map back to raw element bytes.
#[derive(Clone, Copy, PartialEq)]
enum Pipe {
    ShuffleDeflate, // filter pipeline [shuffle, deflate] — rooler + cooler default
    DeflateOnly,    // plain gzip, no shuffle
}

/// Sequential reader over one 1-D chunked SHUFFLE+DEFLATE dataset via raw chunk access.
pub struct ChunkStream {
    dset: hdf5::Dataset,
    esize: usize,
    chunk: usize,   // elements per chunk
    n: usize,       // total elements
    nchunks: usize,
    next: usize,    // next chunk index to hand out
    pipe: Pipe,
}

impl ChunkStream {
    /// `None` if this dataset can't take the fast path (not 1-D chunked, or a filter pipeline
    /// other than [shuffle,] deflate) — the caller then uses the ordinary read API.
    pub fn open(dset: hdf5::Dataset) -> Result<Option<ChunkStream>> {
        let shape = dset.shape();
        if shape.len() != 1 { return Ok(None); }
        let n = shape[0];
        let chunk = match dset.chunk() { Some(c) if c.len() == 1 && c[0] > 0 => c[0], _ => return Ok(None) };
        let pipe = match dset.filters().as_slice() {
            [Filter::Shuffle, Filter::Deflate(_)] => Pipe::ShuffleDeflate,
            [Filter::Deflate(_)] => Pipe::DeflateOnly,
            _ => return Ok(None),
        };
        let esize = dset.dtype()?.size();
        let nchunks = n.div_ceil(chunk).max(if n == 0 { 0 } else { 1 });
        Ok(Some(ChunkStream { dset, esize, chunk, n, nchunks, next: 0, pipe }))
    }

    pub fn n_elems(&self) -> usize { self.n }
    pub fn chunk_elems(&self) -> usize { self.chunk }
    /// Bytes per stored element, as the file declares it.
    pub fn elem_size(&self) -> usize { self.esize }
    pub fn done(&self) -> bool { self.next >= self.nchunks }

    /// Fetch the raw stored bytes of up to `k` chunks (HDF5 calls, this thread, global lock).
    /// Returns (chunk_index, filter_mask, raw_bytes) triples.
    fn fetch_raw(&mut self, k: usize) -> Result<Vec<(usize, u32, Vec<u8>)>> {
        let hi = (self.next + k).min(self.nchunks);
        let mut out = Vec::with_capacity(hi - self.next);
        for c in self.next..hi {
            let offset = [(c * self.chunk) as u64];
            let (mut mask, mut addr, mut size) = (0u32, 0u64, 0u64);
            let raw = hdf5::sync::sync(|| -> Result<Vec<u8>> {
                let rc = unsafe {
                    H5Dget_chunk_info_by_coord(self.dset.id(), offset.as_ptr(), &mut mask,
                                               &mut addr, &mut size)
                };
                if rc < 0 || size == 0 { bail!("chunk {} of {}: no stored data", c, self.dset.name()); }
                let mut buf = vec![0u8; size as usize];
                let mut mask2 = 0u32;
                let rc = unsafe {
                    H5Dread_chunk(self.dset.id(), 0, offset.as_ptr(), &mut mask2,
                                  buf.as_mut_ptr().cast())
                };
                if rc < 0 { bail!("H5Dread_chunk failed at chunk {} of {}", c, self.dset.name()); }
                Ok(buf)
            })?;
            out.push((c, mask, raw));
        }
        self.next = hi;
        Ok(out)
    }
}

/// Decode one raw chunk to element bytes: inflate (unless the mask says deflate was skipped),
/// then un-byte-transpose (unless shuffle was skipped or absent). Returns exactly
/// `valid * esize` bytes. Filter-mask bit i set = pipeline stage i was skipped for this chunk.
fn decode_chunk(raw: &[u8], mask: u32, pipe: Pipe, esize: usize, chunk: usize, valid: usize) -> Result<Vec<u8>> {
    let cbytes = chunk * esize;
    let (shuffle_skipped, deflate_skipped) = match pipe {
        Pipe::ShuffleDeflate => (mask & 1 != 0, mask & 2 != 0),
        Pipe::DeflateOnly => (true, mask & 1 != 0),
    };
    let inflated: Vec<u8> = if deflate_skipped {
        raw.to_vec()
    } else {
        let mut out = vec![0u8; cbytes];
        let m = Decompressor::new().zlib_decompress(raw, &mut out)
            .map_err(|e| anyhow::anyhow!("chunk inflate failed: {:?}", e))?;
        out.truncate(m);
        out
    };
    if inflated.len() < valid * esize { bail!("chunk decoded to {} bytes, need {}", inflated.len(), valid * esize); }
    if shuffle_skipped {
        let mut v = inflated;
        v.truncate(valid * esize);
        return Ok(v);
    }
    // HDF5 SHUFFLE byte-transpose over the n elements the filter actually saw
    let nfull = inflated.len() / esize;
    let mut out = vec![0u8; valid * esize];
    for b in 0..esize {
        let plane = &inflated[b * nfull..];
        for e in 0..valid { out[e * esize + b] = plane[e]; }
    }
    Ok(out)
}

/// Stream a pixel table (`bin1_id`, `bin2_id`, `count`) through `f` in order, decompressing
/// chunks on rayon threads when all three columns are gzip; otherwise the ordinary serial path.
pub fn stream_pixels(
    g: &hdf5::Group, block: usize,
    mut f: impl FnMut(&[i64], &[i64], &[i32]) -> Result<()>,
) -> Result<()> {
    let b1d = g.dataset("pixels/bin1_id")?;
    let b2d = g.dataset("pixels/bin2_id")?;
    let cnd = g.dataset("pixels/count")?;
    // Fast path only when the columns are chunk-aligned with each other (so batches line up)
    // AND stored at the widths the decoders below assume. A cooler written elsewhere may use
    // int32 bin ids or int64 counts; reinterpreting those would silently corrupt the values, so
    // anything unexpected falls back to the serial path, which converts dtypes properly.
    let fast = match (ChunkStream::open(b1d.clone())?, ChunkStream::open(b2d.clone())?, ChunkStream::open(cnd.clone())?) {
        (Some(a), Some(b), Some(c))
            if a.chunk_elems() == b.chunk_elems() && a.chunk_elems() == c.chunk_elems()
                && a.n_elems() == b.n_elems() && a.n_elems() == c.n_elems()
                && a.elem_size() == 8 && b.elem_size() == 8 && c.elem_size() == 4 => Some((a, b, c)),
        _ => None,
    };
    let Some((mut s1, mut s2, mut sc)) = fast else {
        // serial fallback: the pre-existing read path
        let n = cnd.shape()[0];
        let mut pos = 0;
        while pos < n {
            let hi = (pos + block).min(n);
            let a = b1d.read_slice_1d::<i64, _>(pos..hi)?;
            let b = b2d.read_slice_1d::<i64, _>(pos..hi)?;
            let c = cnd.read_slice_1d::<i32, _>(pos..hi)?;
            f(a.as_slice().unwrap(), b.as_slice().unwrap(), c.as_slice().unwrap())?;
            pos = hi;
        }
        return Ok(());
    };
    let chunk = s1.chunk_elems();
    let n = s1.n_elems();
    let batch_chunks = (block / chunk.max(1)).clamp(1, 64);
    let mut base = 0usize; // element index of the batch start
    while !s1.done() {
        let r1 = s1.fetch_raw(batch_chunks)?;
        let r2 = s2.fetch_raw(batch_chunks)?;
        let rc = sc.fetch_raw(batch_chunks)?;
        let m: usize = r1.iter().map(|(c, _, _)| (n - c * chunk).min(chunk)).sum();
        // decode all three columns' chunks in one parallel wave, then convert to typed vectors
        let (o1, (o2, oc)) = rayon::join(
            || decode_i64(&r1, &s1, m, chunk, n),
            || rayon::join(|| decode_i64(&r2, &s2, m, chunk, n),
                           || decode_i32(&rc, &sc, m, chunk, n)),
        );
        let (o1, o2, oc) = (o1?, o2?, oc?);
        f(&o1, &o2, &oc)?;
        base += m;
    }
    debug_assert_eq!(base, n);
    Ok(())
}

fn decode_i64(raws: &[(usize, u32, Vec<u8>)], s: &ChunkStream, m: usize, chunk: usize, n: usize) -> Result<Vec<i64>> {
    let parts: Vec<Vec<u8>> = raws.par_iter().map(|(c, mask, raw)| {
        decode_chunk(raw, *mask, s.pipe, 8, chunk, (n - c * chunk).min(chunk))
    }).collect::<Result<_>>()?;
    let mut out = Vec::with_capacity(m);
    for p in &parts {
        out.extend(p.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())));
    }
    Ok(out)
}

fn decode_i32(raws: &[(usize, u32, Vec<u8>)], s: &ChunkStream, m: usize, chunk: usize, n: usize) -> Result<Vec<i32>> {
    let parts: Vec<Vec<u8>> = raws.par_iter().map(|(c, mask, raw)| {
        decode_chunk(raw, *mask, s.pipe, 4, chunk, (n - c * chunk).min(chunk))
    }).collect::<Result<_>>()?;
    let mut out = Vec::with_capacity(m);
    for p in &parts {
        out.extend(p.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("rooler_parread_{}_{}", tag, std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The load-bearing test: chunks written by BOTH writers (HDF5's own filter pipeline and
    /// our ParColumn direct-chunk path) must decode identically through ChunkStream,
    /// including the partial edge chunk.
    #[test]
    fn chunkstream_reads_both_write_paths_exactly() {
        let d = tmpdir("both");
        let n = 3000usize;
        let chunk = 1024usize;
        let data: Vec<i64> = (0..n as i64).map(|i| i * 977 - 12345).collect();
        let pa = d.join("hdf5.h5");
        {
            let f = hdf5::File::create(pa.to_str().unwrap()).unwrap();
            f.new_dataset::<i64>().shape([n]).chunk([chunk]).shuffle().deflate(4)
                .create("x").unwrap().write(&arr1(&data)).unwrap();
        }
        let pb = d.join("par.h5");
        {
            let f = hdf5::File::create(pb.to_str().unwrap()).unwrap();
            let ds = f.new_dataset::<i64>().shape((0..,)).chunk([chunk]).shuffle().deflate(4)
                .create("x").unwrap();
            let mut col = crate::parwrite::ParColumn::new(ds, 8, chunk, 4).unwrap();
            col.push_i64(&data).unwrap();
            col.finish().unwrap();
        }
        for p in [&pa, &pb] {
            let f = hdf5::File::open(p.to_str().unwrap()).unwrap();
            let mut s = ChunkStream::open(f.dataset("x").unwrap()).unwrap()
                .expect("shuffle+deflate must take the fast path");
            let mut got: Vec<i64> = Vec::new();
            while !s.done() {
                let raws = s.fetch_raw(2).unwrap();
                for (c, mask, raw) in &raws {
                    let valid = (n - c * chunk).min(chunk);
                    let bytes = decode_chunk(raw, *mask, s.pipe, 8, chunk, valid).unwrap();
                    got.extend(bytes.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())));
                }
            }
            assert_eq!(got, data, "{}", p.display());
        }
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn unsupported_filters_fall_back() {
        let d = tmpdir("fb");
        let p = d.join("t.h5");
        let f = hdf5::File::create(p.to_str().unwrap()).unwrap();
        // uncompressed chunked -> no deflate in the pipeline -> not eligible
        let ds = f.new_dataset::<i32>().shape([100]).chunk([32]).create("plain").unwrap();
        ds.write(&arr1(&(0..100).collect::<Vec<i32>>())).unwrap();
        assert!(ChunkStream::open(ds).unwrap().is_none());
        // contiguous -> not eligible
        let ds = f.new_dataset::<i32>().shape([10]).create("contig").unwrap();
        ds.write(&arr1(&(0..10).collect::<Vec<i32>>())).unwrap();
        assert!(ChunkStream::open(ds).unwrap().is_none());
        drop(f);
        std::fs::remove_dir_all(&d).ok();
    }
}
