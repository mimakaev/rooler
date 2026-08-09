//! zoomify: base .cool -> multi-resolution .mcool by cascading integer-factor coarsening.
//! Coarsening is streaming: map each fine bin -> coarse bin (respecting chrom boundaries), and
//! because fine pixels are sorted by bin1 the coarse bin1 is non-decreasing, so we accumulate one
//! coarse row at a time (sort its bin2 + sum) and emit. O(nnz), bounded memory, no external sort.
use crate::cooler::{CoolWriter, Comp};
use anyhow::Result;
use hdf5::File;
use std::time::Instant;

fn offsets(lengths: &[i64], res: i64) -> (Vec<i64>, i64) {
    let mut off = vec![0i64];
    for &l in lengths { off.push(off.last().unwrap() + (l + res - 1) / res); }
    let n = *off.last().unwrap();
    (off, n)
}

fn read_root_meta(path: &str) -> Result<(Vec<String>, Vec<i64>, i64, String)> {
    let m = crate::cooler::read_meta(path, None)?;
    Ok((m.names, m.lengths, m.binsize, m.assembly))
}

/// Stream all pixels of `grp/pixels` in blocks, calling `f(bin1, bin2, count)`.
fn stream_pixels(g: &hdf5::Group, block: usize, mut f: impl FnMut(&[i64], &[i64], &[i32]) -> Result<()>) -> Result<()> {
    let b1 = g.dataset("pixels/bin1_id")?;
    let b2 = g.dataset("pixels/bin2_id")?;
    let cn = g.dataset("pixels/count")?;
    let n = cn.shape()[0];
    let mut pos = 0;
    while pos < n {
        let hi = (pos + block).min(n);
        let a = b1.read_slice_1d::<i64, _>(pos..hi)?;
        let b = b2.read_slice_1d::<i64, _>(pos..hi)?;
        let c = cn.read_slice_1d::<i32, _>(pos..hi)?;
        f(a.as_slice().unwrap(), b.as_slice().unwrap(), c.as_slice().unwrap())?;
        pos = hi;
    }
    Ok(())
}

struct RowAgg {
    cur: i64,
    b2: Vec<i64>,
    c: Vec<i64>,
    ob1: Vec<i64>,
    ob2: Vec<i64>,
    oc: Vec<i32>,
    emit: usize,
    nclamped: u64,
}
impl RowAgg {
    fn new(emit: usize) -> RowAgg {
        RowAgg { cur: -1, b2: Vec::new(), c: Vec::new(),
                 ob1: Vec::with_capacity(emit), ob2: Vec::with_capacity(emit), oc: Vec::with_capacity(emit),
                 emit, nclamped: 0 }
    }
    fn push(&mut self, cb1: i64, cb2: i64, cnt: i64, w: &mut CoolWriter) -> Result<()> {
        if cb1 != self.cur {
            self.flush_row(w)?;
            self.cur = cb1;
        }
        self.b2.push(cb2); self.c.push(cnt);
        Ok(())
    }
    fn flush_row(&mut self, w: &mut CoolWriter) -> Result<()> {
        if self.b2.is_empty() { return Ok(()); }
        // sort (bin2) then run-length sum
        let mut idx: Vec<usize> = (0..self.b2.len()).collect();
        idx.sort_unstable_by_key(|&i| self.b2[i]);
        let mut k = 0;
        while k < idx.len() {
            let key = self.b2[idx[k]];
            let mut s = 0i64;
            while k < idx.len() && self.b2[idx[k]] == key { s += self.c[idx[k]]; k += 1; }
            self.ob1.push(self.cur); self.ob2.push(key);
            self.oc.push(crate::cooler::clamp_count(s, &mut self.nclamped));
        }
        self.b2.clear(); self.c.clear();
        if self.ob1.len() >= self.emit { self.drain(w)?; }
        Ok(())
    }
    fn drain(&mut self, w: &mut CoolWriter) -> Result<()> {
        w.append(&self.ob1, &self.ob2, &self.oc)?;
        self.ob1.clear(); self.ob2.clear(); self.oc.clear();
        Ok(())
    }
    fn finish(&mut self, w: &mut CoolWriter) -> Result<()> {
        self.flush_row(w)?; self.drain(w)
    }
}

fn build_map(lengths: &[i64], fine_res: i64, coarse_res: i64) -> (Vec<i64>, Vec<i64>, i64) {
    let (fine_off, fine_n) = offsets(lengths, fine_res);
    let (coarse_off, coarse_n) = offsets(lengths, coarse_res);
    let factor = coarse_res / fine_res;
    let mut map = vec![0i64; fine_n as usize];
    for c in 0..lengths.len() {
        let (flo, fhi) = (fine_off[c], fine_off[c + 1]);
        for (local, b) in (flo..fhi).enumerate() {
            map[b as usize] = coarse_off[c] + (local as i64) / factor;
        }
    }
    (map, coarse_off, coarse_n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_map_respects_chrom_boundaries() {
        // chrom lengths chosen so bin counts are ODD -> the last fine bin of chrA is a partial
        // coarse bin and must NOT be merged into chrB's first coarse bin.
        let lengths = vec![250i64, 130];
        let (map, coarse_off, coarse_n) = build_map(&lengths, 10, 20);
        let (fine_off, fine_n) = offsets(&lengths, 10);
        assert_eq!(fine_off, vec![0, 25, 38]);          // 25 and 13 fine bins
        assert_eq!(coarse_off, vec![0, 13, 20]);        // 13 and 7 coarse bins
        assert_eq!((fine_n, coarse_n), (38, 20));
        // every fine bin maps inside its own chrom's coarse range
        for c in 0..lengths.len() {
            for b in fine_off[c]..fine_off[c + 1] {
                let cb = map[b as usize];
                assert!(cb >= coarse_off[c] && cb < coarse_off[c + 1],
                    "fine bin {} of chrom {} mapped to coarse {} outside [{}, {})",
                    b, c, cb, coarse_off[c], coarse_off[c + 1]);
            }
        }
        // chrA's last fine bin (24) is the odd one out -> its own coarse bin 12, not chrB's 13
        assert_eq!(map[24], 12);
        assert_eq!(map[25], 13);                        // chrB's first fine bin
        // factor 2 pairing within a chrom
        assert_eq!(&map[0..6], &[0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn coarse_map_handles_factor_4() {
        let lengths = vec![250i64, 130];
        let (map, coarse_off, _) = build_map(&lengths, 10, 40);
        assert_eq!(&map[0..5], &[0, 0, 0, 0, 1]);
        assert_eq!(map[24], coarse_off[1] - 1, "chrA's last fine bin -> chrA's last coarse bin");
        assert_eq!(map[25], coarse_off[1], "chrB starts a fresh coarse bin");
    }

    #[test]
    fn row_agg_sums_duplicate_columns_and_clamps() {
        // RowAgg's clamp path: two coarse-mapped pixels landing in one cell above i32::MAX
        let mut agg = RowAgg::new(1 << 20);
        agg.b2 = vec![5, 5, 7];
        agg.c = vec![i64::from(i32::MAX), 10, 3];
        agg.cur = 0;
        let mut ob = (Vec::new(), Vec::new(), Vec::new());
        // emulate flush_row's aggregation without a writer
        let mut idx: Vec<usize> = (0..agg.b2.len()).collect();
        idx.sort_unstable_by_key(|&i| agg.b2[i]);
        let mut k = 0;
        while k < idx.len() {
            let key = agg.b2[idx[k]];
            let mut s = 0i64;
            while k < idx.len() && agg.b2[idx[k]] == key { s += agg.c[idx[k]]; k += 1; }
            ob.0.push(agg.cur); ob.1.push(key);
            ob.2.push(crate::cooler::clamp_count(s, &mut agg.nclamped));
        }
        assert_eq!(ob.1, vec![5, 7]);
        assert_eq!(ob.2, vec![i32::MAX, 3], "summed count saturates instead of wrapping");
        assert_eq!(agg.nclamped, 1);
    }
}

/// Build the mcool, then (optionally) balance every level. Balancing runs *after* the cascade
/// so the writer's HDF5 handles are all released — an op cannot reopen the file read-write
/// while any File/Group/Dataset from the build is still alive.
pub fn zoomify_and_balance(
    src: &str, out: &str, resolutions: Option<Vec<i64>>, comp: Comp, assembly: Option<&str>,
    balance: bool, nthreads: usize, log: bool,
) -> Result<()> {
    let reslist = zoomify(src, out, resolutions, comp, assembly, log)?;
    if !balance { return Ok(()); }
    let t0 = Instant::now();
    for res in &reslist {
        let uri = format!("{}::resolutions/{}", out, res);
        let nbins = {
            let m = crate::cooler::read_meta(out, Some(&res.to_string()))?;
            m.nbins
        };
        // cache-blocked SpMV pays off once the marginal vectors stop fitting in L3
        let tiled_block = if nbins >= 4_000_000 { Some(65536) } else { None };
        if log { eprintln!("  [zoomify] balancing {}bp ({} bins){}", res, nbins,
            if tiled_block.is_some() { ", tiled" } else { "" }); }
        crate::balance::balance(&uri, crate::balance::Params {
            nthreads, tiled_block, ..Default::default() }, log)?;
    }
    if log { eprintln!("  zoomify: balanced {} resolutions in {:.0}s", reslist.len(), t0.elapsed().as_secs_f64()); }
    Ok(())
}

/// Build the mcool. Returns the resolutions written, coarsest-last.
pub fn zoomify(src: &str, out: &str, resolutions: Option<Vec<i64>>, comp: Comp, assembly: Option<&str>, log: bool) -> Result<Vec<i64>> {
    let t0 = Instant::now();
    let (names, lengths, base_res, src_asm) = read_root_meta(src)?;
    let chromsizes: Vec<(String, i64)> = names.iter().cloned().zip(lengths.iter().cloned()).collect();
    let asm = match assembly {
        Some(a) if !a.trim().is_empty() => a.trim().to_string(),
        _ if !src_asm.is_empty() && src_asm != "unknown" => src_asm.clone(),
        _ => crate::view::detect("", &chromsizes).map(|s| s.to_string()).ok_or_else(|| anyhow::anyhow!(
            "refusing to write an mcool without a genome assembly (source lacks one); pass --assembly <name>"))?,
    };
    // resolution schedule: explicit, or double until very coarse
    let reslist: Vec<i64> = match resolutions {
        Some(mut v) => { v.sort(); v }
        None => {
            let mut v = vec![base_res]; let mut r = base_res;
            let (_, mut n) = offsets(&lengths, r);
            while n > (4 * names.len() as i64) {
                r *= 2; let (_, nn) = offsets(&lengths, r); n = nn; v.push(r);
            }
            v
        }
    };
    let f = File::create(out)?;
    f.new_attr::<hdf5::types::VarLenAscii>().create("format")?
        .write_scalar(&hdf5::types::VarLenAscii::from_ascii("HDF5::MCOOL")?)?;
    f.new_attr::<i64>().create("format-version")?.write_scalar(&2i64)?;
    let _ = f.create_group("resolutions")?;
    let block = 1 << 22;
    let emit = 1 << 20;

    // level 0: copy base from src -> resolutions/{base_res}
    let (base_off, base_n) = offsets(&lengths, base_res);
    {
        let mut w = CoolWriter::create_in(&f, &format!("resolutions/{}", base_res), &names, &lengths,
            base_res, base_n as usize, &base_off, comp, &asm)?;
        let srcf = File::open(src)?;
        let sg = srcf.group("/")?;
        let mut ob1 = Vec::new(); let mut ob2 = Vec::new(); let mut oc = Vec::new();
        stream_pixels(&sg, block, |a, b, c| {
            ob1.extend_from_slice(a); ob2.extend_from_slice(b); oc.extend_from_slice(c);
            if ob1.len() >= emit { w.append(&ob1, &ob2, &oc)?; ob1.clear(); ob2.clear(); oc.clear(); }
            Ok(())
        })?;
        if !ob1.is_empty() { w.append(&ob1, &ob2, &oc)?; }
        w.close()?;
    }
    if log { eprintln!("  [zoomify] {}bp copied ({} bins) {:.0}s", base_res, base_n, t0.elapsed().as_secs_f64()); }

    // cascade coarser levels from the previous level in the output file
    let mut prev_res = base_res;
    for &res in reslist.iter().skip(1) {
        let (map, coarse_off, coarse_n) = build_map(&lengths, prev_res, res);
        let mut w = CoolWriter::create_in(&f, &format!("resolutions/{}", res), &names, &lengths,
            res, coarse_n as usize, &coarse_off, comp, &asm)?;
        let mut agg = RowAgg::new(emit);
        let pg = f.group(&format!("resolutions/{}", prev_res))?;
        stream_pixels(&pg, block, |a, b, c| {
            for i in 0..a.len() {
                let cb1 = map[a[i] as usize];
                let cb2 = map[b[i] as usize];
                agg.push(cb1, cb2, c[i] as i64, &mut w)?;
            }
            Ok(())
        })?;
        agg.finish(&mut w)?;
        let nnz = w.nnz;
        w.close()?;
        if agg.nclamped > 0 {
            eprintln!("  [zoomify] WARNING {}bp: {} pixels clamped at i32::MAX (counts are stored as i32)",
                res, agg.nclamped);
        }
        if log { eprintln!("  [zoomify] {}bp built ({} bins, {} pix) {:.0}s", res, coarse_n, nnz, t0.elapsed().as_secs_f64()); }
        prev_res = res;
    }
    if log { eprintln!("  zoomify DONE: {} resolutions in {:.0}s", reslist.len(), t0.elapsed().as_secs_f64()); }
    Ok(reslist)
}
