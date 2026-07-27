//! expected: cis distance-decay P(s) per region (arms/chroms view). One O(nnz) streaming pass for
//! sum_balanced[region][dist]; n_valid[region][dist] via FFT autocorrelation of the region's
//! valid-mask. Stored in-cooler under {grp}/expected/{view}/weight (+ the view under {grp}/views/).
use crate::view;
use anyhow::{anyhow, Result};
use hdf5::File;
use num_complex::Complex;
use rustfft::FftPlanner;

/// n_valid[s] = sum_i v[i]*v[i+s] for a 0/1 mask, via FFT autocorrelation. len(out)=len(v).
fn autocorr(v: &[f64]) -> Vec<f64> {
    let l = v.len();
    if l == 0 { return Vec::new(); }
    let m = (2 * l).next_power_of_two();
    let mut buf: Vec<Complex<f64>> = (0..m).map(|i| Complex::new(if i < l { v[i] } else { 0.0 }, 0.0)).collect();
    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(m).process(&mut buf);
    for x in buf.iter_mut() { *x = *x * x.conj(); }
    planner.plan_fft_inverse(m).process(&mut buf);
    (0..l).map(|s| (buf[s].re / m as f64).round()).collect()
}

fn stream_pixels(g: &hdf5::Group, block: usize, mut f: impl FnMut(&[i64], &[i64], &[i32])) -> Result<()> {
    let (b1, b2, cn) = (g.dataset("pixels/bin1_id")?, g.dataset("pixels/bin2_id")?, g.dataset("pixels/count")?);
    let n = cn.shape()[0];
    let mut pos = 0;
    while pos < n {
        let hi = (pos + block).min(n);
        let a = b1.read_slice_1d::<i64, _>(pos..hi)?;
        let b = b2.read_slice_1d::<i64, _>(pos..hi)?;
        let c = cn.read_slice_1d::<i32, _>(pos..hi)?;
        f(a.as_slice().unwrap(), b.as_slice().unwrap(), c.as_slice().unwrap());
        pos = hi;
    }
    Ok(())
}

pub fn expected(uri: &str, view_req: Option<&str>, log: bool) -> Result<()> {
    let t0 = std::time::Instant::now();
    let (path, grp) = match uri.split_once("::") { Some((a, b)) => (a.to_string(), b.to_string()), None => (uri.to_string(), "/".to_string()) };
    let f = File::append(&path)?;
    let g = if grp == "/" { f.group("/")? } else { f.group(&grp)? };
    let meta = crate::cooler::read_meta(&path, if grp == "/" { None } else { Some(grp.rsplit('/').next().unwrap()) })?;
    let nbins = meta.nbins;
    if !g.link_exists("bins/weight") { return Err(anyhow!("cooler is not balanced (no bins/weight); run `rooler balance` first")); }
    let weight: Vec<f64> = g.dataset("bins/weight")?.read_1d::<f64>()?.to_vec();
    let ignore_diags = g.dataset("bins/weight")?.attr("ignore_diags").ok()
        .and_then(|a| a.read_scalar::<i64>().ok()).unwrap_or(2);

    let chromsizes: Vec<(String, i64)> = meta.names.iter().cloned().zip(meta.lengths.iter().cloned()).collect();
    let (view_name, regions) = view::resolve(&meta.assembly, &chromsizes, view_req)
        .map_err(|e| anyhow!("cannot determine expected regions: {}", e))?;
    let cid: std::collections::HashMap<&str, usize> = meta.names.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();

    // region bin ranges + bin->region map
    let nreg = regions.len();
    let mut reg_b0 = vec![0i64; nreg];
    let mut reg_len = vec![0usize; nreg];
    let mut region_of = vec![-1i32; nbins];
    for (ri, r) in regions.iter().enumerate() {
        let c = *cid.get(r.chrom.as_str()).ok_or_else(|| anyhow!("region chrom {} not in cooler", r.chrom))?;
        let base = meta.chrom_offset[c];
        let b0 = base + r.start / meta.binsize;
        let b1 = base + (r.end + meta.binsize - 1) / meta.binsize;
        reg_b0[ri] = b0; reg_len[ri] = (b1 - b0) as usize;
        for b in b0..b1 { region_of[b as usize] = ri as i32; }
    }

    // streaming sum_balanced[region][dist]
    let mut sumb: Vec<Vec<f64>> = reg_len.iter().map(|&l| vec![0f64; l]).collect();
    stream_pixels(&g, 1 << 22, |ia, ja, ca| {
        for k in 0..ia.len() {
            let (i, j) = (ia[k] as usize, ja[k] as usize);
            let ri = region_of[i];
            if ri < 0 || region_of[j] != ri { continue; }
            let (wi, wj) = (weight[i], weight[j]);
            if wi.is_nan() || wj.is_nan() { continue; }
            let d = (ja[k] - ia[k]) as usize;
            sumb[ri as usize][d] += ca[k] as f64 * wi * wj;
        }
    })?;
    if log { eprintln!("  expected: sum pass done ({} regions, view={}) {:.0}s", nreg, view_name, t0.elapsed().as_secs_f64()); }

    // n_valid via FFT autocorr of per-region valid mask; assemble long-format output
    let (mut o_reg, mut o_dist, mut o_nval, mut o_sum, mut o_avg) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for ri in 0..nreg {
        let (b0, l) = (reg_b0[ri] as usize, reg_len[ri]);
        let mask: Vec<f64> = (0..l).map(|k| if weight[b0 + k].is_nan() { 0.0 } else { 1.0 }).collect();
        let nval = autocorr(&mask);
        for d in 0..l {
            let nv = nval[d];
            let avg = if d < ignore_diags as usize || nv <= 0.0 { f64::NAN } else { sumb[ri][d] / nv };
            o_reg.push(ri as i32); o_dist.push(d as i64); o_nval.push(nv); o_sum.push(sumb[ri][d]); o_avg.push(avg);
        }
    }

    // store: {grp}/expected/{view}/weight
    let _ = g.unlink("expected");
    let ge = g.create_group(&format!("expected/{}/weight", view_name))?;
    ge.new_dataset::<i32>().shape([o_reg.len()]).create("region_id")?.write(&ndarray::arr1(&o_reg))?;
    ge.new_dataset::<i64>().shape([o_dist.len()]).create("dist")?.write(&ndarray::arr1(&o_dist))?;
    ge.new_dataset::<f64>().shape([o_nval.len()]).deflate(4).create("n_valid")?.write(&ndarray::arr1(&o_nval))?;
    ge.new_dataset::<f64>().shape([o_sum.len()]).deflate(4).create("balanced.sum")?.write(&ndarray::arr1(&o_sum))?;
    ge.new_dataset::<f64>().shape([o_avg.len()]).deflate(4).create("balanced.avg")?.write(&ndarray::arr1(&o_avg))?;
    // store the view (regions)
    let _ = g.unlink(&format!("views/{}", view_name));
    let gv = g.create_group(&format!("views/{}", view_name))?;
    let names: Vec<String> = regions.iter().map(|r| r.name.clone()).collect();
    let chroms: Vec<String> = regions.iter().map(|r| r.chrom.clone()).collect();
    write_str(&gv, "name", &names)?; write_str(&gv, "chrom", &chroms)?;
    gv.new_dataset::<i64>().shape([nreg]).create("start")?.write(&ndarray::arr1(&regions.iter().map(|r| r.start).collect::<Vec<_>>()))?;
    gv.new_dataset::<i64>().shape([nreg]).create("end")?.write(&ndarray::arr1(&regions.iter().map(|r| r.end).collect::<Vec<_>>()))?;
    if log { eprintln!("  expected DONE: {} rows, view '{}' in {:.0}s", o_reg.len(), view_name, t0.elapsed().as_secs_f64()); }
    Ok(())
}

fn write_str(g: &hdf5::Group, name: &str, vals: &[String]) -> Result<()> {
    use hdf5::types::FixedAscii;
    let ml = vals.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
    macro_rules! w { ($n:expr) => {{
        let a: Vec<FixedAscii<$n>> = vals.iter().map(|s| FixedAscii::<$n>::from_ascii(s.as_bytes()).unwrap()).collect();
        g.new_dataset::<FixedAscii<$n>>().shape([vals.len()]).create(name)?.write(&ndarray::arr1(&a))?;
    }}}
    match ml { 0..=16 => w!(16), 17..=32 => w!(32), _ => w!(64) }
    Ok(())
}
