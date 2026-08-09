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

// ---- log-space Gaussian smoothing of the contact-vs-distance curve -------------------------
// Reproduces cooltools' `log_smooth` (cooltools.sandbox.expected_smoothing), which is what
// `expected_cis(smooth=True)` uses with sigma_log10=0.1, window_sigma=5, points_per_sigma=10.
// A raw P(s) curve is noisy at large separations, where few pixel pairs contribute; smoothing
// in log10(distance) is what makes the curve usable for pileups and observed/expected.

/// Thin a sorted array to points ~uniformly spaced in log10, keeping the first and last.
fn log_thin(xs: &[f64], min_log10_step: f64) -> Vec<f64> {
    if xs.is_empty() { return Vec::new(); }
    let min_ratio = 10f64.powf(min_log10_step);
    let mut out = vec![xs[0]];
    let mut prev = xs[0];
    for &x in &xs[1..] {
        if x > prev * min_ratio { out.push(x); prev = x; }
    }
    if *out.last().unwrap() != *xs.last().unwrap() { out.push(*xs.last().unwrap()); }
    out
}

/// numpy.interp in log-log space: exp(interp(ln x, ln xp, ln fp)), clamped at the ends.
/// (The log base is irrelevant — it cancels in both the interpolation weights and the result.)
fn log_interp(xs: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    let lx: Vec<f64> = xp.iter().map(|v| v.ln()).collect();
    let lf: Vec<f64> = fp.iter().map(|v| v.ln()).collect();
    xs.iter().map(|&x| {
        let l = x.ln();
        if l <= lx[0] { return lf[0].exp(); }
        if l >= *lx.last().unwrap() { return lf.last().unwrap().exp(); }
        // j = last knot at or below l. Landing exactly on a knot returns its value, as numpy
        // does — important because knot 0 is ln(dist 0) = -inf and a slope through it is NaN.
        let j = lx.partition_point(|&v| v <= l) - 1;
        if lx[j] == l { return lf[j].exp(); }
        let (x0, x1, y0, y1) = (lx[j], lx[j + 1], lf[j], lf[j + 1]);
        if !x0.is_finite() { return y1.exp(); }
        (y0 + (l - x0) / (x1 - x0) * (y1 - y0)).exp()
    }).collect()
}

/// Smooth `sum` and `nval` against `xs` (= distance in bins) with a Gaussian in log10 space,
/// evaluated on a thinned grid and interpolated back. Returns (smoothed_sum, smoothed_nval);
/// the smoothed average is their ratio, exactly as cooltools computes it.
fn log_smooth_pair(xs: &[f64], sum: &[f64], nval: &[f64]) -> (Vec<f64>, Vec<f64>) {
    const SIGMA: f64 = 0.1;          // cooltools smooth_sigma
    const WINDOW_SIGMA: f64 = 5.0;   // window half-width, in sigmas
    const POINTS_PER_SIGMA: f64 = 10.0;
    let n = xs.len();
    if n == 0 { return (Vec::new(), Vec::new()); }
    let thinned = log_thin(xs, SIGMA / POINTS_PER_SIGMA);
    let log_xs: Vec<f64> = xs.iter().map(|v| v.log10()).collect();  // dist 0 -> -inf, as in numpy
    let (mut s_thin, mut v_thin) = (vec![0f64; thinned.len()], vec![0f64; thinned.len()]);
    for (i, &tx) in thinned.iter().enumerate() {
        let cur = tx.log10();
        // numpy searchsorted(side='left') on a sorted array
        let lo = log_xs.partition_point(|&v| v < cur - SIGMA * WINDOW_SIGMA);
        let hi = log_xs.partition_point(|&v| v < cur + SIGMA * WINDOW_SIGMA);
        if lo >= hi { continue; }
        let w: Vec<f64> = log_xs[lo..hi].iter()
            .map(|&lx| (-((cur - lx).powi(2)) / (2.0 * SIGMA * SIGMA)).exp()).collect();
        let norm: f64 = w.iter().sum();
        if norm > 0.0 {
            for (k, &wk) in w.iter().enumerate() {
                s_thin[i] += sum[lo + k] * wk / norm;
                v_thin[i] += nval[lo + k] * wk / norm;
            }
        }
    }
    (log_interp(xs, &thinned, &s_thin), log_interp(xs, &thinned, &v_thin))
}

fn stream_pixels(g: &hdf5::Group, block: usize, mut f: impl FnMut(&[i64], &[i64], &[i32])) -> Result<()> {
    // parallel chunk decompression for gzip columns; serial fallback otherwise (parread.rs)
    crate::parread::stream_pixels(g, block, |a, b, c| { f(a, b, c); Ok(()) })
}

/// Default-on path (balance/zoomify/repack): compute expected with the per-organism default
/// view, but never fail the parent op over it — an unknown genome just warns and moves on.
pub fn expected_or_warn(uri: &str, log: bool) {
    if let Err(e) = expected(uri, None, log) {
        eprintln!("  WARNING: expected not computed ({}); run `rooler expected --view ...` manually", e);
    }
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

    // streaming per-region, per-distance sums: balanced (count*w_i*w_j over valid pixels) and raw
    let mut sumb: Vec<Vec<f64>> = reg_len.iter().map(|&l| vec![0f64; l]).collect();
    let mut sumc: Vec<Vec<f64>> = reg_len.iter().map(|&l| vec![0f64; l]).collect();
    stream_pixels(&g, 1 << 22, |ia, ja, ca| {
        for k in 0..ia.len() {
            let (i, j) = (ia[k] as usize, ja[k] as usize);
            let ri = region_of[i];
            if ri < 0 || region_of[j] != ri { continue; }
            // the balancing mask applies to BOTH transforms: cooltools drops pixels touching a
            // masked bin before summing, so count.sum is over valid pixels too, not all pixels
            let (wi, wj) = (weight[i], weight[j]);
            if wi.is_nan() || wj.is_nan() { continue; }
            let d = (ja[k] - ia[k]) as usize;
            sumc[ri as usize][d] += ca[k] as f64;
            sumb[ri as usize][d] += ca[k] as f64 * wi * wj;
        }
    })?;
    if log { eprintln!("  expected: sum pass done ({} regions, view={}) {:.0}s", nreg, view_name, t0.elapsed().as_secs_f64()); }

    // n_valid via FFT autocorr of the per-region valid mask; n_total is just the diagonal length.
    // Ignored diagonals are reported as NaN (cooltools' convention) but enter smoothing as 0,
    // because cooltools sums them with pandas' skipna semantics before smoothing.
    let dmax = reg_len.iter().copied().max().unwrap_or(0);
    let mut nvalid: Vec<Vec<f64>> = Vec::with_capacity(nreg);
    for ri in 0..nreg {
        let (b0, l) = (reg_b0[ri] as usize, reg_len[ri]);
        let mask: Vec<f64> = (0..l).map(|k| if weight[b0 + k].is_nan() { 0.0 } else { 1.0 }).collect();
        nvalid.push(autocorr(&mask));
    }
    let idiag = ignore_diags.max(0) as usize;
    let masked = |v: f64, d: usize| if d < idiag { f64::NAN } else { v };

    // genome-wide aggregate: sum both series over regions at each distance, smooth once, then
    // every region reports the same value at that distance (cooltools merges the agg curve on dist)
    let xs_all: Vec<f64> = (0..dmax).map(|d| d as f64).collect();
    let mut agg_sum = vec![0f64; dmax];
    let mut agg_nval = vec![0f64; dmax];
    for ri in 0..nreg {
        for d in 0..reg_len[ri] {
            if d >= idiag { agg_sum[d] += sumb[ri][d]; }
            agg_nval[d] += nvalid[ri][d];
        }
    }
    let (agg_s, agg_v) = log_smooth_pair(&xs_all, &agg_sum, &agg_nval);
    let agg_avg: Vec<f64> = (0..dmax).map(|d| agg_s[d] / agg_v[d]).collect();

    let (mut o_reg, mut o_dist, mut o_ntot, mut o_nval) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut o_csum, mut o_cavg, mut o_sum, mut o_avg) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut o_sm, mut o_smagg) = (Vec::new(), Vec::new());
    for ri in 0..nreg {
        let l = reg_len[ri];
        let xs: Vec<f64> = (0..l).map(|d| d as f64).collect();
        // per-region smoothing, on the same (sum, n_valid) pair, ignored diagonals as 0
        let sm_in: Vec<f64> = (0..l).map(|d| if d < idiag { 0.0 } else { sumb[ri][d] }).collect();
        let (sm_s, sm_v) = log_smooth_pair(&xs, &sm_in, &nvalid[ri]);
        for d in 0..l {
            let nv = nvalid[ri][d];
            let nt = (l - d) as f64;
            o_reg.push(ri as i32); o_dist.push(d as i64);
            o_ntot.push(nt); o_nval.push(nv);
            o_csum.push(masked(sumc[ri][d], d));
            o_cavg.push(masked(sumc[ri][d] / nt, d));
            o_sum.push(masked(sumb[ri][d], d));
            o_avg.push(masked(if nv > 0.0 { sumb[ri][d] / nv } else { f64::NAN }, d));
            // the smoothed curves are deliberately NOT masked on ignored diagonals: the Gaussian
            // window fills those in from neighbouring distances, which is the point of smoothing.
            // dist 0 still comes out NaN on its own (log10(0) leaves it with no window).
            o_sm.push(sm_s[d] / sm_v[d]);
            o_smagg.push(agg_avg[d]);
        }
    }

    // store: {grp}/expected/{view}/weight (scoped: recomputing one view must not drop the others)
    let _ = g.unlink(&format!("expected/{}", view_name));
    let ge = g.create_group(&format!("expected/{}/weight", view_name))?;
    ge.new_dataset::<i32>().shape([o_reg.len()]).create("region_id")?.write(&ndarray::arr1(&o_reg))?;
    ge.new_dataset::<i64>().shape([o_dist.len()]).create("dist")?.write(&ndarray::arr1(&o_dist))?;
    let col = |name: &str, v: &[f64]| -> Result<()> {
        ge.new_dataset::<f64>().shape([v.len()]).shuffle().deflate(4).create(name)?
            .write(&ndarray::arr1(v))?;
        Ok(())
    };
    col("n_total", &o_ntot)?;
    col("n_valid", &o_nval)?;
    col("count.sum", &o_csum)?;
    col("count.avg", &o_cavg)?;
    col("balanced.sum", &o_sum)?;
    col("balanced.avg", &o_avg)?;
    col("balanced.avg.smoothed", &o_sm)?;
    col("balanced.avg.smoothed.agg", &o_smagg)?;
    // what a consumer should use by default, mirroring cooltools' "contact_frequency"
    ge.new_attr::<hdf5::types::VarLenAscii>().create("default_column")?
        .write_scalar(&hdf5::types::VarLenAscii::from_ascii("balanced.avg.smoothed.agg")?)?;
    ge.new_attr::<f64>().create("smooth_sigma")?.write_scalar(&0.1f64)?;
    ge.new_attr::<i64>().create("ignore_diags")?.write_scalar(&ignore_diags)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values produced by cooltools' own smoother (cooltools 0.7.1):
    ///   from cooltools.sandbox.expected_smoothing import log_smooth
    ///   xs = np.arange(12.0); sm = np.linspace(1, 12, 12); nv = np.full(12, 10.0)
    ///   log_smooth(xs, [sm, nv], sigma_log10=0.1, window_sigma=5, points_per_sigma=10)
    /// Pinned here so a refactor of the kernel cannot silently drift away from cooltools.
    #[test]
    fn log_smoothing_matches_cooltools_reference() {
        let xs: Vec<f64> = (0..12).map(|d| d as f64).collect();
        let sum: Vec<f64> = (0..12).map(|d| (d + 1) as f64).collect();
        let nval = vec![10.0f64; 12];
        let (s, v) = log_smooth_pair(&xs, &sum, &nval);
        let want_s = [0.0, 2.0106780713396764, 3.1815689640695157, 4.256973276759025,
                      5.329797111975757, 6.411757698746538, 7.4695854241760085,
                      8.442141916493815, 9.254735954755823, 9.880058265634439,
                      10.340225151643418, 10.674910402595378];
        for (i, w) in want_s.iter().enumerate() {
            assert!((s[i] - w).abs() <= 1e-9 * w.abs().max(1e-9),
                "smoothed sum at dist {}: {} vs cooltools {}", i, s[i], w);
        }
        // n_valid is constant, so its smoothing must reproduce it (except at dist 0, which has
        // no window: log10(0) is -inf, and cooltools leaves it at zero -> NaN average)
        assert_eq!(v[0], 0.0);
        for i in 1..12 {
            assert!((v[i] - 10.0).abs() < 1e-9, "smoothed n_valid at {} = {}", i, v[i]);
        }
        assert!((s[0] / v[0]).is_nan(), "dist 0 must come out NaN, as in cooltools");
    }

    #[test]
    fn log_thin_is_uniform_in_log_space() {
        let xs: Vec<f64> = (0..1000).map(|d| d as f64).collect();
        let t = log_thin(&xs, 0.01);
        assert_eq!(t[0], 0.0, "keeps the first point");
        assert_eq!(*t.last().unwrap(), 999.0, "always keeps the last point");
        assert!(t.len() < xs.len(), "thinning must actually thin ({} of {})", t.len(), xs.len());
        assert!(t.windows(2).all(|w| w[1] > w[0]), "strictly increasing");
        // consecutive kept points are at least the requested log10 step apart (past the origin)
        for w in t.windows(2).skip(2) {
            assert!(w[1] >= w[0] * 10f64.powf(0.01) - 1e-9 || w[1] == *t.last().unwrap());
        }
    }

    #[test]
    fn log_interp_hits_knots_exactly() {
        // knot 0 is ln(0) = -inf; interpolating there must not produce NaN from a -inf slope
        let xp = [0.0, 1.0, 2.0, 8.0];
        let fp = [0.0, 2.0, 4.0, 16.0];
        let got = log_interp(&[0.0, 1.0, 2.0, 8.0], &xp, &fp);
        assert_eq!(got[0], 0.0);
        for (i, w) in [0.0, 2.0, 4.0, 16.0].iter().enumerate() {
            assert!((got[i] - w).abs() < 1e-12, "knot {} -> {} want {}", i, got[i], w);
        }
        // geometric interpolation between knots: f(4) on a pure power law is exact
        let mid = log_interp(&[4.0], &xp, &fp)[0];
        assert!((mid - 8.0).abs() < 1e-12, "log-log interp at 4 = {} want 8", mid);
    }
}
