//! balance: genome-wide iterative correction (IC) over the compressed CSR scratch (built once,
//! parallel SpMV per iteration). Scale-free stop (CV=std/mean < tol). Matches cooler weights.
use crate::scratch::{Scratch, SpMV};
use crate::scratch_tiled::TiledScratch;
use anyhow::Result;
use hdf5::File;

pub struct Params {
    pub ignore_diags: i64,
    pub mad_max: f64,
    pub min_nnz: f64,
    pub min_count: f64,
    pub tol: f64,   // scale-free: stop when CV=std/mean of the marginal < tol
    pub max_iters: usize,
    pub nthreads: usize,
    /// None = auto (tiled above 4M bins); Some(0) = force row-chunk; Some(b) = tiled, tile size b
    pub tiled_block: Option<i64>,
    /// RAM budget in GB. When the estimated scratch would exceed it, blobs go to a disk-backed
    /// mmap next to the cooler instead of the heap (identical results, page-cache speed when
    /// memory is free). Default 8: distiller runs ~8 balances in parallel on a 64GB node.
    pub mem_gb: f64,
}
impl Default for Params {
    fn default() -> Self { Params { ignore_diags: 2, mad_max: 5.0, min_nnz: 10.0, min_count: 0.0,
        tol: 1e-4, max_iters: 200, nthreads: 8, tiled_block: None, mem_gb: 8.0 } }
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

pub fn balance(uri: &str, p: Params, log: bool) -> Result<()> {
    let t0 = std::time::Instant::now();
    let (path, grp) = match uri.split_once("::") { Some((a, b)) => (a.to_string(), b.to_string()), None => (uri.to_string(), "/".to_string()) };
    let f = File::append(&path)?;
    let g = if grp == "/" { f.group("/")? } else { f.group(&grp)? };
    crate::cooler::check_supported(&g)?;
    // probe sizes (dataset shapes only) to route the kernel and place the scratch
    let nbins_probe = g.dataset("bins/start")?.shape()[0];
    let nnz_probe = g.dataset("pixels/count")?.shape()[0];
    // kernel auto-route: tiling pays once the O(nbins) SpMV vectors outgrow L3 — at 4M bins the
    // v + per-thread y vectors are 64MB, past any common L3 (measured 2.8x at 12.5M bins). The
    // tile size itself (65536 x 8B = 512KB slices) is the L2-sizing knob inside the kernel.
    let block = match p.tiled_block {
        Some(0) => None,
        Some(b) => Some(b),
        None => if nbins_probe >= 4_000_000 { Some(65536) } else { None },
    };
    // scratch placement: estimated compressed size (measured ~2.5 B/pix row / ~2.8 tiled; use a
    // margin) plus the O(nbins) vectors we hold regardless (rowptr, bias/y, per-thread partials)
    let est_scratch = (nnz_probe as f64 * if block.is_some() { 3.0 } else { 2.6 }) as usize;
    let est_fixed = (p.nthreads + 6) * nbins_probe * 8;
    let budget = (p.mem_gb * 1e9) as usize;
    let spill_prefix = if est_scratch + est_fixed > budget {
        let pfx = format!("{}.balance_scratch.{}", path, std::process::id());
        if log { eprintln!("  scratch est {:.1}GB + vectors {:.1}GB > --mem {:.1}GB -> disk-backed mmap scratch",
            est_scratch as f64 / 1e9, est_fixed as f64 / 1e9, p.mem_gb); }
        Some(pfx)
    } else { None };
    let sc: Box<dyn SpMV> = match block {
        Some(b) => Box::new(TiledScratch::build(&g, b, p.nthreads, spill_prefix.as_deref())?),
        None => Box::new(Scratch::build(&g, 262144, p.nthreads, spill_prefix.as_deref())?),
    };
    let nbins = sc.nbins();
    if log { eprintln!("  scratch({}{}): {} pix -> {:.1}GB ({:.2} B/pix) in {:.0}s",
        block.map(|b| format!("tiled B={}", b)).unwrap_or("row".into()),
        if spill_prefix.is_some() { ", disk" } else { "" },
        sc.nnz(), sc.comp_bytes() as f64 / 1e9, sc.comp_bytes() as f64 / sc.nnz().max(1) as f64, t0.elapsed().as_secs_f64()); }

    // mask from raw marginals
    let (mraw, mnnz) = sc.marginals(p.ignore_diags);
    let mut good = vec![true; nbins];
    for k in 0..nbins {
        if mnnz[k] < p.min_nnz { good[k] = false; }
        if p.min_count > 0.0 && mraw[k] < p.min_count { good[k] = false; }
    }
    if p.mad_max > 0.0 {
        let mut marg = mraw.clone();
        for c in 0..sc.chrom_offset().len() - 1 {
            let (lo, hi) = (sc.chrom_offset()[c] as usize, sc.chrom_offset()[c + 1] as usize);
            let mut pos: Vec<f64> = marg[lo..hi].iter().cloned().filter(|&x| x > 0.0).collect();
            let med = median(&mut pos);
            if med > 0.0 { for k in lo..hi { marg[k] /= med; } }
        }
        let lognz: Vec<f64> = marg.iter().filter(|&&x| x > 0.0).map(|x| x.ln()).collect();
        let medlog = median(&mut lognz.clone());
        let mut dev: Vec<f64> = lognz.iter().map(|x| (x - medlog).abs()).collect();
        let madlog = median(&mut dev);
        let cutoff = (medlog - p.mad_max * madlog).exp();
        for k in 0..nbins { if marg[k] < cutoff { good[k] = false; } }
    }
    let ngood = good.iter().filter(|&&x| x).count();
    if log { eprintln!("  mask: {}/{} bins ({:.0}s)", ngood, nbins, t0.elapsed().as_secs_f64()); }

    // IC iterations: marg_i = bias_i * SpMV(bias)_i
    let mut bias: Vec<f64> = good.iter().map(|&g| if g { 1.0 } else { 0.0 }).collect();
    let mut mean = 1.0f64; let mut cv = f64::NAN; let mut converged = false;
    for it in 0..p.max_iters {
        let y = sc.spmv(&bias, p.ignore_diags);
        let marg: Vec<f64> = (0..nbins).map(|i| bias[i] * y[i]).collect();
        let nz: Vec<f64> = marg.iter().cloned().filter(|&x| x != 0.0).collect();
        if nz.is_empty() { break; }
        let n = nz.len() as f64;
        mean = nz.iter().sum::<f64>() / n;
        let var = nz.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        cv = var.sqrt() / mean;
        for k in 0..nbins { if marg[k] != 0.0 { bias[k] /= marg[k] / mean; } }
        if log && (it % 5 == 0 || cv < p.tol) { eprintln!("  ic {:3} cv={:.3e}", it, cv); }
        if cv < p.tol { converged = true; break; }
    }

    let s = mean.sqrt();
    let weight: Vec<f64> = (0..nbins).map(|k| if good[k] { bias[k] / s } else { f64::NAN }).collect();

    let gb = g.group("bins")?;
    let _ = gb.unlink("weight");
    let dw = gb.new_dataset::<f64>().shape([nbins]).shuffle().deflate(4).create("weight")?;
    dw.write(&ndarray::arr1(&weight))?;
    let ai = |n: &str, v: i64| -> Result<()> { dw.new_attr::<i64>().create(n)?.write_scalar(&v)?; Ok(()) };
    let af = |n: &str, v: f64| -> Result<()> { dw.new_attr::<f64>().create(n)?.write_scalar(&v)?; Ok(()) };
    let ab = |n: &str, v: bool| -> Result<()> { dw.new_attr::<hdf5::types::VarLenAscii>().create(n)?.write_scalar(&hdf5::types::VarLenAscii::from_ascii(if v {"True"} else {"False"})?)?; Ok(()) };
    ab("converged", converged)?; ab("cis_only", false)?; ab("divisive_weights", false)?;
    ai("ignore_diags", p.ignore_diags)?; af("mad_max", p.mad_max)?; af("min_nnz", p.min_nnz)?;
    ai("min_count", p.min_count as i64)?; af("tol", p.tol)?; af("scale", mean)?; af("var", cv * cv)?;
    if log { eprintln!("  balance DONE: converged={} cv={:.2e} scale={:.1} in {:.0}s", converged, cv, mean, t0.elapsed().as_secs_f64()); }
    Ok(())
}
