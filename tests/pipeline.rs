//! End-to-end pipeline test: synthetic pairs -> cload -> merge -> zoomify -> balance -> expected.
//!
//! Every stage is checked against an independent in-test oracle (a HashMap/dense-matrix
//! recomputation), so this needs no python, no cooler, and no reference files. It is the
//! regression net for the whole op chain: if a refactor breaks binning, merge aggregation,
//! coarsening, IC or expected, one of these assertions fires.
use anyhow::Result;
use rooler::{balance, cload, cooler::Comp, expected, merge, zoomify};
use std::collections::HashMap;
use std::io::Write;

// hg38-prefix chromsizes so the assembly fingerprint (chr1 length) resolves -> no --assembly needed
const CHR1: i64 = 248_956_422;
const CHR2: i64 = 242_193_529;
const BINSIZE: i64 = 10_000;
const SPAN: i64 = 3_000_000; // positions live in the first 3Mb of each chrom

/// deterministic pseudo-random source, so a failure is always reproducible
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, n: u64) -> i64 { (self.next() % n) as i64 }
}

struct Case {
    dir: std::path::PathBuf,
    pairs: std::path::PathBuf,
    /// oracle: (bin1, bin2) -> count, computed independently of rooler
    expect: HashMap<(i64, i64), i64>,
    nbins: i64,
    nbins1: i64, // bins on chr1
}

fn nb(len: i64, res: i64) -> i64 { (len + res - 1) / res }

/// Write a synthetic .pairs file and compute the expected pixel table by hand.
fn make_case(tag: &str, npairs: usize, seed: u64) -> Result<Case> {
    let dir = std::env::temp_dir().join(format!("rooler_e2e_{}_{}", tag, std::process::id()));
    std::fs::remove_dir_all(&dir).ok(); // start from a clean slate even after a failed run
    std::fs::create_dir_all(&dir)?;
    let pairs = dir.join("in.pairs");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&pairs)?);
    writeln!(w, "## pairs format v1.0")?;
    writeln!(w, "#columns: readID chr1 pos1 chr2 pos2 strand1 strand2")?;
    writeln!(w, "#chromsize: chr1 {}", CHR1)?;
    writeln!(w, "#chromsize: chr2 {}", CHR2)?;

    let nbins1 = nb(CHR1, BINSIZE);
    let nbins = nbins1 + nb(CHR2, BINSIZE);
    let mut expect: HashMap<(i64, i64), i64> = HashMap::new();
    let mut g = Lcg(seed);
    for i in 0..npairs {
        // 70% cis (half chr1, half chr2), 30% trans; duplicates arise naturally from the
        // small position span, which is what exercises drain-and-count aggregation
        let roll = g.below(10);
        let (c1, p1, c2, p2) = if roll < 5 {
            (0, g.below(SPAN as u64), 0, g.below(SPAN as u64))
        } else if roll < 7 {
            (1, g.below(SPAN as u64), 1, g.below(SPAN as u64))
        } else {
            (0, g.below(SPAN as u64), 1, g.below(SPAN as u64))
        };
        let name = |c: i64| if c == 0 { "chr1" } else { "chr2" };
        writeln!(w, "r{}\t{}\t{}\t{}\t{}\t+\t-", i, name(c1), p1, name(c2), p2)?;
        let off = |c: i64| if c == 0 { 0 } else { nbins1 };
        let b1 = off(c1) + p1 / BINSIZE;
        let b2 = off(c2) + p2 / BINSIZE;
        let (lo, hi) = if b1 <= b2 { (b1, b2) } else { (b2, b1) };
        *expect.entry((lo, hi)).or_insert(0) += 1;
    }
    w.flush()?;
    Ok(Case { dir, pairs, expect, nbins, nbins1 })
}

/// read a cooler's pixel table as (bin1, bin2, count) vectors
fn read_pixels(path: &str, grp: &str) -> Result<(Vec<i64>, Vec<i64>, Vec<i32>)> {
    let f = hdf5::File::open(path)?;
    let g = if grp == "/" { f.group("/")? } else { f.group(grp)? };
    Ok((g.dataset("pixels/bin1_id")?.read_1d::<i64>()?.to_vec(),
        g.dataset("pixels/bin2_id")?.read_1d::<i64>()?.to_vec(),
        g.dataset("pixels/count")?.read_1d::<i32>()?.to_vec()))
}

fn read_f64(path: &str, grp: &str, name: &str) -> Result<Vec<f64>> {
    let f = hdf5::File::open(path)?;
    let g = if grp == "/" { f.group("/")? } else { f.group(grp)? };
    Ok(g.dataset(name)?.read_1d::<f64>()?.to_vec())
}

/// pixel table must be sorted, upper-triangular, and exactly equal to the oracle
fn assert_matches_oracle(got: &(Vec<i64>, Vec<i64>, Vec<i32>), want: &HashMap<(i64, i64), i64>, scale: i64, ctx: &str) {
    let (b1, b2, c) = got;
    assert_eq!(b1.len(), want.len(), "{}: pixel count", ctx);
    let mut prev = (-1i64, -1i64);
    for i in 0..b1.len() {
        assert!(b1[i] <= b2[i], "{}: pixel {} not upper-triangular ({},{})", ctx, i, b1[i], b2[i]);
        assert!((b1[i], b2[i]) > prev, "{}: pixels not sorted at {}", ctx, i);
        prev = (b1[i], b2[i]);
        let w = want.get(&(b1[i], b2[i]))
            .unwrap_or_else(|| panic!("{}: unexpected pixel ({},{})", ctx, b1[i], b2[i]));
        assert_eq!(c[i] as i64, w * scale, "{}: count at ({},{})", ctx, b1[i], b2[i]);
    }
}

#[test]
fn full_pipeline_matches_oracles() -> Result<()> {
    let case = make_case("full", 200_000, 20260809)?;
    let d = &case.dir;
    let p = |n: &str| d.join(n).to_str().unwrap().to_string();

    // ---- 1. cload, with a tiny --mem so phase A spills many runs (out-of-core path) ----
    let base = p("base.cool");
    let tmp = p("runs");
    let nnz = cload::cload(case.pairs.to_str().unwrap(), BINSIZE, &base, 0.001, 4,
        Comp::parse("blosc:zstd:1")?, &tmp, None, false)?;
    assert_eq!(nnz as usize, case.expect.len(), "cload nnz");
    let got = read_pixels(&base, "/")?;
    assert_matches_oracle(&got, &case.expect, 1, "cload");
    // total counts conserved
    assert_eq!(got.2.iter().map(|&x| x as i64).sum::<i64>(), 200_000, "cload conserves pairs");
    // the spill dir must be cleaned up
    assert!(!std::path::Path::new(&tmp).exists(), "cload left its spill dir behind");
    // assembly was fingerprinted from chr1's length, not left blank
    assert_eq!(rooler::cooler::read_meta(&base, None)?.assembly, "hg38");

    // ---- 2. merge: the same cooler twice -> every count doubled, same keys ----
    let merged = p("merged.cool");
    merge::merge_coolers(&[base.clone(), base.clone()], None, &merged, 0.01,
        Comp::parse("blosc:zstd:1")?, None, false)?;
    assert_matches_oracle(&read_pixels(&merged, "/")?, &case.expect, 2, "merge(serial)");
    // ranged-parallel merge must agree with the serial one exactly
    let merged_par = p("merged_par.cool");
    merge::merge_coolers_parallel(&[base.clone(), base.clone()], None, &merged_par, 0.01, 4,
        Comp::parse("blosc:zstd:1")?, None, false)?;
    assert_eq!(read_pixels(&merged_par, "/")?, read_pixels(&merged, "/")?, "parallel merge != serial merge");

    // ---- 3. zoomify: coarse level must equal brute-force coarsening of the fine level ----
    let mc = p("out.mcool");
    let coarse = BINSIZE * 4;
    zoomify::zoomify(&base, &mc, Some(vec![BINSIZE, coarse]), Comp::parse("blosc:zstd:1")?, None, false)?;
    let g_fine = format!("resolutions/{}", BINSIZE);
    let g_coarse = format!("resolutions/{}", coarse);
    // base level is a faithful copy
    assert_matches_oracle(&read_pixels(&mc, &g_fine)?, &case.expect, 1, "zoomify base");
    // oracle for the coarse level, built from the fine oracle via the same chrom-aware map
    let cn1 = nb(CHR1, coarse);
    let fine1 = case.nbins1;
    let cmap = |b: i64| -> i64 {
        if b < fine1 { b * BINSIZE / coarse } else { cn1 + (b - fine1) * BINSIZE / coarse }
    };
    let mut want_coarse: HashMap<(i64, i64), i64> = HashMap::new();
    for (&(b1, b2), &v) in &case.expect {
        let (x, y) = (cmap(b1), cmap(b2));
        let (lo, hi) = if x <= y { (x, y) } else { (y, x) };
        *want_coarse.entry((lo, hi)).or_insert(0) += v;
    }
    let got_coarse = read_pixels(&mc, &g_coarse)?;
    assert_matches_oracle(&got_coarse, &want_coarse, 1, "zoomify coarse");
    assert_eq!(got_coarse.2.iter().map(|&x| x as i64).sum::<i64>(), 200_000, "zoomify conserves counts");

    // ---- 4. balance: IC must actually equalize the marginals of unmasked bins ----
    let uri = format!("{}::{}", mc, g_fine);
    balance::balance(&uri, balance::Params {
        ignore_diags: 0, mad_max: 0.0, min_nnz: 2.0, min_count: 0.0,
        tol: 1e-8, max_iters: 500, nthreads: 4, tiled_block: None }, false)?;
    let w = read_f64(&mc, &g_fine, "bins/weight")?;
    assert_eq!(w.len(), case.nbins as usize);
    let ngood = w.iter().filter(|x| x.is_finite()).count();
    assert!(ngood > 100, "balance masked nearly everything ({} good bins)", ngood);
    // recompute balanced marginals straight from the pixel table
    let (b1, b2, c) = &got;
    let mut marg = vec![0f64; case.nbins as usize];
    for i in 0..b1.len() {
        let (x, y) = (b1[i] as usize, b2[i] as usize);
        if !w[x].is_finite() || !w[y].is_finite() { continue; }
        let v = c[i] as f64 * w[x] * w[y];
        marg[x] += v;
        if x != y { marg[y] += v; }
    }
    let nz: Vec<f64> = marg.iter().cloned().filter(|&v| v != 0.0).collect();
    let mean = nz.iter().sum::<f64>() / nz.len() as f64;
    let cv = (nz.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / nz.len() as f64).sqrt() / mean;
    assert!(cv < 1e-3, "balanced marginals not flat: CV={:.3e}", cv);
    // cooler convention: weights are multiplicative and the marginal is normalized to ~1
    assert!((mean - 1.0).abs() < 1e-2, "balanced marginal mean {:.6} != ~1", mean);

    // the tiled kernel must agree with the row-chunk kernel it replaces
    let w_row = w.clone();
    balance::balance(&uri, balance::Params {
        ignore_diags: 0, mad_max: 0.0, min_nnz: 2.0, min_count: 0.0,
        tol: 1e-8, max_iters: 500, nthreads: 4, tiled_block: Some(64) }, false)?;
    let w_tiled = read_f64(&mc, &g_fine, "bins/weight")?;
    for i in 0..w_row.len() {
        assert_eq!(w_row[i].is_finite(), w_tiled[i].is_finite(), "mask differs at bin {}", i);
        if w_row[i].is_finite() {
            let rel = (w_row[i] - w_tiled[i]).abs() / w_row[i].abs();
            assert!(rel < 1e-9, "tiled vs row weight at bin {}: {} vs {}", i, w_row[i], w_tiled[i]);
        }
    }

    // ---- 5. expected: compare against a brute-force O(n^2) computation for chr1 ----
    expected::expected(&uri, Some("chroms"), false)?;
    // NOTE: every HDF5 handle (File *and* Group/Dataset) must be dropped before the next op
    // reopens the file read-write, so these reads live in their own scope.
    let (reg, dist, nval, sum) = {
        let f = hdf5::File::open(&mc)?;
        let ge = f.group(&format!("{}/expected/chroms/weight", g_fine))?;
        (ge.dataset("region_id")?.read_1d::<i32>()?.to_vec(),
         ge.dataset("dist")?.read_1d::<i64>()?.to_vec(),
         ge.dataset("n_valid")?.read_1d::<f64>()?.to_vec(),
         ge.dataset("balanced.sum")?.read_1d::<f64>()?.to_vec())
    };
    // brute force over chr1 (region 0) using the tiled weights currently stored
    let wt = &w_tiled;
    let n1 = fine1 as usize;
    let mut want_sum = vec![0f64; n1];
    for i in 0..b1.len() {
        let (x, y) = (b1[i] as usize, b2[i] as usize);
        if x >= n1 || y >= n1 { continue; }             // cis chr1 only
        if !wt[x].is_finite() || !wt[y].is_finite() { continue; }
        want_sum[y - x] += c[i] as f64 * wt[x] * wt[y];
    }
    let mut want_nvalid = vec![0f64; n1];
    for s in 0..n1 {
        let mut n = 0f64;
        for i in 0..(n1 - s) {
            if wt[i].is_finite() && wt[i + s].is_finite() { n += 1.0; }
        }
        want_nvalid[s] = n;
    }
    let mut checked = 0;
    for k in 0..reg.len() {
        if reg[k] != 0 { continue; }
        let s = dist[k] as usize;
        assert_eq!(nval[k], want_nvalid[s], "n_valid at dist {}", s);
        let (a, b) = (sum[k], want_sum[s]);
        assert!((a - b).abs() <= 1e-9 * b.abs().max(1e-12), "balanced.sum at dist {}: {} vs {}", s, a, b);
        checked += 1;
    }
    assert_eq!(checked, n1, "expected must cover every chr1 distance");

    // recomputing another view must not drop the first (P1.7)
    expected::expected(&uri, Some("arms"), false)?;
    {
        let f = hdf5::File::open(&mc)?;
        assert!(f.link_exists(&format!("{}/expected/chroms/weight", g_fine)), "recomputing a view dropped the other");
        assert!(f.link_exists(&format!("{}/expected/arms/weight", g_fine)));
    }

    // ---- 6. custom BED view: expected must honour it and store it under its own name ----
    let bed = d.join("myview.bed");
    std::fs::write(&bed, format!("chr1\t0\t1000000\tfirstMb\nchr1\t1000000\t2000000\tsecondMb\n"))?;
    expected::expected(&uri, Some(&format!("custom:{}", bed.display())), false)?;
    {
        let f = hdf5::File::open(&mc)?;
        let ge = f.group(&format!("{}/expected/myview/weight", g_fine))?;
        let reg = ge.dataset("region_id")?.read_1d::<i32>()?.to_vec();
        assert_eq!(*reg.iter().max().unwrap(), 1, "custom view should have exactly 2 regions");
        // each region is 1Mb / 10kb = 100 bins -> 100 distances each
        assert_eq!(reg.len(), 200, "custom view rows");
        let gv = f.group(&format!("{}/views/myview", g_fine))?;
        assert_eq!(gv.dataset("start")?.read_1d::<i64>()?.to_vec(), vec![0, 1_000_000]);
        assert!(f.link_exists(&format!("{}/expected/chroms/weight", g_fine)), "custom view dropped the others");
    }

    std::fs::remove_dir_all(d).ok();
    Ok(())
}

/// `zoomify --balance` must leave a converged weight vector at every resolution.
#[test]
fn zoomify_balance_covers_every_level() -> Result<()> {
    let case = make_case("zb", 100_000, 4242)?;
    let d = &case.dir;
    let p = |n: &str| d.join(n).to_str().unwrap().to_string();
    let base = p("base.cool");
    cload::cload(case.pairs.to_str().unwrap(), BINSIZE, &base, 0.01, 4,
        Comp::parse("blosc:zstd:1")?, &p("runs"), None, false)?;
    let mc = p("out.mcool");
    let levels = vec![BINSIZE, BINSIZE * 2, BINSIZE * 4];
    zoomify::zoomify_and_balance(&base, &mc, Some(levels.clone()), Comp::parse("blosc:zstd:1")?,
        None, true, 4, false)?;
    for res in &levels {
        let g = format!("resolutions/{}", res);
        let w = read_f64(&mc, &g, "bins/weight")?;
        assert!(w.iter().any(|x| x.is_finite()), "{}bp has no finite weights", res);
        let f = hdf5::File::open(&mc)?;
        let conv = f.group(&g)?.dataset("bins/weight")?.attr("converged")?
            .read_scalar::<hdf5::types::VarLenAscii>()?;
        assert_eq!(conv.as_str(), "True", "{}bp did not converge", res);
    }
    std::fs::remove_dir_all(d).ok();
    Ok(())
}

/// cload must be invariant to --mem / thread count: the same input always yields the same cooler.
#[test]
fn cload_is_deterministic_across_mem_and_threads() -> Result<()> {
    let case = make_case("determ", 50_000, 777)?;
    let d = &case.dir;
    let p = |n: &str| d.join(n).to_str().unwrap().to_string();
    let src = case.pairs.to_str().unwrap();

    let a = p("a.cool");
    cload::cload(src, BINSIZE, &a, 0.0005, 1, Comp::parse("none")?, &p("ra"), None, false)?;
    let b = p("b.cool");
    cload::cload(src, BINSIZE, &b, 4.0, 8, Comp::parse("gzip1")?, &p("rb"), None, false)?;

    let (pa, pb) = (read_pixels(&a, "/")?, read_pixels(&b, "/")?);
    assert_eq!(pa, pb, "cload output depends on --mem/--threads");
    assert_matches_oracle(&pa, &case.expect, 1, "cload(determinism)");
    std::fs::remove_dir_all(d).ok();
    Ok(())
}

/// Guard rails: the ops must refuse rather than silently produce wrong output.
#[test]
fn ops_refuse_bad_input() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("rooler_e2e_guard_{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let p = |n: &str| dir.join(n).to_str().unwrap().to_string();

    // unknown genome + no --assembly => refuse to write a mystery cooler
    let pairs = dir.join("weird.pairs");
    std::fs::write(&pairs, "#chromsize: scaffold_1 12345\nr\tscaffold_1\t1\tscaffold_1\t2\t+\t+\n")?;
    let e = cload::cload(pairs.to_str().unwrap(), 100, &p("w.cool"), 0.01, 1,
        Comp::parse("none")?, &p("rw"), None, false).unwrap_err();
    assert!(e.to_string().contains("assembly"), "expected an assembly refusal, got: {}", e);
    // ... but an explicit assembly is accepted
    cload::cload(pairs.to_str().unwrap(), 100, &p("w.cool"), 0.01, 1,
        Comp::parse("none")?, &p("rw2"), Some("myGenome"), false)?;
    assert_eq!(rooler::cooler::read_meta(&p("w.cool"), None)?.assembly, "myGenome");

    // merging coolers with different bin layouts must error, not silently corrupt
    let pairs2 = dir.join("w2.pairs");
    std::fs::write(&pairs2, "#chromsize: scaffold_1 12345\n#chromsize: scaffold_2 999\nr\tscaffold_1\t1\tscaffold_1\t2\t+\t+\n")?;
    cload::cload(pairs2.to_str().unwrap(), 100, &p("w2.cool"), 0.01, 1,
        Comp::parse("none")?, &p("rw3"), Some("myGenome"), false)?;
    let e = merge::merge_coolers(&[p("w.cool"), p("w2.cool")], None, &p("bad.cool"), 0.01,
        Comp::parse("none")?, Some("myGenome"), false).unwrap_err();
    assert!(e.to_string().contains("merge:"), "expected a merge mismatch error, got: {}", e);

    // expected on an unbalanced cooler must say so, not produce garbage
    let e = expected::expected(&p("w.cool"), Some("chroms"), false).unwrap_err();
    assert!(e.to_string().contains("not balanced"), "unexpected error: {}", e);

    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}
