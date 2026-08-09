//! repack: rewrite an existing cooler/mcool the way rooler would have written it — parallel-gzip
//! default preset (plugin-free, small chunks, smaller files), genome assembly stamped AND verified
//! against the chromosome sizes, balanced if it carries no weights, expected computed. The one-shot
//! "make this old cooler nice" op. In place by default (atomic tmp+rename); `--backup` keeps the
//! original at `<src>.bac`; `--out` writes elsewhere and leaves the source untouched.
use crate::cooler::{read_meta, Comp, CoolWriter, PartialFileGuard};
use anyhow::{anyhow, bail, Result};
use hdf5::File;

pub struct RepackOpts {
    pub out: Option<String>,
    pub backup: bool,
    pub assembly: Option<String>,
    pub comp: Comp,
    pub nthreads: usize,
    pub mem_gb: f64,
    pub expected: bool,
}

pub fn repack(src: &str, o: RepackOpts, log: bool) -> Result<()> {
    let t0 = std::time::Instant::now();
    // enumerate resolution groups (mcool) or the single root cooler
    let resolutions: Vec<Option<String>> = {
        let f = File::open(src)?;
        if f.link_exists("resolutions") {
            let mut names = f.group("resolutions")?.member_names()?;
            names.sort_by_key(|s| s.parse::<i64>().unwrap_or(i64::MAX));
            if names.is_empty() { bail!("{}: empty resolutions group", src); }
            names.into_iter().map(Some).collect()
        } else {
            vec![None]
        }
    };
    let is_mcool = resolutions[0].is_some();

    // resolve the assembly (override > stored > fingerprint > refuse), then VERIFY it: if the
    // name maps to a genome we know AND the chromsizes fingerprint to a genome we know, they
    // must agree — `repack --assembly hg19` on an hg38 file is exactly the mistake to catch.
    let meta0 = read_meta(src, resolutions[0].as_deref())?;
    let chromsizes: Vec<(String, i64)> = meta0.names.iter().cloned().zip(meta0.lengths.iter().cloned()).collect();
    let asm = match &o.assembly {
        Some(a) if !a.trim().is_empty() => a.trim().to_string(),
        _ if !meta0.assembly.is_empty() && meta0.assembly != "unknown" => meta0.assembly.clone(),
        _ => crate::view::detect("", &chromsizes).map(|s| s.to_string()).ok_or_else(|| anyhow!(
            "refusing to repack without a genome assembly: the file has none and the chromsizes \
             are not recognized; pass --assembly <name>"))?,
    };
    let by_name = crate::view::detect(&asm, &[]);
    let by_fp = crate::view::detect("", &chromsizes);
    if let (Some(a), Some(b)) = (by_name, by_fp) {
        if a != b {
            bail!("assembly {:?} contradicts the file's chromosome sizes (they match {}); \
                   not stamping a wrong provenance", asm, b);
        }
    }
    if log { eprintln!("  repack: {} ({} resolution{}), assembly={}{}", src, resolutions.len(),
        if resolutions.len() == 1 { "" } else { "s" }, asm,
        if by_name.is_some() && by_fp.is_some() { " (verified against chromsizes)" } else { "" }); }

    let final_path = o.out.clone().unwrap_or_else(|| src.to_string());
    let tmp = format!("{}.repack_tmp", final_path);
    let mut guard = PartialFileGuard::new(&tmp);

    // ---- write phase: pixels + weights, one output handle, dropped before balance/expected ----
    let mut had_weight = vec![false; resolutions.len()];
    {
        let sf = File::open(src)?;
        let f = File::create(&tmp)?;
        guard.arm();
        if is_mcool {
            f.new_attr::<hdf5::types::VarLenAscii>().create("format")?
                .write_scalar(&hdf5::types::VarLenAscii::from_ascii("HDF5::MCOOL")?)?;
            f.new_attr::<i64>().create("format-version")?.write_scalar(&2i64)?;
            f.create_group("resolutions")?;
        }
        for (ri, res) in resolutions.iter().enumerate() {
            let meta = read_meta(src, res.as_deref())?;
            let group = match res { Some(r) => format!("resolutions/{}", r), None => "/".to_string() };
            let sg = if group == "/" { sf.group("/")? } else { sf.group(&group)? };
            let mut w = CoolWriter::create_in(&f, &group, &meta.names, &meta.lengths, meta.binsize,
                meta.nbins, &meta.chrom_offset, o.comp, &asm)?;
            crate::parread::stream_pixels(&sg, 1 << 22, |a, b, c| w.append(a, b, c))?;
            let n = w.nnz;
            w.close()?;
            let dg = if group == "/" { f.group("/")? } else { f.group(&group)? };
            had_weight[ri] = copy_weight(&sg, &dg)?;
            if log { eprintln!("  repack: {} rewritten ({} pix{}) {:.0}s",
                res.as_deref().map(|r| format!("{}bp", r)).unwrap_or_else(|| "cooler".into()),
                n, if had_weight[ri] { ", weights carried" } else { "" }, t0.elapsed().as_secs_f64()); }
        }
    }

    // ---- move into place (tmp rename is atomic within a filesystem) ----
    if o.out.is_none() {
        if o.backup {
            let bac = format!("{}.bac", src);
            std::fs::rename(src, &bac)?;
            if log { eprintln!("  repack: original kept at {}", bac); }
        }
        std::fs::rename(&tmp, src)?;
    } else {
        std::fs::rename(&tmp, &final_path)?;
    }
    guard.defuse();

    // ---- balance (only if the source had no weights) + expected, per resolution ----
    for (ri, res) in resolutions.iter().enumerate() {
        let uri = match res { Some(r) => format!("{}::resolutions/{}", final_path, r), None => final_path.clone() };
        if !had_weight[ri] {
            if log { eprintln!("  repack: {} has no weights -> balancing", uri); }
            crate::balance::balance(&uri, crate::balance::Params {
                nthreads: o.nthreads, mem_gb: o.mem_gb, ..Default::default() }, log)?;
        }
        if o.expected { crate::expected::expected_or_warn(&uri, log); }
    }
    if log { eprintln!("  repack DONE: {} in {:.0}s", final_path, t0.elapsed().as_secs_f64()); }
    Ok(())
}

/// Copy bins/weight (values + scalar attrs) from the source group into the freshly written one.
/// Attr types vary by writer (rooler: ascii "True"; cooler: numpy bools/ints/floats) — try the
/// common scalar types in turn and skip anything unreadable rather than fail the repack.
fn copy_weight(sg: &hdf5::Group, dg: &hdf5::Group) -> Result<bool> {
    if !sg.link_exists("bins/weight") { return Ok(false); }
    let sd = sg.dataset("bins/weight")?;
    let v = sd.read_1d::<f64>()?;
    let dd = dg.group("bins")?.new_dataset::<f64>().shape([v.len()]).shuffle().deflate(4)
        .create("weight")?;
    dd.write(&v)?;
    for name in sd.attr_names()? {
        let a = sd.attr(&name)?;
        if let Ok(x) = a.read_scalar::<i64>() {
            dd.new_attr::<i64>().create(name.as_str())?.write_scalar(&x)?;
        } else if let Ok(x) = a.read_scalar::<f64>() {
            dd.new_attr::<f64>().create(name.as_str())?.write_scalar(&x)?;
        } else if let Ok(x) = a.read_scalar::<hdf5::types::VarLenAscii>() {
            dd.new_attr::<hdf5::types::VarLenAscii>().create(name.as_str())?.write_scalar(&x)?;
        } else if let Ok(x) = a.read_scalar::<bool>() {
            dd.new_attr::<hdf5::types::VarLenAscii>().create(name.as_str())?
                .write_scalar(&hdf5::types::VarLenAscii::from_ascii(if x { "True" } else { "False" })?)?;
        } else {
            eprintln!("  repack: weight attr {:?} has an unsupported type; not carried", name);
        }
    }
    Ok(true)
}
