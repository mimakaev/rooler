//! repack: rewrite an existing cooler/mcool the way rooler would have written it — parallel-gzip
//! default preset (plugin-free, small chunks, smaller files), genome assembly stamped AND verified
//! against the chromosome sizes, balanced if it carries no weights, expected computed. The one-shot
//! "make this old cooler nice" op. In place by default (atomic tmp+rename); `--backup` keeps the
//! original at `<src>.bac`; `--out` writes elsewhere and leaves the source untouched.
use crate::cooler::{read_meta, read_str_attr, write_str_attr, Comp, CoolWriter, PartialFileGuard};
use anyhow::{bail, Result};
use hdf5::types::TypeDescriptor as T;
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
    let (is_mcool, groups): (bool, Vec<String>) = {
        let f = File::open(src)?;
        if f.link_exists("resolutions") {
            let mut names = f.group("resolutions")?.member_names()?;
            names.sort_by_key(|s| s.parse::<i64>().unwrap_or(i64::MAX));
            if names.is_empty() { bail!("{}: empty resolutions group", src); }
            (true, names.into_iter().map(|r| format!("resolutions/{}", r)).collect())
        } else {
            (false, vec!["/".to_string()])
        }
    };

    // resolve the assembly (override > stored > fingerprint > refuse), then VERIFY it: if the
    // name maps to a genome we know AND the chromsizes fingerprint to a genome we know, they
    // must agree — `repack --assembly hg19` on an hg38 file is exactly the mistake to catch.
    let meta0 = read_meta(src, &groups[0])?;
    let chromsizes = meta0.chromsizes();
    let asm = crate::view::resolve_assembly(o.assembly.as_deref(), &meta0.assembly, &chromsizes)?;
    let by_name = crate::view::detect(&asm, &[]);
    let by_fp = crate::view::detect("", &chromsizes);
    if let (Some(a), Some(b)) = (by_name, by_fp) {
        if a != b {
            bail!("assembly {:?} contradicts the file's chromosome sizes (they match {}); \
                   not stamping a wrong provenance", asm, b);
        }
    }
    if log { eprintln!("  repack: {} ({} resolution{}), assembly={}{}", src, groups.len(),
        if groups.len() == 1 { "" } else { "s" }, asm,
        if by_name.is_some() && by_fp.is_some() { " (verified against chromsizes)" } else { "" }); }

    let final_path = o.out.clone().unwrap_or_else(|| src.to_string());
    let tmp = format!("{}.repack_tmp", final_path);
    let mut guard = PartialFileGuard::new(&tmp);

    // ---- write phase: pixels + weights, one output handle, dropped before balance/expected ----
    let mut had_weight = vec![false; groups.len()];
    {
        let sf = File::open(src)?;
        let f = if is_mcool { crate::cooler::create_mcool(&tmp)? } else { File::create(&tmp)? };
        guard.arm();
        for (ri, group) in groups.iter().enumerate() {
            let meta = read_meta(src, group)?;
            let sg = sf.group(group)?;
            let mut w = CoolWriter::create_in(&f, group, &meta.names, &meta.lengths, meta.binsize,
                o.comp, &asm)?;
            crate::parread::stream_pixels(&sg, 1 << 22, |a, b, c| w.append(a, b, c))?;
            let n = w.nnz;
            w.close()?;
            let dg = f.group(group)?;
            // user metadata (a JSON string) survives the rewrite
            if let Some(m) = read_str_attr(&sg, "metadata") { write_str_attr(&dg, "metadata", &m)?; }
            had_weight[ri] = copy_bin_columns(&sg, &dg)?;
            if log { eprintln!("  repack: {} rewritten ({} pix{}) {:.0}s", group,
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
    for (ri, group) in groups.iter().enumerate() {
        let uri = format!("{}::{}", final_path, group);
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

/// Copy every bins/* column the writer does not create itself — `weight`, and anything a
/// different tool added (GC content, mappability, alternative weights) — with its attributes.
/// repack rewrites in place by default, so dropping or degrading them would be silent data loss
/// on exactly the files repack exists for. Returns whether the source had `bins/weight`.
fn copy_bin_columns(sg: &hdf5::Group, dg: &hdf5::Group) -> Result<bool> {
    let (sb, db) = (sg.group("bins")?, dg.group("bins")?);
    for name in sb.member_names()? {
        if matches!(name.as_str(), "chrom" | "start" | "end") { continue; }
        let Ok(sd) = sb.dataset(&name) else { continue };
        let n = sd.size();
        let mk = |b: hdf5::DatasetBuilderEmpty| b.shape([n]).shuffle().deflate(4);
        // dispatch on the STORED type: HDF5 would happily convert float<->int on read, so
        // "try a type and see if it reads" silently truncates
        let dd = match sd.dtype()?.to_descriptor() {
            Ok(T::Float(_)) => { let d = mk(db.new_dataset::<f64>()).create(name.as_str())?; d.write(&sd.read_1d::<f64>()?)?; d }
            Ok(T::Integer(_) | T::Unsigned(_)) => { let d = mk(db.new_dataset::<i64>()).create(name.as_str())?; d.write(&sd.read_1d::<i64>()?)?; d }
            Ok(T::Boolean) => { let d = mk(db.new_dataset::<bool>()).create(name.as_str())?; d.write(&sd.read_1d::<bool>()?)?; d }
            _ => {
                eprintln!("  repack: WARNING bins/{} has a type this build cannot copy; \
                           it is NOT carried into the output", name);
                continue;
            }
        };
        copy_scalar_attrs(&sd, &dd, &format!("bins/{}", name))?;
    }
    Ok(db.link_exists("weight"))
}

/// Copy scalar attributes, keeping each one's kind (int / float / bool / string) — these are the
/// balancing parameters on bins/weight. Files written by rooler before 0.1.0-alpha.2 stored the
/// boolean flags as the strings "True"/"False"; those are upgraded to real booleans here.
fn copy_scalar_attrs(src: &hdf5::Location, dst: &hdf5::Location, what: &str) -> Result<()> {
    for name in src.attr_names()? {
        let a = src.attr(&name)?;
        let n = name.as_str();
        match a.dtype()?.to_descriptor() {
            Ok(T::Integer(_) | T::Unsigned(_)) if a.is_scalar() =>
                dst.new_attr::<i64>().create(n)?.write_scalar(&a.read_scalar::<i64>()?)?,
            Ok(T::Float(_)) if a.is_scalar() =>
                dst.new_attr::<f64>().create(n)?.write_scalar(&a.read_scalar::<f64>()?)?,
            Ok(T::Boolean) if a.is_scalar() =>
                dst.new_attr::<bool>().create(n)?.write_scalar(&a.read_scalar::<bool>()?)?,
            _ => match read_str_attr(src, n).as_deref() {
                Some("True") => dst.new_attr::<bool>().create(n)?.write_scalar(&true)?,
                Some("False") => dst.new_attr::<bool>().create(n)?.write_scalar(&false)?,
                Some(s) => write_str_attr(dst, n, s)?,
                None => eprintln!("  repack: WARNING {} attribute {:?} has an unsupported type; not carried", what, name),
            },
        }
    }
    Ok(())
}
