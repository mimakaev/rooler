//! Cooler v3 schema: streaming writer + pixel reader. Integer counts (i32) for v1.
use anyhow::{anyhow, bail, Result};
use hdf5::types::FixedAscii;
use hdf5::{File, Group};
use ndarray::{arr1, s, Array1};

/// Compression preset for pixel/bin datasets.
#[derive(Clone, Copy)]
pub enum Comp {
    None,
    Gzip(u8),
    BloscZstd(u8),
    BloscLz4(u8),
}
impl Comp {
    /// "gzip[0-9] | blosc:zstd:N | blosc:lz4:N | none". Unknown presets are an error — a typo
    /// must never silently fall back to a different codec.
    pub fn parse(s: &str) -> Result<Comp> {
        const USAGE: &str = "use gzip[0-9] | blosc:zstd:N | blosc:lz4:N | none";
        let lvl = |x: &str, max: u8| -> Result<u8> {
            let v: u8 = x.parse().map_err(|_| anyhow!("bad compression level {:?} ({})", x, USAGE))?;
            if v > max { bail!("compression level {} out of range 0..={} ({})", v, max, USAGE); }
            Ok(v)
        };
        if s == "none" {
            Ok(Comp::None)
        } else if let Some(l) = s.strip_prefix("gzip") {
            Ok(Comp::Gzip(if l.is_empty() { 4 } else { lvl(l, 9)? }))
        } else if let Some(rest) = s.strip_prefix("blosc:") {
            let mut it = rest.split(':');
            let cname = it.next().unwrap_or("");
            let cl = match it.next() { Some(x) => lvl(x, 9)?, None => 1 };
            if it.next().is_some() { bail!("unknown compression preset {:?} ({})", s, USAGE); }
            match cname {
                "lz4" => Ok(Comp::BloscLz4(cl)),
                "zstd" => Ok(Comp::BloscZstd(cl)),
                _ => bail!("unknown blosc codec {:?} ({})", cname, USAGE),
            }
        } else {
            bail!("unknown compression preset {:?} ({})", s, USAGE)
        }
    }
}

macro_rules! cfg_comp {
    ($b:expr, $c:expr) => {{
        let mut b = $b;
        match $c {
            Comp::None => {}
            Comp::Gzip(l) => { b = b.shuffle().deflate(l); }
            Comp::BloscZstd(l) => { b = b.blosc_zstd(l, hdf5::filters::BloscShuffle::Byte); }
            Comp::BloscLz4(l) => { b = b.blosc_lz4(l, hdf5::filters::BloscShuffle::Byte); }
        }
        b
    }};
}

/// Saturating i64 -> i32 count cast; bumps *nclamped when the value doesn't fit.
/// Counts are stored as i32 by design (bandwidth/cache); a pixel with >2.1e9 reads is an
/// extreme outlier, so we saturate (never wrap) and report how many pixels were affected.
#[inline]
pub fn clamp_count(v: i64, nclamped: &mut u64) -> i32 {
    if v > i32::MAX as i64 { *nclamped += 1; i32::MAX } else { v as i32 }
}

/// Bin-pair keys are `bin1*nbins + bin2` in i64; past ~3.04e9 bins (a ~1bp genome-wide binsize)
/// that multiplication overflows silently. Refuse up front instead.
pub fn check_key_space(nbins: i64) -> Result<()> {
    const MAX_NBINS: i64 = 3_037_000_499; // floor(sqrt(i64::MAX))
    if nbins > MAX_NBINS {
        bail!("{} bins: bin-pair keys (bin1*nbins+bin2) would overflow i64; \
               the maximum supported bin count is {} (use a coarser binsize)", nbins, MAX_NBINS);
    }
    Ok(())
}

/// Deletes `path` on drop unless defused — armed right after an output file is created so a
/// failed op never leaves a valid-looking but incomplete cooler behind, and never touches a
/// pre-existing file the op hadn't started writing.
pub(crate) struct PartialFileGuard<'a> {
    path: &'a str,
    armed: bool,
}
impl<'a> PartialFileGuard<'a> {
    pub(crate) fn new(path: &'a str) -> Self { PartialFileGuard { path, armed: false } }
    pub(crate) fn arm(&mut self) { self.armed = true; }
    pub(crate) fn defuse(&mut self) { self.armed = false; }
}
impl Drop for PartialFileGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            eprintln!("  removing incomplete output {}", self.path);
            let _ = std::fs::remove_file(self.path);
        }
    }
}

/// Metadata needed to construct an output matching a set of inputs.
pub struct Meta {
    pub names: Vec<String>,
    pub lengths: Vec<i64>,
    pub chrom_offset: Vec<i64>,
    pub nbins: usize,
    pub binsize: i64,
    pub assembly: String,
}

impl Meta {
    pub fn chromsizes(&self) -> Vec<(String, i64)> {
        self.names.iter().cloned().zip(self.lengths.iter().copied()).collect()
    }
}

/// "file.mcool::resolutions/1000" -> (path, group); a bare path means the root group "/".
pub fn split_uri(uri: &str) -> (String, String) {
    match uri.split_once("::") {
        Some((p, g)) if !g.is_empty() => (p.to_string(), g.to_string()),
        Some((p, _)) => (p.to_string(), "/".to_string()),
        None => (uri.to_string(), "/".to_string()),
    }
}

/// First bin of every chromosome, plus the total as the last element (cooler's chrom_offset).
pub fn bin_offsets(lengths: &[i64], binsize: i64) -> Vec<i64> {
    let mut off = vec![0i64];
    for &l in lengths { off.push(off.last().unwrap() + (l + binsize - 1) / binsize); }
    off
}

/// Read a scalar string attribute whatever its HDF5 flavour: cooler/h5py write variable-length
/// UTF-8, older rooler files variable-length ASCII, other tools fixed-length. None = absent.
pub fn read_str_attr(loc: &hdf5::Location, name: &str) -> Option<String> {
    use hdf5::types::{VarLenAscii, VarLenUnicode};
    let a = loc.attr(name).ok()?;
    if let Ok(v) = a.read_scalar::<VarLenUnicode>() { return Some(v.as_str().to_string()); }
    if let Ok(v) = a.read_scalar::<VarLenAscii>() { return Some(v.as_str().to_string()); }
    if let Ok(v) = a.read_scalar::<FixedAscii<64>>() { return Some(v.as_str().to_string()); }
    None
}

/// Write a scalar string attribute the way h5py does (variable-length UTF-8), replacing any
/// existing one.
pub fn write_str_attr(loc: &hdf5::Location, name: &str, val: &str) -> Result<()> {
    let v: hdf5::types::VarLenUnicode = val.parse()
        .map_err(|e| anyhow!("attribute {} = {:?} is not storable: {}", name, val, e))?;
    if loc.attr(name).is_ok() {
        // the crate has no attribute delete; a string of a different length cannot be
        // overwritten in place, so drop the old attribute first
        let cname = std::ffi::CString::new(name)?;
        let rc = hdf5::sync::sync(|| unsafe { hdf5_metno_sys::h5a::H5Adelete(loc.id(), cname.as_ptr()) });
        if rc < 0 { bail!("could not replace attribute {}", name); }
    }
    loc.new_attr::<hdf5::types::VarLenUnicode>().create(name)?.write_scalar(&v)?;
    Ok(())
}

/// Create an empty .mcool container: root attributes + the `resolutions` group.
pub fn create_mcool(path: &str) -> Result<File> {
    let f = File::create(path)?;
    write_str_attr(&f, "format", "HDF5::MCOOL")?;
    f.new_attr::<i64>().create("format-version")?.write_scalar(&2i64)?;
    f.create_group("resolutions")?;
    Ok(f)
}

/// `datetime.isoformat()`-style UTC timestamp for a unix time (days -> civil date, no deps).
fn iso_utc(unix_secs: u64) -> String {
    let (days, rem) = ((unix_secs / 86_400) as i64, unix_secs % 86_400);
    let z = days + 719_468;
    let (era, doe) = (z.div_euclid(146_097), z.rem_euclid(146_097));
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let (d, m) = (doy - (153 * mp + 2) / 5 + 1, if mp < 10 { mp + 3 } else { mp - 9 });
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}+00:00", y, m, d, rem / 3600, rem % 3600 / 60, rem % 60)
}

/// Refuse cooler variants rooler would otherwise mis-handle silently.
///
/// rooler assumes the common shape: symmetric-upper storage, fixed-width bins, integer counts.
/// Everything else is legal cooler, and reading it anyway would corrupt data rather than fail —
/// below-diagonal pixels get double-counted by the symmetric kernels and relabelled
/// "symmetric-upper" on write, and float counts get truncated to int32. So check up front.
pub fn check_supported(g: &Group) -> Result<()> {
    let sattr = |n: &str| read_str_attr(g, n);
    match sattr("storage-mode").as_deref() {
        None | Some("symmetric-upper") => {}
        Some(m) => bail!("storage-mode {:?} is not supported (rooler assumes symmetric-upper, \
                          i.e. only the upper triangle is stored). Reading it would double-count \
                          the pixels below the diagonal.", m),
    }
    match sattr("bin-type").as_deref() {
        None | Some("fixed") => {}
        Some(t) => bail!("bin-type {:?} is not supported; rooler needs fixed-width bins", t),
    }
    if let Ok(d) = g.dataset("pixels/count") {
        let td = d.dtype()?.to_descriptor()?;
        if !matches!(td, hdf5::types::TypeDescriptor::Integer(_) | hdf5::types::TypeDescriptor::Unsigned(_)) {
            bail!("pixels/count has dtype {} ; rooler stores integer counts and would truncate \
                   fractional values. Round or rescale them first.", td);
        }
    }
    Ok(())
}

/// Read the layout of the cooler in group `grp` ("/" or e.g. "resolutions/1000") of `path`.
pub fn read_meta(path: &str, grp: &str) -> Result<Meta> {
    let f = File::open(path)?;
    let g = f.group(grp)?;
    check_supported(&g)?;
    let names: Vec<String> = g.dataset("chroms/name")?.read_1d::<FixedAscii<64>>()
        .map_err(|e| anyhow!("{}: cannot read chroms/name (names over 64 bytes are not supported): {}", path, e))?
        .iter().map(|s| s.to_string()).collect();
    let lengths: Vec<i64> = g.dataset("chroms/length")?.read_1d::<i64>()?.to_vec();
    let chrom_offset: Vec<i64> = g.dataset("indexes/chrom_offset")?.read_1d::<i64>()?.to_vec();
    let nbins = g.dataset("bins/start")?.shape()[0];
    let binsize = g.attr("bin-size")?.read_scalar::<i64>()?;
    if binsize <= 0 { bail!("{}: bin-size {} is not positive", path, binsize); }
    // every op rebuilds the bin table from (chromsizes, binsize); a file whose table is laid
    // out differently would be silently re-binned, so it is refused here
    if chrom_offset != bin_offsets(&lengths, binsize) || nbins as i64 != *chrom_offset.last().unwrap_or(&0) {
        bail!("{}::{}: bin table is not the regular fixed-width layout for bin-size {}", path, grp, binsize);
    }
    let assembly = read_str_attr(&g, "genome-assembly").unwrap_or_default();
    Ok(Meta { names, lengths, chrom_offset, nbins, binsize, assembly })
}

/// Lazy block reader over a cooler's sorted pixel table, yielding (key=bin1*nbins+bin2, count) blocks.
pub struct CoolerPix {
    b1: hdf5::Dataset,
    b2: hdf5::Dataset,
    cn: hdf5::Dataset,
    nbins: i64,
    end: usize,
    pos: usize,
    block: usize,
    _f: File,
}
impl CoolerPix {
    pub fn open(path: &str, grp: &str, nbins: usize, block: usize) -> Result<CoolerPix> {
        let f = File::open(path)?;
        let g = f.group(grp)?;
        let cn = g.dataset("pixels/count")?;
        let end = cn.shape()[0];
        Ok(CoolerPix {
            b1: g.dataset("pixels/bin1_id")?, b2: g.dataset("pixels/bin2_id")?, cn,
            nbins: nbins as i64, end, pos: 0, block, _f: f,
        })
    }
    /// reader confined to pixel range [p0, p1) — for a bin1-range partition (ranged-parallel merge).
    pub fn open_slice(path: &str, grp: &str, nbins: usize, block: usize, p0: usize, p1: usize) -> Result<CoolerPix> {
        let mut c = Self::open(path, grp, nbins, block)?;
        c.pos = p0; c.end = p1; Ok(c)
    }
    /// bin1_offset index for this cooler (for slicing bin1 ranges into pixel ranges).
    pub fn bin1_offset(path: &str, grp: &str) -> Result<Vec<i64>> {
        let f = File::open(path)?; let g = f.group(grp)?;
        let mut v = g.dataset("indexes/bin1_offset")?.read_1d::<i64>()?.to_vec();
        let nbins = g.dataset("bins/start")?.shape()[0];
        if v.len() == nbins { v.push(g.dataset("pixels/count")?.shape()[0] as i64); }
        Ok(v)
    }
}
impl crate::merge::BlockSource for CoolerPix {
    fn next(&mut self) -> Result<Option<(Vec<i64>, crate::merge::Counts)>> {
        if self.pos >= self.end { return Ok(None); }
        let hi = std::cmp::min(self.pos + self.block, self.end);
        let b1 = self.b1.read_slice_1d::<i64, _>(self.pos..hi)?;
        let b2 = self.b2.read_slice_1d::<i64, _>(self.pos..hi)?;
        let cn = self.cn.read_slice_1d::<i32, _>(self.pos..hi)?;
        let keys: Vec<i64> = (0..b1.len()).map(|i| b1[i] * self.nbins + b2[i]).collect();
        let cnts: Vec<i64> = cn.iter().map(|&x| x as i64).collect();
        self.pos = hi;
        Ok(Some((keys, crate::merge::Counts::PerKey(cnts))))
    }
}

/// Where appended pixels go. Gzip gets the parallel direct-chunk path (deflate is single-threaded
/// inside HDF5 and dominates `--compat` writes); blosc/none keep HDF5's own write path.
enum PixSink {
    Hdf5 { d1: hdf5::Dataset, d2: hdf5::Dataset, dc: hdf5::Dataset },
    Par { c1: crate::parwrite::ParColumn, c2: crate::parwrite::ParColumn, cc: crate::parwrite::ParColumn },
}

/// Streaming writer: append (bin1,bin2,count) blocks in ascending-bin1 order; bounded RAM.
/// Writes into a target group (root "/" for a flat .cool, "resolutions/{res}" for an mcool).
pub struct CoolWriter {
    _f: File,      // keep the file handle alive
    g: Group,      // target group
    pix: PixSink,
    nbins: usize,
    chrom_offset: Vec<i64>,
    bincount: Vec<i64>,
    pub nnz: usize,
    sum: i64,
    last_bin1: i64,
}

impl CoolWriter {
    pub fn create(
        path: &str, names: &[String], lengths: &[i64], binsize: i64, comp: Comp, assembly: &str,
    ) -> Result<CoolWriter> {
        let f = File::create(path)?;
        Self::create_in(&f, "/", names, lengths, binsize, comp, assembly)
    }

    /// Write a cooler into `group` ("/" for flat .cool, "resolutions/{res}" for an mcool level).
    /// The bin table is the regular fixed-width one implied by (lengths, binsize).
    pub fn create_in(
        file: &File, group: &str, names: &[String], lengths: &[i64], binsize: i64,
        comp: Comp, assembly: &str,
    ) -> Result<CoolWriter> {
        if binsize <= 0 { bail!("binsize must be positive, got {}", binsize); }
        // chrom lengths and bin coordinates are stored int32 (cooler schema); refuse loudly
        // instead of silently wrapping to negative coordinates (lungfish-class chromosomes)
        for (n, &l) in names.iter().zip(lengths) {
            if l > i32::MAX as i64 {
                bail!("chromosome {:?} is {} bp; the cooler format stores coordinates as int32 (max {})",
                      n, l, i32::MAX);
            }
            if l < 0 { bail!("chromosome {:?} has negative length {}", n, l); }
        }
        let chrom_offset = bin_offsets(lengths, binsize);
        let nbins = *chrom_offset.last().unwrap() as usize;
        let g = if group == "/" { file.group("/")? } else { file.create_group(group)? };
        // the full attribute set cooler's own writer produces (cooler.create.write_info), so
        // `Cooler.info` has every key downstream code may index — `sum` and `nnz` included;
        // those two are placeholders here and get their real values in close()
        let sattr = |name: &str, val: &str| write_str_attr(&g, name, val);
        let iattr = |name: &str, val: i64| -> Result<()> {
            g.new_attr::<i64>().create(name)?.write_scalar(&val)?; Ok(())
        };
        sattr("format", "HDF5::Cooler")?;
        iattr("format-version", 3)?;
        sattr("format-url", "https://github.com/open2c/cooler")?;
        sattr("bin-type", "fixed")?;
        iattr("bin-size", binsize)?;
        sattr("storage-mode", "symmetric-upper")?;
        iattr("nbins", nbins as i64)?;
        iattr("nchroms", names.len() as i64)?;
        iattr("nnz", 0)?;
        iattr("sum", 0)?;
        sattr("genome-assembly", assembly)?;
        sattr("generated-by", concat!("rooler-", env!("CARGO_PKG_VERSION")))?;
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs()).unwrap_or(0);
        sattr("creation-date", &iso_utc(now))?;
        sattr("metadata", "{}")?;

        // chroms
        let gc = g.create_group("chroms")?;
        write_fixed_ascii(&gc, "name", names)?;
        gc.new_dataset::<i32>().shape([names.len()]).create("length")?
            .write(&Array1::from(lengths.iter().map(|&l| l as i32).collect::<Vec<_>>()))?;

        // bins
        let gb = g.create_group("bins")?;
        let mut cids = vec![0i32; nbins];
        let mut starts = vec![0i32; nbins];
        let mut ends = vec![0i32; nbins];
        for cid in 0..names.len() {
            let (lo, hi) = (chrom_offset[cid] as usize, chrom_offset[cid + 1] as usize);
            let l = lengths[cid];
            for (k, b) in (lo..hi).enumerate() {
                cids[b] = cid as i32;
                let s0 = (k as i64) * binsize;
                starts[b] = s0 as i32;
                ends[b] = std::cmp::min(s0 + binsize, l) as i32;
            }
        }
        write_chrom_codes(&gb, &cids, names, comp)?;
        build_i32(&gb, "start", &starts, comp)?;
        build_i32(&gb, "end", &ends, comp)?;

        // pixels (resizable, streaming). gzip uses 256K-element chunks (the parallel-writer
        // sweet spot: 64K-256K is fastest AND compresses better than cooler's ~60KB default);
        // blosc keeps 1M-element chunks, which its own path handles well.
        let gp = g.create_group("pixels")?;
        let pix = match comp {
            Comp::Gzip(level) => {
                use crate::parwrite::ParColumn;
                let chunk = 1usize << 18;
                PixSink::Par {
                    c1: ParColumn::new(res_i64(&gp, "bin1_id", chunk, comp)?, 8, chunk, level)?,
                    c2: ParColumn::new(res_i64(&gp, "bin2_id", chunk, comp)?, 8, chunk, level)?,
                    cc: ParColumn::new(res_i32(&gp, "count", chunk, comp)?, 4, chunk, level)?,
                }
            }
            _ => {
                let chunk = 1usize << 20;
                PixSink::Hdf5 {
                    d1: res_i64(&gp, "bin1_id", chunk, comp)?,
                    d2: res_i64(&gp, "bin2_id", chunk, comp)?,
                    dc: res_i32(&gp, "count", chunk, comp)?,
                }
            }
        };

        Ok(CoolWriter {
            _f: file.clone(), g, pix, nbins, chrom_offset,
            bincount: vec![0i64; nbins], nnz: 0, sum: 0, last_bin1: -1,
        })
    }

    pub fn append(&mut self, bin1: &[i64], bin2: &[i64], count: &[i32]) -> Result<()> {
        let n = bin1.len();
        if n == 0 { return Ok(()); }
        // once per multi-million-pixel block: free, and catches a caller that breaks bin1 order
        if bin1[0] < self.last_bin1 {
            bail!("append: blocks must be non-decreasing in bin1 ({} after {})", bin1[0], self.last_bin1);
        }
        self.last_bin1 = bin1[n - 1];
        let (lo, hi) = (self.nnz, self.nnz + n);
        match &mut self.pix {
            PixSink::Hdf5 { d1, d2, dc } => {
                d1.resize([hi])?; d1.write_slice(&arr1(bin1), s![lo..hi])?;
                d2.resize([hi])?; d2.write_slice(&arr1(bin2), s![lo..hi])?;
                dc.resize([hi])?; dc.write_slice(&arr1(count), s![lo..hi])?;
            }
            PixSink::Par { c1, c2, cc } => {
                c1.push_i64(bin1)?; c2.push_i64(bin2)?; cc.push_i32(count)?;
            }
        }
        for &b in bin1 { self.bincount[b as usize] += 1; }
        self.sum += count.iter().map(|&c| c as i64).sum::<i64>();
        self.nnz = hi;
        Ok(())
    }

    pub fn close(mut self) -> Result<()> {
        // flush the parallel writer's tail (partial chunk) before the index/attrs are written
        if let PixSink::Par { c1, c2, cc } = &mut self.pix {
            c1.finish()?; c2.finish()?; cc.finish()?;
        }
        let mut bin1_offset = vec![0i64; self.nbins + 1];
        let mut acc = 0i64;
        for i in 0..self.nbins { bin1_offset[i] = acc; acc += self.bincount[i]; }
        bin1_offset[self.nbins] = acc;
        let gi = self.g.create_group("indexes")?;
        gi.new_dataset::<i64>().shape([self.nbins + 1]).create("bin1_offset")?
            .write(&arr1(&bin1_offset))?;
        gi.new_dataset::<i64>().shape([self.chrom_offset.len()]).create("chrom_offset")?
            .write(&arr1(&self.chrom_offset))?;
        self.g.attr("nnz")?.write_scalar(&(self.nnz as i64))?;
        self.g.attr("sum")?.write_scalar(&self.sum)?;
        // file flushes/closes when the last handle (driver's) drops; don't close here.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cooler variants rooler cannot handle must be refused, not silently mis-read: a square
    /// cooler would have its below-diagonal pixels double-counted and then be relabelled
    /// symmetric-upper, and float counts would be truncated to int32.
    #[test]
    fn unsupported_cooler_variants_are_refused() {
        let d = tmpdir("supported");
        let names = vec!["chrA".to_string()];
        // a normal rooler cooler passes
        let p = d.join("ok.cool");
        {
            let mut w = CoolWriter::create(p.to_str().unwrap(), &names, &[100], 10, Comp::None, "test").unwrap();
            w.append(&[0], &[1], &[3]).unwrap();
            w.close().unwrap();
        }
        {
            let f = File::open(p.to_str().unwrap()).unwrap();
            check_supported(&f.group("/").unwrap()).expect("a plain rooler cooler must pass");
        }
        // flip storage-mode to "square" -> refused, naming the mode
        {
            let f = File::append(p.to_str().unwrap()).unwrap();
            let g = f.group("/").unwrap();
            write_str_attr(&g, "storage-mode", "square").unwrap();
            let e = check_supported(&g).unwrap_err().to_string();
            assert!(e.contains("square") && e.contains("symmetric-upper"), "{}", e);
        }
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn iso_utc_formats_known_instants() {
        assert_eq!(iso_utc(0), "1970-01-01T00:00:00+00:00");
        assert_eq!(iso_utc(1_700_000_000), "2023-11-14T22:13:20+00:00");
        assert_eq!(iso_utc(951_782_400), "2000-02-29T00:00:00+00:00"); // leap day
    }

    #[test]
    fn split_uri_defaults_to_root() {
        assert_eq!(split_uri("a.cool"), ("a.cool".into(), "/".into()));
        assert_eq!(split_uri("a.mcool::resolutions/100"), ("a.mcool".into(), "resolutions/100".into()));
        assert_eq!(split_uri("a.mcool::/resolutions/100"), ("a.mcool".into(), "/resolutions/100".into()));
    }

    #[test]
    fn comp_parse_accepts_valid_presets() {
        assert!(matches!(Comp::parse("none").unwrap(), Comp::None));
        assert!(matches!(Comp::parse("gzip").unwrap(), Comp::Gzip(4)));
        assert!(matches!(Comp::parse("gzip4").unwrap(), Comp::Gzip(4)));
        assert!(matches!(Comp::parse("gzip0").unwrap(), Comp::Gzip(0)));
        assert!(matches!(Comp::parse("gzip9").unwrap(), Comp::Gzip(9)));
        assert!(matches!(Comp::parse("blosc:zstd:1").unwrap(), Comp::BloscZstd(1)));
        assert!(matches!(Comp::parse("blosc:lz4:5").unwrap(), Comp::BloscLz4(5)));
        assert!(matches!(Comp::parse("blosc:zstd").unwrap(), Comp::BloscZstd(1)));
    }

    #[test]
    fn comp_parse_rejects_typos_instead_of_falling_back() {
        // the whole point: a typo must not silently produce a different codec
        for bad in ["gizp4", "blosc:snappy:1", "gzip99", "gzip300", "zstd", "", "blosc:", "blosc:zstd:1:2"] {
            assert!(Comp::parse(bad).is_err(), "{:?} should be rejected", bad);
        }
    }

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("rooler_test_{}_{}", tag, std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn create_rejects_oversize_chromosomes_and_names() {
        let d = tmpdir("guards");
        let p = d.join("t.cool");
        let ps = p.to_str().unwrap();
        let err_of = |r: Result<CoolWriter>| match r { Err(e) => e, Ok(_) => panic!("expected an error") };
        // > 2^31-1 bp cannot be stored in the schema's int32 coordinates -> refuse, don't wrap
        let e = err_of(CoolWriter::create(ps, &["chrBig".to_string()], &[3_000_000_000], 1_000_000,
            Comp::None, "t"));
        assert!(e.to_string().contains("int32"), "unexpected error: {}", e);
        // names longer than 64 bytes are refused with the name in the message, not panicked on
        let long = "x".repeat(65);
        let e = err_of(CoolWriter::create(ps, &[long], &[100], 10, Comp::None, "t"));
        assert!(e.to_string().contains("64"), "unexpected error: {}", e);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn append_rejects_out_of_order_blocks() {
        let d = tmpdir("append");
        let p = d.join("t.cool");
        let names = vec!["chrA".to_string()];
        let mut w = CoolWriter::create(p.to_str().unwrap(), &names, &[100], 10, Comp::None, "test").unwrap();
        w.append(&[0, 1, 5], &[0, 3, 5], &[1, 1, 1]).unwrap();
        // going backwards in bin1 would silently corrupt the bin1_offset index
        let e = w.append(&[4], &[9], &[1]).unwrap_err();
        assert!(e.to_string().contains("non-decreasing"), "unexpected error: {}", e);
        // continuing forward from the last bin1 is fine (blocks may share a bin1)
        w.append(&[5, 6], &[7, 6], &[1, 1]).unwrap();
        w.close().unwrap();
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn writer_roundtrips_pixels_and_indexes() {
        let d = tmpdir("roundtrip");
        let p = d.join("t.cool");
        let names = vec!["chrA".to_string(), "chrB".to_string()];
        // chrA: 4 bins (35bp @10), chrB: 2 bins (20bp @10)
        let (b1, b2, c) = (vec![0i64, 0, 1, 4], vec![0i64, 3, 1, 5], vec![5i32, 2, 7, 3]);
        {
            let mut w = CoolWriter::create(p.to_str().unwrap(), &names, &[35, 20], 10, Comp::Gzip(1), "test").unwrap();
            w.append(&b1[..3], &b2[..3], &c[..3]).unwrap();
            w.append(&b1[3..], &b2[3..], &c[3..]).unwrap();
            assert_eq!(w.nnz, 4);
            w.close().unwrap();
        }
        let f = hdf5::File::open(p.to_str().unwrap()).unwrap();
        let g = f.group("/").unwrap();
        assert_eq!(g.dataset("pixels/bin1_id").unwrap().read_1d::<i64>().unwrap().to_vec(), b1);
        assert_eq!(g.dataset("pixels/bin2_id").unwrap().read_1d::<i64>().unwrap().to_vec(), b2);
        assert_eq!(g.dataset("pixels/count").unwrap().read_1d::<i32>().unwrap().to_vec(), c);
        // bin1_offset must be a valid CSR row pointer over the pixels
        let off = g.dataset("indexes/bin1_offset").unwrap().read_1d::<i64>().unwrap().to_vec();
        assert_eq!(off, vec![0, 2, 3, 3, 3, 4, 4]);
        assert!(off.windows(2).all(|w| w[0] <= w[1]), "bin1_offset must be non-decreasing");
        assert_eq!(*off.last().unwrap(), 4);
        // bins are built from chromsizes: chrA 0-10-20-30-35, chrB 0-10-20
        assert_eq!(g.dataset("bins/start").unwrap().read_1d::<i32>().unwrap().to_vec(), vec![0, 10, 20, 30, 0, 10]);
        assert_eq!(g.dataset("bins/end").unwrap().read_1d::<i32>().unwrap().to_vec(), vec![10, 20, 30, 35, 10, 20]);
        assert_eq!(g.dataset("bins/chrom").unwrap().read_1d::<i32>().unwrap().to_vec(), vec![0, 0, 0, 0, 1, 1]);
        // bins/chrom must be a real HDF5 ENUM of the chrom names (the cooler schema type)
        match g.dataset("bins/chrom").unwrap().dtype().unwrap().to_descriptor().unwrap() {
            hdf5::types::TypeDescriptor::Enum(e) => {
                let names: Vec<&str> = e.members.iter().map(|m| m.name.as_str()).collect();
                assert_eq!(names, vec!["chrA", "chrB"]);
                assert_eq!(e.members[1].value, 1);
            }
            other => panic!("bins/chrom is not an enum: {:?}", other),
        }
        // the attribute set cooler's own writer produces: Cooler.info must have every key
        assert_eq!(g.attr("nnz").unwrap().read_scalar::<i64>().unwrap(), 4);
        assert_eq!(g.attr("sum").unwrap().read_scalar::<i64>().unwrap(), 17);
        assert_eq!(g.attr("nbins").unwrap().read_scalar::<i64>().unwrap(), 6);
        assert_eq!(g.attr("nchroms").unwrap().read_scalar::<i64>().unwrap(), 2);
        assert_eq!(g.attr("bin-size").unwrap().read_scalar::<i64>().unwrap(), 10);
        assert_eq!(g.attr("format-version").unwrap().read_scalar::<i64>().unwrap(), 3);
        for (k, want) in [("format", "HDF5::Cooler"), ("bin-type", "fixed"), ("metadata", "{}"),
                          ("storage-mode", "symmetric-upper"), ("genome-assembly", "test"),
                          ("format-url", "https://github.com/open2c/cooler")] {
            assert_eq!(read_str_attr(&g, k).as_deref(), Some(want), "attribute {}", k);
        }
        assert!(read_str_attr(&g, "generated-by").unwrap().starts_with("rooler-"));
        assert!(read_str_attr(&g, "creation-date").unwrap().starts_with("20"));
        drop(f);
        std::fs::remove_dir_all(&d).ok();
    }
}

/// `bins/chrom` as a real HDF5 ENUM of the chromosome names over int32 codes — the cooler
/// schema type (h5py's enum_dtype). The crate can't build runtime enums, so this is raw FFI;
/// every call runs on this thread under the crate's global lock. Values are identical to the
/// plain int32 codes we wrote before — only the type gains the name mapping.
fn write_chrom_codes(g: &Group, cids: &[i32], names: &[String], comp: Comp) -> Result<()> {
    use hdf5_metno_sys::h5d::{H5Dclose, H5Dcreate2, H5Dwrite};
    use hdf5_metno_sys::h5i::hid_t;
    use hdf5_metno_sys::h5p::{H5P_DEFAULT, H5Pclose, H5Pcreate, H5Pset_chunk, H5Pset_deflate,
                              H5Pset_shuffle, H5P_CLS_DATASET_CREATE};
    use hdf5_metno_sys::h5s::{H5S_ALL, H5Sclose, H5Screate_simple};
    use hdf5_metno_sys::h5t::{H5T_NATIVE_INT32, H5Tclose, H5Tenum_create, H5Tenum_insert};
    let cname = std::ffi::CString::new("chrom")?;
    let n = cids.len();
    hdf5::sync::sync(|| -> Result<()> {
        unsafe {
            let tid = H5Tenum_create(*H5T_NATIVE_INT32);
            if tid < 0 { bail!("H5Tenum_create failed"); }
            // close ids on every exit path
            let fin = |tid: hid_t, sid: hid_t, dcpl: hid_t, did: hid_t| {
                if did >= 0 { H5Dclose(did); }
                if dcpl >= 0 { H5Pclose(dcpl); }
                if sid >= 0 { H5Sclose(sid); }
                H5Tclose(tid);
            };
            for (i, nm) in names.iter().enumerate() {
                let c = std::ffi::CString::new(nm.as_str())?;
                let v: i32 = i as i32;
                if H5Tenum_insert(tid, c.as_ptr(), (&v as *const i32).cast()) < 0 {
                    fin(tid, -1, -1, -1);
                    bail!("H5Tenum_insert failed for chromosome {:?}", nm);
                }
            }
            let dims = [n as u64];
            let sid = H5Screate_simple(1, dims.as_ptr(), std::ptr::null());
            if sid < 0 { fin(tid, -1, -1, -1); bail!("H5Screate_simple failed"); }
            // small dataset: shuffle+deflate for any compressed preset (plain HDF5, no plugins)
            let dcpl = match comp {
                Comp::None => H5P_DEFAULT,
                _ => {
                    let p = H5Pcreate(*H5P_CLS_DATASET_CREATE);
                    let chunk = [n.min(1 << 20) as u64];
                    if p < 0 || H5Pset_chunk(p, 1, chunk.as_ptr()) < 0
                        || H5Pset_shuffle(p) < 0 || H5Pset_deflate(p, 4) < 0 {
                        fin(tid, sid, p, -1);
                        bail!("dcpl setup failed for bins/chrom");
                    }
                    p
                }
            };
            let did = H5Dcreate2(g.id() as hid_t, cname.as_ptr(), tid, sid,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
            if did < 0 { fin(tid, sid, dcpl, -1); bail!("H5Dcreate2 failed for bins/chrom"); }
            let rc = if n > 0 {
                H5Dwrite(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, cids.as_ptr().cast())
            } else { 0 };
            fin(tid, sid, if dcpl == H5P_DEFAULT { -1 } else { dcpl }, did);
            if rc < 0 { bail!("H5Dwrite failed for bins/chrom"); }
            Ok(())
        }
    })
}

/// Fixed-length ASCII string column (chromosome and region names), sized to the longest value.
pub(crate) fn write_fixed_ascii(g: &Group, name: &str, vals: &[String]) -> Result<()> {
    let long = vals.iter().max_by_key(|s| s.len()).map(|s| s.as_str()).unwrap_or("");
    let maxlen = long.len();
    if maxlen > 64 {
        bail!("name {:?} is {} bytes; at most 64 supported", long, long.len());
    }
    macro_rules! w { ($n:expr) => {{
        let a: Vec<FixedAscii<$n>> = vals.iter()
            .map(|s| FixedAscii::<$n>::from_ascii(s.as_bytes())
                .map_err(|e| anyhow!("name {:?} is not ASCII: {}", s, e)))
            .collect::<Result<_>>()?;
        g.new_dataset::<FixedAscii<$n>>().shape([vals.len()]).create(name)?.write(&arr1(&a))?;
    }}}
    // round maxlen up to a supported const size
    match maxlen { 0..=8 => w!(8), 9..=16 => w!(16), 17..=32 => w!(32), _ => w!(64) }
    Ok(())
}
fn build_i32(g: &Group, name: &str, data: &[i32], c: Comp) -> Result<()> {
    let b = cfg_comp!(g.new_dataset::<i32>().shape([data.len()]), c);
    b.create(name)?.write(&arr1(data))?; Ok(())
}
fn res_i64(g: &Group, name: &str, chunk: usize, c: Comp) -> Result<hdf5::Dataset> {
    let b = cfg_comp!(g.new_dataset::<i64>().shape((0.., )).chunk([chunk]), c);
    Ok(b.create(name)?)
}
fn res_i32(g: &Group, name: &str, chunk: usize, c: Comp) -> Result<hdf5::Dataset> {
    let b = cfg_comp!(g.new_dataset::<i32>().shape((0.., )).chunk([chunk]), c);
    Ok(b.create(name)?)
}
