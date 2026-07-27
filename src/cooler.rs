//! Cooler v3 schema: streaming writer + pixel reader. Integer counts (i32) for v1.
use anyhow::Result;
use hdf5::types::{EnumMember, EnumType, FixedAscii, IntSize, TypeDescriptor};
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
    pub fn parse(s: &str) -> Comp {
        if let Some(l) = s.strip_prefix("gzip") {
            Comp::Gzip(l.parse().unwrap_or(4))
        } else if let Some(rest) = s.strip_prefix("blosc:") {
            let mut it = rest.split(':');
            let cname = it.next().unwrap_or("zstd");
            let cl = it.next().and_then(|x| x.parse().ok()).unwrap_or(1);
            match cname { "lz4" => Comp::BloscLz4(cl), _ => Comp::BloscZstd(cl) }
        } else if s == "none" {
            Comp::None
        } else {
            Comp::BloscZstd(1)
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

fn chrom_enum(names: &[String]) -> TypeDescriptor {
    let members = names
        .iter()
        .enumerate()
        .map(|(i, n)| EnumMember { name: n.clone(), value: i as u64 })
        .collect();
    TypeDescriptor::Enum(EnumType { size: IntSize::U4, signed: true, members })
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

fn open_res<'a>(f: &'a File, res: Option<&str>) -> Result<Group> {
    match res {
        Some(r) => Ok(f.group(&format!("resolutions/{}", r))?),
        None => Ok(f.group("/")?),
    }
}

pub fn read_meta(path: &str, res: Option<&str>) -> Result<Meta> {
    let f = File::open(path)?;
    let g = open_res(&f, res)?;
    let raw = g.dataset("chroms/name")?.read_1d::<FixedAscii<64>>()
        .or_else(|_| g.dataset("chroms/name")?.read_1d::<FixedAscii<8>>().map(|a| a.mapv(|x| FixedAscii::<64>::from_ascii(x.as_bytes()).unwrap())))?;
    let names: Vec<String> = raw.iter().map(|s| s.to_string()).collect();
    let lengths: Vec<i64> = g.dataset("chroms/length")?.read_1d::<i32>()?.iter().map(|&x| x as i64).collect();
    let chrom_offset: Vec<i64> = g.dataset("indexes/chrom_offset")?.read_1d::<i64>()?.to_vec();
    let nbins = g.dataset("bins/start")?.shape()[0];
    let binsize = g.attr("bin-size").or_else(|_| f.attr("bin-size"))?.read_scalar::<i64>()?;
    let assembly = g.attr("genome-assembly").or_else(|_| f.attr("genome-assembly")).ok()
        .and_then(|a| a.read_scalar::<hdf5::types::VarLenAscii>().ok())
        .map(|s| s.as_str().to_string()).unwrap_or_default();
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
    pub fn open(path: &str, res: Option<&str>, nbins: usize, block: usize) -> Result<CoolerPix> {
        let f = File::open(path)?;
        let g = open_res(&f, res)?;
        let cn = g.dataset("pixels/count")?;
        let end = cn.shape()[0];
        Ok(CoolerPix {
            b1: g.dataset("pixels/bin1_id")?, b2: g.dataset("pixels/bin2_id")?, cn,
            nbins: nbins as i64, end, pos: 0, block, _f: f,
        })
    }
    /// reader confined to pixel range [p0, p1) — for a bin1-range partition (ranged-parallel merge).
    pub fn open_slice(path: &str, res: Option<&str>, nbins: usize, block: usize, p0: usize, p1: usize) -> Result<CoolerPix> {
        let mut c = Self::open(path, res, nbins, block)?;
        c.pos = p0; c.end = p1; Ok(c)
    }
    /// bin1_offset index for this cooler (for slicing bin1 ranges into pixel ranges).
    pub fn bin1_offset(path: &str, res: Option<&str>) -> Result<Vec<i64>> {
        let f = File::open(path)?; let g = open_res(&f, res)?;
        let mut v = g.dataset("indexes/bin1_offset")?.read_1d::<i64>()?.to_vec();
        let nbins = g.dataset("bins/start")?.shape()[0];
        if v.len() == nbins { v.push(g.dataset("pixels/count")?.shape()[0] as i64); }
        Ok(v)
    }
}
impl crate::merge::BlockSource for CoolerPix {
    fn next(&mut self) -> Result<Option<(Vec<i64>, Vec<i64>)>> {
        if self.pos >= self.end { return Ok(None); }
        let hi = std::cmp::min(self.pos + self.block, self.end);
        let b1 = self.b1.read_slice_1d::<i64, _>(self.pos..hi)?;
        let b2 = self.b2.read_slice_1d::<i64, _>(self.pos..hi)?;
        let cn = self.cn.read_slice_1d::<i32, _>(self.pos..hi)?;
        let keys: Vec<i64> = (0..b1.len()).map(|i| b1[i] * self.nbins + b2[i]).collect();
        let cnts: Vec<i64> = cn.iter().map(|&x| x as i64).collect();
        self.pos = hi;
        Ok(Some((keys, cnts)))
    }
}

/// Streaming writer: append (bin1,bin2,count) blocks in ascending-bin1 order; bounded RAM.
/// Writes into a target group (root "/" for a flat .cool, "resolutions/{res}" for an mcool).
pub struct CoolWriter {
    _f: File,      // keep the file handle alive
    g: Group,      // target group
    d1: hdf5::Dataset,
    d2: hdf5::Dataset,
    dc: hdf5::Dataset,
    nbins: usize,
    chrom_offset: Vec<i64>,
    bincount: Vec<i64>,
    pub nnz: usize,
    last_bin1: i64,
}

impl CoolWriter {
    pub fn create(
        path: &str, names: &[String], lengths: &[i64], binsize: i64,
        nbins: usize, chrom_offset: &[i64], comp: Comp, assembly: &str,
    ) -> Result<CoolWriter> {
        let f = File::create(path)?;
        Self::create_in(&f, "/", names, lengths, binsize, nbins, chrom_offset, comp, assembly)
    }

    /// Write a cooler into `group` ("/" for flat .cool, "resolutions/{res}" for an mcool level).
    pub fn create_in(
        file: &File, group: &str, names: &[String], lengths: &[i64], binsize: i64,
        nbins: usize, chrom_offset: &[i64], comp: Comp, assembly: &str,
    ) -> Result<CoolWriter> {
        let g = if group == "/" { file.group("/")? } else { file.create_group(group)? };
        let sattr = |name: &str, val: &str| -> Result<()> {
            g.new_attr::<hdf5::types::VarLenAscii>().create(name)?
                .write_scalar(&hdf5::types::VarLenAscii::from_ascii(val)?)?; Ok(())
        };
        sattr("format", "HDF5::Cooler")?;
        g.new_attr::<i64>().create("format-version")?.write_scalar(&3i64)?;
        sattr("bin-type", "fixed")?;
        g.new_attr::<i64>().create("bin-size")?.write_scalar(&binsize)?;
        sattr("storage-mode", "symmetric-upper")?;
        g.new_attr::<i64>().create("nbins")?.write_scalar(&(nbins as i64))?;
        g.new_attr::<i64>().create("nchroms")?.write_scalar(&(names.len() as i64))?;
        g.new_attr::<i64>().create("nnz")?.write_scalar(&0i64)?;
        sattr("genome-assembly", assembly)?;
        sattr("generated-by", "rooler")?;

        // chroms
        let gc = g.create_group("chroms")?;
        let maxlen = names.iter().map(|n| n.len()).max().unwrap_or(1);
        write_fixed_ascii(&gc, "name", names, maxlen)?;
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
        write_chrom_enum(&gb, &cids, names)?;
        build_i32(&gb, "start", &starts, comp)?;
        build_i32(&gb, "end", &ends, comp)?;

        // pixels (resizable, streaming)
        let gp = g.create_group("pixels")?;
        let chunk = 1usize << 20;
        let d1 = res_i64(&gp, "bin1_id", chunk, comp)?;
        let d2 = res_i64(&gp, "bin2_id", chunk, comp)?;
        let dc = res_i32(&gp, "count", chunk, comp)?;

        Ok(CoolWriter {
            _f: file.clone(), g, d1, d2, dc, nbins, chrom_offset: chrom_offset.to_vec(),
            bincount: vec![0i64; nbins], nnz: 0, last_bin1: -1,
        })
    }

    pub fn append(&mut self, bin1: &[i64], bin2: &[i64], count: &[i32]) -> Result<()> {
        let n = bin1.len();
        if n == 0 { return Ok(()); }
        debug_assert!(bin1[0] >= self.last_bin1, "append blocks must be sorted by bin1");
        self.last_bin1 = bin1[n - 1];
        let (lo, hi) = (self.nnz, self.nnz + n);
        self.d1.resize([hi])?; self.d1.write_slice(&arr1(bin1), s![lo..hi])?;
        self.d2.resize([hi])?; self.d2.write_slice(&arr1(bin2), s![lo..hi])?;
        self.dc.resize([hi])?; self.dc.write_slice(&arr1(count), s![lo..hi])?;
        for &b in bin1 { self.bincount[b as usize] += 1; }
        self.nnz = hi;
        Ok(())
    }

    pub fn close(self) -> Result<()> {
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
        // file flushes/closes when the last handle (driver's) drops; don't close here.
        Ok(())
    }
}

fn write_chrom_enum(g: &Group, cids: &[i32], _names: &[String]) -> Result<()> {
    // v1: plain int32 codes (cooler fetch/region use chrom_offset; enum fidelity is a TODO)
    g.new_dataset::<i32>().shape([cids.len()]).create("chrom")?.write(&arr1(cids))?;
    Ok(())
}

fn write_fixed_ascii(g: &Group, name: &str, vals: &[String], maxlen: usize) -> Result<()> {
    macro_rules! w { ($n:expr) => {{
        let a: Vec<FixedAscii<$n>> = vals.iter().map(|s| FixedAscii::<$n>::from_ascii(s.as_bytes()).unwrap()).collect();
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
