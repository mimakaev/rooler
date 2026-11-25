//! Pairs as Parquet: schema, row-group encoder/decoder, file writer/reader.
//!
//! Columns: `chrom1 pos1 chrom2 pos2 strand1 strand2` (+ `pair_type`, `mapq1 mapq2` when present).
//! Chromosomes and strands are dictionary-encoded strings, so the file reads naturally from
//! polars/duckdb/pyarrow (`chrom1 == "chr3"`). Parquet has no enum with a defined order, so the
//! chromosome order lives in the footer key-value metadata (`pairs.chromsizes`); the row order is
//! chrom1-pos1-chrom2-pos2 in THAT chromosome order, declared in `pairs.sorted`.
//!
//! Spill runs use an internal variant of the same layout (`PqCtx::new(.., internal = true)`):
//! chromosomes as int32 ids and strands as booleans. The Parquet writer hashes every row of a
//! string-dictionary column, which makes those four columns ~60% of the encode time; runs are
//! never queried by name, so they skip it. `PqFile` reads either variant.
//!
//! Row groups are encoded independently (`PqCtx::encode_row_group`) so callers can encode them
//! on many threads and append the results in order.
use crate::pairs::{Header, Pair, HAS_MAPQ, HAS_PTYPE};
use anyhow::{anyhow, bail, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, UInt8Type};
use arrow_array::{Array, ArrayRef, BooleanArray, DictionaryArray, Int32Array, RecordBatch, StringArray, UInt8Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::arrow::arrow_writer::{compute_leaves, get_column_writers, ArrowColumnChunk};
use parquet::arrow::{add_encoded_arrow_schema_to_metadata, ArrowSchemaConverter};
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::{ColumnPath, SchemaDescriptor};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;

pub const KEY_VERSION: &str = "pairs.format_version";
pub const KEY_SORTED: &str = "pairs.sorted";
pub const KEY_CHROMSIZES: &str = "pairs.chromsizes";
pub const KEY_ASSEMBLY: &str = "pairs.genome_assembly";
/// First (chrom1 id << 32 | pos1) key of every row group, comma-separated hex. Lets a sorted file
/// be merged with others by key range without decoding anything first.
pub const KEY_RG_FIRST: &str = "pairs.row_group_first_keys";
/// Counters accumulated by the tools that produced the file (`name\tvalue` lines).
pub const KEY_STATS: &str = "pairs.stats";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PqCodec { Zstd(i32), Lz4, None }

impl PqCodec {
    /// "zstd" (level 1), "zstdN", "lz4", "none"
    pub fn parse(s: &str) -> Result<PqCodec> {
        match s {
            "lz4" => Ok(PqCodec::Lz4),
            "none" => Ok(PqCodec::None),
            "zstd" => Ok(PqCodec::Zstd(1)),
            _ => s.strip_prefix("zstd").and_then(|l| l.parse().ok()).map(PqCodec::Zstd)
                .ok_or_else(|| anyhow!("unknown parquet codec {:?} (zstd, zstdN, lz4, none)", s)),
        }
    }
}

/// Everything needed to encode row groups of one file; shareable across threads.
pub struct PqCtx {
    schema: Arc<Schema>,
    pq_schema: SchemaDescriptor,
    props: Arc<WriterProperties>,
    names: ArrayRef,
    plus_minus: ArrayRef,
    columns: u8,
    internal: bool,
}

impl PqCtx {
    /// `internal`: the spill-run variant (integer chromosome ids, boolean strands).
    pub fn new(h: &Header, codec: PqCodec, internal: bool) -> Result<PqCtx> {
        // positions are stored as signed 32-bit, the widest type every Parquet reader agrees on
        if let Some((n, l)) = h.chroms.iter().find(|c| c.1 > i32::MAX as u64) {
            bail!("chromosome {} is {} bp: longer than 2^31, not representable", n, l);
        }
        let dict = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let (chrom, strand) = if internal { (DataType::Int32, DataType::Boolean) } else { (dict.clone(), dict.clone()) };
        let mut fields = vec![
            Field::new("chrom1", chrom.clone(), false), Field::new("pos1", DataType::Int32, false),
            Field::new("chrom2", chrom, false), Field::new("pos2", DataType::Int32, false),
            Field::new("strand1", strand.clone(), false), Field::new("strand2", strand, false),
        ];
        if h.columns & HAS_PTYPE != 0 { fields.push(Field::new("pair_type", dict.clone(), false)); }
        if h.columns & HAS_MAPQ != 0 {
            fields.push(Field::new("mapq1", DataType::UInt8, false));
            fields.push(Field::new("mapq2", DataType::UInt8, false));
        }
        let schema = Arc::new(Schema::new(fields));
        let col = |s: &str| ColumnPath::from(s);
        let chromsizes: String = h.chroms.iter().map(|(n, s)| format!("{}\t{}\n", n, s)).collect();
        let mut kv = vec![
            KeyValue::new(KEY_VERSION.to_string(), "0".to_string()),
            KeyValue::new(KEY_SORTED.to_string(), h.sorted.clone()),
            KeyValue::new(KEY_CHROMSIZES.to_string(), chromsizes),
        ];
        if !h.assembly.is_empty() { kv.push(KeyValue::new(KEY_ASSEMBLY.to_string(), h.assembly.clone())); }
        let mut b = WriterProperties::builder()
            .set_compression(match codec {
                PqCodec::Zstd(l) => Compression::ZSTD(ZstdLevel::try_new(l)?),
                PqCodec::Lz4 => Compression::LZ4_RAW,
                PqCodec::None => Compression::UNCOMPRESSED,
            })
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::Chunk)
            .set_max_row_group_size(usize::MAX) // row groups are cut by the caller
            .set_column_encoding(col("pos1"), Encoding::DELTA_BINARY_PACKED)
            .set_column_encoding(col("pos2"), Encoding::DELTA_BINARY_PACKED)
            .set_key_value_metadata(Some(kv));
        if internal {
            // chrom1 is constant over long stretches (delta packs it to nothing); chrom2 jumps
            // between chromosomes on every trans pair, which byte-splitting + the codec handles
            b = b.set_column_encoding(col("chrom1"), Encoding::DELTA_BINARY_PACKED)
                .set_column_encoding(col("chrom2"), Encoding::BYTE_STREAM_SPLIT)
                .set_column_dictionary_enabled(col("pair_type"), true);
        } else {
            for c in ["chrom1", "chrom2", "strand1", "strand2", "pair_type"] {
                b = b.set_column_dictionary_enabled(col(c), true);
            }
        }
        let mut props = b.build();
        add_encoded_arrow_schema_to_metadata(&schema, &mut props);
        let props = Arc::new(props);
        let pq_schema = ArrowSchemaConverter::new().with_coerce_types(props.coerce_types()).convert(&schema)?;
        Ok(PqCtx {
            schema, pq_schema, props,
            names: Arc::new(StringArray::from_iter_values(h.chroms.iter().map(|c| c.0.as_str()))),
            plus_minus: Arc::new(StringArray::from(vec!["+", "-"])),
            columns: h.columns,
            internal,
        })
    }

    /// Encode one row group. Pure with respect to the file: the result is appended later.
    pub fn encode_row_group(&self, pairs: &[Pair]) -> Result<Vec<ArrowColumnChunk>> {
        let dict = |keys: Int32Array, vals: &ArrayRef| -> Result<ArrayRef> {
            Ok(Arc::new(DictionaryArray::<Int32Type>::try_new(keys, vals.clone())?))
        };
        let ints = |f: fn(&Pair) -> i32| Int32Array::from_iter_values(pairs.iter().map(f));
        let (c1, c2) = (ints(|p| p.chrom1() as i32), ints(|p| p.chrom2() as i32));
        let (p1, p2): (ArrayRef, ArrayRef) = (Arc::new(ints(|p| p.pos1() as i32)), Arc::new(ints(|p| p.pos2() as i32)));
        let mut arrays: Vec<ArrayRef> = if self.internal {
            let bools = |bit: u8| -> ArrayRef { Arc::new(BooleanArray::from_iter(pairs.iter().map(|p| Some(p.strands() & bit != 0)))) };
            vec![Arc::new(c1), p1, Arc::new(c2), p2, bools(1), bools(2)]
        } else {
            vec![dict(c1, &self.names)?, p1, dict(c2, &self.names)?, p2,
                 dict(ints(|p| (p.strands() & 1) as i32), &self.plus_minus)?,
                 dict(ints(|p| (p.strands() >> 1) as i32), &self.plus_minus)?]
        };
        if self.columns & HAS_PTYPE != 0 {
            // a handful of distinct two-character types per row group: build its dictionary here
            let mut code: HashMap<[u8; 2], i32> = HashMap::new();
            let mut vals: Vec<String> = Vec::new();
            let keys: Vec<i32> = pairs.iter().map(|p| {
                let t = p.ptype();
                *code.entry(t).or_insert_with(|| { vals.push(String::from_utf8_lossy(&t).into_owned()); vals.len() as i32 - 1 })
            }).collect();
            let vals: ArrayRef = Arc::new(StringArray::from(vals));
            arrays.push(dict(Int32Array::from(keys), &vals)?);
        }
        if self.columns & HAS_MAPQ != 0 {
            arrays.push(Arc::new(UInt8Array::from_iter_values(pairs.iter().map(|p| p.mapq1()))));
            arrays.push(Arc::new(UInt8Array::from_iter_values(pairs.iter().map(|p| p.mapq2()))));
        }
        let mut writers = get_column_writers(&self.pq_schema, &self.props, &self.schema)?.into_iter();
        let mut out = Vec::with_capacity(arrays.len());
        for (arr, field) in arrays.iter().zip(self.schema.fields()) {
            for leaf in compute_leaves(field, arr)? {
                let mut cw = writers.next().ok_or_else(|| anyhow!("parquet column writer count mismatch"))?;
                cw.write(&leaf)?;
                out.push(cw.close()?);
            }
        }
        Ok(out)
    }
}

pub struct PqWriter {
    w: SerializedFileWriter<BufWriter<File>>,
    first: Vec<u64>,
}

impl PqWriter {
    pub fn create(path: &str, ctx: &PqCtx) -> Result<PqWriter> {
        let f = BufWriter::with_capacity(1 << 22, File::create(path)?);
        Ok(PqWriter { w: SerializedFileWriter::new(f, ctx.pq_schema.root_schema_ptr(), ctx.props.clone())?, first: Vec::new() })
    }
    /// Append a row group produced by `PqCtx::encode_row_group` (same ctx as `create`);
    /// `first_k1` is the `k1` of its first pair.
    pub fn append(&mut self, chunks: Vec<ArrowColumnChunk>, first_k1: u64) -> Result<()> {
        self.first.push(first_k1);
        let mut rg = self.w.next_row_group()?;
        for c in chunks { c.append_to_row_group(&mut rg)?; }
        rg.close()?;
        Ok(())
    }
    /// Write the footer. A Parquet file without it is unreadable.
    pub fn close(mut self, stats: Option<&str>) -> Result<()> {
        let keys: Vec<String> = self.first.iter().map(|k| format!("{:x}", k)).collect();
        self.w.append_key_value_metadata(KeyValue::new(KEY_RG_FIRST.to_string(), keys.join(",")));
        if let Some(s) = stats { self.w.append_key_value_metadata(KeyValue::new(KEY_STATS.to_string(), s.to_string())); }
        self.w.into_inner()?.flush()?;
        Ok(())
    }
}

/// An open pairs Parquet file. `read_row_group` takes `&self` and opens its own handle, so row
/// groups can be read from many threads at once.
pub struct PqFile {
    path: String,
    meta: ArrowReaderMetadata,
    pub header: Header,
    /// first key of every row group, when the file records them
    pub first_keys: Option<Vec<u64>>,
    /// the `pairs.stats` footer block ("" when absent)
    pub stats: String,
    ids: HashMap<String, u32>,
}

impl PqFile {
    pub fn open(path: &str) -> Result<PqFile> {
        let meta = ArrowReaderMetadata::load(&File::open(path)?, ArrowReaderOptions::new())?;
        let mut h = Header { sorted: "none".into(), ..Default::default() };
        let (mut first_keys, mut stats) = (None, String::new());
        for kv in meta.metadata().file_metadata().key_value_metadata().map(|v| v.as_slice()).unwrap_or(&[]) {
            let v = kv.value.as_deref().unwrap_or("");
            match kv.key.as_str() {
                KEY_SORTED => h.sorted = v.to_string(),
                KEY_ASSEMBLY => h.assembly = v.to_string(),
                KEY_STATS => stats = v.to_string(),
                KEY_RG_FIRST => first_keys = Some(v.split(',').filter(|x| !x.is_empty())
                    .map(|x| u64::from_str_radix(x, 16).map_err(|_| anyhow!("{}: bad {}", path, KEY_RG_FIRST)))
                    .collect::<Result<Vec<u64>>>()?),
                KEY_CHROMSIZES => for l in v.lines() {
                    match l.split_once('\t').and_then(|(n, s)| Some((n, s.parse::<u64>().ok()?))) {
                        Some((n, s)) => h.chroms.push((n.to_string(), s)),
                        None => bail!("{}: bad {} line {:?}", path, KEY_CHROMSIZES, l),
                    }
                },
                _ => {}
            }
        }
        if h.chroms.is_empty() {
            bail!("{}: no {} in the Parquet metadata — not a pairs Parquet file written by this tool", path, KEY_CHROMSIZES);
        }
        let sch = meta.schema();
        for c in ["chrom1", "pos1", "chrom2", "pos2", "strand1", "strand2"] {
            if sch.field_with_name(c).is_err() { bail!("{}: required column {} is missing", path, c); }
        }
        if sch.field_with_name("pair_type").is_ok() { h.columns |= HAS_PTYPE; }
        if sch.field_with_name("mapq1").is_ok() && sch.field_with_name("mapq2").is_ok() { h.columns |= HAS_MAPQ; }
        let ids = h.chroms.iter().enumerate().map(|(i, c)| (c.0.clone(), i as u32)).collect();
        if first_keys.as_ref().is_some_and(|f| f.len() != meta.metadata().num_row_groups()) {
            bail!("{}: {} does not match the number of row groups", path, KEY_RG_FIRST);
        }
        Ok(PqFile { path: path.to_string(), meta, header: h, first_keys, stats, ids })
    }
    pub fn num_row_groups(&self) -> usize { self.meta.metadata().num_row_groups() }
    pub fn rows(&self, g: usize) -> usize { self.meta.metadata().row_group(g).num_rows() as usize }
    pub fn num_rows(&self) -> u64 { self.meta.metadata().file_metadata().num_rows() as u64 }
    /// Compressed bytes per column over the whole file, in schema order.
    pub fn column_bytes(&self) -> Vec<(String, u64)> {
        let md = self.meta.metadata();
        let mut out: Vec<(String, u64)> = self.meta.schema().fields().iter().map(|f| (f.name().clone(), 0)).collect();
        for rg in md.row_groups() {
            for (j, c) in rg.columns().iter().enumerate() { if j < out.len() { out[j].1 += c.compressed_size() as u64; } }
        }
        out
    }

    /// Decode row group `g`, appending its pairs to `out`.
    pub fn read_row_group(&self, g: usize, out: &mut Vec<Pair>) -> Result<()> {
        let rd = ParquetRecordBatchReaderBuilder::new_with_metadata(File::open(&self.path)?, self.meta.clone())
            .with_row_groups(vec![g]).with_batch_size(1 << 18).build()?;
        out.reserve(self.rows(g));
        for b in rd { self.batch_to_pairs(&b?, out)?; }
        Ok(())
    }

    fn batch_to_pairs(&self, b: &RecordBatch, out: &mut Vec<Pair>) -> Result<()> {
        let col = |name: &str| b.column_by_name(name).ok_or_else(|| anyhow!("{}: column {} missing", self.path, name));
        let dict = |name: &str| -> Result<&DictionaryArray<Int32Type>> {
            col(name)?.as_dictionary_opt::<Int32Type>().filter(|d| d.values().data_type() == &DataType::Utf8)
                .ok_or_else(|| anyhow!("{}: column {} is not a dictionary-encoded string column", self.path, name))
        };
        let i32s = |name: &str| -> Result<&[i32]> {
            Ok(col(name)?.as_primitive_opt::<Int32Type>().ok_or_else(|| anyhow!("{}: column {} is not int32", self.path, name))?.values())
        };
        // (lookup table, keys): pair value = table[key]. Integer/boolean columns of the internal
        // variant go through an identity table so one loop below serves both variants.
        let nchrom = self.header.chroms.len();
        let chrom = |name: &str| -> Result<(Vec<u32>, std::borrow::Cow<[i32]>)> {
            if let Some(a) = col(name)?.as_primitive_opt::<Int32Type>() {
                if a.values().iter().any(|&v| v < 0 || v as usize >= nchrom) { bail!("{}: chromosome id out of range in {}", self.path, name); }
                return Ok(((0..nchrom as u32).collect(), std::borrow::Cow::Borrowed(a.values())));
            }
            let d = dict(name)?;
            let vals = d.values().as_string::<i32>();
            let map = (0..vals.len()).map(|i| self.ids.get(vals.value(i)).copied().ok_or_else(|| anyhow!(
                "{}: chromosome {:?} is not in {}", self.path, vals.value(i), KEY_CHROMSIZES))).collect::<Result<Vec<u32>>>()?;
            Ok((map, std::borrow::Cow::Borrowed(d.keys().values())))
        };
        let strand = |name: &str| -> Result<(Vec<u8>, std::borrow::Cow<[i32]>)> {
            if let Some(a) = col(name)?.as_boolean_opt() {
                return Ok((vec![0, 1], std::borrow::Cow::Owned(a.values().iter().map(|b| b as i32).collect())));
            }
            let d = dict(name)?;
            let vals = d.values().as_string::<i32>();
            let map = (0..vals.len()).map(|i| match vals.value(i) {
                "+" => Ok(0u8), "-" => Ok(1u8), s => Err(anyhow!("{}: bad strand {:?}", self.path, s)),
            }).collect::<Result<Vec<u8>>>()?;
            Ok((map, std::borrow::Cow::Borrowed(d.keys().values())))
        };
        let (m1, k1) = chrom("chrom1")?;
        let (m2, k2) = chrom("chrom2")?;
        let (p1, p2) = (i32s("pos1")?, i32s("pos2")?);
        let (sm1, sk1) = strand("strand1")?;
        let (sm2, sk2) = strand("strand2")?;
        let pt: Option<(Vec<[u8; 2]>, &[i32])> = if self.header.columns & HAS_PTYPE != 0 {
            let d = dict("pair_type")?;
            let vals = d.values().as_string::<i32>();
            let map = (0..vals.len()).map(|i| match vals.value(i).as_bytes() {
                [a, b] => Ok([*a, *b]), _ => Err(anyhow!("{}: pair_type {:?} is not two characters", self.path, vals.value(i))),
            }).collect::<Result<Vec<[u8; 2]>>>()?;
            Some((map, d.keys().values()))
        } else { None };
        let mq: Option<(&[u8], &[u8])> = if self.header.columns & HAS_MAPQ != 0 {
            let u8s = |name: &str| -> Result<&[u8]> {
                Ok(col(name)?.as_primitive_opt::<UInt8Type>().ok_or_else(|| anyhow!("{}: column {} is not uint8", self.path, name))?.values())
            };
            Some((u8s("mapq1")?, u8s("mapq2")?))
        } else { None };
        for i in 0..b.num_rows() {
            let ptype = match &pt { Some((m, k)) => m[k[i] as usize], None => [0, 0] };
            let (q1, q2) = match mq { Some((a, c)) => (a[i], c[i]), None => (0, 0) };
            out.push(Pair::raw(m1[k1[i] as usize], p1[i] as u32, m2[k2[i] as usize], p2[i] as u32,
                               sm1[sk1[i] as usize] | sm2[sk2[i] as usize] << 1, ptype, q1, q2));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(columns: u8) -> Header {
        Header {
            chroms: vec![("chr1".into(), 1_000_000), ("chr10".into(), 2_000_000_000), ("chr2".into(), 500)],
            sorted: crate::pairs::SORTED_CANONICAL.into(), assembly: "test".into(), columns,
        }
    }

    fn pairs(n: usize, columns: u8) -> Vec<Pair> {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut v: Vec<Pair> = (0..n).map(|_| {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let x = s;
            let pt = if columns & HAS_PTYPE != 0 { [b"UMR"[(x >> 40) as usize % 3], b"UMR"[(x >> 44) as usize % 3]] } else { [0, 0] };
            let (q1, q2) = if columns & HAS_MAPQ != 0 { ((x >> 48) as u8, (x >> 56) as u8) } else { (0, 0) };
            Pair::flipped((x % 3) as u32, ((x >> 8) % 400) as u32, ((x >> 4) % 3) as u32, ((x >> 20) % 400) as u32,
                          x >> 62 & 1 != 0, x >> 63 != 0, pt, q1, q2)
        }).collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn parquet_round_trip_with_and_without_optional_columns() {
        let dir = std::env::temp_dir().join(format!("rooler_pq_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (k, columns) in [0u8, HAS_PTYPE, HAS_MAPQ, HAS_PTYPE | HAS_MAPQ].into_iter().enumerate() {
            let path = dir.join(format!("t{}.parquet", k)).to_string_lossy().into_owned();
            let (h, v) = (header(columns), pairs(5000, columns));
          for internal in [false, true] {
            let ctx = PqCtx::new(&h, PqCodec::Zstd(1), internal).unwrap();
            let mut w = PqWriter::create(&path, &ctx).unwrap();
            for ch in v.chunks(1500) { w.append(ctx.encode_row_group(ch).unwrap(), ch[0].k1).unwrap(); }
            w.close(None).unwrap();
            let f = PqFile::open(&path).unwrap();
            // chromosome order comes back from the metadata, not from any lexicographic order
            assert_eq!((&f.header.chroms, f.header.columns, f.header.sorted.as_str()), (&h.chroms, columns, "chr1-pos1-chr2-pos2"));
            assert_eq!((f.num_row_groups(), f.num_rows()), (4, 5000));
            let mut got = Vec::new();
            for g in 0..f.num_row_groups() { f.read_row_group(g, &mut got).unwrap(); }
            assert_eq!(got, v);
          }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn oversized_chromosome_is_refused() {
        let mut h = header(0);
        h.chroms.push(("big".into(), 3_000_000_000));
        assert!(PqCtx::new(&h, PqCodec::None, false).is_err());
    }
}
