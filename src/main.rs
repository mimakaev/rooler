use anyhow::Result;
use clap::{Parser, Subcommand};
use rooler::{balance, cload, cooler, expected, merge, pairs_parse, pairs_pq, pairs_tools, repack, zoomify};

#[derive(Parser)]
#[command(name = "rooler", about = "Fast out-of-core cooler engine")]
struct Cli { #[command(subcommand)] cmd: Cmd }

#[derive(Subcommand)]
enum Cmd {
    /// Merge N coolers (sum counts on matching pixels)
    Merge {
        out: String,
        #[arg(required = true)] inputs: Vec<String>,
        #[arg(long)] res: Option<String>,
        /// RAM budget in GB (default 4)
        #[arg(long, conflicts_with = "chunksize")] mem: Option<f64>,
        #[arg(long, default_value = "gzip4")] preset: String,
        #[arg(long)] assembly: Option<String>,
        #[arg(long, alias = "nproc", default_value = "8")] threads: usize,
        /// cooler compatibility: pixels per chunk; maps to a --mem budget (mutually exclusive with --mem)
        #[arg(long)] chunksize: Option<u64>,
    },
    /// Load a .pairs.gz (or plain .pairs, or "-" for stdin) into a .cool at a fixed resolution
    Cload {
        pairs: String,
        binsize: i64,
        out: String,
        /// RAM budget in GB (default 4)
        #[arg(long, conflicts_with = "chunksize")] mem: Option<f64>,
        #[arg(long, alias = "nproc", default_value = "8")] threads: usize,
        #[arg(long, default_value = "gzip4")] preset: String,
        #[arg(long)] assembly: Option<String>,
        /// cooler compatibility: pixels per chunk; maps to a --mem budget (mutually exclusive with --mem)
        #[arg(long)] chunksize: Option<u64>,
        /// treat pairs positions as 0-based (the .pairs spec, and the default here, is 1-based)
        #[arg(long)] zero_based: bool,
    },
    /// Build a multi-resolution .mcool from a base .cool
    Zoomify {
        src: String,
        out: String,
        #[arg(long, value_delimiter=',')] resolutions: Option<Vec<i64>>,
        #[arg(long, default_value = "gzip4")] preset: String,
        #[arg(long)] assembly: Option<String>,
        /// balance every resolution after building (distiller-style)
        #[arg(long)] balance: bool,
        #[arg(long, alias = "nproc", default_value = "8")] threads: usize,
        /// RAM budget in GB for the balancing passes (scratch spills to disk-backed mmap beyond it)
        #[arg(long, default_value = "8.0")] mem: f64,
        /// skip the expected computation that otherwise follows each --balance pass
        #[arg(long)] no_expected: bool,
    },
    /// Balance a cooler (genome-wide IC); writes bins/weight
    Balance {
        uri: String,
        #[arg(long, default_value="2")] ignore_diags: i64,
        #[arg(long, default_value="5")] mad_max: f64,
        #[arg(long, default_value="10")] min_nnz: f64,
        #[arg(long, default_value="0")] min_count: f64,
        #[arg(long, default_value="1e-4")] tol: f64,
        #[arg(long, default_value="200")] max_iters: usize,
        #[arg(long, alias = "nproc", default_value="8")] threads: usize,
        /// SpMV tile size. Default: auto (tiled 65536 above 4M bins, row-chunk below). 0 forces row-chunk.
        #[arg(long)] block: Option<i64>,
        /// RAM budget in GB; the scratch spills to a disk-backed mmap beyond it (same results)
        #[arg(long, default_value="8.0")] mem: f64,
        /// skip the expected computation that otherwise follows balancing
        #[arg(long)] no_expected: bool,
    },
    /// Compute + store cis expected P(s) per region (arms/chroms)
    Expected {
        uri: String,
        #[arg(long)] view: Option<String>,
    },
    /// Rewrite a cooler/mcool with rooler defaults: parallel gzip, assembly stamped + verified,
    /// balanced if it has no weights, expected computed. In place unless --out.
    Repack {
        src: String,
        /// write here instead of replacing the source in place
        #[arg(long, conflicts_with = "backup")] out: Option<String>,
        /// keep the original at <src>.bac
        #[arg(long)] backup: bool,
        #[arg(long)] assembly: Option<String>,
        #[arg(long, default_value = "gzip4")] preset: String,
        #[arg(long, alias = "nproc", default_value = "8")] threads: usize,
        /// RAM budget in GB for a balancing pass, if one is needed
        #[arg(long, default_value = "8.0")] mem: f64,
        /// skip the expected computation
        #[arg(long)] no_expected: bool,
    },
    /// Pairs as Parquet (prototype): sort, scan, view
    Pairs { #[command(subcommand)] cmd: PairsCmd },
}

#[derive(Subcommand)]
enum PairsCmd {
    /// Text .pairs(.gz) (or "-") -> pairs file sorted chrom1-pos1-chrom2-pos2 in the header's
    /// chromosome order. Output is Parquet
    Sort {
        input: String,
        out: String,
        /// RAM budget in GB
        #[arg(long, default_value = "4.0")] mem: f64,
        #[arg(long, alias = "nproc", default_value = "8")] threads: usize,
        /// codec of the output: zstd, zstdN, lz4, none
        #[arg(long, default_value = "zstd1")] codec: String,
        /// codec of the spill runs
        #[arg(long, default_value = "zstd1")] run_codec: String,
    },
    /// Decode a whole pairs file: pair count, order check, bytes per pair by column, speed
    Scan {
        file: String,
        #[arg(long, alias = "nproc", default_value = "8")] threads: usize,
    },
    /// pairs file -> 4DN text on stdout
    View {
        file: String,
        /// omit the header
        #[arg(long)] no_header: bool,
        /// keep only pairs with mapq1 and mapq2 at least this
        #[arg(long, default_value = "0")] min_mapq: u8,
        /// chromsizes file: keep only pairs (and header lines) on these chromosomes
        #[arg(long)] chroms: Option<String>,
    },
    /// Name-grouped SAM text from a mapper (or "-") -> sorted pairs file. Reports every ligation
    /// junction of a read pair, as `pairtools parse --walks-policy all --add-columns mapq`;
    /// only pairs with both sides uniquely mapped are stored, the rest are counted
    Parse {
        input: String,
        out: String,
        /// chromosome order (first column of a chromsizes file); other contigs follow by name
        #[arg(long)] chroms_path: Option<String>,
        #[arg(long, default_value = "")] assembly: String,
        #[arg(long, default_value = "1")] min_mapq: u8,
        #[arg(long, default_value = "20")] max_inter_align_gap: i32,
        #[arg(long, default_value = "750")] max_molecule_size: i64,
        /// RAM budget in GB
        #[arg(long, default_value = "4.0")] mem: f64,
        #[arg(long, alias = "nproc", default_value = "4")] threads: usize,
        #[arg(long, default_value = "zstd1")] codec: String,
    },
    /// Merge sorted pairs files into one; with --dedup N also drop duplicates (same chromosomes
    /// and strands, both positions within N bp of a kept pair) and write pairtools-style stats
    Merge {
        out: String,
        #[arg(required = true)] inputs: Vec<String>,
        /// remove duplicates with this max mismatch in bp (0 = exact)
        #[arg(long)] dedup: Option<u32>,
        /// write the stats here (they are always stored in the output's footer)
        #[arg(long)] stats: Option<String>,
        #[arg(long, default_value = "8.0")] mem: f64,
        #[arg(long, alias = "nproc", default_value = "8")] threads: usize,
        #[arg(long, default_value = "zstd1")] codec: String,
    },
}

/// cooler-compat shim: `--chunksize C` with `--nproc N` implies roughly C*N*40 bytes of
/// working set (see MEMORY_CALIBRATION.md). --mem and --chunksize are mutually exclusive
/// (clap `conflicts_with`), so there is no precedence question; neither given -> 4 GB.
fn resolve_mem(mem: Option<f64>, chunksize: Option<u64>, threads: usize) -> f64 {
    match (mem, chunksize) {
        (Some(m), _) => m,
        (None, Some(c)) => {
            let m = (c as f64 * threads as f64 * 40e-9).max(0.25);
            eprintln!("  --chunksize {} x --nproc {} -> --mem {:.2} GB", c, threads, m);
            m
        }
        (None, None) => 4.0,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Merge { out, inputs, res, mem, preset, assembly, threads, chunksize } => {
            let mem = resolve_mem(mem, chunksize, threads);
            // one group for every input: --res R, else the `::group` of the first input, else "/"
            let grp = res.map(|r| format!("resolutions/{}", r))
                .unwrap_or_else(|| cooler::split_uri(&inputs[0]).1);
            let paths: Vec<String> = inputs.iter().map(|s| cooler::split_uri(s).0).collect();
            merge::merge_coolers(&paths, &grp, &out, mem, threads, cooler::Comp::parse(&preset)?, assembly.as_deref(), true)?;
        }
        Cmd::Cload { pairs, binsize, out, mem, threads, preset, assembly, chunksize, zero_based } => {
            let mem = resolve_mem(mem, chunksize, threads);
            let tmp = format!("{}.runs", out);
            cload::cload(&pairs, binsize, &out, mem, threads, cooler::Comp::parse(&preset)?, &tmp,
                assembly.as_deref(), !zero_based, true)?;
        }
        Cmd::Zoomify { src, out, resolutions, preset, assembly, balance, threads, mem, no_expected } => {
            zoomify::zoomify_and_balance(&src, &out, resolutions, cooler::Comp::parse(&preset)?,
                assembly.as_deref(), balance, threads, mem, !no_expected, true)?;
        }
        Cmd::Balance { uri, ignore_diags, mad_max, min_nnz, min_count, tol, max_iters, threads, block, mem, no_expected } => {
            balance::balance(&uri, balance::Params{ignore_diags, mad_max, min_nnz, min_count, tol,
                max_iters, nthreads: threads, tiled_block: block, mem_gb: mem}, true)?;
            // expected is on by default once weights exist (per-organism default view);
            // an unknown genome warns instead of failing the balance that just succeeded
            if !no_expected { expected::expected_or_warn(&uri, true); }
        }
        Cmd::Expected { uri, view } => {
            expected::expected(&uri, view.as_deref(), true)?;
        }
        Cmd::Repack { src, out, backup, assembly, preset, threads, mem, no_expected } => {
            repack::repack(&src, repack::RepackOpts {
                out, backup, assembly, comp: cooler::Comp::parse(&preset)?,
                nthreads: threads, mem_gb: mem, expected: !no_expected }, true)?;
        }
        Cmd::Pairs { cmd } => match cmd {
            PairsCmd::Sort { input, out, mem, threads, codec, run_codec } => {
                if threads == 0 { anyhow::bail!("--threads must be positive"); }
                pairs_tools::sort(&input, &out, &pairs_tools::SortOpts {
                    mem_gb: mem, threads, tmpdir: format!("{}.runs", out),
                    run_codec: pairs_pq::PqCodec::parse(&run_codec)?,
                    out_codec: pairs_pq::PqCodec::parse(&codec)?,
                }, true)?;
            }
            PairsCmd::Scan { file, threads } => {
                let r = pairs_tools::scan(&file, threads.max(1))?;
                let sz = std::fs::metadata(&file)?.len();
                let np = r.npairs.max(1) as f64;
                println!("pairs\t{}\nblocks\t{}\nsorted\t{}\ndigest\t{:016x}", r.npairs, r.nblocks, r.sorted, r.digest);
                println!("file_bytes\t{}\nbytes_per_pair\t{:.4}", sz, sz as f64 / np);
                for (name, bytes) in &r.col_bytes { println!("  {}\t{:.4}", name, *bytes as f64 / np); }
                println!("seconds\t{:.3}\nMpairs_per_s\t{:.1}", r.seconds, r.npairs as f64 / r.seconds / 1e6);
            }
            PairsCmd::View { file, no_header, min_mapq, chroms } => {
                let so = std::io::stdout();
                let mut w = std::io::BufWriter::with_capacity(1 << 20, so.lock());
                pairs_tools::view(&file, &mut w, !no_header, min_mapq, chroms.as_deref())?;
            }
            PairsCmd::Parse { input, out, chroms_path, assembly, min_mapq, max_inter_align_gap, max_molecule_size, mem, threads, codec } => {
                if threads == 0 { anyhow::bail!("--threads must be positive"); }
                let c = pairs_pq::PqCodec::parse(&codec)?;
                pairs_parse::parse(&input, &out, pairs_parse::ParseOpts {
                    min_mapq, max_gap: max_inter_align_gap, max_molecule: max_molecule_size, chroms_path, assembly,
                }, &pairs_tools::SortOpts { mem_gb: mem, threads, tmpdir: format!("{}.runs", out), run_codec: c, out_codec: c }, true)?;
            }
            PairsCmd::Merge { out, inputs, dedup, stats, mem, threads, codec } => {
                if threads == 0 { anyhow::bail!("--threads must be positive"); }
                let c = pairs_pq::PqCodec::parse(&codec)?;
                let st = pairs_tools::merge(&inputs, &out, &pairs_tools::SortOpts {
                    mem_gb: mem, threads, tmpdir: String::new(), run_codec: c, out_codec: c }, dedup, true)?;
                if let Some(p) = stats {
                    let f = pairs_pq::PqFile::open(&out)?;
                    std::fs::write(&p, st.to_text(&f.header.chroms))?;
                }
            }
        },
    }
    Ok(())
}
