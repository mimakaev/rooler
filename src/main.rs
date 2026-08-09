use anyhow::Result;
use clap::{Parser, Subcommand};
use rooler::{balance, cload, cooler, expected, merge, zoomify};

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
        #[arg(long)] block: Option<i64>,
    },
    /// Compute + store cis expected P(s) per region (arms/chroms)
    Expected {
        uri: String,
        #[arg(long)] view: Option<String>,
    },
    /// internal: write a tiny test cooler
    TestWrite { out: String, #[arg(long, default_value="0")] variant: i32 },
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
            let paths: Vec<String> = inputs.iter().map(|s| s.split("::").next().unwrap().to_string()).collect();
            let r = res.or_else(|| inputs[0].split("::").nth(1).map(|g| g.rsplit('/').next().unwrap().to_string()));
            merge::merge_coolers_parallel(&paths, r.as_deref(), &out, mem, threads, cooler::Comp::parse(&preset)?, assembly.as_deref(), true)?;
        }
        Cmd::Cload { pairs, binsize, out, mem, threads, preset, assembly, chunksize, zero_based } => {
            let mem = resolve_mem(mem, chunksize, threads);
            let tmp = format!("{}.runs", out);
            cload::cload(&pairs, binsize, &out, mem, threads, cooler::Comp::parse(&preset)?, &tmp,
                assembly.as_deref(), !zero_based, true)?;
        }
        Cmd::Zoomify { src, out, resolutions, preset, assembly, balance, threads } => {
            zoomify::zoomify_and_balance(&src, &out, resolutions, cooler::Comp::parse(&preset)?,
                assembly.as_deref(), balance, threads, true)?;
        }
        Cmd::Balance { uri, ignore_diags, mad_max, min_nnz, min_count, tol, max_iters, threads, block } => {
            balance::balance(&uri, balance::Params{ignore_diags, mad_max, min_nnz, min_count, tol, max_iters, nthreads: threads, tiled_block: block}, true)?;
        }
        Cmd::Expected { uri, view } => {
            expected::expected(&uri, view.as_deref(), true)?;
        }
        Cmd::TestWrite { out, variant } => {
            let names = vec!["chrA".to_string(), "chrB".to_string()];
            let lengths = vec![35i64, 20]; let chrom_offset = vec![0i64, 4, 6];
            let mut w = cooler::CoolWriter::create(&out, &names, &lengths, 10, 6, &chrom_offset,
                cooler::Comp::parse("gzip4")?, "test")?;
            if variant == 0 {
                w.append(&[0, 0, 1], &[0, 3, 1], &[5, 2, 7])?; w.append(&[4], &[5], &[3])?;
            } else {
                w.append(&[0, 1, 4], &[3, 1, 5], &[10, 1, 100])?;
            }
            w.close()?; println!("wrote {}", out);
        }
    }
    Ok(())
}
