mod cooler;
mod merge;
mod cload;
mod zoomify;
mod balance;
mod scratch;
mod expected;
#[allow(dead_code)]
mod view;
use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rooler", about = "Fast out-of-core cooler engine")]
struct Cli { #[command(subcommand)] cmd: Cmd }

#[derive(Subcommand)]
enum Cmd {
    /// Merge N coolers (sum counts on matching pixels)
    Merge {
        out: String,
        inputs: Vec<String>,
        #[arg(long)] res: Option<String>,
        #[arg(long, default_value = "4.0")] mem: f64,
        #[arg(long, default_value = "blosc:zstd:1")] preset: String,
        #[arg(long)] assembly: Option<String>,
    },
    /// Load a .pairs.gz into a .cool at a fixed resolution
    Cload {
        pairs: String,
        binsize: i64,
        out: String,
        #[arg(long, default_value = "4.0")] mem: f64,
        #[arg(long, default_value = "8")] threads: usize,
        #[arg(long, default_value = "blosc:zstd:1")] preset: String,
        #[arg(long)] assembly: Option<String>,
    },
    /// Build a multi-resolution .mcool from a base .cool
    Zoomify {
        src: String,
        out: String,
        #[arg(long, value_delimiter=',')] resolutions: Option<Vec<i64>>,
        #[arg(long, default_value = "blosc:zstd:1")] preset: String,
        #[arg(long)] assembly: Option<String>,
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
        #[arg(long, default_value="8")] threads: usize,
    },
    /// Compute + store cis expected P(s) per region (arms/chroms)
    Expected {
        uri: String,
        #[arg(long)] view: Option<String>,
    },
    /// internal: write a tiny test cooler
    TestWrite { out: String, #[arg(long, default_value="0")] variant: i32 },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Merge { out, inputs, res, mem, preset, assembly } => {
            let paths: Vec<String> = inputs.iter().map(|s| s.split("::").next().unwrap().to_string()).collect();
            let r = res.or_else(|| inputs[0].split("::").nth(1).map(|g| g.rsplit('/').next().unwrap().to_string()));
            merge::merge_coolers(&paths, r.as_deref(), &out, mem, cooler::Comp::parse(&preset), assembly.as_deref(), true)?;
        }
        Cmd::Cload { pairs, binsize, out, mem, threads, preset, assembly } => {
            let tmp = format!("{}.runs", out);
            cload::cload(&pairs, binsize, &out, mem, threads, cooler::Comp::parse(&preset), &tmp, assembly.as_deref(), true)?;
        }
        Cmd::Zoomify { src, out, resolutions, preset, assembly } => {
            zoomify::zoomify(&src, &out, resolutions, cooler::Comp::parse(&preset), assembly.as_deref(), true)?;
        }
        Cmd::Balance { uri, ignore_diags, mad_max, min_nnz, min_count, tol, max_iters, threads } => {
            balance::balance(&uri, balance::Params{ignore_diags, mad_max, min_nnz, min_count, tol, max_iters, nthreads: threads}, true)?;
        }
        Cmd::Expected { uri, view } => {
            expected::expected(&uri, view.as_deref(), true)?;
        }
        Cmd::TestWrite { out, variant } => {
            let names = vec!["chrA".to_string(), "chrB".to_string()];
            let lengths = vec![35i64, 20]; let chrom_offset = vec![0i64, 4, 6];
            let mut w = cooler::CoolWriter::create(&out, &names, &lengths, 10, 6, &chrom_offset,
                cooler::Comp::parse("gzip4"), "test")?;
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
