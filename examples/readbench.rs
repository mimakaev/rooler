//! Pixel-table read throughput, via the same path the streaming ops use.
//!   cargo run --release --example readbench -- <cool> [group] [--serial]
use anyhow::Result;

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let serial = args.iter().any(|a| a == "--serial");
    args.retain(|a| a != "--serial");
    let path = args.first().expect("usage: readbench <cool> [group] [--serial]").clone();
    let grp = args.get(1).cloned().unwrap_or_else(|| "/".into());
    let f = hdf5::File::open(&path)?;
    let g = if grp == "/" { f.group("/")? } else { f.group(&grp)? };
    let n = g.dataset("pixels/count")?.shape()[0];

    let t0 = std::time::Instant::now();
    let mut seen = 0usize;
    if serial {
        // the ordinary HDF5 read path, one chunk inflated at a time
        let (b1, b2, cn) = (g.dataset("pixels/bin1_id")?, g.dataset("pixels/bin2_id")?, g.dataset("pixels/count")?);
        let (mut pos, block) = (0usize, 1 << 22);
        while pos < n {
            let hi = (pos + block).min(n);
            let _ = b1.read_slice_1d::<i64, _>(pos..hi)?;
            let _ = b2.read_slice_1d::<i64, _>(pos..hi)?;
            let _ = cn.read_slice_1d::<i32, _>(pos..hi)?;
            seen += hi - pos;
            pos = hi;
        }
    } else {
        rooler::parread::stream_pixels(&g, 1 << 22, |a, _b, _c| { seen += a.len(); Ok(()) })?;
    }
    let s = t0.elapsed().as_secs_f64();
    let gb = seen as f64 * 20.0 / 1e9;   // i64 + i64 + i32 of decompressed data
    println!("{:>22}  {:>4}  {:>13} pix  {:>6.1}s  {:>5.2} GB/s  ({:>5.0} Mpix/s)  [file {:.2} GB]",
        path.rsplit('/').next().unwrap(), if serial { "ser" } else { "par" }, seen, s, gb / s,
        seen as f64 / s / 1e6, std::fs::metadata(&path)?.len() as f64 / 1e9);
    Ok(())
}
