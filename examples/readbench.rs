//! Measure raw pixel-table read throughput (decompression bound), per preset.
//!   cargo run --release --example readbench -- <cool> [group]
use anyhow::Result;

fn main() -> Result<()> {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: readbench <cool> [group]");
    let grp = a.next().unwrap_or_else(|| "/".into());
    let f = hdf5::File::open(&path)?;
    let g = if grp == "/" { f.group("/")? } else { f.group(&grp)? };
    let (b1, b2, cn) = (g.dataset("pixels/bin1_id")?, g.dataset("pixels/bin2_id")?, g.dataset("pixels/count")?);
    let n = cn.shape()[0];
    let t0 = std::time::Instant::now();
    let mut acc = 0i64;
    let mut pos = 0usize;
    let block = 1 << 22;
    while pos < n {
        let hi = (pos + block).min(n);
        acc += b1.read_slice_1d::<i64, _>(pos..hi)?.iter().take(1).sum::<i64>();
        acc += b2.read_slice_1d::<i64, _>(pos..hi)?.iter().take(1).sum::<i64>();
        acc += cn.read_slice_1d::<i32, _>(pos..hi)?.iter().take(1).map(|&x| x as i64).sum::<i64>();
        pos = hi;
    }
    let s = t0.elapsed().as_secs_f64();
    let bytes = n as f64 * 20.0;            // i64 + i64 + i32 decompressed
    let file = std::fs::metadata(&path)?.len() as f64;
    println!("{:>28}  {:>13} pix  {:>6.1}s  decompressed {:>6.2} GB/s  ({:.1} Mpix/s)  [file {:.1} GB, chk {}]",
        path.rsplit('/').next().unwrap(), n, s, bytes / s / 1e9, n as f64 / s / 1e6, file / 1e9, acc.min(1));
    Ok(())
}
