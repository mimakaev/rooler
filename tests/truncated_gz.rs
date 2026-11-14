//! A truncated/corrupt `.pairs.gz` makes the decompressor die mid-stream, which on the pipe looks
//! exactly like a normal EOF. cload must notice (exit status) and refuse, not write a
//! valid-looking cooler from the part it got.
//!
//! This lives in its own test binary on purpose: cload spawns `bgzip`, and a child process
//! inherits every file descriptor libhdf5 has open at that moment — including the file locks
//! on them. Run alongside the other pipeline tests (threads of one process), that made their
//! HDF5 files intermittently un-reopenable until the child exited.
use anyhow::Result;
use rooler::cload;
use rooler::cooler::Comp;
use std::io::Write;

#[test]
fn cload_refuses_truncated_gz() -> Result<()> {
    if std::process::Command::new("bgzip").arg("--version").output().is_err() {
        eprintln!("bgzip not found; skipping");
        return Ok(());
    }
    let dir = std::env::temp_dir().join(format!("rooler_truncgz_{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir)?;
    let p = |n: &str| dir.join(n).to_str().unwrap().to_string();

    // ~50k pairs over one 1 Mb chromosome, pseudo-random so the gzip stream is not tiny
    let mut raw = Vec::new();
    writeln!(raw, "## pairs format v1.0\n#chromsize: chr1 1000000")?;
    let mut s = 12345u64;
    let mut next = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) % 1_000_000 + 1 };
    let npairs = 50_000;
    for i in 0..npairs { writeln!(raw, "r{}\tchr1\t{}\tchr1\t{}\t+\t-", i, next(), next())?; }
    let mut gz = vec![0u8; raw.len() + 1024];
    let n = libdeflater::Compressor::new(libdeflater::CompressionLvl::default())
        .gzip_compress(&raw, &mut gz).unwrap();

    // the intact file loads, and every pair is counted
    std::fs::write(p("ok.pairs.gz"), &gz[..n])?;
    cload::cload(&p("ok.pairs.gz"), 1000, &p("ok.cool"), 0.01, 2,
        Comp::parse("none")?, &p("runs_ok"), Some("test"), true, false)?;
    assert_eq!(hdf5::File::open(p("ok.cool"))?.attr("sum")?.read_scalar::<i64>()?, npairs);

    // cut in half, it must fail and leave nothing behind
    std::fs::write(p("cut.pairs.gz"), &gz[..n / 2])?;
    let e = cload::cload(&p("cut.pairs.gz"), 1000, &p("cut.cool"), 0.01, 2,
        Comp::parse("none")?, &p("runs_cut"), Some("test"), true, false).unwrap_err();
    assert!(e.to_string().contains("truncated or corrupt"), "unexpected error: {}", e);
    assert!(!std::path::Path::new(&p("cut.cool")).exists(), "partial cooler left behind");
    assert!(!std::path::Path::new(&p("runs_cut")).exists(), "spill dir left behind");
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}
