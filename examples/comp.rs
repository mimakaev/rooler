fn main() -> anyhow::Result<()> {
    let n = 20_000_000usize;
    let data: Vec<i64> = (0..n as i64).map(|x| x/3).collect();
    for (tag, path) in [("lz4","/tmp/c_lz4.h5"),("zstd","/tmp/c_zstd.h5"),("blosclz","/tmp/c_bl.h5"),("gzip","/tmp/c_gz.h5")] {
        let f = hdf5::File::create(path)?;
        let b = f.new_dataset::<i64>().shape([n]).chunk([1<<20]);
        let b = match tag {
            "lz4" => b.blosc_lz4(1, hdf5::filters::BloscShuffle::Byte),
            "zstd" => b.blosc_zstd(1, hdf5::filters::BloscShuffle::Byte),
            "blosclz" => b.blosc_blosclz(1, hdf5::filters::BloscShuffle::Byte),
            _ => b.shuffle().deflate(4),
        };
        b.create("x")?.write(&ndarray::arr1(&data))?;
        f.close()?;
        let sz = std::fs::metadata(path)?.len();
        println!("{:8} {:.3} bytes/elem ({} MB)", tag, sz as f64/n as f64, sz/1_000_000);
    }
    Ok(())
}
