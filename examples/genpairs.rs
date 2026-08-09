//! Synthetic 4DN .pairs generator for scale testing — streams to stdout, stores nothing.
//!
//!     cargo build --release --example genpairs
//!     ./target/release/examples/genpairs <npairs> [seed] | rooler cload - 64 out.cool --mem 48
//!
//! Contacts follow a plausible Hi-C shape so the resulting cooler is realistically sparse:
//! 70% cis with P(s) ~ 1/s over [500, chrom length] (inverse-CDF sampled), 30% trans uniform
//! over the genome. hg38 chromsizes are baked in. Deterministic: same (npairs, seed) gives the
//! same stream, so a run can be reproduced or resumed.
//!
//! Generation is parallel (worker threads format line blocks, one writer drains them) because a
//! single formatting thread cannot keep a multi-threaded `rooler cload` fed.
use std::io::Write;

const HG38: &[(&str, i64)] = &[
    ("chr1", 248956422), ("chr2", 242193529), ("chr3", 198295559), ("chr4", 190214555),
    ("chr5", 181538259), ("chr6", 170805979), ("chr7", 159345973), ("chr8", 145138636),
    ("chr9", 138394717), ("chr10", 133797422), ("chr11", 135086622), ("chr12", 133275309),
    ("chr13", 114364328), ("chr14", 107043718), ("chr15", 101991189), ("chr16", 90338345),
    ("chr17", 83257441), ("chr18", 80373285), ("chr19", 58617616), ("chr20", 64444167),
    ("chr21", 46709983), ("chr22", 50818468), ("chrX", 156040895), ("chrY", 57227415),
];

const BLOCK_PAIRS: usize = 250_000; // pairs formatted per work unit

struct Lcg(u64);
impl Lcg {
    #[inline]
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    #[inline]
    fn below(&mut self, n: u64) -> u64 { self.next() % n.max(1) }
    #[inline]
    fn unit(&mut self) -> f64 { (self.next() & ((1 << 53) - 1)) as f64 / (1u64 << 53) as f64 }
}

/// append a non-negative integer without going through fmt (this is the hot path)
#[inline]
fn put_int(out: &mut Vec<u8>, mut v: i64) {
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 { break; }
    }
    out.extend_from_slice(&buf[i..]);
}

/// cumulative genome offsets, for picking a chromosome proportionally to its length
fn cum() -> (Vec<i64>, i64) {
    let mut c = Vec::with_capacity(HG38.len());
    let mut acc = 0i64;
    for &(_, l) in HG38 { acc += l; c.push(acc); }
    (c, acc)
}

#[inline]
fn pick(cum: &[i64], total: i64, r: &mut Lcg) -> usize {
    let x = r.below(total as u64) as i64;
    cum.partition_point(|&c| c <= x)
}

fn gen_block(seed: u64, n: usize, cum: &[i64], total: i64, out: &mut Vec<u8>) {
    let mut r = Lcg(seed);
    for i in 0..n {
        let c1 = pick(cum, total, &mut r);
        let (name1, len1) = HG38[c1];
        let p1 = r.below(len1 as u64) as i64;
        let (name2, p2) = if r.below(10) < 7 {
            // cis: separation s ~ 1/s on [500, len1] via inverse CDF, random direction
            let lo = 500f64;
            let hi = (len1 as f64).max(lo * 2.0);
            let s = (lo * (hi / lo).powf(r.unit())) as i64;
            let q = if r.below(2) == 0 { p1 + s } else { p1 - s };
            (name1, q.clamp(0, len1 - 1))
        } else {
            let c2 = pick(cum, total, &mut r);
            let (n2, l2) = HG38[c2];
            (n2, r.below(l2 as u64) as i64)
        };
        out.extend_from_slice(b"r");
        put_int(out, i as i64);
        out.push(b'\t');
        out.extend_from_slice(name1.as_bytes());
        out.push(b'\t');
        put_int(out, p1);
        out.push(b'\t');
        out.extend_from_slice(name2.as_bytes());
        out.push(b'\t');
        put_int(out, p2);
        out.extend_from_slice(b"\t+\t-\n");
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let npairs: u64 = args.next().and_then(|s| s.replace('_', "").parse().ok())
        .unwrap_or_else(|| { eprintln!("usage: genpairs <npairs> [seed] [threads]"); std::process::exit(2) });
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(20260809);
    let nthreads: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(6);

    let (cum, total) = cum();
    let mut hdr = Vec::new();
    hdr.extend_from_slice(b"## pairs format v1.0\n#sorted: none\n");
    hdr.extend_from_slice(b"#columns: readID chr1 pos1 chr2 pos2 strand1 strand2\n");
    hdr.extend_from_slice(b"#genome_assembly: hg38\n");
    for &(n, l) in HG38 { hdr.extend_from_slice(format!("#chromsize: {} {}\n", n, l).as_bytes()); }

    let stdout = std::io::stdout();
    let mut w = std::io::BufWriter::with_capacity(1 << 22, stdout.lock());
    w.write_all(&hdr).unwrap();

    let nblocks = npairs.div_ceil(BLOCK_PAIRS as u64);
    let (tx, rx) = crossbeam_channel::bounded::<(u64, Vec<u8>)>(nthreads * 4);
    let counter = std::sync::atomic::AtomicU64::new(0);

    std::thread::scope(|s| {
        let counter = &counter;
        let cum = &cum;
        for _ in 0..nthreads {
            let tx = tx.clone();
            s.spawn(move || {
                let mut out = Vec::with_capacity(BLOCK_PAIRS * 48);
                loop {
                    let b = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if b >= nblocks { break; }
                    let n = BLOCK_PAIRS.min((npairs - b * BLOCK_PAIRS as u64) as usize);
                    out.clear();
                    // seed per block so the stream is reproducible regardless of scheduling
                    gen_block(seed ^ b.wrapping_mul(0x9E3779B97F4A7C15), n, cum, total, &mut out);
                    if tx.send((b, std::mem::take(&mut out))).is_err() { return; }
                    out = Vec::with_capacity(BLOCK_PAIRS * 48);
                }
            });
        }
        drop(tx);
        // pairs are unordered, so blocks may be written in whatever order they finish
        for (_, buf) in rx {
            if w.write_all(&buf).is_err() { std::process::exit(0); } // downstream closed the pipe
        }
    });
    w.flush().ok();
}
