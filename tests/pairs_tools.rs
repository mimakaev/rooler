//! End-to-end tests of the pairs tools: SAM parsing against pairtools' output, sort, merge, dedup.
use rooler::pairs::Pair;
use rooler::pairs_parse::{parse, ParseOpts};
use rooler::pairs_pq::{PqCodec, PqFile};
use rooler::pairs_tools::{merge, sort, view, Deduper, SortOpts};

/// A fresh scratch directory, removed when dropped.
struct Tmp(std::path::PathBuf);
impl Tmp {
    fn new(tag: &str) -> Tmp {
        let d = std::env::temp_dir().join(format!("rooler_pairs_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Tmp(d)
    }
    fn path(&self) -> &std::path::Path { &self.0 }
}
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

fn opts(tmp: &std::path::Path, mem_gb: f64) -> SortOpts {
    SortOpts { mem_gb, threads: 3, tmpdir: tmp.join("runs").to_str().unwrap().into(),
               run_codec: PqCodec::Zstd(1), out_codec: PqCodec::Zstd(1) }
}

fn text(path: &str, min_mapq: u8) -> Vec<String> {
    let mut out = Vec::new();
    view(path, &mut out, false, min_mapq, None).unwrap();
    String::from_utf8(out).unwrap().lines().map(|l| l.split_once('\t').unwrap().1.to_string()).collect()
}

/// `tests/data/parse_small.sam` is 1500 read pairs of a real mouse Hi-C library (bwa mem -SP5M),
/// chosen to be rich in multi-alignment reads; the expected file is the UU pairs that
/// `pairtools parse --walks-policy all --add-columns mapq` (1.1.2) makes of it.
#[test]
fn parse_matches_pairtools() {
    let d = Tmp::new("parse");
    let out = d.path().join("p.parquet");
    let out = out.to_str().unwrap();
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/");
    let n = parse(&format!("{}parse_small.sam", data), out, ParseOpts {
        min_mapq: 1, max_gap: 20, max_molecule: 750,
        chroms_path: Some(format!("{}parse_small.chroms", data)), assembly: "mm10".into(),
    }, &opts(d.path(), 0.001), false).unwrap();
    let mut got = text(out, 0);
    got.sort();
    let mut want: Vec<String> = std::fs::read_to_string(format!("{}parse_small.pairtools_uu.txt", data)).unwrap()
        .lines().map(|s| s.to_string()).collect();
    want.sort();
    assert_eq!(n as usize, want.len());
    assert_eq!(got, want);
    // pair-type counts of the same pairtools run
    let f = PqFile::open(out).unwrap();
    for line in ["pair_types/UU\t1436", "pair_types/MU\t225", "pair_types/MM\t67", "pair_types/NU\t20", "pair_types/NM\t2", "read_pairs\t1500"] {
        assert!(f.stats.lines().any(|l| l == line), "missing {:?} in\n{}", line, f.stats);
    }
    // the mapq filter of `view`
    let hi = text(out, 30);
    assert!(!hi.is_empty() && hi.len() < got.len());
    assert!(hi.iter().all(|l| l.split('\t').skip(6).all(|q| q.parse::<u8>().unwrap() >= 30)));
}

fn lcg(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s >> 33 }

/// Random pile-ups: many pairs within a few bp of each other on both sides.
fn random_pairs(n: usize, seed: u64) -> Vec<Pair> {
    let mut s = seed;
    let mut v: Vec<Pair> = (0..n).map(|_| {
        let c1 = (lcg(&mut s) % 2) as u32;
        let c2 = c1 + (lcg(&mut s) % 2) as u32;
        Pair::flipped(c1, 1 + (lcg(&mut s) % 40) as u32, c2, 1 + (lcg(&mut s) % 40) as u32,
                      lcg(&mut s) % 2 == 0, lcg(&mut s) % 2 == 0, [0, 0], (lcg(&mut s) % 61) as u8, 60)
    }).collect();
    v.sort();
    v
}

/// The rule, written the slow obvious way.
fn dedup_reference(v: &[Pair], mm: u32) -> Vec<Pair> {
    let mut kept: Vec<Pair> = Vec::new();
    for p in v {
        let dup = kept.iter().any(|q| q.chrom1() == p.chrom1() && q.chrom2() == p.chrom2() && q.strands() == p.strands()
            && q.pos1().abs_diff(p.pos1()) <= mm && q.pos2().abs_diff(p.pos2()) <= mm);
        if !dup { kept.push(*p); }
    }
    kept
}

#[test]
fn dedup_matches_reference_however_the_stream_is_cut() {
    for mm in [0u32, 1, 3] {
        let v = random_pairs(6000, 7 + mm as u64);
        let want = dedup_reference(&v, mm);
        assert!(want.len() < v.len());
        for slab in [v.len(), 1000, 37, 1] {
            let mut d = Deduper::new(mm);
            let mut got = Vec::new();
            let mut ndup = 0;
            for c in v.chunks(slab) {
                let mut c = c.to_vec();
                ndup += d.run(&mut c);
                got.extend(c);
            }
            assert_eq!(got, want, "mm {} slab {}", mm, slab);
            assert_eq!(ndup as usize, v.len() - want.len());
        }
    }
}

#[test]
fn sort_then_merge_dedup_end_to_end() {
    let d = Tmp::new("e2e");
    let p = |n: &str| d.path().join(n).to_str().unwrap().to_string();
    let chroms = ["chrA", "chrB", "chrC"];
    let all = random_pairs(50_000, 99);
    // two unsorted text files, sides in arbitrary order
    let mut s = 5u64;
    for (k, part) in all.chunks(30_000).enumerate() {
        let lines: Vec<String> = part.iter().map(|q| {
            let (a, b) = ((chroms[q.chrom1() as usize], q.pos1(), q.strands() & 1, q.mapq1()),
                          (chroms[q.chrom2() as usize], q.pos2(), q.strands() >> 1, q.mapq2()));
            // a pair with both sides at one position has no defined side order: leave those alone
            let (a, b) = if lcg(&mut s) % 2 == 0 || (a.0, a.1) == (b.0, b.1) { (a, b) } else { (b, a) };
            let st = |x: u8| if x != 0 { '-' } else { '+' };
            format!("r\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", a.0, a.1, b.0, b.1, st(a.2), st(b.2), a.3, b.3)
        }).collect();
        let mut keyed: Vec<(u64, String)> = lines.into_iter().map(|l| (lcg(&mut s), l)).collect();
        keyed.sort();
        let lines: Vec<String> = keyed.into_iter().map(|x| x.1).collect();
        let hdr = "## pairs format v1.0\n#chromsize: chrA 1000\n#chromsize: chrB 1000\n#chromsize: chrC 1000\n\
                   #columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2 mapq1 mapq2\n";
        std::fs::write(p(&format!("in{}.pairs", k)), format!("{}{}\n", hdr, lines.join("\n"))).unwrap();
        // tiny memory budget: many spill runs and several merge units
        sort(&p(&format!("in{}.pairs", k)), &p(&format!("s{}.parquet", k)), &opts(d.path(), 0.0005), false).unwrap();
    }
    let ins = vec![p("s0.parquet"), p("s1.parquet")];
    let fmt = |v: &[Pair]| -> Vec<String> { v.iter().map(|q| format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        chroms[q.chrom1() as usize], q.pos1(), chroms[q.chrom2() as usize], q.pos2(),
        if q.strands() & 1 != 0 { '-' } else { '+' }, if q.strands() & 2 != 0 { '-' } else { '+' }, q.mapq1(), q.mapq2())).collect() };
    merge(&ins, &p("m.parquet"), &opts(d.path(), 0.0005), None, false).unwrap();
    assert_eq!(text(&p("m.parquet"), 0), fmt(&all));
    let st = merge(&ins, &p("d.parquet"), &opts(d.path(), 0.0005), Some(1), false).unwrap();
    let want = dedup_reference(&all, 1);
    assert_eq!(text(&p("d.parquet"), 0), fmt(&want));
    assert_eq!((st.nodups as usize, st.dups as usize), (want.len(), all.len() - want.len()));
    assert_eq!(st.cis + st.trans, st.nodups);
}
