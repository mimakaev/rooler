//! SAM text -> pairs. A port of `pairtools parse --walks-policy all` (pairtools 1.1.2), i.e. the
//! parse2 walk algorithm reporting outer ends in pair orientation: every ligation junction seen
//! in a read pair is reported, and junctions sequenced from both ends are reported once.
//!
//! The port follows pairtools/lib/parse.py statement by statement, including where the original
//! is quirky, so that results can be compared pair for pair:
//!   * `_convert_gaps_into_alignments` does not examine the last k alignments of a read after
//!     inserting k gap alignments, and never looks at the 3' tail of the read;
//!   * `format_pair` tests for pair type "R1&R2" while the walk code produces "R1&2", so a
//!     junction confirmed by both reads is reported at the 5' ends of the two READ 1 alignments
//!     with read orientation (not the outer ends);
//!   * CIGAR bookkeeping counts only M as matching (`=`/`X` are ignored), and the 5'/3' clip is
//!     the last S/H seen before/after the first M.
//! Only pairs with both sides uniquely mapped are emitted; the rest are counted by pair type.
//!
//! Input is name-grouped SAM (mapper output order), all records of a read pair adjacent.
use crate::pairs::{Header, Pair, HAS_MAPQ, SORTED_CANONICAL};
use crate::pairs_tools::{BlockParser, SortInput, SortOpts, Stats};
use anyhow::{anyhow, bail, Result};
use memchr::{memchr, memrchr};
use std::collections::HashMap;
use std::io::Read;
use std::sync::{Arc, Mutex};

pub struct ParseOpts {
    /// alignments with MAPQ below this are "multi-mapped" (pairtools --min-mapq)
    pub min_mapq: u8,
    /// unaligned read stretches longer than this become null alignments (--max-inter-align-gap)
    pub max_gap: i32,
    /// pairtools --max-molecule-size; used as the insert-size limit of the read-through check
    pub max_molecule: i64,
    /// chromosome order file (first column = names); unlisted contigs follow, sorted by name
    pub chroms_path: Option<String>,
    pub assembly: String,
}

/// One alignment (a SAM record, or a null alignment standing for an unaligned stretch).
#[derive(Clone, Copy, Debug, Default)]
struct Al {
    mapped: bool,
    /// mapped and MAPQ >= min_mapq
    uniq: bool,
    chrom: u32,
    pos5: i64,
    pos3: i64,
    minus: bool,
    mapq: u8,
    dist5: i32,
    read_span: i32,
    read_len: i32,
    /// pysam query_alignment_start: leading soft clip in SAM (reference) orientation
    qstart: i32,
    read1: bool,
    read2: bool,
}

impl Al {
    fn null() -> Al { Al { minus: true, ..Default::default() } }
}

struct SamCtx { cmap: HashMap<Vec<u8>, u32>, lengths: Vec<u64>, o: ParseOpts }

#[inline]
fn num(b: &[u8]) -> Option<i64> {
    if b.is_empty() || b.len() > 18 { return None; }
    let mut v = 0i64;
    for &c in b { if !c.is_ascii_digit() { return None; } v = v * 10 + (c - b'0') as i64; }
    Some(v)
}

/// QNAME of a SAM line.
#[inline]
fn qname(line: &[u8]) -> &[u8] { &line[..memchr(b'\t', line).unwrap_or(line.len())] }

fn parse_record<'a>(line: &'a [u8], cx: &SamCtx, last: &mut (Vec<u8>, u32)) -> Result<(&'a [u8], Al)> {
    let bad = || anyhow!("malformed SAM line: {:?}", String::from_utf8_lossy(&line[..line.len().min(200)]));
    let mut f: [&[u8]; 6] = [&[]; 6];
    let mut start = 0usize;
    for slot in f.iter_mut() {
        let rel = memchr(b'\t', &line[start..]).ok_or_else(bad)?;
        *slot = &line[start..start + rel];
        start += rel + 1;
    }
    let flag = num(f[1]).ok_or_else(bad)?;
    let mapq = num(f[4]).ok_or_else(bad)?.min(255) as u8;
    let mut a = Al { mapped: flag & 4 == 0, mapq, read1: flag & 0x40 != 0, read2: flag & 0x80 != 0, minus: true, ..Default::default() };
    // CIGAR bookkeeping, as pairtools' cigar_dict
    let (mut matched, mut ref_span, mut read_span, mut read_len, mut clip5, mut clip3) = (0i32, 0i32, 0i32, 0i32, 0i32, 0i32);
    let (mut qstart, mut leading) = (0i32, true);
    if f[5] != b"*" {
        let mut n = 0i32;
        for &c in f[5] {
            if c.is_ascii_digit() { n = n.wrapping_mul(10).wrapping_add((c - b'0') as i32); continue; }
            match c {
                b'M' => { matched += n; ref_span += n; read_span += n; read_len += n; }
                b'I' => { read_span += n; read_len += n; }
                b'D' => { ref_span += n; }
                b'S' | b'H' => { read_len += n; if matched == 0 { clip5 = n; } else { clip3 = n; } }
                _ => {}
            }
            if leading { match c { b'S' => qstart += n, b'H' => {}, _ => leading = false } }
            n = 0;
        }
    }
    a.read_span = read_span; a.read_len = read_len; a.qstart = qstart;
    if a.mapped {
        let minus = flag & 0x10 != 0;
        a.dist5 = if minus { clip3 } else { clip5 };
        a.uniq = mapq >= cx.o.min_mapq;
        if a.uniq {
            let id = if f[2] == &last.0[..] { last.1 } else {
                let id = *cx.cmap.get(f[2]).ok_or_else(|| anyhow!(
                    "SAM record on {:?}, which is not in the @SQ header", String::from_utf8_lossy(f[2])))?;
                last.0.clear(); last.0.extend_from_slice(f[2]); last.1 = id;
                id
            };
            let pos = num(f[3]).ok_or_else(bad)?; // 1-based leftmost
            if pos < 1 { return Err(bad()); }
            // an alignment hanging over the chromosome end is clamped to it
            let len = cx.lengths[id as usize] as i64;
            let (pos, right) = (pos.min(len), (pos - 1 + ref_span as i64).clamp(1, len));
            a.chrom = id;
            a.minus = minus;
            if minus { a.pos5 = right; a.pos3 = pos; } else { a.pos5 = pos; a.pos3 = right; }
        }
    }
    Ok((f[0], a))
}

/// pairtools `_convert_gaps_into_alignments`, including its loop quirk (see module doc).
fn convert_gaps(v: &mut Vec<Al>, max_gap: i32) {
    if v.len() == 1 && !v[0].mapped { return; }
    let n0 = v.len();
    let mut last = 0i32;
    for i in 0..n0 {
        let a = v[i];
        if a.dist5 - last > max_gap {
            let mut e = Al::null();
            e.dist5 = last;
            e.read_span = a.dist5 - last;
            e.read_len = a.read_len;
            last = a.dist5 + a.read_span;
            v.insert(i, e);
        } else {
            last = last.max(a.dist5 + a.read_span);
        }
    }
}

fn normalize(v: &mut Vec<Al>, max_gap: i32) {
    if v.is_empty() { v.push(Al::null()); }
    v.sort_by_key(|a| a.dist5); // stable, like Python's sorted()
    convert_gaps(v, max_gap);
}

fn pairs_overlap(l5: &Al, l3: &Al, r5: &Al, r3: &Al, mm: i64) -> bool {
    let side = |a: &Al, b: &Al| match (a.uniq, b.uniq) {
        (false, false) => true,
        (true, true) => a.chrom == b.chrom && a.minus != b.minus,
        _ => false,
    };
    side(l5, r3) && side(l3, r5) && (l5.pos3 - r3.pos5).abs() <= mm && (l3.pos5 - r5.pos3).abs() <= mm
}

fn partial_overlap(a1: &Al, a2: &Al, max_insert: i64, mm: i64) -> bool {
    if !a1.uniq && !a2.uniq { return true; }
    if !a1.uniq || !a2.uniq { return false; }
    let (min_size, dist) = if !a1.minus {
        ((a1.pos3 - a1.pos5).max(a2.pos5 - a2.pos3), a2.pos5 - a1.pos5)
    } else {
        ((a1.pos5 - a1.pos3).max(a2.pos3 - a2.pos5), a1.pos5 - a2.pos5)
    };
    a1.chrom == a2.chrom && a1.minus != a2.minus && dist <= max_insert + mm && dist >= min_size - mm
}

/// One reported side: (alignment, position, is minus strand).
type Side = (Al, i64, bool);
#[inline] fn as_is(a: &Al) -> Side { (*a, a.pos5, a.minus) }
/// the alignment seen from its other end: 3' position, opposite strand
#[inline] fn outer(a: &Al, pos3: Option<i64>) -> Side { (*a, pos3.unwrap_or(a.pos3), !a.minus) }

/// pairtools `parse_complex_walk` with report_position="outer", report_orientation="pair",
/// dedup_max_mismatch=3, no expansion. Calls `emit` once per reported pair.
fn complex_walk(a1: &[Al], a2: &[Al], max_insert: i64, emit: &mut impl FnMut(Side, Side)) {
    const MM: i64 = 3;
    let (n1, n2) = (a1.len(), a2.len());
    let mut cur_right = 1usize;
    if n1 >= 2 && n2 >= 2 {
        let (rem_left, mut rem_right, mut checked) = (n1 - 1, n2 - 1, 0usize);
        let mut found = false;
        while rem_left > checked && rem_right > 0 {
            let mut ov = pairs_overlap(&a1[n1 - 2], &a1[n1 - 1], &a2[n2 - cur_right - 1], &a2[n2 - cur_right], MM);
            if ov {
                let (mut l, mut r, mut c) = (1usize, cur_right, checked);
                while ov && c > 0 {
                    l += 1; r -= 1;
                    ov &= pairs_overlap(&a1[n1 - l - 1], &a1[n1 - l], &a2[n2 - r - 1], &a2[n2 - r], MM);
                    c -= 1;
                }
                if ov { cur_right += 1; found = true; break; }
            }
            cur_right += 1; checked += 1; rem_right -= 1;
        }
        if !found { cur_right = 1; }
    }
    let (last_left, last_right);
    if cur_right == 1 {
        let (mut ll, mut lr) = (1usize, 1usize);
        if partial_overlap(&a1[n1 - 1], &a2[n2 - 1], max_insert, MM) {
            // read 2 reads through into the last junction of read 1 (or the other way round)
            if n1 >= 2 { emit(as_is(&a1[n1 - 2]), outer(&a1[n1 - 1], Some(a2[n2 - 1].pos5))); ll = 2; }
            if n2 >= 2 { emit(outer(&a2[n2 - 1], Some(a1[n1 - 1].pos5)), as_is(&a2[n2 - 2])); lr = 2; }
        } else {
            emit(as_is(&a1[n1 - 1]), as_is(&a2[n2 - 1]));
        }
        last_left = ll; last_right = lr;
    } else {
        last_left = cur_right; last_right = cur_right;
    }
    // junctions inside read 1
    for i in 0..n1.saturating_sub(last_left) { emit(as_is(&a1[i]), outer(&a1[i + 1], None)); }
    // junctions seen by both reads ("R1&2": reported unflipped, see module doc)
    for k in 0..cur_right - 1 { let i = n1 - cur_right + k; emit(as_is(&a1[i]), as_is(&a1[i + 1])); }
    // junctions inside read 2
    for i in 0..cur_right.min(n2.saturating_sub(last_right)) { emit(outer(&a2[i + 1], None), as_is(&a2[i])); }
}

/// Pair-type counters: NN NM NU MM MU UU XX.
pub const PTYPES: [&str; 7] = ["NN", "NM", "NU", "MM", "MU", "UU", "XX"];

#[derive(Default)]
struct Scratch { recs: Vec<Al>, s1: Vec<Al>, s2: Vec<Al> }

/// All records of one read pair -> pairs.
fn parse_group(sc: &mut Scratch, o: &ParseOpts, counts: &mut [u64; 7], out: &mut Vec<Pair>) {
    // pairtools: sorted(sams, key=(is_read2, query_alignment_start)), then split by the read-1 flag
    sc.recs.sort_by_key(|a| (a.read2, a.qstart));
    sc.s1.clear(); sc.s2.clear();
    for a in &sc.recs { if a.read1 { sc.s1.push(*a); } else { sc.s2.push(*a); } }
    if (sc.s1.is_empty() && sc.s2.len() < 2) || (sc.s2.is_empty() && sc.s1.len() < 2) {
        counts[6] += 1;
        return;
    }
    normalize(&mut sc.s1, o.max_gap);
    normalize(&mut sc.s2, o.max_gap);
    let mut emit = |x: Side, y: Side| {
        let rank = |a: &Al| if !a.mapped { 0usize } else if !a.uniq { 1 } else { 2 };
        let (r1, r2) = (rank(&x.0).min(rank(&y.0)), rank(&x.0).max(rank(&y.0)));
        counts[[[0, 1, 2], [1, 3, 4], [2, 4, 5]][r1][r2]] += 1;
        if r1 == 2 {
            out.push(Pair::flipped(x.0.chrom, x.1 as u32, y.0.chrom, y.1 as u32, x.2, y.2, [0, 0], x.0.mapq, y.0.mapq));
        }
    };
    if sc.s1.len() == 1 && sc.s2.len() == 1 {
        emit(as_is(&sc.s1[0]), as_is(&sc.s2[0]));
    } else {
        complex_walk(&sc.s1, &sc.s2, o.max_molecule, &mut emit);
    }
}

/// Block boundary for name-grouped SAM: the start of the last read group in the buffer, so no
/// group is split between blocks.
fn cut_sam(buf: &[u8]) -> Option<usize> {
    let end = memrchr(b'\n', buf)?;
    let mut start = memrchr(b'\n', &buf[..end]).map(|p| p + 1).unwrap_or(0);
    let q = qname(&buf[start..end]);
    loop {
        if start == 0 { return None; }
        let prev = memrchr(b'\n', &buf[..start - 1]).map(|p| p + 1).unwrap_or(0);
        if qname(&buf[prev..start - 1]) == q { start = prev; } else { return Some(start); }
    }
}

/// Read the @-header of a SAM stream: (name, length) per @SQ, and the body bytes already read.
fn read_sam_header(rd: &mut dyn Read) -> Result<(Vec<(String, u64)>, Vec<u8>)> {
    let mut sq = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; 1 << 20];
    let mut start = 0usize;
    loop {
        while let Some(rel) = memchr(b'\n', &buf[start..]) {
            let line = &buf[start..start + rel];
            if line.first() != Some(&b'@') { return Ok((sq, buf[start..].to_vec())); }
            if line.starts_with(b"@SQ\t") {
                let l = String::from_utf8_lossy(line);
                let (mut name, mut len) = (None, None);
                for f in l.split('\t') {
                    if let Some(v) = f.strip_prefix("SN:") { name = Some(v.to_string()); }
                    if let Some(v) = f.strip_prefix("LN:") { len = v.parse::<u64>().ok(); }
                }
                match (name, len) { (Some(n), Some(l)) => sq.push((n, l)), _ => bail!("bad @SQ line: {:?}", l) }
            }
            start += rel + 1;
        }
        buf.drain(..start);
        start = 0;
        let n = rd.read(&mut chunk)?;
        if n == 0 { return Ok((sq, buf)); }
        buf.extend_from_slice(&chunk[..n]);
    }
}

/// Name-grouped SAM text ("-" = stdin) -> sorted pairs Parquet. Returns the pair count.
pub fn parse(input: &str, out: &str, po: ParseOpts, so: &SortOpts, log: bool) -> Result<u64> {
    let mut rd: Box<dyn Read> = if input == "-" { Box::new(std::io::stdin()) } else { Box::new(std::fs::File::open(input)?) };
    let (sq, body_tail) = read_sam_header(&mut *rd)?;
    if sq.is_empty() { bail!("{}: no @SQ lines — the SAM header is required", input); }
    // chromosome order, as pairtools: names of the chroms file that the SAM has, then the rest by name
    let sizes: HashMap<&str, u64> = sq.iter().map(|(n, l)| (n.as_str(), *l)).collect();
    let mut chroms: Vec<(String, u64)> = Vec::new();
    if let Some(p) = &po.chroms_path {
        for l in std::fs::read_to_string(p)?.lines() {
            let n = l.trim().split('\t').next().unwrap_or("");
            if let Some(&len) = sizes.get(n) { if !chroms.iter().any(|c| c.0 == n) { chroms.push((n.to_string(), len)); } }
        }
    }
    let mut rest: Vec<(String, u64)> = sq.iter().filter(|c| !chroms.iter().any(|d| d.0 == c.0)).cloned().collect();
    rest.sort();
    chroms.extend(rest);
    let header = Header { chroms: chroms.clone(), sorted: SORTED_CANONICAL.into(), assembly: po.assembly.clone(), columns: HAS_MAPQ };
    let cx = Arc::new(SamCtx {
        cmap: chroms.iter().enumerate().map(|(i, (n, _))| (n.as_bytes().to_vec(), i as u32)).collect(),
        lengths: chroms.iter().map(|c| c.1).collect(),
        o: po,
    });
    let totals = Arc::new(Mutex::new(([0u64; 7], 0u64)));
    let make_parser = || -> BlockParser {
        let (cx, totals) = (cx.clone(), totals.clone());
        let mut sc = Scratch::default();
        let mut last = (Vec::new(), 0u32);
        Box::new(move |blk: &[u8], out: &mut Vec<Pair>| {
            let (mut counts, mut groups) = ([0u64; 7], 0u64);
            let mut start = 0usize;
            let mut cur: &[u8] = &[];
            sc.recs.clear();
            while let Some(rel) = memchr(b'\n', &blk[start..]) {
                let line = &blk[start..start + rel];
                start += rel + 1;
                if line.is_empty() || line[0] == b'@' { continue; }
                let (q, a) = parse_record(line, &cx, &mut last)?;
                if q != cur {
                    if !sc.recs.is_empty() { parse_group(&mut sc, &cx.o, &mut counts, out); groups += 1; }
                    sc.recs.clear();
                    cur = q;
                }
                sc.recs.push(a);
            }
            if !sc.recs.is_empty() { parse_group(&mut sc, &cx.o, &mut counts, out); groups += 1; }
            let mut t = totals.lock().unwrap();
            for k in 0..7 { t.0[k] += counts[k]; }
            t.1 += groups;
            Ok(())
        })
    };
    let nchrom = chroms.len();
    let stats = || {
        let t = totals.lock().unwrap();
        let mut s = Stats::new(nchrom);
        for (k, name) in PTYPES.iter().enumerate() {
            if t.0[k] > 0 { s.counts.insert(format!("pair_types/{}", name), t.0[k]); }
        }
        s.counts.insert("total".into(), t.0.iter().sum());
        s.counts.insert("read_pairs".into(), t.1);
        s
    };
    let n = crate::pairs_tools::sort_input(
        SortInput { rd, body_tail, child: None, name: input.to_string(), header, cut: cut_sam, make_parser: &make_parser },
        out, so, &stats, log)?;
    if log {
        let t = totals.lock().unwrap();
        let tot: u64 = t.0.iter().sum();
        eprintln!("  parse: {} read pairs -> {} pairs; {}", t.1, tot,
                  PTYPES.iter().zip(t.0.iter()).filter(|x| *x.1 > 0).map(|(n, c)| format!("{} {}", n, c)).collect::<Vec<_>>().join(", "));
    }
    Ok(n)
}
