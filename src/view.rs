//! Genome "view" (regions for expected): arms / whole-chroms / custom.
//! Opinionated per-genome defaults, no external (bioframe) dependency. Centromere midpoints for
//! arm genomes are baked in (approximate is fine for per-arm expected; regenerate from UCSC
//! cytoBand via scripts/arms.py). Unknown genomes must pass an explicit --view or we refuse.

#[derive(Clone, Debug, PartialEq)]
pub enum ViewKind { Arms, Chroms }

#[derive(Clone, Debug)]
pub struct Region { pub name: String, pub chrom: String, pub start: i64, pub end: i64 }

// approximate hg38 centromere midpoints (bp), chr1..22,X,Y — fine for arm-splitting expected.
const HG38_CEN: &[(&str, i64)] = &[
    ("chr1",123400000),("chr2",93900000),("chr3",90900000),("chr4",50000000),("chr5",48800000),
    ("chr6",59800000),("chr7",60100000),("chr8",45200000),("chr9",43000000),("chr10",39800000),
    ("chr11",53400000),("chr12",35500000),("chr13",17700000),("chr14",17200000),("chr15",19000000),
    ("chr16",36800000),("chr17",25100000),("chr18",18500000),("chr19",26200000),("chr20",28100000),
    ("chr21",12000000),("chr22",15000000),("chrX",61000000),("chrY",10400000),
];
const HG19_CEN: &[(&str, i64)] = &[
    ("chr1",125000000),("chr2",93300000),("chr3",91000000),("chr4",50400000),("chr5",48400000),
    ("chr6",61000000),("chr7",59900000),("chr8",45600000),("chr9",49000000),("chr10",40200000),
    ("chr11",53700000),("chr12",35800000),("chr13",17900000),("chr14",17600000),("chr15",19000000),
    ("chr16",36600000),("chr17",24000000),("chr18",17200000),("chr19",26500000),("chr20",27500000),
    ("chr21",13200000),("chr22",14700000),("chrX",60600000),("chrY",12500000),
];

// sacCer3 CEN midpoints (SGD/UCSC, approximate to ~1kb — fine for arm-splitting expected).
const SACCER3_CEN: &[(&str, i64)] = &[
    ("chrI",151465),("chrII",238207),("chrIII",114385),("chrIV",449711),
    ("chrV",151987),("chrVI",148510),("chrVII",496920),("chrVIII",105586),
    ("chrIX",355629),("chrX",436307),("chrXI",440129),("chrXII",150828),
    ("chrXIII",268031),("chrXIV",628758),("chrXV",326584),("chrXVI",555957),
];

/// Resolve the assembly name to stamp into a cooler. Explicit user value wins (any non-empty
/// string is trusted as provenance); otherwise fingerprint the chromsizes. None => indeterminate,
/// and callers must REFUSE to write (no mystery coolers).
pub fn resolve_assembly(user: Option<&str>, chromsizes: &[(String, i64)]) -> Option<String> {
    if let Some(u) = user { let u = u.trim(); if !u.is_empty() { return Some(u.to_string()); } }
    detect("", chromsizes).map(|s| s.to_string())
}

/// Normalize an assembly name and/or fingerprint chromsizes -> canonical genome id.
pub fn detect(assembly: &str, chromsizes: &[(String, i64)]) -> Option<&'static str> {
    let a = assembly.to_lowercase();
    let a = a.trim_end_matches("_ebv").trim_end_matches("_analysis_set");
    let by_name = match a {
        "grch38" | "hg38" => Some("hg38"),
        "grch37" | "hg19" => Some("hg19"),
        "grcm38" | "mm10" => Some("mm10"),
        "grcm39" | "mm39" => Some("mm39"),
        "dm6" | "bdgp6" => Some("dm6"),
        "saccer3" | "r64" => Some("saccer3"),
        "ce11" | "wbcel235" | "ce10" => Some("ce11"),
        _ => None,
    };
    if by_name.is_some() { return by_name; }
    // fingerprint by chr1 length (our writers stamp assembly="unknown")
    let chr1 = chromsizes.iter().find(|(c, _)| c == "chr1" || c == "1").map(|(_, l)| *l);
    match chr1 {
        Some(248956422) => Some("hg38"),
        Some(249250621) => Some("hg19"),
        Some(195471971) => Some("mm10"),
        Some(195154279) => Some("mm39"),
        _ => None,
    }
}

/// Default view kind for a known genome, or None if we don't have an opinion.
fn default_kind(genome: &str) -> Option<ViewKind> {
    match genome {
        "hg38" | "hg19" | "saccer3" => Some(ViewKind::Arms),
        "mm10" | "mm39" | "dm6" | "ce11" => Some(ViewKind::Chroms),
        _ => None,
    }
}

fn centromeres(genome: &str) -> Option<&'static [(&'static str, i64)]> {
    match genome {
        "hg38" => Some(HG38_CEN), "hg19" => Some(HG19_CEN), "saccer3" => Some(SACCER3_CEN), _ => None,
    }
}

fn whole_chroms(chromsizes: &[(String, i64)]) -> Vec<Region> {
    chromsizes.iter().map(|(c, l)| Region { name: c.clone(), chrom: c.clone(), start: 0, end: *l }).collect()
}

fn arms_view(chromsizes: &[(String, i64)], cen: &[(&str, i64)]) -> Vec<Region> {
    let mut out = Vec::new();
    for (c, l) in chromsizes {
        if let Some(&(_, mid)) = cen.iter().find(|(cc, _)| cc == c) {
            let mid = mid.min(*l);
            out.push(Region { name: format!("{}_p", c), chrom: c.clone(), start: 0, end: mid });
            out.push(Region { name: format!("{}_q", c), chrom: c.clone(), start: mid, end: *l });
        } else {
            out.push(Region { name: c.clone(), chrom: c.clone(), start: 0, end: *l });
        }
    }
    out
}

/// Resolve the region list for expected. `requested`: None=use genome default; Some("chroms"|"arms").
/// Returns Err for unknown genome with no explicit request, or "arms" requested without a centromere table.
pub fn resolve(assembly: &str, chromsizes: &[(String, i64)], requested: Option<&str>)
    -> Result<(String, Vec<Region>), String> {
    let genome = detect(assembly, chromsizes);
    let kind = match requested {
        Some("chroms") => ViewKind::Chroms,
        Some("arms") => ViewKind::Arms,
        Some(other) => return Err(format!("unknown --view '{}' (use chroms|arms|custom:<bed>)", other)),
        None => match genome.and_then(default_kind) {
            Some(k) => k,
            None => return Err(format!(
                "unknown genome (assembly='{}'); pass --view chroms|arms|custom:<bed> to compute expected",
                assembly)),
        },
    };
    let name = match &kind { ViewKind::Arms => "arms", ViewKind::Chroms => "chroms" }.to_string();
    match kind {
        ViewKind::Chroms => Ok((name, whole_chroms(chromsizes))),
        ViewKind::Arms => match genome.and_then(centromeres) {
            Some(cen) => Ok((name, arms_view(chromsizes, cen))),
            None => Err(format!(
                "no centromere table for genome (assembly='{}'); use --view chroms or custom:<bed>",
                assembly)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hg38_sizes() -> Vec<(String, i64)> {
        vec![("chr1".into(), 248956422), ("chr2".into(), 242193529), ("chrX".into(), 156040895)]
    }
    #[test]
    fn hg38_defaults_to_arms() {
        let (name, regs) = resolve("unknown", &hg38_sizes(), None).unwrap();
        assert_eq!(name, "arms");
        // chr1 -> chr1_p, chr1_q
        assert!(regs.iter().any(|r| r.name == "chr1_p" && r.start == 0 && r.end == 123400000));
        assert!(regs.iter().any(|r| r.name == "chr1_q" && r.start == 123400000 && r.end == 248956422));
    }
    #[test]
    fn mouse_defaults_to_chroms() {
        let sizes = vec![("chr1".into(), 195471971i64)];
        let (name, regs) = resolve("GRCm38", &sizes, None).unwrap();
        assert_eq!(name, "chroms");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].end, 195471971);
    }
    #[test]
    fn unknown_genome_refuses() {
        let sizes = vec![("scaffold_1".into(), 12345i64)];
        assert!(resolve("weirdbug", &sizes, None).is_err());
        // but explicit chroms works
        assert!(resolve("weirdbug", &sizes, Some("chroms")).is_ok());
    }
}
