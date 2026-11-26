//! Core types of the pairs tools: the fixed-width in-memory record and the file header.
//!
//! Canonical order is (chrom1, pos1, chrom2, pos2, strand1, strand2, ...), chromosome ids being
//! positions in the header's chromsizes, with every pair flipped so side 1 <= side 2. In that
//! order bin1 is monotonic at any bin size, so a cooler can be built with no global sort.

pub const SORTED_CANONICAL: &str = "chr1-pos1-chr2-pos2";

/// Optional-column bits (`Header::columns`).
pub const HAS_PTYPE: u8 = 1;
pub const HAS_MAPQ: u8 = 2;

/// One pair. Field order is the sort order: derived `Ord` is the canonical order.
/// `k1`/`k2` = chrom id << 32 | position; `aux` packs, high to low: strand1 (bit 33, 1 = '-'),
/// strand2 (bit 32), the two pair-type characters (bits 31..16), mapq1, mapq2.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Debug)]
pub struct Pair {
    pub k1: u64,
    pub k2: u64,
    pub aux: u64,
}

impl Pair {
    /// Build a pair in upper-triangle form: the sides are swapped (strands, mapqs and the two
    /// pair-type characters with them) when side 1 sorts after side 2.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn flipped(c1: u32, p1: u32, c2: u32, p2: u32, minus1: bool, minus2: bool,
                   ptype: [u8; 2], mapq1: u8, mapq2: u8) -> Pair {
        let (k1, k2) = ((c1 as u64) << 32 | p1 as u64, (c2 as u64) << 32 | p2 as u64);
        if k1 <= k2 {
            Pair { k1, k2, aux: pack_aux(minus1, minus2, ptype, mapq1, mapq2) }
        } else {
            Pair { k1: k2, k2: k1, aux: pack_aux(minus2, minus1, [ptype[1], ptype[0]], mapq2, mapq1) }
        }
    }
    /// Assemble a pair from already-flipped fields (`strands`: bit 0 = strand1 '-', bit 1 = strand2 '-').
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn raw(c1: u32, p1: u32, c2: u32, p2: u32, strands: u8, ptype: [u8; 2], mapq1: u8, mapq2: u8) -> Pair {
        Pair { k1: (c1 as u64) << 32 | p1 as u64, k2: (c2 as u64) << 32 | p2 as u64,
               aux: pack_aux(strands & 1 != 0, strands & 2 != 0, ptype, mapq1, mapq2) }
    }
    #[inline] pub fn chrom1(&self) -> u32 { (self.k1 >> 32) as u32 }
    #[inline] pub fn pos1(&self) -> u32 { self.k1 as u32 }
    #[inline] pub fn chrom2(&self) -> u32 { (self.k2 >> 32) as u32 }
    #[inline] pub fn pos2(&self) -> u32 { self.k2 as u32 }
    /// bit 0 = strand1 is '-', bit 1 = strand2 is '-'
    #[inline] pub fn strands(&self) -> u8 { (((self.aux >> 33) & 1) | ((self.aux >> 32) & 1) << 1) as u8 }
    #[inline] pub fn ptype(&self) -> [u8; 2] { [(self.aux >> 24) as u8, (self.aux >> 16) as u8] }
    #[inline] pub fn mapq1(&self) -> u8 { (self.aux >> 8) as u8 }
    #[inline] pub fn mapq2(&self) -> u8 { self.aux as u8 }
}

#[inline]
fn pack_aux(minus1: bool, minus2: bool, ptype: [u8; 2], mapq1: u8, mapq2: u8) -> u64 {
    (minus1 as u64) << 33 | (minus2 as u64) << 32 | (ptype[0] as u64) << 24 | (ptype[1] as u64) << 16
        | (mapq1 as u64) << 8 | mapq2 as u64
}

/// What a pairs file says about itself, whatever the container.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Header {
    /// (name, length) in sort order
    pub chroms: Vec<(String, u64)>,
    /// `SORTED_CANONICAL`, or "none"
    pub sorted: String,
    pub assembly: String,
    /// which optional columns the file carries (HAS_* bits)
    pub columns: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_swaps_everything() {
        let p = Pair::flipped(2, 10, 1, 99, true, false, *b"UR", 7, 60);
        assert_eq!((p.chrom1(), p.pos1(), p.chrom2(), p.pos2()), (1, 99, 2, 10));
        assert_eq!((p.strands(), p.ptype(), p.mapq1(), p.mapq2()), (0b10, *b"RU", 60, 7));
        let q = Pair::flipped(1, 99, 2, 10, false, true, *b"RU", 60, 7);
        assert_eq!(p, q);
        // raw() is the inverse of the accessors
        assert_eq!(Pair::raw(p.chrom1(), p.pos1(), p.chrom2(), p.pos2(), p.strands(), p.ptype(), p.mapq1(), p.mapq2()), p);
    }

    #[test]
    fn derived_order_is_chrom1_pos1_chrom2_pos2() {
        let mk = |c1, p1, c2, p2| Pair::raw(c1, p1, c2, p2, 0, [0, 0], 0, 0);
        let mut v = vec![mk(1, 5, 1, 9), mk(0, 7, 3, 1), mk(0, 7, 1, 8), mk(0, 2, 5, 5), mk(0, 7, 1, 2)];
        v.sort();
        assert_eq!(v, vec![mk(0, 2, 5, 5), mk(0, 7, 1, 2), mk(0, 7, 1, 8), mk(0, 7, 3, 1), mk(1, 5, 1, 9)]);
    }
}
