"""rooler — thin Python read API over rooler/cooler-format .cool/.mcool files.

    r = rooler.open("f.mcool", 1000)      # or "f.mcool::resolutions/1000", or "f.cool"
    r.raw().fetch("chr1")                 # dense raw counts (symmetric)
    r.balance().fetch("chr1", "chr2")     # balanced (w_i * w_j), trans
    r.fetch("chr1:0-2,000,000", balance=True)
    r.pixels(), r.bins(), r.chroms(), r.info
    r.matrix(balance=True).fetch("chr1")  # cooler-compatible shim
"""
import numpy as np
import pandas as pd
import h5py
try:
    import hdf5plugin  # noqa: F401  (registers blosc/zstd/lz4 filters for reading)
except Exception:
    pass

__all__ = ["open", "Rooler"]


def open(uri, resolution=None):
    return Rooler(uri, resolution)


def _parse_region(region, chromsizes):
    """'chr1' or 'chr1:0-2,000,000' -> (chrom, start, end)."""
    if ":" not in region:
        return region, 0, int(chromsizes[region])
    chrom, span = region.split(":")
    lo, hi = span.replace(",", "").split("-")
    return chrom, int(lo), int(hi)


class Rooler:
    def __init__(self, uri, resolution=None):
        if "::" in uri:
            path, grp = uri.split("::", 1)
        elif resolution is not None:
            path, grp = uri, f"resolutions/{resolution}"
        else:
            path, grp = uri, "/"
        self._f = h5py.File(path, "r")
        self._g = self._f if grp in ("/", "") else self._f[grp]
        self.binsize = int(self._g.attrs.get("bin-size", self._f.attrs.get("bin-size", 0)))
        names = self._g["chroms/name"][:]
        self.chromnames = [n.decode() if isinstance(n, bytes) else n for n in names]
        self._clen = self._g["chroms/length"][:].astype(np.int64)
        self.chromsizes = pd.Series(self._clen, index=self.chromnames)
        self.chrom_offset = self._g["indexes/chrom_offset"][:].astype(np.int64)
        self.bin1_offset = self._g["indexes/bin1_offset"][:].astype(np.int64)
        self.nbins = int(self._g["bins/start"].shape[0])
        self.nnz = int(self._g["pixels/count"].shape[0])
        self._cid = {n: i for i, n in enumerate(self.chromnames)}
        self._b1 = self._g["pixels/bin1_id"]
        self._b2 = self._g["pixels/bin2_id"]
        self._cn = self._g["pixels/count"]
        self._weight = self._g["bins/weight"] if "weight" in self._g["bins"] else None

    # ---- metadata (cooler-compatible) ----
    @property
    def info(self):
        return {"nbins": self.nbins, "nnz": self.nnz, "bin-size": self.binsize,
                "nchroms": len(self.chromnames), "balanced": self._weight is not None}

    @property
    def shape(self):
        return (self.nbins, self.nbins)

    def chroms(self):
        return pd.DataFrame({"name": self.chromnames, "length": self._clen})

    def extent(self, region):
        """(bin0, bin1) for a region — cooltools uses this heavily."""
        return self._region_bins(region)

    def offset(self, region):
        return self._region_bins(region)[0]

    def bins(self):
        return _Table(self, "bins")

    def pixels(self):
        return _Table(self, "pixels")

    def weights(self):
        return None if self._weight is None else self._weight[:]

    def _bins_df(self, lo, hi):
        d = {"chrom": pd.Categorical.from_codes(self._g["bins/chrom"][lo:hi], self.chromnames),
             "start": self._g["bins/start"][lo:hi], "end": self._g["bins/end"][lo:hi]}
        if self._weight is not None:
            d["weight"] = self._weight[lo:hi]
        return pd.DataFrame(d)

    def _pixels_df(self, lo, hi):
        return pd.DataFrame({"bin1_id": self._b1[lo:hi], "bin2_id": self._b2[lo:hi],
                             "count": self._cn[lo:hi]})

    # ---- region -> bin range ----
    def _region_bins(self, region):
        chrom, start, end = _parse_region(region, self.chromsizes)
        base = self.chrom_offset[self._cid[chrom]]
        b0 = base + start // self.binsize
        b1 = base + (end + self.binsize - 1) // self.binsize
        return int(b0), int(b1)

    # ---- fetch (the operation) ----
    def raw(self, region1=None, region2=None, sparse=False):
        """r.raw("chr1:1-2M") -> dense raw matrix. No region -> a slicer: r.raw()[a:b, c:d]."""
        if region1 is None:
            return _Slicer(self, balance=False)
        return self._fetch_region(region1, region2, balance=False, sparse=sparse)

    def balanced(self, region1=None, region2=None, sparse=False):
        if region1 is None:
            return _Slicer(self, balance=True)
        return self._fetch_region(region1, region2, balance=True, sparse=sparse)

    balance = balanced  # alias

    def matrix(self, balance=False, sparse=False):  # cooler-compatible: .fetch()/[...]
        return _MatrixSelector(self, balance, sparse)

    def _fetch_region(self, region1, region2, balance, sparse):
        a0, a1 = self._region_bins(region1)
        b0, b1 = (a0, a1) if region2 is None else self._region_bins(region2)
        return self._fetch_bins(a0, a1, b0, b1, balance, sparse)

    def _read_rows(self, r0, r1):
        """All pixels with bin1 in [r0,r1): (bin1, bin2, count) as arrays."""
        p0, p1 = int(self.bin1_offset[r0]), int(self.bin1_offset[r1])
        if p1 <= p0:
            return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float64)
        return (self._b1[p0:p1].astype(np.int64), self._b2[p0:p1].astype(np.int64),
                self._cn[p0:p1].astype(np.float64))

    def _fetch_bins(self, a0, a1, b0, b1, balance, sparse):
        na, nb = a1 - a0, b1 - b0
        M = np.zeros((na, nb), dtype=np.float64)
        # direct: stored pixels with bin1 in a-range, col j in b-range
        i, j, c = self._read_rows(a0, a1)
        m = (j >= b0) & (j < b1)
        np.add.at(M, (i[m] - a0, j[m] - b0), c[m])
        # transpose (symmetric-upper): stored pixels with bin1 in b-range, col in a-range
        i2, j2, c2 = self._read_rows(b0, b1)
        m2 = (j2 >= a0) & (j2 < a1)
        np.add.at(M, (j2[m2] - a0, i2[m2] - b0), c2[m2])
        # the diagonal (p==q, p in a & b range) is the only cell hit by both passes -> subtract once
        diag = m & (i == j) & (i >= b0) & (i < b1)
        np.add.at(M, (i[diag] - a0, j[diag] - b0), -c[diag])
        if balance:
            w = self.weights()
            if w is None:
                raise ValueError("no weights in this cooler; run balance first")
            M *= np.outer(w[a0:a1], w[b0:b1])
        if sparse:
            from scipy.sparse import coo_matrix
            r, cc = np.nonzero(M)
            return coo_matrix((M[r, cc], (r, cc)), shape=M.shape)
        return M


class _Slicer:
    """r.raw()[a:b, c:d] — bin-index access."""
    def __init__(self, roo, balance):
        self.roo, self.balance = roo, balance

    def __getitem__(self, key):
        s1, s2 = key if isinstance(key, tuple) else (key, key)
        n = self.roo.nbins
        a0, a1 = (s1.start or 0), (s1.stop if s1.stop is not None else n)
        b0, b1 = (s2.start or 0), (s2.stop if s2.stop is not None else n)
        return self.roo._fetch_bins(a0, a1, b0, b1, self.balance, False)


class _MatrixSelector:
    """cooler-compatible: clr.matrix(balance=True).fetch("chr1") and clr.matrix()[i0:i1, j0:j1]."""
    def __init__(self, roo, balance, sparse=False):
        self.roo, self.balance, self.sparse = roo, balance, sparse

    def fetch(self, region1, region2=None):
        return self.roo._fetch_region(region1, region2, self.balance, self.sparse)

    def __getitem__(self, key):
        s1, s2 = key if isinstance(key, tuple) else (key, key)
        n = self.roo.nbins
        a0, a1 = (s1.start or 0), (s1.stop if s1.stop is not None else n)
        b0, b1 = (s2.start or 0), (s2.stop if s2.stop is not None else n)
        return self.roo._fetch_bins(a0, a1, b0, b1, self.balance, self.sparse)


class _Table:
    """cooler-compatible bins()/pixels() selector: [:], [lo:hi], and (bins) .fetch(region)."""
    def __init__(self, roo, kind):
        self.roo, self.kind = roo, kind
        self._n = roo.nbins if kind == "bins" else roo.nnz

    def __len__(self):
        return self._n

    @property
    def columns(self):
        if self.kind == "bins":
            c = ["chrom", "start", "end"]
            return c + (["weight"] if self.roo._weight is not None else [])
        return ["bin1_id", "bin2_id", "count"]

    def _slice(self, lo, hi):
        return self.roo._bins_df(lo, hi) if self.kind == "bins" else self.roo._pixels_df(lo, hi)

    def __getitem__(self, key):
        if isinstance(key, slice):
            lo = key.start or 0
            hi = key.stop if key.stop is not None else self._n
            return self._slice(lo, hi)
        if isinstance(key, str):  # column access: clr.bins()['weight']
            return self._slice(0, self._n)[key]
        raise TypeError(key)

    def fetch(self, region):
        if self.kind != "bins":
            raise NotImplementedError("pixels().fetch not supported; use [lo:hi]")
        b0, b1 = self.roo._region_bins(region)
        return self._slice(b0, b1)
