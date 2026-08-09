"""rooler — thin Python read API over rooler/cooler-format .cool/.mcool files.

    r = rooler.open("f.mcool", 1000)      # or "f.mcool::resolutions/1000", or "f.cool"
    r.raw("chr1")                         # dense raw counts (symmetric)
    r.balanced("chr1", "chr2")            # balanced (w_i * w_j), trans
    r.ooe("chr1:0-2,000,000")             # observed / expected, cis
    r.expected()                          # P(s) table, smoothed by default
    r.pixels(), r.bins(), r.chroms(), r.info   # pixels/bins are polars; chroms is pandas
    r.matrix(balance=True).fetch("chr1")  # cooler-compatible shim

Keep the handle open and reuse it.
    Opening a Rooler reads and caches the things every fetch needs — chromosome names and
    lengths, the chrom offsets and the full `bin1_offset` index — and it lazily caches the
    balancing weights and the expected table on first use. Those caches live on the object,
    so re-opening the file per fetch re-reads all of it and throws the caches away. Open once,
    hold it, fetch many times.

    The handle is read-only, but it is still a resource worth scoping: it holds an OS file
    descriptor, and while it is alive HDF5 will not let anything in the same process reopen
    that file for writing (so a rooler op run in-process would fail). `Rooler` is therefore a
    context manager, and has `.close()`:

        with rooler.open("f.mcool", 1000) as r:
            m = r.ooe("chr1")

    Prefer a long-lived handle for a whole analysis; use `with` when you are about to write
    to the same file, or in a long-running process that opens many coolers.
"""
import numpy as np
import pandas as pd
import h5py
try:
    import hdf5plugin  # noqa: F401  (registers blosc/zstd/lz4 filters for reading)
except Exception:
    pass
try:
    import polars as pl
except ImportError:  # pixels()/bins() fall back to pandas
    pl = None

__version__ = "0.1.0a1"
__all__ = ["open", "Rooler", "__version__"]


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
        self._wcache = None  # weights read from disk once, on first balanced fetch
        self._ecache = {}    # (view, column) -> (bin->region map, per-region expected arrays)
        self._uri = uri

    # ---- lifetime ----
    # Read-only, but still worth closing: it holds a file descriptor, and while it is open
    # HDF5 refuses to reopen the same file for writing in this process.
    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        state = "closed" if self._f is None else f"{self.nbins} bins, {self.nnz} pixels"
        return f"<Rooler {self._uri!r} binsize={self.binsize} {state}>"

    # ---- metadata (cooler-compatible) ----
    @property
    def info(self):
        return {"nbins": self.nbins, "nnz": self.nnz, "bin-size": self.binsize,
                "nchroms": len(self.chromnames), "balanced": self._weight is not None}

    @property
    def shape(self):
        return (self.nbins, self.nbins)

    def chroms(self):
        """cooler-compatible chromosome table: `r.chroms()[:]`, `[lo:hi]`, `.fetch(name)`.
        Returns pandas with the same columns and dtypes cooler gives (name, length)."""
        return _ChromTable(self)

    def extent(self, region):
        """(bin0, bin1) for a region — cooltools uses this heavily."""
        return self._region_bins(region)

    def offset(self, region):
        return self._region_bins(region)[0]

    def bins(self, frame="polars"):
        """Bin table. `frame="pandas"` gives cooler's exact layout (chrom as a categorical)."""
        return _Table(self, "bins", frame)

    def pixels(self, join=False, frame="polars"):
        """Pixel table, as **polars** by default — it is the bulk table, and polars reads it
        without pandas' per-column copies. `frame="pandas"` returns cooler's exact layout, and
        `cooler.Cooler(path).pixels()` works on rooler files unchanged if you want cooler's
        selector itself. `join=True` expands bin ids to chrom1/start1/end1/chrom2/start2/end2.
        """
        return _Table(self, "pixels", frame, join)

    def weights(self):
        if self._weight is None:
            return None
        if self._wcache is None:
            self._wcache = self._weight[:]
        return self._wcache

    # ---- expected: cis contact-vs-distance, as stored by `rooler expected` ----
    def expected_views(self):
        """Names of the stored expected views (e.g. ['arms', 'chroms'])."""
        return sorted(self._g["expected"].keys()) if "expected" in self._g else []

    def expected(self, view=None, column=None):
        """Stored cis expected P(s) as a DataFrame, in cooltools' layout.

        Columns: region1/region2, dist, dist_bp, n_total, n_valid, count.sum, count.avg,
        balanced.sum, balanced.avg, balanced.avg.smoothed, balanced.avg.smoothed.agg, plus
        `contact_frequency` — the value you normally want.

        `contact_frequency` defaults to the **log-smoothed, genome-wide aggregated** curve
        (`balanced.avg.smoothed.agg`), matching `cooltools.expected_cis`'s default. Raw
        `balanced.avg` is noisy at large separations, where few pixel pairs contribute; pass
        `column="balanced.avg"` to opt out, or any other stored column name.

        `view` selects among stored views; with one view stored it is picked automatically.
        """
        if "expected" not in self._g:
            raise ValueError("no expected stored; run `rooler expected` on this cooler")
        avail = self.expected_views()
        if view is None:
            if len(avail) != 1:
                raise ValueError(
                    f"several expected views stored ({', '.join(avail)}); pass view=")
            view = avail[0]
        if view not in avail:
            raise ValueError(f"no expected view {view!r} (have: {', '.join(avail) or 'none'})")

        ge = self._g[f"expected/{view}/weight"]
        df = pd.DataFrame({k: ge[k][:] for k in ge.keys() if k != "region_id"})
        names = [n.decode() if isinstance(n, bytes) else n
                 for n in self._g[f"views/{view}/name"][:]]
        reg = np.asarray([names[i] for i in ge["region_id"][:]])
        default = ge.attrs.get("default_column", "balanced.avg.smoothed.agg")
        if isinstance(default, bytes):
            default = default.decode()
        col = column or default
        if col not in df.columns:
            raise ValueError(f"no column {col!r} (have: {', '.join(df.columns)})")
        df["region1"] = reg
        df["region2"] = reg                   # cis only; mirrors cooltools' layout
        df["dist_bp"] = df["dist"] * self.binsize
        df["contact_frequency"] = df[col].values
        order = ["region1", "region2", "dist", "dist_bp", "contact_frequency", "n_total",
                 "n_valid", "count.sum", "balanced.sum", "count.avg", "balanced.avg",
                 "balanced.avg.smoothed", "balanced.avg.smoothed.agg"]
        cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
        return (df[cols].sort_values(["region1", "dist"], kind="stable")
                .reset_index(drop=True))

    # Column dicts, so bins()/pixels() can hand the same arrays to polars or pandas without
    # an extra copy. chrom is emitted as plain strings; the pandas path re-categoricalises it
    # to match cooler, and polars stores strings natively.
    def _chrom_names_for(self, codes):
        names = np.asarray(self.chromnames, dtype=object)
        return names[codes]

    def _bins_cols(self, lo, hi):
        d = {"chrom": self._chrom_names_for(self._g["bins/chrom"][lo:hi]),
             "start": self._g["bins/start"][lo:hi], "end": self._g["bins/end"][lo:hi]}
        if self._weight is not None:
            d["weight"] = self._weight[lo:hi]
        return d

    def _pixels_cols(self, lo, hi, join=False):
        b1, b2 = self._b1[lo:hi], self._b2[lo:hi]
        cnt = self._cn[lo:hi]
        if not join:
            return {"bin1_id": b1, "bin2_id": b2, "count": cnt}
        # expand bin ids to coordinates, the way cooler's pixels(join=True) does
        cid = self._g["bins/chrom"]
        st, en = self._g["bins/start"], self._g["bins/end"]
        def coords(b):
            b = np.asarray(b)
            order = np.argsort(b, kind="stable")          # h5py needs increasing fancy indices
            uniq, inv = np.unique(b[order], return_inverse=True)
            c = self._chrom_names_for(cid[uniq])[inv]
            s, e = st[uniq][inv], en[uniq][inv]
            back = np.empty_like(order)
            back[order] = np.arange(len(order))
            return c[back], s[back], e[back]
        c1, s1, e1 = coords(b1)
        c2, s2, e2 = coords(b2)
        return {"chrom1": c1, "start1": s1, "end1": e1,
                "chrom2": c2, "start2": s2, "end2": e2, "count": cnt}

    # ---- region -> bin range ----
    def _region_span(self, chrom, start, end):
        """Bin range of a *view region*, assigning each bin to the region holding most of it
        (midpoint rule). Must match src/expected.rs, or ooe would divide by the wrong curve."""
        ci = self._cid[chrom]
        base = int(self.chrom_offset[ci])
        cbins = int(self.chrom_offset[ci + 1]) - base
        mid = lambda x: min(max(int(np.ceil(x / self.binsize - 0.5)), 0), cbins)
        return base + mid(start), base + mid(end)

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

    # ---- observed / expected ----
    def _expected_lookup(self, view, column):
        """(bin -> view-region id, {region id: expected-by-distance}, region names).
        Cached on the handle: built once per (view, column), reused by every ooe() call."""
        key = (view, column)
        if key in self._ecache:
            return self._ecache[key]
        df = self.expected(view=view, column=column)
        if view is None:
            view = self.expected_views()[0]
        gv = self._g[f"views/{view}"]
        chroms = [c.decode() if isinstance(c, bytes) else c for c in gv["chrom"][:]]
        starts, ends = gv["start"][:], gv["end"][:]
        names = [n.decode() if isinstance(n, bytes) else n for n in gv["name"][:]]
        region_of = np.full(self.nbins, -1, dtype=np.int32)
        for rid, (c, s, e) in enumerate(zip(chroms, starts, ends)):
            b0, b1 = self._region_span(c, s, e)
            region_of[b0:b1] = rid
        by_name = {n: rid for rid, n in enumerate(names)}
        tables = {}
        for name, sub in df.groupby("region1", sort=False):
            tables[by_name[name]] = sub.sort_values("dist")["contact_frequency"].to_numpy()
        self._ecache[key] = (region_of, tables, names)
        return self._ecache[key]

    def _view_region_bins(self, region, names, view):
        """Bin range for a region string. A stored view-region name (e.g. 'chr1_p') resolves to
        that region — the natural unit for ooe, and what the boundary error tells you to use —
        otherwise it is parsed as a chromosome or 'chrom:start-end'."""
        if isinstance(region, str) and region in names:
            gv = self._g[f"views/{view}"]
            rid = names.index(region)
            c = gv["chrom"][rid]
            c = c.decode() if isinstance(c, bytes) else c
            return self._region_span(c, int(gv["start"][rid]), int(gv["end"][rid]))
        return self._region_bins(region)

    def _one_region(self, rng, region_of, names, view, what):
        """The bin range must lie inside exactly one view region; otherwise explain why not."""
        lo, hi = rng
        r = region_of[lo:hi]
        if len(r) == 0:
            raise ValueError(f"{what} is empty")
        if (r < 0).any():
            raise ValueError(
                f"{what} includes bins outside view {view!r} — expected is only defined "
                f"inside its regions")
        first = int(r[0])
        if not (r == first).all():
            spanned = [names[i] for i in dict.fromkeys(r.tolist())]
            raise ValueError(
                f"{what} spans {len(spanned)} regions of view {view!r} ({', '.join(spanned)}). "
                f"Expected is defined per region, so a fetch crossing that boundary has no "
                f"single P(s) to divide by — fetch each region separately, or store expected "
                f"with a coarser view (e.g. `rooler expected --view chroms`).")
        return first

    def ooe(self, region1=None, region2=None, view=None, column=None):
        """Observed-over-expected: balanced counts divided by the stored cis expected.

        Each cell (i, j) is divided by the expected value for its genomic separation |i-j|.
        Both sides must lie inside a single region of the expected view; a fetch that crosses
        an arm or chromosome boundary — or a trans fetch — **raises**, because there is no one
        P(s) curve that applies to it. Fetch the regions separately, or store expected with a
        coarser view.

        Uses the same default column as `expected()`: the log-smoothed, genome-wide aggregated
        curve (`balanced.avg.smoothed.agg`). Pass `column="balanced.avg.smoothed"` for the
        per-region curve, or `"balanced.avg"` for the unsmoothed one. Note that OOE only
        averages to ~1 over the *same* extent the expected was computed for — a sub-region of
        an arm is generally off by whatever makes it differ from its arm, which is usually the
        signal you are looking for.

        Requires `rooler expected` to have been run (balance does it by default).
        """
        if region1 is None:
            raise ValueError("ooe() needs a region, e.g. r.ooe('chr1:0-5,000,000')")
        vname = view if view is not None else (self.expected_views() or [None])[0]
        region_of, tables, names = self._expected_lookup(view, column)
        a0, a1 = self._view_region_bins(region1, names, vname)
        b0, b1 = ((a0, a1) if region2 is None
                  else self._view_region_bins(region2, names, vname))
        ra = self._one_region((a0, a1), region_of, names, vname, f"region1 {region1!r}")
        rb = ra if region2 is None else self._one_region(
            (b0, b1), region_of, names, vname, f"region2 {region2!r}")
        if ra != rb:
            raise ValueError(
                f"observed/expected needs both sides in the same region of view {vname!r}; "
                f"got {names[ra]!r} and {names[rb]!r}. There is no cis expected across "
                f"regions (trans expected is a different quantity rooler does not store yet).")
        curve = tables.get(ra)
        if curve is None:
            raise ValueError(f"no expected curve stored for region {names[ra]!r}")

        obs = self._fetch_bins(a0, a1, b0, b1, balance=True, sparse=False)
        # The expected matrix is Toeplitz (exp[i,j] = curve[|i-j|]), so build it as a zero-copy
        # strided view over a 1-D array of length na+nb-1 rather than materializing na*nb
        # values. ooe() then costs one divide pass over the observed matrix.
        na, nb = a1 - a0, b1 - b0
        d = np.abs(np.arange(a0 - b1 + 1, a1 - b0))
        ext = np.full(d.shape, np.nan)
        ok = d < len(curve)
        ext[ok] = curve[d[ok]]
        exp = np.lib.stride_tricks.as_strided(
            ext[nb - 1:], shape=(na, nb),
            strides=(ext.strides[0], -ext.strides[0]), writeable=False)
        # obs is freshly allocated by the fetch, so divide in place: saves an allocation and
        # a full pass over the matrix (~a third of ooe's cost on large fetches)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(obs, exp, out=obs)
        return obs


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
        # transpose (symmetric-upper): stored pixels with bin1 in b-range, col in a-range.
        # A cis square fetch (the common case) has identical ranges — reuse the rows already
        # read instead of hitting HDF5 a second time.
        if (b0, b1) == (a0, a1):
            i2, j2, c2 = i, j, c
        else:
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


class _ChromTable:
    """cooler-compatible chromosome table. Small metadata, so it stays pandas and matches
    cooler's columns and dtypes exactly (name: str, length: int32)."""
    def __init__(self, roo):
        self.roo = roo

    def __len__(self):
        return len(self.roo.chromnames)

    @property
    def columns(self):
        return ["name", "length"]

    @property
    def shape(self):
        return (len(self), 2)

    def keys(self):
        return list(self.roo.chromnames)

    def _slice(self, lo, hi):
        return pd.DataFrame(
            {"name": pd.array(self.roo.chromnames[lo:hi], dtype="string"),
             "length": self.roo._clen[lo:hi].astype(np.int32)})

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._slice(key.start or 0,
                               key.stop if key.stop is not None else len(self))
        if isinstance(key, str):
            return self._slice(0, len(self))[key]
        if isinstance(key, (int, np.integer)):
            return self._slice(key, key + 1)
        raise TypeError(key)

    def fetch(self, name):
        """The row for one chromosome. (cooler's chroms().fetch raises NotImplementedError;
        this is an addition, not a compatibility requirement.)"""
        if name not in self.roo._cid:
            raise ValueError(f"no chromosome {name!r}")
        i = self.roo._cid[name]
        return self._slice(i, i + 1)

    def __repr__(self):
        return f"<chroms {len(self)} rows>"


class _Table:
    """bins()/pixels() selector: `[:]`, `[lo:hi]`, `['col']`, and `.fetch(region)`.

    Returns polars by default — these are the bulk tables, and a billion-pixel slice should
    not pay pandas' per-column conversions. Pass frame="pandas" for cooler's exact layout.
    """
    def __init__(self, roo, kind, frame="polars", join=False):
        if frame not in ("polars", "pandas"):
            raise ValueError(f"frame must be 'polars' or 'pandas', got {frame!r}")
        if frame == "polars" and pl is None:
            frame = "pandas"
        self.roo, self.kind, self.frame, self.join = roo, kind, frame, join
        self._n = roo.nbins if kind == "bins" else roo.nnz

    def __len__(self):
        return self._n

    @property
    def shape(self):
        return (self._n, len(self.columns))

    @property
    def columns(self):
        if self.kind == "bins":
            c = ["chrom", "start", "end"]
            return c + (["weight"] if self.roo._weight is not None else [])
        if self.join:
            return ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "count"]
        return ["bin1_id", "bin2_id", "count"]

    def _slice(self, lo, hi):
        lo, hi = max(int(lo), 0), min(int(hi), self._n)
        if hi < lo:
            hi = lo
        d = (self.roo._bins_cols(lo, hi) if self.kind == "bins"
             else self.roo._pixels_cols(lo, hi, self.join))
        if self.frame == "polars":
            return pl.DataFrame(d)
        df = pd.DataFrame(d)
        # cooler stores chrom as an *ordered* categorical over the chromosome order
        if self.kind == "bins":
            df["chrom"] = pd.Categorical(df["chrom"], categories=self.roo.chromnames, ordered=True)
        elif self.join:
            for c in ("chrom1", "chrom2"):
                df[c] = pd.Categorical(df[c], categories=self.roo.chromnames, ordered=True)
        return df

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._slice(key.start or 0,
                               key.stop if key.stop is not None else self._n)
        if isinstance(key, str):          # column access: r.bins()['weight']
            return self._slice(0, self._n)[key]
        if isinstance(key, (int, np.integer)):
            return self._slice(key, key + 1)
        raise TypeError(key)

    def fetch(self, region, region2=None):
        """Rows for a region. For pixels this is every stored pixel whose *bin1* lies in the
        region (the row-block the bin1_offset index addresses), optionally further restricted
        to pixels whose bin2 lies in `region2`."""
        b0, b1 = self.roo._region_bins(region)
        if self.kind == "bins":
            return self._slice(b0, b1)
        p0, p1 = int(self.roo.bin1_offset[b0]), int(self.roo.bin1_offset[b1])
        out = self._slice(p0, p1)
        if region2 is None:
            return out
        c0, c1 = self.roo._region_bins(region2)
        if self.join:
            raise NotImplementedError("fetch(region, region2) needs join=False")
        if self.frame == "polars":
            return out.filter((pl.col("bin2_id") >= c0) & (pl.col("bin2_id") < c1))
        m = (out["bin2_id"] >= c0) & (out["bin2_id"] < c1)
        return out[m].reset_index(drop=True)

    def __repr__(self):
        return f"<{self.kind} {self._n} rows, {self.frame}>"
