"""Accessor tests: rooler's tables must equal cooler's, column for column.

Run with:  pytest python/tests -q
Needs `cooler` and a rooler binary on PATH (or ROOLER=/path/to/rooler) to build the fixture.
"""
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

cooler = pytest.importorskip("cooler")
import rooler  # noqa: E402

ROOLER = os.environ.get("ROOLER", shutil.which("rooler"))
GENPAIRS = os.environ.get("GENPAIRS")


@pytest.fixture(scope="module")
def cool(tmp_path_factory):
    """A small cooler written by the rooler binary, balanced (so weights + expected exist)."""
    if not ROOLER:
        pytest.skip("no rooler binary; set ROOLER=/path/to/rooler")
    gen = GENPAIRS or os.path.join(os.path.dirname(ROOLER), "examples", "genpairs")
    if not os.path.exists(gen):
        pytest.skip("no genpairs example binary; set GENPAIRS=")
    d = tmp_path_factory.mktemp("rooler")
    path = str(d / "t.cool")
    pairs = subprocess.Popen([gen, "2000000", "7", "4"], stdout=subprocess.PIPE)
    subprocess.run([ROOLER, "cload", "-", "500000", path, "--nproc", "4"],
                   stdin=pairs.stdout, check=True, capture_output=True)
    pairs.wait()
    subprocess.run([ROOLER, "balance", path, "--nproc", "4"], check=True, capture_output=True)
    return path


def test_pixels_match_cooler(cool):
    c, r = cooler.Cooler(cool), rooler.open(cool)
    assert c.pixels()[:5000].equals(r.pixels(frame="pandas")[:5000])
    assert list(r.pixels().columns) == list(c.pixels().columns)
    assert len(r.pixels()) == c.info["nnz"]
    r.close()


def test_info_matches_cooler(cool):
    """r.info is the file's attributes, i.e. cooler's own info dict — and a rooler-written
    file must carry every key cooler's writer produces (`sum` and `nnz` included)."""
    c, r = cooler.Cooler(cool), rooler.open(cool)
    assert r.info == c.info
    for key in ("bin-size", "bin-type", "creation-date", "format", "format-url",
                "format-version", "generated-by", "genome-assembly", "metadata", "nbins",
                "nchroms", "nnz", "storage-mode", "sum"):
        assert key in r.info, key
    assert r.info["sum"] == 2_000_000          # one count per generated pair
    assert r.info["sum"] == int(c.pixels()["count"][:].sum())
    assert r.info["nnz"] == len(c.pixels())
    # balance flags are real booleans, as cooler writes them ("False" as a string is truthy)
    import h5py
    with h5py.File(cool, "r") as f:
        attrs = dict(f["bins/weight"].attrs)
    assert attrs["converged"] is np.True_ and attrs["cis_only"] is np.False_
    assert not attrs["divisive_weights"]
    r.close()


def test_matrix_matches_cooler(cool):
    """Dense fetches (cis square, trans, overlapping rectangles, bin slices) equal cooler's."""
    c, r = cooler.Cooler(cool), rooler.open(cool)
    for bal in (False, True):
        cm, rm = c.matrix(balance=bal), r.matrix(balance=bal)
        for args in (("chr2",), ("chr1", "chr3"), ("chr3", "chr1"),
                     ("chr1:10,000,000-60,000,000", "chr1:40,000,000-90,000,000")):
            assert np.allclose(cm.fetch(*args), rm.fetch(*args), equal_nan=True), (bal, args)
        assert np.allclose(cm[10:60, 30:90], rm[10:60, 30:90], equal_nan=True)
    assert np.array_equal(r.raw()[5:40, 5:40], c.matrix(balance=False)[5:40, 5:40])
    r.close()


def test_pixels_join_matches_cooler(cool):
    c, r = cooler.Cooler(cool), rooler.open(cool)
    want = c.pixels(join=True)[:2000].reset_index(drop=True)
    assert want.equals(r.pixels(join=True, frame="pandas")[:2000])
    r.close()


def test_bins_match_cooler(cool):
    c, r = cooler.Cooler(cool), rooler.open(cool)
    assert c.bins()[:3000].equals(r.bins(frame="pandas")[:3000])
    # cooler stores chrom as an *ordered* categorical; that must survive
    got = r.bins(frame="pandas")[:10]["chrom"]
    assert isinstance(got.dtype, pd.CategoricalDtype) and got.cat.ordered
    r.close()


def test_chroms_match_cooler(cool):
    c, r = cooler.Cooler(cool), rooler.open(cool)
    a, b = c.chroms()[:], r.chroms()[:]
    assert list(a.columns) == list(b.columns)
    assert (a["name"].to_numpy() == b["name"].to_numpy()).all()
    assert (a["length"].to_numpy() == b["length"].to_numpy()).all()
    assert b["length"].dtype == np.int32
    assert r.chroms().fetch(a["name"].iloc[1])["length"].iloc[0] == a["length"].iloc[1]
    r.close()


def test_polars_is_the_default_and_agrees_with_pandas(cool):
    pl = pytest.importorskip("polars")
    r = rooler.open(cool)
    p, q = r.pixels()[:1000], r.pixels(frame="pandas")[:1000]
    assert isinstance(p, pl.DataFrame)
    assert isinstance(r.bins()[:10], pl.DataFrame)
    for col in ("bin1_id", "bin2_id", "count"):
        assert (p[col].to_numpy() == q[col].to_numpy()).all()
    r.close()


def test_slicing_and_fetch(cool):
    r = rooler.open(cool)
    assert r.pixels()[10:20].height == 10
    assert r.pixels()[:5].height == 5
    b0, b1 = r.extent("chr2")
    assert r.bins().fetch("chr2").height == b1 - b0
    # pixels().fetch gives the bin1 row-block, and every row must start in the region
    px = r.pixels().fetch("chr2")
    assert px["bin1_id"].min() >= b0 and px["bin1_id"].max() < b1
    # restricting bin2 as well keeps it inside the region
    cis = r.pixels().fetch("chr2", "chr2")
    assert cis["bin2_id"].min() >= b0 and cis["bin2_id"].max() < b1
    r.close()


def test_context_manager_closes(cool):
    with rooler.open(cool) as r:
        assert len(r.chroms()) > 0
    with pytest.raises(Exception):
        r.pixels()[:1]


def test_ooe_and_expected(cool):
    r = rooler.open(cool)
    e = r.expected()
    for col in ("region1", "dist", "contact_frequency", "balanced.avg",
                "balanced.avg.smoothed", "balanced.avg.smoothed.agg"):
        assert col in e.columns
    name = e["region1"].iloc[0]
    m = r.ooe(name)
    assert m.shape[0] == m.shape[1] and np.isfinite(m).any()
    # crossing a view-region boundary must raise, not return NaN
    with pytest.raises(ValueError):
        r.ooe("chr1", "chr2")
    r.close()
