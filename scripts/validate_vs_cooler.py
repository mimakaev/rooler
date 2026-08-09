#!/usr/bin/env python3
"""Manual cross-check of a rooler cooler against a reference (or against python cooler).

The cargo test suite validates rooler against in-test oracles and needs no python. This
script is the *external* check: it compares two cooler files pixel-for-pixel and, if the
`cooler` package is importable, verifies the file also reads correctly through cooler
itself (which is the real compatibility claim).

    scripts/validate_vs_cooler.py a.cool b.cool              # compare two files
    scripts/validate_vs_cooler.py a.cool b.cool --grp resolutions/1000
    scripts/validate_vs_cooler.py a.cool --self-check        # cooler-readability only
    scripts/validate_vs_cooler.py a.cool b.cool --region chr1:0-5,000,000

Exit code 0 = all checks passed, 1 = a mismatch was found.
"""
import argparse
import sys

import numpy as np

try:
    import hdf5plugin  # noqa: F401  registers blosc/zstd/lz4 so blosc coolers are readable
except ImportError:
    pass
import h5py

FAIL = []


def check(ok, msg):
    print(("  OK   " if ok else "  FAIL ") + msg)
    if not ok:
        FAIL.append(msg)
    return ok


def grp(f, path):
    return f if path in ("/", "") else f[path]


def compare_pixels(a_path, b_path, path):
    """Pixel tables, bins and indexes must be identical."""
    with h5py.File(a_path, "r") as fa, h5py.File(b_path, "r") as fb:
        ga, gb = grp(fa, path), grp(fb, path)
        na, nb = ga["pixels/count"].shape[0], gb["pixels/count"].shape[0]
        if not check(na == nb, f"nnz equal ({na} vs {nb})"):
            return
        for col, ds in [("bin1_id", "pixels/bin1_id"), ("bin2_id", "pixels/bin2_id"),
                        ("count", "pixels/count")]:
            # stream in chunks so this works on multi-billion-pixel coolers
            same, step, total_a = True, 1 << 24, 0
            for lo in range(0, na, step):
                hi = min(lo + step, na)
                xa, xb = ga[ds][lo:hi], gb[ds][lo:hi]
                if not np.array_equal(xa, xb):
                    same = False
                    bad = int(np.flatnonzero(xa != xb)[0]) + lo
                    check(False, f"pixels/{col} differs at index {bad}: "
                                 f"{ga[ds][bad]} vs {gb[ds][bad]}")
                    break
                if col == "count":
                    total_a += int(xa.sum())
            if same:
                check(True, f"pixels/{col} identical ({na} values)"
                            + (f", total counts {total_a}" if col == "count" else ""))
        for ds in ["bins/start", "bins/end", "bins/chrom", "indexes/bin1_offset",
                   "indexes/chrom_offset"]:
            check(np.array_equal(ga[ds][:], gb[ds][:]), f"{ds} identical")
        ca = [x.decode() if isinstance(x, bytes) else x for x in ga["chroms/name"][:]]
        cb = [x.decode() if isinstance(x, bytes) else x for x in gb["chroms/name"][:]]
        check(ca == cb, "chroms/name identical")
        check(np.array_equal(ga["chroms/length"][:], gb["chroms/length"][:]), "chroms/length identical")
        for attr in ["bin-size", "nbins", "nnz", "storage-mode", "format-version"]:
            va, vb = ga.attrs.get(attr), gb.attrs.get(attr)
            check(va == vb, f"attr {attr} equal ({va!r} vs {vb!r})")
        if "weight" in ga["bins"] and "weight" in gb["bins"]:
            wa, wb = ga["bins/weight"][:], gb["bins/weight"][:]
            check(np.array_equal(np.isnan(wa), np.isnan(wb)), "weight mask identical")
            m = ~np.isnan(wa)
            if m.any():
                rel = np.abs(wa[m] - wb[m]) / np.maximum(np.abs(wb[m]), 1e-300)
                check(rel.max() < 1e-9, f"weights agree (max rel {rel.max():.3e})")


def cooler_selfcheck(path, cooler_uri, region):
    """The file must be readable by python cooler — that is the compatibility claim."""
    try:
        import cooler
    except ImportError:
        print("  SKIP cooler not importable — pixel comparison only")
        return
    clr = cooler.Cooler(cooler_uri)
    check(clr.info["nnz"] > 0, f"cooler opens it (nnz={clr.info['nnz']}, binsize={clr.binsize})")
    check(len(clr.chromnames) > 0, f"cooler sees {len(clr.chromnames)} chromosomes")
    reg = region or (clr.chromnames[0] + ":0-5,000,000")
    try:
        m = clr.matrix(balance=False).fetch(reg)
        check(np.isfinite(m).all(), f"cooler fetch {reg} -> {m.shape}, all finite")
        # the fetched block must be symmetric (storage-mode symmetric-upper expanded correctly)
        check(np.allclose(m, m.T), f"fetched {reg} is symmetric")
        if "weight" in clr.bins().columns:
            mb = clr.matrix(balance=True).fetch(reg)
            check(np.isnan(mb).sum() < mb.size, f"balanced fetch {reg} has finite values")
    except Exception as e:  # noqa: BLE001
        check(False, f"cooler fetch {reg} raised {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b", nargs="?", help="second cooler to compare against")
    ap.add_argument("--grp", default="/", help="group inside the file, e.g. resolutions/1000")
    ap.add_argument("--region", help="region for the cooler fetch check")
    ap.add_argument("--self-check", action="store_true", help="only run the cooler-readability check")
    args = ap.parse_args()

    uri = args.a if args.grp in ("/", "") else f"{args.a}::{args.grp}"
    if args.b and not args.self_check:
        print(f"comparing {args.a} vs {args.b} (group {args.grp})")
        compare_pixels(args.a, args.b, args.grp)
    print(f"checking {uri} through python cooler")
    cooler_selfcheck(args.a, uri, args.region)

    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)} check(s)")
        for m in FAIL:
            print("  - " + m)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
