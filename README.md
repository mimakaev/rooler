# rooler

A fast, out-of-core, opinionated reimplementation of cooler's heavy CLI path — **cload, merge,
zoomify, balance, expected** — plus a thin Python read API. Built for genome-wide micro-C scale
(tens to hundreds of billions of pairs, 64–256 bp resolutions) on commodity RAM.

Rust engine (the ops) + a small Python package (the read/analysis API). Output files are valid
**cooler / mcool** format, so cooler and cooltools work on them natively.

## Why
- **Out-of-core everything.** A `--mem` budget bounds RAM; cload/merge scale to any size.
- **Fast.** Parallel pairs parsing (~95 Mpairs/s), streaming k-way merge, streaming zoomify,
  compressed-scratch parallel-SpMV balance, and an O(nnz) `expected` (seconds, not hours).
- **No mystery coolers.** rooler refuses to write a cooler without a genome assembly.
- **cooler/cooltools compatible.** Files round-trip through `cooler.Cooler`; `cooltools` runs on them.

## Build
```
cargo build --release          # needs: rust, libhdf5-dev, liblz4-dev, pkg-config
# runtime: bgzip (htslib) for cload input
```
Reading blosc-compressed coolers written elsewhere: `export HDF5_PLUGIN_PATH=<hdf5plugin>/plugins`.
rooler writes blosc:zstd:1 by default (statically linked) and gzip on request.

## CLI
```
rooler cload   <pairs.gz> <binsize> <out.cool> [--mem 4] [--threads 8] [--preset blosc:zstd:1] [--assembly hg38]
rooler merge   <out.cool> <in1.cool> <in2.cool> ...      [--mem 4] [--res R] [--assembly hg38]
rooler zoomify <base.cool> <out.mcool> [--resolutions a,b,c] [--assembly hg38]
rooler balance <cool[::resolutions/R]> [--ignore-diags 2] [--mad-max 5] [--min-nnz 10] [--tol 1e-4] [--threads 8]
rooler expected <cool[::resolutions/R]> [--view chroms|arms]
```
- **cload**: bgzip-parallel decode → parse/bin → sort `--mem`-chunks → k-way merge → write. Auto-detects
  and stamps the genome assembly (or `--assembly`); refuses if it can't determine one.
- **merge**: streaming k-way drain-and-count merge over the (pre-sorted) inputs. No sort, no spill.
- **zoomify**: streaming integer-factor coarsen (respects chrom boundaries); cascades level to level.
- **balance**: genome-wide IC over a compressed CSR scratch (built once, parallel SpMV per iteration).
  Scale-free stop (`CV = std/mean < tol`). Writes cooler-compatible `bins/weight`.
- **expected**: cis distance-decay P(s) per region (arms/chroms view). One O(nnz) pass for
  `sum_balanced`, FFT autocorrelation for `n_valid`. Stored in-cooler under `expected/{view}/weight`.

## Python read API (`python/rooler`)
```python
import rooler
r = rooler.open("f.mcool", 1000)      # or "f.mcool::resolutions/1000", or "f.cool"
r.raw("chr1:10_000_000-20_000_000")   # dense raw matrix (symmetric)
r.balanced("chr1", "chr2")            # balanced (w_i*w_j), trans
r.raw()[a:b, c:d]                     # bin-index slicing
r.pixels()[lo:hi]; r.bins()[:]; r.chroms(); r.extent("chr1"); r.info
r.matrix(balance=True).fetch("chr1")  # cooler-compatible shim
```
For cooltools, use `cooler.Cooler(rooler_file)` directly — the files are cooler-format.

## Memory
`--mem` is the RAM knob; peak RSS ≈ `--mem` + O(nbins) overhead for cload/merge. balance sizes a
compressed scratch (~2 B/pixel) to the data (no `--mem`). See `MEMORY_CALIBRATION.md`.

## Tests
```
cargo test --release          # ~0.6s, no python / network / fixture files needed
```
- **Unit tests** cover the codecs (bin2-delta shuffle+LZ4, u8+exception counts, spill runs),
  the k-way drain-and-count merge, pairs line parsing, the coarsening bin map, genome views,
  and the cooler writer (round-trip, index validity, append ordering, preset parsing).
- **`tests/pipeline.rs`** runs the whole chain — synthetic pairs → cload → merge → zoomify →
  balance → expected — against independent in-test oracles: a HashMap pixel table for
  cload/merge/zoomify, a marginal-flatness (CV) property check plus row-vs-tiled agreement for
  balance, and a brute-force O(n²) recomputation for expected. It also asserts cload is
  invariant to `--mem`/`--threads`, and that the ops refuse mystery coolers, mismatched merges
  and unbalanced `expected`.

*Embedding note:* the ops open the file read-write, so drop every HDF5 handle — `Group` and
`Dataset` too, not just `File` — before calling the next op on the same file in-process.

External cross-check against python cooler (manual, needs h5py/cooler):
```
scripts/validate_vs_cooler.py a.cool b.cool [--grp resolutions/1000] [--region chr1:0-5,000,000]
scripts/validate_vs_cooler.py a.cool --self-check
```

## Validation (vs cooler / cooltools, on real data)
- cload/merge/zoomify: pixel-exact vs numpy reference / cooler.coarsen_cooler.
- balance: weights median rel diff **6e-6** vs `cooler.balance_cooler` (0 mask disagreements).
- expected: `balanced.avg` median rel diff **6e-16** vs `cooltools.expected_cis` (machine precision).
- Full chain cload→zoomify→balance runs end-to-end; the balanced mcool works in cooler + cooltools.

## Status
Working: cload, merge, zoomify, balance, expected, read API, assembly enforcement.
See `STATUS.md` for the current state + review findings, `PLAN.md` for the roadmap,
`PROGRESS.md` for the historical build log.
