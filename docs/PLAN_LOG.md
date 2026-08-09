# PLAN_LOG — measurements while executing PLAN.md

## Phase 0 — baseline (2026-08-09)
- `cargo build --release` clean (1 dead-code warning: `chrom_enum`); `cargo test --release` 3/3 pass.
- Baseline balance on `/workspace/scratch_bench/plan_base.cool` (copy of enc514_256.cool,
  2,563,532,430 pix / 12,537,161 bins), `--block 65536 --threads 8`:
  - scratch(tiled) 6.4GB (2.51 B/pix) built in **28s**
  - mask **11239212/12537161** bins at 32s
  - `converged=true cv=7.52e-5 scale=461.0` total **50s** (7 IC iters)
  - matches the pre-plan measurement (27s build / 48s total) within noise.
- Disk: /workspace 1.2T free.
- Note: `/usr/bin/time` is not installed in this container; used rooler's own timing lines.

## Phase 1 (2026-08-09)
- Build warning-free; 26 unit tests pass.
- Regression checks: balance on plan_base.cool unchanged (`converged=true cv=7.52e-5 scale=461.0`,
  50s); cload of the 5M-pair subset @100kb -> **248,914 pixels**, matching the historical
  reference exactly.
- New guard rails verified by hand: `--preset gizp4` / `gzip99` / `blosc:snappy:1` all error;
  merging coolers with different chrom names errors naming the file and the first difference;
  a pairs line with an unknown chromosome errors instead of panicking.

## Phase 2 (2026-08-09)
- Test suite 3 -> **29 tests, 0.6s** (26 unit + 3 integration). No python/network/fixtures.
- Found and fixed a *test* bug worth remembering: dropping only the `File` handle is not enough
  to release an HDF5 file — a live `Group`/`Dataset` keeps it open, so the next op's read-write
  open fails with a confusing `H5Fcreate: file exists`. All in-process handle use must be scoped.
- `scripts/validate_vs_cooler.py` cross-check: cload with `blosc:zstd:1` vs `gzip4` produce
  byte-identical pixel tables, and the files read through python `cooler` (fetch + symmetry).

## Phase 3 — parallel gzip writer (2026-08-09)
FFI smoke test passed first try: `H5Dwrite_chunk` (hdf5-metno-sys 0.10.1) + libdeflater zlib,
chunks compressed on rayon threads, all HDF5 calls on the caller's thread. Verified the written
chunks round-trip through the ordinary HDF5 API and that the dataset advertises the same
SHUFFLE+DEFLATE pipeline as HDF5's own write path (so stock cooler/h5py read it, no plugins).

Benchmark — single-input merge (a pure 1.1B-pixel re-write), `e2e/base.cool`, 8 threads:

| preset | wall | user | output size |
|---|---|---|---|
| gzip4 **before** (HDF5 serial deflate) | **100s** | 1m56 | 820.9 MB |
| gzip4 **after** (parallel direct-chunk) | **26s** | 2m24 | **786.1 MB** |
| blosc:zstd:1 (default, unchanged path) | 26s | 0m40 | 1227.5 MB |
| none (read+merge floor) | 24s | 0m29 | 22378 MB |

- **3.8x faster** on the gzip path, and the file is **4.2% smaller** (256K-element chunks
  compress better than the 1M chunk the serial path used — matches the prototype's finding).
- Pixel-identical to the serial-gzip output (all 1.118B pixels, bins, indexes, attrs) and
  cooler-readable; both files are plain gzip+shuffle, differing only in chunk size.
- **gzip compat is now free**: same wall time as blosc:zstd:1 while producing a **36% smaller**
  file. Costs ~5.5 cores vs blosc's ~1.5.
- **merge is no longer write-bound**: gzip 26s / blosc 26s vs a 24s uncompressed floor, i.e.
  within 8% of the read+merge path itself. Further write optimization (P3.5 blosc direct-chunk)
  would buy ~nothing for merge; the remaining cost is the k-way merge/read side.
- Scales: the 2.56B-pixel cooler re-writes with gzip4 in **64s** (5.9 GB out).

## Phase 4 gate — cload phase split after P3 (2026-08-09)
`cload ENCFF514KZU.pairs.gz 256 --mem 8 --threads 8` (2,610,545,790 pairs -> 2,563,532,430 pixels):
- phase A (parse + bin + sort + compressed spill, 24 runs): **35s** (74 Mpairs/s)
- phase B (single-thread k-way merge + write): **123s**
- total **158s** -> phase B is **78% of wall time**, far above the 35% skip threshold.
=> **P4 proceeds.** (P3 did not touch this: cload's default preset is blosc, whose write path
is unchanged; and phase B's cost is dominated by the single-thread k-way merge itself.)

## Phase 4 — ranged-parallel cload phase B (2026-08-09)
`cload ENCFF514KZU.pairs.gz 256 --mem 8 --threads 8`, 2.61B pairs -> 2,563,532,430 pixels:

| | phase A | phase B | total |
|---|---|---|---|
| before (serial phase B) | 35s | 123s | **158s** |
| after (8 ranges x 8 threads) | 35s | **71s** | **106s** |

- **1.5x end to end, 1.7x on phase B**; output **pixel-identical** to the reference cooler
  (all 2,563,532,430 pixels, bins, indexes, attrs — `validate_vs_cooler.py`).
- Not 8x because the single writer thread serializes the append side; that is by design
  (HDF5 stays on one thread) and is the next limit if phase B is ever revisited.
- Cross-check that the write is no longer the phase-B bottleneck: the same cload with the
  *parallel* gzip writer takes 116s (vs 108s blosc) while burning 9m54 user vs 6m31 — more
  CPU, no wall-clock win. Phase B is now merge/read-bound. (gzip does give a 12% smaller
  file: 5.88 GB vs 6.70 GB.)
- SPILL_BLK 1M -> 128K as planned, so resident RAM (ranges x runs x one decoded block) stays
  small enough to keep P ranges parallel at high run counts. Measured compression cost:
  **1.866 vs 1.860 B/key = 0.3%**, far under the 10% budget.
- `RunReader::open_range` seek rule, and the bug the test caught: seeking to the last block
  with `first_key <= lo` **drops pixels** when a run of equal keys straddles a block boundary
  and lo lands on that key — the trailing duplicates in the earlier block are skipped. The
  predicate must be strict (`first_key < lo`). Found by the range-partition unit test before
  it ever ran at scale; it would have silently lost counts.

## Phase 5 — distiller parity + convenience (2026-08-09)
- `zoomify --balance` balances every level after the cascade. It must run *after* zoomify
  returns: the builder's HDF5 handles have to be released before balance can reopen the file
  read-write (same handle-scoping rule as P2). Picks the tiled SpMV automatically above 4M bins.
- `--nproc` accepted as an alias for `--threads` on cload/merge/balance/zoomify; `--chunksize C`
  maps to `--mem = C x nproc x 40 B` (MEMORY_CALIBRATION.md) when `--mem` is left at default,
  and logs the mapping. Verified: `--nproc 4 --chunksize 10000000 -> --mem 1.60 GB`.
- `--view custom:<bed>`: 3-4 column BED, validated against the cooler (known chrom,
  0 <= start < end <= length, no overlaps per chrom), named after the file stem so several
  custom views coexist. Verified end to end (3 regions -> 800 rows, stored under
  `expected/v` + `views/v` alongside the existing `chroms` view); an overlapping BED errors.
- Genome table: added name aliases danRer11/GRCz11, rn6, rn7/mRatBN7.2, galGal6/GRCg6a,
  bosTau9/ARS-UCD1.2, susScr11, canFam4, TAIR10, IRGSP-1.0 -> all default to whole chromosomes.
  Deliberately NO length fingerprints for these: an unverified fingerprint would mislabel data.
- Not a regression, noted while testing: balance does not converge on the 5M-pair *subset* at
  100kb (only 2311/32353 bins survive the mask; the surviving matrix is effectively
  disconnected, so IC plateaus at cv~2.1). Byte-identical behaviour on the pre-P4 binary, and
  it correctly reports `converged=false`. Dense/real data converges normally.
- Tests: 39 (35 unit + 4 integration), including custom-BED parse/validation cases and
  `zoomify --balance` asserting a converged weight at every level.

## Phase 6 — large-scale stretch test (2026-08-09)

### Sizing: why 100B pairs and not 300B
Disk math before starting (1.20 TB free on /workspace), using the measured 1.87 B/key spill and
~2.2 B/pixel output, with pixels ≈ pairs at 64bp (the bin space is so large that nearly every
pair lands on its own pixel):

| pairs | spill | cooler | peak during cload |
|---|---|---|---|
| 300B | 0.56 TB | ~0.62 TB | **~1.18 TB = 98% of free — refused** |
| 150B | 0.28 TB | ~0.31 TB | ~0.59 TB |
| 100B | 0.19 TB | ~0.21 TB | ~0.40 TB (chosen; leaves room for the mcool too) |

300B is **disk-limited on this box, not architecture-limited** — and the estimate is uncertain
enough that a 10% miss would fill the volume mid-run. Ran 100B as the plan's fallback allows.
(The synthetic stream actually compresses worse than real data: **2.83 B/key**, because its trans
pairs are uniform and so delta-code poorly. Real spill was 1.87 B/key.)

### Result: `genpairs 100e9 | rooler cload - 64 mega64.cool --mem 32 --threads 8`

| | |
|---|---|
| input | **100,000,000,000 pairs** (streamed, never stored as text) |
| output | **81,477,686,796 pixels** over **48,254,229 bins** at **64 bp**, hg38 |
| phase A | 1787s — parse + bin + sort + spill, 205 runs, 56 Mpairs/s |
| phase B | 5114s — ranged-parallel merge, **8 ranges x 8 threads even at 205 runs** |
| total | **6901s (1h55m)** |
| **peak RSS** | **29.8 GB** (with `--mem 32`) — comfortably inside a 64 GB budget |
| file | 175.9 GB = **2.16 B/pixel** |

- The SPILL_BLK 1M -> 128K change (P4) paid off exactly as intended: with 205 runs the memory
  budget still allowed the full 8 parallel ranges. At 1M blocks the same budget would have
  allowed only 6.
- RAM is bounded by `--mem`, not by the data: 100B pairs and 48M bins ran in 30 GB.

### zoomify at 64bp scale (same run)
`rooler zoomify mega64.cool mega64.mcool --resolutions 64,128,256,512,1024`, streaming, RSS ~8 GB:

| level | bins | pixels | cumulative time |
|---|---|---|---|
| 64 bp (base copy) | 48,254,229 | 81,477,686,796 | 1917s |
| 128 bp | 24,127,123 | 73,965,326,303 | 5601s |
| 256 bp | 12,063,568 | 66,268,934,613 | 9135s |
| 512 bp | 6,031,791 | 58,537,355,526 | 12265s |
| 1024 bp | 3,015,901 | 50,807,785,752 | 15042s |

Complete: **5 resolutions in 15042s (4h11m), peak RSS 0.8 GB** — zoomify is genuinely streaming;
its whole working set is the fine->coarse bin map plus the writer's per-bin counter
(48.25M x 8 B + 24.1M x 8 B at the widest point). Final mcool 622 GB.

Note how little the pixel count drops when coarsening (81.5B -> 74.0B -> 66.3B): at this depth
and resolution the matrix is so sparse that most pixels stay distinct, so each coarser level
costs nearly as much as the last. That is a property of the data, not of the implementation.

### Verification
- `nnz` attr == pixel-table length == `bin1_offset[-1]` at **all five** levels.
- **Count conservation across the whole cascade.** Coarsening maps fine bin b -> b//factor
  monotonically within a chromosome and preserves (lo,hi) order, so the pixels with
  `bin1 < K` at 64bp are exactly those with `bin1 < K/factor` at each coarser level. Summing
  that slice (K = 100,000 fine bins of chr1, 247,606,698 pixels at 64bp):

  | level | pixels in slice | counts |
  |---|---|---|
  | 64 bp | 247,606,698 | 484,364,054 |
  | 128 bp | 230,926,281 | 484,364,054 |
  | 256 bp | 214,714,425 | 484,364,054 |
  | 512 bp | 198,762,128 | 484,364,054 |
  | 1024 bp | 182,649,198 | 484,364,054 |

  **Exactly equal at every level** — no counts created or lost by the cascade.
  (Summing all 81.5B counts is impractical here; this 247M-pixel slice is the check that fits.)
- python `cooler` opens both the finest and coarsest levels (`nnz=81477686796, binsize=64` and
  `nnz=50807785752, binsize=1024`, 24 chroms) and `clr.matrix().fetch("chr1:0-500,000")` returns
  correct, symmetric 7813x7813 and 489x489 blocks.

### Verdict
The 64bp claim holds: **100B pairs -> an 81.5B-pixel, 48.3M-bin cooler and a multi-resolution
mcool (5 levels, 622 GB), entirely out of core, with a 29.8 GB peak for cload (`--mem 32`) and
0.8 GB for zoomify** — a 176 GB cooler and a 622 GB mcool on a 125 GB machine. 300B was not attempted because it needs ~1.18 TB of a 1.20 TB volume —
a **disk** limit on this box, not an architectural one; the run is linear in pairs, so 300B
would take ~3x the time and ~3x the space.
