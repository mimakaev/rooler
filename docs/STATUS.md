# rooler — status report (2026-08-09)

Canonical current-state doc. `PLAN.md` is the work plan, `PLAN_LOG.md` the measurements taken
while executing it, `PROGRESS.md` the historical build log, `MEMORY_CALIBRATION.md` the --mem
sizing data, `../GZIP_PARALLEL_FINDINGS.md` the parallel-gzip study that P3 implements.

## What exists and works

| op | approach | validation | perf (i9-13900KF, 8 threads) |
|---|---|---|---|
| **cload** | bgzip/plain/stdin → parallel parse/bin/sort → compressed spill runs (1.87 B/key) → **ranged-parallel** k-way drain-and-count merge → writer | pixel-exact vs reference (all 2.56B pixels); e2e oracle test | 2.61B pairs @256bp in **106s** (was 158s) |
| **merge** | streaming k-way heap merge; ranged-parallel over bin1 partitions; input compatibility checked | pixel-exact incl. megacooler; serial == parallel in tests | 1.1B-pixel rewrite in 26s (≈ the 24s uncompressed floor) |
| **zoomify** | streaming integer-factor coarsen, cascade; `--balance` balances every level | exact vs cooler.coarsen_cooler; e2e brute-force oracle | O(nnz)/level |
| **balance** | genome-wide IC over a compressed in-RAM scratch (~2–2.5 B/pix); row-chunk or 2D-tiled (`--block`) SpMV, atomic-counter reduction | 0 mask disagreements + ~1e-6 vs cooler; tiled == row to 3e-16 | 2.56B pix/12.5M bins: **48s** tiled (IC 0.98 Gpix/s) |
| **expected** | O(nnz) streaming `sum_balanced` + FFT mask-autocorr `n_valid`; arms/chroms/custom-BED views | 6.4e-16 vs cooltools; e2e brute-force O(n²) oracle | seconds |
| **write path** | blosc:zstd:1 (default) or **parallel direct-chunk gzip** (`H5Dwrite_chunk` + libdeflater) | pixel-identical to serial gzip; plain SHUFFLE+DEFLATE, no plugins needed | gzip **3.8× faster** and 4.2% smaller than before |
| **Python read API** | `rooler.open()`, `raw()/balanced()`, cooler-compat `matrix()/bins()/pixels()` | exact vs cooler; cooltools runs natively on the files | — |

Scale proofs (real data): **megacooler** = 10 ENCODE files (~40B pairs) → 26.3B-pixel 53GB
cooler @256bp in ~63 min, balanced at 70GB peak RSS.

Scale proof (synthetic, 64bp): **100,000,000,000 pairs → 81,477,686,796 pixels over 48,254,229
bins at 64 bp** in 1h55m (`--mem 32`), **peak RSS 29.8 GB** — data ~30× larger than RAM. The
mcool cascade (64→128→256→…) then streams at ~8 GB RSS. The 64bp cooler opens in python
`cooler` and fetches correctly. 300B pairs is a *disk* limit on this box (~1.18 TB of a 1.20 TB
volume), not an architectural one. See `PLAN_LOG.md`.

"No mystery coolers" (assembly required) enforced everywhere.

## Tests
`cargo test --release` — **42 tests, ~0.6s**, no python/network/fixtures.
- Unit: codecs (bin2-delta shuffle+LZ4, u8+exception counts, spill runs incl. ranged readers),
  k-way merge semantics, pairs parsing, coarsening bin map, genome/BED views, cooler writer
  (round-trip, index validity, append order, preset parsing), parallel-gzip direct chunks.
- Integration (`tests/pipeline.rs`): the whole chain against independent oracles — HashMap pixel
  table for cload/merge/zoomify, marginal-flatness property + row-vs-tiled agreement for balance,
  brute-force O(n²) for expected; plus cload determinism across `--mem`/`--threads`,
  `zoomify --balance` convergence at every level, and refusal cases.
External cross-check: `scripts/validate_vs_cooler.py` (streams a full pixel/bins/index/attr
comparison at billion-pixel scale, then verifies the file through python `cooler`).

## Review findings — status

Fixed in the post-alpha review batch (2026-08-09, see CHANGELOG "Unreleased"): raw
`H5Dwrite_chunk` raced the safe-API readers in parallel merge (now under the crate's global
lock; no measurable cost — 1.1B-pixel gzip merge 29s vs 32s for the pre-fix binary on the same
warm cache); `zoomify --resolutions` accepted non-divisible/finer-than-base lists (silent
coordinate corruption — now validated, and the cascade builds each level from the coarsest
built divisor, so base-omitting/non-chain lists work); `bins/chrom` restored to a real HDF5
ENUM (schema-compliant, values unchanged); failed merge/cload no longer leave valid-looking
partial outputs (workers joined before finalize + delete-on-error); cload validates positions
against chromosome lengths; parallel-merge RAM formula includes the input count; assorted
loud-refusals (zero merge inputs, >2.1Gb chroms, >64-byte names, i64 key overflow at ~3e9
bins, `--mem`/`--chunksize` now mutually exclusive); python API caches weights and reuses
rows on cis fetches. Test suite 39 → 42.

Fixed in this pass (P1): silent `as i32` count wrap at ~2.1e9 (now **saturating with a warning**;
i32 storage is a deliberate choice, accumulators stay i64); merge accepted mismatched inputs;
cload **panicked** on an unknown chromosome; `Comp::parse` silently swallowed typos; sacCer3
defaulted to arms with no centromere table; append order was only a `debug_assert`; `expected`
wiped sibling views; dead code and a misleading `merge_sources_parallel` name.

Fixed later: no regression tests (P2, 3 → 39); serial gzip writes (P3); serial cload phase B
(P4); missing `zoomify --balance`, `--nproc/--chunksize`, custom views, thin genome table (P5).

Bugs caught by the new tests before they reached data:
- **Ranged spill readers dropped pixels** when a run of equal keys straddled a block boundary
  and a range began on that key — the seek predicate had to be strict (`first_key < lo`).
- In-process HDF5 handle scoping: dropping the `File` is not enough; a live `Group`/`Dataset`
  keeps the file open and makes the next op's read-write open fail confusingly.

## Known limits / open items
- **balance scratch spills to a disk-backed mmap past `--mem` (default 8 GB)** — anonymous RAM
  stays O(nbins) (2.4 GB anon at 2.56 B pix / 12.5 M bins) and results are identical; the cost
  is disk traffic per matvec when the page cache can't hold the scratch (~2.5 B/pixel of it).
  A 100 B-pixel balance is therefore *possible* on a workstation but NVMe-bound: ~250 GB of
  scratch re-read per iteration when far beyond RAM. The kernel (row vs tiled) also
  auto-routes now (tiled at ≥4 M bins, `--block 0`/`--block B` to override).
- Chrom lengths are stored i32 → 2.1 Gb per chromosome (same limit as cooler; now a loud
  refusal instead of a silent wrap).
- `read_meta` reads fixed-length chrom names only (variable-length string names from foreign
  writers are not handled).
- IC can plateau without converging on very sparse/disconnected masked matrices (it reports
  `converged=false`); this matches cooler's behaviour.
- Balance perf levers not taken: per-tile CSR for bin1, tile-size sweep, persistent rayon pool.
- P3.5 (blosc via direct-chunk) was **deliberately skipped**: merge is now within 8% of the
  uncompressed floor and cload phase B is merge-bound, so it would buy ~nothing.

## Suggested next steps
1. Run the full distiller chain against rooler as a drop-in and fix whatever friction appears.
2. KR balancing if anyone actually wants it (deprioritized — tolerance is the real lever).
3. `noodles-bgzf` to drop the external `bgzip` dependency.
4. Real centromere tables for the newly added genomes (currently whole-chromosome defaults).
