# Changelog

## 0.1.0-alpha.1 — 2026-08-09

First public alpha. All five ops work, are validated against `cooler`/`cooltools`, and have
been exercised on datasets up to 100 billion pairs.

### Ops
- **`cload`** — `.pairs.gz`, plain text or stdin → `.cool`. Parallel parse and bin, compressed
  spill runs, ranged-parallel k-way merge. 2.61 B pairs → 2.56 B pixels at 256 bp in 106 s.
- **`merge`** — streaming k-way merge over pre-sorted inputs, parallel over bin1 ranges.
  Validates that inputs share a bin layout instead of silently producing a corrupt table.
- **`zoomify`** — streaming coarsening cascade; `--balance` balances every level.
- **`balance`** — genome-wide iterative correction over a compressed in-memory matrix, with an
  optional cache-blocked kernel (`--block`) worth ~2.8× on very large coolers.
- **`expected`** — cis P(s) per region in one O(nnz) pass plus an FFT. Views: `chroms`, `arms`,
  or `custom:<bed>`; several views coexist in one file.
- **Python read API** — `rooler.open()`, `raw()` / `balanced()`, and a cooler-compatible
  `matrix()` / `bins()` / `pixels()` surface.

### Notable choices
- **gzip by default.** Chunks are compressed in parallel and written with HDF5's direct-chunk
  API, making gzip **3.8× faster** than the standard single-threaded path while producing
  ordinary shuffle+deflate files that any HDF5 reader opens without plugins. `blosc:zstd:1`
  remains available; blosc coolers written elsewhere are readable.
- **Counts are `int32`,** saturating at `i32::MAX` with a warning naming the affected pixel
  count. Accumulators are 64-bit throughout.
- **No mystery coolers** — rooler refuses to write a cooler without a genome assembly.
- `--nproc` and `--chunksize` accepted for cooler compatibility.

### Testing
39 tests, ~0.6 s, with no network, fixtures or Python. Each op is checked against an
independent oracle rather than a stored answer: a recomputed pixel table for
cload/merge/zoomify, a marginal-flatness property plus cross-kernel agreement for balance, and
a brute-force O(n²) recomputation for expected. `scripts/validate_vs_cooler.py` cross-checks
against `cooler` itself at billion-pixel scale.

### Known limits
- `balance` holds the matrix in memory (~2 B/pixel); balancing at 100-billion-pixel scale is
  the main open problem. Every other op streams.
- Chromosomes capped at 2.1 Gb (as in cooler).
- Iterative correction may plateau on very sparse or disconnected matrices; it reports
  `converged=false` rather than pretending otherwise.
