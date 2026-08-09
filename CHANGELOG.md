# Changelog

## Unreleased

Review-fix batch (post-alpha code review; all outputs verified pixel-identical to alpha.1
where behavior was meant to be unchanged, and the 1.1 B-pixel gzip merge benchmark shows no
performance regression).

### Fixed
- **`.pairs` positions are now read as 1-based** (the 4DN spec, and cooler's default), i.e.
  binned as `(pos-1)/binsize` rather than `pos/binsize`. The old convention shifted every read
  whose position was an exact multiple of the binsize into the next bin (~1 read-end in
  `binsize`; 0.4% at 256 bp) and let a read at the end of a chromosome spill into the next
  chromosome. `cload` output is now **byte-identical to `cooler cload`** — verified on all
  2,563,532,077 pixels of a real micro-C file and again at 10 kb. `--zero-based` opts out.
  This was found by benchmarking against cooler at 2.5 B pixels; earlier "pixel-exact" checks
  had compared rooler against itself.
- **Thread-safety of the parallel gzip writer.** The raw `H5Dwrite_chunk` calls now run under
  the hdf5 crate's global lock. Previously, `merge --threads >1` (gzip is the default preset)
  had worker threads reading inputs through libhdf5 while the writer wrote chunks outside the
  lock — undefined behavior on the stock non-threadsafe libhdf5 builds most systems ship.
- **`zoomify --resolutions` is validated.** A resolution finer than the base or not an integer
  multiple of it is a hard error (a truncated coarsening factor silently corrupted
  coordinates before). Lists are deduplicated, and the cascade now builds each requested level
  from the coarsest already-built level that divides it — so a list that omits the base or is
  not chainwise divisible (e.g. `2000,3000` on a 1000 bp base) builds exactly the levels it
  names, correctly. The base level is only written if requested.
- **`bins/chrom` is a real HDF5 ENUM again** (the cooler schema type), with the chromosome
  names as members — not bare int32 codes. Values are unchanged; h5py/cooler see the same
  codes plus the name mapping.
- **Failed ops no longer leave a valid-looking output behind.** merge joins its workers before
  finalizing the file (a failed range used to be silently skipped, leaving a consistent cooler
  missing pixels), and cload/merge delete a partially-written output on error. cload also
  cleans its spill directory on failure.
- **cload validates positions against chromosome lengths.** An out-of-range position (bad
  chromsizes, malformed field) used to land silently in the next chromosome's bins.
- **Parallel merge memory formula includes the input count** — merging many coolers under
  `--mem` no longer overshoots the budget by a factor of k.
- A pairs file whose final body line lacks a trailing newline no longer drops that pair when
  it is the first body line; `merge` with zero inputs is a usage error, not a panic;
  chromosomes over 2^31−1 bp and names over 64 bytes are refused loudly instead of silently
  wrapping/panicking; `--mem` and `--chunksize` are mutually exclusive (precedence was
  previously decided by sniffing whether `--mem` equalled its default); bin counts above
  ~3.04e9 (i64 key overflow) are refused.
- **Python read API:** balancing weights are read from disk once and cached on the open
  handle (each `balanced()` fetch used to re-read the entire weight column), and a cis square
  fetch reuses the pixel rows it already read instead of reading them from HDF5 twice.

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
