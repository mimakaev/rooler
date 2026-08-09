# Changelog

## Unreleased

### Added
- **Expected is computed by default whenever weights are written** — after `balance`, after
  each level of `zoomify --balance`, and in `repack`. One O(nnz) pass + FFT, with the
  per-organism default view (arms for hg38/hg19/sacCer3, whole chromosomes for mm10/mm39/dm6/
  ce11 and others); an unknown genome warns and skips instead of failing the parent op.
  `--no-expected` opts out. Departure from cooler, on purpose: this is the quantity everyone
  recomputes with cooltools at default parameters anyway, at 100–1000× the cost.
- **New op: `rooler repack <cool|mcool>`** — rewrite an existing cooler the way rooler would
  have written it: parallel-gzip preset (plugin-free, smaller), genome assembly stamped **and
  verified against the chromosome-size fingerprint** (`repack --assembly hg19` on an hg38 file
  is refused), balanced if it carries no weights (weights and their attrs are carried over
  otherwise), expected stored. In place by default via atomic tmp+rename; `--backup` keeps the
  original at `<src>.bac`; `--out` writes elsewhere.
- **`balance` now honors a `--mem` budget (default 8 GB) and is no longer RAM-only.** When the
  estimated compressed scratch plus working vectors exceed the budget, the scratch blobs are
  written to an unlinked temp file next to the cooler and mmap'd read-only instead of held on
  the heap. Results are identical; anonymous memory stays O(nbins) (measured: 2.56 B-pixel /
  12.5 M-bin balance at `--mem 8` peaks at **2.4 GB anon RSS** vs 8.6 GB in-RAM, 59 s vs 50 s
  with a warm page cache). The mmap pages are file-backed and evictable, so 8 balances sharing
  a 64 GB node degrade gracefully instead of OOMing. Temp files are unlinked immediately after
  mapping — nothing leaks even on SIGKILL. `zoomify --balance` takes the same `--mem`.
- **`balance --block` now auto-routes.** Default: tiled SpMV (B=65536) at ≥4 M bins — where
  the O(nbins) vectors outgrow L3 and tiling measured 2.8× — row-chunk below. `--block 0`
  forces row-chunk; an explicit `--block B` forces tiled. (`zoomify --balance` previously had
  this auto-pick; plain `balance` defaulted to row-chunk at any size.)

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
