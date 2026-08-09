# rooler implementation plan (hand-off)

Ordered work plan derived from the 2026-08-08 code review (`STATUS.md`). Each task says exactly
what to change, where, and how to prove it works. Do the phases in order; **one commit per
numbered task**, message format: `P<phase>.<task>: <summary>`. After every task:
`source ~/.cargo/env && cargo build --release && cargo test --release` must pass.

Ground rules
- **Counts are stored as i32, by decision.** Never widen storage dtypes. Internal accumulators
  stay i64; on emit, **clamp** to `i32::MAX` (saturate, don't wrap, don't error) and count how
  many pixels were clamped; print one warning line per op if the count is nonzero.
- Do NOT touch the SpMV hot loops in `scratch.rs` / `scratch_tiled.rs` (`spmv`, `marginals`,
  `decode`, `decode_chunk`) except where a task explicitly says so. They are perf-tuned and
  bit-validated.
- Do NOT change the on-disk cooler schema (dataset names/dtypes/attrs). Everything must keep
  round-tripping through python `cooler`.
- Big test coolers live in `/workspace/scratch_bench/` (e.g. `enc514_256.cool`, 2.56B pixels).
  Copy before writing into one (`balance`/`expected` modify in place). Real pairs:
  `/workspace/encode_hic_pairs/*.pairs.gz`.
- Machine: 24 cores (8P+16E), 125GB RAM. Default `--threads 8` conventions stay.

---

## Phase 0 — baseline snapshot (no code changes)

Record, in `PLAN_LOG.md` (create it; append as you go):
```
cargo build --release && cargo test --release        # expect 3 tests pass
cp /workspace/scratch_bench/enc514_256.cool /workspace/scratch_bench/plan_base.cool
time ./target/release/rooler balance /workspace/scratch_bench/plan_base.cool --block 65536 --threads 8
```
Expected ballpark: scratch build ~27s, total ~48s, `converged=true`, mask 11239212/12537161.
Keep `plan_base.cool` around; later phases reuse it.

---

## Phase 1 — correctness batch

### P1.1 clamp i64→i32 count casts (zoomify, merge, cload)
Add to `src/cooler.rs`:
```rust
/// Saturating i64 -> i32 count cast; bumps *nclamped when the value doesn't fit.
pub fn clamp_count(v: i64, nclamped: &mut u64) -> i32 {
    if v > i32::MAX as i64 { *nclamped += 1; i32::MAX } else { v as i32 }
}
```
Replace every silent `as i32` on a count value:
- `src/zoomify.rs:72` (`RowAgg::flush_row`, `s as i32`) — thread a `nclamped: u64` field
  through `RowAgg`; after each level, if nonzero:
  `eprintln!("  [zoomify] WARNING {}bp: {} pixels clamped at i32::MAX", res, n)`.
- `src/merge.rs:85, 142, 185` (the three emit closures mapping `cnts` → i32) — accumulate a
  clamp counter (for the parallel path use `AtomicU64`), warn once at end of the merge.
- Audit `src/cload.rs` — its counts flow through the same merge emit; no extra sites.
Test (add to a new `tests/` — see P2.2 for the lib-target prerequisite; if you prefer, fold this
test into P2): write a tiny cooler via `CoolWriter` with two pixels of count `i32::MAX - 5` that
coarsen into one bin at the next zoom level; run `zoomify`; assert the coarse pixel's count is
exactly `i32::MAX` and the warning fired.

### P1.2 merge input validation
In `merge_coolers` AND `merge_coolers_parallel` (`src/merge.rs`): after reading `meta` from
`paths[0]`, loop over `paths[1..]`, `read_meta` each, and error unless `names`, `lengths`,
`binsize`, and `nbins` all equal the first — message must name the offending file and field:
`bail!("merge: {} has binsize {} but {} has {}", ...)`.
Test: create two `test-write` coolers (variant 0 and a hand-made one with different chroms via
`CoolWriter` in a unit test) and assert merge errors; assert same-schema merge still works.

### P1.3 cload: error (not panic) on unknown chromosome
`src/cload.rs:77-78`: replace `bins.cmap[c1]` indexing with a lookup that returns
`Err(anyhow!("pairs line references chromosome {:?} not present in the #chromsize header",
String::from_utf8_lossy(c1)))`. Same for `c2`. The worker already returns `Result`; make sure the
error propagates out of `cload()` (the `h.join().unwrap()?` path already does this once it's an
`Err` instead of a panic).
Test: unit test on a 3-line pairs body (make the parse loop testable by extracting the per-line
parse into `fn parse_line(line: &[u8], bins: &Bins, ...) -> Result<Option<i64>>` — pure function,
easy to test; keep it `#[inline]` so perf is unchanged). Verify a line with a bogus chrom yields
the error and a good line yields the right key.

### P1.4 strict `Comp::parse`
`src/cooler.rs`: change signature to `pub fn parse(s: &str) -> Result<Comp>`; unknown strings /
bad levels are errors (`bail!("unknown compression preset '{}' (use gzip[N] | blosc:zstd:N |
blosc:lz4:N | none)", s)`). Update the four call sites in `src/main.rs` to `Comp::parse(&preset)?`.
Test: unit test — `"gzip4"`, `"gzip"`, `"blosc:zstd:1"`, `"blosc:lz4:5"`, `"none"` parse;
`"gizp4"`, `"blosc:snappy:1"`, `"gzip99"` (level >9) error.

### P1.5 sacCer3 centromere table (fixes yeast `expected` refusing by default)
`src/view.rs`: add
```rust
// sacCer3 CEN midpoints (SGD, approximate to ~1kb — fine for arm-splitting expected)
const SACCER3_CEN: &[(&str, i64)] = &[
    ("chrI",151465),("chrII",238207),("chrIII",114385),("chrIV",449711),
    ("chrV",151987),("chrVI",148510),("chrVII",496920),("chrVIII",105586),
    ("chrIX",355629),("chrX",436307),("chrXI",440129),("chrXII",150828),
    ("chrXIII",268031),("chrXIV",628758),("chrXV",326584),("chrXVI",555957),
];
```
and return it from `centromeres("saccer3")`. If network access is available, verify the values
against SGD/UCSC sacCer3 cytoBand and correct any that differ; if not, ship as-is (the module
header already declares approximate is fine).
Test: unit test `resolve("sacCer3", &[("chrI",230218),("chrXII",1078177)], None)` → Ok, name
"arms", chrI split at 151465, 32-region count logic (2 per chrom present).

### P1.6 `CoolWriter::append`: real sorted-order check
`src/cooler.rs:224`: `debug_assert!` → real check:
`if bin1[0] < self.last_bin1 { bail!("append: blocks must be non-decreasing in bin1 ({} after {})", bin1[0], self.last_bin1); }`
It runs once per multi-million-pixel block — cost is zero.
Test: unit test that out-of-order appends error.

### P1.7 `expected`: don't wipe sibling views
`src/expected.rs:98`: `let _ = g.unlink("expected");` → `let _ = g.unlink(&format!("expected/{}", view_name));`
(the later `create_group("expected/{view}/weight")` already creates intermediates, and the
existing `views/{view}` unlink is already scoped correctly).
Test: on a small balanced cooler run `expected --view chroms` then `expected --view arms`
(hg38-fingerprinted chromsizes), then assert **both** `expected/chroms` and `expected/arms`
groups exist.

### P1.8 naming/dead-code cleanup (no behavior change)
- `src/merge.rs`: rename `merge_sources_parallel` → `merge_sources_to_writer` (it is serial);
  drop the unused `_nthreads` param; update the call in `src/cload.rs` and its misleading
  "ranged-parallel" comment (say: single-thread heap merge; ranged-parallel is P4).
- `src/cload.rs`: `RunReader::open(p, _block)` → `open(p)`; delete the dead `block` computation
  at cload.rs:176 (leave a comment: phase-B RAM = #runs × SPILL_BLK decode buffers; see P4).
- `src/balance.rs:81`: delete `let _ = &mut marg;`.
- `src/cooler.rs`: delete unused `chrom_enum` (it's the dead-code warning in every build) — the
  enum-fidelity TODO lives in PROGRESS.md, we don't need the stub.
Acceptance: `cargo build --release 2>&1 | grep warning` → no dead-code warnings; behavior
identical (`rooler balance` on `plan_base.cool` still converges with same mask count).

---

## Phase 2 — regression test suite

### P2.1 make modules testable: add a lib target
Create `src/lib.rs`:
```rust
pub mod cooler; pub mod merge; pub mod cload; pub mod zoomify;
pub mod balance; pub mod scratch; pub mod scratch_tiled; pub mod expected; pub mod view;
```
`src/main.rs`: drop the `mod` declarations, `use rooler::{...}` instead. Make the items tests
need `pub` (codec fns in `scratch.rs`: `shuffle4/unshuffle4/enc_count/dec_count` are
`pub(crate)` — promote to `pub`; same for cload's `shuffle8/unshuffle8`, `fast_atoi`,
`parse_line`, `Bins`, `RunReader`, merge's `BlockSource/merge_sources`).

### P2.2 unit tests (in-module `#[cfg(test)]` blocks)
- **codecs** (`scratch.rs`): round-trip `shuffle4`→`unshuffle4` and `enc_count`→`dec_count` on:
  empty-ish (1 elem), all-small counts, counts with >255 exceptions (incl. exactly 255/256 and
  `i32::MAX`), random 10k vector (use a fixed-seed LCG, no rand crate).
- **cload codecs** (`cload.rs`): `shuffle8`→`unshuffle8` round trip; spill-block write→read:
  extract the block-encode body of the `spill` closure into
  `pub fn write_spill(w: &mut impl Write, sorted_keys: &[i64]) -> Result<()>` (used by the
  closure), then test write→`RunReader` read reproduces sorted keys across multiple
  `SPILL_BLK` boundaries.
- **merge** (`merge.rs`): a `VecSource` (Vec of blocks) `BlockSource` impl in the test module.
  Cases: 3 sources with overlapping keys → summed counts; a key's run spanning a block boundary
  within one source; empty blocks; single source; empty source list.
- **zoomify** (`zoomify.rs`): `build_map` respects chrom boundaries (2 chroms, odd bin counts,
  factor 2 and 4: last fine bin of chrom A never maps into chrom B's first coarse bin).
- **view**: keep existing 3, add the sacCer3 test (P1.5).

### P2.3 end-to-end pipeline test (`tests/pipeline.rs`)
Prerequisite feature: **cload must accept plain-text (non-gz) pairs** so the test doesn't need
bgzip. In `cload()`: if `pairs` doesn't end with `.gz`, read the file directly instead of
spawning bgzip (`Box<dyn Read>` over either source). Also accept `-` = stdin (needed by P6).
Test flow (all in a `tempfile`-style scratch dir under `std::env::temp_dir()`; use
hg38-truncated chromsizes so assembly detection passes, e.g. `chr1:248956422`, `chr2:242193529`
— pairs positions only in the first few Mb):
1. Deterministically generate ~200k pairs (fixed-seed LCG; mix cis/trans; positions in 0..3Mb;
   include header `## pairs format v1.0` + `#chromsize:` lines; some duplicate pairs to
   exercise drain-and-count).
2. Compute the expected pixel map in-test (HashMap<(b1,b2),i64> with the same lo/hi swap).
3. `cload::cload(...)` at 10kb with tiny `--mem` (force multiple spill runs; assert
   `run_paths.len() > nthreads` by using mem=0.001) → open the output with the `hdf5` crate →
   assert pixel table **exactly** equals the sorted expected map; assert `bin1_offset` and
   `chrom_offset` consistent (offsets sum, monotone).
4. `merge::merge_coolers_parallel([out, out]) `→ every count doubled, same keys.
5. `zoomify` to 20kb+40kb → per-level total count conserved AND 20kb table equals brute-force
   coarsening of the 10kb table computed in-test.
6. `balance::balance` (tol 1e-6, ignore_diags 1, min_nnz 2, mad_max 0 for the synthetic) →
   read `bins/weight`; property check: balanced marginals of good bins have CV < 1e-3
   (compute directly from the pixel table + weights in the test).
7. `expected::expected(..., Some("chroms"))` → compare `balanced.sum`/`n_valid` for chr1
   against a brute-force O(n²) computation from the dense balanced matrix.
These are pure property/oracle checks — **no python, no cooler dependency** in CI path.
Acceptance: `cargo test --release` runs the whole suite < ~60s.

### P2.4 keep a manual cross-check script
`scripts/validate_vs_cooler.py` (new): given `a.cool b.cool`, assert equal pixels via h5py, and
optionally `cooler.Cooler(a).matrix().fetch("chr1:0-5,000,000")` equality. Used manually after
big changes; not part of cargo test. Document in README ("Validation").

---

## Phase 3 — parallel gzip writer (port the prototype into CoolWriter)

Context: `--compat` gzip writes are single-thread deflate inside HDF5 ≈ 220MB/s; the validated
python prototype (`/workspace/GZIP_PARALLEL_FINDINGS.md`, `/workspace/scratch_gzip/`) gets 8.6×
by compressing chunks in parallel and writing with `H5Dwrite_chunk`. Files stay byte-compatible
(each chunk is an independent zlib stream; any valid zlib stream is readable). Port to Rust.

### P3.1 dependencies + FFI smoke test
- `Cargo.toml`: add `libdeflater = "1"` and `hdf5-metno-sys = "0.10.1"` (already in the lock
  file as hdf5-metno 0.9.4's sys crate — same version, no duplicate).
- FFI (verified present): 
  `hdf5_metno_sys::h5d::H5Dwrite_chunk(dset_id: hid_t, dxpl_id: hid_t, filters: u32, offset: *const hsize_t, data_size: size_t, buf: *const c_void) -> herr_t`.
  Call with `dxpl_id = hdf5_metno_sys::h5p::H5P_DEFAULT` (= 0), `filters = 0` (0 = no filter
  skipped; a SET bit means that filter was SKIPPED for the chunk).
- Smoke test (unit test): create a 1-D chunked+shuffle+deflate i32 dataset of 3000 elems, chunk
  1024; for each of the 3 chunks: pad to 1024 elems (zeros), byte-shuffle (transpose: dst
  `b*n+e` ← src byte `b` of elem `e`), zlib-compress (libdeflater `Compressor`, level 4,
  `zlib_compress` — NOT `deflate_compress`/`gzip_compress`; HDF5 expects the 2-byte 0x78 zlib
  wrapper), `H5Dwrite_chunk` at offset `[i*1024]`. Read back through the normal API → equals
  input. This one test de-risks the whole phase.

### P3.2 `src/parwrite.rs`: parallel chunked column writer
```rust
pub struct ParColumn { dset: hdf5::Dataset, elem_size: usize, chunk: usize, level: i32,
                       staged: Vec<u8> /* raw little-endian bytes */, nwritten_chunks: usize,
                       total_elems: usize }
```
Behavior:
- `push_i64(&mut self, v: &[i64])` / `push_i32(&mut self, v: &[i32])` append raw LE bytes to
  `staged`; when `staged` holds ≥ K full chunks (K = 16), `flush_full_chunks()`.
- `flush_full_chunks()`: split staged into whole-chunk byte slices; **rayon par_iter** each:
  shuffle (byte transpose over `chunk` elems × `elem_size`) then libdeflater zlib level
  `self.level` into a fresh Vec (bound via `zlib_compress_bound`); collect in order; then, on
  the calling thread only (HDF5 is not thread-safe — every H5 call stays on the caller):
  `dset.resize(total_elems_so_far)` once, then `H5Dwrite_chunk` each chunk at offset
  `[(nwritten_chunks + i) * chunk]`.
- `finish(&mut self)`: `dset.resize(exact_total)` FIRST, then pad the final partial chunk with
  zeros to full chunk size, shuffle+compress+write it (writing a full-size edge chunk after
  set_extent is the standard pattern — h5py's `write_direct_chunk` does exactly this; the
  padding tail is never readable).
- Simplicity over pipelining: batch-parallel compression already captures the win (deflate is
  ~97% of the cost); do NOT build a background writer thread.

### P3.3 wire into `CoolWriter` for the gzip preset
In `create_in`, when `comp` is `Comp::Gzip(l)`: create `pixels/bin1_id|bin2_id|count` exactly as
today (`.shuffle().deflate(l)`, resizable) but with **chunk = 262144 elems** (findings: 64–256K
optimal; today's 1M chunk is fine for blosc, keep 1M there), and hold three `ParColumn`s.
`append()` routes to `ParColumn::push_*` (bin-order bookkeeping/`bincount` unchanged);
`close()` calls `finish()` on each. Blosc/None presets keep the current write path untouched.
Default gzip level stays whatever `gzip4` parses to; findings say L4 = right default, never >6.

### P3.4 validate + bench
- `cargo test`: e2e pipeline test (P2.3) rerun with `preset = gzip4` → identical pixel oracle.
- Cross-check with python cooler (`scripts/validate_vs_cooler.py`) on a real file:
  `rooler merge /tmp_out.cool /workspace/scratch_bench/plan_base.cool --preset gzip4 --threads 8`
  vs the same merge with the pre-P3 binary (build it from the P2 commit) — pixel-identical, and
  `h5dump -pH` on both shows the same SHUFFLE+DEFLATE filter pipeline.
- Bench (append results to `PLAN_LOG.md`): time that single-input merge (≈ pure re-write of
  2.56B pixels) with `--preset gzip4` before vs after; expect ≥4× on the write-bound path, and
  gzip-after ≈ within ~1.5× of blosc:zstd:1. Also record output file sizes.

### P3.5 (stretch, only if P3.1–P3.4 landed cleanly) blosc via the same path
Same direct-chunk trick for `Comp::BloscZstd/Lz4`: compress chunks with the `blosc-src` C API in
rayon threads (`blosc_compress_ctx` — the _ctx variant is thread-safe/stateless), write with
`H5Dwrite_chunk`. Byte layout must match the blosc HDF5 filter's chunk format (blosc frames are
self-describing — the filter stores the raw blosc frame; verify by `H5Dread_chunk`-ing a chunk
written by the current path and comparing against `blosc_compress_ctx` output on the same
input, params: clevel, shuffle=BYTE, typesize=elem_size, blocksize 0). If the frames don't
match byte-for-byte but round-trip, that's fine — acceptance is read-back equality via the
normal HDF5 API + python cooler. This makes merge/cload write-parallel for the DEFAULT preset,
not just --compat. If it turns messy, stop and leave a note in PLAN_LOG.md.

---

## Phase 4 — cload phase-B: ranged-parallel merge of spill runs

Do this AFTER P3 and re-measure first: with parallel writes, re-run
`rooler cload /workspace/encode_hic_pairs/ENCFF514KZU.pairs.gz 256 /tmp.cool --mem 8 --threads 8`
(before P4, log the phase split). If phase-B is now < ~35% of wall time, SKIP P4 (diminishing
returns) and note it. Otherwise:

### P4.1 spill format: add per-block first_key
`write_spill` block header becomes `[n: u32][clen: u32][first_key: i64]` (16 bytes). Bump a
format marker: first 8 bytes of every run file = magic `b"RKZ2\0\0\0\0"`. `RunReader` checks the
magic. (Runs are transient within one cload call — no back-compat needed.)

### P4.2 don't materialize counts for count=1 sources
`BlockSource::next` returns `(Vec<i64>, Counts)` where
`enum Counts { Ones, PerKey(Vec<i64>) }` with `#[inline] fn at(&self, i: usize) -> i64`.
`merge_sources` uses `cnt.at(pos)`. `CoolerPix` returns `PerKey`, `RunReader` returns `Ones`
(halves phase-B resident memory). Re-run the full test suite — merge oracle tests cover this.

### P4.3 ranged-parallel phase B
- Reduce `SPILL_BLK` to `131_072` (resident/source ≈ 1MB decoded; compression ratio barely
  moves at 128K keys — verify on one run file, expect ≤10% worse than 1M-key blocks).
- `RunReader::open_range(path, lo_key, hi_key)`: scan block headers (read 16-byte header, seek
  `clen` forward — no decompression) until `first_key` of the NEXT block > `lo_key`; decode
  from the current block on, and clip yielded slices to `[lo_key, hi_key)` (keys within a block
  are sorted → binary search the boundaries; a key's duplicates never straddle ranges since
  ranges partition key space).
- Partition: collect every block's `first_key` across all runs (headers only), sort, take P-1
  quantile pivots → P key ranges. Then copy the `merge_coolers_parallel` pattern verbatim:
  per-range thread = `merge_sources` over `open_range` readers → bounded channel → single
  writer drains ranges in order. Delete the now-unused serial call in `cload`.
- Memory statement in the log: resident ≈ P × nruns × 1MB; with the default P=8 cap this is
  fine to ~1000 runs; add `let p = nthreads.min( (mem_gb*0.5*1e9 / (nruns as f64*1.1e6)) as usize ).max(1);`
  so `--mem` genuinely bounds it.
Acceptance: pipeline e2e (which forces many runs) still pixel-exact; ENCFF514KZU @256bp cload
matches the pre-P4 output cooler pixel-for-pixel (`scripts/validate_vs_cooler.py`), and phase-B
wall time drops ≥3× (log before/after).

---

## Phase 5 — distiller parity + convenience

### P5.1 `zoomify --balance`
Flag on the CLI: after each level is written (including the base copy), call `balance::balance`
on `out::resolutions/{res}` with defaults, `nthreads` from a new `--threads` arg (add it to
Zoomify, default 8), and `tiled_block: if nbins >= 4_000_000 { Some(65536) } else { None }`.
Progress line per level. Acceptance: e2e test extended — zoomify --balance leaves a
`bins/weight` with `converged=True` at every resolution.

### P5.2 cooler CLI shims
- `--nproc N` as a clap alias for `--threads` on cload/merge/balance (`#[arg(long, alias = "nproc")]`).
- cload/merge: optional `--chunksize C` → if given and `--mem` left at default, set
  `mem_gb = C as f64 * threads as f64 * 40e-9` (from MEMORY_CALIBRATION.md), and log the mapping.
### P5.3 custom view: `--view custom:<path>`
`view.rs`: parse a 3–4 column BED (chrom, start, end[, name]; tab-separated, `#` comments) into
`Vec<Region>` (name defaults to `chrom:start-end`), returned with view name = the file stem.
Validate: chroms exist in the cooler, 0 ≤ start < end ≤ chrom length, regions non-overlapping
per chrom (sort + check). Wire through `expected --view custom:regions.bed`. Unit tests: good
bed, overlapping bed (error), unknown chrom (error).
### P5.4 genome table growth (small, optional)
Add name aliases only (no fingerprints without verified lengths): `danRer11/GRCz11`,
`rn6`, `rn7/mRatBN7.2`, `galGal6/GRCg6a`, `bosTau9/ARS-UCD1.2`, `susScr11`, `canFam4`,
`TAIR10`, `IRGSP-1.0` → all `ViewKind::Chroms` default. If network is available, fetch UCSC
chromInfo for each and also add chr1-length fingerprints; otherwise names only.

---

## Phase 6 — 300B-pair stretch test (only after P1–P4 are green)

Goal: prove cload→merge→zoomify at 64bp on a ~64GB budget, out-of-core, at 300B pairs — using
a streaming generator (nothing this size is ever stored as text).
- `examples/genpairs.rs` (cargo example): deterministic LCG; hg38 chromsizes baked in; 70% cis
  with distance ~ s^-1 on [500, chrlen] (inverse-CDF: `s = exp(u * ln(chrlen/500)) * 500`),
  30% trans uniform; writes 4-column body lines to stdout with the `#chromsize` header.
  Throughput target ≥50M lines/s with `write!` into a 4MB BufWriter (it will be the producer
  for cload's stdin path from P2.3).
- Disk math FIRST (log it): 300B keys × ~1.8B spill ≈ 540GB spill + output cooler
  (~2-2.5 B/pix × nnz). Check `df /workspace`; if insufficient, scale the run down (100B pairs
  ≈ 180GB spill) and say so in the log — a 100B run still proves the architecture.
- Run: `./target/release/genpairs 100000000000 | rooler cload - 64 mega64.cool --mem 48
  --threads 16`, then zoomify 64→…; record wall/RSS via `/usr/bin/time -v` in `PLAN_LOG.md`.
  Watch for the P1.1 clamp warning at coarse levels — at this scale it SHOULD fire; that's the
  designed behavior, log the clamped-pixel counts per level.

---

## Explicitly out of scope (do not start)
- KR balancing; enum `bins/chrom` fidelity; loser-tree merge; per-tile CSR bin1 / B sweep /
  persistent rayon pool in balance; lzf read plugin; any Python API changes beyond docs.

## Definition of done, per phase
Phase is done when: all its tests pass in `cargo test --release`, the phase's bench/validation
numbers are appended to `PLAN_LOG.md`, `STATUS.md`'s findings list is updated (mark items
fixed), and the work is committed in the task-per-commit structure.
