# rooler — status report & code review (2026-08-08)

Canonical current-state doc. `PROGRESS.md` is the historical build log; `MEMORY_CALIBRATION.md`
holds the --mem sizing data; `../GZIP_PARALLEL_FINDINGS.md` holds the parallel-gzip-writer study.

## What exists and works (all validated against cooler/cooltools or references)

| op | approach | validation | perf (i9-13900KF, 8 P-cores) |
|---|---|---|---|
| **cload** | bgzip -@ decode → parallel parse/bin/sort → compressed spill runs (delta+shuffle+LZ4, 1.76 B/key) → k-way drain-and-count merge → writer | pixel-exact vs coolerx/numpy (5M subset + full 2.61B-pair file) | 2.61B pairs @256bp in 145s (parse 95 Mpairs/s; phase-B merge+write 118s = bottleneck) |
| **merge** | streaming k-way heap merge of cooler readers; ranged-parallel (bin1-partition, in-order writer) | pixel-exact, incl. megacooler | 26.3B pix; WRITE-bound (single-thread blosc), par ~1.5× serial |
| **zoomify** | streaming integer-factor coarsen, cascade; fine→coarse bin map respects chrom bounds | pixel-exact vs cooler.coarsen_cooler; counts conserved every level | O(nnz)/level |
| **balance** | genome-wide IC over compressed in-RAM scratch (~2–2.5 B/pix); row-chunk or 2D-tiled (`--block`) SpMV; atomic-counter reduction | weights vs cooler: 0 mask disagreements, ~1e-6 rel; tiled vs row 3e-16 | 2.56B pix/12.5M bins: tiled 48s total (IC 0.98 Gpix/s), row 80s; **matches C prototype** |
| **expected** | O(nnz) streaming sum_balanced[region][dist] + FFT mask-autocorr n_valid; arms/chroms views baked in | 6.4e-16 vs cooltools | seconds |
| **Python read API** | `rooler.open()`, `raw()/balanced()` direct region fetch + cooler-compat `matrix()/bins()/pixels()` | exact vs cooler (cis/trans/cross); cooltools runs natively on the files | — |

Scale proof: **megacooler** = 10 ENCODE files (~40B pairs) → cload+merge → 26.3B-pixel 53GB
@256bp in ~63 min; balanced at 70GB peak RSS. "No mystery coolers" enforced everywhere.

## Code review findings (2026-08-08, full read of all 10 src files + python)

### Correctness — must fix before the next scale-up
1. **i32 count overflow at design scale** — `zoomify.rs` RowAgg emits `s as i32`; `merge.rs`
   emit closures cast `x as i32`. At ~40B contacts the coarsest chr1-cis pixel is **~2.09B —
   3% under i32::MAX**; at 100B+ pairs it silently wraps negative. This is exactly the
   advertised scale. Fix: checked cast that errors cleanly + `--count-dtype i64` escape hatch
   (CoolWriter parameterized over count dtype; cooler reads int64 counts fine). Accumulators
   are already i64 — only the final casts are unsafe.
2. **merge never validates input compatibility** — meta comes from `paths[0]` only; inputs with
   different nbins/binsize/chrom order would produce silently corrupt output. Fix: read_meta all
   inputs, require identical chroms+binsize+nbins.
3. **cload phase-B RAM ignores `--mem`** — the computed `block` (cload.rs:176) is passed to
   `RunReader::open` which **ignores it**; each run holds a decoded 1M-key block (8MB) + a
   materialized `vec![1i64; n]` counts (8MB). ~16MB × #runs explains the measured RSS ≈ 2×--mem;
   at 300B pairs/--mem 8 → ~300 runs → ~5GB. Fix options: sub-slice the decoded block; stop
   materializing counts for count=1 sources; halve SPILL_BLK.
4. **saccer3 defaults to Arms but has no centromere table** — `rooler expected` on yeast refuses
   by default with a confusing error (view.rs `default_kind` vs `centromeres` mismatch). Fix:
   add the 16 yeast CEN midpoints, or default saccer3→chroms until then.
5. **cload panics on unknown chrom** — `bins.cmap[c1]` indexes the HashMap; a pairs line naming a
   chrom absent from `#chromsize:` headers panics a worker, surfacing as a bare join/unwrap panic.
   Fix: proper error naming the chromosome.
6. **`Comp::parse` never fails** — a typo'd `--preset` silently becomes blosc:zstd:1. Should error.

### Minor / polish
- `CoolWriter::append` sort-order check is `debug_assert!` → release builds skip it (make it a real check; it's once per block, free).
- `expected` recompute wipes ALL stored views (`g.unlink("expected")`), not just the one being rewritten.
- `merge_sources_parallel` is serial (name + cload.rs "ranged-parallel" comment mislead); `RunReader`'s `_block` and `write_chrom_enum`'s `_names` are dead params; `balance.rs:81` dead `let _ = &mut marg;`.
- Python `_parse_region` lacks k/M suffixes; `read_meta` can't read VarLenAscii chrom names (some 3rd-party coolers); chrom lengths as i32 caps chroms at 2.1Gb (same limit as cooler; axolotl-class genomes break — note only).
- Genome table is 7 named + 4 fingerprints, not the "20-ish" intended; dm6/ce11/sacCer3 can't be fingerprinted (no chr1). `--view custom:<bed>` advertised in errors but not implemented.

### Maintainability assessment
- **Good**: 2,006 lines total for five ops + read API; every module has a design-rationale header;
  one merge primitive reused everywhere (BlockSource); SpMV trait cleanly splits row/tiled kernels;
  docs (README/PROGRESS/MEMORY_CALIBRATION) are current and honest; repo history is clean.
- **The gap: no regression tests.** 3 unit tests (view only). Every "pixel-exact/1e-6" claim above
  was verified by ad-hoc scripts that are not checked in — nothing prevents a refactor from
  silently breaking cload or balance. Highest-leverage maintainability investment:
  (a) Rust unit tests for the codecs (shuffle4/8, delta, enc/dec_count round-trips), merge
      drain-and-count, zoomify RowAgg, cload line parsing;
  (b) a checked-in e2e test: synthetic pairs → cload → merge → zoomify → balance → expected,
      asserted against golden values + optional python-cooler cross-check (seed: `rooler
      test-write` already exists).
- `unsafe` is confined to the two SpMV hot loops + unshuffle4, each with a safety comment —
  acceptable per the "unsafe kernels are a valid strategy" decision.

## Next steps (proposed order)

1. **Correctness batch** (small, closes silent-corruption risks): overflow-checked count casts +
   `--count-dtype i64`; merge input validation; cload unknown-chrom error; strict `Comp::parse`;
   yeast CEN table; release-mode append assert. ~a day.
2. **Regression test suite** (protect everything before more perf work): codec unit tests +
   synthetic e2e with golden outputs. ~a day.
3. **Port the parallel gzip writer into CoolWriter** (`../GZIP_PARALLEL_FINDINGS.md`: 8.6× vs
   serial, byte-compatible, h5py-verified prototype). H5Dwrite_chunk via hdf5-metno-sys +
   libdeflate-6 (L4 default / L3 fast), rayon compress → single writer thread. This attacks the
   #1 remaining bottleneck: merge and cload phase B are both WRITE-bound. Generalize to blosc
   too (same direct-chunk path) so every preset gets a parallel writer. Biggest user-visible win;
   also makes gzip-compat files (readable by stock cooler, no hdf5plugin) fast enough to default.
4. **cload phase-B ranged-parallel merge**: add per-block first-key to the spill format (trivial
   header field) → binary-search runs into key ranges → reuse merge_coolers_parallel's pattern.
   With #3, cload should approach parse speed (~95 Mpairs/s end-to-end).
5. **Distiller parity**: `zoomify --balance` (balance every level), `--nproc/--chunksize→--mem`
   shim, custom BED views. Then run the full distiller chain swap-in test.
6. **300B-pair synthetic stretch test** (proves the 64GB/300B claim; also exercises #1's overflow
   path for real). Needs a synthetic pair generator.
7. Balance leftovers (optional): per-tile CSR for bin1 (drop a decode stream from tiled build),
   tile-size B sweep, persistent rayon pool.
