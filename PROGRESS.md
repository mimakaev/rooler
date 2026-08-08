# rooler — Rust cooler engine, build progress

> Historical build log. For current state, code-review findings, and the roadmap see **STATUS.md**.

Host: `/ssdhome/magus/work/cooler-redo/rooler/`. Toolchain: rust 1.97, libhdf5 1.10.10 (`hdf5-metno` crate).
Build: `source ~/.cargo/env && cargo build --release`. Binary: `target/release/rooler`.

## Design (agreed)
- ONE merge primitive: heap-based **drain-and-count k-way merge** of sorted (key,count) block sources.
  key = bin1*nbins+bin2 (i64, safe to ~3e9 bins). Draining all equal keys (cross- and within-stream)
  yields an aggregated sorted stream. Bounded RAM = K·block + emit buffer, set by `--mem`.
- **merge** = merge over cooler pixel readers directly (inputs pre-sorted; no sort, no spill).
- **cload** = phase A: parse pairs (Rust), bin, sort `--mem`-sized chunks of bare keys, spill compressed
  sorted runs; phase B: same k-way merge over run-readers (count=1 → drain-and-count aggregates). TODO.
- counts i32 (v1), accumulate i64; f32 later. `--nproc`: single-thread merge core + prefetch/writer threads.
- `--ram`/`--mem` is primary knob; map cooler `--chunksize`/`--nproc` → mem for drop-in. TODO.

## Status
- [x] crate scaffold; hdf5 links; reads gzip coolers natively.
- [x] `cooler::CoolWriter` — streaming cooler-v3 writer (append blocks, bounded RAM). **Round-trips
      through Python cooler** (pixels/matrix/fetch/chromnames all correct). gzip preset; chrom as int32
      (cooler maps to category via chrom_offset — enum fidelity is a TODO).
- [x] `cooler::CoolerPix` — lazy block reader (key,count).
- [x] `merge::merge_sources` (heap drain-and-count) + `merge_coolers`. CLI `rooler merge`.
      **VALIDATED**: tiny synthetic exact; real heart+vessels @262144 (62M+60M→62M pixels) **pixel-exact
      vs coolerx merge**, sum 41,705,798,220, out-of-core at --mem 0.05 (1.5M blocks), 9.1s.
- [x] `cload` — Rust pairs parser (shells `bgzip -@` for bgzf) + fast tab/atoi + chrom-cache; phase-A
      sort `--mem` chunks of bare keys, spill raw sorted runs; phase-B RunReader (count=1) → drain-and-count
      merge → writer. CLI `rooler cload`. **VALIDATED**: 5M-pair subset @100kb → 248,914 pixels **pixel-exact
      vs coolerx/numpy ref**, out-of-core at --mem 0.02 (2 runs). Full file ENCFF514KZU @10kb: 2.61B pairs
      -> 2.093B pixels (EXACT), --mem 8 = 3 runs; parse 17.5 Mpairs/s (1 thread, beats polars 14), total 354s.
- [x] **blosc:zstd:1 write** — via hdf5-metno `blosc` feature + forcing `blosc-src` features `["lz4","zstd"]`
      (default blosc-src bundles ONLY blosclz; lz4/zstd silently no-op'd at ratio 1.0 without this). Default preset.
- [x] **parallel-parse cload** — bgzip -@ decompress; producer splits stream into line-aligned blocks over a
      bounded crossbeam channel; N worker threads each parse (hand tab/atoi + chrom-cache) + bin + sort_unstable
      + spill their own raw-i64 runs. RAM = N×(mem/N). Phase B = single-thread heap merge (ranged-parallel shim
      in place, TODO). VALIDATED exact. **ENCFF514KZU @256bp: 2.61B pairs -> 2.56B pixels, 145s** (parse 27s =
      **95 Mpairs/s, 5.4x** the 1-thread 17.5; merge+blosc-write 118s), output **6.7GB (2.6 B/pix, 7.7x)**.
- [~] Megacooler test RUNNING (megacooler.sh/.log): cload all 10 ENCODE files @256bp (--mem 8) then merge into
      one. Confirms the single-thread merge (~100s/2.6B) is now the bottleneck -> motivates ranged-parallel merge.
- [ ] ranged-parallel merge: partition bin1 into P ranges (coolers slice via bin1_offset; runs via binary
      search on key), merge ranges in parallel. Under --mem this needs temp-spill or bounded-overlap (HDF5
      parallel writes to one file unsafe). NEXT after megacooler baseline.
- [x] **MEGACOOLER built** (real-world test): cload all 10 ENCODE hg38 files @256bp (--mem 8) + merge ->
      **26.3B-pixel cooler, 53GB, ~63 min total**. Merge alone = **1678s (28 min) single-thread** = confirmed
      bottleneck (motivates ranged-parallel merge, ~8x). Reads fine via API (region fetch 0.4s).
- [x] **Python read API** (`python/rooler/__init__.py`): `rooler.open(uri, res)`; **`r.raw("chr1:..")` /
      `r.balanced("chr1:..")`** direct dense fetch (region1[, region2]); `r.raw()[a:b,c:d]` slicer.
      cooler-compat surface for cooltools: `matrix(balance=).fetch()/[...]`, `bins()[:]/.fetch()/['weight']/.columns`,
      `pixels()[lo:hi]`, `chroms()`, `chromsizes`, `binsize`, `info`, `shape`, `extent()`, `offset()`.
      **VALIDATED vs cooler**: raw cis/trans/cross-chrom exact; balanced allclose. Symmetric-upper expand +
      diagonal double-count handled. hdf5plugin auto-imported for blosc read.
- [x] **cooltools works natively on rooler output** via `cooler.Cooler(rooler_file)` (files are valid cooler
      format): `cooltools.expected_cis` runs. So cooltools-compat = the FILE FORMAT; rooler read obj is the
      ergonomic layer, not a 100% cooler-internal drop-in (avoid that rabbit hole).
- [x] **zoomify** (`zoomify.rs`): CoolWriter generalized to `create_in(file, group, ...)` so it writes into
      `resolutions/{res}`. Streaming integer-factor coarsen via a fine->coarse bin map (respects chrom
      boundaries); coarse bin1 non-decreasing -> accumulate one coarse row (sort bin2 + sum) + emit. Cascade
      level-to-level. CLI `rooler zoomify src.cool out.mcool [--resolutions a,b,c]` (default: double until coarse).
      **VALIDATED**: hv_rust 262144 -> 9-level mcool in 4s; count conserved at every level; 524288 level
      **pixel-EXACT vs cooler.coarsen_cooler**; cooler opens the mcool.
- [x] **NO MYSTERY COOLERS**: cload/merge/zoomify REFUSE to write without a genome assembly. Resolution:
      --assembly override > input's stored assembly > chr1-length fingerprint (view::detect) > refuse. Stamped
      into genome-assembly attr. cload auto-detects hg38 from ENCODE chromsizes. read_meta now returns assembly.
- [x] **balance** (`balance.rs`): genome-wide IC, streaming out-of-core (re-streams pixel table per iteration,
      RAM=O(nbins)). Mask (min_nnz + per-chrom MAD on log-marginal) + IC iterations + multiplicative weight
      /sqrt(scale) + cooler-compatible weight attrs. **VALIDATED vs cooler.balance_cooler** (hv_rust): scale
      identical, 0 mask disagreements, weight median rel diff 8.4e-7 (p99 1.6e-5), balanced matrix allclose.
      CLI `rooler balance uri [--ignore-diags/--mad-max/--min-nnz/--tol/--max-iters]`.
- [~] full distiller chain e2e (cload->zoomify->balance) RUNNING (e2e.sh/.log).

## NONTRIVIAL DECISIONS (for review)
1. No mystery coolers: refuse without assembly (detect/override/fingerprint). Opinionated, per user.
2. balance v1 = streaming IC over the cooler pixel table (re-decompress/iter), NOT compressed-scratch/tiled
   kernel. Bounded RAM, correct, bit-close; single-thread. PERF FOLLOWUP: compressed scratch + parallel SpMV
   (coolerx showed tiled cache-blocking = 2.5x). At megacooler scale (26B pix x ~15 iters) this is slow -> that's
   the next perf port.
3. IC only (cooler default); KR deferred. 4. Genome-wide (cis_only=false) default. 5. Defaults match cooler.
6. bins/chrom still plain int32 (cooler maps via chrom_offset); enum fidelity deferred.
- [x] **view module** (`view.rs`): genome->regions for expected, NO bioframe dep. ViewKind Arms/Chroms;
      opinionated defaults (hg38/hg19/saccer3=arms, mm10/mm39/dm6/ce11=chroms); detect by assembly name +
      chr1-length fingerprint (our writers stamp assembly="unknown"); unknown genome REFUSES unless --view
      chroms|arms|custom. hg38/hg19 centromere midpoints baked (APPROX, fine for per-arm expected;
      regenerate from UCSC cytoBand via the python snippet when net available). 3 unit tests pass.
      TODO: saccer3/other arm centromere tables (currently arms-default genomes w/o table error); custom BED
      parsing; cload should capture real assembly from pairs header instead of "unknown".
- [ ] expected: fold sum_balanced[region][dist] into balance pass (weights just computed); n_valid via FFT
      mask-autocorr per arm (rustfft); store under resolutions/{res}/expected/{view}/{weight}; view (arms via
      bioframe from genome-assembly) under .../views/. Read API r.expected(). Fast standalone `rooler expected`.
- [ ] CLI decision: Rust binary for ops (cload/merge/zoomify/balance); Python for read API. Rust CLI = distiller
      drop-in (shell `rooler cload ...`). --chunksize/--nproc -> --mem shim.
- [ ] ranged-parallel merge (parked); shuffle+lz4 spill; noodles-bgzf.
- [ ] blosc write (via hdf5plugin filter) + lzf read (build h5py lzf plugin) for real-world coolers.
- [ ] loser-tree (heap fine to ~thousands-way); cooler-flag compat shim; --nproc pipeline threads.

## Notes
- Reading blosc coolers from Rust: set HDF5_PLUGIN_PATH=<venv>/site-packages/hdf5plugin/plugins.
- T2 coolers are lzf-compressed → need an lzf HDF5 plugin to read from Rust (not yet built).
