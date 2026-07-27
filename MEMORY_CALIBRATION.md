# rooler memory calibration (measured 2026-07-27)
Peak RSS by op (VmHWM), from real runs:

| op | data | --mem | peak RSS | model |
|---|---|---|---|---|
| cload | 2.61B pairs @256bp | 8 | 16.0 GB* | ~2x --mem (phaseA buffers + phaseB blocks, glibc retains) |
| merge | 2 coolers @262kb | 2 | 3.0 GB | --mem + ~1 GB (K readers + emit + O(nbins)) |
| balance | 1.12B pix, 64K bins | (auto) | 1.24 GB | compressed scratch (~1-2.6 B/pix * nnz) + ~O(nbins) vectors |
| zoomify | 62M pix | (auto) | 0.23 GB | streaming + coarsen map (fine_nbins i64) |
| expected | 1.12B pix | (auto) | 0.11 GB | per-region sum arrays (~nbins) + FFT buffers |
*cload phase-B block now capped -> RSS ~= --mem + ~1-2 GB after fix.

## Rules of thumb
- cload/merge peak RSS ~= --mem + O(nbins)-ish overhead (post phase-B cap). Set --mem = RAM_budget * ~0.5.
- balance peak RSS ~= compressed_scratch = ~2 B/pixel * nnz  (megacooler 26B pix -> ~55 GB, fits 64 GB).
  NO --mem for balance (scratch sizes to data); balance a huge cooler needs RAM >= ~2 B/pix * nnz.
- zoomify/expected: small, O(nbins). Safe anywhere.

## cooler --chunksize/--nproc -> --mem shim (for distiller drop-in)
cooler holds ~chunksize pixel-rows per proc as pandas frames (~40 B/row). So:
    rooler --mem_GB  ~=  chunksize * nproc * 40e-9      (matches cooler's footprint)
e.g. cooler --chunksize 10_000_000 --nproc 8  ->  --mem ~= 3.2 GB.
(Constant ~40 B/row is approximate; recalibrate against a cooler run if exactness needed.)

## Defaults set
cload/merge --mem default = 4 GB (RSS ~5-6 GB). balance --threads 8. All safe on a 32/64 GB box,
incl. distiller's ~5 parallel merges (5 x ~5 GB = 25 GB < 32).

## Large-scale test results (megacooler: 26.3B pixels, 256bp, 12.5M bins)
- **balance**: OK, 11.37M/12.54M bins weighted (90.7%), converged. Peak RSS 70 GB (scratch ~52GB +
  single-thread-build overhead). ~30 min: dominated by (a) single-thread scratch build [now parallelized],
  (b) IC SpMV hitting the 12M-bin cache-miss wall [tiled kernel = the next perf lever, coolerx showed 2.5x].
- **merge (10 coolers, ranged-parallel, --threads 8)**: EXACT, peak RSS 8.33 GB (= --mem 8). ~18 min vs
  28 min single-thread = only ~1.5x. FINDING: at scale merge is WRITE-BOUND (single-thread blosc compress of
  the 26B-pixel output) + hdf5 crate serializes reads via a global lock. Ranged-parallel helps read+merge, not
  the serial write. REAL LEVER: parallelize the WRITE (set BLOSC_NTHREADS / blosc filter nthreads, or a
  multi-threaded writer). ~125% CPU observed confirms write/read-lock bound, not merge-compute bound.

## Compressed spill (cload) — done
cload phase-A spills sorted key runs as delta + byte-shuffle(8) + LZ4 blocks (1M keys/block).
Measured ~1.76 B/key vs 8.0 raw = 4.5x less spill IO. Validated exact. Makes the ~300B-pair @64bp
run disk-feasible (2.4 TB raw spill -> ~529 GB compressed). Parse throughput unchanged (84 Mpairs/s;
compress overlaps spill). Phase-B RunReader decompresses blocks (1M keys/reader in flight -> bounded).

## Tiled (cache-blocked) SpMV — result (256bp monster: 2.56B pix, 12.5M bins)
Uniform 2D tiling (block B), tile (I,J) block-local bin1/bin2, cache-blocked SpMV. Bit-IDENTICAL to
row-chunk (2.55e-16) and cooler. Speed (B=65536, --threads 8):
  row-chunk: build 17s, mask 38s, IC 21s/iter, total 201s
  tiled:     build 42s, mask 25s, IC 14.6s/iter, total 169s
=> ~1.4x SpMV, 1.5x marginals, 1.2x overall. LESS than the coolerx C prototype's 2.5x, because
(a) the extra per-pixel bin1-local decode stream adds decode cost offsetting the cache gain,
(b) tiled BUILD is slower (counting-sort + 2 shuffle-encodes/tile: 42 vs 17s).
LEVERS to close the gap: (1) store bin1 as per-tile CSR (rowptr) instead of per-pixel u32 stream ->
removes a whole decode stream; (2) B sweep (32k/65k/131k); (3) persistent rayon pool (currently a new
ThreadPool per spmv/marginals call). CLI: `rooler balance ... --block 65536`. Row-chunk is still default.
