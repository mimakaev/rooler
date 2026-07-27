# 8-hour autonomous run (2026-07-27)
1. [x] balance perf port: compressed CSR scratch (lz4+shuffle, u8+exc) + rayon SpMV. VALIDATED vs cooler
       (6.1e-6 median, 0 mask disagree); 4s vs 19s streaming on hv_rust (~5x).
2. [x] expected: streaming sum_balanced + FFT n_valid, stored in-cooler + view. VALIDATED vs cooltools
       (balanced.avg median rel 6.4e-16 = machine eps!); 10s whole-genome (cooltools = glacial).
3. [x] validate balance + expected -> both bit-close/exact.
4. [~] memory harvest -> chunksize/nproc->ram calibration + defaults.
5. [ ] large-scale zoomify tests on big coolers.
6. [ ] initial docs.
7. [ ] git init + commit.
8. [x] ranged-parallel merge: bin1-range partition, parallel range-merges streamed to writer in order via bounded channels (no temp). VALIDATED exact (h262+v262).
STRETCH (user): compressed spill, then synthetic ~300B-pair set (perturb existing pairs) -> cload/merge/zoomify @64bp on 64GB RAM.
