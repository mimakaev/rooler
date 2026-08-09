# Benchmarks

Every comparison below ran the reference tool on the *same input*, on the *same machine*, in the
same session. Where a comparison would have taken hours of reference-tool time it is reported as
rooler-only and labelled as such, never extrapolated.

**Machine.** Intel i9-13900KF (8 P-cores + 16 E-cores), 125 GB DDR5, NVMe SSD. Linux 6.8,
libhdf5 1.10.10, `cooler` 0.10.4, `cooltools` 0.7.1, Python 3.12.
rooler runs use `--threads 8` (its kernels saturate the 8 performance cores; memory-bound work
gains little from the E-cores); cooler runs use `-p 8`.

Reproduce with `scripts/bench_2p5b.sh` and `scripts/bench_vs_cooler.sh`.

---

## At 2.5 billion pixels

The size where waiting for the reference implementation stops being reasonable, and the
comparison that matters most. One cooler throughout — **2,563,532,077 pixels over 12,537,161
bins at 256 bp**, from an ENCODE hg38 micro-C file — with both tools at their defaults.

| op | cooler 0.10.4 | rooler | speedup |
|---|---|---|---|
| `cload` 2.61 B pairs → 256 bp | 3294 s (55 min) | **110 s** | **30×** |
| `balance` genome-wide IC, 12.5 M bins | 747 s | **89 s** | **8.4×** |
| `coarsen` 256 → 512 → 1024 → 2048 | 1932 s | **334 s** | **5.8×** |

Outputs are smaller as well as faster: 5.83 GB vs 6.03 GB for the cooler, 19.80 GB vs 20.54 GB
for the mcool.

Two notes so the balance row is read correctly. It uses `--no-expected`, because cooler's
`balance` does not compute expected while rooler's default does; with expected included rooler
takes 136 s — still 5.5× faster while doing strictly more work. And 57–67 s of those 89 s is
building the compressed matrix, a cost paid once (see *Where the time goes*).

## At 50 million pairs

The same ops on a 50 M-pair subset at 10 kb, for a sense of small-file behaviour.

| op | cooler 0.10.4 | rooler | speedup |
|---|---|---|---|
| `cload` 50 M pairs → 10 kb | 32.0 s | **1.1 s** | **29×** |
| `balance` | 18.9 s | **1.7 s** | **11×** |
| `coarsen` 5 resolutions | 13.4 s | **2.3 s** | **5.8×** |
| output — 10 kb cooler | 32.6 MB | **31.5 MB** | |
| output — 5-level mcool | 68.6 MB | **61.6 MB** | |

---

## Correctness

Speed only matters if the answers match.

| op | check | result |
|---|---|---|
| `cload` | pixel table vs `cooler cload` | **byte-identical** — all 2,563,532,077 pixels, and again at 10 kb |
| `merge` | pixel table vs reference | exact |
| `coarsen` | vs `cooler.coarsen_cooler` | exact; counts conserved at every level |
| `balance` | vs `cooler.balance_cooler` | identical bin mask (0 disagreements); weights agree to **2.5e-6** at matched tolerance |
| `expected` | vs `cooltools.expected_cis`, all columns including the log-smoothed ones | **2.7e-15** |
| tables | `pixels`, `pixels(join=True)`, `bins`, `chroms` vs cooler's accessors | identical frames |

Two differences are deliberate rather than error:

- **Balance stopping rule.** At default settings the weights differ by ~2e-4 because the tools
  stop at different points — rooler uses a scale-free criterion (`CV = std/mean < 1e-4`), cooler
  an absolute variance threshold. At matched tolerance they agree to 2.5e-6.
- **Bins straddling a region boundary.** When an arm boundary falls inside a bin (possible at
  some resolutions, never for a whole-chromosome view), cooltools counts that bin in `n_valid`
  but omits it from the sums; rooler includes it in both. Brute force over the pixel table gives
  rooler's value. Affects ~4% of rows by ~0.2% for an arms view.

---

## Scale

rooler-only: the reference pipeline does not finish these in reasonable time.

**40 billion pairs, real data.** Ten ENCODE hg38 micro-C files, `cload` at 256 bp then `merge`:
a **26.3 billion-pixel**, 53 GB cooler in ~63 min, balanced at 70 GB peak RSS.

**100 billion pairs, synthetic, at 64 bp.** As much about memory as speed — it builds a matrix
far larger than the machine has RAM.

| stage | result |
|---|---|
| input | **100,000,000,000 pairs**, streamed, never stored as text |
| `cload` | **81,477,686,796 pixels** over **48,254,229 bins** at **64 bp** — 176 GB — in 1 h 55 m |
| — peak RSS | **29.8 GB** (`--mem 32`) |
| `coarsen` | 5 levels (64 → 1024 bp), 622 GB mcool, in 4 h 11 m |
| — peak RSS | **0.8 GB** |

The 176 GB cooler was built in 29.8 GB — an output ~6× the memory that produced it — and the
622 GB mcool in 0.8 GB, ~780× the memory that produced it, because coarsening's working set is
the fine→coarse bin map and a per-bin counter, never the matrix.

Verified afterwards: metadata consistent at all five levels; counts conserved **exactly** across
the whole cascade (484,364,054 over a 247-million-pixel chr1 slice, identical at 64, 128, 256,
512 and 1024 bp); `cooler` opens the result and fetches correct symmetric blocks.

Pixel counts fall slowly when coarsening this deep a map — 81.5 B → 74.0 B → 66.3 B → 58.5 B →
50.8 B — because the matrix is sparse enough that most pixels stay distinct. That is a property
of the data, and it means each coarser level costs nearly as much as the last.

> 300 billion pairs was not attempted: it needs ~1.18 TB on a 1.20 TB volume. A disk limit on
> this machine, not an architectural one — the pipeline is linear in pairs.

---

## Compression and I/O

rooler writes gzip by default, which is only reasonable because both directions are parallel.
HDF5 stores each chunk as an independent stream, so rooler packs and unpacks chunks on worker
threads and exchanges finished bytes with the direct-chunk API. The files stay ordinary
shuffle+deflate, readable anywhere with no filter plugins.

Rewriting a 1.1 B-pixel cooler:

| preset | write | output |
|---|---|---|
| gzip, single-threaded (the usual HDF5 path) | 100 s | 820.9 MB |
| **gzip, rooler** | **26 s** | **786.1 MB** |
| blosc:zstd:1 | 26 s | 1227.5 MB |
| no compression | 24 s | 22.4 GB |

Reading the same table back:

| preset | read | notes |
|---|---|---|
| no compression | 3.77 GB/s (188 Mpix/s) | HDF5's own overhead floor |
| **gzip, parallel chunks** | **3.66 GB/s (183 Mpix/s)** | at that floor — compression is free |
| blosc:zstd:1 | 2.42 GB/s (121 Mpix/s) | |
| gzip, serial (HDF5's own path) | 0.97 GB/s (49 Mpix/s) | what the parallel reader replaces |

gzip is **3.8× faster to write** than the standard path and **3.8× faster to read**, landing
within 0.3% of uncompressed throughput while producing files a third smaller than blosc and
readable by every HDF5 install. Compression level does not measurably affect read speed.

Python-side, reading a 40 M-pixel slice: **0.88 s** into polars (rooler's default), 1.12 s into
pandas, 1.20 s through cooler.

---

## Where the time goes

- **`cload`** (2.61 B pairs → 2.56 B pixels at 256 bp, 110 s): ~35 s parsing, binning and
  spilling; the rest in the k-way merge.
- **`coarsen`**: the streaming read and write of each level.
- **`balance`** (89 s): 57–67 s of it builds the compressed matrix, the rest is the iteration.
  That build still goes through the serial HDF5 read path rather than the parallel chunk reader
  — the clearest remaining win. It is a one-time cost, so re-balancing the same cooler at a
  different tolerance is far cheaper than the first run. Above 4 M bins a cache-blocked kernel
  engages automatically, worth ~2.8× on the iteration itself.
- **`expected`**: one O(nnz) pass plus an FFT — 47 s on the 2.5 B-pixel cooler.

The open problem is balancing at 100-billion-pixel scale: it is the one op that holds a
compressed matrix instead of streaming, so it is bounded by `--mem` and spills to a disk-backed
map beyond that.
