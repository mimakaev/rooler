# Benchmarks

All numbers below were measured on one machine, with the reference tool run on the *same input*
in the *same session*. Where a comparison would have taken hours of reference-tool time, it is
reported as rooler-only and labelled as such rather than extrapolated.

**Machine.** Intel i9-13900KF (8 P-cores + 16 E-cores, 32 threads), 125 GB DDR5, NVMe SSD.
Linux 6.8, libhdf5 1.10.10. `cooler` 0.10.4 / `cooltools`, Python 3.12.
All rooler runs use `--threads 8` (its kernels saturate the 8 performance cores; the E-cores add
little for memory-bound work). All cooler runs use `-p 8`.

Reproduce with `scripts/bench_vs_cooler.sh <pairs.gz> <workdir> 8`.

---

## Head to head vs cooler

Input: 49,999,640 real Hi-C pairs (an ENCODE hg38 micro-C file, subset), loaded at **10 kb**.
Every op runs on the identical file; balance and zoomify both start from the same cooler.

| op | cooler 0.10.4 | rooler | speedup |
|---|---|---|---|
| `cload` (50 M pairs → 10 kb) | 31.4 s | **1.2 s** | **26×** |
| `balance` (genome-wide IC) | 18.7 s | **1.2 s** | **16×** |
| `zoomify` (5 resolutions) | 13.6 s | **3.7 s** | **3.7×** |

Output sizes are equivalent — rooler is not buying speed by writing bigger files:

| file | cooler | rooler |
|---|---|---|
| 10 kb cooler | 32.64 MB | 32.73 MB (+0.3%) |
| 5-level mcool | 69.85 MB | **63.98 MB (−8.4%)** |

### Balance at a larger size

rooler decompresses the matrix once into a compact in-memory form and iterates over that, rather
than re-reading the pixel table every pass.

| cooler | bins | pixels | cooler 0.10.4 | rooler | speedup |
|---|---|---|---|---|---|
| 10 kb | 310 K | 33 M | 18.7 s | **1.2 s** | **16×** |
| 50 kb | 64 K | 1.12 B | 80.6 s | **12.0 s** | **6.7×** |

Both agree with cooler exactly on which bins to keep (0 mask disagreements at either size).

The speedup is *smaller* at the larger size, which is worth being straight about: 8 of rooler's
12 s there is the one-time cost of decompressing 1.12 B pixels into the in-memory form. The
iteration itself is ~4 s. That build cost is paid once and amortised across re-balances, but it
is real, and it means the headline ratio depends on how compressed your input is.

---

## At 2.5 billion pixels

This is the size where waiting for the reference implementation stops being reasonable, so it
is the comparison that matters most. Everything below runs on one cooler —
**2,563,532,430 pixels over 12,537,161 bins at 256 bp** (an ENCODE hg38 micro-C file) — with
both tools at their own defaults, i.e. what you would actually type.

| op | cooler 0.10.4 | rooler | speedup |
|---|---|---|---|
| `balance` (genome-wide IC, 12.5 M bins) | 670 s | **88 s** | **7.6×** |
| `balance` with `--block 65536` | 670 s | **51 s** | **13×** |
| `coarsen` 256 → 512 → 1024 → 2048 | 1955 s | **460 s** | **4.3×** |
| `cload` 2.61 B pairs → 256 bp | 3258 s (54 min) | **114 s** | **29×** |

**The results are the same results.** `cload` output is **byte-identical to cooler's** — all
2,563,532,077 pixels, every bin1/bin2/count value. All three balance runs select an identical
**11,239,212** good bins — zero mask disagreements — and the weights agree to a median of
**9.9e-6**. Every coarsened level matches cooler exactly, bin for bin and pixel for pixel:

| level | bins (rooler / cooler) | pixels (rooler / cooler) |
|---|---|---|
| 256 bp | 12,537,161 / 12,537,161 | 2,563,532,430 / 2,563,532,430 |
| 512 bp | 6,268,696 / 6,268,696 | 2,519,298,536 / 2,519,298,536 |
| 1024 bp | 3,134,464 / 3,134,464 | 2,459,691,793 / 2,459,691,793 |
| 2048 bp | 1,567,341 / 1,567,341 | 2,384,753,647 / 2,384,753,647 |

rooler's outputs are also smaller: 5.83 GB vs 6.03 GB for the cooler, 19.89 GB vs 21.49 GB for
the mcool (−7.4%).

*(The balance and coarsen inputs are one cooler of 2,563,532,430 pixels, built before the
1-based binning fix below; both tools were given that same file, so the comparison stands. The
`cload` row rebuilds from the pairs and lands on 2,563,532,077 — cooler's count exactly.)*

### Per iteration, the gap is much larger than the wall clock suggests

rooler pays a one-time cost to decompress the matrix into a compact in-memory form, then
iterates over that. Splitting the balance runs accordingly:

| | build | per IC iteration | iterations |
|---|---|---|---|
| cooler | — | **67 s** | 10 |
| rooler (default) | 18 s | **8.1 s** | 7 |
| rooler (`--block`) | 29 s | **2.7 s** | 7 |

So the *iteration* — the thing that dominates when you tighten tolerance, re-balance, or move
to a finer resolution — is **25× faster** with the cache-blocked kernel. The end-to-end 13× is
the honest number for a single run; 25× is what you feel on the second one.

---

## Correctness

Speed is only interesting if the answers match. They do.

| op | check | result |
|---|---|---|
| `cload` | pixel table vs **cooler** | **byte-identical** (all 2,563,532,077 pixels, and again at 10 kb) |
| `merge` | pixel table vs reference | **exact** |
| `zoomify` | vs `cooler.coarsen_cooler` | **exact**; counts conserved at every level |
| `balance` | vs `cooler.balance_cooler`, same tolerance | **0** mask disagreements; weights agree to **2.5e-6** (median; p99 3.4e-6) |
| `balance` | 1.12 B-pixel cooler, both at default tolerance | **0** mask disagreements; median **8.1e-6** (p99 4.5e-5) |
| `expected` | vs `cooltools.expected_cis`, all 13 columns incl. log-smoothed | **2.4e-15** — machine precision, 0 NaN mismatches |

### The benchmark found a bug

Running `cload` head to head at this size is what surfaced it: rooler's pixel count was 353
higher than cooler's, with total counts identical. `.pairs` coordinates are **1-based** (4DN
spec, and cooler's default), but rooler was binning them as `pos / binsize` instead of
`(pos-1) / binsize`. That shifts every read whose position is an exact multiple of the binsize
into the next bin — about 1 read-end in `binsize`, so 0.4% at 256 bp — and a read at the very
end of a chromosome spilled into the *next chromosome*.

It had gone unnoticed because the earlier "pixel-exact" checks compared rooler against rooler
(and against a prototype sharing the same convention). Comparing against cooler itself is what
caught it. Fixed, with `--zero-based` for files that genuinely are 0-based; output is now
byte-identical to cooler at both 10 kb and 256 bp.

At *default* settings the balance weights differ by ~2.4e-4, purely because the two tools stop
at different points: rooler's default is a scale-free criterion (`CV = std/mean < 1e-4`) rather
than cooler's absolute variance threshold. Tighten `--tol` and the two converge onto each other
at 2.5e-6, which is the number that says the algorithms agree.

---

## Scale

These are rooler-only: the reference pipeline does not finish these in a reasonable time.

### Real data — 40 billion pairs

Ten ENCODE hg38 micro-C files, `cload` each at 256 bp then `merge`:

| | |
|---|---|
| output | **26.3 billion pixels**, 53 GB, 256 bp |
| wall time | ~63 min end to end |
| balance | completed at 70 GB peak RSS |

### Synthetic — 100 billion pairs at 64 bp

The point of this run is memory, not speed. The machine has 125 GB of RAM; the finished
cooler is 176 GB and the full mcool 622 GB, and neither op ever held more than a small
fraction of that.

| stage | result |
|---|---|
| input | **100,000,000,000 pairs**, streamed (never stored as text) |
| `cload` | **81,477,686,796 pixels** over **48,254,229 bins** at **64 bp**, 176 GB, in 1 h 55 m |
| — peak RSS | **29.8 GB** (`--mem 32`) |
| `zoomify` | 5 levels (64 → 1024 bp), 622 GB mcool, in 4 h 11 m |
| — peak RSS | **0.8 GB** |

Read those RSS figures against the outputs. The 176 GB cooler was built in **29.8 GB** — an
output ~6× the memory that produced it. The 622 GB mcool was built in **0.8 GB**, an output
~780× the memory that produced it, because zoomify's working set is the fine→coarse bin map
plus a per-bin counter and never the matrix itself.

Verified afterwards: `nnz` metadata consistent at all five levels; counts conserved **exactly**
across the whole cascade (484,364,054 over a 247-million-pixel chr1 slice, identical at 64, 128,
256, 512 and 1024 bp); and `cooler` opens the result and fetches correct symmetric blocks.

Levels coarsen slowly at this depth — 81.5 B → 74.0 B → 66.3 B → 58.5 B → 50.8 B pixels — because
the matrix is sparse enough that most pixels stay distinct. That is a property of the data, and
it means each coarser level costs nearly as much as the last.

> 300 billion pairs was not attempted: it needs ~1.18 TB on a 1.20 TB volume. That is a disk
> limit on this machine, not an architectural one — the pipeline is linear in pairs.

---

## Compression: gzip is the default, and it is not the bottleneck

The usual argument for exotic codecs is that gzip is too slow. That is true of the *single-
threaded* gzip path HDF5 uses by default, and it is not true here: rooler compresses chunks in
parallel and hands HDF5 finished bytes, which is both faster and produces standard files.

Rewriting a 1.1-billion-pixel cooler:

| preset | wall | output |
|---|---|---|
| gzip, single-threaded (the usual path) | 100 s | 820.9 MB |
| **gzip, rooler (default)** | **26 s** | **786.1 MB** |
| blosc:zstd:1 | 26 s | 1227.5 MB |
| no compression (the floor) | 24 s | 22.4 GB |

gzip is now **3.8× faster than the standard gzip path** and produces a slightly smaller file. It
matches blosc on wall time while being **36% smaller**, and it sits within 8% of the
uncompressed floor — so the write path is no longer what limits the pipeline.

That is why rooler defaults to gzip: the files need no filter plugins, every HDF5 reader on
earth opens them, and compatibility costs nothing. (This is not hypothetical — while preparing
these benchmarks, `cooler balance` failed outright on a blosc-compressed cooler because the
worker processes could not find the blosc plugin. A gzip cooler simply works.)

---

## Where the time goes now

Useful if you plan to work on this.

- **`cload`** (2.61 B pairs → 2.56 B pixels @256 bp, 106 s total): 35 s parsing/binning/spilling,
  71 s in the k-way merge. The merge is now the cost, not the write.
- **`merge`**: within 8% of the uncompressed floor — effectively I/O and merge bound.
- **`balance`**: the iteration is memory-bandwidth bound. A cache-blocked kernel (`--block`) is
  worth **2.8×** on the largest coolers (0.35 → 0.98 Gpixel/s on a 2.5 B-pixel, 12.5 M-bin
  matrix), bringing total balance time there from 80 s to 48 s.
- **`expected`**: seconds — a single O(nnz) pass plus an FFT.

The open problem is balancing at 100-billion-pixel scale: it is the one op that holds the matrix
in memory rather than streaming it.
