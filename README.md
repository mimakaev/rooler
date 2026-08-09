# rooler

**A fast, out-of-core engine for Hi-C / micro-C `.cool` files.**
`cload` · `merge` · `zoomify` · `balance` · `expected` — plus a small Python read API.

> **Alpha.** The ops are validated against `cooler`/`cooltools` and have been run on
> hundred-billion-pair datasets, but this is a young project. Interfaces may still move, and
> you should spot-check results against `cooler` on your own data before trusting it.

rooler is a reimplementation of the heavy part of the [cooler](https://github.com/open2c/cooler)
CLI — the distiller-style pipeline `pairs → cload → merge → zoomify → balance` — aimed at the
scale modern micro-C has reached: **tens to hundreds of billions of contacts, at 64–256 bp
resolution, on a normal workstation**.

It writes ordinary **cooler files**. `cooler`, `cooltools`, HiGlass and everything else in that
ecosystem read rooler output directly, with no plugins and no conversion step.

## Why

Deep micro-C broke the assumptions the original tools were built on. A 40-billion-pair dataset
at 256 bp is not a bigger version of a 2-billion-pair dataset at 5 kb — it stops fitting in
memory, and the parts of the pipeline that used to be "fast enough" become overnight jobs.

rooler's answer is engineering every layer of the pipeline for that scale:

- **Truly streaming, explicitly algorithmic ops.** Each operation is implemented as the
  algorithm it claims to be — merge is a k-way merge, coarsen is a streaming accumulation —
  not a chunk-sort in disguise. Rust kernels work in place, without the memcopies that
  numpy-based chunking cannot avoid. Pixels are processed as they stream past, so `--mem` is a
  real budget rather than a hint, and no op ever holds the matrix.
- **Compact custom intermediates.** Spill runs and scratch use purpose-built compressed binary
  formats (delta + byte-shuffle + LZ4), ~4× smaller than raw and fast enough to decode inside
  the compute loop.
- **In-memory compression for balance.** The matrix iterates from a compressed in-RAM form at
  ~2 bytes per pixel — and past `--mem` it moves to a disk-backed memory map, same results.
- **A parallel gzip writer.** Each HDF5 chunk is shuffle+deflate-packed by hand on worker
  threads and handed to the direct-chunk API: 2–3 GB/s of standard, plugin-free gzip output.
- **Cache-blocked kernels.** When the bin table outgrows the CPU cache (fine resolutions), a
  2D-tiled SpMV keeps the hot vectors cache-resident — worth ~2.8× at 12 M bins.

The largest run so far took **100 billion pairs to an 81-billion-pixel, 48-million-bin cooler
at 64 bp in under two hours**, and cascaded it into a five-level 622 GB `.mcool` in another
four. For reference, that single cooler is roughly thirty times the size of a typical deep
Hi-C map. It also ran in a **29.8 GB peak of RAM**, and the coarsening cascade in under 1 GB —
memory is bounded by `--mem`, not by the data.

Against `cooler` 0.10.4 on the same machine and inputs (details in
[BENCHMARKS.md](BENCHMARKS.md)):

| op | cooler 0.10.4 | rooler | speedup |
|---|---|---|---|
| `cload` — 2.61 B pairs → 256 bp | 3258 s | **114 s** | **29×** |
| `balance` — 2.5 B pixels, 12.5 M bins | 670 s | **51 s** | **13×** |
| `coarsen` — 2.5 B pixels, 3 levels | 1955 s | **460 s** | **4.3×** |
| `cload` — 50 M pairs → 10 kb | 31.4 s | **1.2 s** | **26×** |

## Install

```bash
# engine (needs: rust, libhdf5-dev, pkg-config; bgzip/htslib at runtime for .pairs.gz input)
cargo install --git https://github.com/mimakaev/rooler

# python read API (optional)
pip install "git+https://github.com/mimakaev/rooler#subdirectory=python"
```

## Quickstart

```bash
# pairs -> cooler at 1 kb
rooler cload sample.pairs.gz 1000 sample.cool --assembly hg38

# combine replicates
rooler merge merged.cool rep1.cool rep2.cool rep3.cool

# multi-resolution mcool; --balance balances every level AND stores expected P(s) for each
rooler zoomify merged.cool merged.mcool --balance

# or balance one resolution (expected is computed automatically; --no-expected opts out)
rooler balance merged.mcool::resolutions/1000

# adopt a cooler made elsewhere: recompress to plugin-free gzip, stamp + verify the assembly,
# balance it if it has no weights, store expected. In place; --backup keeps <file>.bac
rooler repack old.mcool --backup --assembly hg38
```

```python
import rooler

with rooler.open("merged.mcool", 1000) as r:
    r.raw("chr1:5,000,000-6,000,000")    # dense raw counts
    r.balanced("chr1", "chr2")           # balanced, trans
    r.ooe("chr1_p")                      # observed / expected, cis
    r.expected()                         # P(s) table, smoothed by default
    r.matrix(balance=True).fetch("chr17")  # cooler-compatible form
```

**Keep the handle open.** Opening a `Rooler` reads and caches everything a fetch needs — chrom
names and lengths, chrom offsets, the whole `bin1_offset` index — and lazily caches the
balancing weights and the expected table on first use. Re-opening per fetch re-reads all of
that and discards the caches. Open once, hold it, fetch many times.

`ooe()` divides balanced counts by the stored expected at each cell's genomic separation. Both
sides must sit inside one region of the expected view — a fetch crossing an arm or chromosome
boundary, or a trans fetch, **raises** rather than quietly returning NaN, because no single
P(s) applies to it. It defaults to the smoothed genome-wide curve; `column=` picks another.

Because the output is a real cooler, this also just works:

```python
import cooler, cooltools
clr = cooler.Cooler("merged.mcool::resolutions/1000")
cooltools.expected_cis(clr)
```

## CLI

```
rooler cload   <pairs[.gz]|-> <binsize> <out.cool>  [--mem 4] [--threads 8] [--assembly hg38]
rooler merge   <out.cool> <in1.cool> <in2.cool> ...  [--mem 4] [--res R] [--assembly hg38]
rooler zoomify <base.cool> <out.mcool>  [--resolutions a,b,c] [--balance] [--threads 8]
rooler balance <cool[::resolutions/R]>  [--ignore-diags 2] [--mad-max 5] [--min-nnz 10]
                                        [--tol 1e-4] [--threads 8] [--mem 8] [--no-expected]
rooler expected <cool[::resolutions/R]> [--view chroms|arms|custom:<bed>]
rooler repack  <cool|mcool>  [--out new.cool | --backup] [--assembly hg38] [--mem 8]
```

- **cload** — reads bgzipped `.pairs`, plain text, or `-` for stdin.
- **merge** — refuses inputs whose bin layouts disagree, rather than producing quiet garbage.
- **zoomify** — `--balance` balances every resolution as it goes and stores expected for each.
- **balance** — genome-wide iterative correction, followed by expected by default. The
  cache-blocked kernel (worth ~2.8× on fine-resolution coolers) engages automatically above
  4 M bins; `--block` overrides, `--mem` bounds RAM (scratch spills to a disk-backed mmap
  beyond it).
- **expected** — cis distance-decay P(s) per region, stored in the cooler with the full
  cooltools column set including the **log-smoothed** curves. Views can be `chroms`, `arms`,
  or your own BED; several views coexist in one file.
- **repack** — rewrite an existing cooler/mcool the way rooler would have written it:
  parallel-gzip compression, assembly stamped **and checked against the chromosome sizes**,
  balanced if it carries no weights, expected stored. In place by default (`--backup` keeps
  the original), or to a new path with `--out`.

Coming from cooler: `--nproc` works as an alias for `--threads`, and `--chunksize` maps onto
the `--mem` budget.

### Opinionated choices

**No mystery coolers.** rooler refuses to write a cooler without a genome assembly. It will take
`--assembly`, or infer one from the chromsizes.

**Expected comes built in.** In practice, people compute cis expected with cooltools at default
parameters over chromosome arms — and wait hours-to-days for what is one O(nnz) pass. rooler
computes it **by default whenever weights are written** (`balance`, `zoomify --balance`,
`repack`), with a per-organism default view: arms where arms are meaningful (human, yeast),
whole chromosomes where they are not (mouse, fly, worm).

**Counts are `int32`.** Internal accumulators are 64-bit, but stored counts saturate at
2,147,483,647 with a loud warning telling you how many pixels were affected — a value unlikely
to represent a true pixel of a Hi-C map.

## Compatibility

rooler writes **gzip-compressed coolers by default** — the same shuffle+deflate pipeline cooler
itself uses, so every HDF5 reader on earth opens them with no filter plugins and no conversion.

Defaulting to gzip only became reasonable because of the writer. HDF5 deflates each chunk on a
single thread, which made gzip the slow option; rooler packs chunks on worker threads and hands
the finished bytes to the direct-chunk API, reaching 2–3 GB/s and making gzip **3.8× faster than
the standard path** while producing ordinary, plugin-free files. Writing is no longer the
bottleneck.

**Reading still is, and there is no fast reader yet.** Streaming a pixel table back through
HDF5's own filter pipeline runs at about **1 GB/s** on this machine, against a **3.8 GB/s**
ceiling for the same data uncompressed — so decompression costs roughly three quarters of the
read. blosc sits in between at 2.4 GB/s:

| pixel-table read (1.12 B pixels, single thread) | throughput |
|---|---|
| uncompressed (HDF5 overhead only) | 3.77 GB/s |
| blosc:zstd:1 | 2.39 GB/s |
| gzip (level 1 or 4 — no measurable difference) | ~0.95 GB/s |

Note this is well below raw inflate speed: HDF5 uses zlib rather than a modern deflate, adds an
un-shuffle pass, and charges its own per-chunk overhead. The cure is the mirror image of the
writer — read raw chunks and inflate them in parallel — and it is not written yet. Until it is,
read-heavy work is slower than it should be.

We first built this on **blosc**, which is faster in both directions. The problem is that a
blosc cooler is not really a cooler: reading it needs a filter plugin, so it fails in a plain
`h5py` or `cooler` install. Fixing that upstream is close to a one-line change plus a small
dependency — but even if it landed tomorrow, it would be years before enough installed coolers
had it. That is not a bet worth making for a file format whose whole value is that everyone can
read it, so gzip is the default and `--preset blosc:zstd:1` is there for private intermediates.
rooler reads blosc coolers written by other tools either way.

Validated against the reference implementations: `cload` output is **byte-identical** to
`cooler cload` (verified on all 2.56 billion pixels of a real micro-C file); `merge` and
`zoomify` are pixel-exact; `balance` picks the identical set of bins and its weights agree with
`cooler.balance_cooler` to **2.5e-6** at matched tolerance; every `expected` column matches
`cooltools.expected_cis` to **2.4e-15**.

`.pairs` coordinates are read as **1-based**, per the 4DN spec and cooler's default; pass
`--zero-based` for a file that genuinely is not.

`expected` stores the same columns `cooltools.expected_cis` returns — `n_total`, `n_valid`,
`count.sum/avg`, `balanced.sum/avg`, and the log-smoothed `balanced.avg.smoothed` (per region)
and `balanced.avg.smoothed.agg` (genome-wide), at cooltools' own smoothing defaults. A raw P(s)
is noisy at large separations, where few pixel pairs contribute, so the smoothed curve is what
analyses actually want: `r.expected()` and `r.ooe()` both default to it, exactly as
`cooltools.expected_cis` does, and `column=` selects another.

## Tests

```bash
cargo test --release      # ~0.6 s, no network, no fixture files, no python
```

The suite generates its own data and checks each op against an independent oracle — a
brute-force recomputation rather than a stored blessed answer. There is also
`scripts/validate_vs_cooler.py` for comparing two coolers, or a cooler against `cooler` itself,
at billion-pixel scale.

## Status and limits

Working: all five ops, the Python read API, assembly enforcement. Known limits:

- **`balance` respects a `--mem` budget (default 8 GB).** It builds a compressed matrix
  (~2.5 bytes per pixel) in RAM when it fits, and in a disk-backed memory map next to the
  cooler when it doesn't — identical results, and committed memory stays small either way
  (a 2.5-billion-pixel balance peaks at ~2.4 GB of anonymous RSS under an 8 GB budget). Far
  beyond RAM the cost is re-reading the scratch from NVMe each iteration, so very deep
  balances become disk-bandwidth-bound rather than impossible.
- Chromosomes are capped at 2.1 Gb (same limit as cooler).
- Iterative correction can plateau without converging on very sparse or disconnected matrices;
  it reports `converged=false` rather than pretending.

## Development

`docs/` holds the development record: `STATUS.md` (current state and review findings),
`PLAN.md` / `PLAN_LOG.md` (the work plan and every measurement taken), `PROGRESS.md` (build
log), `MEMORY_CALIBRATION.md` (`--mem` sizing data).

## Authorship

**rooler was written entirely by [Claude](https://claude.ai) (Anthropic), in collaboration with
Max Imakaev.** Every line of Rust and Python in this repository — the engine, the tests, the
benchmarks and this README — was produced by the model. The human contribution was direction:
choosing the problem, setting the architecture and the priorities, pushing back on bad ideas,
and deciding what "correct" and "fast enough" had to mean.

This is stated plainly because it is unusual, and because you should know it when judging the
code. It is also why the validation is emphasised so heavily: correctness here rests on
oracle-based tests and agreement with the reference implementations, not on an author's
authority.

## License

MIT — see [LICENSE](LICENSE).
