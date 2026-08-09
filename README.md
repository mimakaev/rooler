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
  numpy-based chunking cannot avoid. A `--mem` budget bounds RAM; nothing loads a pixel table.
- **Compact custom intermediates.** Spill runs and scratch use purpose-built compressed binary
  formats (delta + byte-shuffle + LZ4), ~4× smaller than raw and fast enough to decode inside
  the compute loop.
- **In-memory compression for balance.** The matrix iterates from a compressed in-RAM form at
  ~2 bytes per pixel — and past `--mem` it moves to a disk-backed memory map, same results.
- **A parallel gzip writer.** Each HDF5 chunk is shuffle+deflate-packed by hand on worker
  threads and handed to the direct-chunk API: 2–3 GB/s of standard, plugin-free gzip output.
- **Cache-blocked kernels.** When the bin table outgrows the CPU cache (fine resolutions), a
  2D-tiled SpMV keeps the hot vectors cache-resident — worth ~2.8× at 12 M bins.

The largest run so far: **100 billion pairs → an 81-billion-pixel, 48-million-bin cooler at
64 bp using 30 GB of RAM** — data roughly 30× larger than memory — and the full
multi-resolution `.mcool` from it in **under a gigabyte**.

Speed, measured on the same machine against `cooler` 0.10.4 (details in
[BENCHMARKS.md](BENCHMARKS.md)):

| op | cooler 0.10.4 | rooler | speedup |
|---|---|---|---|
| `cload` — 2.61 B pairs → 256 bp | 3258 s | **114 s** | **29×** |
| `balance` — 2.5 B pixels, 12.5 M bins | 670 s | **51 s** | **13×** |
| `coarsen` — 2.5 B pixels, 3 levels | 1955 s | **460 s** | **4.3×** |
| `cload` — 50 M pairs → 10 kb | 31.4 s | **1.2 s** | **26×** |

Same inputs, same machine, same session — and the same answers: `cload` output is
**byte-identical to cooler's**, coarsened levels match pixel for pixel, and balance selects an
identical bin mask with weights agreeing to ~1e-5.

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
r = rooler.open("merged.mcool", 1000)

r.raw("chr1:5,000,000-6,000,000")        # dense raw counts
r.balanced("chr1", "chr2")               # balanced, trans
r.matrix(balance=True).fetch("chr17")    # cooler-compatible form
```

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
- **expected** — cis distance-decay P(s) per region, stored in the cooler. Views can be
  `chroms`, `arms`, or your own BED; several views coexist in one file.
- **repack** — rewrite an existing cooler/mcool the way rooler would have written it:
  parallel-gzip compression, assembly stamped **and checked against the chromosome sizes**,
  balanced if it carries no weights, expected stored. In place by default (`--backup` keeps
  the original), or to a new path with `--out`.

Coming from cooler: `--nproc` works as an alias for `--threads`, and `--chunksize` maps onto
the `--mem` budget.

### Opinionated choices

**No mystery coolers.** rooler refuses to write a cooler without a genome assembly. It will take
`--assembly`, or infer one from the chromsizes, but it will not silently produce a file whose
provenance nobody can reconstruct later.

**Expected comes built in.** In practice, people compute cis expected with cooltools at default
parameters over chromosome arms — and wait hours-to-days for what is one O(nnz) pass. rooler
computes it **by default whenever weights are written** (`balance`, `zoomify --balance`,
`repack`), with a per-organism default view: arms where arms are meaningful (human, yeast),
whole chromosomes where they are not (mouse, fly, worm). Unknown genomes get a warning, not a
failure; `--no-expected` opts out; explicitly requested views (`arms`, `chroms`, custom BEDs)
coexist in the file.

**Counts are `int32`.** Internal accumulators are 64-bit, but stored counts saturate at
2,147,483,647 with a loud warning telling you how many pixels were affected. Half-width counts
mean smaller files and better cache behaviour, and a pixel with two billion reads in it is an
outlier, not a measurement.

## Compatibility

rooler writes **gzip-compressed coolers by default** — the same shuffle+deflate pipeline cooler
itself uses, so any HDF5 reader opens them with no filter plugins installed. It is also
substantially faster than the usual gzip path, so compatibility costs you nothing; the pipeline
is limited by its algorithms, not by the codec. `--preset blosc:zstd:1` is available if you
prefer, and rooler reads blosc coolers written by other tools.

Validated against the reference implementations: `cload` output is **byte-identical** to
`cooler cload` (verified on all 2.56 billion pixels of a real micro-C file); `merge` and
`zoomify` are pixel-exact; `balance` picks the identical set of bins and its weights agree with
`cooler.balance_cooler` to **2.5e-6** at matched tolerance; `expected` matches
`cooltools.expected_cis` to machine precision (6.4e-16).

`.pairs` coordinates are read as **1-based**, per the 4DN spec and cooler's default; pass
`--zero-based` for a file that genuinely is not.

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
