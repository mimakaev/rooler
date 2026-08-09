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

rooler's answer is that **every op streams**. A `--mem` budget bounds RAM; nothing loads a pixel
table. The largest test so far built an **81-billion-pixel, 48-million-bin cooler at 64 bp
resolution from 100 billion pairs, using 30 GB of RAM** — data roughly 30× larger than memory —
and then built the full multi-resolution `.mcool` from it in **under a gigabyte**.

Speed, measured on the same machine against `cooler` 0.10.4 (details in
[BENCHMARKS.md](BENCHMARKS.md)):

| op | cooler 0.10.4 | rooler | speedup |
|---|---|---|---|
| `cload` — 50 M pairs → 10 kb | 31.4 s | **1.2 s** | **26×** |
| `balance` — genome-wide IC | 18.7 s | **1.2 s** | **16×** |
| `balance` — 1.12 B-pixel cooler | 80.6 s | **12.0 s** | **6.7×** |
| `zoomify` — 5 resolutions | 13.6 s | **3.7 s** | **3.7×** |

Same inputs, same machine, same session, with matching output sizes and matching results.

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

# multi-resolution mcool, balancing every level
rooler zoomify merged.cool merged.mcool --balance

# or balance one resolution, then compute expected P(s)
rooler balance merged.mcool::resolutions/1000
rooler expected merged.mcool::resolutions/1000
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
                                        [--tol 1e-4] [--threads 8] [--block 65536]
rooler expected <cool[::resolutions/R]> [--view chroms|arms|custom:<bed>]
```

- **cload** — reads bgzipped `.pairs`, plain text, or `-` for stdin.
- **merge** — refuses inputs whose bin layouts disagree, rather than producing quiet garbage.
- **zoomify** — `--balance` balances every resolution as it goes.
- **balance** — genome-wide iterative correction. `--block` enables a cache-blocked kernel that
  is worth ~2.8× on very large, fine-resolution coolers (above a few million bins).
- **expected** — cis distance-decay P(s) per region, stored in the cooler. Views can be
  `chroms`, `arms`, or your own BED; several views coexist in one file.

Coming from cooler: `--nproc` works as an alias for `--threads`, and `--chunksize` maps onto
the `--mem` budget.

### Two opinionated choices

**No mystery coolers.** rooler refuses to write a cooler without a genome assembly. It will take
`--assembly`, or infer one from the chromsizes, but it will not silently produce a file whose
provenance nobody can reconstruct later.

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

Validated against the reference implementations: `cload`, `merge` and `zoomify` are
**pixel-exact**; `balance` picks the identical set of bins and its weights agree with
`cooler.balance_cooler` to **2.5e-6** at matched tolerance; `expected` matches
`cooltools.expected_cis` to machine precision (6.4e-16).

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

- **`balance` is memory-bound by design.** It holds a compressed matrix in RAM (~2 bytes per
  pixel), so a 100-billion-pixel cooler needs far more than a workstation has. Everything else
  streams. Balancing at that scale is the main open problem.
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
