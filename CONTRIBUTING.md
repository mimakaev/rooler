# Contributing

rooler is alpha, and issues are more valuable than patches right now — especially **a case
where rooler and `cooler` disagree**. If you have one, that is the most useful thing you can
send. Please include the assembly, resolution, and roughly how big the input is.

## Ground rules

**Nothing lands without a test.** The suite is fast on purpose (~0.6 s) so there is no excuse
to skip it:

```bash
cargo test --release
```

Tests check ops against an *independent oracle* — a brute-force recomputation — rather than a
stored blessed answer. A blessed answer only tells you the behaviour changed; an oracle tells
you which one is wrong. Please keep that pattern.

**Measure performance claims.** Numbers in this repo come with the input, the machine and the
command that produced them — see `BENCHMARKS.md`, and keep that property. A benchmark without a
stated input size is not a benchmark.

**Correctness before speed.** Several ops write files people will analyse and publish from. An
op that is quietly wrong is worse than one that is slow, so prefer erroring out over guessing —
that principle is why merge validates its inputs and why counts saturate loudly instead of
wrapping.

## Notes for working in this codebase

- **HDF5 handles must be scoped.** Dropping a `File` is not enough — a live `Group` or `Dataset`
  keeps the file open, and the next op's read-write open then fails with a confusing error.
  Scope your reads in a block.
- **The SpMV kernels in `src/scratch*.rs` are tuned and validated.** They use `unsafe` indexing
  deliberately and both kernels must produce identical weights. Benchmark before and after any
  change there, and keep the cross-kernel test passing.
- **Do not change the on-disk schema casually.** Files must keep round-tripping through
  `cooler.Cooler`; `scripts/validate_vs_cooler.py` is the check.

## Docs

`docs/VALIDATION.md` records what has been verified against cooler/cooltools and what has not;
if you add or change an op, update it. `docs/MEMORY.md` holds the measured RSS figures.

## Authorship

This codebase was written entirely by Claude (Anthropic) under human direction — see the
[README](README.md#authorship). Contributions from humans and models are equally welcome; the
bar is the same either way, and it is the tests and the measurements, not the author.
