# Memory

`--mem` is a real budget, not a hint: every op except `balance` streams, so peak RSS is set by
the buffers you allow rather than by the size of the data.

## Measured peak RSS

VmHWM from real runs on the machine described in [BENCHMARKS.md](../BENCHMARKS.md).

| op | data | `--mem` | peak RSS |
|---|---|---|---|
| `cload` | 100 B pairs → 81.5 B pixels @64 bp | 32 | **29.8 GB** |
| `cload` | 2.61 B pairs → 2.56 B pixels @256 bp | 8 | ~10 GB |
| `balance` | 2.56 B pixels, 12.5 M bins | 8 | **8.2 GB** (scratch spills to disk) |
| `balance` | same | 24 | **8.7 GB** (scratch in RAM) |
| `coarsen` | 81.5 B pixels → 5 levels | — | **0.8 GB** |
| `expected` | 2.56 B pixels, arms view | — | **2.4 GB** |
| `merge` | 2 coolers @262 kb | 2 | ~3 GB |

## Rules of thumb

- **`cload` and `merge`**: peak ≈ `--mem` plus an O(nbins) overhead. Setting `--mem` to about
  half the RAM you want to give the job is a safe starting point.
- **`coarsen` and `expected`**: small and O(nbins) — a bin map and per-bin accumulators, never
  the matrix. Safe to run anywhere, at any data size.
- **`balance`** is the exception. It holds the matrix in a compressed form at roughly
  **2–2.6 bytes per pixel**, plus about `(threads + 6) × nbins × 8` bytes of vectors. Give it
  more than that via `--mem` and it stays in RAM; give it less and the compressed matrix moves
  to a disk-backed memory map — same results, slower, and peak RSS stays near `--mem`. Estimate
  it as `2.6 × nnz` bytes: a 2.5 B-pixel cooler wants ~7 GB, a 26 B-pixel one ~55 GB.

## Coming from cooler

cooler holds roughly `chunksize` pixel rows per process as pandas frames, about 40 B/row, so

```
--mem GB  ≈  chunksize × nproc × 40e-9
```

`--chunksize 10000000 --nproc 8` ≈ `--mem 3.2`. rooler accepts `--chunksize` directly and
applies this conversion, printing what it chose. The 40 B/row constant is approximate;
recalibrate against your own cooler run if you need it exact.
