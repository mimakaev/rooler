# PLAN_LOG — measurements while executing PLAN.md

## Phase 0 — baseline (2026-08-09)
- `cargo build --release` clean (1 dead-code warning: `chrom_enum`); `cargo test --release` 3/3 pass.
- Baseline balance on `/workspace/scratch_bench/plan_base.cool` (copy of enc514_256.cool,
  2,563,532,430 pix / 12,537,161 bins), `--block 65536 --threads 8`:
  - scratch(tiled) 6.4GB (2.51 B/pix) built in **28s**
  - mask **11239212/12537161** bins at 32s
  - `converged=true cv=7.52e-5 scale=461.0` total **50s** (7 IC iters)
  - matches the pre-plan measurement (27s build / 48s total) within noise.
- Disk: /workspace 1.2T free.
- Note: `/usr/bin/time` is not installed in this container; used rooler's own timing lines.
