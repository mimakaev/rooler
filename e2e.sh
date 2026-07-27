#!/bin/bash
set -e
BIN=/workspace/rooler/target/release/rooler
D=/workspace/scratch_bench/e2e
mkdir -p "$D"
echo "=== [1] cload @50kb $(date +%T) ==="
"$BIN" cload /workspace/encode_hic_pairs/ENCFF514KZU.pairs.gz 50000 "$D/base.cool" --mem 8 --threads 8
echo "=== [2] zoomify $(date +%T) ==="
"$BIN" zoomify "$D/base.cool" "$D/out.mcool"
echo "=== [3] balance 50kb $(date +%T) ==="
"$BIN" balance "$D/out.mcool::resolutions/50000"
echo "=== DONE $(date +%T) ==="
