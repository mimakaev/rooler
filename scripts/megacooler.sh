#!/bin/bash
set -e
BIN=/workspace/rooler/target/release/rooler
OUT=/workspace/scratch_bench/mega
mkdir -p "$OUT"
cools=""
for f in /workspace/encode_hic_pairs/*.pairs.gz; do
  name=$(basename "$f" .pairs.gz)
  echo "===== cload $name  $(date +%T) ====="
  /usr/bin/env time -v true 2>/dev/null || true
  t=$(date +%s)
  "$BIN" cload "$f" 256 "$OUT/$name.cool" --mem 8 --threads 8
  echo "  [$name cload took $(( $(date +%s) - t ))s]"
  cools="$cools $OUT/$name.cool"
done
echo "===== MERGE all -> megacooler_256  $(date +%T) ====="
t=$(date +%s)
"$BIN" merge "$OUT/megacooler_256.cool" $cools --mem 8
echo "  [merge took $(( $(date +%s) - t ))s]"
echo "===== DONE $(date +%T) ====="
ls -la "$OUT/megacooler_256.cool"
