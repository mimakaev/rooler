#!/usr/bin/env bash
# Large-scale stretch test: synthetic pairs -> cload -> zoomify at 64bp, with RSS tracking.
#   scripts/stretch64.sh <npairs> <outdir> [mem_gb] [threads]
# Streams pairs from the generator straight into cload (nothing is ever stored as text).
set -u
N=${1:-100000000000}
OUT=${2:?usage: stretch64.sh <npairs> <outdir> [mem_gb] [threads]}
MEM=${3:-32}
THREADS=${4:-8}
BIN=$(dirname "$0")/../target/release
mkdir -p "$OUT"

# peak RSS of a pid tree, sampled (no /usr/bin/time in this container)
watch_rss() {
  local pid=$1 name=$2 peak=0
  while kill -0 "$pid" 2>/dev/null; do
    local tot=0
    for p in $(pgrep -P "$pid" 2>/dev/null) "$pid"; do
      local v; v=$(awk '/VmRSS/{print $2}' /proc/"$p"/status 2>/dev/null)
      [ -n "${v:-}" ] && tot=$((tot + v))
    done
    [ "$tot" -gt "$peak" ] && peak=$tot
    sleep 2
  done
  echo "PEAK_RSS_${name}_GB=$(awk -v k=$peak 'BEGIN{printf "%.1f", k/1048576}')"
}

echo "=== stretch64: $N pairs @64bp, --mem $MEM --threads $THREADS -> $OUT"
df -h "$OUT" | tail -1

echo "--- cload ---"
S=$(date +%s)
"$BIN/examples/genpairs" "$N" 20260809 6 \
  | "$BIN/rooler" cload - 64 "$OUT/mega64.cool" --mem "$MEM" --threads "$THREADS" &
CL=$!
watch_rss $CL cload &
W=$!
wait $CL; RC=$?
wait $W
echo "cload rc=$RC elapsed=$(( $(date +%s) - S ))s"
[ $RC -ne 0 ] && exit $RC
ls -la "$OUT/mega64.cool"; df -h "$OUT" | tail -1

echo "--- zoomify (64 -> 128 -> 256 -> 512 -> 1024) ---"
S=$(date +%s)
"$BIN/rooler" zoomify "$OUT/mega64.cool" "$OUT/mega64.mcool" --resolutions 64,128,256,512,1024 &
ZP=$!
watch_rss $ZP zoomify &
W=$!
wait $ZP; RC=$?
wait $W
echo "zoomify rc=$RC elapsed=$(( $(date +%s) - S ))s"
ls -la "$OUT/mega64.mcool"; df -h "$OUT" | tail -1
