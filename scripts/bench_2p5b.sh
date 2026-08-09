#!/usr/bin/env bash
# Head-to-head at ~2.5 billion pixels / 12.5 M bins @ 256 bp -- the scale where waiting for the
# reference implementation stops being acceptable. Runs balance, coarsen and cload for both
# tools on identical inputs, at each tool's own defaults (i.e. what a user actually types).
#   scripts/bench_2p5b.sh <workdir>
set -u
W=${1:?usage: bench_2p5b.sh <workdir>}
SRC=${SRC:?set SRC=<cooler>}          # the ~2.5 B-pixel, 12.5 M-bin cooler at 256 bp
PAIRS=${PAIRS:?set PAIRS=<pairs.gz>}  # the pairs file SRC was built from
CHROMS=${CHROMS:-$W/chrom.sizes}      # derived from the pairs header if absent
NP=${NP:-8}
BIN=$(cd "$(dirname "$0")/.." && pwd)/target/release
COOLER=${COOLER:-$(command -v cooler)}
# cooler's workers need this only if SRC is blosc-compressed; harmless otherwise
[ -n "${HDF5_PLUGIN_PATH:-}" ] || export HDF5_PLUGIN_PATH=$(python -c \
  'import hdf5plugin,os;print(os.path.join(os.path.dirname(hdf5plugin.__file__),"plugins"))' 2>/dev/null || echo "")
mkdir -p "$W"
[ -s "$CHROMS" ] || bgzip -dc "$PAIRS" | sed -n 's/^#chromsize: \([^ ]*\) \(.*\)/\1\t\2/p' > "$CHROMS"

t() { # t <label...> -- <cmd...>
  local label=(); while [ "$1" != "--" ]; do label+=("$1"); shift; done; shift
  local s e rc
  s=$(date +%s)
  "$@" >"$W/$(echo "${label[*]}" | tr ' ' '_').out" 2>&1; rc=$?
  e=$(date +%s)
  printf 'RESULT\t%s\t%ss\trc=%d\n' "$(IFS=$'\t'; echo "${label[*]}")" "$((e-s))" "$rc"
  [ $rc -ne 0 ] && tail -5 "$W/$(echo "${label[*]}" | tr ' ' '_').out"
  return 0
}

echo "### 2.5B-pixel head to head, $(date -u +%FT%TZ), nproc=$NP"
echo "### src=$SRC"

# ---------- 1. balance (12.5 M bins) ----------
echo "## balance"
cp "$SRC" "$W/bal_r.cool"; cp "$SRC" "$W/bal_t.cool"; cp "$SRC" "$W/bal_c.cool"
t rooler balance default -- "$BIN/rooler" balance "$W/bal_r.cool" --nproc $NP
t rooler balance tiled   -- "$BIN/rooler" balance "$W/bal_t.cool" --nproc $NP --block 65536
t cooler balance default -- "$COOLER" balance -p $NP --force "$W/bal_c.cool"

# ---------- 2. coarsen 256 -> 512 -> 1024 -> 2048 ----------
echo "## coarsen"
rm -f "$W/r.mcool" "$W/c.mcool"
t rooler zoomify 3levels -- "$BIN/rooler" zoomify "$SRC" "$W/r.mcool" --resolutions 256,512,1024,2048
t cooler zoomify 3levels -- "$COOLER" zoomify -p $NP -r 512,1024,2048 -o "$W/c.mcool" "$SRC"
ls -la "$W/r.mcool" "$W/c.mcool" 2>/dev/null | awk '{print "# size", $NF, $5}'

# ---------- 3. cload 2.61 B pairs @ 256 bp ----------
echo "## cload"
rm -f "$W/cl_r.cool" "$W/cl_c.cool"
t rooler cload 256bp -- "$BIN/rooler" cload "$PAIRS" 256 "$W/cl_r.cool" --nproc $NP
t cooler cload 256bp -- "$COOLER" cload pairs -c1 2 -p1 3 -c2 4 -p2 5 --assembly hg38 \
    "$CHROMS:256" "$PAIRS" "$W/cl_c.cool"
ls -la "$W/cl_r.cool" "$W/cl_c.cool" 2>/dev/null | awk '{print "# size", $NF, $5}'

echo "### done $(date -u +%FT%TZ)"
