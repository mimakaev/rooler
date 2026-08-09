#!/usr/bin/env bash
# Head-to-head benchmark: rooler vs cooler on the same inputs, same machine.
#   scripts/bench_vs_cooler.sh <pairs.gz> <workdir> [nproc]
# Prints one TSV line per measurement: <tool>\t<op>\t<detail>\t<seconds>\t<bytes>
set -u
PAIRS=${1:?pairs.gz}
W=${2:?workdir}
NP=${3:-8}
BIN=$(cd "$(dirname "$0")/.." && pwd)/target/release
COOLER=${COOLER:-$(command -v cooler)}
[ -x "$COOLER" ] || { echo "need cooler on PATH (or COOLER=/path/to/cooler)"; exit 1; }
mkdir -p "$W"

t() { # t <label...> -- <cmd...>
  local label=() ; while [ "$1" != "--" ]; do label+=("$1"); shift; done; shift
  local s e
  s=$(date +%s.%N)
  "$@" >"$W/last.out" 2>&1; local rc=$?
  e=$(date +%s.%N)
  printf '%s\t%s\tRC=%d\n' "$(IFS=$'\t'; echo "${label[*]}")" \
      "$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.1f", b-a}')" "$rc"
  [ $rc -ne 0 ] && tail -3 "$W/last.out"
  return 0
}

# chromsizes for cooler cload (from the pairs header)
bgzip -dc "$PAIRS" | sed -n 's/^#chromsize: \([^ ]*\) \(.*\)/\1\t\2/p' > "$W/chrom.sizes"
NPAIRS=$(bgzip -dc "$PAIRS" | grep -vc '^#')
echo "# pairs=$NPAIRS  nproc=$NP  chroms=$(wc -l < "$W/chrom.sizes")"
echo -e "tool\top\tdetail\tseconds\trc"

for BS in 10000; do
  rm -f "$W/rooler.$BS.cool" "$W/cooler.$BS.cool"
  t rooler cload "${BS}bp" -- "$BIN/rooler" cload "$PAIRS" "$BS" "$W/rooler.$BS.cool" --nproc $NP
  t cooler cload "${BS}bp" -- "$COOLER" cload pairs -c1 2 -p1 3 -c2 4 -p2 5 \
      --assembly hg38 "$W/chrom.sizes:$BS" "$PAIRS" "$W/cooler.$BS.cool"
  ls -la "$W/rooler.$BS.cool" "$W/cooler.$BS.cool" 2>/dev/null | awk '{print "# size", $NF, $5}'
done

# --- balance on the same cooler (copy each, they write in place) ---
cp "$W/rooler.10000.cool" "$W/bal_r.cool"; cp "$W/rooler.10000.cool" "$W/bal_c.cool"
t rooler balance 10000bp -- "$BIN/rooler" balance "$W/bal_r.cool" --nproc $NP
t cooler balance 10000bp -- "$COOLER" balance -p $NP --force "$W/bal_c.cool"

# --- zoomify ---
rm -f "$W/r.mcool" "$W/c.mcool"
t rooler zoomify 5levels -- "$BIN/rooler" zoomify "$W/rooler.10000.cool" "$W/r.mcool" \
    --resolutions 10000,20000,40000,80000,160000
t cooler zoomify 5levels -- "$COOLER" zoomify -p $NP -r 20000,40000,80000,160000 \
    -o "$W/c.mcool" "$W/rooler.10000.cool"
ls -la "$W/r.mcool" "$W/c.mcool" 2>/dev/null | awk '{print "# size", $NF, $5}'
echo "# done"

# --- medium-scale balance only (a 1.1B-pixel, 64k-bin 50kb cooler). Deliberately NOT the
# --- 2.5B/12.5M-bin or 81B-pixel coolers: cooler balance there would run for hours.
MED=${MED:-}   # optional: a larger cooler for a second balance comparison
if [ -f "$MED" ]; then
  cp "$MED" "$W/med_r.cool"; cp "$MED" "$W/med_c.cool"
  echo "# medium balance input: $MED"
  t rooler balance med50kb -- "$BIN/rooler" balance "$W/med_r.cool" --nproc $NP
  t cooler balance med50kb -- "$COOLER" balance -p $NP --force "$W/med_c.cool"
  rm -f "$W/med_r.cool" "$W/med_c.cool"
fi
