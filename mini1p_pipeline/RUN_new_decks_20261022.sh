#!/bin/bash
# RUN_new_decks_20261022.sh - the decks in the layout of your own two decks (make_new_deck.py), EN + KR:
#   Day1+2_FULL, Day1+2_PI, Day1_FULL, Day1_PI, Day2_FULL, Day2_PI  ->  lab_meeting\decks_Day1_Day2\NEW_STYLE\EN | KR
# with previews (NEW_STYLE\previews). One PSTH cache for all (analysis\new_decks\psth_cache.npz; --fresh-cache to remake
# it after the neuron selection changed). The single-day decks use the design / neurons-per-mouse figures of the old
# Day-d PI decks, so the Day-1 ones wait for RUN_fix3's Day1_PI_summary rebuild (1191's merged neuron).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
N=$BASE/lab_meeting/decks_Day1_Day2/NEW_STYLE
L=$BASE/mini1p_out/20261021/logs
PC="$(cygpath -w $BASE/analysis/new_decks/psth_cache.npz)"
w() { cygpath -w "$1"; }
mkdir -p $N/EN $N/KR $N/previews $N/logs
[ "$1" = "--fresh-cache" ] && rm -f $BASE/analysis/new_decks/psth_cache.npz
one() {   # name, days, extra flags
  local name=$1 days=$2; shift 2
  for lg in en kr; do
    local sub=EN out=$name
    [ $lg = kr ] && { sub=KR; out=${name}_KR; }
    local f="$N/$sub/$out.pptx"
    # a deck you have open in PowerPoint cannot be replaced: write next to it
    if [ -f "$f" ] && ! ( : >> "$f" ) 2> /dev/null; then f="$N/$sub/${out}_new.pptx"; echo "  $out.pptx is open - writing $(basename $f)"; fi
    echo "[$(date '+%T')] $(basename $f)"
    CA_DAY=2 python -I $PIPE/make_new_deck.py --base "$B" --out "$(w $f)" --days $days --lang $lg --psth-cache "$PC" "$@" \
        > "$N/logs/$out.log" 2>&1 || { echo "*** $out failed (see $N/logs/$out.log)"; continue; }
    rm -f "$N/previews/${out}_slide"*.png
    python -I $PIPE/pptx_preview.py "$(w $f)" "$(w $N/previews)" --dpi 60 > /dev/null 2>&1
  done
}
one Day1+2_FULL "1 2"
one Day1+2_PI "1 2" --short
one Day2_FULL "2"
one Day2_PI "2" --short
# Day 1: wait for the rebuilt Day-1 PI figures (RUN_fix3 E step)
until grep -q "Day1_per_mouse\|all done" $L/fix_E_decks.log 2> /dev/null; do sleep 30; done
one Day1_FULL "1"
one Day1_PI "1" --short
echo "all done $(date '+%F %T')"
