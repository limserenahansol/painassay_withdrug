#!/bin/bash
# RUN_new_decks_day12_20261022.sh - the Day 1 + Day 2 new-style decks again (FULL + PI, EN + KR) with the CHECKS and OTHER WAYS
# slides (ca_event_validate.py) added after the event analysis; nothing removed. Previews in NEW_STYLE\previews. A deck open
# in PowerPoint is written next to it as <name>_new.pptx.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
N=$BASE/lab_meeting/decks_Day1_Day2/NEW_STYLE
PC="$(cygpath -w $BASE/analysis/new_decks/psth_cache.npz)"
w() { cygpath -w "$1"; }
one() {
  local name=$1 days=$2; shift 2
  for lg in en kr; do
    local sub=EN out=$name
    [ $lg = kr ] && { sub=KR; out=${name}_KR; }
    local f="$N/$sub/$out.pptx"
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
echo "all done $(date '+%F %T')"
