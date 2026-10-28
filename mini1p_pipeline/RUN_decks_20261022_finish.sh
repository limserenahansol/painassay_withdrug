#!/bin/bash
# RUN_decks_20261022_finish.sh - after RUN_decks_20261022_evening.sh: the Day 1 + 2 longitudinal deck again (fig_within_day /
# split_rows restored - an edit of 10-22 had removed them) and the Korean versions of the decks translated before this
# evening's new templates (Day1_PI_summary, Day2_PI_summary, Day2_per_mouse). Open decks -> <name>_new.pptx.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
D=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2
until grep -q "all done" $D/logs/evening_20261022.log; do sleep 20; done
w() { cygpath -w "$1"; }
target() {
  local f=$1
  if [ -f "$f" ] && ! ( : >> "$f" ) 2> /dev/null; then echo "${f%.pptx}_new.pptx"; else echo "$f"; fi
}
out=$(target $D/EN/Day1+2_longitudinal.pptx)
echo "[$(date '+%T')] Day1+2_longitudinal -> $(basename $out)"
CA_DAY=2 python -I $PIPE/make_longitudinal_deck.py --base "$B" --out "$(w $out)" > $D/logs/Day1+2_longitudinal.log 2>&1 \
  || echo "*** Day1+2_longitudinal failed (see $D/logs/Day1+2_longitudinal.log)"
rm -f $D/EN/previews/$(basename ${out%.pptx})_slide*.png
python -I $PIPE/pptx_preview.py "$(w $out)" "$(w $D/EN/previews)" --dpi 60 > /dev/null 2>&1
bash $PIPE/RUN_kr_only_20261022.sh Day1+2_longitudinal Day1_PI_summary Day2_PI_summary Day2_per_mouse
echo "finish done $(date '+%F %T')"
