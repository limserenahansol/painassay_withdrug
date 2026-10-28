#!/bin/bash
# RUN_all_decks.sh [day1|day2|long|all] - every deck from the current analysis files, English + Korean.
# After you change neurons (ca_exclude_rois.py, then ca_post_mouse*.sh + the cohort script), run this once: all figures
# and numbers are re-read from the files, so only what changed changes. Korean = pptx_translate.py (number-masked
# templates, translations_kr.json); figure labels stay English. Previews (PNG per slide) next to every deck.
#   Day 1 / Day 2:  PI summary (make_day1_full_deck), FULL (make_day1_complete_deck), per mouse (make_day1_per_mouse_deck)
#   Day 1 + 2:      longitudinal (make_longitudinal_deck)
# Output: lab_meeting\decks_Day1_Day2\EN and \KR (overwritten by every run; DECKS=<folder name> for another folder)
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B='C:/Users/hsollim/Documents/HEAL_mini1p_SBI553'
DECKS=${DECKS:-decks_Day1_Day2}
D=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/$DECKS
mkdir -p $D/EN $D/KR $D/logs
w() { cygpath -w "$1"; }
WHAT=${1:-all}
build() {   # day builder name
  local day=$1 mod=$2 name=$3
  echo "[$(date '+%T')] $name"
  CA_DAY=$day python -I $PIPE/$mod.py --base "$B" --out "$(w $D/EN/$name.pptx)" > $D/logs/$name.log 2>&1 \
    || { echo "*** $name failed (see $D/logs/$name.log)"; return 1; }
  python -I $PIPE/pptx_translate.py --in "$(w $D/EN/$name.pptx)" --out "$(w $D/KR/${name}_KR.pptx)" \
      --dump "$(w $D/logs/missing_kr.json)" | head -1
}
if [ "$WHAT" = day1 ] || [ "$WHAT" = all ]; then
  build 1 make_day1_full_deck Day1_PI_summary
  build 1 make_day1_per_mouse_deck Day1_per_mouse
  build 1 make_day1_complete_deck Day1_FULL
fi
if [ "$WHAT" = day2 ] || [ "$WHAT" = all ]; then
  build 2 make_day1_full_deck Day2_PI_summary
  build 2 make_day1_per_mouse_deck Day2_per_mouse
  build 2 make_day1_complete_deck Day2_FULL
fi
if [ "$WHAT" = long ] || [ "$WHAT" = all ]; then
  build 2 make_longitudinal_deck Day1+2_longitudinal
fi
for f in $D/KR/*.pptx; do
  python -I $PIPE/pptx_preview.py "$(w $f)" "$(w $D/KR/previews)" --dpi 60 > /dev/null 2>&1
done
echo "decks -> $D (EN, KR; previews in KR/previews and next to every EN deck)"
