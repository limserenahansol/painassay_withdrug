#!/bin/bash
# RUN_decks_20261022_evening.sh - every deck again after your 10-22 decisions (all 4 mice confirmed, 3-colour neuron marks,
# * / ** / *** on the figures, the event analysis): A) the old-style Day-1 and Day-2 PI decks first (their design /
# neurons-per-mouse figures feed the new single-day decks), then in parallel B1) the new-style decks (RUN_new_decks_20261022.sh,
# 6 decks x EN + KR) and B2) the other old-style decks. A deck that is open in PowerPoint is never replaced: it is written
# next to it as <name>_new.pptx. EN + KR previews of every deck.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
D=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2
mkdir -p $D/logs $D/EN/previews $D/KR/previews $D/NEW_STYLE/logs
w() { cygpath -w "$1"; }
target() {   # the path to write: <name>_new.pptx when <name>.pptx is open in PowerPoint
  local f=$1
  if [ -f "$f" ] && ! ( : >> "$f" ) 2> /dev/null; then echo "${f%.pptx}_new.pptx"; else echo "$f"; fi
}
kr() {       # $1 = deck name, $2 = the EN file just written
  local name=$1 en=$2 out
  out=$(target $D/KR/${name}_KR.pptx)
  python -I $PIPE/pptx_translate.py --in "$(w $en)" --out "$(w $out)" --dump "$(w $D/logs/missing_kr_$name.json)" | head -1
  rm -f $D/KR/previews/$(basename ${out%.pptx})_slide*.png
  python -I $PIPE/pptx_preview.py "$(w $out)" "$(w $D/KR/previews)" --dpi 60 > /dev/null 2>&1
}
build() {
  local day=$1 mod=$2 name=$3 out
  out=$(target $D/EN/$name.pptx)
  echo "[$(date '+%T')] $name -> $(basename $out)"
  CA_DAY=$day python -I $PIPE/$mod.py --base "$B" --out "$(w $out)" > $D/logs/$name.log 2>&1 \
    || { echo "*** $name failed (see $D/logs/$name.log)"; return 1; }
  rm -f $D/EN/previews/$(basename ${out%.pptx})_slide*.png
  python -I $PIPE/pptx_preview.py "$(w $out)" "$(w $D/EN/previews)" --dpi 60 > /dev/null 2>&1
  kr $name $out
}
echo "start $(date '+%F %T')"
build 1 make_day1_full_deck Day1_PI_summary
build 2 make_day1_full_deck Day2_PI_summary
bash $PIPE/RUN_new_decks_20261022.sh > $D/NEW_STYLE/logs/run_evening_20261022.log 2>&1 &
NEWPID=$!
build 1 make_day1_per_mouse_deck Day1_per_mouse
build 2 make_day1_per_mouse_deck Day2_per_mouse
build 2 make_longitudinal_deck Day1+2_longitudinal
build 1 make_day1_complete_deck Day1_FULL
build 2 make_day1_complete_deck Day2_FULL
wait $NEWPID
echo "new-style decks:"; cat $D/NEW_STYLE/logs/run_evening_20261022.log
echo "all done $(date '+%F %T')"
