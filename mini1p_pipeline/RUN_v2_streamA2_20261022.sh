#!/bin/bash
# RUN_v2_streamA2_20261022.sh - the new-style decks again (10-22 evening): legibility redraws (P7, V1, V2, O1 parts, S3, P6 parts,
# W2 cut, FOV titles) and the pipeline slides (logger from the raw file, n and test on every slide).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
V=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2/v2_20261022
PC="$(cygpath -w /c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis/new_decks/psth_cache_win3.npz)"
w() { cygpath -w "$1"; }
echo "start $(date '+%F %T')"
for spec in "Day1+2_FULL|1 2|" "Day1+2_PI|1 2|--short" "Day1_FULL|1|" "Day1_PI|1|--short" "Day2_FULL|2|" "Day2_PI|2|--short"; do
  IFS='|' read name days extra <<< "$spec"
  for lg in en kr; do
    sub=EN; out=$name; [ $lg = kr ] && { sub=KR; out=${name}_KR; }
    echo "[$(date '+%T')] A $out"
    CA_DAY=2 python -I $PIPE/make_new_deck.py --base "$B" --out "$(w $V/NEW_STYLE/$sub/$out.pptx)" --days $days --lang $lg \
        --psth-cache "$PC" $extra > $V/NEW_STYLE/logs/$out.log 2>&1 || echo "*** $out failed"
  done
done
for lg in en kr; do
  sub=EN; out=PerMouse_Day1+2; [ $lg = kr ] && { sub=KR; out=PerMouse_Day1+2_KR; }
  echo "[$(date '+%T')] A $out"
  CA_DAY=2 python -I $PIPE/make_per_mouse_v2.py --lang $lg --out "$(w $V/NEW_STYLE/$sub/$out.pptx)" > $V/NEW_STYLE/logs/$out.log 2>&1 || echo "*** $out failed"
  out=Sex_analysis; [ $lg = kr ] && out=Sex_analysis_KR
  CA_DAY=2 python -I $PIPE/make_sex_deck.py --lang $lg --out "$(w $V/NEW_STYLE/$sub/$out.pptx)" > $V/NEW_STYLE/logs/$out.log 2>&1 || echo "*** $out failed"
done
echo "[$(date '+%T')] A decks done; PI video"
echo "(video already made)"
echo "all done $(date '+%F %T')"
