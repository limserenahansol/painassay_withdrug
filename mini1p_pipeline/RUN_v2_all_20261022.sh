#!/bin/bash
# RUN_v2_all_20261022.sh - every deck again into lab_meeting\decks_Day1_Day2\v2_20261022 (the old files stay):
#   A) new style, best window, every event, stars with their pair, split figures, sex + P2 slides: 6 decks x EN + KR,
#      the per-mouse v2 deck (EN + KR) and the sex deck alone (EN + KR)
#   B) the old-style decks again (pair-marked stars), EN + KR, kept as history
# Max 2 heavy jobs at a time (A and B run in parallel, each sequential).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
V=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2/v2_20261022
PC="$(cygpath -w /c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis/new_decks/psth_cache_win3.npz)"
w() { cygpath -w "$1"; }
streamA() {
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
  echo "[$(date '+%T')] A done"
}
streamB() {
  for spec in "1|make_day1_full_deck|Day1_PI_summary" "2|make_day1_full_deck|Day2_PI_summary" "1|make_day1_per_mouse_deck|Day1_per_mouse" \
              "2|make_day1_per_mouse_deck|Day2_per_mouse" "2|make_longitudinal_deck|Day1+2_longitudinal" "1|make_day1_complete_deck|Day1_FULL" \
              "2|make_day1_complete_deck|Day2_FULL"; do
    IFS='|' read day mod name <<< "$spec"
    echo "[$(date '+%T')] B $name"
    CA_DAY=$day python -I $PIPE/$mod.py --base "$B" --out "$(w $V/OLD_STYLE/EN/$name.pptx)" > $V/OLD_STYLE/logs/$name.log 2>&1 \
      || { echo "*** $name failed"; continue; }
    python -I $PIPE/pptx_translate.py --in "$(w $V/OLD_STYLE/EN/$name.pptx)" --out "$(w $V/OLD_STYLE/KR/${name}_KR.pptx)" \
      --dump "$(w $V/OLD_STYLE/logs/missing_kr_$name.json)" > $V/OLD_STYLE/logs/${name}_kr.log 2>&1 || echo "*** $name KR failed"
  done
  echo "[$(date '+%T')] B done"
}
echo "start $(date '+%F %T')"
streamA > $V/NEW_STYLE/logs/run_A.log 2>&1 &
PA=$!
streamB > $V/OLD_STYLE/logs/run_B.log 2>&1 &
PB=$!
wait $PA $PB
cat $V/NEW_STYLE/logs/run_A.log $V/OLD_STYLE/logs/run_B.log
echo "all done $(date '+%F %T')"
