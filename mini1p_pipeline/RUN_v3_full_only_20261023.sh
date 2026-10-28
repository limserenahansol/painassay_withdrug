#!/bin/bash
# RUN_v3_full_only_20261023.sh - the three FULL new-style decks again (the check-3 figure title, 10-23), EN and KR side by side,
# into NEW_STYLE_v3; then the final assembly (RUN_v3_finish_20261023.sh)
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
V=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2/v2_20261022
NS=$V/NEW_STYLE_v3
PC="$(cygpath -w /c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis/new_decks/psth_cache_win3.npz)"
w() { cygpath -w "$1"; }
echo "start $(date '+%F %T')"
one_lang() {
  lg=$1; sub=EN; [ $lg = kr ] && sub=KR
  for spec in "Day1+2_FULL|1 2" "Day1_FULL|1" "Day2_FULL|2"; do
    IFS='|' read name days <<< "$spec"
    out=$name; [ $lg = kr ] && out=${name}_KR
    echo "[$(date '+%T')] $out"
    CA_DAY=2 python -I $PIPE/make_new_deck.py --base "$B" --out "$(w $NS/$sub/$out.pptx)" --days $days --lang $lg \
        --psth-cache "$PC" > $NS/logs/$out.log 2>&1 || echo "*** $out failed"
  done
}
one_lang en > $NS/logs/_full_en.log 2>&1 &
one_lang kr > $NS/logs/_full_kr.log 2>&1
wait
cat $NS/logs/_full_en.log $NS/logs/_full_kr.log
grep -l "Traceback" $NS/logs/*FULL*.log 2>/dev/null && echo "*** tracebacks"
echo "decks done $(date '+%T')"
bash $PIPE/RUN_v3_finish_20261023.sh
