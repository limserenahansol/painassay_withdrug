#!/bin/bash
# RUN_v3_newstyle_20261023.sh - every new-style deck again after the round-2 triple check (P5 / V4 / O1 / population / map
# figures, the Korean design figure), into NEW_STYLE_v3 (NEW_STYLE stays as it was: Day2_FULL_KR is open in PowerPoint).
# EN and KR run side by side (2 jobs).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
V=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2/v2_20261022
NS=$V/NEW_STYLE_v3
PC="$(cygpath -w /c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis/new_decks/psth_cache_win3.npz)"
w() { cygpath -w "$1"; }
mkdir -p $NS/EN $NS/KR $NS/logs
echo "start $(date '+%F %T')"
one_lang() {
  lg=$1; sub=EN; [ $lg = kr ] && sub=KR
  for spec in "Day1+2_FULL|1 2|" "Day1+2_PI|1 2|--short" "Day1_FULL|1|" "Day1_PI|1|--short" "Day2_FULL|2|" "Day2_PI|2|--short"; do
    IFS='|' read name days extra <<< "$spec"
    out=$name; [ $lg = kr ] && out=${name}_KR
    echo "[$(date '+%T')] $out"
    CA_DAY=2 python -I $PIPE/make_new_deck.py --base "$B" --out "$(w $NS/$sub/$out.pptx)" --days $days --lang $lg \
        --psth-cache "$PC" $extra > $NS/logs/$out.log 2>&1 || echo "*** $out failed"
  done
  out=PerMouse_Day1+2; [ $lg = kr ] && out=PerMouse_Day1+2_KR
  echo "[$(date '+%T')] $out"
  CA_DAY=2 python -I $PIPE/make_per_mouse_v2.py --lang $lg --out "$(w $NS/$sub/$out.pptx)" > $NS/logs/$out.log 2>&1 || echo "*** $out failed"
  out=Sex_analysis; [ $lg = kr ] && out=Sex_analysis_KR
  CA_DAY=2 python -I $PIPE/make_sex_deck.py --lang $lg --out "$(w $NS/$sub/$out.pptx)" > $NS/logs/$out.log 2>&1 || echo "*** $out failed"
}
one_lang en > $NS/logs/_en.log 2>&1 &
one_lang kr > $NS/logs/_kr.log 2>&1
wait
cat $NS/logs/_en.log $NS/logs/_kr.log
grep -l "Traceback" $NS/logs/*.log 2>/dev/null && echo "*** tracebacks above"
ls -la $NS/EN $NS/KR
echo "all done $(date '+%F %T')"
