#!/bin/bash
# RUN_timefix_decks_20261023.sh - every new-style deck again from the timing-corrected analyses (10-23) into NEW_STYLE_v4
# (NEW_STYLE_v3 stays as it was), with a NEW trace cache (psth_cache_win3_tf.npz; the old cache was built from the old marks),
# and the pipeline figures (old ones kept in pipeline_figs_v5pre_timefix). The first deck builds the cache alone, then EN and
# KR run side by side (2 jobs).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
A=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis
V=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2/v2_20261022
NS=$V/NEW_STYLE_v4
PC="$(cygpath -w $A/new_decks/psth_cache_win3_tf.npz)"
w() { cygpath -w "$1"; }
export SCORING_SUFFIX=_timefix
mkdir -p $NS/EN $NS/KR $NS/logs
echo "start $(date '+%F %T')"
[ -d $A/pipeline_figs_v5pre_timefix ] || cp -r $A/pipeline_figs $A/pipeline_figs_v5pre_timefix
python -I -X utf8 $PIPE/make_pipeline_figs.py > $NS/logs/pipeline_figs.log 2>&1 || echo "*** pipeline figs failed"
# 10-23 fix: the split parts (_p1/_p2, _q1-_q4) of the event-analysis / validation figures are crops made by crop_split.py;
# make_new_deck shows the parts whenever they exist, so they must be re-cut after every analysis run
python -I -X utf8 $PIPE/crop_split.py > $NS/logs/crop_split.log 2>&1 || echo "*** crop_split failed"
echo "[$(date '+%T')] Day1+2_FULL (builds the trace cache)"
CA_DAY=2 python -I $PIPE/make_new_deck.py --base "$B" --out "$(w $NS/EN/Day1+2_FULL.pptx)" --days 1 2 --lang en \
    --psth-cache "$PC" > $NS/logs/Day1+2_FULL.log 2>&1 || echo "*** Day1+2_FULL failed"
one_lang() {
  lg=$1; sub=EN; [ $lg = kr ] && sub=KR
  for spec in "Day1+2_FULL|1 2|" "Day1+2_PI|1 2|--short" "Day1_FULL|1|" "Day1_PI|1|--short" "Day2_FULL|2|" "Day2_PI|2|--short"; do
    IFS='|' read name days extra <<< "$spec"
    out=$name; [ $lg = kr ] && out=${name}_KR
    [ $lg = en ] && [ $name = Day1+2_FULL ] && continue
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
ls -la $NS/EN $NS/KR | grep pptx
echo "all done $(date '+%F %T')"
